"""
D-PeRFlow Training Script

This script trains a student model using the D-PeRFlow algorithm to distill
a pre-trained SEDD teacher model for few-step discrete diffusion generation.

Usage:
    # Single GPU
    python run_train_d_perflow.py --config configs/d_perflow.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 run_train_d_perflow.py --config configs/d_perflow.yaml
"""

import datetime
import os
import os.path
import argparse
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

import math
from collections import Counter

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
import d_perflow
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


torch.backends.cudnn.benchmark = True


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def load_teacher_model(cfg, device):
    """
    Load pre-trained teacher model from checkpoint.

    Args:
        cfg: Configuration object
        device: Target device

    Returns:
        teacher_model: Loaded teacher model in eval mode
    """
    # Create teacher model
    teacher_model = SEDD(cfg).to(device)

    # Load checkpoint
    if cfg.d_perflow.teacher_checkpoint:
        checkpoint = torch.load(
            cfg.d_perflow.teacher_checkpoint,
            map_location=device
        )
        # Handle DDP state dict
        state_dict = checkpoint.get('model', checkpoint)
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()

        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        teacher_model.load_state_dict(new_state_dict)
        print(f"Loaded teacher model from {cfg.d_perflow.teacher_checkpoint}")

    # Set to eval mode and freeze
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    return teacher_model


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Check if using distributed training
    distributed = world_size > 1

    # Create directories
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # Logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))

    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint("{} \t Memory: {:.2f}GB".format(
                props.name, props.total_memory / (1024 ** 3)))
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # Build graph and noise
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    if distributed:
        noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5

    # Load teacher model
    mprint("Loading teacher model...")
    teacher_model = load_teacher_model(cfg, device)
    mprint("Teacher model loaded successfully.")

    # Build student model (initialized from teacher)
    mprint("Initializing student model from teacher weights...")
    student_model = SEDD(cfg).to(device)

    if cfg.d_perflow.init_from_teacher:
        # Copy teacher weights to student
        student_model.load_state_dict(teacher_model.state_dict())
        mprint("Student initialized from teacher weights.")

    if distributed:
        student_model = DDP(student_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in student_model.parameters())
    mprint(f"Number of parameters in student model: {num_parameters}")

    # EMA for student
    ema = ExponentialMovingAverage(student_model.parameters(), decay=cfg.training.ema)
    mprint(f"Student Model: {student_model}")
    mprint(f"EMA: {ema}")

    # Optimizer
    optimizer = losses.get_optimizer(cfg, student_model.parameters())
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")

    state = dict(
        optimizer=optimizer,
        scaler=scaler,
        model=student_model,
        noise=noise,
        ema=ema,
        step=0
    )

    # Restore from checkpoint if available
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Data loaders
    train_ds, eval_ds = data.get_dataloaders(cfg, distributed=distributed)
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # D-PeRFlow training step function
    optimize_fn = losses.optimization_manager(cfg)

    # Debug log file path
    debug_log_file = os.path.join(work_dir, 'debug_log.txt')
    mprint(f"Debug log will be saved to: {debug_log_file}")

    lambda_fwd_kl = cfg.d_perflow.get('lambda_fwd_kl', 1.0)
    lambda_rev_kl = cfg.d_perflow.get('lambda_rev_kl', 0.0)
    lambda_perceptual = cfg.d_perflow.get('lambda_perceptual', 0.0)

    # Perceptual loss (frozen GPT-2 features)
    perceptual_loss_fn = None
    if lambda_perceptual > 0:
        from perceptual_loss import GPT2PerceptualLoss
        perceptual_layers = cfg.d_perflow.get('perceptual_layers', [2, 5, 8, 11])
        perceptual_loss_type = cfg.d_perflow.get('perceptual_loss_type', 'l2')
        perceptual_loss_fn = GPT2PerceptualLoss(
            model_name="gpt2",
            feature_layers=tuple(perceptual_layers),
            loss_type=perceptual_loss_type,
        ).to(device)
        mprint(f"Perceptual loss: GPT-2 layers {list(perceptual_layers)}, type={perceptual_loss_type}")

    teacher_method = cfg.d_perflow.get('teacher_method', 'analytic')
    student_method = cfg.d_perflow.get('student_method', 'analytic')

    train_step_fn = d_perflow.get_d_perflow_step_fn(
        noise=noise,
        graph=graph,
        teacher_model=teacher_model,
        num_time_windows=cfg.d_perflow.num_time_windows,
        sampling_eps=cfg.d_perflow.sampling_eps,
        teacher_steps=cfg.d_perflow.teacher_steps,
        train=True,
        optimize_fn=optimize_fn,
        accum=cfg.training.accum,
        train_temperature=cfg.d_perflow.get('train_temperature', 1.0),
        debug_log_file=debug_log_file,
        lambda_fwd_kl=lambda_fwd_kl,
        lambda_rev_kl=lambda_rev_kl,
        lambda_perceptual=lambda_perceptual,
        perceptual_loss_fn=perceptual_loss_fn,
        teacher_method=teacher_method,
        student_method=student_method,
    )

    eval_step_fn = d_perflow.get_d_perflow_step_fn(
        noise=noise,
        graph=graph,
        teacher_model=teacher_model,
        num_time_windows=cfg.d_perflow.num_time_windows,
        sampling_eps=cfg.d_perflow.sampling_eps,
        teacher_steps=cfg.d_perflow.teacher_steps,
        train=False,
        optimize_fn=optimize_fn,
        accum=cfg.training.accum,
        train_temperature=cfg.d_perflow.get('train_temperature', 1.0),
        debug_log_file=debug_log_file,
        lambda_fwd_kl=lambda_fwd_kl,
        lambda_rev_kl=lambda_rev_kl,
        lambda_perceptual=lambda_perceptual,
        perceptual_loss_fn=perceptual_loss_fn,
        teacher_method=teacher_method,
        student_method=student_method,
    )

    # D-PeRFlow sampler for snapshot sampling
    if cfg.training.snapshot_sampling:
        sampling_shape = (
            cfg.training.batch_size // (cfg.ngpus * cfg.training.accum),
            cfg.model.length
        )
        d_perflow_sampler = d_perflow.get_d_perflow_sampler(
            graph=graph,
            noise=noise,
            num_time_windows=cfg.d_perflow.num_time_windows,
            sampling_eps=cfg.d_perflow.sampling_eps
        )

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting D-PeRFlow (pure KL) training at step {initial_step}.")
    mprint(f"Number of time windows: {cfg.d_perflow.num_time_windows}")
    mprint(f"Teacher method: {teacher_method} (steps: {cfg.d_perflow.teacher_steps})")
    mprint(f"Reverse KL weight (lambda_rev_kl): {lambda_rev_kl}")
    mprint(f"Perceptual loss weight (lambda_perceptual): {lambda_perceptual}")
    loss_str = "fwd_KL"
    if lambda_rev_kl > 0:
        loss_str += f" + {lambda_rev_kl} * rev_KL"
    if lambda_perceptual > 0:
        loss_str += f" + {lambda_perceptual} * perceptual"
    mprint(f"Loss: {loss_str}")

    while state['step'] < num_train_steps + 1:
        step = state['step']

        # Get batch
        if cfg.data.train != "text8":
            batch = next(train_iter)['input_ids'].to(device)
        else:
            batch = next(train_iter).to(device)

        # Training step
        loss = train_step_fn(state, batch)

        # Check if step was incremented (full accumulation)
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                if distributed:
                    dist.all_reduce(loss)
                    loss /= world_size
                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))

            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                if cfg.data.valid != "text8":
                    eval_batch = next(eval_iter)['input_ids'].to(device)
                else:
                    eval_batch = next(train_iter).to(device)
                eval_loss = eval_step_fn(state, eval_batch)

                if distributed:
                    dist.all_reduce(eval_loss)
                    eval_loss /= world_size
                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save checkpoint
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(
                        os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'),
                        state
                    )

                # Generate samples
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step} with {cfg.d_perflow.num_time_windows}-step sampling")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(student_model.parameters())
                    ema.copy_to(student_model.parameters())

                    # Use D-PeRFlow sampler
                    sample = d_perflow_sampler.sample(
                        student_model,
                        sampling_shape,
                        device
                    )

                    ema.restore(student_model.parameters())

                    sentences = tokenizer.batch_decode(sample)

                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("=" * 92 + "\n")

                    # Compute perplexity
                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                            batches = sample.shape[0] // cfg.eval.perplexity_batch_size
                            total_perplexity = 0

                            for i in range(batches):
                                s = sample[i * cfg.eval.perplexity_batch_size:(i + 1) * cfg.eval.perplexity_batch_size]
                                loss_val, logits = eval_model(s, labels=s)[:2]
                                logits = logits.transpose(-1, -2)
                                perplexity = F.cross_entropy(
                                    logits[..., :-1], s[..., 1:], reduction="none"
                                ).mean(dim=-1).exp().mean()
                                total_perplexity += perplexity

                            total_perplexity /= batches
                            if distributed:
                                dist.all_reduce(total_perplexity)
                                total_perplexity /= world_size
                            # --- Lightweight diversity metrics ---
                            # Distinct-n on decoded text
                            texts = tokenizer.batch_decode(sample)

                            def _distinct_n(texts, n):
                                all_ngrams = []
                                for t in texts:
                                    tokens = t.split()
                                    if len(tokens) >= n:
                                        all_ngrams.extend(
                                            tuple(tokens[j:j+n])
                                            for j in range(len(tokens) - n + 1)
                                        )
                                return len(set(all_ngrams)) / max(len(all_ngrams), 1)

                            distinct1 = _distinct_n(texts, 1)
                            distinct2 = _distinct_n(texts, 2)
                            distinct3 = _distinct_n(texts, 3)

                            # Token-level entropy & unique ratio
                            flat_tokens = sample.reshape(-1).tolist()
                            tok_counts = Counter(flat_tokens)
                            total_tok = len(flat_tokens)
                            tok_entropy = -sum(
                                (c / total_tok) * math.log2(c / total_tok)
                                for c in tok_counts.values()
                            )
                            unique_ratio = len(tok_counts) / total_tok

                            mprint(
                                f"Eval at step {step} "
                                f"({cfg.d_perflow.num_time_windows}-step sampling):\n"
                                f"  PPL: {total_perplexity:.3f} | "
                                f"Distinct-1: {distinct1:.4f} | "
                                f"Distinct-2: {distinct2:.4f} | "
                                f"Distinct-3: {distinct3:.4f}\n"
                                f"  Token Entropy: {tok_entropy:.3f} bits | "
                                f"Unique Token Ratio: {unique_ratio:.4f} "
                                f"({len(tok_counts)}/{total_tok})"
                            )

                            del eval_model, logits, loss_val

                    if distributed:
                        dist.barrier()


@hydra.main(version_base=None, config_path="configs", config_name="d_perflow")
def main(cfg: DictConfig):
    # Add work_dir based on hydra
    cfg.work_dir = os.getcwd()

    ngpus = cfg.ngpus
    port = int(np.random.randint(10000, 20000))

    if ngpus > 1:
        torch.multiprocessing.spawn(
            run_multiprocess,
            args=(ngpus, cfg, port),
            nprocs=ngpus,
            join=True
        )
    else:
        _run(0, 1, cfg)


if __name__ == "__main__":
    main()
