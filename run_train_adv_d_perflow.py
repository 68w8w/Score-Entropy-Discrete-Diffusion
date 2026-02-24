"""
Adversarial D-PeRFlow Training Script

Trains a student model using D-PeRFlow with adversarial distillation.
Combines KL divergence with a discriminator loss for improved sample quality.

Usage:
    # Single GPU
    python run_train_adv_d_perflow.py --config configs/adv_d_perflow.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 run_train_adv_d_perflow.py --config configs/adv_d_perflow.yaml
"""

import datetime
import os
import os.path
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
import d_perflow
from adversarial_distillation import (
    create_discriminator,
    AdversarialDistillationLoss,
)
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
    """Load pre-trained teacher model from checkpoint."""
    teacher_model = SEDD(cfg).to(device)

    if cfg.d_perflow.teacher_checkpoint:
        checkpoint = torch.load(
            cfg.d_perflow.teacher_checkpoint,
            map_location=device
        )
        state_dict = checkpoint.get('model', checkpoint)
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        teacher_model.load_state_dict(new_state_dict)
        print(f"Loaded teacher model from {cfg.d_perflow.teacher_checkpoint}")

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    return teacher_model


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir
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

    # Build student model
    mprint("Initializing student model from teacher weights...")
    student_model = SEDD(cfg).to(device)

    if cfg.d_perflow.init_from_teacher:
        student_model.load_state_dict(teacher_model.state_dict())
        mprint("Student initialized from teacher weights.")

    if distributed:
        student_model = DDP(student_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in student_model.parameters())
    mprint(f"Number of parameters in student model: {num_parameters}")

    # EMA for student
    ema = ExponentialMovingAverage(student_model.parameters(), decay=cfg.training.ema)

    # =========================================================================
    # Create projected discriminator using teacher embeddings
    # =========================================================================
    adv_cfg = cfg.adversarial

    # Extract teacher's pre-trained token embedding for projected discrimination.
    # This provides a semantically organized V->D_model projection without
    # learning a 50258-dim linear layer from scratch.
    teacher_embed_weight = teacher_model.vocab_embed.embedding.data  # [V, D_model]
    mprint(f"Teacher embedding shape: {teacher_embed_weight.shape}")

    discriminator = create_discriminator(
        teacher_embed_weight=teacher_embed_weight,
        hidden_size=adv_cfg.disc_hidden_size,
        n_heads=adv_cfg.disc_n_heads,
        n_blocks=adv_cfg.disc_n_blocks,
        dropout=adv_cfg.disc_dropout,
        max_seq_len=cfg.model.length,
    ).to(device)

    if distributed:
        # broadcast_buffers=False is required because the R1 gradient penalty
        # uses create_graph=True, and DDP buffer broadcasting is an in-place
        # operation that would corrupt the autograd version tracking.
        # The only buffer (embed_weight) is already identical across ranks
        # (cloned from the same teacher checkpoint), so no sync is needed.
        discriminator = DDP(discriminator, device_ids=[rank],
                            find_unused_parameters=True,
                            broadcast_buffers=False)

    disc_params = sum(p.numel() for p in discriminator.parameters())
    mprint(f"Discriminator parameters: {disc_params}")

    # Wrap in adversarial loss module
    adv_loss_module = AdversarialDistillationLoss(
        discriminator=discriminator,
        lambda_adv=adv_cfg.lambda_adv,
        r1_gamma=adv_cfg.r1_gamma,
    ).to(device)

    # Discriminator optimizer
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=adv_cfg.disc_lr,
        betas=(adv_cfg.disc_beta1, adv_cfg.disc_beta2),
        weight_decay=adv_cfg.disc_weight_decay,
    )
    mprint(f"Discriminator optimizer: {disc_optimizer}")

    # =========================================================================
    # Student optimizer
    # =========================================================================
    optimizer = losses.get_optimizer(cfg, student_model.parameters())
    mprint(f"Student optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()

    state = dict(
        optimizer=optimizer,
        scaler=scaler,
        model=student_model,
        noise=noise,
        ema=ema,
        step=0,
        disc_metrics={},
        gen_metrics={},
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

    # Optimization function
    optimize_fn = losses.optimization_manager(cfg)

    # Debug log file
    debug_log_file = os.path.join(work_dir, 'debug_log.txt')
    mprint(f"Debug log will be saved to: {debug_log_file}")

    # =========================================================================
    # Create adversarial training step function
    # =========================================================================
    train_step_fn = d_perflow.get_adversarial_d_perflow_step_fn(
        noise=noise,
        graph=graph,
        teacher_model=teacher_model,
        adv_loss_module=adv_loss_module,
        num_time_windows=cfg.d_perflow.num_time_windows,
        delta_t=cfg.d_perflow.delta_t,
        sampling_eps=cfg.d_perflow.sampling_eps,
        euler_steps=cfg.d_perflow.euler_steps,
        train=True,
        optimize_fn=optimize_fn,
        disc_optimizer=disc_optimizer,
        accum=cfg.training.accum,
        disc_steps_per_gen=adv_cfg.disc_steps_per_gen,
        debug_log_file=debug_log_file,
    )

    eval_step_fn = d_perflow.get_adversarial_d_perflow_step_fn(
        noise=noise,
        graph=graph,
        teacher_model=teacher_model,
        adv_loss_module=adv_loss_module,
        num_time_windows=cfg.d_perflow.num_time_windows,
        delta_t=cfg.d_perflow.delta_t,
        sampling_eps=cfg.d_perflow.sampling_eps,
        euler_steps=cfg.d_perflow.euler_steps,
        train=False,
        optimize_fn=optimize_fn,
        disc_optimizer=disc_optimizer,
        accum=cfg.training.accum,
        disc_steps_per_gen=adv_cfg.disc_steps_per_gen,
        debug_log_file=debug_log_file,
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
    mprint(f"Starting Adversarial D-PeRFlow training at step {initial_step}.")
    mprint(f"Number of time windows: {cfg.d_perflow.num_time_windows}")
    mprint(f"Adversarial weight (lambda_adv): {adv_cfg.lambda_adv}")
    mprint(f"Discriminator steps per generator step: {adv_cfg.disc_steps_per_gen}")
    mprint(f"R1 gradient penalty: {adv_cfg.r1_gamma}")

    while state['step'] < num_train_steps + 1:
        step = state['step']

        # Get batch
        if cfg.data.train != "text8":
            batch = next(train_iter)['input_ids'].to(device)
        else:
            batch = next(train_iter).to(device)

        # Training step (alternates disc/gen updates internally)
        loss = train_step_fn(state, batch)

        # Check if step was incremented (full accumulation for gen update)
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                if distributed:
                    dist.all_reduce(loss)
                    loss /= world_size

                # Log both generator and discriminator metrics
                gen_metrics = state.get('gen_metrics', {})
                disc_metrics = state.get('disc_metrics', {})

                log_msg = "step: %d, loss: %.5e" % (step, loss.item())
                if gen_metrics:
                    log_msg += (
                        f", kl: {gen_metrics.get('gen_kl_loss', 0):.6f}"
                        f", adv: {gen_metrics.get('gen_adv_loss', 0):.6f}"
                        f", total: {gen_metrics.get('gen_total_loss', 0):.6f}"
                    )
                if disc_metrics:
                    d_real = disc_metrics.get('disc_real_logit_mean', 0)
                    d_fake = disc_metrics.get('disc_fake_logit_mean', 0)
                    log_msg += (
                        f", disc: {disc_metrics.get('disc_loss', 0):.4f}"
                        f", d_tok: {disc_metrics.get('disc_token_loss', 0):.4f}"
                        f", d_seq: {disc_metrics.get('disc_seq_loss', 0):.4f}"
                        f", d_gap: {d_real - d_fake:.3f}"
                        f", tok_acc_r: {disc_metrics.get('disc_real_tok_acc', 0):.2f}"
                        f", tok_acc_f: {disc_metrics.get('disc_fake_tok_acc', 0):.2f}"
                        f", r1: {disc_metrics.get('disc_r1_penalty', 0):.4f}"
                    )
                mprint(log_msg)

                # Also write to debug log file for persistent tracking
                if rank == 0:
                    with open(debug_log_file, 'a') as f:
                        f.write(f"[TRAIN] {log_msg}\n")

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

                    # Also save discriminator checkpoint
                    disc_state = {
                        'discriminator': discriminator.state_dict(),
                        'disc_optimizer': disc_optimizer.state_dict(),
                        'step': step,
                    }
                    torch.save(
                        disc_state,
                        os.path.join(checkpoint_dir, f'disc_checkpoint_{save_step}.pth'),
                    )

                # Generate samples
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step} with {cfg.d_perflow.num_time_windows}-step sampling")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(student_model.parameters())
                    ema.copy_to(student_model.parameters())

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
                            mprint(f"Generative Perplexity at step: {step}. "
                                   f"Perplexity: {total_perplexity:.3f}. "
                                   f"(Using {cfg.d_perflow.num_time_windows}-step sampling)")

                            del eval_model, logits, loss_val

                    if distributed:
                        dist.barrier()


@hydra.main(version_base=None, config_path="configs", config_name="adv_d_perflow")
def main(cfg: DictConfig):
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
