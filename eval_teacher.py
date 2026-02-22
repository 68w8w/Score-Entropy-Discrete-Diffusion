"""
Evaluate teacher model (original SEDD) PPL at various step counts.

Generates samples using the SEDD AnalyticPredictor at different steps
and computes GPT-2 perplexity for comparison with D-PeRFlow student.

Usage:
    # Evaluate teacher at 8, 16, 32, 64, 128 steps
    python eval_teacher.py --checkpoint path/to/teacher.pth \
        --steps 8 16 32 64 128 --num_samples 256 --batch_size 32

    # Quick test
    python eval_teacher.py --checkpoint path/to/teacher.pth \
        --steps 16 128 --num_samples 64 --batch_size 16
"""

import argparse
import os
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import noise_lib
import graph_lib
import sampling
from model import SEDD
from model.ema import ExponentialMovingAverage


def load_teacher(cfg, checkpoint_path, device):
    """Load teacher model from checkpoint."""
    model = SEDD(cfg)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Support multiple checkpoint formats
    if 'ema' in checkpoint:
        # If EMA weights are saved, use them (they are the best weights)
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
        ema.load_state_dict(checkpoint['ema'])
        ema.copy_to(model.parameters())
        print("Loaded EMA weights from checkpoint.")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        # Remove 'module.' prefix from DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.removeprefix('module.')] = v
        model.load_state_dict(new_state_dict)
        print("Loaded model weights from checkpoint.")
    else:
        # Assume the checkpoint is the state_dict itself
        model.load_state_dict(checkpoint)
        print("Loaded raw state_dict from checkpoint.")

    model = model.to(device)
    model.eval()
    return model


def compute_perplexity(samples, batch_size, device):
    """Compute perplexity using GPT-2 large, consistent with SEDD/D-PeRFlow eval."""
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
    gpt2.eval()

    total_ppl = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            outputs = gpt2(input_ids=batch, labels=batch)
            logits = outputs.logits.transpose(-1, -2)  # [B, V, L]

            ppl = F.cross_entropy(
                logits[..., :-1], batch[..., 1:], reduction="none"
            ).mean(dim=-1).exp().mean()

            total_ppl += ppl.item()
            num_batches += 1

    del gpt2
    torch.cuda.empty_cache()

    return total_ppl / num_batches


def generate_samples(model, graph, noise, num_samples, batch_size, steps, device, seq_len, predictor='analytic'):
    """Generate samples using SEDD sampler."""
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            cur_bs = min(batch_size, num_samples - i * batch_size)
            batch_dims = (cur_bs, seq_len)

            sampler_fn = sampling.get_pc_sampler(
                graph=graph,
                noise=noise,
                batch_dims=batch_dims,
                predictor=predictor,
                steps=steps,
                denoise=True,
                eps=1e-5,
                device=device,
            )
            samples = sampler_fn(model)
            all_samples.append(samples)
            print(f"    batch {i+1}/{num_batches} ({cur_bs} samples)")

    return torch.cat(all_samples, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEDD teacher model PPL")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to teacher checkpoint')
    parser.add_argument('--steps', type=int, nargs='+', default=[8, 16, 32, 64, 128],
                        help='List of sampling steps to evaluate')
    parser.add_argument('--num_samples', type=int, default=256,
                        help='Number of samples to generate per step count')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for generation')
    parser.add_argument('--ppl_batch_size', type=int, default=8,
                        help='Batch size for PPL computation')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--predictor', type=str, default='analytic',
                        choices=['analytic', 'euler'],
                        help='SEDD predictor type')
    parser.add_argument('--show_samples', type=int, default=3,
                        help='Number of sample texts to display per step count')
    args = parser.parse_args()

    # Load config
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    base_cfg = OmegaConf.load(os.path.join(config_dir, "d_perflow.yaml"))
    model_cfg = OmegaConf.load(os.path.join(config_dir, "model", "small.yaml"))
    base_cfg.model = model_cfg
    cfg = base_cfg

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"Loading teacher from: {args.checkpoint}")
    model = load_teacher(cfg, args.checkpoint, device)

    # Build graph & noise
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    seq_len = cfg.model.length  # 1024

    # Results table
    results = []

    print(f"\n{'='*60}")
    print(f"Teacher Model Evaluation")
    print(f"  Predictor: {args.predictor}")
    print(f"  Samples per step count: {args.num_samples}")
    print(f"  Steps to evaluate: {args.steps}")
    print(f"{'='*60}\n")

    for step_count in args.steps:
        print(f"--- Steps = {step_count} ---")

        # Generate
        print(f"  Generating {args.num_samples} samples...")
        samples = generate_samples(
            model, graph, noise,
            args.num_samples, args.batch_size,
            step_count, device, seq_len,
            predictor=args.predictor,
        )

        # Show a few samples
        if args.show_samples > 0:
            print(f"\n  Sample texts:")
            for idx in range(min(args.show_samples, len(samples))):
                text = tokenizer.decode(samples[idx])
                print(f"  [Sample {idx+1}] {text[:200]}...")
            print()

        # Compute PPL
        print(f"  Computing PPL...")
        ppl = compute_perplexity(samples, args.ppl_batch_size, device)
        results.append((step_count, ppl))
        print(f"  => Steps={step_count}, PPL={ppl:.2f}\n")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'RESULTS SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'Steps':>8} | {'Predictor':>12} | {'PPL':>10}")
    print(f"{'-'*8}-+-{'-'*12}-+-{'-'*10}")
    for step_count, ppl in results:
        print(f"{step_count:>8} | {args.predictor:>12} | {ppl:>10.2f}")

    # Compare with student
    print(f"\n{'='*60}")
    print(f"{'COMPARISON WITH D-PeRFlow STUDENT':^60}")
    print(f"{'='*60}")
    print(f"{'Model':>20} | {'Steps':>6} | {'PPL':>10}")
    print(f"{'-'*20}-+-{'-'*6}-+-{'-'*10}")
    for step_count, ppl in results:
        print(f"{'Teacher (SEDD)':>20} | {step_count:>6} | {ppl:>10.2f}")
    print(f"{'Student (D-PeRFlow)':>20} | {'16':>6} | {'80.00':>10}")
    print(f"{'Student (D-PeRFlow)':>20} | {'8':>6} | {'160.00':>10}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
