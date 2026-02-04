"""
Evaluation script for D-PeRFlow model.
Generates samples in batches and computes PPL on all samples together.

Usage:
    python eval_d_perflow.py --checkpoint checkpoints-meta/checkpoint_2002.pth \
        --num_samples 1024 --batch_size 32 --device cuda:0
"""

import torch
import torch.nn.functional as F
import sys
import os

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import noise_lib
import graph_lib
from model import SEDD
from model import utils as mutils
from model.ema import ExponentialMovingAverage
from d_perflow import DPerflowSampler
import losses
from transformers import GPT2LMHeadModel


def load_model(cfg, checkpoint_path, device):
    """Load trained model from checkpoint."""
    # Create model
    model = SEDD(cfg)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DDP state dict
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    return model


def compute_perplexity(samples, batch_size=8):
    """Compute perplexity using GPT-2, consistent with SEDD evaluation.

    Uses E[exp(L)]: per-sample cross-entropy -> exp -> average.
    """
    device = samples.device

    # Load GPT-2
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
    gpt2_model.eval()

    total_perplexity = 0.0
    total_batches = 0

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]

            # Get logits from GPT-2
            outputs = gpt2_model(input_ids=batch, labels=batch)
            logits = outputs.logits.transpose(-1, -2)  # [B, V, L]

            # Per-sample cross-entropy, then exp, then average (consistent with SEDD)
            perplexity = F.cross_entropy(
                logits[..., :-1], batch[..., 1:], reduction="none"
            ).mean(dim=-1).exp().mean()

            total_perplexity += perplexity.item()
            total_batches += 1

            print(f"  Batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}, "
                  f"batch_ppl: {perplexity.item():.4f}")

    avg_perplexity = total_perplexity / total_batches

    return avg_perplexity


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to D-PeRFlow checkpoint')
    parser.add_argument('--num_samples', type=int, default=1024,
                        help='Total number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for generation (per iteration)')
    parser.add_argument('--ppl_batch_size', type=int, default=8,
                        help='Batch size for PPL computation')
    parser.add_argument('--num_time_windows', type=int, default=8,
                        help='Number of time windows (sampling steps)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    args = parser.parse_args()

    # Load config manually (merge base config with model config)
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    base_cfg = OmegaConf.load(os.path.join(config_dir, "d_perflow.yaml"))
    model_cfg = OmegaConf.load(os.path.join(config_dir, "model", "small.yaml"))

    # Merge model config into base config
    base_cfg.model = model_cfg
    cfg = base_cfg

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(cfg, args.checkpoint, device)

    # Create sampler
    noise = noise_lib.get_noise(cfg).to(device)
    graph = graph_lib.get_graph(cfg, device)

    sampler = DPerflowSampler(
        graph=graph,
        noise=noise,
        num_time_windows=args.num_time_windows,
        sampling_eps=cfg.d_perflow.sampling_eps
    )

    # Generate samples in batches
    print(f"\nGenerating {args.num_samples} samples with {args.num_time_windows} steps...")
    all_samples = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
            batch_dims = (current_batch_size, cfg.model.length)

            samples = sampler.sample(model, batch_dims, device)
            all_samples.append(samples)

            print(f"  Generated batch {i+1}/{num_batches} ({current_batch_size} samples)")

    all_samples = torch.cat(all_samples, dim=0)
    print(f"Total samples generated: {all_samples.shape[0]}")

    # Compute perplexity
    print(f"\nComputing perplexity...")
    ppl = compute_perplexity(all_samples, batch_size=args.ppl_batch_size)

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Steps (K): {args.num_time_windows}")
    print(f"  NFEs: {args.num_time_windows}")
    print(f"  PPL: {ppl:.3f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
