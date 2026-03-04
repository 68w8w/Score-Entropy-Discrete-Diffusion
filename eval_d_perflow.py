"""
Evaluation script for D-PeRFlow model.
Generates samples in batches and computes PPL, MAUVE, Self-BLEU on all samples.

Usage:
    python eval_d_perflow.py --checkpoint checkpoints-meta/checkpoint_2002.pth \
        --num_samples 1024 --batch_size 32 --device cuda:0

Metrics:
    - PPL: Perplexity using GPT-2 (lower = better quality)
    - MAUVE: Distribution similarity to real text (higher = better, 0-1)
    - Self-BLEU: Sample diversity (lower = more diverse)
    - Distinct-n: N-gram diversity (higher = more diverse)
"""

import torch
import torch.nn.functional as F
import sys
import os
import json

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
import sampling
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


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


def compute_distinct_n(texts, n=2):
    """
    Compute Distinct-n: ratio of unique n-grams to total n-grams.
    Higher = more diverse (0-1).
    """
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        if len(tokens) >= n:
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            all_ngrams.extend(ngrams)

    if len(all_ngrams) == 0:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def compute_self_bleu(texts, n=4, sample_size=100):
    """
    Compute Self-BLEU: average BLEU of each sample against others.
    Lower = more diverse.

    Args:
        texts: List of generated texts
        n: N-gram order for BLEU
        sample_size: Number of samples to use (for speed)
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        print("Warning: nltk not installed. Skipping Self-BLEU. Install with: pip install nltk")
        return None

    # Sample if too many texts
    import random
    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)

    if len(texts) < 2:
        return 0.0

    smoothing = SmoothingFunction().method1
    scores = []

    # Compute BLEU for each sample against all others
    for i, text in enumerate(texts):
        hypothesis = text.split()
        references = [t.split() for j, t in enumerate(texts) if j != i]

        if len(hypothesis) == 0 or len(references) == 0:
            continue

        try:
            # Use uniform weights up to n-gram
            weights = [1.0/n] * n
            score = sentence_bleu(references, hypothesis,
                                  weights=weights,
                                  smoothing_function=smoothing)
            scores.append(score)
        except:
            continue

    if len(scores) == 0:
        return 0.0

    return sum(scores) / len(scores)


def compute_mauve(generated_texts, reference_texts=None, device_id=0, max_len=256):
    """
    Compute MAUVE score: distribution similarity between generated and real text.
    Higher = better (0-1).

    Args:
        generated_texts: List of generated texts
        reference_texts: List of reference texts (if None, uses wikitext)
        device_id: GPU device ID
        max_len: Maximum text length for MAUVE computation
    """
    try:
        import mauve
    except ImportError:
        print("Warning: mauve-text not installed. Skipping MAUVE. Install with: pip install mauve-text")
        return None

    # If no reference texts provided, load some from a dataset
    if reference_texts is None:
        try:
            from datasets import load_dataset
            print("  Loading reference texts from wikitext-103...")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            # Filter and get similar number of samples
            reference_texts = [t for t in dataset["text"] if len(t.split()) > 20]
            reference_texts = reference_texts[:len(generated_texts)]
        except Exception as e:
            print(f"Warning: Could not load reference dataset: {e}")
            print("  Skipping MAUVE computation.")
            return None

    # Truncate texts to max_len tokens for efficiency
    def truncate(text, max_tokens=max_len):
        tokens = text.split()[:max_tokens]
        return " ".join(tokens)

    generated_texts = [truncate(t) for t in generated_texts]
    reference_texts = [truncate(t) for t in reference_texts]

    # Filter empty texts
    generated_texts = [t for t in generated_texts if len(t.strip()) > 0]
    reference_texts = [t for t in reference_texts if len(t.strip()) > 0]

    if len(generated_texts) == 0 or len(reference_texts) == 0:
        print("Warning: No valid texts for MAUVE computation.")
        return None

    print(f"  Computing MAUVE with {len(generated_texts)} generated and {len(reference_texts)} reference texts...")

    try:
        result = mauve.compute_mauve(
            p_text=reference_texts,
            q_text=generated_texts,
            device_id=device_id,
            max_text_length=max_len,
            verbose=False
        )
        return result.mauve
    except Exception as e:
        print(f"Warning: MAUVE computation failed: {e}")
        return None


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
    parser.add_argument('--sampler', type=str, default='d_perflow',
                        choices=['d_perflow', 'sedd'],
                        help='Sampler to use: d_perflow or sedd (original SEDD sampler)')
    parser.add_argument('--sedd_steps', type=int, default=128,
                        help='Number of steps for SEDD sampler (only used when --sampler=sedd)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for softmax (higher = more diverse outputs)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to print distribution statistics at each step')
    parser.add_argument('--debug_log_file', type=str, default=None,
                        help='Path to save debug logs (if None, print to console only)')
    parser.add_argument('--compute_mauve', action='store_true',
                        help='Compute MAUVE score (requires mauve-text package)')
    parser.add_argument('--compute_self_bleu', action='store_true',
                        help='Compute Self-BLEU score (requires nltk package)')
    parser.add_argument('--compute_distinct', action='store_true',
                        help='Compute Distinct-n scores')
    parser.add_argument('--all_metrics', action='store_true',
                        help='Compute all metrics (PPL, MAUVE, Self-BLEU, Distinct-n)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation results as JSON (e.g., eval_results.json)')
    args = parser.parse_args()

    # If --all_metrics, enable all
    if args.all_metrics:
        args.compute_mauve = True
        args.compute_self_bleu = True
        args.compute_distinct = True

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

    if args.sampler == 'd_perflow':
        sampler = DPerflowSampler(
            graph=graph,
            noise=noise,
            num_time_windows=args.num_time_windows,
            sampling_eps=cfg.d_perflow.sampling_eps,
            temperature=args.temperature,
            debug=args.debug,
            debug_log_file=args.debug_log_file
        )
        steps_info = f"{args.num_time_windows} steps (D-PeRFlow, T={args.temperature})"
    else:
        # Use SEDD original sampler (AnalyticPredictor)
        sampler = None  # Will use get_pc_sampler directly
        steps_info = f"{args.sedd_steps} steps (SEDD AnalyticPredictor)"

    # Generate samples in batches
    print(f"\nGenerating {args.num_samples} samples with {steps_info}...")
    all_samples = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
            batch_dims = (current_batch_size, cfg.model.length)

            if args.sampler == 'd_perflow':
                samples = sampler.sample(model, batch_dims, device)
            else:
                # Use SEDD original sampler
                sedd_sampler = sampling.get_pc_sampler(
                    graph=graph,
                    noise=noise,
                    batch_dims=batch_dims,
                    predictor='analytic',
                    steps=args.sedd_steps,
                    denoise=True,
                    eps=1e-5,
                    device=device
                )
                samples = sedd_sampler(model)
            all_samples.append(samples)

            print(f"  Generated batch {i+1}/{num_batches} ({current_batch_size} samples)")

    all_samples = torch.cat(all_samples, dim=0)
    print(f"Total samples generated: {all_samples.shape[0]}")

    # Decode all samples to text
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    all_texts = [tokenizer.decode(sample) for sample in all_samples]

    # Print sample texts for quality inspection
    print(f"\n{'='*50}")
    print("Sample texts (first 5):")
    print(f"{'='*50}")
    for idx in range(min(5, len(all_texts))):
        print(f"\n--- Sample {idx+1} ---")
        print(all_texts[idx][:500])
    print(f"\n{'='*50}")

    # Initialize results dict
    results = {}

    # Compute perplexity (always computed)
    print(f"\nComputing perplexity...")
    ppl = compute_perplexity(all_samples, batch_size=args.ppl_batch_size)
    results['PPL'] = ppl

    # Compute Distinct-n
    if args.compute_distinct:
        print(f"\nComputing Distinct-n...")
        distinct_1 = compute_distinct_n(all_texts, n=1)
        distinct_2 = compute_distinct_n(all_texts, n=2)
        distinct_3 = compute_distinct_n(all_texts, n=3)
        results['Distinct-1'] = distinct_1
        results['Distinct-2'] = distinct_2
        results['Distinct-3'] = distinct_3
        print(f"  Distinct-1: {distinct_1:.4f}")
        print(f"  Distinct-2: {distinct_2:.4f}")
        print(f"  Distinct-3: {distinct_3:.4f}")

    # Compute Self-BLEU
    if args.compute_self_bleu:
        print(f"\nComputing Self-BLEU...")
        self_bleu = compute_self_bleu(all_texts, n=4, sample_size=100)
        if self_bleu is not None:
            results['Self-BLEU'] = self_bleu
            print(f"  Self-BLEU: {self_bleu:.4f}")

    # Compute MAUVE
    if args.compute_mauve:
        print(f"\nComputing MAUVE...")
        # Extract device ID from device string
        device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        mauve_score = compute_mauve(all_texts, device_id=device_id)
        if mauve_score is not None:
            results['MAUVE'] = mauve_score
            print(f"  MAUVE: {mauve_score:.4f}")

    # Print final results
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"{'='*60}")
    print(f"  Sampler: {args.sampler}")
    print(f"  Samples: {args.num_samples}")
    if args.sampler == 'd_perflow':
        print(f"  Steps (K): {args.num_time_windows}")
        print(f"  NFEs: {args.num_time_windows}")
    else:
        print(f"  Steps: {args.sedd_steps}")
        print(f"  NFEs: {args.sedd_steps}")
    print(f"-" * 60)
    print(f"  PPL: {results['PPL']:.3f} (lower = better quality)")
    if 'Distinct-1' in results:
        print(f"  Distinct-1: {results['Distinct-1']:.4f} (higher = more diverse)")
        print(f"  Distinct-2: {results['Distinct-2']:.4f} (higher = more diverse)")
        print(f"  Distinct-3: {results['Distinct-3']:.4f} (higher = more diverse)")
    if 'Self-BLEU' in results:
        print(f"  Self-BLEU: {results['Self-BLEU']:.4f} (lower = more diverse)")
    if 'MAUVE' in results:
        print(f"  MAUVE: {results['MAUVE']:.4f} (higher = closer to real text)")
    print(f"{'='*60}")

    # Save results to file
    if args.output_file:
        save_data = {
            'sampler': args.sampler,
            'num_samples': args.num_samples,
            'steps': args.num_time_windows if args.sampler == 'd_perflow' else args.sedd_steps,
            'temperature': args.temperature,
            'checkpoint': args.checkpoint,
            'metrics': results,
        }
        with open(args.output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()
