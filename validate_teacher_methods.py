"""
Quick validation script to compare teacher distribution methods:
  - Euler (Taylor approximation: I + dt*Q)
  - Analytic multi-step (exact exp(ΔσQ), 1st-order Exponential Integrator)
  - Exp-Midpoint (2nd-order Exponential Integrator)

Evaluation approach:
  1. Take real text x_0 from the dataset
  2. Add noise to get x_{t_k} at various time windows
  3. Use each method to compute P_{t_{k-1}} (reverse distribution)
  4. Measure how much probability mass P_{t_{k-1}} places on the true x_0 tokens
     (higher = more accurate reverse process)
  5. Also measure distribution entropy and KL divergence between methods

This does NOT require training - it directly measures teacher quality.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from load_model import load_model
from model import utils as mutils
from d_perflow import DPerflowTrainer
from transformers import GPT2TokenizerFast
import noise_lib


def compute_metrics(probs, x_0, method_name):
    """Compute quality metrics for a predicted distribution."""
    B, L, V = probs.shape

    # 1. Average probability on true tokens (higher = better)
    true_token_probs = probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    avg_true_prob = true_token_probs.mean().item()

    # 2. Average log-probability on true tokens (higher = better, more numerically stable)
    avg_log_prob = (true_token_probs + 1e-10).log().mean().item()

    # 3. Distribution entropy (informational)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()

    # 4. Top-1 accuracy: does argmax match x_0?
    pred_tokens = probs.argmax(dim=-1)  # [B, L]
    top1_acc = (pred_tokens == x_0).float().mean().item()

    # 5. Fraction of negative probabilities before clamping (for Euler only, informational)
    return {
        "method": method_name,
        "avg_true_prob": avg_true_prob,
        "avg_log_prob": avg_log_prob,
        "entropy": entropy,
        "top1_accuracy": top1_acc,
    }


def compute_kl_divergence(p, q):
    """Compute KL(P || Q) element-wise, averaged."""
    p_safe = p.clamp(min=1e-10)
    q_safe = q.clamp(min=1e-10)
    kl = (p_safe * (p_safe.log() - q_safe.log())).sum(dim=-1).mean().item()
    return kl


def main():
    parser = argparse.ArgumentParser(description="Compare teacher distribution methods")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str,
                        help="HuggingFace model ID or local training run directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_time_windows", type=int, default=16,
                        help="K: number of time windows (larger = smaller steps = easier)")
    parser.add_argument("--euler_steps", type=int, default=4,
                        help="Number of sub-steps for Euler and Analytic methods")
    parser.add_argument("--num_batches", type=int, default=4,
                        help="Number of batches to average over")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for datasets (optional)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, graph, noise = load_model(args.model_path, device)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Get score function
    score_fn = mutils.get_score_fn(model, train=False, sampling=True)

    # Create trainer
    trainer = DPerflowTrainer(
        graph=graph,
        noise=noise,
        num_time_windows=args.num_time_windows,
    )

    # Load real text data
    print("Loading dataset...")
    from datasets import load_dataset
    load_kwargs = {}
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = args.cache_dir
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", **load_kwargs)
    texts = [t for t in ds["text"] if len(t.strip()) > 100]

    # Tokenize
    all_tokens = []
    for text in texts[:200]:
        tokens = tokenizer.encode(text)
        if len(tokens) >= args.seq_len:
            all_tokens.append(tokens[:args.seq_len])
    all_tokens = torch.tensor(all_tokens, device=device)
    print(f"Prepared {len(all_tokens)} sequences of length {args.seq_len}")

    # Results storage
    all_results = {
        "euler": [], "analytic": [], "exp_midpoint": []
    }

    # Test across different windows
    K = args.num_time_windows
    windows_to_test = list(range(1, K + 1))

    print(f"\n{'='*80}")
    print(f"Comparing teacher methods: K={K}, euler_steps={args.euler_steps}")
    print(f"{'='*80}\n")

    for window_k in windows_to_test:
        window_results = {"euler": [], "analytic": [], "exp_midpoint": []}

        for batch_idx in range(args.num_batches):
            # Sample a batch of real data
            idx = torch.randint(0, len(all_tokens), (args.batch_size,))
            x_0 = all_tokens[idx]  # [B, L]

            # Get window boundaries
            k_tensor = torch.full((args.batch_size,), window_k, device=device, dtype=torch.long)
            t_k_minus_1, t_k = trainer.get_window_boundaries(k_tensor, device)
            t_k = t_k.unsqueeze(-1)
            t_k_minus_1 = t_k_minus_1.unsqueeze(-1)

            # Construct noisy state at t_k
            with torch.no_grad():
                x_t_k = trainer.construct_noisy_state(x_0, t_k)

                # Count how many tokens are masked
                mask_token = graph.dim - 1
                mask_ratio = (x_t_k == mask_token).float().mean().item()

                # === Method 1: Euler ===
                P_euler = trainer.compute_euler_distribution(
                    score_fn, x_t_k, t_k, t_k_minus_1, num_steps=args.euler_steps
                )
                metrics_euler = compute_metrics(P_euler, x_0, "euler")
                window_results["euler"].append(metrics_euler)

                # === Method 2: Analytic multi-step ===
                P_analytic = trainer.compute_analytic_multi_step(
                    score_fn, x_t_k, t_k, t_k_minus_1, num_steps=args.euler_steps
                )
                metrics_analytic = compute_metrics(P_analytic, x_0, "analytic")
                window_results["analytic"].append(metrics_analytic)

                # === Method 3: Exp-Midpoint ===
                P_midpoint = trainer.compute_exp_midpoint_distribution(
                    score_fn, x_t_k, t_k, t_k_minus_1
                )
                metrics_midpoint = compute_metrics(P_midpoint, x_0, "exp_midpoint")
                window_results["exp_midpoint"].append(metrics_midpoint)

        # Average metrics for this window
        t_k_val = trainer.time_boundaries[window_k].item()
        t_km1_val = trainer.time_boundaries[window_k - 1].item()
        print(f"Window k={window_k}/{K}  t: {t_k_val:.4f} -> {t_km1_val:.4f}  "
              f"(mask_ratio ≈ {mask_ratio:.2f})")
        print(f"  {'Method':<16} {'AvgTrueProb':>12} {'AvgLogProb':>12} {'Top1Acc':>10} {'Entropy':>10}")
        print(f"  {'-'*62}")

        for method in ["euler", "analytic", "exp_midpoint"]:
            results = window_results[method]
            avg = {
                key: np.mean([r[key] for r in results])
                for key in ["avg_true_prob", "avg_log_prob", "top1_accuracy", "entropy"]
            }
            all_results[method].append(avg)
            print(f"  {method:<16} {avg['avg_true_prob']:>12.6f} {avg['avg_log_prob']:>12.4f} "
                  f"{avg['top1_accuracy']:>10.4f} {avg['entropy']:>10.4f}")
        print()

    # Summary: average across all windows
    print(f"\n{'='*80}")
    print("SUMMARY (averaged across all windows)")
    print(f"{'='*80}")
    print(f"  {'Method':<16} {'AvgTrueProb':>12} {'AvgLogProb':>12} {'Top1Acc':>10} {'Entropy':>10}")
    print(f"  {'-'*62}")
    for method in ["euler", "analytic", "exp_midpoint"]:
        results = all_results[method]
        avg = {
            key: np.mean([r[key] for r in results])
            for key in ["avg_true_prob", "avg_log_prob", "top1_accuracy", "entropy"]
        }
        print(f"  {method:<16} {avg['avg_true_prob']:>12.6f} {avg['avg_log_prob']:>12.4f} "
              f"{avg['top1_accuracy']:>10.4f} {avg['entropy']:>10.4f}")

    print(f"\nInterpretation:")
    print(f"  - AvgTrueProb: probability placed on ground truth token (higher = better)")
    print(f"  - AvgLogProb:  log of above (higher = better, more sensitive)")
    print(f"  - Top1Acc:     argmax matches ground truth (higher = better)")
    print(f"  - Entropy:     distribution sharpness (informational)")


if __name__ == "__main__":
    main()
