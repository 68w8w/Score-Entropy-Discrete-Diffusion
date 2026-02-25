# Adversarial D-PeRFlow Training Log Analysis

Branch: `claude/adversarial-distillation-integration-WfCcO`
Date: 2026-02-24, 2000 steps on 4x RTX 4090

## Training Configuration

| Parameter | Value |
|-----------|-------|
| GPUs | 4x NVIDIA RTX 4090 (23.64GB) |
| Model | DiT-small: 768 hidden, 12 blocks, 12 heads (169.6M params) |
| Discriminator | ProjectedDisc: 256 hidden, 4 heads, 4 blocks (4.14M params) |
| D-PeRFlow | 16 time windows, teacher 2-step Euler, student 1-step |
| lambda_adv | 0.1 |
| R1 | gamma=10.0, lazy interval=16 |
| Batch size | 16 per GPU |
| LR | student=1e-4, disc=2e-4 |
| Total steps | 2000 (~29 minutes) |
| Data | OpenWebText (train) / WikiText103 (valid) |

## 1. Generative Perplexity — Improving (Positive)

| Step | Perplexity | Delta |
|------|-----------|-------|
| 200  | 246.917   | —     |
| 400  | 244.735   | -2.2  |
| 600  | 237.019   | -7.7  |
| 800  | **214.873** | **-22.1 (best)** |
| 1000 | 225.548   | +10.7 |
| 1200 | 230.362   | +4.8  |
| 1400 | 233.172   | +2.8  |
| 1600 | 218.288   | -14.9 |
| 1800 | 220.554   | +2.3  |
| 2000 | 215.842   | -4.7  |

**Overall: 246.9 → 215.8, ~12.6% improvement.** Best at step 800 (214.87).

The perplexity oscillates but trends downward. This improvement comes **entirely from KL distillation** (see below).

## 2. Discriminator Collapse — Critical Failure

### Timeline of Collapse

| Step Range | tok_acc_r | tok_acc_f | d_gap | Status |
|-----------|-----------|-----------|-------|--------|
| 0-350     | ~0.50     | ~0.50     | ~0.000 | Random guessing from the start |
| 400       | 1.00      | 0.00      | ~0.000 | Momentary flip (1 batch artifact) |
| 450-1200  | ~0.50     | ~0.50     | ~0.000 | Back to random |
| 1250      | 0.19      | 0.82      | ~0.000 | Collapse begins |
| 1300      | 0.10      | 0.89      | ~0.000 | Accelerating |
| 1350-2000 | **0.00**  | **1.00**  | ~0.000 | **Fully collapsed — predicts everything as fake** |

### Key Evidence

1. **Seq logit gap ≈ 0.000 throughout entire training** — discriminator output for real and fake is identical
2. **D(teacher) prob ≈ D(student) prob ≈ 0.48-0.50** — no separation at any point
3. **d_seq loss ≈ 1.3863 = 2×ln(2)** — this is exactly the binary cross-entropy of random guessing
4. **R1 penalty = 0.0000 at all logged steps** — R1 fires every 16 steps but at logging intervals (100), the metrics show 0 because the R1 step didn't coincide

### Adversarial Loss Becomes Constant

| Step | adv_loss | Interpretation |
|------|----------|---------------|
| 0    | 2.621    | Initial (meaningful) |
| 50   | 1.352    | Rapid drop |
| 100+ | 1.35-1.42 | **Flat — no useful gradient signal** |

`adv ≈ 1.386 ≈ 2×ln(2)` = non-saturating loss with 50% discriminator accuracy. The adversarial component provides **zero learning signal** after the first few steps.

## 3. Distribution Quality (from debug_log)

Despite discriminator failure, the student-teacher alignment is good:

| Metric | Step 100 | Step 1000 | Step 2000 |
|--------|----------|-----------|-----------|
| Argmax agreement | 97.2% | 97.0% | 97.0% |
| JS divergence | 0.022 | 0.020 | 0.022 |
| L2 distance | 0.061 | 0.062 | 0.067 |
| Entropy ratio (S/T) | 1.92 | 1.87 | 1.87 |
| Teacher max_prob | 0.974 | 0.969 | 0.970 |
| Student max_prob | 0.949 | 0.939 | 0.940 |

The student has **higher entropy** (~1.87x teacher) — its distributions are more diffuse. This is the subtle difference the discriminator should learn to exploit, but cannot.

## 4. Root Cause Analysis

### Primary cause: Teacher-student outputs are too similar for the discriminator

- Student initialized from teacher weights → 97%+ argmax agreement from step 0
- The distributional difference is subtle (entropy ratio, slightly more diffuse peaks)
- A 4.14M param discriminator with 4 transformer blocks operating on **soft probability → embedding projection** features cannot detect this subtle difference
- The token-level and sequence-level classification signals both converge to random chance immediately

### Secondary causes:

1. **No discriminator warm-up**: The discriminator trains simultaneously with the generator from step 0, never getting a head start to learn real/fake features

2. **Generator backward leaks gradients into discriminator**: In `generator_loss()` (line 380), `self.discriminator(student_probs, x_t, t)` runs without detaching discriminator parameters. The generator backward computes and accumulates gradients on discriminator params, which are stale (not used for disc update but pollute memory). This also explains the misleading `Disc grad norm: 53000-59000` in logs — it captures accumulated generator-phase gradients, not the actual disc training gradients (which ARE properly clipped to 10.0).

3. **Feature space may be too smooth**: The projected discriminator's `softmax → matmul(embed)` projection creates a feature space where teacher and student look nearly identical (since argmax tokens match 97% of the time)

## 5. Evaluation Loss Trend

| Step | eval_loss | Notes |
|------|-----------|-------|
| 0    | 0.4473    | Initial |
| 100  | 0.3118    | Big drop |
| 1100 | 0.3014    | Best |
| 1400 | 0.2816    | **Best overall** |
| 2000 | 0.3432    | Slight regression |

Evaluation loss improves 0.447 → 0.282 (best), showing KL distillation is effective.

## 6. Recommendations for Next Training Run

### A. Fix discriminator to provide useful signal

1. **Discriminator pre-training / warm-up**: Train discriminator for N steps (e.g., 200-500) before starting generator updates, so it learns meaningful features first

2. **Add noise to student init**: Instead of exact teacher weight copy, add small Gaussian noise to student initialization. This creates more distributional difference for the discriminator to learn from

3. **Operate on logit space, not probability space**: Feed raw logits (before softmax) to the discriminator. Logit-space differences are much larger than probability-space differences when distributions are peaked (max_prob > 0.94)

4. **Increase discriminator capacity**: Current 4-block, 4-head, 256-hidden is potentially too small. Try 6-8 blocks, 8 heads, 512 hidden

5. **Spectral normalization**: Add spectral norm to discriminator layers for Lipschitz constraint — this is more effective than R1 alone for stabilizing GAN training

### B. Fix gradient hygiene

6. **Freeze discriminator during generator loss**: In `generator_loss()`, wrap discriminator call with `torch.no_grad()` or set `disc.requires_grad_(False)` before generator forward, `True` after. This saves memory and prevents stale gradient accumulation

### C. Tune hyperparameters

7. **Increase lambda_adv**: Since the discriminator signal is weak, 0.1 might be too small. Try 0.5-1.0 once discriminator is fixed

8. **Lower disc_lr with warm-up**: Use cosine schedule or linear warm-up for disc LR to prevent early instability

9. **Increase disc_steps_per_gen**: Train discriminator 2-5 times per generator step (WGAN-style) to keep it ahead

### D. Alternative approaches

10. **Feature matching loss**: Instead of adversarial loss, use L2 distance between intermediate discriminator features of real and fake — this is more stable than adversarial training and still provides distributional alignment

11. **Contrastive discriminator**: Use InfoNCE/contrastive loss instead of binary classification — better suited for detecting subtle distributional differences
