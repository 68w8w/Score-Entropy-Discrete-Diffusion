"""
Adversarial Distillation for Discrete Diffusion Models

This module implements adversarial training to complement the KL divergence loss
in the D-PeRFlow distillation framework. The key idea is to train a discriminator
to distinguish between teacher (multi-step) and student (single-step) denoised
distributions, providing a richer training signal than KL divergence alone.

Architecture:
- Discriminator: Lightweight transformer that takes soft token distributions
  (probability vectors over vocabulary) and a timestep conditioning signal,
  producing a real/fake classification per sequence.
- Training: Alternating updates — discriminator learns to classify, student
  learns to fool discriminator while also minimizing KL divergence.

References:
- Adversarial Diffusion Distillation (ADD), Sauer et al. 2023
- Consistency Training with GAN loss, Kim et al. 2023
- Score Distillation with GAN, Wang et al. 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiscriminatorBlock(nn.Module):
    """A single transformer block for the discriminator."""

    def __init__(self, hidden_size, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_size, hidden_size),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        Returns:
            x: [B, L, D]
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        B, L, D = x.shape

        qkv = self.attn_qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, L, L]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v)  # [B, H, L, D_h]
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.attn_out(out)
        x = residual + out

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout2(x)
        x = residual + x

        return x


class DistributionDiscriminator(nn.Module):
    """
    Discriminator for adversarial distillation of discrete diffusion models.

    Takes soft token probability distributions [B, L, V] and a timestep
    conditioning signal, and outputs a real/fake logit per sequence.

    The discriminator operates on log-probability distributions rather than
    raw probabilities, which amplifies differences in the distribution tails
    and makes it easier to distinguish teacher from student outputs.

    Uses spectral normalization on linear layers for stable training.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=256,
        n_heads=4,
        n_blocks=4,
        mlp_ratio=4,
        dropout=0.1,
        max_seq_len=1024,
        use_spectral_norm=True,
    ):
        """
        Args:
            vocab_size: Size of the token vocabulary (including mask token if absorbing)
            hidden_size: Hidden dimension of the discriminator
            n_heads: Number of attention heads
            n_blocks: Number of transformer blocks
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            use_spectral_norm: Whether to apply spectral normalization
        """
        super().__init__()

        self.use_spectral_norm = use_spectral_norm

        def maybe_sn(layer):
            """Apply spectral normalization if enabled."""
            if use_spectral_norm and isinstance(layer, nn.Linear):
                return nn.utils.spectral_norm(layer)
            return layer

        # Project vocab-sized log-probability vectors to hidden size
        self.input_proj = maybe_sn(nn.Linear(vocab_size, hidden_size))

        # Timestep conditioning
        self.time_embed = nn.Sequential(
            maybe_sn(nn.Linear(256, hidden_size)),
            nn.SiLU(),
            maybe_sn(nn.Linear(hidden_size, hidden_size)),
        )

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(hidden_size, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])

        # Output head: pool over sequence then classify
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            maybe_sn(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
            maybe_sn(nn.Linear(hidden_size, 1)),
        )

    @staticmethod
    def timestep_embedding(t, dim=256, max_period=10000):
        """Sinusoidal timestep embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, probs, t):
        """
        Args:
            probs: Token probability distributions [B, L, V]
            t: Timestep [B] (window boundary time)

        Returns:
            logits: Real/fake logits [B, 1]
        """
        B, L, V = probs.shape

        # Convert to log-probabilities to amplify distributional differences.
        # Raw probs are nearly one-hot (max_prob > 0.93), making teacher/student
        # indistinguishable in probability space. Log-space spreads out the
        # distribution tail where the meaningful differences exist.
        # Clamp to [-20, 0] to avoid -inf while preserving dynamic range.
        log_probs = torch.log(probs + 1e-8).clamp(min=-20.0)

        # Project log-probabilities to hidden space
        x = self.input_proj(log_probs)  # [B, L, D]

        # Add positional embedding
        x = x + self.pos_embed[:, :L, :]

        # Add timestep conditioning (broadcast over sequence)
        t_emb = self.time_embed(self.timestep_embedding(t))  # [B, D]
        x = x + t_emb.unsqueeze(1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Pool and classify
        x = self.norm(x)
        x = x.mean(dim=1)  # [B, D] - mean pooling over sequence
        logits = self.head(x)  # [B, 1]

        return logits


class AdversarialDistillationLoss(nn.Module):
    """
    Combines KL divergence and adversarial losses for distillation.

    The total generator (student) loss is:
        L_student = L_KL + lambda_adv * L_adv

    where:
    - L_KL: Forward KL divergence between teacher and student distributions
    - L_adv: Adversarial loss (non-saturating GAN loss)

    The discriminator loss is the standard binary cross-entropy:
        L_disc = -E[log D(teacher)] - E[log(1 - D(student))]

    We use R1 gradient penalty for discriminator regularization.
    """

    def __init__(
        self,
        discriminator,
        lambda_adv=0.1,
        r1_gamma=10.0,
        label_smoothing=0.0,
    ):
        """
        Args:
            discriminator: DistributionDiscriminator instance
            lambda_adv: Weight for adversarial loss relative to KL loss
            r1_gamma: R1 gradient penalty coefficient (0 to disable)
            label_smoothing: Label smoothing for discriminator targets
                (default 0.0 — label smoothing is counterproductive when
                teacher/student distributions are already very similar)
        """
        super().__init__()
        self.discriminator = discriminator
        self.lambda_adv = lambda_adv
        self.r1_gamma = r1_gamma
        self.label_smoothing = label_smoothing

    def discriminator_loss(self, teacher_probs, student_probs, t):
        """
        Compute discriminator loss.

        Args:
            teacher_probs: Teacher distribution [B, L, V] (detached, no grad)
            student_probs: Student distribution [B, L, V] (detached, no grad)
            t: Timestep [B]

        Returns:
            loss: Discriminator loss scalar
            metrics: Dict with loss components
        """
        # Detach both to ensure no grad flows to generator
        teacher_probs = teacher_probs.detach()
        student_probs = student_probs.detach()

        # R1 gradient penalty on real (teacher) samples
        if self.r1_gamma > 0:
            teacher_probs.requires_grad_(True)

        # Discriminator predictions
        real_logits = self.discriminator(teacher_probs, t)  # [B, 1]
        fake_logits = self.discriminator(student_probs, t)  # [B, 1]

        # Labels with smoothing
        real_label = 1.0 - self.label_smoothing
        fake_label = self.label_smoothing

        # Binary cross-entropy with logits
        real_loss = F.binary_cross_entropy_with_logits(
            real_logits,
            torch.full_like(real_logits, real_label),
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_logits,
            torch.full_like(fake_logits, fake_label),
        )

        loss = real_loss + fake_loss

        # R1 gradient penalty
        r1_penalty = torch.tensor(0.0, device=loss.device)
        if self.r1_gamma > 0 and teacher_probs.requires_grad:
            grad_real = torch.autograd.grad(
                outputs=real_logits.sum(),
                inputs=teacher_probs,
                create_graph=True,
            )[0]
            r1_penalty = grad_real.pow(2).sum(dim=[1, 2]).mean()
            loss = loss + self.r1_gamma * 0.5 * r1_penalty

        metrics = {
            "disc_loss": loss.item(),
            "disc_real_loss": real_loss.item(),
            "disc_fake_loss": fake_loss.item(),
            "disc_r1_penalty": r1_penalty.item(),
            "disc_real_logit_mean": real_logits.mean().item(),
            "disc_fake_logit_mean": fake_logits.mean().item(),
        }

        return loss, metrics

    def generator_loss(self, student_probs, t):
        """
        Compute adversarial loss for the student (generator).

        Uses non-saturating loss: -log(D(student))
        This provides stronger gradients when the discriminator is confident.

        Args:
            student_probs: Student distribution [B, L, V] (with grad)
            t: Timestep [B]

        Returns:
            loss: Generator adversarial loss [B]
        """
        fake_logits = self.discriminator(student_probs, t)  # [B, 1]

        # Non-saturating GAN loss: -log(sigmoid(D(x))) = BCE with label=1
        loss = F.binary_cross_entropy_with_logits(
            fake_logits,
            torch.ones_like(fake_logits),
            reduction='none',
        )

        return loss.squeeze(-1)  # [B]

    def combined_loss(self, kl_loss, student_probs, t):
        """
        Compute combined KL + adversarial loss for the student.

        Args:
            kl_loss: KL divergence loss [B]
            student_probs: Student distribution [B, L, V] (with grad)
            t: Timestep [B]

        Returns:
            total_loss: Combined loss [B]
            metrics: Dict with loss components
        """
        adv_loss = self.generator_loss(student_probs, t)
        total_loss = kl_loss + self.lambda_adv * adv_loss

        metrics = {
            "gen_kl_loss": kl_loss.mean().item(),
            "gen_adv_loss": adv_loss.mean().item(),
            "gen_total_loss": total_loss.mean().item(),
            "lambda_adv": self.lambda_adv,
        }

        return total_loss, metrics


def create_discriminator(
    vocab_size,
    hidden_size=256,
    n_heads=4,
    n_blocks=4,
    dropout=0.1,
    max_seq_len=1024,
    use_spectral_norm=True,
):
    """
    Factory function to create a discriminator.

    Args:
        vocab_size: Token vocabulary size
        hidden_size: Discriminator hidden dimension
        n_heads: Number of attention heads
        n_blocks: Number of transformer blocks
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        use_spectral_norm: Whether to apply spectral normalization

    Returns:
        DistributionDiscriminator instance
    """
    return DistributionDiscriminator(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_blocks=n_blocks,
        dropout=dropout,
        max_seq_len=max_seq_len,
        use_spectral_norm=use_spectral_norm,
    )
