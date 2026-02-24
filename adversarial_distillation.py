"""
Adversarial Distillation for Discrete Diffusion Models

Projected discriminator design for adversarial distillation in the D-PeRFlow
framework. Key design principles drawn from Projected GAN (Sauer et al. 2021)
and Adversarial Diffusion Distillation (ADD, Sauer et al. 2023):

1. Projected features: Use pre-trained teacher token embeddings to project
   probability distributions from V=50258 to D_model=768. This "soft embedding
   lookup" (probs @ embed_weight) naturally lives in a semantically organized
   space and avoids learning a 50258->256 projection from scratch.

2. Conditional discrimination: The discriminator receives the noisy input x_t
   as context, so it only needs to judge "is this output correct for this
   input?" rather than learning the unconditional teacher/student difference.

3. Per-token + sequence-level dual heads: Per-token classification provides
   L=1024x denser training signal than a single sequence logit. The sequence
   head captures global coherence.

References:
- Projected GAN Converges Faster, Sauer et al. NeurIPS 2021
- Adversarial Diffusion Distillation, Sauer et al. ECCV 2024
- Consistency Training with GAN loss, Kim et al. 2023
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
        residual = x
        x = self.norm1(x)
        B, L, D = x.shape

        qkv = self.attn_qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.attn_out(out)
        x = residual + out

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout2(x)
        x = residual + x

        return x


class ProjectedDiscriminator(nn.Module):
    """
    Projected discriminator for adversarial distillation of discrete diffusion.

    Instead of projecting raw V-dim probability vectors through a learned
    Linear(V, D), this discriminator uses the teacher model's pre-trained
    token embeddings as a frozen projection:

        feat = probs @ teacher_embed   # [B, L, V] @ [V, D_model] -> [B, L, D_model]

    This "soft embedding lookup" has three key advantages:
    1. The projection is semantically organized (similar tokens are close)
    2. No need to learn a V-dimensional projection from scratch
    3. Clean gradient flow: d(feat)/d(probs) = embed^T (no log, no clamp)

    The discriminator is also conditioned on x_t (noisy input tokens), which
    provides the context "what was the model asked to denoise?". This converts
    the task from unconditional distribution classification (very hard) to
    conditional output verification (much easier).

    Dual output heads provide dense training signal:
    - Per-token head: classifies each position independently (L signals/sample)
    - Sequence head: captures global coherence (1 signal/sample)
    """

    def __init__(
        self,
        teacher_embed_weight,
        hidden_size=256,
        n_heads=4,
        n_blocks=4,
        mlp_ratio=4,
        dropout=0.1,
        max_seq_len=1024,
    ):
        """
        Args:
            teacher_embed_weight: Pre-trained teacher embedding [V, D_model]
            hidden_size: Discriminator hidden dimension
            n_heads: Number of attention heads
            n_blocks: Number of transformer blocks
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        vocab_size, d_model = teacher_embed_weight.shape

        # Frozen teacher embedding as projection matrix
        self.register_buffer('embed_weight', teacher_embed_weight.detach().clone())

        # Project from teacher embedding dim to discriminator hidden dim
        self.feat_proj = nn.Linear(d_model, hidden_size)

        # Condition projection: noisy input tokens -> hidden
        # Uses the same frozen embedding, then a learned projection
        self.cond_proj = nn.Linear(d_model, hidden_size)

        # Fuse distribution features with noisy-input condition
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
        )

        # Timestep conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(hidden_size, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        # Per-token head: dense supervision signal
        self.token_head = nn.Linear(hidden_size, 1)

        # Sequence-level head: global coherence
        self.seq_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def timestep_embedding(t, dim=256, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, probs, x_t, t):
        """
        Args:
            probs: Token probability distributions [B, L, V]
            x_t: Noisy input token IDs [B, L] (conditioning context)
            t: Timestep [B]

        Returns:
            token_logits: Per-token real/fake logits [B, L, 1]
            seq_logit: Sequence-level real/fake logit [B, 1]
        """
        B, L, V = probs.shape

        # 1. Soft embedding: project V-dim probs to D_model via teacher embedding
        #    This is differentiable: d(feat)/d(probs) = embed_weight^T
        feat = torch.matmul(probs, self.embed_weight)  # [B, L, D_model]
        feat = self.feat_proj(feat)  # [B, L, hidden]

        # 2. Condition: embed the noisy input tokens
        cond = self.embed_weight[x_t]  # [B, L, D_model]  (index into frozen embed)
        cond = self.cond_proj(cond)  # [B, L, hidden]

        # 3. Fuse features and condition
        h = self.fusion(torch.cat([feat, cond], dim=-1))  # [B, L, hidden]

        # 4. Add position and time embeddings
        h = h + self.pos_embed[:, :L, :]
        t_emb = self.time_embed(self.timestep_embedding(t))  # [B, hidden]
        h = h + t_emb.unsqueeze(1)

        # 5. Transformer blocks
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)

        # 6. Dual heads
        token_logits = self.token_head(h)  # [B, L, 1]
        seq_logit = self.seq_head(h.mean(dim=1))  # [B, 1]

        return token_logits, seq_logit


class AdversarialDistillationLoss(nn.Module):
    """
    Adversarial distillation loss with projected discriminator.

    Generator (student) loss:
        L_student = L_KL + lambda_adv * L_adv

    Discriminator loss (per-token + sequence-level):
        L_disc = L_token(real, fake) + L_seq(real, fake) + R1_penalty

    The per-token loss provides L=1024 independent training signals per sample,
    making discriminator training dramatically more data-efficient than the
    single sequence-level logit of conventional approaches.
    """

    def __init__(
        self,
        discriminator,
        lambda_adv=0.1,
        r1_gamma=10.0,
        token_loss_weight=1.0,
        seq_loss_weight=1.0,
        r1_interval=16,
    ):
        """
        Args:
            discriminator: ProjectedDiscriminator instance
            lambda_adv: Weight for adversarial loss in generator update
            r1_gamma: R1 gradient penalty coefficient (0 to disable)
            token_loss_weight: Weight for per-token discrimination loss
            seq_loss_weight: Weight for sequence-level discrimination loss
            r1_interval: Lazy R1 interval (StyleGAN2). Compute R1 only every
                N discriminator steps and scale gamma by N. Set to 1 to
                compute every step (original behavior). Default 16.
        """
        super().__init__()
        self.discriminator = discriminator
        self.lambda_adv = lambda_adv
        self.r1_gamma = r1_gamma
        self.token_loss_weight = token_loss_weight
        self.seq_loss_weight = seq_loss_weight
        self.r1_interval = r1_interval
        self._disc_step = 0

    def discriminator_loss(self, teacher_probs, student_probs, x_t, t):
        """
        Compute discriminator loss with per-token + sequence-level heads.

        Uses lazy R1 regularization (StyleGAN2): compute R1 penalty only every
        r1_interval discriminator steps, scaling gamma by the interval so the
        effective regularization strength is unchanged. This avoids the cost
        of create_graph=True second-order gradients on most steps.

        Args:
            teacher_probs: Teacher distribution [B, L, V] (detached)
            student_probs: Student distribution [B, L, V] (detached)
            x_t: Noisy input tokens [B, L]
            t: Timestep [B]

        Returns:
            loss: Discriminator loss scalar
            metrics: Dict with loss components
        """
        self._disc_step += 1
        do_r1 = (self.r1_gamma > 0 and
                 self._disc_step % self.r1_interval == 0)

        teacher_probs = teacher_probs.detach()
        student_probs = student_probs.detach()

        if do_r1:
            teacher_probs.requires_grad_(True)

        # Forward pass
        real_tok, real_seq = self.discriminator(teacher_probs, x_t, t)
        fake_tok, fake_seq = self.discriminator(student_probs, x_t, t)

        # Per-token loss: BCE at each position
        real_tok_loss = F.binary_cross_entropy_with_logits(
            real_tok, torch.ones_like(real_tok),
        )
        fake_tok_loss = F.binary_cross_entropy_with_logits(
            fake_tok, torch.zeros_like(fake_tok),
        )
        token_loss = real_tok_loss + fake_tok_loss

        # Sequence-level loss
        real_seq_loss = F.binary_cross_entropy_with_logits(
            real_seq, torch.ones_like(real_seq),
        )
        fake_seq_loss = F.binary_cross_entropy_with_logits(
            fake_seq, torch.zeros_like(fake_seq),
        )
        seq_loss = real_seq_loss + fake_seq_loss

        loss = self.token_loss_weight * token_loss + self.seq_loss_weight * seq_loss

        # Lazy R1 gradient penalty on real samples (StyleGAN2)
        # Scale gamma by r1_interval so the time-averaged penalty is unchanged.
        r1_penalty = torch.tensor(0.0, device=loss.device)
        if do_r1:
            r1_target = real_tok.sum() + real_seq.sum()
            grad_real = torch.autograd.grad(
                outputs=r1_target,
                inputs=teacher_probs,
                create_graph=True,
            )[0]
            r1_penalty = grad_real.pow(2).sum(dim=[1, 2]).mean()
            lazy_gamma = self.r1_gamma * self.r1_interval
            loss = loss + lazy_gamma * 0.5 * r1_penalty

        metrics = {
            "disc_loss": loss.item(),
            "disc_token_loss": token_loss.item(),
            "disc_seq_loss": seq_loss.item(),
            "disc_r1_penalty": r1_penalty.item(),
            "disc_real_logit_mean": real_seq.mean().item(),
            "disc_fake_logit_mean": fake_seq.mean().item(),
            "disc_real_tok_acc": (real_tok > 0).float().mean().item(),
            "disc_fake_tok_acc": (fake_tok < 0).float().mean().item(),
        }

        return loss, metrics

    def generator_loss(self, student_probs, x_t, t):
        """
        Compute adversarial loss for the student (generator).

        Non-saturating loss at both per-token and sequence levels.

        Args:
            student_probs: Student distribution [B, L, V] (with grad)
            x_t: Noisy input tokens [B, L]
            t: Timestep [B]

        Returns:
            loss: Generator adversarial loss [B]
        """
        fake_tok, fake_seq = self.discriminator(student_probs, x_t, t)

        # Per-token: non-saturating loss averaged over positions
        tok_loss = F.binary_cross_entropy_with_logits(
            fake_tok, torch.ones_like(fake_tok), reduction='none',
        ).squeeze(-1).mean(dim=-1)  # [B]

        # Sequence-level: non-saturating loss
        seq_loss = F.binary_cross_entropy_with_logits(
            fake_seq, torch.ones_like(fake_seq), reduction='none',
        ).squeeze(-1)  # [B]

        loss = self.token_loss_weight * tok_loss + self.seq_loss_weight * seq_loss
        return loss

    def combined_loss(self, kl_loss, student_probs, x_t, t):
        """
        Compute combined KL + adversarial loss for the student.

        Args:
            kl_loss: KL divergence loss [B]
            student_probs: Student distribution [B, L, V] (with grad)
            x_t: Noisy input tokens [B, L]
            t: Timestep [B]

        Returns:
            total_loss: Combined loss [B]
            metrics: Dict with loss components
        """
        adv_loss = self.generator_loss(student_probs, x_t, t)
        total_loss = kl_loss + self.lambda_adv * adv_loss

        metrics = {
            "gen_kl_loss": kl_loss.mean().item(),
            "gen_adv_loss": adv_loss.mean().item(),
            "gen_total_loss": total_loss.mean().item(),
            "lambda_adv": self.lambda_adv,
        }

        return total_loss, metrics


def create_discriminator(
    teacher_embed_weight,
    hidden_size=256,
    n_heads=4,
    n_blocks=4,
    dropout=0.1,
    max_seq_len=1024,
):
    """
    Factory function to create a projected discriminator.

    Args:
        teacher_embed_weight: Pre-trained teacher embedding weight [V, D_model]
        hidden_size: Discriminator hidden dimension
        n_heads: Number of attention heads
        n_blocks: Number of transformer blocks
        dropout: Dropout rate
        max_seq_len: Maximum sequence length

    Returns:
        ProjectedDiscriminator instance
    """
    return ProjectedDiscriminator(
        teacher_embed_weight=teacher_embed_weight,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_blocks=n_blocks,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )
