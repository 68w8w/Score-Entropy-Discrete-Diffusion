"""
Adversarial Distillation for Discrete Diffusion Models

Two discriminator designs for adversarial distillation in the D-PeRFlow framework:

A. ProjectedDiscriminator (teacher-embedding projection):
   Uses pre-trained teacher token embeddings to project probability distributions
   from V=50258 to D_model=768. Operates on continuous probability vectors.

B. GPT2ProjectedDiscriminator (GPT-2 feature-space projection):
   Uses Straight-Through Gumbel-Softmax to sample quasi-discrete tokens from
   probability distributions, then extracts features from a frozen GPT-2 model.
   A lightweight discriminator head classifies GPT-2 hidden states as real/fake.
   This mirrors image ADD's use of DINO features for projected discrimination.

Both support:
- Conditional discrimination on noisy input x_t
- Per-token + sequence-level dual heads for dense training signal
- Lazy R1 gradient penalty (StyleGAN2)

References:
- Projected GAN Converges Faster, Sauer et al. NeurIPS 2021
- Adversarial Diffusion Distillation, Sauer et al. ECCV 2024
- Consistency Training with GAN loss, Kim et al. 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Model, GPT2Config


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

        # Store as a plain tensor, NOT a registered buffer. This is critical
        # because DDP modifies registered buffers in-place (broadcasting,
        # version tracking) which breaks autograd.grad() in the R1 penalty.
        # By keeping it outside Module's buffer system, DDP cannot touch it.
        # The tensor is frozen and identical across all ranks (cloned from the
        # same teacher checkpoint), so no cross-rank sync is needed.
        self._frozen_embed = teacher_embed_weight.detach().clone()

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

    def _apply(self, fn, recurse=True):
        """Override to handle device/dtype transfer for _frozen_embed."""
        result = super()._apply(fn, recurse)
        self._frozen_embed = fn(self._frozen_embed)
        return result

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

        # Use the frozen embed directly — it's a plain tensor outside Module's
        # buffer system, so DDP cannot modify it in-place.
        embed_w = self._frozen_embed

        # 1. Soft embedding: project V-dim probs to D_model via teacher embedding
        #    This is differentiable: d(feat)/d(probs) = embed_weight^T
        feat = torch.matmul(probs, embed_w)  # [B, L, D_model]
        feat = self.feat_proj(feat)  # [B, L, hidden]

        # 2. Condition: embed the noisy input tokens
        cond = embed_w[x_t]  # [B, L, D_model]  (index into frozen embed)
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


def gumbel_softmax_sample(logits, tau=0.5):
    """
    Straight-Through Gumbel-Softmax: discrete forward, continuous backward.

    Forward pass returns one-hot vectors (hard samples).
    Backward pass uses the continuous softmax gradient (straight-through estimator).

    Args:
        logits: Unnormalized log-probabilities [B, L, V]
        tau: Temperature (lower = harder, higher = softer). Default 0.5.

    Returns:
        y: One-hot samples with straight-through gradient [B, L, V]
    """
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = F.softmax((logits + gumbels) / tau, dim=-1)
    # Straight-through: hard forward, soft backward
    index = y_soft.argmax(dim=-1)
    y_hard = F.one_hot(index, logits.size(-1)).float()
    return (y_hard - y_soft).detach() + y_soft


class GPT2ProjectedDiscriminator(nn.Module):
    """
    GPT-2 feature-space projected discriminator for discrete diffusion distillation.

    Instead of discriminating on continuous probability distributions (which are
    too easy to distinguish by their smoothness alone), this discriminator:

    1. Samples discrete tokens from probability distributions via Gumbel-Softmax
    2. Feeds the (quasi-discrete) token embeddings through a frozen GPT-2 model
    3. Classifies the GPT-2 hidden states as real (teacher) vs fake (student)

    This mirrors image ADD's use of frozen DINO features: the pre-trained GPT-2
    provides a rich, semantically meaningful feature space where real text and
    generated text differ in ways that correlate with actual quality.

    Key design decisions:
    - GPT-2 is completely frozen (no gradients, no fine-tuning)
    - Gumbel-Softmax provides gradient flow: student → sample → GPT-2 → disc → loss
    - Multi-layer feature extraction (like Projected GAN's multi-scale features)
    - Lightweight discriminator heads on top of GPT-2 features
    """

    def __init__(
        self,
        gpt2_model_name="gpt2",
        hidden_size=256,
        n_heads=4,
        n_blocks=2,
        dropout=0.1,
        max_seq_len=1024,
        gumbel_tau=0.5,
        feature_layers=(-1, -3),
    ):
        """
        Args:
            gpt2_model_name: HuggingFace GPT-2 model name (e.g., "gpt2", "gpt2-medium")
            hidden_size: Discriminator head hidden dimension
            n_heads: Number of attention heads in discriminator blocks
            n_blocks: Number of transformer blocks in discriminator head
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            gumbel_tau: Gumbel-Softmax temperature
            feature_layers: Which GPT-2 layers to extract features from (negative indexing)
        """
        super().__init__()

        self.gumbel_tau = gumbel_tau
        self.feature_layers = feature_layers

        # Load frozen GPT-2
        self._gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
        self._gpt2.eval()
        for param in self._gpt2.parameters():
            param.requires_grad = False

        gpt2_hidden = self._gpt2.config.n_embd  # 768 for gpt2
        n_gpt2_layers = self._gpt2.config.n_layer

        # Resolve negative layer indices
        self._layer_indices = [
            idx if idx >= 0 else n_gpt2_layers + idx
            for idx in feature_layers
        ]

        # Feature projection: project concatenated multi-layer GPT-2 features
        total_feat_dim = gpt2_hidden * len(feature_layers)
        self.feat_proj = nn.Linear(total_feat_dim, hidden_size)

        # Condition projection: noisy input tokens via GPT-2 embedding
        self.cond_proj = nn.Linear(gpt2_hidden, hidden_size)

        # Fuse features with condition
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

        # Discriminator transformer blocks (lightweight)
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(hidden_size, n_heads, 4, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_size)

        # Per-token head
        self.token_head = nn.Linear(hidden_size, 1)

        # Sequence-level head
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

    def _extract_gpt2_features(self, token_probs):
        """
        Extract GPT-2 hidden states from Gumbel-Softmax sampled tokens.

        Uses straight-through Gumbel-Softmax: forward pass samples discrete
        tokens (one-hot), backward pass uses continuous softmax gradient.
        The one-hot vectors are multiplied by GPT-2's embedding matrix to get
        "quasi-discrete" embeddings that GPT-2 can process.

        Args:
            token_probs: Probability distributions [B, L, V]

        Returns:
            features: Concatenated multi-layer GPT-2 features [B, L, D_feat]
        """
        # Gumbel-Softmax: sample quasi-discrete tokens
        log_probs = (token_probs + 1e-10).log()
        gumbel_onehot = gumbel_softmax_sample(log_probs, tau=self.gumbel_tau)

        # Soft embedding lookup: one_hot @ embedding_weight
        # This is differentiable through the Gumbel-Softmax straight-through
        wte = self._gpt2.wte.weight  # [V, D_gpt2]
        inputs_embeds = torch.matmul(gumbel_onehot, wte)  # [B, L, D_gpt2]

        # Run GPT-2 preserving gradient through inputs_embeds only.
        # GPT-2's own parameters are frozen (requires_grad=False), so no
        # gradients accumulate for them, but the chain rule still flows
        # through inputs_embeds → gumbel_onehot → student logits.
        outputs = self._gpt2(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )

        # Extract features from specified layers
        hidden_states = outputs.hidden_states  # tuple of [B, L, D_gpt2]
        features = []
        for idx in self._layer_indices:
            # +1 because hidden_states[0] is the embedding output
            features.append(hidden_states[idx + 1])

        return torch.cat(features, dim=-1)  # [B, L, D_feat]

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

        # 1. Extract GPT-2 features via Gumbel-Softmax sampling
        gpt2_feat = self._extract_gpt2_features(probs)  # [B, L, D_feat]
        feat = self.feat_proj(gpt2_feat)  # [B, L, hidden]

        # 2. Condition: embed the noisy input tokens via GPT-2 embedding
        with torch.no_grad():
            cond = self._gpt2.wte(x_t)  # [B, L, D_gpt2]
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
    Adversarial distillation loss supporting both discriminator types.

    Generator (student) loss:
        L_student = L_fwd_KL + lambda_rev_kl * L_rev_KL + lambda_adv * L_adv

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
        gpt2_discriminator=None,
        lambda_gpt2_adv=0.1,
    ):
        """
        Args:
            discriminator: ProjectedDiscriminator instance (teacher-embedding based)
            lambda_adv: Weight for projected adversarial loss in generator update
            r1_gamma: R1 gradient penalty coefficient (0 to disable)
            token_loss_weight: Weight for per-token discrimination loss
            seq_loss_weight: Weight for sequence-level discrimination loss
            r1_interval: Lazy R1 interval (StyleGAN2). Compute R1 only every
                N discriminator steps and scale gamma by N. Set to 1 to
                compute every step (original behavior). Default 16.
            gpt2_discriminator: Optional GPT2ProjectedDiscriminator instance
            lambda_gpt2_adv: Weight for GPT-2 adversarial loss in generator update
        """
        super().__init__()
        self.discriminator = discriminator
        self.lambda_adv = lambda_adv
        self.r1_gamma = r1_gamma
        self.token_loss_weight = token_loss_weight
        self.seq_loss_weight = seq_loss_weight
        self.r1_interval = r1_interval
        self._disc_step = 0

        # GPT-2 projected discriminator (Direction 1)
        self.gpt2_discriminator = gpt2_discriminator
        self.lambda_gpt2_adv = lambda_gpt2_adv
        self._gpt2_disc_step = 0

    def discriminator_loss(self, teacher_probs, student_probs, x_t, t):
        """
        Compute discriminator loss with per-token + sequence-level heads.

        Uses lazy R1 regularization (StyleGAN2): compute R1 penalty only every
        r1_interval discriminator steps, scaling gamma by the interval so the
        effective regularization strength is unchanged.

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

    def gpt2_discriminator_loss(self, teacher_probs, student_probs, x_t, t):
        """
        Compute GPT-2 projected discriminator loss.

        Discriminates in GPT-2 feature space on Gumbel-Softmax sampled tokens.
        No R1 penalty here — Gumbel-Softmax already provides implicit regularization
        through stochastic sampling.

        Args:
            teacher_probs: Teacher distribution [B, L, V] (detached)
            student_probs: Student distribution [B, L, V] (detached)
            x_t: Noisy input tokens [B, L]
            t: Timestep [B]

        Returns:
            loss: GPT-2 discriminator loss scalar
            metrics: Dict with loss components
        """
        if self.gpt2_discriminator is None:
            return torch.tensor(0.0, device=t.device), {}

        self._gpt2_disc_step += 1

        teacher_probs = teacher_probs.detach()
        student_probs = student_probs.detach()

        # Forward pass through GPT-2 discriminator
        real_tok, real_seq = self.gpt2_discriminator(teacher_probs, x_t, t)
        fake_tok, fake_seq = self.gpt2_discriminator(student_probs, x_t, t)

        # Per-token loss
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

        metrics = {
            "gpt2_disc_loss": loss.item(),
            "gpt2_disc_token_loss": token_loss.item(),
            "gpt2_disc_seq_loss": seq_loss.item(),
            "gpt2_disc_real_logit_mean": real_seq.mean().item(),
            "gpt2_disc_fake_logit_mean": fake_seq.mean().item(),
            "gpt2_disc_real_tok_acc": (real_tok > 0).float().mean().item(),
            "gpt2_disc_fake_tok_acc": (fake_tok < 0).float().mean().item(),
        }

        return loss, metrics

    def generator_loss(self, student_probs, x_t, t):
        """
        Compute adversarial loss for the student (generator).

        Non-saturating loss at both per-token and sequence levels,
        from both the projected discriminator and GPT-2 discriminator.

        Args:
            student_probs: Student distribution [B, L, V] (with grad)
            x_t: Noisy input tokens [B, L]
            t: Timestep [B]

        Returns:
            loss: Generator adversarial loss [B]
            gpt2_loss: GPT-2 adversarial loss [B] (0 if no GPT-2 discriminator)
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

        proj_loss = self.token_loss_weight * tok_loss + self.seq_loss_weight * seq_loss

        # GPT-2 discriminator loss
        gpt2_loss = torch.zeros_like(proj_loss)
        if self.gpt2_discriminator is not None:
            g_fake_tok, g_fake_seq = self.gpt2_discriminator(student_probs, x_t, t)

            g_tok_loss = F.binary_cross_entropy_with_logits(
                g_fake_tok, torch.ones_like(g_fake_tok), reduction='none',
            ).squeeze(-1).mean(dim=-1)  # [B]

            g_seq_loss = F.binary_cross_entropy_with_logits(
                g_fake_seq, torch.ones_like(g_fake_seq), reduction='none',
            ).squeeze(-1)  # [B]

            gpt2_loss = self.token_loss_weight * g_tok_loss + self.seq_loss_weight * g_seq_loss

        return proj_loss, gpt2_loss

    def combined_loss(self, kl_loss, student_probs, x_t, t, rev_kl_loss=None, lambda_rev_kl=0.0):
        """
        Compute combined loss for the student:
            L = L_fwd_KL + lambda_rev_kl * L_rev_KL
              + lambda_adv * L_proj_adv + lambda_gpt2_adv * L_gpt2_adv

        Args:
            kl_loss: Forward KL divergence loss [B]
            student_probs: Student distribution [B, L, V] (with grad)
            x_t: Noisy input tokens [B, L]
            t: Timestep [B]
            rev_kl_loss: Reverse KL divergence loss [B] (optional)
            lambda_rev_kl: Weight for reverse KL loss

        Returns:
            total_loss: Combined loss [B]
            metrics: Dict with loss components
        """
        proj_adv_loss, gpt2_adv_loss = self.generator_loss(student_probs, x_t, t)

        total_loss = kl_loss + self.lambda_adv * proj_adv_loss + self.lambda_gpt2_adv * gpt2_adv_loss

        metrics = {
            "gen_fwd_kl_loss": kl_loss.mean().item(),
            "gen_proj_adv_loss": proj_adv_loss.mean().item(),
            "gen_gpt2_adv_loss": gpt2_adv_loss.mean().item(),
            "gen_total_loss": total_loss.mean().item(),
            "lambda_adv": self.lambda_adv,
            "lambda_gpt2_adv": self.lambda_gpt2_adv,
        }

        # Add reverse KL if provided
        if rev_kl_loss is not None and lambda_rev_kl > 0:
            total_loss = total_loss + lambda_rev_kl * rev_kl_loss
            metrics["gen_rev_kl_loss"] = rev_kl_loss.mean().item()
            metrics["lambda_rev_kl"] = lambda_rev_kl
            metrics["gen_total_loss"] = total_loss.mean().item()

        # Keep backward-compatible key
        metrics["gen_kl_loss"] = metrics["gen_fwd_kl_loss"]
        metrics["gen_adv_loss"] = metrics["gen_proj_adv_loss"]

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


def create_gpt2_discriminator(
    gpt2_model_name="gpt2",
    hidden_size=256,
    n_heads=4,
    n_blocks=2,
    dropout=0.1,
    max_seq_len=1024,
    gumbel_tau=0.5,
    feature_layers=(-1, -3),
):
    """
    Factory function to create a GPT-2 projected discriminator.

    Args:
        gpt2_model_name: HuggingFace GPT-2 model name
        hidden_size: Discriminator head hidden dimension
        n_heads: Number of attention heads
        n_blocks: Number of transformer blocks in discriminator head
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        gumbel_tau: Gumbel-Softmax temperature
        feature_layers: Which GPT-2 layers to extract features from

    Returns:
        GPT2ProjectedDiscriminator instance
    """
    return GPT2ProjectedDiscriminator(
        gpt2_model_name=gpt2_model_name,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_blocks=n_blocks,
        dropout=dropout,
        max_seq_len=max_seq_len,
        gumbel_tau=gumbel_tau,
        feature_layers=feature_layers,
    )
