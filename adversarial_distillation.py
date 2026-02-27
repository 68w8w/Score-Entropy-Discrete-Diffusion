"""
Adversarial Distillation for Discrete Diffusion Models — v4 (LSGAN)

Key insight from v3 experiment: Hinge loss achieves margin immediately on
near-identical teacher/student distributions (96-97% argmax agreement),
then gradient = 0 → discriminators freeze. Meanwhile, feature matching
(weight=1.0, loss≈7) dominated total loss at 90%, overwhelming KL (≈0.2).

v4 solution — Least Squares GAN (LSGAN, Mao et al. ICCV 2017):
  D: (D(real) - 1)² + D(fake)²
  G: (D(fake) - 1)²

Why LSGAN is ideal for distillation:
1. Quadratic penalty naturally bounds outputs near {0, 1} — no need for
   the hinge margin trick that caused saturation in v3
2. Always non-zero gradients (no dead zone like hinge at |D|>1)
3. Self-regularizing: overshooting is quadratically penalized, so disc
   never reaches g2_gap=19 (v2 BCE problem) or stops learning (v3 hinge)
4. Simpler than hinge + LeCam while achieving better properties

Retained from v3:
- Spectral Normalization (SNGAN) — gradient stability
- Adaptive Lambda (VQGAN) — gradient magnitude balancing
- Feature Matching — reduced weight (0.1, not 1.0)
- LeCam Regularization — extra disc smoothing

References:
- Least Squares GANs, Mao et al. ICCV 2017
- Spectral Normalization for GANs, Miyato et al. ICLR 2018
- Taming Transformers (VQGAN), Esser et al. CVPR 2021
- Adversarial Diffusion Distillation, Sauer et al. ECCV 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Model


# ---------------------------------------------------------------------------
# Utility: apply spectral normalization to all Linear layers in a module
# ---------------------------------------------------------------------------
def apply_spectral_norm(module):
    """Apply spectral normalization to all Linear layers (recursively).
    Skips layers that are already spectrally normalized."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            try:
                nn.utils.parametrizations.spectral_norm(child)
            except Exception:
                # Already normalized or not applicable
                pass
        else:
            apply_spectral_norm(child)


# ---------------------------------------------------------------------------
# Discriminator Block (Transformer)
# ---------------------------------------------------------------------------
class DiscriminatorBlock(nn.Module):
    """Transformer block for discriminator. All Linear layers will receive
    spectral normalization from the parent module's __init__."""

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


# ---------------------------------------------------------------------------
# Projected Discriminator (teacher-embedding)
# ---------------------------------------------------------------------------
class ProjectedDiscriminator(nn.Module):
    """
    Projected discriminator using teacher embeddings as a frozen projection.

    v3 changes:
    - Spectral normalization on all learned Linear layers
    - Returns intermediate features for feature matching
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
        super().__init__()

        vocab_size, d_model = teacher_embed_weight.shape

        # Frozen teacher embedding (plain tensor, not a buffer — see v1 notes)
        self._frozen_embed = teacher_embed_weight.detach().clone()

        self.feat_proj = nn.Linear(d_model, hidden_size)
        self.cond_proj = nn.Linear(d_model, hidden_size)

        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            DiscriminatorBlock(hidden_size, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_size)

        self.token_head = nn.Linear(hidden_size, 1)
        self.seq_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        # Apply spectral normalization to all Linear layers
        apply_spectral_norm(self)

    def _apply(self, fn, recurse=True):
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

    def forward(self, probs, x_t, t, return_features=False):
        """
        Args:
            probs: [B, L, V]
            x_t: [B, L]
            t: [B]
            return_features: if True, also return last-layer features

        Returns:
            token_logits: [B, L, 1]
            seq_logit: [B, 1]
            (optional) features: [B, L, hidden]
        """
        B, L, V = probs.shape
        embed_w = self._frozen_embed

        feat = torch.matmul(probs, embed_w)  # [B, L, D_model]
        feat = self.feat_proj(feat)

        cond = embed_w[x_t]
        cond = self.cond_proj(cond)

        h = self.fusion(torch.cat([feat, cond], dim=-1))

        h = h + self.pos_embed[:, :L, :]
        t_emb = self.time_embed(self.timestep_embedding(t))
        h = h + t_emb.unsqueeze(1)

        for block in self.blocks:
            h = block(h)
        h = self.norm(h)

        token_logits = self.token_head(h)
        seq_logit = self.seq_head(h.mean(dim=1))

        if return_features:
            return token_logits, seq_logit, h
        return token_logits, seq_logit


# ---------------------------------------------------------------------------
# Gumbel-Softmax
# ---------------------------------------------------------------------------
def gumbel_softmax_sample(logits, tau=0.5):
    """Straight-Through Gumbel-Softmax."""
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = F.softmax((logits + gumbels) / tau, dim=-1)
    index = y_soft.argmax(dim=-1)
    y_hard = F.one_hot(index, logits.size(-1)).float()
    return (y_hard - y_soft).detach() + y_soft


# ---------------------------------------------------------------------------
# GPT-2 Projected Discriminator
# ---------------------------------------------------------------------------
class GPT2ProjectedDiscriminator(nn.Module):
    """
    GPT-2 feature-space projected discriminator.

    v3 changes:
    - Spectral normalization on all discriminator head Linear layers
    - Returns intermediate features for feature matching
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
        super().__init__()

        self.gumbel_tau = gumbel_tau
        self.feature_layers = feature_layers

        # Load frozen GPT-2
        self._gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
        self._gpt2.eval()
        for param in self._gpt2.parameters():
            param.requires_grad = False

        gpt2_hidden = self._gpt2.config.n_embd
        n_gpt2_layers = self._gpt2.config.n_layer

        self._layer_indices = [
            idx if idx >= 0 else n_gpt2_layers + idx
            for idx in feature_layers
        ]

        total_feat_dim = gpt2_hidden * len(feature_layers)
        self.feat_proj = nn.Linear(total_feat_dim, hidden_size)
        self.cond_proj = nn.Linear(gpt2_hidden, hidden_size)

        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            DiscriminatorBlock(hidden_size, n_heads, 4, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_size)

        self.token_head = nn.Linear(hidden_size, 1)
        self.seq_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        # Apply spectral norm to all learned Linear layers (NOT frozen GPT-2)
        # We only normalize the discriminator head, not the frozen backbone
        for module_name in ['feat_proj', 'cond_proj', 'fusion', 'time_embed',
                            'blocks', 'norm', 'token_head', 'seq_head']:
            submod = getattr(self, module_name)
            if isinstance(submod, nn.Linear):
                nn.utils.parametrizations.spectral_norm(submod)
            else:
                apply_spectral_norm(submod)

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
        """Extract GPT-2 hidden states from Gumbel-Softmax sampled tokens."""
        wte = self._gpt2.wte.weight
        v_gpt2 = wte.shape[0]
        probs_real = token_probs[..., :v_gpt2]
        probs_real = probs_real / (probs_real.sum(dim=-1, keepdim=True) + 1e-10)

        log_probs = (probs_real + 1e-10).log()
        gumbel_onehot = gumbel_softmax_sample(log_probs, tau=self.gumbel_tau)
        inputs_embeds = torch.matmul(gumbel_onehot, wte)

        outputs = self._gpt2(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states
        features = []
        for idx in self._layer_indices:
            features.append(hidden_states[idx + 1])

        return torch.cat(features, dim=-1)

    def forward(self, probs, x_t, t, return_features=False):
        """
        Args:
            probs: [B, L, V]
            x_t: [B, L]
            t: [B]
            return_features: if True, also return last-layer features

        Returns:
            token_logits: [B, L, 1]
            seq_logit: [B, 1]
            (optional) features: [B, L, hidden]
        """
        B, L, V = probs.shape

        gpt2_feat = self._extract_gpt2_features(probs)
        feat = self.feat_proj(gpt2_feat)

        v_gpt2 = self._gpt2.wte.weight.shape[0]
        x_t_clamped = x_t.clamp(max=v_gpt2 - 1)
        with torch.no_grad():
            cond = self._gpt2.wte(x_t_clamped)
        cond = self.cond_proj(cond)

        h = self.fusion(torch.cat([feat, cond], dim=-1))

        h = h + self.pos_embed[:, :L, :]
        t_emb = self.time_embed(self.timestep_embedding(t))
        h = h + t_emb.unsqueeze(1)

        for block in self.blocks:
            h = block(h)
        h = self.norm(h)

        token_logits = self.token_head(h)
        seq_logit = self.seq_head(h.mean(dim=1))

        if return_features:
            return token_logits, seq_logit, h
        return token_logits, seq_logit


# ---------------------------------------------------------------------------
# LSGAN loss utilities (Mao et al. ICCV 2017)
# ---------------------------------------------------------------------------
def lsgan_loss_disc(real_logits, fake_logits):
    """LSGAN discriminator loss: (D(real)-1)² + D(fake)².
    Optimal D: D(real)→1, D(fake)→0. Natural output range ~[0,1]."""
    return ((real_logits - 1.0).pow(2).mean() + fake_logits.pow(2).mean())


def lsgan_loss_gen(fake_logits):
    """LSGAN generator loss: (D(fake)-1)².
    Returns per-element loss (same shape as input)."""
    return (fake_logits - 1.0).pow(2)


# ---------------------------------------------------------------------------
# Adversarial Distillation Loss — v4 (LSGAN)
# ---------------------------------------------------------------------------
class AdversarialDistillationLoss(nn.Module):
    """
    Adversarial distillation loss with LSGAN + SOTA stabilization.

    Key differences from v3:
    - LSGAN (not hinge) — always-non-zero gradients, self-regularizing outputs
    - Reduced feature matching weight (auxiliary, not dominant)
    - Retained: spectral norm, LeCam, adaptive lambda
    """

    def __init__(
        self,
        discriminator,
        lambda_adv=0.1,
        token_loss_weight=1.0,
        seq_loss_weight=1.0,
        gpt2_discriminator=None,
        lambda_gpt2_adv=0.1,
        lecam_weight=0.001,
        feature_matching_weight=0.0,
        adaptive_lambda=True,
        max_lambda=10.0,
        # Keep these for backward compatibility but they're no longer used
        r1_gamma=0.0,
        r1_interval=16,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_adv = lambda_adv
        self.token_loss_weight = token_loss_weight
        self.seq_loss_weight = seq_loss_weight

        # GPT-2 discriminator
        self.gpt2_discriminator = gpt2_discriminator
        self.lambda_gpt2_adv = lambda_gpt2_adv

        # LeCam regularization (replaces R1)
        self.lecam_weight = lecam_weight
        self.register_buffer('lecam_ema_real', torch.tensor(0.0))
        self.register_buffer('lecam_ema_fake', torch.tensor(0.0))
        self.register_buffer('g2_lecam_ema_real', torch.tensor(0.0))
        self.register_buffer('g2_lecam_ema_fake', torch.tensor(0.0))
        self.lecam_decay = 0.999

        # Feature matching
        self.feature_matching_weight = feature_matching_weight

        # Adaptive lambda (VQGAN-style)
        self.adaptive_lambda = adaptive_lambda
        self.max_lambda = max_lambda

        self._disc_step = 0
        self._gpt2_disc_step = 0

    def _lecam_reg(self, d_real_mean, d_fake_mean, ema_real, ema_fake):
        """LeCam regularization (Tseng et al. ICLR 2022).
        Prevents the discriminator from becoming overconfident by penalizing
        when real scores drop below EMA of fake scores (and vice versa)."""
        reg = (F.relu(d_real_mean - ema_fake).pow(2)
               + F.relu(ema_real - d_fake_mean).pow(2))
        return reg

    def discriminator_loss(self, teacher_probs, student_probs, x_t, t):
        """Projected discriminator loss with LSGAN + LeCam reg."""
        self._disc_step += 1

        teacher_probs = teacher_probs.detach()
        student_probs = student_probs.detach()

        # Forward pass
        real_tok, real_seq = self.discriminator(teacher_probs, x_t, t)
        fake_tok, fake_seq = self.discriminator(student_probs, x_t, t)

        # LSGAN loss (per-token + sequence)
        token_loss = lsgan_loss_disc(real_tok, fake_tok)
        seq_loss = lsgan_loss_disc(real_seq, fake_seq)
        loss = self.token_loss_weight * token_loss + self.seq_loss_weight * seq_loss

        # LeCam regularization
        d_real_mean = real_seq.mean().detach()
        d_fake_mean = fake_seq.mean().detach()

        # Update EMA
        self.lecam_ema_real.mul_(self.lecam_decay).add_(
            d_real_mean * (1 - self.lecam_decay))
        self.lecam_ema_fake.mul_(self.lecam_decay).add_(
            d_fake_mean * (1 - self.lecam_decay))

        lecam_reg = self._lecam_reg(
            real_seq.mean(), fake_seq.mean(),
            self.lecam_ema_real, self.lecam_ema_fake
        )
        loss = loss + self.lecam_weight * lecam_reg

        metrics = {
            "disc_loss": loss.item(),
            "disc_token_loss": token_loss.item(),
            "disc_seq_loss": seq_loss.item(),
            "disc_lecam_reg": lecam_reg.item(),
            "disc_r1_penalty": 0.0,  # backward compat
            "disc_real_logit_mean": real_seq.mean().item(),
            "disc_fake_logit_mean": fake_seq.mean().item(),
            "disc_real_tok_acc": (real_tok > 0).float().mean().item(),
            "disc_fake_tok_acc": (fake_tok < 0).float().mean().item(),
        }

        return loss, metrics

    def gpt2_discriminator_loss(self, teacher_probs, student_probs, x_t, t):
        """GPT-2 discriminator loss with LSGAN + LeCam reg."""
        if self.gpt2_discriminator is None:
            return torch.tensor(0.0, device=t.device), {}

        self._gpt2_disc_step += 1

        teacher_probs = teacher_probs.detach()
        student_probs = student_probs.detach()

        real_tok, real_seq = self.gpt2_discriminator(teacher_probs, x_t, t)
        fake_tok, fake_seq = self.gpt2_discriminator(student_probs, x_t, t)

        # LSGAN loss
        token_loss = lsgan_loss_disc(real_tok, fake_tok)
        seq_loss = lsgan_loss_disc(real_seq, fake_seq)
        loss = self.token_loss_weight * token_loss + self.seq_loss_weight * seq_loss

        # LeCam regularization for GPT-2 disc
        d_real_mean = real_seq.mean().detach()
        d_fake_mean = fake_seq.mean().detach()

        self.g2_lecam_ema_real.mul_(self.lecam_decay).add_(
            d_real_mean * (1 - self.lecam_decay))
        self.g2_lecam_ema_fake.mul_(self.lecam_decay).add_(
            d_fake_mean * (1 - self.lecam_decay))

        lecam_reg = self._lecam_reg(
            real_seq.mean(), fake_seq.mean(),
            self.g2_lecam_ema_real, self.g2_lecam_ema_fake
        )
        loss = loss + self.lecam_weight * lecam_reg

        metrics = {
            "gpt2_disc_loss": loss.item(),
            "gpt2_disc_token_loss": token_loss.item(),
            "gpt2_disc_seq_loss": seq_loss.item(),
            "gpt2_disc_lecam_reg": lecam_reg.item(),
            "gpt2_disc_real_logit_mean": real_seq.mean().item(),
            "gpt2_disc_fake_logit_mean": fake_seq.mean().item(),
            "gpt2_disc_real_tok_acc": (real_tok > 0).float().mean().item(),
            "gpt2_disc_fake_tok_acc": (fake_tok < 0).float().mean().item(),
        }

        return loss, metrics

    def generator_loss(self, student_probs, teacher_probs, x_t, t):
        """
        Generator adversarial loss with LSGAN + feature matching.

        Returns:
            proj_loss: [B] projected adversarial loss
            gpt2_loss: [B] GPT-2 adversarial loss
            fm_loss: scalar feature matching loss
        """
        # --- Projected discriminator ---
        out = self.discriminator(student_probs, x_t, t, return_features=True)
        fake_tok, fake_seq, fake_feat = out

        # LSGAN: (D(fake) - 1)²
        tok_loss = lsgan_loss_gen(fake_tok.squeeze(-1)).mean(dim=-1)  # [B]
        seq_loss = lsgan_loss_gen(fake_seq.squeeze(-1))  # [B]
        proj_loss = self.token_loss_weight * tok_loss + self.seq_loss_weight * seq_loss

        # --- Feature matching (projected disc) ---
        fm_loss = torch.tensor(0.0, device=student_probs.device)
        if self.feature_matching_weight > 0:
            with torch.no_grad():
                _, _, real_feat = self.discriminator(
                    teacher_probs.detach(), x_t, t, return_features=True)
            fm_loss = (fake_feat.mean(dim=[0, 1]) - real_feat.mean(dim=[0, 1])).pow(2).mean()

        # --- GPT-2 discriminator ---
        gpt2_loss = torch.zeros_like(proj_loss)
        if self.gpt2_discriminator is not None:
            g_out = self.gpt2_discriminator(student_probs, x_t, t, return_features=True)
            g_fake_tok, g_fake_seq, g_fake_feat = g_out

            g_tok_loss = lsgan_loss_gen(g_fake_tok.squeeze(-1)).mean(dim=-1)
            g_seq_loss = lsgan_loss_gen(g_fake_seq.squeeze(-1))
            gpt2_loss = self.token_loss_weight * g_tok_loss + self.seq_loss_weight * g_seq_loss

            # Feature matching (GPT-2 disc)
            if self.feature_matching_weight > 0:
                with torch.no_grad():
                    _, _, g_real_feat = self.gpt2_discriminator(
                        teacher_probs.detach(), x_t, t, return_features=True)
                fm_loss = fm_loss + (
                    g_fake_feat.mean(dim=[0, 1]) - g_real_feat.mean(dim=[0, 1])
                ).pow(2).mean()

        return proj_loss, gpt2_loss, fm_loss

    def compute_adaptive_lambda(self, distill_loss, adv_loss, last_layer_param):
        """
        VQGAN-style adaptive λ: balance gradient magnitudes.

        λ = ‖∂L_distill/∂θ_last‖ / (‖∂L_adv/∂θ_last‖ + δ)

        This prevents the adversarial loss from ever dominating.
        """
        if not self.adaptive_lambda or last_layer_param is None:
            return self.lambda_adv, self.lambda_gpt2_adv

        try:
            grad_distill = torch.autograd.grad(
                distill_loss.mean(), last_layer_param, retain_graph=True
            )[0]
            grad_adv = torch.autograd.grad(
                adv_loss.mean(), last_layer_param, retain_graph=True
            )[0]

            ratio = torch.norm(grad_distill) / (torch.norm(grad_adv) + 1e-6)
            adaptive_w = torch.clamp(ratio, 0.0, self.max_lambda).detach()
            return adaptive_w.item(), adaptive_w.item()
        except RuntimeError:
            # Fallback if autograd fails (e.g., no graph)
            return self.lambda_adv, self.lambda_gpt2_adv

    def combined_loss(self, kl_loss, student_probs, teacher_probs, x_t, t,
                      rev_kl_loss=None, lambda_rev_kl=0.0,
                      last_layer_param=None):
        """
        Combined generator loss:
            L = L_fwd_KL + λ_rev * L_rev_KL
              + λ_proj * L_proj_adv + λ_gpt2 * L_gpt2_adv
              + λ_fm * L_feature_matching
        """
        proj_adv_loss, gpt2_adv_loss, fm_loss = self.generator_loss(
            student_probs, teacher_probs, x_t, t
        )

        # Compute distillation loss for adaptive lambda
        distill_loss = kl_loss.clone()
        if rev_kl_loss is not None and lambda_rev_kl > 0:
            distill_loss = distill_loss + lambda_rev_kl * rev_kl_loss

        # Adaptive lambda computation
        total_adv = proj_adv_loss + gpt2_adv_loss
        lambda_proj, lambda_gpt2 = self.compute_adaptive_lambda(
            distill_loss, total_adv, last_layer_param
        )

        # Clamp adaptive lambda by the configured base lambda
        # adaptive_lambda acts as a multiplier on the base lambda
        if self.adaptive_lambda and last_layer_param is not None:
            effective_proj = self.lambda_adv * lambda_proj
            effective_gpt2 = self.lambda_gpt2_adv * lambda_gpt2
        else:
            effective_proj = self.lambda_adv
            effective_gpt2 = self.lambda_gpt2_adv

        total_loss = kl_loss + effective_proj * proj_adv_loss + effective_gpt2 * gpt2_adv_loss

        # Feature matching
        if self.feature_matching_weight > 0:
            total_loss = total_loss + self.feature_matching_weight * fm_loss

        metrics = {
            "gen_fwd_kl_loss": kl_loss.mean().item(),
            "gen_proj_adv_loss": proj_adv_loss.mean().item(),
            "gen_gpt2_adv_loss": gpt2_adv_loss.mean().item(),
            "gen_fm_loss": fm_loss.item(),
            "gen_total_loss": total_loss.mean().item(),
            "lambda_adv": self.lambda_adv,
            "lambda_gpt2_adv": self.lambda_gpt2_adv,
            "effective_lambda_proj": effective_proj if isinstance(effective_proj, float) else effective_proj,
            "effective_lambda_gpt2": effective_gpt2 if isinstance(effective_gpt2, float) else effective_gpt2,
        }

        # Add reverse KL if provided
        if rev_kl_loss is not None and lambda_rev_kl > 0:
            total_loss = total_loss + lambda_rev_kl * rev_kl_loss
            metrics["gen_rev_kl_loss"] = rev_kl_loss.mean().item()
            metrics["lambda_rev_kl"] = lambda_rev_kl
            metrics["gen_total_loss"] = total_loss.mean().item()

        # Backward-compatible keys
        metrics["gen_kl_loss"] = metrics["gen_fwd_kl_loss"]
        metrics["gen_adv_loss"] = metrics["gen_proj_adv_loss"]

        return total_loss, metrics


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
def create_discriminator(
    teacher_embed_weight,
    hidden_size=256,
    n_heads=4,
    n_blocks=4,
    dropout=0.1,
    max_seq_len=1024,
):
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
