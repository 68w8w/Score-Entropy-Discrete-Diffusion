"""
GPT-2 Perceptual Loss for Discrete Diffusion Distillation

Uses frozen GPT-2 intermediate features to provide semantic-level
supervision for student-teacher distillation. Unlike adversarial training,
this approach has no training dynamics issues (no discriminator collapse,
no mode collapse, no gradient vanishing).

Key idea: Teacher and student probability distributions are projected into
GPT-2's learned representation space via soft embeddings (P @ W_embed).
The L2 distance between multi-layer features penalizes semantic differences
that token-level KL divergence misses.

References:
- Perceptual Losses for Real-Time Style Transfer, Johnson et al. ECCV 2016
- The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
  (LPIPS), Zhang et al. CVPR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model


class GPT2PerceptualLoss(nn.Module):
    """
    Perceptual loss using frozen GPT-2 intermediate layer features.

    Extracts hidden states at multiple transformer layers for both teacher
    and student probability distributions, then computes feature distance.

    No learnable parameters — purely a loss function.
    """

    def __init__(
        self,
        model_name="gpt2",
        feature_layers=(2, 5, 8, 11),
        loss_type="l2",
    ):
        """
        Args:
            model_name: HuggingFace GPT-2 model name
            feature_layers: Tuple of GPT-2 layer indices to extract features from.
                           GPT-2 base has 12 layers (0-11). Default uses 4 evenly
                           spaced layers for multi-scale feature matching.
            loss_type: 'l2' (MSE), 'cosine' (1 - cos_sim), or 'l1' (MAE)
        """
        super().__init__()

        self.feature_layers = feature_layers
        self.loss_type = loss_type

        # Load frozen GPT-2
        self._gpt2 = GPT2Model.from_pretrained(model_name)
        self._gpt2.eval()
        for param in self._gpt2.parameters():
            param.requires_grad = False

        self._gpt2_vocab_size = self._gpt2.wte.weight.shape[0]  # 50257

    def _extract_features(self, probs):
        """
        Extract GPT-2 hidden states from probability distributions.

        Uses soft embeddings (P @ W_embed) for full differentiability.
        No Gumbel-Softmax needed — gradients flow cleanly through
        the matrix multiply to the student's output probabilities.

        Args:
            probs: [B, L, V] probability distribution over vocabulary

        Returns:
            features: list of [B, L, D] tensors, one per selected layer
        """
        wte = self._gpt2.wte.weight  # [V_gpt2, D]
        v_gpt2 = self._gpt2_vocab_size

        # Handle SEDD vocab size mismatch (V=50258 with absorb state vs GPT-2 V=50257)
        if probs.shape[-1] > v_gpt2:
            probs_gpt2 = probs[..., :v_gpt2]
            # Re-normalize after truncating absorb token
            probs_gpt2 = probs_gpt2 / (probs_gpt2.sum(dim=-1, keepdim=True) + 1e-10)
        else:
            probs_gpt2 = probs

        # Soft embedding: differentiable weighted sum of token embeddings
        inputs_embeds = torch.matmul(probs_gpt2, wte)  # [B, L, D]

        # Forward through GPT-2 with hidden state output
        outputs = self._gpt2(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )

        # hidden_states: tuple of (n_layers + 1) tensors
        # Index 0 = embedding output, index i+1 = layer i output
        hidden_states = outputs.hidden_states

        features = []
        for layer_idx in self.feature_layers:
            feat = hidden_states[layer_idx + 1]  # +1 to skip embedding layer
            features.append(feat)

        return features

    def forward(self, teacher_probs, student_probs):
        """
        Compute perceptual loss between teacher and student distributions.

        Args:
            teacher_probs: [B, L, V] teacher probability distribution (detached)
            student_probs: [B, L, V] student probability distribution (has grad)

        Returns:
            loss: scalar perceptual loss (averaged over layers)
        """
        # Teacher features: no grad needed (frozen teacher + frozen GPT-2)
        with torch.no_grad():
            teacher_features = self._extract_features(teacher_probs)

        # Student features: grad flows through soft embeddings to student model
        student_features = self._extract_features(student_probs)

        # Per-layer feature distance
        total_loss = 0.0
        for t_feat, s_feat in zip(teacher_features, student_features):
            if self.loss_type == "l2":
                layer_loss = F.mse_loss(s_feat, t_feat)
            elif self.loss_type == "cosine":
                cos_sim = F.cosine_similarity(s_feat, t_feat, dim=-1)
                layer_loss = (1 - cos_sim).mean()
            elif self.loss_type == "l1":
                layer_loss = F.l1_loss(s_feat, t_feat)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            total_loss = total_loss + layer_loss

        # Average across layers
        total_loss = total_loss / len(self.feature_layers)

        return total_loss
