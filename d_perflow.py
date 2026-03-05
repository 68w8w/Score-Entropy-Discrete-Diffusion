"""
D-PeRFlow: Discrete Piecewise Rectified Flow Training Algorithm

This module implements the D-PeRFlow training algorithm for distilling discrete diffusion models.
The algorithm trains a student model to match teacher distributions at time window boundaries,
enabling few-step generation.

Key Components:
- Teacher distribution computation using AnalyticPredictor and EulerPredictor logic
- Distribution interpolation for intermediate timesteps
- KL divergence loss between student predictions and teacher targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from catsample import sample_categorical
from model import utils as mutils


class DPerflowTrainer:
    """
    D-PeRFlow Trainer class that handles:
    1. Teacher distribution computation at window boundaries
    2. Distribution interpolation and resampling
    3. Loss computation for student model optimization
    """

    def __init__(
        self,
        graph,
        noise,
        num_time_windows: int = 4,
        sampling_eps: float = 1e-3,
    ):
        """
        Args:
            graph: The discrete graph structure (Absorbing or Uniform)
            noise: The noise schedule module
            num_time_windows: Number of time windows K (default: 4 for 4-step generation)
            sampling_eps: Small epsilon to avoid t=0 numerical issues (default: 1e-3)
        """
        self.graph = graph
        self.noise = noise
        self.num_time_windows = num_time_windows
        self.sampling_eps = sampling_eps

        # Precompute time window boundaries: [eps, t_1, t_2, ..., t_{K-1}, 1.0]
        # Windows are: (eps, t_1], (t_1, t_2], ..., (t_{K-1}, 1.0]
        self.time_boundaries = torch.linspace(
            sampling_eps, 1.0, num_time_windows + 1
        )

    def get_window_boundaries(self, k: torch.Tensor, device: torch.device):
        """
        Get the boundaries for window k.

        Args:
            k: Window indices [B], values in {1, 2, ..., K}
            device: Target device

        Returns:
            t_k_minus_1: Lower boundary of window [B]
            t_k: Upper boundary of window [B]
        """
        boundaries = self.time_boundaries.to(device)
        t_k_minus_1 = boundaries[k - 1]  # Lower bound
        t_k = boundaries[k]               # Upper bound
        return t_k_minus_1, t_k

    def construct_noisy_state(self, x_0: torch.Tensor, t_k: torch.Tensor):
        """
        Construct the noisy state x_{t_k} at window boundary using SEDD masking.

        For AbsorbingGraph: tokens are masked with probability (1 - exp(-sigma(t_k)))

        Args:
            x_0: Original clean data [B, L]
            t_k: Time at window upper boundary [B] or [B, 1]

        Returns:
            x_t_k: Noisy state at time t_k [B, L]
        """
        # Ensure t_k is 1D [B] for noise, then expand for graph
        t_k_1d = t_k.squeeze(-1) if t_k.dim() > 1 else t_k
        sigma_t_k = self.noise(t_k_1d)[0]  # [B]
        # Expand sigma for graph.sample_transition which expects [B, 1] or broadcastable
        sigma_t_k_expanded = sigma_t_k[:, None]  # [B, 1]
        x_t_k = self.graph.sample_transition(x_0, sigma_t_k_expanded)
        return x_t_k

    def compute_forward_distribution(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Compute the exact forward process distribution q(x_t | x_0).

        This is the TRUE marginal distribution at time t conditioned on x_0,
        computed analytically from the forward process without any neural network.

        For Absorbing graph:
            q(x_t = v | x_0) = exp(-σ(t)) * δ(v, x_0) + (1-exp(-σ(t))) * δ(v, MASK)

        For Uniform graph:
            q(x_t = v | x_0) = exp(-σ(t)) * δ(v, x_0) + (1-exp(-σ(t))) / V

        Args:
            x_0: Original clean data [B, L]
            t: Time [B] or [B, 1]

        Returns:
            probs: Forward process distribution [B, L, V]
        """
        t_1d = t.squeeze(-1) if t.dim() > 1 else t
        sigma = self.noise(t_1d)[0]  # [B]
        sigma_expanded = sigma[:, None]  # [B, 1]

        V = self.graph.dim
        stay_prob = (-sigma_expanded).exp()  # [B, 1] — probability of staying at x_0

        # One-hot of x_0: [B, L, V]
        one_hot_x0 = F.one_hot(x_0, num_classes=V).float()

        if self.graph.absorb:
            # Absorbing: move to MASK state (last dim) with prob 1-exp(-σ)
            mask_one_hot = F.one_hot(
                torch.full_like(x_0, V - 1), num_classes=V
            ).float()
            probs = stay_prob.unsqueeze(-1) * one_hot_x0 + (1 - stay_prob).unsqueeze(-1) * mask_one_hot
        else:
            # Uniform: move to any state uniformly with prob (1-exp(-σ))/V
            probs = stay_prob.unsqueeze(-1) * one_hot_x0 + (1 - stay_prob).unsqueeze(-1) / V

        return probs

    def compute_analytic_distribution(
        self,
        score_fn,
        x: torch.Tensor,
        t: torch.Tensor,
        step_size: torch.Tensor,
    ):
        """
        Compute the probability distribution using AnalyticPredictor logic.

        This computes P(x'|x) for a small step, returning the full distribution
        instead of sampling from it.

        Args:
            score_fn: Score function from teacher model
            x: Current discrete state [B, L]
            t: Current time [B] or [B, 1]
            step_size: Step size for the transition [B], [B, 1] or scalar

        Returns:
            probs: Transition probability distribution [B, L, V]
        """
        # Ensure time tensors are 1D [B]
        t_1d = t.squeeze(-1) if t.dim() > 1 else t
        if isinstance(step_size, torch.Tensor):
            step_size_1d = step_size.squeeze(-1) if step_size.dim() > 1 else step_size
        else:
            step_size_1d = step_size

        # Get noise levels
        curr_sigma = self.noise(t_1d)[0]  # [B]
        next_sigma = self.noise(t_1d - step_size_1d)[0]  # [B]

        dsigma = curr_sigma - next_sigma  # [B]

        # Compute score from teacher model
        score = score_fn(x, curr_sigma)  # [B, L, V]

        # Apply staggered score adjustment
        # dsigma needs to be [B, 1] for graph methods
        dsigma_expanded = dsigma[:, None]  # [B, 1]
        stag_score = self.graph.staggered_score(score, dsigma_expanded)  # [B, L, V]

        # Compute transition probabilities
        # transp_transition gives the transition matrix row for each position
        probs = stag_score * self.graph.transp_transition(x, dsigma_expanded)  # [B, L, V]

        # Normalize to ensure valid probability distribution
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

        return probs

    def compute_euler_distribution(
        self,
        score_fn,
        x: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        num_steps: int = 1,
    ):
        """
        Compute the target distribution using Euler ODE solver logic.

        This traces the probability flow ODE from t_start to t_end and returns
        the final distribution.

        Args:
            score_fn: Score function from teacher model
            x: Starting discrete state [B, L]
            t_start: Start time [B] or [B, 1]
            t_end: End time [B] or [B, 1]
            num_steps: Number of Euler steps (default: 1 for single window step)

        Returns:
            probs: Target probability distribution at t_end [B, L, V]
        """
        # Ensure time tensors are 1D [B]
        t_start_1d = t_start.squeeze(-1) if t_start.dim() > 1 else t_start
        t_end_1d = t_end.squeeze(-1) if t_end.dim() > 1 else t_end

        # Total time to traverse (scalar per batch)
        dt = (t_start_1d - t_end_1d) / num_steps  # [B]

        # Initialize current state and time
        current_x = x
        current_t = t_start_1d.clone()  # [B]
        probs = None

        for step in range(num_steps):
            sigma, dsigma = self.noise(current_t)  # Both [B]
            score = score_fn(current_x, sigma)  # [B, L, V]

            # Compute reverse rate: step_size * dsigma * reverse_rate_matrix
            scale = (dt * dsigma)[:, None, None]  # [B, 1, 1]
            rev_rate = scale * self.graph.reverse_rate(current_x, score)  # [B, L, V]

            # Compute distribution for this step: one-hot of CURRENT state + rate
            one_hot_current = F.one_hot(current_x, num_classes=self.graph.dim).float()
            probs = one_hot_current + rev_rate

            # Clamp and normalize
            probs = probs.clamp(min=0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

            # Update time
            current_t = current_t - dt

            # For multi-step, sample intermediate states (like SEDD)
            # Only the last step returns distribution without sampling
            # This is consistent with how discrete diffusion actually works
            if step < num_steps - 1:
                current_x = sample_categorical(probs, method="hard")

        return probs

    def compute_analytic_multi_step(
        self,
        score_fn,
        x: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        num_steps: int = 4,
    ):
        """
        Compute target distribution using Analytic predictor with multiple steps.

        This is an Exponential Euler (1st-order Exponential Integrator):
        each step uses the EXACT matrix exponential exp(ΔσQ) via SEDD's
        staggered_score * transp_transition, instead of the Taylor approximation
        I + dt*Q used by Euler.

        The only approximation is that the score is held constant within each sub-step.
        More steps = more score updates = higher accuracy.

        Args:
            score_fn: Score function from teacher model
            x: Starting discrete state [B, L]
            t_start: Start time [B] or [B, 1]
            t_end: End time [B] or [B, 1]
            num_steps: Number of analytic sub-steps

        Returns:
            probs: Target probability distribution at t_end [B, L, V]
        """
        t_start_1d = t_start.squeeze(-1) if t_start.dim() > 1 else t_start
        t_end_1d = t_end.squeeze(-1) if t_end.dim() > 1 else t_end

        # Linearly spaced time points from t_start to t_end
        # dt per step (t_start > t_end, so dt > 0)
        dt = (t_start_1d - t_end_1d) / num_steps  # [B]

        current_x = x
        current_t = t_start_1d.clone()
        probs = None

        for step in range(num_steps):
            next_t = current_t - dt
            # Clamp to avoid going below t_end due to floating point
            next_t = torch.max(next_t, t_end_1d)

            # Compute sigma at current and next time
            curr_sigma = self.noise(current_t)[0]  # [B]
            next_sigma = self.noise(next_t)[0]      # [B]
            dsigma = curr_sigma - next_sigma         # [B]

            # Score at current state and time
            score = score_fn(current_x, curr_sigma)  # [B, L, V]

            # Analytic predictor: stag_score * transp_transition = exact exp(ΔσQ)
            dsigma_expanded = dsigma[:, None]  # [B, 1]
            stag_score = self.graph.staggered_score(score, dsigma_expanded)
            probs = stag_score * self.graph.transp_transition(current_x, dsigma_expanded)

            # Normalize
            probs = probs.clamp(min=0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

            # Update time
            current_t = next_t

            # Sample intermediate state for multi-step (except last step)
            if step < num_steps - 1:
                current_x = sample_categorical(probs, method="hard")

        return probs

    def compute_exp_midpoint_distribution(
        self,
        score_fn,
        x: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
    ):
        """
        Compute target distribution using 2nd-order Exponential Midpoint method.

        This is a 2nd-order Exponential Integrator that uses 2 score evaluations
        to achieve O(Δσ³) accuracy, compared to O(Δσ²) for Exp-Euler (Analytic).

        Algorithm:
            1. Score at start: s₁ = score(x_t, σ(t_start))
            2. Half-step with s₁: P_half = stag_score(s₁, Δσ/2) * exp(Δσ/2 · Q^T)
            3. Sample midpoint: x_mid ~ P_half
            4. Score at midpoint: s₂ = score(x_mid, σ(t_mid))
            5. Full step with s₂: P_final = stag_score(s₂, Δσ) * exp(Δσ · Q^T)

        The midpoint score s₂ better represents the average behavior over the
        full interval, cancelling the first-order error term by symmetry.

        Args:
            score_fn: Score function from teacher model
            x: Starting discrete state [B, L]
            t_start: Start time [B] or [B, 1]
            t_end: End time [B] or [B, 1]

        Returns:
            probs: Target probability distribution at t_end [B, L, V]
        """
        t_start_1d = t_start.squeeze(-1) if t_start.dim() > 1 else t_start
        t_end_1d = t_end.squeeze(-1) if t_end.dim() > 1 else t_end
        t_mid = (t_start_1d + t_end_1d) / 2.0

        # === Step 1: Half-step predictor (Exp-Euler from t_start to t_mid) ===
        sigma_start = self.noise(t_start_1d)[0]   # [B]
        sigma_mid = self.noise(t_mid)[0]           # [B]
        sigma_end = self.noise(t_end_1d)[0]        # [B]

        dsigma_half = sigma_start - sigma_mid       # [B]

        # Score at start
        score_start = score_fn(x, sigma_start)      # [B, L, V]

        # Analytic half-step
        dsigma_half_exp = dsigma_half[:, None]
        stag_half = self.graph.staggered_score(score_start, dsigma_half_exp)
        probs_half = stag_half * self.graph.transp_transition(x, dsigma_half_exp)
        probs_half = probs_half.clamp(min=0)
        probs_half = probs_half / (probs_half.sum(dim=-1, keepdim=True) + 1e-10)

        # Sample midpoint state
        x_mid = sample_categorical(probs_half, method="hard")

        # === Step 2: Full step using midpoint score (Exp-Euler from t_start to t_end, with s₂) ===
        # Score at midpoint (the key improvement: this score is more representative)
        score_mid = score_fn(x_mid, sigma_mid)       # [B, L, V]

        # Full analytic step from original x using midpoint score
        dsigma_full = sigma_start - sigma_end         # [B]
        dsigma_full_exp = dsigma_full[:, None]
        stag_full = self.graph.staggered_score(score_mid, dsigma_full_exp)
        probs_final = stag_full * self.graph.transp_transition(x, dsigma_full_exp)
        probs_final = probs_final.clamp(min=0)
        probs_final = probs_final / (probs_final.sum(dim=-1, keepdim=True) + 1e-10)

        return probs_final

    def _sigma_to_t(self, sigma):
        """Invert the LogLinear noise schedule: σ(t) = -log(1-(1-ε)t) => t = (1-exp(-σ))/(1-ε)"""
        eps = getattr(self.noise, 'eps', 1e-3)
        return (1 - torch.exp(-sigma)) / (1 - eps)

    def compute_exp_midpoint_sigma_distribution(
        self,
        score_fn,
        x: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
    ):
        """
        Exponential Midpoint method with σ-space adaptive midpoint.

        Instead of t_mid = (t_start + t_end) / 2, computes the midpoint in σ-space:
            σ_mid = (σ(t_start) + σ(t_end)) / 2
            t_mid = σ⁻¹(σ_mid)

        This is motivated by the fact that the LogLinear noise schedule σ(t) = -log(1-(1-ε)t)
        is nonlinear in t. Near t=1, σ changes extremely fast while near t=0 it's slow.
        Taking the midpoint in σ-space places it where the dynamics are more uniform,
        yielding better 2nd-order accuracy with no extra computational cost.

        Accuracy: O(Δσ³) with better constants than t-space midpoint.
        NFE: 2 (same as exp_midpoint)
        """
        t_start_1d = t_start.squeeze(-1) if t_start.dim() > 1 else t_start
        t_end_1d = t_end.squeeze(-1) if t_end.dim() > 1 else t_end

        # Compute σ values at boundaries
        sigma_start = self.noise(t_start_1d)[0]   # [B]
        sigma_end = self.noise(t_end_1d)[0]        # [B]

        # Midpoint in σ-space (arithmetic mean of σ values)
        sigma_mid = (sigma_start + sigma_end) / 2.0
        t_mid = self._sigma_to_t(sigma_mid)

        # === Step 1: Half-step predictor (t_start -> t_mid) ===
        dsigma_half = sigma_start - sigma_mid       # [B]
        score_start = score_fn(x, sigma_start)      # [B, L, V]

        dsigma_half_exp = dsigma_half[:, None]
        stag_half = self.graph.staggered_score(score_start, dsigma_half_exp)
        probs_half = stag_half * self.graph.transp_transition(x, dsigma_half_exp)
        probs_half = probs_half.clamp(min=0)
        probs_half = probs_half / (probs_half.sum(dim=-1, keepdim=True) + 1e-10)

        # Sample midpoint state
        x_mid = sample_categorical(probs_half, method="hard")

        # === Step 2: Full step using midpoint score ===
        score_mid = score_fn(x_mid, sigma_mid)      # [B, L, V]

        dsigma_full = sigma_start - sigma_end        # [B]
        dsigma_full_exp = dsigma_full[:, None]
        stag_full = self.graph.staggered_score(score_mid, dsigma_full_exp)
        probs_final = stag_full * self.graph.transp_transition(x, dsigma_full_exp)
        probs_final = probs_final.clamp(min=0)
        probs_final = probs_final / (probs_final.sum(dim=-1, keepdim=True) + 1e-10)

        return probs_final

    def compute_exp_corrector_distribution(
        self,
        score_fn,
        x: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
    ):
        """
        Exponential Predictor-Corrector (Heun's method) for discrete diffusion.

        Uses the trapezoidal rule to approximate the score integral:
            1. Predict: Take a full analytic step using score at t_start
            2. Correct: Evaluate score at predicted endpoint
            3. Average the two scores and redo the step

        This avoids the sampling noise at the midpoint (unlike exp_midpoint),
        and achieves 2nd-order accuracy via the trapezoidal rule:
            s_avg = (s(x, σ_start) + s(x_pred, σ_end)) / 2

        Combined with σ-space optimization for the prediction step.

        Accuracy: O(Δσ³) with reduced variance (no midpoint sampling)
        NFE: 2 (same as exp_midpoint)
        """
        t_start_1d = t_start.squeeze(-1) if t_start.dim() > 1 else t_start
        t_end_1d = t_end.squeeze(-1) if t_end.dim() > 1 else t_end

        sigma_start = self.noise(t_start_1d)[0]   # [B]
        sigma_end = self.noise(t_end_1d)[0]        # [B]
        dsigma_full = sigma_start - sigma_end       # [B]
        dsigma_full_exp = dsigma_full[:, None]

        # === Step 1: Predict - full analytic step with score at t_start ===
        score_start = score_fn(x, sigma_start)      # [B, L, V]

        stag_pred = self.graph.staggered_score(score_start, dsigma_full_exp)
        probs_pred = stag_pred * self.graph.transp_transition(x, dsigma_full_exp)
        probs_pred = probs_pred.clamp(min=0)
        probs_pred = probs_pred / (probs_pred.sum(dim=-1, keepdim=True) + 1e-10)

        # Sample predicted endpoint
        x_pred = sample_categorical(probs_pred, method="hard")

        # === Step 2: Correct - evaluate score at predicted endpoint ===
        score_end = score_fn(x_pred, sigma_end)      # [B, L, V]

        # === Step 3: Trapezoidal average of scores ===
        score_avg = (score_start + score_end) / 2.0

        # === Step 4: Full step with averaged score ===
        stag_final = self.graph.staggered_score(score_avg, dsigma_full_exp)
        probs_final = stag_final * self.graph.transp_transition(x, dsigma_full_exp)
        probs_final = probs_final.clamp(min=0)
        probs_final = probs_final / (probs_final.sum(dim=-1, keepdim=True) + 1e-10)

        return probs_final

    def compute_teacher_distributions(
        self,
        teacher_score_fn,
        x_t_k: torch.Tensor,
        t_k: torch.Tensor,
        t_k_minus_1: torch.Tensor,
        euler_steps: int = 1,
        teacher_method: str = "euler",
        x_0: torch.Tensor = None,
    ):
        """
        Compute teacher distributions at window boundaries.

        Args:
            teacher_score_fn: Score function from teacher model
            x_t_k: Noisy state at window start [B, L]
            t_k: Window upper boundary time [B, 1]
            t_k_minus_1: Window lower boundary time [B, 1]
            euler_steps: Number of steps for Euler/Analytic methods
            teacher_method: Which method to compute P_{t_{k-1}}:
                - "euler": Euler ODE solver (I + dt*Q, Taylor approximation)
                - "analytic": Analytic multi-step (exact exp(ΔσQ), 1st-order Exp Integrator)
                - "exp_midpoint": Exponential Midpoint (2nd-order, t-space midpoint)
                - "exp_midpoint_sigma": Exponential Midpoint with σ-space adaptive midpoint
                - "exp_corrector": Predictor-Corrector / Heun's method (2nd-order, trapezoidal)
            x_0: Original clean data [B, L]. If provided, computes P_t_k using
                the exact forward distribution q(x_{t_k} | x_0) instead of the
                delta_t approximation.

        Returns:
            P_t_k: Distribution at window start [B, L, V]
            P_t_k_minus_1: Target distribution at window end [B, L, V]
        """
        # A. Compute start distribution P_{t_k}
        # Use exact forward process distribution q(x_{t_k} | x_0) — no neural network needed
        P_t_k = self.compute_forward_distribution(x_0, t_k)

        # B. Compute target distribution P_{t_{k-1}}
        if teacher_method == "euler":
            P_t_k_minus_1 = self.compute_euler_distribution(
                teacher_score_fn, x_t_k, t_k, t_k_minus_1, num_steps=euler_steps
            )
        elif teacher_method == "analytic":
            P_t_k_minus_1 = self.compute_analytic_multi_step(
                teacher_score_fn, x_t_k, t_k, t_k_minus_1, num_steps=euler_steps
            )
        elif teacher_method == "exp_midpoint":
            P_t_k_minus_1 = self.compute_exp_midpoint_distribution(
                teacher_score_fn, x_t_k, t_k, t_k_minus_1
            )
        elif teacher_method == "exp_midpoint_sigma":
            P_t_k_minus_1 = self.compute_exp_midpoint_sigma_distribution(
                teacher_score_fn, x_t_k, t_k, t_k_minus_1
            )
        elif teacher_method == "exp_corrector":
            P_t_k_minus_1 = self.compute_exp_corrector_distribution(
                teacher_score_fn, x_t_k, t_k, t_k_minus_1
            )
        else:
            raise ValueError(f"Unknown teacher_method: {teacher_method}")

        return P_t_k, P_t_k_minus_1

    def interpolate_distribution(
        self,
        P_t_k: torch.Tensor,
        P_t_k_minus_1: torch.Tensor,
        t: torch.Tensor,
        t_k: torch.Tensor,
        t_k_minus_1: torch.Tensor,
    ):
        """
        Linearly interpolate between window boundary distributions.

        P_t = alpha * P_{t_k} + (1 - alpha) * P_{t_{k-1}}
        where alpha = (t - t_{k-1}) / (t_k - t_{k-1})

        Args:
            P_t_k: Distribution at window start [B, L, V]
            P_t_k_minus_1: Distribution at window end [B, L, V]
            t: Target time [B, 1]
            t_k: Window upper boundary [B, 1]
            t_k_minus_1: Window lower boundary [B, 1]

        Returns:
            P_t: Interpolated distribution at time t [B, L, V]
        """
        # Compute interpolation coefficient
        alpha = (t - t_k_minus_1) / (t_k - t_k_minus_1 + 1e-10)  # [B, 1]
        alpha = alpha.unsqueeze(-1)  # [B, 1, 1] for broadcasting

        # Linear interpolation
        P_t = alpha * P_t_k + (1 - alpha) * P_t_k_minus_1

        # Ensure valid probability distribution
        P_t = P_t.clamp(min=0)
        P_t = P_t / (P_t.sum(dim=-1, keepdim=True) + 1e-10)

        return P_t

    def resample_from_distribution(self, probs: torch.Tensor):
        """
        Resample discrete tokens from probability distribution.

        Args:
            probs: Probability distribution [B, L, V]

        Returns:
            x: Sampled tokens [B, L]
        """
        return sample_categorical(probs, method="hard")

    def compute_kl_loss(
        self,
        target_probs: torch.Tensor,
        student_logits: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Compute KL divergence loss: D_KL(target || student)

        Args:
            target_probs: Target probability distribution P [B, L, V]
            student_logits: Student model log-probabilities [B, L, V]
            mask: Optional mask for positions to include [B, L]

        Returns:
            loss: KL divergence loss [B] (per sample)
        """
        # Student log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # KL divergence: sum_v P[v] * (log P[v] - log Q[v])
        # = sum_v P[v] * log P[v] - sum_v P[v] * log Q[v]
        # We compute -sum_v P[v] * log Q[v] (cross-entropy) and add entropy of P

        # Cross-entropy term: -sum_v P[v] * log Q[v]
        cross_entropy = -(target_probs * student_log_probs).sum(dim=-1)  # [B, L]

        # Entropy term: -sum_v P[v] * log P[v]
        target_log_probs = (target_probs + 1e-10).log()
        entropy = -(target_probs * target_log_probs).sum(dim=-1)  # [B, L]

        # KL = cross_entropy - entropy = -sum P log Q + sum P log P
        kl_div = cross_entropy - entropy  # [B, L]

        if mask is not None:
            kl_div = kl_div * mask
            loss = kl_div.sum(dim=-1) / (mask.sum(dim=-1) + 1e-10)
        else:
            loss = kl_div.mean(dim=-1)  # [B]

        return loss

    def compute_kl_loss_probs(
        self,
        target_probs: torch.Tensor,
        student_probs: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Compute KL divergence loss between two probability distributions: D_KL(target || student)

        Args:
            target_probs: Target probability distribution P [B, L, V]
            student_probs: Student probability distribution Q [B, L, V]
            mask: Optional mask for positions to include [B, L]

        Returns:
            loss: KL divergence loss [B] (per sample)
        """
        # Student log probabilities (from probability distribution)
        student_log_probs = (student_probs + 1e-10).log()

        # Cross-entropy term: -sum_v P[v] * log Q[v]
        cross_entropy = -(target_probs * student_log_probs).sum(dim=-1)  # [B, L]

        # Entropy term: -sum_v P[v] * log P[v]
        target_log_probs = (target_probs + 1e-10).log()
        entropy = -(target_probs * target_log_probs).sum(dim=-1)  # [B, L]

        # KL = cross_entropy - entropy
        kl_div = cross_entropy - entropy  # [B, L]

        if mask is not None:
            kl_div = kl_div * mask
            loss = kl_div.sum(dim=-1) / (mask.sum(dim=-1) + 1e-10)
        else:
            loss = kl_div.mean(dim=-1)  # [B]

        return loss

    def compute_reverse_kl_loss_probs(
        self,
        target_probs: torch.Tensor,
        student_probs: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Compute Reverse KL divergence loss: D_KL(student || target)

        Reverse KL is more robust to mode collapse as it penalizes
        student for putting mass where target has low probability.

        Args:
            target_probs: Target probability distribution P [B, L, V]
            student_probs: Student probability distribution Q [B, L, V]
            mask: Optional mask for positions to include [B, L]

        Returns:
            loss: Reverse KL divergence loss [B] (per sample)
        """
        # Student log probabilities
        student_log_probs = (student_probs + 1e-10).log()

        # Target log probabilities
        target_log_probs = (target_probs + 1e-10).log()

        # Reverse KL: sum_v Q[v] * (log Q[v] - log P[v])
        # = -H(Q) + CE(Q, P)

        # Cross-entropy term: -sum_v Q[v] * log P[v]
        cross_entropy = -(student_probs * target_log_probs).sum(dim=-1)  # [B, L]

        # Negative entropy term: sum_v Q[v] * log Q[v] = -H(Q)
        neg_entropy = (student_probs * student_log_probs).sum(dim=-1)  # [B, L]

        # Reverse KL = -H(Q) + CE(Q, P) = neg_entropy + cross_entropy
        kl_div = neg_entropy + cross_entropy  # [B, L]

        if mask is not None:
            kl_div = kl_div * mask
            loss = kl_div.sum(dim=-1) / (mask.sum(dim=-1) + 1e-10)
        else:
            loss = kl_div.mean(dim=-1)  # [B]

        return loss


def get_d_perflow_loss_fn(
    noise,
    graph,
    teacher_model,
    num_time_windows: int = 4,
    sampling_eps: float = 1e-3,
    euler_steps: int = 1,
    train: bool = True,
    train_temperature: float = 1.0,
    debug_log_file: str = None,
    lambda_fwd_kl: float = 1.0,
    lambda_rev_kl: float = 0.0,
    lambda_perceptual: float = 0.0,
    perceptual_loss_fn=None,
    teacher_method: str = "euler",
    student_method: str = "analytic",
):
    """
    Create the D-PeRFlow loss function.

    Args:
        noise: Noise schedule module
        graph: Graph structure (Absorbing or Uniform)
        teacher_model: Pre-trained teacher model
        num_time_windows: Number of time windows K
        sampling_eps: Small epsilon to avoid t=0
        euler_steps: Number of Euler steps per window
        train: Whether in training mode
        train_temperature: Temperature for softening distributions during training
                          Higher temperature = softer distributions = more diverse outputs
                          Default 1.0 means no temperature scaling
        debug_log_file: Path to save debug logs (if None, print to console)
        lambda_fwd_kl: Weight for forward KL loss KL(P_teacher || P_student) (default: 1.0)
        lambda_rev_kl: Weight for reverse KL loss KL(P_student || P_teacher) (default: 0.0)
        lambda_perceptual: Weight for perceptual loss (default: 0.0)
        perceptual_loss_fn: GPT2PerceptualLoss instance (required if lambda_perceptual > 0)
        student_method: Method for student distribution computation:
            - "euler": Taylor approximation I + dt*Q (fast)
            - "analytic": Exact exp(ΔσQ) (more accurate, recommended)

    Returns:
        loss_fn: Loss function that takes (model, batch) and returns loss
    """
    trainer = DPerflowTrainer(
        graph=graph,
        noise=noise,
        num_time_windows=num_time_windows,
        sampling_eps=sampling_eps,
    )

    # Get teacher score function (always in eval mode, returns true score for Euler)
    teacher_score_fn = mutils.get_score_fn(teacher_model, train=False, sampling=True)

    def student_score_wrapper(model, train_mode):
        """
        Create a score function for the student that returns TRUE scores (not log).
        Uses sampling=False (allows train=True for gradients) + manual exp() with clamping.
        """
        log_score_fn = mutils.get_score_fn(model, train=train_mode, sampling=False)

        def wrapped_score_fn(x, sigma):
            log_score = log_score_fn(x, sigma)
            # Clamp to avoid overflow in exp()
            log_score = log_score.clamp(min=-30, max=30)
            return log_score.exp()

        return wrapped_score_fn

    def loss_fn(model, batch):
        """
        Compute D-PeRFlow loss with window interpolation.

        1. Compute P_{t_k} (exact forward) and P_{t_{k-1}} (teacher denoised)
        2. Sample random time t within window, interpolate to get P_t
        3. Sample discrete x_t ~ P_t
        4. Student denoises x_t from t to t_{k-1}
        5. Loss = KL(P_{t_{k-1}} || P_student)

        Args:
            model: Student model
            batch: Input batch of token ids [B, L]

        Returns:
            loss: Loss value [B]
        """
        device = batch.device
        batch_size = batch.shape[0]

        # 1. Sample window index k uniformly from {1, ..., K}
        k = torch.randint(1, num_time_windows + 1, (batch_size,), device=device)

        # Get window boundaries
        t_k_minus_1, t_k = trainer.get_window_boundaries(k, device)
        t_k_minus_1 = t_k_minus_1.unsqueeze(-1)  # [B, 1]
        t_k = t_k.unsqueeze(-1)  # [B, 1]

        # 2. Construct noisy state x_{t_k} at window upper boundary
        with torch.no_grad():
            x_t_k = trainer.construct_noisy_state(batch, t_k)

        # 3. Teacher computes distributions at both window boundaries
        with torch.no_grad():
            P_t_k, P_t_k_minus_1 = trainer.compute_teacher_distributions(
                teacher_score_fn, x_t_k, t_k, t_k_minus_1, euler_steps,
                teacher_method=teacher_method,
                x_0=batch,
            )

            # 4. Sample random time t within window [t_{k-1}, t_k]
            u = torch.rand(batch_size, 1, device=device)  # [B, 1]
            t = t_k_minus_1 + u * (t_k - t_k_minus_1)     # [B, 1]

            # 5. Interpolate to get P_t, then sample discrete x_t
            P_t = trainer.interpolate_distribution(P_t_k, P_t_k_minus_1, t, t_k, t_k_minus_1)
            x_t = trainer.resample_from_distribution(P_t)

        # 6. Student: 1-step from x_t at time t to t_{k-1}
        student_score_fn_wrapped = student_score_wrapper(model, train_mode=train)
        if student_method == "analytic":
            P_student = trainer.compute_analytic_multi_step(
                student_score_fn_wrapped, x_t, t, t_k_minus_1, num_steps=1
            )
        else:
            P_student = trainer.compute_euler_distribution(
                student_score_fn_wrapped, x_t, t, t_k_minus_1, num_steps=1
            )

        # 7. Compute weighted loss: lambda_fwd * fwd_KL + lambda_rev * rev_KL + lambda_perc * perceptual
        loss = torch.zeros(batch_size, device=device)

        if lambda_fwd_kl > 0:
            fwd_kl = trainer.compute_kl_loss_probs(P_t_k_minus_1, P_student)
            loss = loss + lambda_fwd_kl * fwd_kl

        if lambda_rev_kl > 0:
            rev_kl = trainer.compute_reverse_kl_loss_probs(P_t_k_minus_1, P_student)
            loss = loss + lambda_rev_kl * rev_kl

        if lambda_perceptual > 0 and perceptual_loss_fn is not None:
            p_loss = perceptual_loss_fn(P_t_k_minus_1, P_student)
            loss = loss + lambda_perceptual * p_loss

        # Debug: print distribution statistics every 100 steps
        if hasattr(loss_fn, 'debug_step'):
            loss_fn.debug_step += 1
            if loss_fn.debug_step % 100 == 0:
                with torch.no_grad():
                    # Teacher distribution entropy
                    teacher_entropy = -(P_t_k_minus_1 * (P_t_k_minus_1 + 1e-10).log()).sum(dim=-1).mean()

                    # Student distribution entropy
                    student_entropy = -(P_student * (P_student + 1e-10).log()).sum(dim=-1).mean()

                    # Argmax tokens analysis
                    teacher_argmax = P_t_k_minus_1.argmax(dim=-1)  # [B, L]
                    student_argmax = P_student.argmax(dim=-1)  # [B, L]

                    # Unique argmax tokens per sample (diversity)
                    seq_len = teacher_argmax.shape[1]
                    teacher_unique = torch.tensor([len(torch.unique(row)) for row in teacher_argmax]).float().mean()
                    student_unique = torch.tensor([len(torch.unique(row)) for row in student_argmax]).float().mean()

                    # Agreement rate
                    agreement = (teacher_argmax == student_argmax).float().mean() * 100

                    # Top-5 most common student argmax tokens
                    student_flat = student_argmax.flatten()
                    token_counts = torch.bincount(student_flat, minlength=P_student.shape[-1])
                    top5_tokens = token_counts.topk(5)

                    # Sampled k values for context
                    k_mean = k.float().mean()

                    # Format debug message
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    debug_msg = (
                        f"\n[DEBUG Step {loss_fn.debug_step}] {timestamp}\n"
                        f"  Method: SEDD native Euler (1-step student, {euler_steps}-step teacher)\n"
                        f"  Avg window k: {k_mean:.1f}/{num_time_windows}\n"
                        f"  Teacher: entropy={teacher_entropy:.4f}, unique_argmax={teacher_unique:.1f}/{seq_len}\n"
                        f"  Student: entropy={student_entropy:.4f}, unique_argmax={student_unique:.1f}/{seq_len}\n"
                        f"  Argmax agreement: {agreement:.1f}%\n"
                        f"  Student top5 tokens: {top5_tokens.indices.tolist()} counts: {top5_tokens.values.tolist()}\n"
                        f"  Loss: {loss.mean():.4f}\n"
                    )

                    # Print to console
                    print(debug_msg)

                    # Also write to file if specified
                    if debug_log_file is not None:
                        with open(debug_log_file, 'a') as f:
                            f.write(debug_msg)
        else:
            loss_fn.debug_step = 0

        return loss

    return loss_fn


def get_d_perflow_step_fn(
    noise,
    graph,
    teacher_model,
    num_time_windows: int = 4,
    sampling_eps: float = 1e-3,
    euler_steps: int = 1,
    train: bool = True,
    optimize_fn=None,
    accum: int = 1,
    train_temperature: float = 1.0,
    debug_log_file: str = None,
    lambda_fwd_kl: float = 1.0,
    lambda_rev_kl: float = 0.0,
    lambda_perceptual: float = 0.0,
    perceptual_loss_fn=None,
    teacher_method: str = "euler",
    student_method: str = "analytic",
):
    """
    Create the D-PeRFlow training step function.

    Args:
        noise: Noise schedule module
        graph: Graph structure
        teacher_model: Pre-trained teacher model
        num_time_windows: Number of time windows
        sampling_eps: Epsilon to avoid t=0
        euler_steps: Euler steps per window
        train: Training mode flag
        optimize_fn: Optimization function
        accum: Gradient accumulation steps
        train_temperature: Temperature for softening distributions during training
        debug_log_file: Path to save debug logs (if None, print to console only)
        lambda_fwd_kl: Weight for forward KL loss (default: 1.0)
        lambda_rev_kl: Weight for reverse KL loss (default: 0.0)
        lambda_perceptual: Weight for perceptual loss (default: 0.0)
        perceptual_loss_fn: GPT2PerceptualLoss instance (required if lambda_perceptual > 0)
        student_method: "euler" or "analytic" for student distribution computation

    Returns:
        step_fn: Training step function
    """
    loss_fn = get_d_perflow_loss_fn(
        noise=noise,
        graph=graph,
        teacher_model=teacher_model,
        num_time_windows=num_time_windows,
        sampling_eps=sampling_eps,
        euler_steps=euler_steps,
        train=train,
        train_temperature=train_temperature,
        debug_log_file=debug_log_file,
        lambda_fwd_kl=lambda_fwd_kl,
        lambda_rev_kl=lambda_rev_kl,
        lambda_perceptual=lambda_perceptual,
        perceptual_loss_fn=perceptual_loss_fn,
        teacher_method=teacher_method,
        student_method=student_method,
    )

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch):
        nonlocal accum_iter
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch).mean() / accum

            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()

            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()

                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn


class DPerflowSampler:
    """
    D-PeRFlow sampler for few-step generation.

    After training, the student model can generate samples in K steps,
    where K is the number of time windows.
    """

    def __init__(self, graph, noise, num_time_windows: int = 4, sampling_eps: float = 1e-3, temperature: float = 1.0, debug: bool = False, debug_log_file: str = None):
        self.graph = graph
        self.noise = noise
        self.num_time_windows = num_time_windows
        self.sampling_eps = sampling_eps
        self.temperature = temperature
        self.debug = debug
        self.debug_log_file = debug_log_file

        # Time boundaries
        self.time_boundaries = torch.linspace(
            sampling_eps, 1.0, num_time_windows + 1
        )

    @torch.no_grad()
    def sample(self, model, batch_dims, device):
        """
        Generate samples using the trained student model.

        Uses SEDD's native Euler step mechanism — the same mechanism used during training.
        This eliminates distribution shift between training and inference.

        At each window k: compute 1-step Euler from t_k to t_{k-1}, then sample.

        Args:
            model: Trained student model
            batch_dims: Tuple of (batch_size, seq_length)
            device: Target device

        Returns:
            x: Generated samples [B, L]
        """
        # Use sampling=True for inference (eval mode, returns true scores via .exp())
        score_fn = mutils.get_score_fn(model, train=False, sampling=True)

        # Start from pure noise (all masks for absorbing graph)
        x = self.graph.sample_limit(*batch_dims).to(device)

        # Sample through each window in reverse order using Euler steps
        for k in range(self.num_time_windows, 0, -1):
            t_k = self.time_boundaries[k].to(device)
            t_k_minus_1 = self.time_boundaries[k - 1].to(device)

            # Time tensors [B]
            t_start = t_k * torch.ones(batch_dims[0], device=device)
            t_end = t_k_minus_1 * torch.ones(batch_dims[0], device=device)

            # Compute 1-step analytic predictor from t_k to t_{k-1}
            # Uses exact matrix exponential exp(ΔσQ) instead of Taylor approx I + dt*Q
            sigma_start = self.noise(t_start)[0]  # [B]
            sigma_end = self.noise(t_end)[0]       # [B]
            dsigma = sigma_start - sigma_end       # [B]

            score = score_fn(x, sigma_start)  # [B, L, V] - true scores

            dsigma_expanded = dsigma[:, None]  # [B, 1]
            stag_score = self.graph.staggered_score(score, dsigma_expanded)
            probs = stag_score * self.graph.transp_transition(x, dsigma_expanded)

            # Clamp and normalize
            probs = probs.clamp(min=0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

            # Debug: print distribution statistics at each step
            if self.debug:
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()

                # Argmax analysis
                argmax_tokens = probs.argmax(dim=-1)  # [B, L]
                seq_len = argmax_tokens.shape[1]
                unique_argmax = torch.tensor([len(torch.unique(row)) for row in argmax_tokens]).float().mean()

                # Top-5 most common argmax tokens
                flat = argmax_tokens.flatten()
                token_counts = torch.bincount(flat, minlength=probs.shape[-1])
                top5 = token_counts.topk(5)

                # Count unique tokens in current x (before sampling)
                unique_in_x = len(torch.unique(x))

                # How many positions actually change?
                # (probs at one_hot position vs new token probability)
                stay_prob = probs.gather(-1, x.unsqueeze(-1)).squeeze(-1).mean()

                debug_msg = (
                    f"[Sampling Step {self.num_time_windows - k + 1}/{self.num_time_windows}] "
                    f"t={t_k:.4f} -> {t_k_minus_1:.4f}\n"
                    f"  entropy={entropy:.4f}, unique_argmax={unique_argmax:.1f}/{seq_len}, "
                    f"unique_in_x={unique_in_x}\n"
                    f"  stay_prob={stay_prob:.4f} (prob of keeping current token)\n"
                    f"  top5_tokens: {top5.indices.tolist()} counts: {top5.values.tolist()}\n"
                )

                print(debug_msg)

                if self.debug_log_file is not None:
                    with open(self.debug_log_file, 'a') as f:
                        f.write(debug_msg)

            # Sample from distribution
            x = sample_categorical(probs, method="hard")

        # Final denoising step: for absorbing graph, unmask remaining mask tokens
        if self.graph.absorb:
            # At t=eps, use analytic predictor to remove remaining masks
            t = self.sampling_eps * torch.ones(batch_dims[0], device=device)
            sigma, dsigma = self.noise(t)
            score = score_fn(x, sigma)

            # Use stag_score * transp_transition for final denoising
            stag_score = self.graph.staggered_score(score, dsigma[:, None])
            probs = stag_score * self.graph.transp_transition(x, dsigma[:, None])
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

            # Exclude mask token for final output
            probs = probs[..., :-1]
            x = sample_categorical(probs, method="hard")

        return x


def get_d_perflow_sampler(graph, noise, num_time_windows: int = 4, sampling_eps: float = 1e-3, temperature: float = 1.0, debug: bool = False, debug_log_file: str = None):
    """
    Create a D-PeRFlow sampler.

    Args:
        graph: Graph structure
        noise: Noise schedule
        num_time_windows: Number of windows (= generation steps)
        sampling_eps: Epsilon to avoid t=0
        temperature: Temperature for softmax (higher = more diverse)
        debug: Whether to print debug information during sampling
        debug_log_file: Path to save debug logs (if None, print to console only)

    Returns:
        sampler: DPerflowSampler instance
    """
    return DPerflowSampler(graph, noise, num_time_windows, sampling_eps, temperature, debug, debug_log_file)


# ==============================================================================
# Adversarial Distillation Integration
# ==============================================================================


def get_adversarial_d_perflow_loss_fn(
    noise,
    graph,
    teacher_model,
    adv_loss_module,
    num_time_windows: int = 4,
    sampling_eps: float = 1e-3,
    euler_steps: int = 1,
    train: bool = True,
    debug_log_file: str = None,
    lambda_rev_kl: float = 0.0,
):
    """
    Create D-PeRFlow loss function with adversarial distillation.

    Returns two loss functions:
    - generator_loss_fn: For updating the student model (KL + rev_KL + adversarial)
    - discriminator_loss_fn: For updating the discriminator(s)

    Args:
        noise: Noise schedule module
        graph: Graph structure (Absorbing or Uniform)
        teacher_model: Pre-trained teacher model
        adv_loss_module: AdversarialDistillationLoss instance
        num_time_windows: Number of time windows K
        sampling_eps: Small epsilon to avoid t=0
        euler_steps: Number of Euler steps per window
        train: Whether in training mode
        debug_log_file: Path to save debug logs
        lambda_rev_kl: Weight for reverse KL loss (mode-seeking signal)

    Returns:
        generator_loss_fn: Loss function for student updates
        discriminator_loss_fn: Loss function for discriminator updates
    """
    trainer = DPerflowTrainer(
        graph=graph,
        noise=noise,
        num_time_windows=num_time_windows,
        sampling_eps=sampling_eps,
    )

    teacher_score_fn = mutils.get_score_fn(teacher_model, train=False, sampling=True)

    def student_score_wrapper(model, train_mode):
        log_score_fn = mutils.get_score_fn(model, train=train_mode, sampling=False)

        def wrapped_score_fn(x, sigma):
            log_score = log_score_fn(x, sigma)
            log_score = log_score.clamp(min=-30, max=30)
            return log_score.exp()

        return wrapped_score_fn

    def _compute_distributions(model, batch):
        """Shared computation for both generator and discriminator losses."""
        device = batch.device
        batch_size = batch.shape[0]

        # Sample window index k
        k = torch.randint(1, num_time_windows + 1, (batch_size,), device=device)

        # Get window boundaries
        t_k_minus_1, t_k = trainer.get_window_boundaries(k, device)
        t_k_minus_1 = t_k_minus_1.unsqueeze(-1)  # [B, 1]
        t_k = t_k.unsqueeze(-1)  # [B, 1]

        # Construct noisy state
        with torch.no_grad():
            x_t_k = trainer.construct_noisy_state(batch, t_k)

        # Teacher distributions at both boundaries
        with torch.no_grad():
            P_t_k, P_teacher = trainer.compute_teacher_distributions(
                teacher_score_fn, x_t_k, t_k, t_k_minus_1, euler_steps,
                x_0=batch,
            )

            # Sample random time t within window, interpolate, resample
            u = torch.rand(batch_size, 1, device=device)
            t = t_k_minus_1 + u * (t_k - t_k_minus_1)
            P_t = trainer.interpolate_distribution(P_t_k, P_teacher, t, t_k, t_k_minus_1)
            x_t = trainer.resample_from_distribution(P_t)

        # Student distribution (with gradients) from x_t at time t
        student_score_fn_wrapped = student_score_wrapper(model, train_mode=train)
        P_student = trainer.compute_analytic_multi_step(
            student_score_fn_wrapped, x_t, t, t_k_minus_1, num_steps=1
        )

        return P_teacher, P_student, x_t, t.squeeze(-1), k

    def _write_debug(msg):
        """Helper to print and optionally write debug message to file."""
        print(msg)
        if debug_log_file is not None:
            with open(debug_log_file, 'a') as f:
                f.write(msg)

    def generator_loss_fn(model, batch):
        """
        Compute combined loss for the student model:
            L = L_fwd_KL + lambda_rev_kl * L_rev_KL + lambda_adv * L_adv + lambda_gpt2 * L_gpt2

        Args:
            model: Student model
            batch: Input batch [B, L]

        Returns:
            loss: Combined loss [B]
            metrics: Dict with loss components
        """
        P_teacher, P_student, x_t_k, t, k = _compute_distributions(model, batch)

        # Forward KL divergence loss (mode-covering)
        kl_loss = trainer.compute_kl_loss_probs(P_teacher, P_student)

        # Reverse KL divergence loss (mode-seeking, sharpens student)
        rev_kl_loss = None
        if lambda_rev_kl > 0:
            rev_kl_loss = trainer.compute_reverse_kl_loss_probs(P_teacher, P_student)

        # Get last layer param for adaptive lambda (VQGAN-style)
        # Uses the student model's final projection layer (DDitFinalLayer.linear)
        last_layer_param = None
        if adv_loss_module.adaptive_lambda:
            raw_model = model.module if hasattr(model, 'module') else model
            if hasattr(raw_model, 'output_layer') and hasattr(raw_model.output_layer, 'linear'):
                last_layer_param = raw_model.output_layer.linear.weight
            elif hasattr(raw_model, 'output') and hasattr(raw_model.output, 'weight'):
                last_layer_param = raw_model.output.weight
            elif hasattr(raw_model, 'lm_head') and hasattr(raw_model.lm_head, 'weight'):
                last_layer_param = raw_model.lm_head.weight

        # Combined loss: fwd_KL + rev_KL + proj_adv + gpt2_adv + feature_matching
        total_loss, metrics = adv_loss_module.combined_loss(
            kl_loss, P_student, P_teacher, x_t_k, t,
            rev_kl_loss=rev_kl_loss, lambda_rev_kl=lambda_rev_kl,
            last_layer_param=last_layer_param,
        )

        # Comprehensive debug logging every 100 steps
        if hasattr(generator_loss_fn, 'debug_step'):
            generator_loss_fn.debug_step += 1
            if generator_loss_fn.debug_step % 100 == 0:
                with torch.no_grad():
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # === Distribution entropy ===
                    teacher_entropy = -(P_teacher * (P_teacher + 1e-10).log()).sum(dim=-1).mean()
                    student_entropy = -(P_student * (P_student + 1e-10).log()).sum(dim=-1).mean()

                    # === Per-position max probability (confidence) ===
                    teacher_max_prob = P_teacher.max(dim=-1).values.mean()
                    student_max_prob = P_student.max(dim=-1).values.mean()

                    # === Argmax analysis ===
                    teacher_argmax = P_teacher.argmax(dim=-1)  # [B, L]
                    student_argmax = P_student.argmax(dim=-1)  # [B, L]
                    seq_len = teacher_argmax.shape[1]

                    # Agreement rate
                    agreement = (teacher_argmax == student_argmax).float().mean() * 100

                    # Unique tokens per sample (diversity measure)
                    teacher_unique = torch.tensor([len(torch.unique(row)) for row in teacher_argmax]).float()
                    student_unique = torch.tensor([len(torch.unique(row)) for row in student_argmax]).float()

                    # === Top-5 most common tokens ===
                    teacher_flat = teacher_argmax.flatten()
                    student_flat = student_argmax.flatten()
                    teacher_counts = torch.bincount(teacher_flat, minlength=P_teacher.shape[-1])
                    student_counts = torch.bincount(student_flat, minlength=P_student.shape[-1])
                    teacher_top5 = teacher_counts.topk(5)
                    student_top5 = student_counts.topk(5)

                    # === KL divergence per window ===
                    kl_per_sample = kl_loss.detach()
                    kl_min, kl_max, kl_mean, kl_std = kl_per_sample.min(), kl_per_sample.max(), kl_per_sample.mean(), kl_per_sample.std()

                    # === Discriminator predictions on current batch ===
                    _, disc_seq_teacher = adv_loss_module.discriminator(P_teacher, x_t_k, t)
                    disc_tok_student, disc_seq_student = adv_loss_module.discriminator(P_student, x_t_k, t)
                    disc_prob_teacher = torch.sigmoid(disc_seq_teacher).mean()
                    disc_prob_student = torch.sigmoid(disc_seq_student).mean()
                    # Discriminator accuracy (sequence-level)
                    disc_acc_real = (disc_seq_teacher > 0).float().mean() * 100
                    disc_acc_fake = (disc_seq_student < 0).float().mean() * 100
                    disc_acc_total = (disc_acc_real + disc_acc_fake) / 2
                    # Token-level accuracy
                    tok_acc_fake = (disc_tok_student < 0).float().mean() * 100

                    # === JS divergence (symmetric measure) ===
                    M = 0.5 * (P_teacher + P_student)
                    js_div = 0.5 * (P_teacher * ((P_teacher + 1e-10) / (M + 1e-10)).log()).sum(-1).mean() + \
                             0.5 * (P_student * ((P_student + 1e-10) / (M + 1e-10)).log()).sum(-1).mean()

                    # === L2 distance between distributions ===
                    l2_dist = (P_teacher - P_student).pow(2).sum(dim=-1).sqrt().mean()

                    # === Window-wise breakdown ===
                    k_values = k.detach()
                    window_strs = []
                    for w in range(1, num_time_windows + 1):
                        mask = (k_values == w)
                        if mask.sum() > 0:
                            w_kl = kl_per_sample[mask].mean()
                            window_strs.append(f"k={w}: kl={w_kl:.4f}(n={mask.sum()})")

                    # Reverse KL info for debug
                    rev_kl_str = ""
                    if 'gen_rev_kl_loss' in metrics:
                        rev_kl_str = f"  Rev KL:     {metrics['gen_rev_kl_loss']:.6f}  (lambda={metrics.get('lambda_rev_kl', 0):.3f})\n"
                    gpt2_adv_str = ""
                    if metrics.get('gen_gpt2_adv_loss', 0) > 0:
                        gpt2_adv_str = f"  GPT2 Adv:   {metrics['gen_gpt2_adv_loss']:.6f}  (lambda={metrics.get('lambda_gpt2_adv', 0):.3f})\n"

                    debug_msg = (
                        f"\n{'='*80}\n"
                        f"[ADV-GEN Step {generator_loss_fn.debug_step}] {timestamp}\n"
                        f"{'='*80}\n"
                        f"  Config: Adversarial D-PeRFlow | 1-step student | {euler_steps}-step teacher\n"
                        f"  Lambda_adv: {metrics['lambda_adv']} | Lambda_gpt2: {metrics.get('lambda_gpt2_adv', 0)} | Lambda_rev_kl: {metrics.get('lambda_rev_kl', 0)}\n"
                        f"\n  --- Loss Breakdown ---\n"
                        f"  Fwd KL:     {metrics['gen_fwd_kl_loss']:.6f}  (min={kl_min:.4f} max={kl_max:.4f} std={kl_std:.4f})\n"
                        f"{rev_kl_str}"
                        f"  Proj Adv:   {metrics['gen_proj_adv_loss']:.6f}\n"
                        f"{gpt2_adv_str}"
                        f"  Total loss: {metrics['gen_total_loss']:.6f}\n"
                        f"\n  --- Distribution Quality ---\n"
                        f"  Teacher entropy:  {teacher_entropy:.4f}  |  Student entropy:  {student_entropy:.4f}  |  ratio: {student_entropy/(teacher_entropy+1e-10):.4f}\n"
                        f"  Teacher max_prob: {teacher_max_prob:.4f}  |  Student max_prob: {student_max_prob:.4f}\n"
                        f"  JS divergence:    {js_div:.6f}\n"
                        f"  L2 distance:      {l2_dist:.6f}\n"
                        f"\n  --- Token Diversity ---\n"
                        f"  Argmax agreement: {agreement:.1f}%\n"
                        f"  Teacher unique tokens: mean={teacher_unique.mean():.1f} std={teacher_unique.std():.1f} / {seq_len}\n"
                        f"  Student unique tokens: mean={student_unique.mean():.1f} std={student_unique.std():.1f} / {seq_len}\n"
                        f"  Teacher top5 tokens: {teacher_top5.indices.tolist()} counts: {teacher_top5.values.tolist()}\n"
                        f"  Student top5 tokens: {student_top5.indices.tolist()} counts: {student_top5.values.tolist()}\n"
                        f"\n  --- Discriminator Status ---\n"
                        f"  D(teacher) prob: {disc_prob_teacher:.4f}  |  D(student) prob: {disc_prob_student:.4f}\n"
                        f"  D seq accuracy: {disc_acc_total:.1f}% (real={disc_acc_real:.1f}%, fake={disc_acc_fake:.1f}%)\n"
                        f"  D tok accuracy (fake): {tok_acc_fake:.1f}%\n"
                        f"  D seq logit gap: {(disc_seq_teacher.mean() - disc_seq_student.mean()):.4f}\n"
                        f"\n  --- Per-Window Breakdown ---\n"
                        f"  Avg window k: {k.float().mean():.2f}/{num_time_windows}\n"
                        f"  {' | '.join(window_strs)}\n"
                        f"{'='*80}\n"
                    )
                    _write_debug(debug_msg)
        else:
            generator_loss_fn.debug_step = 0

        return total_loss, metrics

    def discriminator_loss_fn(model, batch):
        """
        Compute discriminator loss for both projected and GPT-2 discriminators.

        Args:
            model: Student model (used to generate student distribution)
            batch: Input batch [B, L]

        Returns:
            loss: Total discriminator loss scalar
            metrics: Dict with loss components from both discriminators
        """
        with torch.no_grad():
            P_teacher, P_student, x_t_k, t, k = _compute_distributions(model, batch)

        disc_loss, metrics = adv_loss_module.discriminator_loss(P_teacher, P_student, x_t_k, t)

        # GPT-2 discriminator loss (if enabled)
        if adv_loss_module.gpt2_discriminator is not None:
            gpt2_disc_loss, gpt2_metrics = adv_loss_module.gpt2_discriminator_loss(
                P_teacher, P_student, x_t_k, t
            )
            disc_loss = disc_loss + gpt2_disc_loss
            metrics.update(gpt2_metrics)

        # Discriminator-specific debug logging every 100 steps
        if hasattr(discriminator_loss_fn, 'debug_step'):
            discriminator_loss_fn.debug_step += 1
            if discriminator_loss_fn.debug_step % 100 == 0:
                with torch.no_grad():
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Gradient norm of discriminator
                    disc_params = adv_loss_module.discriminator.parameters()
                    grad_norm = 0.0
                    param_norm = 0.0
                    for p in disc_params:
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                        param_norm += p.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    param_norm = param_norm ** 0.5

                    debug_msg = (
                        f"\n[ADV-DISC Step {discriminator_loss_fn.debug_step}] {timestamp}\n"
                        f"  Disc total loss:  {metrics['disc_loss']:.6f}\n"
                        f"  Disc token loss:  {metrics['disc_token_loss']:.6f}\n"
                        f"  Disc seq loss:    {metrics['disc_seq_loss']:.6f}\n"
                        f"  R1 penalty:       {metrics['disc_r1_penalty']:.6f}\n"
                        f"  D(real) seq logit:  {metrics['disc_real_logit_mean']:.4f}\n"
                        f"  D(fake) seq logit:  {metrics['disc_fake_logit_mean']:.4f}\n"
                        f"  Seq logit gap:      {metrics['disc_real_logit_mean'] - metrics['disc_fake_logit_mean']:.4f}\n"
                        f"  Tok acc (real):     {metrics['disc_real_tok_acc']:.4f}\n"
                        f"  Tok acc (fake):     {metrics['disc_fake_tok_acc']:.4f}\n"
                        f"  Disc grad norm:   {grad_norm:.4f}\n"
                        f"  Disc param norm:  {param_norm:.4f}\n"
                    )
                    _write_debug(debug_msg)
        else:
            discriminator_loss_fn.debug_step = 0

        return disc_loss, metrics

    return generator_loss_fn, discriminator_loss_fn


def get_adversarial_d_perflow_step_fn(
    noise,
    graph,
    teacher_model,
    adv_loss_module,
    num_time_windows: int = 4,
    sampling_eps: float = 1e-3,
    euler_steps: int = 1,
    train: bool = True,
    optimize_fn=None,
    disc_optimizer=None,
    accum: int = 1,
    disc_steps_per_gen: int = 1,
    debug_log_file: str = None,
    lambda_rev_kl: float = 0.0,
):
    """
    Create D-PeRFlow training step with adversarial distillation.

    Alternates between discriminator and generator (student) updates:
    - disc_steps_per_gen discriminator updates per generator update
    - Generator update uses combined KL + reverse KL + adversarial loss

    Args:
        noise: Noise schedule module
        graph: Graph structure
        teacher_model: Pre-trained teacher model
        adv_loss_module: AdversarialDistillationLoss instance
        num_time_windows: Number of time windows
        sampling_eps: Epsilon to avoid t=0
        euler_steps: Euler steps per window
        train: Training mode flag
        optimize_fn: Optimization function for student
        disc_optimizer: Optimizer for discriminator
        accum: Gradient accumulation steps
        disc_steps_per_gen: Number of discriminator updates per generator update
        debug_log_file: Path to save debug logs
        lambda_rev_kl: Weight for reverse KL loss (mode-seeking)

    Returns:
        step_fn: Training step function
    """
    gen_loss_fn, disc_loss_fn = get_adversarial_d_perflow_loss_fn(
        noise=noise,
        graph=graph,
        teacher_model=teacher_model,
        adv_loss_module=adv_loss_module,
        num_time_windows=num_time_windows,
        sampling_eps=sampling_eps,
        euler_steps=euler_steps,
        train=train,
        debug_log_file=debug_log_file,
        lambda_rev_kl=lambda_rev_kl,
    )

    accum_iter = 0
    total_gen_loss = 0
    # Track where we are in the disc/gen cycle.
    # Phase: 0..disc_steps_per_gen-1 = discriminator updates, then gen update
    phase_counter = 0

    def step_fn(state, batch):
        nonlocal accum_iter, total_gen_loss, phase_counter

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']

            # --- Discriminator update phase ---
            if phase_counter < disc_steps_per_gen:
                phase_counter += 1

                disc_loss, disc_metrics = disc_loss_fn(model, batch)

                disc_optimizer.zero_grad()
                disc_loss.backward()
                # Clip gradients for projected discriminator
                torch.nn.utils.clip_grad_norm_(
                    adv_loss_module.discriminator.parameters(), max_norm=100.0
                )
                # Clip gradients for GPT-2 discriminator head (if enabled)
                if adv_loss_module.gpt2_discriminator is not None:
                    torch.nn.utils.clip_grad_norm_(
                        adv_loss_module.gpt2_discriminator.parameters(), max_norm=100.0
                    )
                disc_optimizer.step()

                state['disc_metrics'] = disc_metrics
                return disc_loss.detach()

            # --- Generator (student) update phase ---
            # Reset phase counter for next cycle
            phase_counter = 0

            gen_loss, gen_metrics = gen_loss_fn(model, batch)
            loss = gen_loss.mean() / accum

            scaler.scale(loss).backward()

            accum_iter += 1
            total_gen_loss += loss.detach()

            if accum_iter == accum:
                accum_iter = 0
                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()

                loss = total_gen_loss
                total_gen_loss = 0

            state['gen_metrics'] = gen_metrics
            return loss
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                gen_loss, _ = gen_loss_fn(model, batch)
                loss = gen_loss.mean()
                ema.restore(model.parameters())
            return loss

    return step_fn
