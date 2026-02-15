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
        delta_t: float = 1e-4,
        sampling_eps: float = 1e-3,
    ):
        """
        Args:
            graph: The discrete graph structure (Absorbing or Uniform)
            noise: The noise schedule module
            num_time_windows: Number of time windows K (default: 4 for 4-step generation)
            delta_t: Infinitesimal step for computing start distribution (default: 1e-4)
            sampling_eps: Small epsilon to avoid t=0 numerical issues (default: 1e-3)
        """
        self.graph = graph
        self.noise = noise
        self.num_time_windows = num_time_windows
        self.delta_t = delta_t
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

    def compute_teacher_distributions(
        self,
        teacher_score_fn,
        x_t_k: torch.Tensor,
        t_k: torch.Tensor,
        t_k_minus_1: torch.Tensor,
        euler_steps: int = 1,
    ):
        """
        Compute teacher distributions at window boundaries.

        Args:
            teacher_score_fn: Score function from teacher model
            x_t_k: Noisy state at window start [B, L]
            t_k: Window upper boundary time [B, 1]
            t_k_minus_1: Window lower boundary time [B, 1]
            euler_steps: Number of Euler steps for ODE solving

        Returns:
            P_t_k: Distribution at window start [B, L, V]
            P_t_k_minus_1: Target distribution at window end [B, L, V]
        """
        # A. Compute start distribution P_{t_k} using analytic predictor with small step
        delta_t_tensor = torch.full_like(t_k, self.delta_t)
        P_t_k = self.compute_analytic_distribution(
            teacher_score_fn, x_t_k, t_k, delta_t_tensor
        )

        # B. Compute target distribution P_{t_{k-1}} using Euler ODE solver
        P_t_k_minus_1 = self.compute_euler_distribution(
            teacher_score_fn, x_t_k, t_k, t_k_minus_1, num_steps=euler_steps
        )

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


def get_d_perflow_loss_fn(
    noise,
    graph,
    teacher_model,
    num_time_windows: int = 4,
    delta_t: float = 1e-4,
    sampling_eps: float = 1e-3,
    euler_steps: int = 1,
    train: bool = True,
):
    """
    Create the D-PeRFlow loss function.

    Args:
        noise: Noise schedule module
        graph: Graph structure (Absorbing or Uniform)
        teacher_model: Pre-trained teacher model
        num_time_windows: Number of time windows K
        delta_t: Small step for analytic distribution computation
        sampling_eps: Small epsilon to avoid t=0
        euler_steps: Number of Euler steps per window
        train: Whether in training mode

    Returns:
        loss_fn: Loss function that takes (model, batch) and returns loss
    """
    trainer = DPerflowTrainer(
        graph=graph,
        noise=noise,
        num_time_windows=num_time_windows,
        delta_t=delta_t,
        sampling_eps=sampling_eps,
    )

    # Get teacher score function (always in eval mode)
    teacher_score_fn = mutils.get_score_fn(teacher_model, train=False, sampling=True)

    def loss_fn(model, batch):
        """
        Compute D-PeRFlow loss for a batch.

        Args:
            model: Student model
            batch: Input batch of token ids [B, L]

        Returns:
            loss: Loss value [B]
        """
        device = batch.device
        batch_size = batch.shape[0]

        # 1. Sample window index k uniformly from {1, ..., K}
        k = torch.randint(
            1, num_time_windows + 1, (batch_size,), device=device
        )

        # Get window boundaries
        t_k_minus_1, t_k = trainer.get_window_boundaries(k, device)
        t_k_minus_1 = t_k_minus_1.unsqueeze(-1)  # [B, 1]
        t_k = t_k.unsqueeze(-1)  # [B, 1]

        # 2. Sample time t uniformly within window (t_{k-1}, t_k]
        u = torch.rand(batch_size, 1, device=device)
        t = t_k_minus_1 + u * (t_k - t_k_minus_1)

        # 3. Construct noisy state x_{t_k} at window boundary
        with torch.no_grad():
            x_t_k = trainer.construct_noisy_state(batch, t_k)

        # 4. Teacher guidance: compute boundary distributions
        with torch.no_grad():
            P_t_k, P_t_k_minus_1 = trainer.compute_teacher_distributions(
                teacher_score_fn, x_t_k, t_k, t_k_minus_1, euler_steps
            )

        # 5. Distribution interpolation
        with torch.no_grad():
            P_t = trainer.interpolate_distribution(
                P_t_k, P_t_k_minus_1, t, t_k, t_k_minus_1
            )

        # 6. Resample student input from mixed distribution
        with torch.no_grad():
            x_t = trainer.resample_from_distribution(P_t)

        # 7. Student prediction
        student_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        sigma_t = noise(t)[0]  # [B, 1]
        student_logits = student_score_fn(x_t, sigma_t)  # [B, L, V]

        # 8. Compute KL divergence loss
        # Target is P_{t_{k-1}}, student predicts at time t
        loss = trainer.compute_kl_loss(P_t_k_minus_1, student_logits)

        return loss

    return loss_fn


def get_d_perflow_step_fn(
    noise,
    graph,
    teacher_model,
    num_time_windows: int = 4,
    delta_t: float = 1e-4,
    sampling_eps: float = 1e-3,
    euler_steps: int = 1,
    train: bool = True,
    optimize_fn=None,
    accum: int = 1,
):
    """
    Create the D-PeRFlow training step function.

    Args:
        noise: Noise schedule module
        graph: Graph structure
        teacher_model: Pre-trained teacher model
        num_time_windows: Number of time windows
        delta_t: Small step for analytic distribution
        sampling_eps: Epsilon to avoid t=0
        euler_steps: Euler steps per window
        train: Training mode flag
        optimize_fn: Optimization function
        accum: Gradient accumulation steps

    Returns:
        step_fn: Training step function
    """
    loss_fn = get_d_perflow_loss_fn(
        noise=noise,
        graph=graph,
        teacher_model=teacher_model,
        num_time_windows=num_time_windows,
        delta_t=delta_t,
        sampling_eps=sampling_eps,
        euler_steps=euler_steps,
        train=train,
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

    def __init__(self, graph, noise, num_time_windows: int = 4, sampling_eps: float = 1e-3):
        self.graph = graph
        self.noise = noise
        self.num_time_windows = num_time_windows
        self.sampling_eps = sampling_eps

        # Time boundaries
        self.time_boundaries = torch.linspace(
            sampling_eps, 1.0, num_time_windows + 1
        )

    @torch.no_grad()
    def sample(self, model, batch_dims, device):
        """
        Generate samples using the trained student model.

        Uses the same probability computation as SEDD's AnalyticPredictor:
        stag_score * transp_transition, but with K large steps (one per window).

        Args:
            model: Trained student model
            batch_dims: Tuple of (batch_size, seq_length)
            device: Target device

        Returns:
            x: Generated samples [B, L]
        """
        # Use sampling=True to get true score (exp of model output), same as SEDD
        score_fn = mutils.get_score_fn(model, train=False, sampling=True)

        # Start from pure noise (all masks for absorbing graph)
        x = self.graph.sample_limit(*batch_dims).to(device)

        # Sample through each window in reverse order
        for k in range(self.num_time_windows, 0, -1):
            t_k = self.time_boundaries[k].to(device)
            t_k_minus_1 = self.time_boundaries[k - 1].to(device)

            # Compute sigma at window boundaries
            t = t_k * torch.ones(batch_dims[0], device=device)
            t_prev = t_k_minus_1 * torch.ones(batch_dims[0], device=device)

            curr_sigma = self.noise(t)[0]
            next_sigma = self.noise(t_prev)[0]
            dsigma = curr_sigma - next_sigma

            # Get model score (true score, not log)
            score = score_fn(x, curr_sigma)

            # Compute transition probabilities (same as SEDD AnalyticPredictor)
            stag_score = self.graph.staggered_score(score, dsigma)
            probs = stag_score * self.graph.transp_transition(x, dsigma)

            # Sample from distribution
            x = sample_categorical(probs, method="hard")

        # Final denoising step (same as SEDD Denoiser)
        if self.graph.absorb:
            t = self.sampling_eps * torch.ones(batch_dims[0], device=device)
            sigma = self.noise(t)[0]

            score = score_fn(x, sigma)
            stag_score = self.graph.staggered_score(score, sigma)
            probs = stag_score * self.graph.transp_transition(x, sigma)

            # Exclude mask token
            probs = probs[..., :-1]
            x = sample_categorical(probs, method="hard")

        return x


def get_d_perflow_sampler(graph, noise, num_time_windows: int = 4, sampling_eps: float = 1e-3):
    """
    Create a D-PeRFlow sampler.

    Args:
        graph: Graph structure
        noise: Noise schedule
        num_time_windows: Number of windows (= generation steps)
        sampling_eps: Epsilon to avoid t=0

    Returns:
        sampler: DPerflowSampler instance
    """
    return DPerflowSampler(graph, noise, num_time_windows, sampling_eps)
