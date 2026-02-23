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

    def compute_analytical_distribution(
        self,
        score_fn,
        x: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        num_steps: int = 2,
        dsigma_threshold: float = 0.1,
    ):
        """
        Compute the target distribution using pure Euler method.

        This uses the same Euler ODE solver as compute_euler_distribution,
        but with the num_steps parameter from analytical config (default: 2).

        Euler: probs = one_hot(x) + dt * dsigma * reverse_rate(x, score)
        As dsigma → 0, probs → one_hot(x), which is the correct identity limit.

        Args:
            score_fn: Score function from teacher model (returns true scores)
            x: Starting discrete state [B, L]
            t_start: Start time [B] or [B, 1]
            t_end: End time [B] or [B, 1]
            num_steps: Number of Euler steps (default: 2)
            dsigma_threshold: Unused, kept for API compatibility

        Returns:
            probs: Target probability distribution at t_end [B, L, V]
        """
        # Ensure time tensors are 1D [B]
        t_start_1d = t_start.squeeze(-1) if t_start.dim() > 1 else t_start
        t_end_1d = t_end.squeeze(-1) if t_end.dim() > 1 else t_end

        # Total time to traverse
        dt = (t_start_1d - t_end_1d) / num_steps  # [B]

        # Initialize current state and time
        current_x = x
        current_t = t_start_1d.clone()  # [B]
        probs = None

        for step in range(num_steps):
            # Get current sigma and dsigma/dt
            curr_sigma, curr_dsigma_dt = self.noise(current_t)  # Both [B]
            next_t = (current_t - dt).clamp(min=self.sampling_eps)
            step_dt = current_t - next_t  # Actual dt after clamping

            # Compute score
            score = score_fn(current_x, curr_sigma)  # [B, L, V]

            # Pure Euler distribution
            scale = (step_dt * curr_dsigma_dt)[:, None, None]  # [B, 1, 1]
            rev_rate = scale * self.graph.reverse_rate(current_x, score)  # [B, L, V]
            one_hot_current = F.one_hot(current_x, num_classes=self.graph.dim).float()
            probs = one_hot_current + rev_rate
            probs = probs.clamp(min=0)

            # Normalize
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

            # Update time
            current_t = next_t

            # For multi-step, sample intermediate states
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
    delta_t: float = 1e-4,
    sampling_eps: float = 1e-3,
    teacher_steps: int = 2,
    teacher_sampler: str = "analytical",
    train: bool = True,
    train_temperature: float = 1.0,
    debug_log_file: str = None,
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
        teacher_steps: Number of steps for teacher sampling (default: 2 as in SDTT)
        teacher_sampler: Teacher sampling method - 'analytical' (SEDD native) or 'euler'
                        'analytical' uses discrete transition matrix (recommended)
                        'euler' uses continuous ODE approximation
        train: Whether in training mode
        train_temperature: Temperature for softening distributions during training
                          Higher temperature = softer distributions = more diverse outputs
                          Default 1.0 means no temperature scaling
        debug_log_file: Path to save debug logs (if None, print to console)

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

    # Get teacher score function (always in eval mode, returns true score)
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
        Compute D-PeRFlow loss using Forward KL.

        Teacher computes target distribution P_{t_{k-1}} via multi-step sampling.
        Student computes distribution via 1-step using its own scores.
        Loss = KL(P_target || P_student)

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

        # 3. Teacher computes target distribution P_{t_{k-1}}
        # Using either Analytical (SEDD native) or Euler sampling
        with torch.no_grad():
            if teacher_sampler == "analytical":
                # SEDD native discrete sampling (recommended)
                P_t_k_minus_1 = trainer.compute_analytical_distribution(
                    teacher_score_fn, x_t_k, t_k, t_k_minus_1, num_steps=teacher_steps
                )
            else:
                # Euler ODE approximation (legacy)
                P_t_k_minus_1 = trainer.compute_euler_distribution(
                    teacher_score_fn, x_t_k, t_k, t_k_minus_1, num_steps=teacher_steps
                )

        # 4. Student: 1-step using student's scores
        # Use the SAME method as teacher for consistency
        student_score_fn_wrapped = student_score_wrapper(model, train_mode=train)
        if teacher_sampler == "analytical":
            P_student = trainer.compute_analytical_distribution(
                student_score_fn_wrapped, x_t_k, t_k, t_k_minus_1, num_steps=1
            )
        else:
            P_student = trainer.compute_euler_distribution(
                student_score_fn_wrapped, x_t_k, t_k, t_k_minus_1, num_steps=1
            )

        # 5. Forward KL divergence loss: KL(P_target || P_student)
        loss = trainer.compute_kl_loss_probs(P_t_k_minus_1, P_student)

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
                        f"  Method: {teacher_sampler.upper()} ({teacher_steps}-step teacher, 1-step student)\n"
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
    delta_t: float = 1e-4,
    sampling_eps: float = 1e-3,
    teacher_steps: int = 2,
    teacher_sampler: str = "analytical",
    train: bool = True,
    optimize_fn=None,
    accum: int = 1,
    train_temperature: float = 1.0,
    debug_log_file: str = None,
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
        teacher_steps: Number of steps for teacher sampling (default: 2)
        teacher_sampler: 'analytical' (SEDD native, recommended) or 'euler'
        train: Training mode flag
        optimize_fn: Optimization function
        accum: Gradient accumulation steps
        train_temperature: Temperature for softening distributions during training
        debug_log_file: Path to save debug logs (if None, print to console only)

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
        teacher_steps=teacher_steps,
        teacher_sampler=teacher_sampler,
        train=train,
        train_temperature=train_temperature,
        debug_log_file=debug_log_file,
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

    def __init__(self, graph, noise, num_time_windows: int = 4, sampling_eps: float = 1e-3,
                 sampler_type: str = "analytical", temperature: float = 1.0,
                 debug: bool = False, debug_log_file: str = None):
        self.graph = graph
        self.noise = noise
        self.num_time_windows = num_time_windows
        self.sampling_eps = sampling_eps
        self.sampler_type = sampler_type  # 'analytical' or 'euler'
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

        Uses either Analytical (SEDD native) or Euler sampling mechanism.
        Should match the method used during training.

        At each window k: compute 1-step from t_k to t_{k-1}, then sample.

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

        # Sample through each window in reverse order
        for k in range(self.num_time_windows, 0, -1):
            t_k = self.time_boundaries[k].to(device)
            t_k_minus_1 = self.time_boundaries[k - 1].to(device)

            # Time tensors [B]
            t_start = t_k * torch.ones(batch_dims[0], device=device)
            t_end = t_k_minus_1 * torch.ones(batch_dims[0], device=device)

            # Get sigma values
            curr_sigma = self.noise(t_start)[0]  # [B]
            next_sigma = self.noise(t_end)[0]  # [B]
            dsigma = curr_sigma - next_sigma  # [B]

            # Compute score
            score = score_fn(x, curr_sigma)  # [B, L, V] - true scores

            if self.sampler_type == "analytical":
                # SEDD native Analytical sampling (recommended)
                # probs = staggered_score(score, dsigma) * transp_transition(x, dsigma)
                dsigma_expanded = dsigma[:, None]  # [B, 1]
                stag_score = self.graph.staggered_score(score, dsigma_expanded)  # [B, L, V]
                probs = stag_score * self.graph.transp_transition(x, dsigma_expanded)  # [B, L, V]
            else:
                # Euler ODE approximation (legacy)
                dt = t_start - t_end  # [B]
                _, dsigma_euler = self.noise(t_start)  # dsigma from noise schedule
                scale = (dt * dsigma_euler)[:, None, None]  # [B, 1, 1]
                rev_rate = scale * self.graph.reverse_rate(x, score)  # [B, L, V]
                one_hot_current = F.one_hot(x, num_classes=self.graph.dim).float()
                probs = one_hot_current + rev_rate
                probs = probs.clamp(min=0)

            # Normalize
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


def get_d_perflow_sampler(graph, noise, num_time_windows: int = 4, sampling_eps: float = 1e-3,
                          sampler_type: str = "analytical", temperature: float = 1.0,
                          debug: bool = False, debug_log_file: str = None):
    """
    Create a D-PeRFlow sampler.

    Args:
        graph: Graph structure
        noise: Noise schedule
        num_time_windows: Number of windows (= generation steps)
        sampling_eps: Epsilon to avoid t=0
        sampler_type: 'analytical' (SEDD native, recommended) or 'euler'
        temperature: Temperature for softmax (higher = more diverse)
        debug: Whether to print debug information during sampling
        debug_log_file: Path to save debug logs (if None, print to console only)

    Returns:
        sampler: DPerflowSampler instance
    """
    return DPerflowSampler(graph, noise, num_time_windows, sampling_eps, sampler_type, temperature, debug, debug_log_file)
