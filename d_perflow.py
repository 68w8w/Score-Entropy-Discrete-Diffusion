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
    euler_steps: int = 1,
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
        euler_steps: Number of Euler steps per window
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
        Compute D-PeRFlow loss using Forward KL with SEDD native Euler steps.

        Teacher computes target distribution P_{t_{k-1}} via multi-step Euler.
        Student computes distribution via 1-step Euler using its own scores.
        Loss = KL(P_target || P_student)

        This ensures training and inference use the SAME mechanism (Euler step),
        eliminating the distribution shift that caused mode collapse with softmax.

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

        # 3. Teacher computes target distribution P_{t_{k-1}} via multi-step Euler
        with torch.no_grad():
            _, P_t_k_minus_1 = trainer.compute_teacher_distributions(
                teacher_score_fn, x_t_k, t_k, t_k_minus_1, euler_steps
            )

        # 4. Student: 1-step Euler using student's scores
        # This uses the SAME Euler mechanism as SEDD sampling:
        #   probs = one_hot(x) + dt * dsigma * reverse_rate(x, score)
        # The student's scores flow through this differentiably
        student_score_fn_wrapped = student_score_wrapper(model, train_mode=train)
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
    delta_t: float = 1e-4,
    sampling_eps: float = 1e-3,
    euler_steps: int = 1,
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
        euler_steps: Euler steps per window
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
        euler_steps=euler_steps,
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

            # Compute 1-step Euler from t_k to t_{k-1}
            # This is the SAME mechanism used during training
            dt = t_start - t_end  # [B]
            sigma, dsigma = self.noise(t_start)  # Both [B]
            score = score_fn(x, sigma)  # [B, L, V] - true scores

            # SEDD native Euler step: probs = one_hot(x) + dt * dsigma * reverse_rate(x, score)
            scale = (dt * dsigma)[:, None, None]  # [B, 1, 1]
            rev_rate = scale * self.graph.reverse_rate(x, score)  # [B, L, V]
            one_hot_current = F.one_hot(x, num_classes=self.graph.dim).float()
            probs = one_hot_current + rev_rate

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


# =============================================================================
# Consistency Training for Discrete Diffusion
# =============================================================================

def get_consistency_loss_fn(
    noise,
    graph,
    teacher_model,
    num_time_windows: int = 4,
    sampling_eps: float = 1e-3,
    train: bool = True,
    consistency_weight: float = 1.0,
    distillation_weight: float = 1.0,
    debug_log_file: str = None,
):
    """
    Create the Consistency Training loss function for discrete diffusion.

    Consistency Training learns a model that can directly predict x_0 from any
    point on the diffusion trajectory. The key constraint is that predictions
    from different points on the SAME trajectory should be CONSISTENT.

    Loss = consistency_loss + distillation_loss

    consistency_loss: f(x_t, t) should equal f(x_{t-dt}, t-dt)
    distillation_loss: f(x_t, t) should match teacher's prediction of x_0

    Args:
        noise: Noise schedule module
        graph: Graph structure (Absorbing)
        teacher_model: Pre-trained teacher model
        num_time_windows: Number of time windows (for sampling time points)
        sampling_eps: Epsilon to avoid t=0
        train: Training mode
        consistency_weight: Weight for consistency loss
        distillation_weight: Weight for distillation loss (matching teacher)
        debug_log_file: Path for debug logs

    Returns:
        loss_fn: Loss function
    """
    # Get teacher score function
    teacher_score_fn = mutils.get_score_fn(teacher_model, train=False, sampling=True)

    def compute_x0_prediction(score, x, mask_token_id=None):
        """
        Compute P(x_0 | x_t) from score predictions.

        For SEDD with absorbing graph:
        - At MASKED positions: P(x_0 = k) ∝ score_k (normalize over non-mask tokens)
        - At non-masked positions: P(x_0 = x_t) = 1 (token is revealed)

        Args:
            score: Model scores [B, L, V]
            x: Current noisy tokens [B, L]
            mask_token_id: ID of the mask token (default: V-1 for absorbing graph)

        Returns:
            x0_probs: Predicted P(x_0 | x_t) distribution [B, L, V-1] (excludes mask token)
        """
        B, L, V = score.shape

        if mask_token_id is None:
            mask_token_id = V - 1  # Absorbing graph uses last token as mask

        # Initialize output probabilities (excluding mask token)
        x0_probs = torch.zeros(B, L, V - 1, device=score.device, dtype=score.dtype)

        # Identify masked and non-masked positions
        is_masked = (x == mask_token_id)  # [B, L]

        # For MASKED positions: normalize score over non-mask tokens
        # score[:, :, :-1] gives scores for actual tokens (not mask)
        score_for_tokens = score[:, :, :-1]  # [B, L, V-1]

        # Softmax to get probabilities (for masked positions)
        masked_probs = F.softmax(score_for_tokens, dim=-1)  # [B, L, V-1]

        # For NON-MASKED positions: one-hot at current token
        non_mask_probs = F.one_hot(x.clamp(max=V-2), num_classes=V-1).float()  # [B, L, V-1]

        # Combine: use masked_probs where masked, non_mask_probs otherwise
        is_masked_expanded = is_masked.unsqueeze(-1)  # [B, L, 1]
        x0_probs = torch.where(is_masked_expanded, masked_probs, non_mask_probs)

        return x0_probs

    def teacher_one_step_euler(x, t, dt):
        """
        Run one Euler step using the teacher to get x_{t-dt}.

        Args:
            x: Current state [B, L]
            t: Current time [B]
            dt: Step size (scalar or [B])

        Returns:
            x_next: State at t-dt [B, L]
        """
        sigma, dsigma = noise(t)  # Both [B]
        score = teacher_score_fn(x, sigma)  # [B, L, V]

        # SEDD Euler step
        if isinstance(dt, float):
            dt_tensor = torch.full_like(t, dt)
        else:
            dt_tensor = dt

        scale = (dt_tensor * dsigma)[:, None, None]  # [B, 1, 1]
        rev_rate = scale * graph.reverse_rate(x, score)  # [B, L, V]

        one_hot_current = F.one_hot(x, num_classes=graph.dim).float()
        probs = one_hot_current + rev_rate
        probs = probs.clamp(min=0)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

        # Sample
        x_next = sample_categorical(probs, method="hard")
        return x_next

    def loss_fn(model, batch):
        """
        Compute Consistency Training loss.

        1. Sample time t uniformly in [eps, 1]
        2. Construct x_t from x_0
        3. Student predicts x_0 from x_t: f(x_t, t)
        4. Teacher takes one step: x_t → x_{t-dt}
        5. Student predicts x_0 from x_{t-dt}: f(x_{t-dt}, t-dt)
        6. Consistency loss: KL(f(x_{t-dt}, t-dt) || f(x_t, t))
        7. Distillation loss: KL(teacher_x0_pred || student_x0_pred)
        """
        device = batch.device
        batch_size = batch.shape[0]
        x_0 = batch

        # 1. Sample time uniformly
        t = torch.rand(batch_size, device=device) * (1.0 - sampling_eps) + sampling_eps  # [B]

        # 2. Construct x_t
        sigma_t = noise(t)[0]  # [B]
        x_t = graph.sample_transition(x_0, sigma_t[:, None])  # [B, L]

        # 3. Student predicts x_0 from x_t
        student_log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        student_log_score_t = student_log_score_fn(x_t, sigma_t)  # [B, L, V]
        # Convert log-score to score for x0 prediction
        student_log_score_t_clamped = student_log_score_t.clamp(min=-30, max=30)
        student_x0_pred_t = compute_x0_prediction(student_log_score_t_clamped, x_t)  # [B, L, V-1]

        # 4. Teacher takes one step: x_t → x_{t-dt}
        # dt = small step (we use fraction of remaining time)
        dt = 0.1 * t  # 10% of current time as step size
        t_next = t - dt  # [B]
        t_next = t_next.clamp(min=sampling_eps)  # Don't go below eps
        actual_dt = t - t_next  # Actual step taken

        with torch.no_grad():
            x_t_next = teacher_one_step_euler(x_t, t, actual_dt)  # [B, L]

        # 5. Student predicts x_0 from x_{t-dt} (STOP GRADIENT on target)
        sigma_t_next = noise(t_next)[0]

        # For consistency loss, we use EMA model or stop gradient on target
        # Here we use stop gradient on the target prediction
        with torch.no_grad():
            student_log_score_t_next = student_log_score_fn(x_t_next, sigma_t_next)
            student_log_score_t_next_clamped = student_log_score_t_next.clamp(min=-30, max=30)
            student_x0_pred_t_next = compute_x0_prediction(student_log_score_t_next_clamped, x_t_next)

        # 6. Consistency loss: predictions should match
        # KL(target || source) where target is from t-dt (stopped gradient)
        consistency_loss = F.kl_div(
            (student_x0_pred_t + 1e-10).log(),  # source (has gradient)
            student_x0_pred_t_next,  # target (no gradient)
            reduction='none'
        ).sum(dim=-1).mean(dim=-1)  # [B]

        # 7. Distillation loss: match teacher's x0 prediction
        with torch.no_grad():
            teacher_score_t = teacher_score_fn(x_t, sigma_t)  # [B, L, V]
            teacher_x0_pred = compute_x0_prediction(teacher_score_t, x_t)  # [B, L, V-1]

        distillation_loss = F.kl_div(
            (student_x0_pred_t + 1e-10).log(),
            teacher_x0_pred,
            reduction='none'
        ).sum(dim=-1).mean(dim=-1)  # [B]

        # Combined loss
        loss = consistency_weight * consistency_loss + distillation_weight * distillation_loss

        # Debug logging
        if hasattr(loss_fn, 'debug_step'):
            loss_fn.debug_step += 1
            if loss_fn.debug_step % 100 == 0:
                with torch.no_grad():
                    # Student prediction entropy
                    student_entropy = -(student_x0_pred_t * (student_x0_pred_t + 1e-10).log()).sum(dim=-1).mean()

                    # Teacher prediction entropy
                    teacher_entropy = -(teacher_x0_pred * (teacher_x0_pred + 1e-10).log()).sum(dim=-1).mean()

                    # Agreement rate
                    student_argmax = student_x0_pred_t.argmax(dim=-1)
                    teacher_argmax = teacher_x0_pred.argmax(dim=-1)
                    agreement = (student_argmax == teacher_argmax).float().mean() * 100

                    # Unique predictions
                    seq_len = student_argmax.shape[1]
                    student_unique = torch.tensor([len(torch.unique(row)) for row in student_argmax]).float().mean()

                    # Masked ratio
                    mask_token_id = graph.dim - 1
                    masked_ratio = (x_t == mask_token_id).float().mean() * 100

                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    debug_msg = (
                        f"\n[DEBUG Step {loss_fn.debug_step}] {timestamp}\n"
                        f"  Method: Consistency Training\n"
                        f"  Avg t: {t.mean():.4f}, masked_ratio: {masked_ratio:.1f}%\n"
                        f"  Teacher: entropy={teacher_entropy:.4f}\n"
                        f"  Student: entropy={student_entropy:.4f}, unique_argmax={student_unique:.1f}/{seq_len}\n"
                        f"  Argmax agreement: {agreement:.1f}%\n"
                        f"  Consistency loss: {consistency_loss.mean():.4f}\n"
                        f"  Distillation loss: {distillation_loss.mean():.4f}\n"
                        f"  Total loss: {loss.mean():.4f}\n"
                    )

                    print(debug_msg)

                    if debug_log_file is not None:
                        with open(debug_log_file, 'a') as f:
                            f.write(debug_msg)
        else:
            loss_fn.debug_step = 0

        return loss

    return loss_fn


def get_consistency_step_fn(
    noise,
    graph,
    teacher_model,
    num_time_windows: int = 4,
    sampling_eps: float = 1e-3,
    train: bool = True,
    optimize_fn=None,
    accum: int = 1,
    consistency_weight: float = 1.0,
    distillation_weight: float = 1.0,
    debug_log_file: str = None,
):
    """
    Create the Consistency Training step function.
    """
    loss_fn = get_consistency_loss_fn(
        noise=noise,
        graph=graph,
        teacher_model=teacher_model,
        num_time_windows=num_time_windows,
        sampling_eps=sampling_eps,
        train=train,
        consistency_weight=consistency_weight,
        distillation_weight=distillation_weight,
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


class ConsistencySampler:
    """
    Sampler for Consistency-trained models.

    Consistency models can generate in 1 step by directly predicting x_0,
    or in multiple steps for better quality.
    """

    def __init__(self, graph, noise, num_steps: int = 1, sampling_eps: float = 1e-3, debug: bool = False):
        self.graph = graph
        self.noise = noise
        self.num_steps = num_steps
        self.sampling_eps = sampling_eps
        self.debug = debug

    @torch.no_grad()
    def sample(self, model, batch_dims, device):
        """
        Generate samples using consistency model.

        For num_steps=1: Direct one-shot generation from noise
        For num_steps>1: Iterative refinement
        """
        score_fn = mutils.get_score_fn(model, train=False, sampling=False)

        # Start from pure noise
        x = self.graph.sample_limit(*batch_dims).to(device)

        # Time steps for multi-step generation
        times = torch.linspace(1.0, self.sampling_eps, self.num_steps + 1, device=device)

        mask_token_id = self.graph.dim - 1

        for i in range(self.num_steps):
            t = times[i] * torch.ones(batch_dims[0], device=device)
            sigma = self.noise(t)[0]

            # Get model predictions
            log_score = score_fn(x, sigma)
            log_score = log_score.clamp(min=-30, max=30)

            # For masked positions: predict x_0 using softmax over non-mask tokens
            score_for_tokens = log_score[:, :, :-1]  # Exclude mask token
            x0_probs = F.softmax(score_for_tokens, dim=-1)  # [B, L, V-1]

            if self.debug:
                entropy = -(x0_probs * (x0_probs + 1e-10).log()).sum(dim=-1).mean()
                masked_ratio = (x == mask_token_id).float().mean() * 100
                print(f"[Step {i+1}/{self.num_steps}] t={times[i]:.4f}, masked_ratio={masked_ratio:.1f}%, entropy={entropy:.4f}")

            # Sample x_0 for masked positions
            is_masked = (x == mask_token_id)

            # Sample from x0_probs
            sampled_x0 = sample_categorical(x0_probs, method="hard")  # [B, L]

            # Update only masked positions
            x = torch.where(is_masked, sampled_x0, x)

            # For multi-step: optionally add noise back for intermediate steps
            if i < self.num_steps - 1:
                # Re-noise to intermediate time
                t_next = times[i + 1]
                sigma_next = self.noise(t_next * torch.ones(batch_dims[0], device=device))[0]
                x = self.graph.sample_transition(x, sigma_next[:, None])

        return x


def get_consistency_sampler(graph, noise, num_steps: int = 1, sampling_eps: float = 1e-3, debug: bool = False):
    """
    Create a Consistency model sampler.

    Args:
        graph: Graph structure
        noise: Noise schedule
        num_steps: Number of sampling steps (1 for one-shot, >1 for iterative)
        sampling_eps: Epsilon to avoid t=0
        debug: Print debug info

    Returns:
        sampler: ConsistencySampler instance
    """
    return ConsistencySampler(graph, noise, num_steps, sampling_eps, debug)
