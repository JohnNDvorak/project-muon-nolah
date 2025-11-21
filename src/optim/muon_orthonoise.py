"""
MuonOrthoNoise Optimizer - Method #1: Low-Magnitude Orthogonal Noise

Adds TRUE orthogonal perturbations to momentum buffer via QR decomposition.
Geometrically principled approach with rigorous mathematical foundation.

Core concept: O_tilde = O_t + epsilon * Q
- O_t: Current momentum buffer
- Q: Orthogonal noise from QR decomposition (guarantees Q^T @ Q = I)
- epsilon: Gradient-norm scaled noise magnitude

Key improvements over NOLAH:
1. QR decomposition guarantees orthogonality (not heuristic)
2. Gradient-norm scaling has principled interpretation
3. Newton-Schulz re-orthogonalization restores conditioning
4. Adaptive triggering based on effective rank

Reference: TRANSITION_TO_ORTHONOISE.md
"""

import torch
from torch.optim import Optimizer
from typing import Dict, Any
from .muon import Muon


class MuonOrthoNoise(Muon):
    """
    Muon optimizer with orthogonal noise perturbations.

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-4)
        momentum: Momentum coefficient (default: 0.95)
        alpha: Noise scale hyperparameter (default: 1e-2)
        annealing: Enable noise annealing schedule (default: True)
        adaptive: Enable adaptive triggering based on effective rank (default: True)
        rank_threshold_ratio: Ratio of min(dims) for rank threshold (default: 0.5)
        newton_schulz_iters: Newton-Schulz re-orthogonalization iterations (default: 3)
        adamw_lr: Learning rate for non-matrix params (default: 3e-4)
        adamw_betas: AdamW beta coefficients (default: (0.9, 0.95))
        adamw_eps: AdamW epsilon (default: 1e-8)
        adamw_wd: AdamW weight decay (default: 0.01)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.95,
        alpha: float = 1e-2,
        annealing: bool = True,
        adaptive: bool = True,
        rank_threshold_ratio: float = 0.5,
        newton_schulz_iters: int = 3,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.01,
    ):
        # Initialize parent Muon optimizer
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        # OrthoNoise specific parameters
        self.alpha = alpha
        self.annealing = annealing
        self.adaptive = adaptive
        self.rank_threshold_ratio = rank_threshold_ratio
        self.newton_schulz_iters = newton_schulz_iters

        # Tracking for metrics
        self.noise_added_count = 0
        self.noise_skipped_count = 0

    def _generate_orthogonal_noise(self, p: torch.Tensor) -> torch.Tensor:
        """
        Generate orthogonal noise via QR decomposition.

        QR decomposition of Gaussian random matrix guarantees orthonormality:
        Q^T @ Q = I

        Args:
            p: Parameter tensor (used for shape)

        Returns:
            Orthogonal noise matrix Q
        """
        # Generate random Gaussian matrix with same shape as momentum buffer
        R = torch.randn_like(self.buffers[p])

        # QR decomposition - Q is orthonormal by construction
        # CRITICAL: May need float32 conversion like SVD
        if R.dtype == torch.bfloat16:
            R_float32 = R.float()
            Q, _ = torch.linalg.qr(R_float32)
            Q = Q.to(R.dtype)
        else:
            Q, _ = torch.linalg.qr(R)

        return Q

    def _compute_epsilon(self, g: torch.Tensor) -> float:
        """
        Compute noise scale proportional to gradient norm.

        epsilon = alpha * ||g||_F

        With annealing: alpha decays from initial value to 0.1 * alpha over 1000 steps
        - Exponential decay: alpha * (0.1 ** (step / 1000))

        Args:
            g: Gradient tensor

        Returns:
            Noise scale epsilon
        """
        # Compute Frobenius norm of gradient
        grad_norm = torch.norm(g, p='fro')

        # Apply annealing if enabled
        if self.annealing:
            # Exponential decay: alpha from 1e-2 -> 1e-3 over 1000 steps
            decay_factor = 0.1 ** (self.step_count / 1000.0)
            alpha_current = self.alpha * decay_factor
        else:
            alpha_current = self.alpha

        epsilon = alpha_current * grad_norm.item()
        return epsilon

    def _compute_effective_rank(self, M: torch.Tensor) -> float:
        """
        Compute effective rank via entropy of normalized singular values.

        effective_rank = exp(H(S))
        where H(S) = -sum(s_i * log(s_i)) for normalized singular values

        Low effective rank indicates poor conditioning - momentum buffer
        has collapsed to low-dimensional subspace.

        Args:
            M: Momentum buffer matrix

        Returns:
            Effective rank (float)
        """
        # Convert to float32 for SVD
        M_float32 = M.float()

        try:
            # Compute singular values only (faster than full SVD)
            S = torch.linalg.svdvals(M_float32)

            # Normalize singular values
            S_norm = S / (S.sum() + 1e-10)  # Add epsilon for stability

            # Compute entropy
            entropy = -(S_norm * torch.log(S_norm + 1e-8)).sum()

            # Effective rank is exp(entropy)
            effective_rank = torch.exp(entropy).item()

            return effective_rank

        except RuntimeError:
            # If SVD fails, assume low rank (trigger noise)
            return 0.0

    def _should_add_noise(self, p: torch.Tensor) -> bool:
        """
        Adaptive triggering based on effective rank.

        Add noise only when momentum buffer has low effective rank,
        indicating poor conditioning that would benefit from perturbation.

        Args:
            p: Parameter tensor

        Returns:
            True if noise should be added
        """
        if not self.adaptive:
            # Always add noise if adaptive triggering disabled
            return True

        # Compute effective rank of momentum buffer
        effective_rank = self._compute_effective_rank(self.buffers[p])

        # Threshold: 50% of minimum dimension
        # E.g., for 768x3072 matrix, threshold = 0.5 * 768 = 384
        rank_threshold = self.rank_threshold_ratio * min(p.shape)

        return effective_rank < rank_threshold

    def _reorthogonalize(self, M: torch.Tensor, iterations: int = None) -> torch.Tensor:
        """
        Newton-Schulz iterations to restore orthogonality.

        Newton-Schulz method for computing orthogonal projection:
        M_{k+1} = 1.5 * M_k - 0.5 * M_k @ M_k^T @ M_k

        Converges to nearest orthogonal matrix. 3 iterations typically sufficient.

        Args:
            M: Matrix to re-orthogonalize
            iterations: Number of iterations (default: use self.newton_schulz_iters)

        Returns:
            Re-orthogonalized matrix
        """
        if iterations is None:
            iterations = self.newton_schulz_iters

        M_orth = M
        for _ in range(iterations):
            M_orth = 1.5 * M_orth - 0.5 * M_orth @ M_orth.T @ M_orth

        return M_orth

    def _muon_step(self, p: torch.Tensor, group: dict) -> None:
        """
        Apply Muon update with orthogonal noise to matrix parameter.

        Overrides parent _muon_step to add OrthoNoise modifications:
        1. Standard momentum update: M = momentum * M + g
        2. Add orthogonal noise: M_tilde = M + epsilon * Q
        3. Re-orthogonalize: M_orth = NewtonSchulz(M_tilde)
        4. SVD-based update: p -= lr * U @ V^T

        Args:
            p: Parameter tensor
            group: Optimizer group with hyperparameters
        """
        if p.grad is None:
            return

        g = p.grad.data

        # Initialize momentum buffer if needed
        if p not in self.buffers:
            self.buffers[p] = torch.zeros(
                p.shape[0], p.shape[1],
                device=p.device,
                dtype=p.dtype
            )

        # Standard Muon momentum update
        # M = momentum * M + g
        self.buffers[p].mul_(group['momentum']).add_(
            g.view(p.shape[0], p.shape[1])
        )

        # ORTHONOISE MODIFICATION
        if self._should_add_noise(p):
            # Generate orthogonal noise via QR decomposition
            Q = self._generate_orthogonal_noise(p)

            # Compute gradient-norm scaled epsilon
            epsilon = self._compute_epsilon(g)

            # Add orthogonal noise: M_tilde = M + epsilon * Q
            self.buffers[p].add_(Q, alpha=epsilon)

            # Re-orthogonalize with Newton-Schulz iterations
            self.buffers[p] = self._reorthogonalize(self.buffers[p])

            self.noise_added_count += 1
        else:
            self.noise_skipped_count += 1

        # Standard SVD-based orthogonal update (from parent Muon)
        try:
            # Convert to float32 for SVD (BFloat16 not supported on CUDA)
            M_float32 = self.buffers[p].float()

            # Compute SVD of momentum buffer
            U, S, Vt = torch.linalg.svd(M_float32, full_matrices=False)

            # Orthogonal update: p -= lr * U @ V^T
            update = U @ Vt

            # Convert back to original dtype and apply
            update = update.to(p.dtype)
            p.data.add_(update, alpha=-group['lr'])

        except RuntimeError as e:
            # SVD can fail on ill-conditioned matrices
            # Fallback to simple gradient descent
            print(f"Warning: SVD failed, using gradient descent fallback: {e}")
            p.data.add_(g, alpha=-group['lr'])

    def get_noise_stats(self) -> Dict[str, int]:
        """
        Get statistics on noise addition.

        Returns:
            Dict with noise_added_count and noise_skipped_count
        """
        return {
            'noise_added_count': self.noise_added_count,
            'noise_skipped_count': self.noise_skipped_count,
        }

    def reset_noise_stats(self):
        """Reset noise statistics counters."""
        self.noise_added_count = 0
        self.noise_skipped_count = 0
