"""
MuonIsotropic Optimizer - Control: Isotropic Gaussian Noise

Adds Gaussian noise to momentum buffer WITHOUT QR decomposition.
This is a control experiment to validate that orthogonality matters.

Core concept: O_tilde = O_t + epsilon * R
- O_t: Current momentum buffer
- R: Gaussian random noise (NOT orthogonalized)
- epsilon: Gradient-norm scaled noise magnitude

If OrthoNoise outperforms MuonIsotropic, it validates that the orthogonal
structure from QR decomposition is important, not just any noise.

Reference: TRANSITION_TO_ORTHONOISE.md (control experiment)
"""

import torch
from torch.optim import Optimizer
from typing import Dict, Any
from .muon import Muon


class MuonIsotropic(Muon):
    """
    Muon optimizer with isotropic Gaussian noise (control).

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-4)
        momentum: Momentum coefficient (default: 0.95)
        alpha: Noise scale hyperparameter (default: 1e-2)
        annealing: Enable noise annealing schedule (default: True)
        adaptive: Enable adaptive triggering based on effective rank (default: True)
        rank_threshold_ratio: Ratio of min(dims) for rank threshold (default: 0.5)
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

        # Isotropic noise parameters (same as OrthoNoise for fair comparison)
        self.alpha = alpha
        self.annealing = annealing
        self.adaptive = adaptive
        self.rank_threshold_ratio = rank_threshold_ratio

        # Tracking for metrics
        self.noise_added_count = 0
        self.noise_skipped_count = 0

    def _generate_isotropic_noise(self, p: torch.Tensor) -> torch.Tensor:
        """
        Generate isotropic Gaussian noise (NOT orthogonalized).

        This is standard Gaussian noise: R ~ N(0, 1)
        Unlike OrthoNoise, we do NOT apply QR decomposition.

        Args:
            p: Parameter tensor (used for shape)

        Returns:
            Gaussian noise matrix R
        """
        # Generate random Gaussian matrix with same shape as momentum buffer
        R = torch.randn_like(self.buffers[p])
        return R

    def _compute_epsilon(self, g: torch.Tensor) -> float:
        """
        Compute noise scale proportional to gradient norm.

        epsilon = alpha * ||g||_F

        With annealing: alpha decays from initial value to 0.1 * alpha over 1000 steps

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
            S_norm = S / (S.sum() + 1e-10)

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
        rank_threshold = self.rank_threshold_ratio * min(p.shape)

        return effective_rank < rank_threshold

    def _muon_step(self, p: torch.Tensor, group: dict) -> None:
        """
        Apply Muon update with isotropic Gaussian noise to matrix parameter.

        Similar to OrthoNoise but WITHOUT orthogonalization:
        1. Standard momentum update: M = momentum * M + g
        2. Add Gaussian noise: M_tilde = M + epsilon * R (R is NOT orthogonalized)
        3. SVD-based update: p -= lr * U @ V^T

        Note: No Newton-Schulz re-orthogonalization since we're not claiming
        orthogonality.

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

        # ISOTROPIC NOISE (control)
        if self._should_add_noise(p):
            # Generate Gaussian noise (NOT orthogonalized)
            R = self._generate_isotropic_noise(p)

            # Compute gradient-norm scaled epsilon
            epsilon = self._compute_epsilon(g)

            # Add isotropic noise: M_tilde = M + epsilon * R
            self.buffers[p].add_(R, alpha=epsilon)

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
