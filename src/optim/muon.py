"""
Muon Optimizer - Matrix Updates On Non-orthogonal Subspaces

Implementation based on the Muon paper.

Matrix parameters (2D): SVD-based orthogonal updates
Other parameters (biases, norms): AdamW fallback
"""

import torch
from torch.optim import Optimizer
from typing import Dict, Any


class Muon(Optimizer):
    """
    Muon optimizer with momentum-based matrix updates.

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-4)
        momentum: Momentum coefficient (default: 0.95)
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
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.01,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

        # Momentum buffers for matrix parameters
        self.buffers: Dict[torch.Tensor, torch.Tensor] = {}

        # AdamW states for non-matrix parameters
        self.adamw_state: Dict[torch.Tensor, Dict[str, Any]] = {}
        self.step_count = 0

    def _is_matrix_param(self, p: torch.Tensor) -> bool:
        """Check if parameter is a matrix (2D with both dims > 1)."""
        return p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1

    def _muon_step(self, p: torch.Tensor, group: dict) -> None:
        """Apply Muon update to matrix parameter."""
        if p.grad is None:
            return

        g = p.grad.data

        # Initialize momentum buffer if needed
        if p not in self.buffers:
            self.buffers[p] = torch.zeros(p.shape[0], p.shape[1], device=p.device, dtype=p.dtype)

        # Update momentum buffer
        # M = momentum * M + g
        self.buffers[p].mul_(group['momentum']).add_(g.view(p.shape[0], p.shape[1]))

        # SVD-based orthogonal update
        try:
            # Compute SVD of momentum buffer
            U, S, Vt = torch.linalg.svd(self.buffers[p], full_matrices=False)

            # Orthogonal update: p -= lr * U @ V^T
            # This projects the update onto the Grassmann manifold
            update = U @ Vt
            p.data.add_(update, alpha=-group['lr'])

        except RuntimeError as e:
            # SVD can fail on ill-conditioned matrices
            # Fallback to simple gradient descent
            print(f"Warning: SVD failed, using gradient descent fallback: {e}")
            p.data.add_(g, alpha=-group['lr'])

    def _adamw_step(self, p: torch.Tensor, group: dict) -> None:
        """Apply AdamW update to non-matrix parameter."""
        if p.grad is None:
            return

        g = p.grad.data

        # Initialize AdamW state if needed
        if p not in self.adamw_state:
            self.adamw_state[p] = {
                'exp_avg': torch.zeros_like(p),
                'exp_avg_sq': torch.zeros_like(p),
            }

        state = self.adamw_state[p]
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['adamw_betas']

        # Weight decay
        p.data.mul_(1 - group['adamw_lr'] * group['adamw_wd'])

        # Update biased first and second moment estimates
        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1 ** self.step_count
        bias_correction2 = 1 - beta2 ** self.step_count

        # Compute step
        step_size = group['adamw_lr'] / bias_correction1
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['adamw_eps'])

        # Update parameters
        p.data.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Route to appropriate optimizer
                if self._is_matrix_param(p):
                    self._muon_step(p, group)
                else:
                    self._adamw_step(p, group)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero out gradients."""
        super().zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Get optimizer state for checkpointing."""
        state = super().state_dict()
        state['buffers'] = {id(k): v for k, v in self.buffers.items()}
        state['adamw_state'] = {id(k): v for k, v in self.adamw_state.items()}
        state['step_count'] = self.step_count
        return state

    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint."""
        super().load_state_dict(state_dict)
        # Note: buffers and adamw_state will be reconstructed on first step
        self.step_count = state_dict.get('step_count', 0)
