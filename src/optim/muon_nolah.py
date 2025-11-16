"""
NOLAH-modified Muon Optimizer - Non-Linear Activation Heuristics

Modifications to Muon:
1. Gradient gating: Apply non-linear transformation to gradients
2. M velocity scaling: Scale momentum by gradient magnitude
3. Non-linear projection: Apply activation-aware manifold projection before SVD
"""

import torch
from .muon import Muon


class MuonNOLAH(Muon):
    """
    NOLAH-modified Muon optimizer.

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-4)
        momentum: Momentum coefficient (default: 0.95)
        gate_type: Gate function type ('tanh', 'sigmoid', 'relu')
        scale_factor: Momentum scaling factor (default: 0.95)
        **kwargs: Additional arguments passed to Muon
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.95,
        gate_type: str = 'tanh',
        scale_factor: float = 0.95,
        **kwargs
    ):
        super().__init__(params, lr=lr, momentum=momentum, **kwargs)

        self.gate_type = gate_type
        self.scale_factor = scale_factor

        # Validate gate type
        if gate_type not in ['tanh', 'sigmoid', 'relu']:
            raise ValueError(f"Invalid gate_type: {gate_type}. Must be 'tanh', 'sigmoid', or 'relu'")

    def _apply_gate(self, g: torch.Tensor) -> torch.Tensor:
        """
        Apply non-linear gating to gradients.

        Stabilizes updates in high-gradient regions while preserving magnitude information.
        """
        if self.gate_type == 'tanh':
            # tanh(g) * |g| - bounded direction, scaled by magnitude
            return torch.tanh(g) * torch.abs(g)

        elif self.gate_type == 'sigmoid':
            # sigmoid(g) * g - smooth gating with original magnitude
            return torch.sigmoid(g) * g

        elif self.gate_type == 'relu':
            # ReLU(g) - simple clipping at zero
            return torch.relu(g)

        return g

    def _scale_momentum(self, M: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Scale momentum buffer by gradient magnitude percentile.

        Reduces momentum in high-gradient regions to prevent overshooting.
        """
        # Compute gradient magnitude percentile
        g_mag = torch.abs(g)
        percentile_95 = torch.quantile(g_mag.flatten(), 0.95)

        # Scale momentum where gradients are large
        scale = torch.where(
            g_mag > percentile_95,
            torch.tensor(self.scale_factor, device=M.device, dtype=M.dtype),
            torch.tensor(1.0, device=M.device, dtype=M.dtype)
        )

        return M * scale

    def _nolah_projection(self, M: torch.Tensor) -> torch.Tensor:
        """
        Apply non-linear projection to momentum buffer before SVD.

        Projects updates through activation-aware manifold.
        """
        # Apply sigmoid activation to create smooth non-linear manifold
        return M * torch.sigmoid(M)

    def _muon_step(self, p: torch.Tensor, group: dict) -> None:
        """Apply NOLAH-modified Muon update to matrix parameter."""
        if p.grad is None:
            return

        g = p.grad.data

        # NOLAH Step 1: Apply gradient gating
        g_gated = self._apply_gate(g)

        # Initialize momentum buffer if needed
        if p not in self.buffers:
            self.buffers[p] = torch.zeros(p.shape[0], p.shape[1], device=p.device, dtype=p.dtype)

        # Update momentum buffer with gated gradients
        self.buffers[p].mul_(group['momentum']).add_(g_gated.view(p.shape[0], p.shape[1]))

        # NOLAH Step 2: Scale momentum by gradient magnitude
        self.buffers[p] = self._scale_momentum(self.buffers[p], g)

        # NOLAH Step 3: Apply non-linear projection
        M_projected = self._nolah_projection(self.buffers[p])

        # SVD-based orthogonal update on projected manifold
        try:
            U, S, Vt = torch.linalg.svd(M_projected, full_matrices=False)
            update = U @ Vt
            p.data.add_(update, alpha=-group['lr'])

        except RuntimeError as e:
            # SVD can fail - fallback to standard Muon
            print(f"Warning: NOLAH SVD failed, using standard Muon: {e}")
            try:
                U, S, Vt = torch.linalg.svd(self.buffers[p], full_matrices=False)
                update = U @ Vt
                p.data.add_(update, alpha=-group['lr'])
            except RuntimeError:
                # If even standard Muon fails, use gradient descent
                print(f"Warning: Standard Muon SVD also failed, using gradient descent")
                p.data.add_(g, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single NOLAH optimization step."""
        # Same structure as Muon, but uses NOLAH-modified _muon_step
        return super().step(closure)
