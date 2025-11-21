# Transition to OrthoNoise (Method #1)

**Status:** Ready to pivot from NOLAH to geometrically principled orthogonal perturbations
**Date:** 2025-11-18
**Reason:** NOLAH showed marginal, inconsistent gains; Grok's constructive feedback provides rigorous alternative

## Why Pivot from NOLAH?

### Honest Assessment of NOLAH

**What NOLAH Actually Is:**
- Ad-hoc momentum tweaking with three heuristic modifications
- Gradient gating: `tanh(g) * abs(g)` - arbitrary choice, no geometric justification
- Percentile scaling: Reduce momentum by 10% where gradient > 95th percentile - magic numbers
- Sigmoid projection: `M * sigmoid(M)` - breaks Muon's orthogonal update invariants

**Results Summary:**
- 350M: 0.03% improvement, 33% faster convergence ✅
- 1B: 0.10% improvement, 9.4% worse early training, no convergence speedup ❌
- **Conclusion:** Marginal, scale-dependent gains within noise margins

### Grok's Critical Feedback (Accepted as Valid)

**Key Criticisms:**
1. "Orthogonal perturbations" claim was misleading - not actually orthogonal
2. Three modifications are heuristics without theoretical foundation
3. 0.05-0.12% improvements are not statistically significant
4. Empty WandB and no public code undermines scientific validity

**Constructive Alternative (Method #1):**
TRUE orthogonal perturbations via QR decomposition with geometric rigor

## Method #1: OrthoNoise - Low-Magnitude Orthogonal Noise

### Core Concept

**Add orthogonal noise to momentum buffer before SVD:**

```
O_tilde = O_t + epsilon * Q
```

Where:
- `O_t` = Current momentum buffer (Muon's M matrix)
- `Q` = Orthogonal noise matrix from QR decomposition
- `epsilon` = Noise scale proportional to gradient norm

### Mathematical Foundation

**QR Decomposition for Orthogonal Noise:**
```python
# Generate random Gaussian matrix
R ~ N(0, 1) with shape matching momentum buffer

# QR decomposition
Q, _ = torch.linalg.qr(R)

# Q is orthonormal: Q^T @ Q = I
```

**Noise Scaling (Adaptive):**
```python
epsilon = alpha * ||g||_F
```
- `alpha` = Hyperparameter (start at 1e-2)
- `||g||_F` = Frobenius norm of current gradient
- Makes noise magnitude proportional to optimization progress

**Re-orthogonalization (Newton-Schulz):**
```python
# After adding noise, restore orthogonality
# Newton-Schulz iterations: 3 steps sufficient
for _ in range(3):
    O_tilde = 1.5 * O_tilde - 0.5 * O_tilde @ O_tilde.T @ O_tilde
```

### Implementation Pseudocode

```python
class MuonOrthoNoise(Muon):
    def __init__(self, params, lr=1e-4, momentum=0.95,
                 alpha=1e-2, annealing=True, adaptive=True):
        super().__init__(params, lr, momentum)
        self.alpha = alpha
        self.annealing = annealing
        self.adaptive = adaptive
        self.step_count = 0

    def _muon_step(self, p: torch.Tensor, group: dict) -> None:
        g = p.grad.data

        # Standard Muon momentum update
        self.buffers[p].mul_(group['momentum']).add_(
            g.view(p.shape[0], p.shape[1])
        )

        # ORTHONOISE MODIFICATION
        if self._should_add_noise(p):
            noise = self._generate_orthogonal_noise(p)
            epsilon = self._compute_epsilon(g)
            self.buffers[p].add_(noise, alpha=epsilon)
            self.buffers[p] = self._reorthogonalize(self.buffers[p])

        # Standard SVD update (CRITICAL: float32 conversion)
        M_float32 = self.buffers[p].float()
        U, S, Vt = torch.linalg.svd(M_float32, full_matrices=False)
        update = U @ Vt
        update = update.to(p.dtype)
        p.data.add_(update, alpha=-group['lr'])

        self.step_count += 1

    def _generate_orthogonal_noise(self, p: torch.Tensor) -> torch.Tensor:
        """Generate orthogonal noise via QR decomposition."""
        R = torch.randn_like(self.buffers[p])
        Q, _ = torch.linalg.qr(R)
        return Q

    def _compute_epsilon(self, g: torch.Tensor) -> float:
        """Compute noise scale proportional to gradient norm."""
        grad_norm = torch.norm(g, p='fro')

        # Annealing: alpha from 1e-2 -> 1e-3 over 1000 steps
        if self.annealing:
            alpha = self.alpha * (0.1 ** (self.step_count / 1000))
        else:
            alpha = self.alpha

        return alpha * grad_norm.item()

    def _should_add_noise(self, p: torch.Tensor) -> bool:
        """Adaptive triggering based on effective rank."""
        if not self.adaptive:
            return True  # Always add noise

        # Compute effective rank via SVD
        M_float32 = self.buffers[p].float()
        _, S, _ = torch.linalg.svd(M_float32, full_matrices=False)

        # Effective rank: exp(entropy of normalized singular values)
        S_norm = S / S.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-8)).sum()
        effective_rank = torch.exp(entropy).item()

        # Trigger noise if rank is low (momentum buffer poorly conditioned)
        rank_threshold = 0.5 * min(p.shape)  # 50% of max possible rank
        return effective_rank < rank_threshold

    def _reorthogonalize(self, M: torch.Tensor, iterations: int = 3) -> torch.Tensor:
        """Newton-Schulz iterations to restore orthogonality."""
        M_orth = M
        for _ in range(iterations):
            M_orth = 1.5 * M_orth - 0.5 * M_orth @ M_orth.T @ M_orth
        return M_orth
```

### Why This is Better Than NOLAH

**Geometric Rigor:**
- QR decomposition GUARANTEES orthogonality (Q^T @ Q = I)
- Newton-Schulz provably converges to nearest orthogonal matrix
- Gradient-norm scaling has principled interpretation (noise ~ optimization progress)

**Testable Hypotheses:**
- H1: Orthogonal noise improves conditioning of momentum buffer
- H2: Improves effective rank → better gradient flow
- H3: Helps escape flat regions (true saddle point escape, not just momentum tweaking)
- H4: Benefits most pronounced during early training (initialization phase)

**Ablation Clarity:**
- Baseline vs OrthoNoise (clean comparison)
- OrthoNoise vs Isotropic Noise (tests orthogonality importance)
- With/without annealing (tests noise schedule importance)
- With/without adaptive triggering (tests rank-based gating)

## 3-Phase Validation Plan

### Phase 1: Code & Sanity (~$30-50, 10-15 hours)

**Goal:** Implement OrthoNoise and verify it doesn't break training

**Tasks:**
1. Create `src/optim/muon_orthonoise.py` with implementation above
2. Update `src/train_ddp.py` to support OrthoNoise optimizer
3. Add CLI command: `python muon.py pretrain --orthonoise --steps 500`

**Experiments (350M model, 500 steps each):**
- Baseline Muon × 2 seeds
- OrthoNoise alpha=1e-3 × 2 seeds
- Isotropic noise control × 2 seeds (Gaussian noise without QR)
- OrthoNoise with annealing × 2 seeds

**Success Criteria:**
- OrthoNoise trains without crashes
- Loss curves comparable or better than baseline
- Effective rank metrics logged and sensible
- No obvious pathologies (NaN, explosion, etc.)

**Cost:** 8 runs × 500 steps × ~$3.50 = **~$28**

### Phase 2: Signal Hunt (~$80-120, 25-40 hours)

**Goal:** Find hyperparameter regime where OrthoNoise shows clear advantage

**Experiments (350M model):**

**Alpha sweep (5 values × 2 seeds):**
- alpha = 1e-4, 1e-3, 5e-3, 1e-2, 5e-2
- 500 steps each
- Both with and without annealing
- **Cost:** 20 runs × $3.50 = $70

**Annealing schedule sweep (3 schedules × 2 seeds):**
- Constant alpha
- Linear decay (1e-2 → 1e-3)
- Exponential decay (1e-2 → 1e-3, half-life = 500 steps)
- **Cost:** 6 runs × $3.50 = $21

**Adaptive triggering (2 thresholds × 2 seeds):**
- rank_threshold = 0.3 * min(dims)
- rank_threshold = 0.7 * min(dims)
- **Cost:** 4 runs × $3.50 = $14

**Total Phase 2:** ~$105

**Success Criteria:**
- Find alpha where OrthoNoise beats baseline by >1% (statistically significant)
- Effective rank improves measurably
- Early training stability better than baseline

### Phase 3: 1B Validation (~$80-120, 25-35 hours)

**Goal:** Validate best configuration scales to 1B

**Experiments (1B model, best alpha from Phase 2):**
- Baseline Muon × 3 seeds × 500 steps
- OrthoNoise (best config) × 3 seeds × 500 steps
- **Cost:** 6 runs × $3.50 = $21

**From-scratch pretraining (4× H100 DDP, 5000 steps):**
- Baseline Muon × 2 seeds
- OrthoNoise × 2 seeds
- **Cost:** 4 runs × 13 hours × $11.20/hour = **$582** ⚠️ (over budget)

**Revised Phase 3 (budget-conscious):**
- 1B fine-tuning: 6 runs × $3.50 = $21
- From-scratch 1B: 2 runs × 1000 steps × $2.80/hour × 2.5 hours = **$14**
- **Total Phase 3:** ~$35

**Success Criteria:**
- OrthoNoise advantage replicates at 1B scale
- Early training stability significantly better than NOLAH
- From-scratch training shows clear benefit (validates initialization hypothesis)

### Budget Summary

| Phase | Experiments | Cost |
|-------|-------------|------|
| Phase 1: Sanity | 8 runs × 500 steps | $28 |
| Phase 2: Signal Hunt | 30 runs × 500 steps | $105 |
| Phase 3: 1B Validation | 8 runs (mixed) | $35 |
| **Total** | **46 runs** | **$168** |

**Contingency:** $50 for re-runs if needed
**Grand Total:** **$218** (within $200-300 budget)

## File Structure Changes

### Archive NOLAH Work

```
archive/
├── muon_nolah.py           # Move from src/optim/
├── NOLAH_RESULTS.md        # Summary of 350M + 1B experiments
└── nolah_ablation_v1.py    # Original ablation experiments
```

### New OrthoNoise Implementation

```
src/optim/
├── muon.py                 # Baseline (unchanged)
├── muon_orthonoise.py      # NEW: Method #1 implementation
└── muon_isotropic.py       # NEW: Control (Gaussian noise, no QR)
```

### Documentation Updates

```
docs/
├── ORTHONOISE_THEORY.md    # Mathematical derivation
├── ABLATION_RESULTS.md     # Phase 1-3 results
└── PIVOT_RATIONALE.md      # Why we pivoted from NOLAH
```

## CLI Commands (After Implementation)

### Phase 1: Sanity Checks

```bash
# Baseline runs (2 seeds)
python muon.py pretrain --steps 500 --seed 42 --name "baseline-seed42"
python muon.py pretrain --steps 500 --seed 123 --name "baseline-seed123"

# OrthoNoise runs
python muon.py pretrain --orthonoise --alpha 1e-3 --steps 500 --seed 42 --name "orthonoise-1e3-seed42"
python muon.py pretrain --orthonoise --alpha 1e-3 --steps 500 --seed 123 --name "orthonoise-1e3-seed123"

# Isotropic control (no QR, just Gaussian noise)
python muon.py pretrain --isotropic --alpha 1e-3 --steps 500 --seed 42 --name "isotropic-1e3-seed42"
python muon.py pretrain --isotropic --alpha 1e-3 --steps 500 --seed 123 --name "isotropic-1e3-seed123"

# With annealing
python muon.py pretrain --orthonoise --alpha 1e-2 --anneal --steps 500 --seed 42 --name "orthonoise-anneal-seed42"
python muon.py pretrain --orthonoise --alpha 1e-2 --anneal --steps 500 --seed 123 --name "orthonoise-anneal-seed123"
```

### Phase 2: Alpha Sweep

```bash
# Alpha sweep (use script to automate)
for alpha in 1e-4 1e-3 5e-3 1e-2 5e-2; do
  for seed in 42 123; do
    python muon.py pretrain --orthonoise --alpha $alpha --steps 500 --seed $seed --name "alpha-${alpha}-seed${seed}"
  done
done
```

### Phase 3: 1B Validation

```bash
# Update .env to use 1B model
# MODEL_NAME=ibm-granite/granite-4.0-h-1b-base

# Best configuration from Phase 2 (example: alpha=5e-3 with annealing)
python muon.py pretrain --orthonoise --alpha 5e-3 --anneal --steps 500 --seed 42 --name "1b-orthonoise-best-seed42"

# From-scratch DDP (4× H100)
python muon.py pretrain --orthonoise --alpha 5e-3 --anneal --steps 1000 --ngpus 4 --seed 42 --name "1b-scratch-orthonoise"
```

## Implementation Checklist

### Week 1: Phase 1 Implementation

- [ ] Create `src/optim/muon_orthonoise.py`
- [ ] Implement QR noise generation
- [ ] Implement Newton-Schulz re-orthogonalization
- [ ] Implement gradient-norm scaling
- [ ] Implement annealing schedule
- [ ] Implement adaptive triggering (effective rank)
- [ ] Add effective rank logging to WandB
- [ ] Update `src/train_ddp.py` to support `--orthonoise` flag
- [ ] Update `muon.py` CLI with OrthoNoise options
- [ ] Create isotropic control optimizer
- [ ] Run 8 sanity check experiments
- [ ] Analyze Phase 1 results

### Week 2: Phase 2 Signal Hunt

- [ ] Create automation script for alpha sweep
- [ ] Run 20 alpha sweep experiments
- [ ] Run 6 annealing schedule experiments
- [ ] Run 4 adaptive triggering experiments
- [ ] Analyze all Phase 2 results
- [ ] Identify best configuration
- [ ] Document findings in `ABLATION_RESULTS.md`

### Week 3: Phase 3 Validation

- [ ] Run 6 × 1B fine-tuning experiments
- [ ] Run 2 × 1B from-scratch experiments (1000 steps)
- [ ] Compare to NOLAH 1B results
- [ ] Analyze statistical significance (3 seeds)
- [ ] Create final results visualization
- [ ] Update PROJECT_PLAN.md with conclusions

## Success Metrics

### Primary Metrics
- **Validation loss improvement:** >1% better than baseline (statistically significant with 2-3 seeds)
- **Convergence speed:** Steps to reach baseline's final loss
- **Early training stability:** Loss at step 100 comparable or better than baseline

### Secondary Metrics (Diagnostic)
- **Effective rank:** Should increase when noise is added
- **Gradient norm:** Should remain stable (no explosion)
- **Update norm:** Monitor `||U @ V^T||_F` for pathologies

### Success Criteria by Phase

**Phase 1 Pass:**
- No training crashes
- Loss curves reasonable (within 5% of baseline)
- Effective rank logging works

**Phase 2 Pass:**
- At least one alpha configuration beats baseline by >1%
- Effective rank measurably higher with OrthoNoise
- Isotropic control underperforms OrthoNoise (validates orthogonality)

**Phase 3 Pass:**
- 1B advantage replicates (>0.5% improvement)
- From-scratch training stable and faster than baseline
- Results publishable (clear signal, proper ablations, multi-seed validation)

## Next Session Action Items

1. **Review this document** - Ensure you understand the pivot rationale
2. **Read 1B_RESULTS.md** - Context on NOLAH limitations
3. **Read PROJECT_PLAN.md** - Updated status and phase structure
4. **Implement `muon_orthonoise.py`** - Start with Phase 1
5. **Update CLI** - Add `--orthonoise`, `--isotropic`, `--alpha`, `--anneal` flags
6. **Run Phase 1 sanity checks** - 8 experiments, ~$28

## Open Questions to Resolve

1. **QR on BFloat16?** - May need float32 conversion like SVD
2. **Newton-Schulz iterations:** 3 sufficient or need 5?
3. **Rank threshold:** 0.5 × min(dims) good default or should sweep?
4. **Noise schedule:** Start adding noise immediately or after warmup?
5. **Multi-GPU:** Does QR noise need to be synchronized across ranks?

## References

**Grok's Method #1 (verbatim):**
```
Let O_t be your momentum buffer. Instead of messing with the update itself:
O_tilde = O_t + epsilon * Q

Where Q is sampled from the Grassmannian (use QR decomp on Gaussian noise),
epsilon is scaled by ||grad||. After, re-orthogonalize O_tilde with
Newton-Schulz iterations before feeding to SVD.

Why it works: You're literally adding structured noise in the tangent space,
then projecting back. Gradient norm scaling ensures you're not being chaotic,
but you ARE exploring nearby orthogonal frames.
```

**Math Background:**
- Grassmannian manifold: Space of k-dimensional linear subspaces
- QR decomposition: Gram-Schmidt orthogonalization (stable numerically)
- Newton-Schulz: Iterative method for matrix square root (faster than SVD for re-orthogonalization)

---

**Status:** Ready to implement OrthoNoise
**Next Steps:** Implement Phase 1, run sanity checks
**Budget:** $168 for 3 phases (+$50 contingency)
**Timeline:** 3 weeks (1 week per phase)
