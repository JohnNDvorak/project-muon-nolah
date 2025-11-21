# 1B Granite Experiments - Complete Results

**Date:** 2025-11-18
**Model:** IBM Granite 1B (ibm-granite/granite-4.0-h-1b-base)
**Dataset:** FineWeb-Edu (sample-10BT), 10K examples
**Infrastructure:** RunPod H100 PCIe (80GB), $2.80/hour

## Experimental Configuration

### Model & Training
- **Parameters:** ~1B (3× larger than 350M baseline)
- **Sequence length:** 512 tokens
- **Precision:** BFloat16 (model), Float32 (SVD operations)
- **Batch size:** 4 per GPU (50% reduction from 350M due to memory constraints)
- **Learning rate:** 1e-4
- **Warmup:** 50 steps
- **Max steps:** 500
- **Evaluation interval:** 50 steps

### Optimizer Configurations

**Baseline Muon:**
- Standard SVD-based orthogonal updates
- Momentum: 0.95
- No modifications

**NOLAH (tanh gate, scale=0.90):**
- Gradient gating: `tanh(g) * abs(g)`
- Momentum scaling: 10% reduction where gradient > 95th percentile
- Non-linear projection: `M * sigmoid(M)`

## Results Timeline

### Baseline Muon - 1B

| Step | Val Loss | Perplexity | Notes |
|------|----------|------------|-------|
| 50   | ~2.90    | ~18.2      | Initial warmup |
| 100  | 2.1059   | 8.22       | Rapid convergence |
| 150  | ~2.08    | ~8.0       | Continued improvement |
| 200  | ~2.07    | ~7.9       | Slowing down |
| 250  | 2.0617   | 7.86       | Approaching plateau |
| 300  | 2.0598   | 7.85       | **Best performance** |
| 350  | ~2.06    | ~7.85      | Plateau region |
| 400  | 2.0599   | 7.85       | Stable |
| 450  | ~2.062   | ~7.86      | Final phase |
| 500  | 2.0622   | 7.86       | **FINAL** |

**Key Observations:**
- Smooth, monotonic convergence
- Plateau reached around step 300
- Final loss: 2.0622
- No training instability

### NOLAH (tanh, scale=0.90) - 1B

| Step | Val Loss | Perplexity | vs Baseline | Notes |
|------|----------|------------|-------------|-------|
| 50   | ~3.2     | ~24.5      | -10.3% worse | **Severe early instability** |
| 100  | 2.3034   | 10.01      | -9.4% worse | Still struggling |
| 150  | ~2.15    | ~8.6       | -3.4% worse | Starting recovery |
| 200  | ~2.09    | ~8.1       | -1.0% worse | Catching up |
| 250  | 2.0612   | 7.86       | +0.02% better | **Surpassed baseline** |
| 300  | 2.0574   | 7.83       | +0.12% better | Best advantage |
| 350  | ~2.058   | ~7.83      | +0.10% better | Maintaining edge |
| 400  | 2.0590   | 7.84       | +0.04% better | Slight regression |
| 450  | ~2.060   | ~7.85      | +0.10% better | Final phase |
| 500  | 2.0601   | 7.85       | +0.10% better | **FINAL** |

**Key Observations:**
- **Severe early instability** (steps 0-150): 9.4% worse at step 100
- Recovery phase (steps 150-250): Gradually caught up
- Final advantage: **0.10% better** (2.0601 vs 2.0622)
- Instability suggests scale=0.90 too aggressive for 1B initialization

## Comparative Analysis

### 350M vs 1B Performance

| Metric | 350M Results | 1B Results | Interpretation |
|--------|--------------|------------|----------------|
| **NOLAH at Step 100** | 2.1% better than baseline | 9.4% worse than baseline | ❌ Instability at scale |
| **Final NOLAH Advantage** | 0.03% better | 0.10% better | ⚠️ Within noise margins |
| **Convergence Speed** | 33% faster (100 vs 150 steps) | Same or slower | ❌ Primary benefit disappeared |
| **Early Stability** | Stable from step 0 | Unstable until step 150 | ❌ Poor conditioning |
| **Optimal Configuration** | scale=0.90, gate=tanh | Unclear (early instability) | ⚠️ May need scale=0.95+ |

### Critical Findings

**What Worked:**
1. ✅ NOLAH eventually recovered and surpassed baseline (steps 250-500)
2. ✅ Final performance marginally better (0.10%)
3. ✅ No catastrophic failures or divergence

**What Didn't Work:**
1. ❌ **Early training instability** - 9.4% worse at step 100 is unacceptable for production use
2. ❌ **No convergence speedup** - The 33% faster convergence seen in 350M completely disappeared
3. ❌ **Marginal final gains** - 0.10% improvement is within noise margins and not scientifically significant
4. ❌ **Scale dependence** - Benefits did NOT scale from 350M to 1B

## Statistical Significance

**Final Results:**
- Baseline: 2.0622716999053954
- NOLAH: 2.0601661467552184
- Difference: 0.0021055531501770 (0.102%)

**Assessment:**
- Absolute difference: ~0.002 loss points
- Relative improvement: 0.10%
- **Conclusion:** Likely within noise margins for a single run
- **Recommendation:** Would need 3-5 seeds per configuration to establish statistical significance

## Cost Analysis

**Total Compute:**
- Baseline 500 steps: ~75 minutes × $2.80/hour = **~$3.50**
- NOLAH 500 steps: ~75 minutes × $2.80/hour = **~$3.50**
- Smoke tests + 100-step runs: ~$2.00
- **Total 1B experiments: ~$9.00**

**Cost per 0.1% improvement:**
- Required dual 500-step runs: $7.00
- **$70 per percentage point improvement** (not economically viable)

## Hypotheses for Scale Dependence

### Why did NOLAH's 33% convergence speedup disappear at 1B?

**Hypothesis 1: Gradient Landscape Geometry**
- Larger models have different loss surface curvature
- 350M: NOLAH's aggressive momentum scaling helped escape shallow local minima
- 1B: More complex landscape requires conservative updates during early training

**Hypothesis 2: Initialization Sensitivity**
- Random weight initialization at 1B scale creates larger initial gradients
- scale=0.90 (10% momentum reduction) disrupts critical early feature learning
- Smaller models (350M) more robust to momentum perturbations

**Hypothesis 3: Effective Rank Dynamics**
- NOLAH's non-linear projection (`M * sigmoid(M)`) affects matrix rank
- At 1B scale, more parameters → higher-rank momentum matrices
- Projection may collapse rank too aggressively, losing gradient information

**Hypothesis 4: Warmup Interaction**
- 50-step warmup insufficient for NOLAH at 1B scale
- Baseline Muon benefits from gradual LR scaling during instability
- NOLAH's modifications interfere with warmup stabilization

## Recommendations

### For Continuing NOLAH Research

**If pursuing NOLAH further:**
1. Test scale=0.95, 0.98 to reduce early instability
2. Increase warmup to 100-150 steps for 1B models
3. Run multi-seed experiments (3-5 seeds) to establish significance
4. Consider adaptive scaling: start at 0.99, anneal to 0.90

**Estimated cost:** $30-50 for full ablation

### For Pivoting to Method #1 (OrthoNoise)

**Based on Grok's constructive feedback:**
1. ✅ NOLAH is "ad-hoc momentum tweaking" with marginal, inconsistent gains
2. ✅ No theoretical justification for tanh gating, percentile scaling, sigmoid projection
3. ✅ Method #1 (true orthogonal perturbations via QR decomposition) has geometric rigor
4. ✅ Early training instability suggests initialization problem - orthogonal noise may help

**Recommendation:** Archive NOLAH, pivot to Method #1 implementation

See `TRANSITION_TO_ORTHONOISE.md` for detailed plan.

## WandB Links

**Runs:**
- Baseline: https://wandb.ai/fishhooks1-independent-researcher/granite-muon-nolah/runs/granite_baseline_1b-baseline-500
- NOLAH: https://wandb.ai/fishhooks1-independent-researcher/granite-muon-nolah/runs/granite_nolah_1b-nolah-500

**Project Dashboard:** https://wandb.ai/fishhooks1-independent-researcher/granite-muon-nolah

## Conclusion

**1B experiments completed successfully with mixed results:**

**Positive:**
- NOLAH achieved 0.10% final improvement over baseline
- Recovered from early instability (demonstrates robustness)
- Infrastructure (DDP, RunPod, WandB) working well

**Negative:**
- 9.4% worse performance during early training (steps 0-150)
- 33% convergence speedup from 350M experiments did NOT replicate
- 0.10% final gain is within noise margins and not significant
- Scale dependence suggests NOLAH modifications are not robust

**Next Steps:**
1. Archive NOLAH results with honest assessment
2. Pivot to Method #1 (OrthoNoise) for geometrically principled approach
3. Focus on from-scratch pretraining where initialization stability matters most
4. Budget: $200-300 for 3-phase validation (350M sanity → signal hunt → 1B validation)

---

**Status:** 1B experiments complete - ready to pivot to OrthoNoise
**Date:** 2025-11-18
