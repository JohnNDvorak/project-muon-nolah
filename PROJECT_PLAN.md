# Project Muon-NOLAH: Implementation Plan

**Status:** Phase 2 Complete ✅ | Scaling to Larger Models 🚀

**Last Updated:** 2025-11-17

---

## Overview

Implement and benchmark NOLAH (Non-Linear Activation Heuristics) optimizer modifications on the Muon optimizer. Testing on IBM Granite models (350M → 1B → from scratch) on FineWeb-Edu.

**Key Finding:** NOLAH converges **33% faster** than baseline Muon while achieving marginally better final performance.

---

## Phase 1: Foundation & Setup ✅ COMPLETE

**Status:** All systems operational

### Completed Items

✅ **Infrastructure Setup**
- RunPod H100 pod configured and tested
- SSH access working (pod: `g86sub94x3kvx5`, port: `13147`)
- Environment variables configured in `secrets/.env`
- Python dependencies installed on pod

✅ **Project Structure**
- Complete directory hierarchy created
- Git repository initialized with proper `.gitignore`
- CLI interface (`muon.py`) with all commands working
- Utilities for SSH, WandB, and RunPod management

✅ **Optimizer Implementation**
- Baseline Muon optimizer with SVD-based updates
- NOLAH optimizer with gradient gating, momentum scaling, and non-linear projection
- **Critical Fix:** BFloat16 → Float32 conversion for CUDA SVD compatibility

✅ **Training Pipeline**
- Complete training script with WandB integration
- Auto-scaling batch size (4/8/16) based on experiment size
- Auto-scaling dataset size (1K/10K/100K examples) based on steps
- Validation loop and checkpointing

✅ **Testing & Validation**
- 10-step smoke test completed successfully
- SVD updates verified (0 fallbacks to gradient descent)
- WandB logging confirmed: https://wandb.ai/fishhooks1-independent-researcher/granite-muon-nolah
- Training metrics: ~9.5s/step on H100 with batch_size=4

### Key Lessons Learned

1. **BFloat16 SVD Issue:** PyTorch SVD doesn't support BFloat16 on CUDA
   - Solution: Convert to float32 for SVD, then back to original dtype

2. **Memory Management:** 350M model + 512 seq length requires careful batch sizing
   - Small tests (≤50 steps): batch_size=4
   - Medium runs (≤500 steps): batch_size=8
   - Full runs: batch_size=16

3. **Dataset Loading:** HuggingFace `load_dataset` downloads metadata for full dataset
   - Use `split=f"train[:{num_examples}]"` to limit examples
   - 1K examples sufficient for 10-step tests

4. **Config Injection:** Use `pprint.pformat()` instead of `json.dumps()` for Python syntax
   - Prevents `false` vs `False` errors

---

## Phase 2: 350M Fine-tuning Experiments ✅ COMPLETE

**Goal:** Compare Muon baseline vs. NOLAH on Granite 350M

### Completed Experiments

✅ **100-step Ablation Tests**
- Baseline: 2.42 final validation loss
- NOLAH scale=0.90: 2.37 ✅ (best)
- NOLAH scale=0.99: 2.39
- NOLAH scale=0.95: 2.45 (worst)

✅ **500-step Full Comparison**
- Baseline: 2.3409 final validation loss (plateau at step 150)
- NOLAH scale=0.90: 2.3402 final validation loss ✅ (slightly better)

### Key Findings

1. **Faster Convergence:** NOLAH reaches optimal performance in 100 steps vs 150 steps for baseline (33% improvement)
2. **Marginal Final Performance Gain:** NOLAH achieves 0.0007 better validation loss
3. **Optimal Scale Factor:** 0.90 performs best, 0.99 second-best
4. **Convergence Limit:** Both methods converge to ~2.34, suggesting dataset/model limit

---

## Phase 3: Scale-up Experiments 🔄 IN PROGRESS

**Goal:** Test if NOLAH advantages scale with model size

### Planned Experiments

- [ ] **Granite 1B Fine-tuning**
  - Model: `ibm-granite/granite-4.0-h-1b-base` (~1B parameters)
  - Test: Does 33% convergence speedup scale?
  - Expected: More pronounced benefits with larger optimization landscape

### Hypothesis

Larger models have more complex loss landscapes with more critical points. NOLAH's gradient gating and momentum scaling should provide even greater benefits at scale.

---

## Phase 4: From-scratch Training 📋 PLANNED

**Goal:** Test NOLAH for training stability from random initialization

### Planned Experiments

- [ ] **Train from Scratch (350M)**
  - Start from random initialization (not pretrained)
  - Test early training stability (first 1000 steps)
  - Metric: Loss variance, gradient explosions, convergence stability

- [ ] **Train from Scratch (1B)**
  - Same test on larger model if 350M shows promise

### Why This Matters

- Real-world training often starts from scratch
- Early training phase is most chaotic
- NOLAH's gradient gating could prevent catastrophic updates

---

## Phase 3: NOLAH Experiments 📋 PLANNED

**Goal:** Compare NOLAH modifications against baseline

### Experiment Matrix

#### Gate Type Ablation
- [ ] NOLAH-tanh-100 (100 steps)
- [ ] NOLAH-sigmoid-100 (100 steps)
- [ ] NOLAH-relu-100 (100 steps)

#### Scale Factor Sweep
- [ ] NOLAH-scale-0.90 (100 steps)
- [ ] NOLAH-scale-0.95 (100 steps, default)
- [ ] NOLAH-scale-0.99 (100 steps)

#### Full Comparison Runs
- [ ] Baseline-500 (established in Phase 2)
- [ ] NOLAH-best-config-500 (best config from ablations)

### Metrics to Track

- **Convergence Speed:** Steps to reach baseline's final loss
- **Stability:** Loss variance across training
- **Final Performance:** Validation loss and perplexity
- **Gradient Norms:** Track update magnitudes
- **Training Efficiency:** Time per step

---

## Phase 4: Analysis & Reporting 📊 PLANNED

**Goal:** Comprehensive comparison and documentation

### Analysis Tasks

- [ ] Download all checkpoints
- [ ] Generate loss curve visualizations
- [ ] Compute validation perplexity for all runs
- [ ] Sample text generation comparison
- [ ] Statistical significance testing
- [ ] Document findings in final report

### Deliverables

- [ ] WandB dashboard with all runs
- [ ] Comparative analysis document
- [ ] Sample outputs from baseline vs. NOLAH
- [ ] Recommendations for NOLAH usage

---

## Cost Estimates

| Experiment Type | Steps | Time | GPU Cost |
|----------------|-------|------|----------|
| Smoke test | 10 | ~2 min | $0.10 |
| Quick baseline | 100 | ~15 min | $0.70 |
| Full baseline | 500 | ~75 min | $3.50 |
| NOLAH ablation (each) | 100 | ~15 min | $0.70 |
| Full NOLAH | 500 | ~75 min | $3.50 |

**Total Estimated Cost for Full Project:** ~$25-30

*Based on RunPod H100 at ~$2.80/hour*

---

## Technical Specifications

### Model
- **Name:** `ibm-granite/granite-4.0-h-350m-base`
- **Parameters:** 340.3M
- **Precision:** BFloat16 (model), Float32 (SVD operations)

### Dataset
- **Source:** `HuggingFaceFW/fineweb-edu` (sample-10BT)
- **Sequence Length:** 512 tokens
- **Train/Val Split:** 95% / 5%

### Optimizer Config
- **Learning Rate:** 1e-4
- **Momentum:** 0.95
- **Warmup Steps:** 50
- **AdamW Fallback:** lr=3e-4, betas=(0.9, 0.95)

### NOLAH Config
- **Gate Types:** tanh, sigmoid, relu
- **Scale Factor:** 0.90, 0.95, 0.99
- **Projection:** sigmoid activation on momentum buffer

---

## Commands Reference

```bash
# Baseline experiments
python muon.py baseline --steps 100
python muon.py baseline --steps 500

# NOLAH experiments
python muon.py nolah --gate tanh --steps 100
python muon.py nolah --gate sigmoid --steps 100
python muon.py nolah --scale 0.90 --steps 100

# Monitoring
python muon.py status --gpu
python muon.py logs --tail 50

# Download results
python muon.py download --run baseline_v4

# Commit results
python muon.py commit "Completed baseline experiments"
```

---

## Current Status

**Infrastructure:** ✅ Fully operational
**Baseline Code:** ✅ Tested and working
**NOLAH Code:** ✅ Implemented, ready to test
**Next Action:** Run Phase 2 baseline experiments

**Pod Status:**
- ID: `g86sub94x3kvx5`
- IP: `216.81.245.148:13147`
- GPU: NVIDIA H100 PCIe (80GB)
- Status: Running

**WandB Project:** https://wandb.ai/fishhooks1-independent-researcher/granite-muon-nolah
