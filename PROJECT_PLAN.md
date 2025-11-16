# Project Muon-NOLAH: Implementation Plan

**Status:** Phase 1 Complete ✅ | Ready for Experimentation 🚀

**Last Updated:** 2025-11-16

---

## Overview

Implement and benchmark NOLAH (Non-Linear Activation Heuristics) optimizer modifications on the Muon optimizer, training IBM Granite 350M on FineWeb-Edu.

**Key Goal:** Compare Muon baseline vs. NOLAH-modified Muon to evaluate convergence speed, stability, and final performance.

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

## Phase 2: Baseline Experiments 🔄 IN PROGRESS

**Goal:** Establish solid Muon baseline metrics for comparison

### Planned Experiments

- [ ] **Baseline-100:** 100 steps, 10K examples (~15 min, ~$0.70)
  - Command: `python muon.py baseline --steps 100`
  - Purpose: Quick baseline for NOLAH comparison

- [ ] **Baseline-500:** 500 steps, 10K examples (~75 min, ~$3.50)
  - Command: `python muon.py baseline --steps 500`
  - Purpose: Full baseline run
  - Expected loss: ~2.8-3.2 (based on FineWeb-Edu benchmarks)

### Success Criteria

- Training completes without OOM errors
- Loss curves show smooth convergence
- WandB logs all metrics (train_loss, val_loss, lr)
- Final validation loss documented for comparison

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
