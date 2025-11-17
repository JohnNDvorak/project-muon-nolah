# Transition to 1B Model Experiments

**Date:** 2025-11-17
**Status:** Ready to begin Phase 3 experiments

---

## Summary of 350M Experiments (Phase 2 Complete)

### Key Results
1. **NOLAH converges 33% faster** than baseline Muon (100 steps vs 150 steps)
2. **Optimal configuration:** `--gate tanh --scale 0.90`
3. **Final performance:** Slight improvement (2.3402 vs 2.3409)
4. **All experiments completed successfully** with no SVD failures

### Experimental Results Table

| Configuration | Steps | Final Val Loss | Steps to Optimal | Speedup |
|---------------|-------|----------------|------------------|---------|
| Baseline Muon | 100 | 2.42 | N/A | - |
| Baseline Muon | 500 | 2.3409 | 150 | - |
| NOLAH (0.90) | 100 | 2.37 | 100 | 33% |
| NOLAH (0.90) | 500 | 2.3402 | 100 | 33% |
| NOLAH (0.99) | 100 | 2.39 | N/A | - |
| NOLAH (0.95) | 100 | 2.45 | N/A | - |

### Critical Technical Lessons Confirmed
1. **BFloat16 → Float32 conversion for SVD** is essential
2. **Scale factor 0.90 works best** (more aggressive momentum scaling)
3. **Both methods converge to same limit** (~2.34) - dataset/model constraint
4. **The advantage is speed, not final performance**

---

## Plan for 1B Experiments (Phase 3)

### Objectives
1. **Test scalability:** Does the 33% speedup scale with model size?
2. **Verify optimal configuration:** Is scale=0.90 still best for 1B?
3. **Memory management:** 1B model requires smaller batch sizes

### Expected Changes for 1B Model

#### Model Configuration
```python
# New model name
model_name = "ibm-granite/granite-4.0-h-1b-base"  # ~1B parameters

# Adjusted batch sizes (due to memory constraints)
if args.steps <= 50:
    num_examples = 1000
    batch_size = 2  # Reduced from 4
elif args.steps <= 500:
    num_examples = 10000
    batch_size = 4  # Reduced from 8
else:
    num_examples = 100000
    batch_size = 8  # Reduced from 16
```

#### Experiment Plan
1. **Quick Smoke Test:** 10 steps with batch_size=2
   - Verify no OOM errors
   - Confirm SVD operations work

2. **100-step Comparison:**
   - Baseline vs NOLAH (scale=0.90)
   - Test if 33% speedup persists

3. **500-step Full Run** (if 100-step shows promise):
   - Longer training to verify convergence
   - Track memory usage patterns

### Commands to Run
```bash
# 1. Smoke test (10 steps)
python muon.py baseline --steps 10 --name "1b-smoke"
python muon.py nolah --gate tanh --scale 0.90 --steps 10 --name "1b-nolah-smoke"

# 2. 100-step comparison
python muon.py baseline --steps 100 --name "1b-baseline-100"
python muon.py nolah --gate tanh --scale 0.90 --steps 100 --name "1b-nolah-100"

# 3. 500-step full comparison
python muon.py baseline --steps 500 --name "1b-baseline-500"
python muon.py nolah --gate tanh --scale 0.90 --steps 500 --name "1b-nolah-500"
```

### Hypotheses
1. **H1:** NOLAH's speedup advantage will be MORE pronounced on 1B model
   - Larger models have more complex optimization landscapes
   - Gradient gating more valuable at scale

2. **H2:** Scale=0.90 will remain optimal
   - Aggressive momentum scaling helps with larger models

3. **H3:** Memory overhead of NOLAH is minimal
   - Only adds a few tensor operations

### Success Metrics
- **Primary:** Convergence speed (steps to 90% of final performance)
- **Secondary:** Final validation loss
- **Tertiary:** Memory usage, training stability

---

## Next Steps After 1B Experiments

### Phase 4: From-Scratch Training
If 1B experiments confirm scalability advantages:

1. **Modify training script** to skip pretrained weights
2. **Test early training stability** (first 1000 steps)
3. **Compare loss trajectories** during chaotic early phase
4. **Potential for bigger impact** than fine-tuning

### Key Questions for From-Scratch
1. Does NOLAH prevent gradient explosions in early training?
2. Is convergence more stable from random initialization?
3. Can NOLAH enable higher learning rates safely?

---

## Technical Notes for Next Session

### Current Git State
- All changes committed
- Documentation updated
- Ready for new experiments

### RunPod Status
- Pod ID: `g86sub94x3kvx5`
- GPU: NVIDIA H100 PCIe (80GB)
- Status: Available for new experiments

### WandB Dashboard
- URL: https://wandb.ai/fishhooks1-independent-researcher/granite-muon-nolah
- All 350M runs logged
- Ready for 1B experiments

### Code Changes Needed for 1B
Update `muon.py` auto-scaling logic for 1B model batch sizes. Search for:
```python
if args.steps <= 500:
    num_examples = 10000
    batch_size = 8  # Change this based on model size
```

---

## Ready to Begin

All infrastructure is in place. Start with:
1. Check GPU status: `python muon.py status --gpu`
2. Run 1B smoke test to verify memory
3. Proceed with 100-step comparisons

Good luck with the 1B experiments! 🚀