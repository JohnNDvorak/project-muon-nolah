# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Muon-NOLAH** - Benchmarking NOLAH (Non-Linear Activation Heuristics) optimizer modifications on the Muon optimizer by training IBM Granite 350M on FineWeb-Edu.

**IMPORTANT:** Full implementation plan and progress tracking is in `PROJECT_PLAN.md`. Always read that file first for current status, phase details, and experiment plans.

**Tech Stack:** Python 3.11+ | PyTorch 2.4 | RunPod H100 | WandB | IBM Granite 350M | FineWeb-Edu

**Infrastructure:** RunPod H100 PCIe (80GB) via SSH | Local CLI control | WandB tracking

## Quick Start

```bash
# Check pod status
python muon.py status --gpu

# Run baseline experiment (100 steps)
python muon.py baseline --steps 100

# Run NOLAH experiment
python muon.py nolah --gate tanh --steps 100

# View logs
python muon.py logs --tail 50

# Download results
python muon.py download --run baseline_v4
```

## Critical Technical Lessons

### 1. BFloat16 SVD Incompatibility (MOST CRITICAL)

**Problem:** PyTorch's SVD operation doesn't support BFloat16 on CUDA, causing all Muon updates to fall back to gradient descent.

**Symptom:** Training runs but prints "Warning: SVD failed, using gradient descent fallback"

**Impact:** Optimizer isn't actually running Muon algorithm, just standard gradient descent

**Solution:** Convert tensors to float32 before SVD, then back to original dtype

```python
# CRITICAL FIX in src/optim/muon.py and src/optim/muon_nolah.py
# Before (BROKEN):
# U, S, Vt = torch.linalg.svd(self.buffers[p], full_matrices=False)

# After (CORRECT):
M_float32 = self.buffers[p].float()  # Convert to float32
U, S, Vt = torch.linalg.svd(M_float32, full_matrices=False)
update = U @ Vt
update = update.to(p.dtype)  # Convert back to original dtype
p.data.add_(update, alpha=-group['lr'])
```

**Verification:** Check training logs for "SVD failures: 0" to confirm fix is working

### 2. Memory Management

**Problem:** 350M model + 512 seq length requires careful batch sizing to avoid OOM

**Solution:** Auto-scale batch size and dataset size based on experiment length

```python
# In muon.py - auto-scaling logic
if args.steps <= 50:
    num_examples = 1000  # Small test
    batch_size = 4  # Small batch to fit in memory
elif args.steps <= 500:
    num_examples = 10000  # Medium run
    batch_size = 8  # Medium batch
else:
    num_examples = 100000  # Full run (1% of 10BT)
    batch_size = int(args.batch_size)  # Use command line arg for full runs
```

**Key Insight:** Dataset loading with `load_dataset()` downloads metadata for full dataset even when using `split`. Always limit with `split=f"train[:{num_examples}]"`

### 3. Configuration Injection

**Problem:** Using `json.dumps()` created JSON syntax (`false`) instead of Python syntax (`False`), causing `NameError: name 'false' is not defined`

**Solution:** Use `pprint.pformat()` for proper Python syntax

```python
# In src/utils/runpod_ssh.py
import pprint
config_str = pprint.pformat(config_dict, indent=2, width=120)
modified_script = f"""# Auto-generated configuration
CONFIG = {config_str}

{script_content}
"""
```

## Project Structure

```
project-muon-nolah/
├── muon.py                 # Main CLI interface (USE THIS)
├── PROJECT_PLAN.md         # Project status and experiment plans
├── CLAUDE.md               # This file - technical guidance
├── README.md               # User-facing documentation
├── requirements.txt        # Python dependencies
├── secrets/
│   ├── .env               # API keys and pod configuration (DO NOT COMMIT)
│   └── .env.template      # Template for environment setup
├── src/
│   ├── train.py           # Training script (runs on RunPod)
│   ├── optim/
│   │   ├── muon.py        # Baseline Muon optimizer
│   │   └── muon_nolah.py  # NOLAH-modified Muon
│   └── utils/
│       ├── runpod_ssh.py  # SSH utilities for RunPod
│       └── wandb_setup.py # WandB integration
├── results/               # Experiment outputs (symlink to Google Drive)
└── scripts/               # Utility scripts
```

## File Purposes

### Core Files (Most Frequently Modified)

**`muon.py`** - Unified CLI interface
- All experiment launches go through this file
- Auto-scales batch size and dataset size
- Commands: baseline, nolah, status, logs, download, commit

**`src/optim/muon.py`** - Baseline Muon optimizer
- **CRITICAL:** Must convert to float32 for SVD operations
- SVD-based orthogonal updates for 2D matrix parameters
- AdamW fallback for biases and layer norms

**`src/optim/muon_nolah.py`** - NOLAH modifications
- Inherits from Muon, overrides `_muon_step()`
- Three modifications: gradient gating, momentum scaling, non-linear projection
- **CRITICAL:** Both primary and fallback SVD paths must handle BFloat16 conversion

**`src/train.py`** - Training script executed on RunPod
- Receives injected CONFIG dictionary from SSH launcher
- Uses configurable `num_train_examples` to control dataset size
- WandB integration with offline fallback

**`src/utils/runpod_ssh.py`** - SSH utilities
- Handles script transfer via base64 encoding
- Config injection using pprint.pformat()
- Status monitoring and log retrieval

### Configuration Files

**`secrets/.env`** - Environment variables (DO NOT COMMIT)
```bash
RUNPOD_API_KEY=...
POD_ID=g86sub94x3kvx5
POD_IP=216.81.245.148
SSH_PORT=13147
SSH_KEY_PATH=/Users/johnandmegandvorak/.ssh/id_ed25519_runpod
WANDB_API_KEY=...
WANDB_ENTITY=fishhooks1-independent-researcher
WANDB_PROJECT=granite-muon-nolah
```

## Development Workflow

### 1. Running Experiments

```bash
# Always check pod status first
python muon.py status --gpu

# Small test (10 steps, ~2 min, ~$0.10)
python muon.py baseline --steps 10 --name "test"

# Quick baseline (100 steps, ~15 min, ~$0.70)
python muon.py baseline --steps 100

# Full baseline (500 steps, ~75 min, ~$3.50)
python muon.py baseline --steps 500

# NOLAH experiments
python muon.py nolah --gate tanh --steps 100
python muon.py nolah --gate sigmoid --scale 0.90 --steps 100
```

### 2. Monitoring

```bash
# View recent logs
python muon.py logs --tail 50

# Follow logs in real-time
python muon.py logs --follow

# Check GPU utilization
python muon.py status --gpu
```

### 3. Downloading Results

```bash
# Download specific run
python muon.py download --run baseline_v4

# Skip checkpoints (faster)
python muon.py download --run baseline_v4 --no-checkpoints
```

### 4. Committing Changes

```bash
# Commit code and metadata (not large checkpoints)
python muon.py commit "Completed baseline experiments"

# With push to GitHub
python muon.py commit "Completed baseline experiments" --push
```

## Common Issues and Fixes

### Issue: "SVD failures: X" in training logs

**Cause:** BFloat16 tensors being passed to SVD without conversion

**Fix:** Check src/optim/muon.py and src/optim/muon_nolah.py for proper float32 conversion

**Verify:** Should see "SVD failures: 0" in logs

### Issue: Out of Memory (OOM) errors

**Cause 1:** Batch size too large for model + sequence length
- **Fix:** Reduce batch size via auto-scaling (edit muon.py)

**Cause 2:** Dataset too large (loading into memory)
- **Fix:** Reduce num_train_examples (edit muon.py)

**Cause 3:** Gradient accumulation issues
- **Fix:** Check training script memory management

### Issue: "NameError: name 'false' is not defined"

**Cause:** Config injection using json.dumps() instead of pprint

**Fix:** Already fixed in src/utils/runpod_ssh.py - use pprint.pformat()

### Issue: SSH connection refused

**Cause:** Pod restarted, SSH port changed

**Fix:** Check RunPod dashboard for new port, update secrets/.env

```bash
# Get new connection details from RunPod dashboard
# Update secrets/.env:
POD_IP=<new_ip>
SSH_PORT=<new_port>
```

### Issue: WandB login failed

**Cause:** API key not set or invalid

**Fix:** Check secrets/.env has correct WANDB_API_KEY

**Fallback:** Training continues with offline mode, logs saved locally

## Optimizer Implementation Details

### Muon Optimizer

**Key Concept:** SVD-based orthogonal updates for matrix parameters (weights), AdamW for others (biases, norms)

**Critical Code Pattern:**
```python
def _muon_step(self, p: torch.Tensor, group: dict) -> None:
    g = p.grad.data

    # Update momentum buffer: M = momentum * M + g
    self.buffers[p].mul_(group['momentum']).add_(g.view(p.shape[0], p.shape[1]))

    # CRITICAL: Convert to float32 for SVD
    M_float32 = self.buffers[p].float()

    # SVD decomposition
    U, S, Vt = torch.linalg.svd(M_float32, full_matrices=False)

    # Orthogonal update: p -= lr * U @ V^T
    update = U @ Vt
    update = update.to(p.dtype)  # Convert back
    p.data.add_(update, alpha=-group['lr'])
```

**Parameter Routing:**
- 2D parameters with both dims > 1: Muon (SVD updates)
- All others (biases, 1D norms): AdamW fallback

### NOLAH Modifications

**Three Key Modifications:**

1. **Gradient Gating** - Apply non-linear transformation to gradients
   - Tanh: `torch.tanh(g) * torch.abs(g)` - bounded direction, scaled by magnitude
   - Sigmoid: `torch.sigmoid(g) * g` - smooth gating
   - ReLU: `torch.relu(g)` - simple clipping

2. **Momentum Scaling** - Reduce momentum in high-gradient regions
   - Compute 95th percentile of gradient magnitude
   - Scale momentum by scale_factor (0.90-0.99) where g > percentile

3. **Non-Linear Projection** - Apply activation before SVD
   - `M_projected = M * torch.sigmoid(M)` - activation-aware manifold

**Critical Implementation:**
```python
def _muon_step(self, p: torch.Tensor, group: dict) -> None:
    g = p.grad.data

    # Step 1: Gradient gating
    g_gated = self._apply_gate(g)

    # Update momentum with gated gradients
    self.buffers[p].mul_(group['momentum']).add_(g_gated.view(p.shape[0], p.shape[1]))

    # Step 2: Momentum scaling
    self.buffers[p] = self._scale_momentum(self.buffers[p], g)

    # Step 3: Non-linear projection
    M_projected = self._nolah_projection(self.buffers[p])

    # SVD on projected manifold (MUST convert to float32)
    M_float32 = M_projected.float()
    U, S, Vt = torch.linalg.svd(M_float32, full_matrices=False)
    update = U @ Vt
    update = update.to(p.dtype)
    p.data.add_(update, alpha=-group['lr'])
```

## Experiment Configuration

### Auto-Scaling Logic

| Steps | Examples | Batch Size | Time | Cost |
|-------|----------|------------|------|------|
| ≤50   | 1K       | 4          | ~2 min | ~$0.10 |
| ≤500  | 10K      | 8          | ~15 min | $0.70 |
| >500  | 100K     | CLI arg    | ~75 min | $3.50 |

### Hyperparameters

**Muon Baseline:**
- Learning rate: 1e-4
- Momentum: 0.95
- Warmup steps: 50

**NOLAH Variations:**
- Gate types: tanh (default), sigmoid, relu
- Scale factors: 0.90, 0.95 (default), 0.99

**Model:**
- IBM Granite 350M (340.3M params)
- Sequence length: 512 tokens
- Precision: BFloat16 (model), Float32 (SVD operations)

**Dataset:**
- FineWeb-Edu sample-10BT
- Configurable examples: 1K/10K/100K
- Train/Val split: 95%/5%

## Testing and Validation

### Smoke Test (Always Run Before Full Experiments)

```bash
python muon.py baseline --steps 10 --name "smoke_test"
```

**Expected Output:**
- No OOM errors
- SVD failures: 0
- Training loss decreasing
- WandB logging successful
- Time: ~9-10s/step on H100

### Validation Checklist

- [ ] Check logs for SVD failures (should be 0)
- [ ] Verify batch size matches expectations
- [ ] Confirm dataset examples loaded correctly
- [ ] WandB run appears in dashboard
- [ ] Training loss curve smooth
- [ ] No memory errors

## Environment Setup

### Initial Setup

```bash
# 1. Create secrets/.env from template
cp secrets/.env.template secrets/.env
# Edit secrets/.env with your API keys

# 2. Install local dependencies
pip install -r requirements.txt

# 3. Set up RunPod pod with PyTorch 2.4 template
# Use RunPod dashboard, add SSH public key
# Update secrets/.env with pod details

# 4. Install dependencies on pod (done via CLI)
# Dependencies auto-installed on first training run
```

### RunPod Pod Configuration

**GPU:** NVIDIA H100 PCIe (80GB)
**Template:** RunPod PyTorch 2.4.0
**Storage:** 100GB container + 100GB volume
**Ports:** HTTP 8888 (JupyterLab), SSH auto-configured
**SSH Key:** Add your public key in pod settings

### Google Drive Integration (Optional)

```bash
# Create symlink for results
python muon.py setup --drive-path "~/Google Drive/muon-nolah-results"
```

## Git Workflow

### What to Commit

**Always commit:**
- Source code (src/)
- CLI interface (muon.py)
- Documentation (README.md, PROJECT_PLAN.md, CLAUDE.md)
- Requirements (requirements.txt)
- Metadata (results/**/metadata.json)

**Never commit:**
- API keys (secrets/.env)
- Model checkpoints (results/**/*.pth)
- WandB logs (wandb/)
- Large datasets

### Commit Examples

```bash
# After implementing features
python muon.py commit "feat: add NOLAH sigmoid gate implementation"

# After experiments
python muon.py commit "exp: complete baseline-100 run, loss=3.2"

# After fixes
python muon.py commit "fix: resolve BFloat16 SVD incompatibility"
```

## Cost Tracking

**RunPod H100:** ~$2.80/hour

| Experiment | Steps | Time | Cost |
|-----------|-------|------|------|
| Smoke test | 10 | ~2 min | $0.10 |
| Quick test | 100 | ~15 min | $0.70 |
| Full baseline | 500 | ~75 min | $3.50 |
| NOLAH ablation | 100 | ~15 min | $0.70 |
| Full project | All | ~10 hrs | $25-30 |

**Budget Management:**
- Stop pod when not in use (RunPod dashboard)
- Use smoke tests (10 steps) to validate changes
- Reserve full runs (500 steps) for final comparisons

## WandB Integration

**Project:** https://wandb.ai/fishhooks1-independent-researcher/granite-muon-nolah

**Logged Metrics:**
- train_loss, val_loss (every eval_steps)
- learning_rate (every step)
- grad_norm (every step)
- svd_failures (per epoch)

**Run Naming:**
- Baseline: `granite_baseline_<name>`
- NOLAH: `granite_nolah_<name>`

**Offline Mode:**
- Automatically enabled if login fails
- Logs saved to `wandb/` directory
- Can sync later with `wandb sync`

## Key Technical Concepts

### SVD-Based Orthogonal Updates

**Purpose:** Projects weight updates onto Grassmann manifold, preserving orthogonality

**Mathematics:** Given momentum buffer M, compute U @ V^T from SVD(M) = U @ S @ V^T

**Why float32?** PyTorch SVD only supports float32/float64 on CUDA, not BFloat16

### Gradient Gating

**Purpose:** Stabilize updates in high-gradient regions while preserving direction information

**Tanh gate:** Bounds gradient direction to [-1, 1] but scales by original magnitude

### Momentum Scaling

**Purpose:** Prevent overshooting in high-gradient regions

**Implementation:** Scale momentum by 0.90-0.99 where gradient magnitude > 95th percentile

### Non-Linear Projection

**Purpose:** Project updates through activation-aware manifold before SVD

**Implementation:** Element-wise multiplication with sigmoid activation

## Success Criteria

**Phase 1 (Complete):**
- [x] Infrastructure operational
- [x] Baseline Muon working with SVD
- [x] NOLAH implemented and tested
- [x] 10-step smoke test passing

**Phase 2 (Current):**
- [ ] Baseline-100 completed
- [ ] Baseline-500 completed
- [ ] Loss curves documented

**Phase 3 (Planned):**
- [ ] NOLAH gate ablation
- [ ] NOLAH scale sweep
- [ ] Best config identified

**Phase 4 (Planned):**
- [ ] Comparative analysis
- [ ] Final report

## Resources

**Project Documentation:**
- `PROJECT_PLAN.md` - Complete project status and experiment plans
- `README.md` - User-facing quick start
- WandB Dashboard: https://wandb.ai/fishhooks1-independent-researcher/granite-muon-nolah

**External References:**
- Muon Optimizer Paper: [Link when published]
- IBM Granite Model: https://huggingface.co/ibm-granite/granite-4.0-h-350m-base
- FineWeb-Edu Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

---

**Last Updated:** 2025-11-16
**Current Phase:** Phase 2 - Baseline Experiments
**Status:** Ready to run experiments
