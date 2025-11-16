# Project Muon-NOLAH

**Benchmark NOLAH optimizer modifications on the Muon optimizer**

Train IBM Granite 350M on FineWeb-Edu and compare baseline Muon vs. NOLAH-modified Muon.

## Quick Start

### 1. Setup

```bash
# Clone and navigate
cd ~/Documents/github/project-muon-nolah

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp secrets/.env.template secrets/.env
# Edit secrets/.env with your API keys

# Setup Google Drive sync (optional)
python muon.py setup --drive-path "~/Google Drive/muon-nolah-results"
```

### 2. Run Experiments

```bash
# Baseline Muon (500 steps)
python muon.py baseline --steps 500

# NOLAH with tanh gating (100 steps, faster iteration)
python muon.py nolah --gate tanh --steps 100

# Monitor training
python muon.py status --gpu
python muon.py logs --tail 50
```

### 3. Download Results

```bash
# Download results (metadata + checkpoints)
python muon.py download --run baseline_v4

# Download metadata only (skip large .pth files)
python muon.py download --run nolah_tanh_v1 --no-checkpoints
```

### 4. Commit to Git

```bash
# Commit code and metadata (not large checkpoints)
python muon.py commit "Completed baseline run - val loss 2.89"
```

---

## Architecture

### Optimizers

**Baseline Muon** (`src/optim/muon.py`)
- Matrix parameters (2D): SVD-based orthogonal updates
- Other parameters: AdamW fallback
- Momentum: 0.95

**NOLAH-modified Muon** (`src/optim/muon_nolah.py`)
- **Gradient gating**: `g_gated = tanh(g) * |g|`
- **Momentum scaling**: Scale by gradient magnitude percentile
- **Non-linear projection**: `M * sigmoid(M)` before SVD

### Storage Strategy

```
project-muon-nolah/
├── src/                    → Git tracked
├── config/                 → Git tracked
├── results/                → Symlinked to Google Drive
│   ├── baseline_v4/
│   │   ├── metadata.json   → Git tracked (small)
│   │   └── model.pth       → Drive only (large, gitignored)
│   └── nolah_v1/
└── secrets/.env            → Local only (gitignored)
```

**Benefits**:
- Code versioned in Git
- Large checkpoints auto-synced to Drive
- No large files in Git history

---

## CLI Reference

### Training Commands

```bash
# Baseline
python muon.py baseline \
  --steps 500 \
  --lr 0.0001 \
  --batch-size 16 \
  --name "my_baseline_run"

# NOLAH
python muon.py nolah \
  --gate tanh \           # tanh | sigmoid | relu
  --scale 0.95 \          # Momentum scale factor
  --steps 100 \
  --name "tanh_experiment"
```

### Monitoring

```bash
# Check if training is running
python muon.py status

# GPU utilization
python muon.py status --gpu

# View logs
python muon.py logs --tail 100

# Follow logs in real-time
python muon.py logs --follow
```

### Data Management

```bash
# Download everything
python muon.py download --run baseline_v4

# Skip checkpoints (faster, for metadata only)
python muon.py download --run baseline_v4 --no-checkpoints
```

### Git Integration

```bash
# Commit code + metadata
python muon.py commit "Add NOLAH gating experiments"

# Commit and push
python muon.py commit "Baseline complete" --push
```

---

## Configuration

### Environment Variables (`secrets/.env`)

```bash
# RunPod
RUNPOD_API_KEY=your_key
POD_ID=your_pod_id
POD_IP=62.169.159.96
SSH_PORT=34538
SSH_KEY_PATH=~/.ssh/id_ed25519_runpod

# WandB
WANDB_API_KEY=your_key
WANDB_ENTITY=fishhooks1-independent-researcher
WANDB_PROJECT=granite-muon-nolah

# Model/Dataset
MODEL_NAME=ibm-granite/granite-4.0-h-350m-base
DATASET_NAME=HuggingFaceFW/fineweb-edu
DATASET_CONFIG=sample-10BT
```

### SSH Key Setup (One-time)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_runpod -N ""

# Copy public key
cat ~/.ssh/id_ed25519_runpod.pub

# Add to RunPod:
# 1. Go to RunPod pod page
# 2. Edit Pod → Advanced Settings → SSH Public Keys
# 3. Paste public key
# 4. Restart pod
```

---

## Expected Results

### Baseline (500 steps on 1% FineWeb-Edu)
- **Train Loss**: 2.8-3.2
- **Val Loss**: 3.0-3.4
- **Runtime**: ~60 minutes on H100
- **Cost**: ~$2.80

### NOLAH Goals
- Faster convergence (fewer steps to same loss)
- Better stability (lower loss variance)
- Improved final perplexity

---

## Troubleshooting

### "Permission denied (publickey)"
SSH key not added to RunPod or pod not restarted.

```bash
# Test SSH connection
ssh -i ~/.ssh/id_ed25519_runpod -p 34538 root@YOUR_POD_IP "echo success"
```

### "WandB timeout"
Network issues on RunPod. Training continues in offline mode.

```bash
# Check if training is still running
python muon.py status

# Sync offline run later
wandb sync /workspace/wandb/offline-run-xxx
```

### "Training not running"
Process crashed. Check logs for errors.

```bash
python muon.py logs --tail 100
```

---

## Development Workflow

### Adding New Features

1. **Create branch**
   ```bash
   git checkout -b feature/nolah-relu-gating
   ```

2. **Modify code** (e.g., add new gate type)
   ```bash
   # Edit src/optim/muon_nolah.py
   ```

3. **Test locally** (dry run to verify config)
   ```bash
   python muon.py nolah --gate relu --steps 10 --dry-run
   ```

4. **Run on RunPod**
   ```bash
   python muon.py nolah --gate relu --steps 100
   ```

5. **Download and analyze**
   ```bash
   python muon.py download --run nolah_relu_v1
   ```

6. **Commit results**
   ```bash
   python muon.py commit "Add ReLU gating experiment" --push
   ```

### Ablation Experiments

```bash
# Gate type sweep
python muon.py nolah --gate tanh --steps 100 --name "gate_tanh"
python muon.py nolah --gate sigmoid --steps 100 --name "gate_sigmoid"
python muon.py nolah --gate relu --steps 100 --name "gate_relu"

# Scale factor sweep
python muon.py nolah --scale 0.90 --steps 100 --name "scale_90"
python muon.py nolah --scale 0.95 --steps 100 --name "scale_95"
python muon.py nolah --scale 0.99 --steps 100 --name "scale_99"
```

---

## WandB Dashboard

All experiments are logged to:
https://wandb.ai/fishhooks1-independent-researcher/granite-muon-nolah

**Key Metrics**:
- `train_loss`: Training loss per step
- `val_loss`: Validation loss (every 50 steps)
- `lr`: Learning rate with warmup

---

## Cost Estimates

| Configuration | Steps | Runtime | Cost (H100) |
|--------------|-------|---------|-------------|
| Quick test   | 50    | ~6 min  | $0.28       |
| Short run    | 100   | ~12 min | $0.56       |
| Standard     | 500   | ~60 min | $2.80       |
| Full         | 1000  | ~2 hrs  | $5.60       |

*Based on RunPod H100 at ~$2.80/hour*

---

## Project Status

✅ **Phase 1: Baseline** - COMPLETE
- [x] Infrastructure setup
- [x] SSH execution pipeline
- [x] WandB integration
- [x] Baseline Muon implementation
- [x] Training script tested

🔬 **Phase 2: NOLAH** - READY
- [x] NOLAH optimizer implemented
- [ ] NOLAH baseline run (100 steps)
- [ ] Gate type ablation
- [ ] Scale factor sweep

📊 **Phase 3: Analysis** - PENDING
- [ ] Full 500-step runs
- [ ] Loss curve comparison
- [ ] Sample text generation
- [ ] Final report

---

## References

- **Muon Optimizer**: [Paper/Implementation URL]
- **IBM Granite**: https://huggingface.co/ibm-granite/granite-4.0-h-350m-base
- **FineWeb-Edu**: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **WandB**: https://wandb.ai/

---

## License

MIT

## Contributing

This is a research project. For questions or collaboration:
- Open an issue on GitHub
- Contact via WandB project page
