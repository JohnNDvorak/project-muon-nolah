"""
Main training script for Muon/NOLAH experiments.

This script is uploaded to RunPod and executed remotely.
Configuration is injected at upload time via muon.py CLI.
"""

import os
import json
import torch
import wandb
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# CONFIG will be injected here by upload_script()
# CONFIG = {...}

# Import optimizer (will be in same directory on RunPod)
import sys
sys.path.insert(0, '/workspace')
from src.optim.muon import Muon


def setup_wandb(config):
    """Initialize WandB with timeout and offline fallback."""
    try:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            name=config['run_name'],
            config=config,
            settings=wandb.Settings(init_timeout=300),
        )
        print("✅ WandB initialized successfully")
    except Exception as e:
        print(f"⚠️  WandB init failed, using offline mode: {e}")
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            name=config['run_name'],
            config=config,
            mode="offline"
        )


def load_model_and_tokenizer(config):
    """Load model and tokenizer."""
    print(f"📦 Loading model: {config['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    return model, tokenizer


def load_data(config, tokenizer):
    """Load and prepare FineWeb-Edu dataset."""
    print(f"📚 Loading dataset: {config['dataset_name']}")

    # For quick tests, use very small dataset; for full runs, use 1%
    num_examples = config.get('num_train_examples', 10000)
    dataset = load_dataset(
        config['dataset_name'],
        config['dataset_config'],
        split=f"train[:{num_examples}]"  # Configurable number of examples
    )

    # Split into train/val
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    def tokenize_function(examples):
        tokens = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        tokens['labels'] = tokens['input_ids'].clone()
        return tokens

    print("🔄 Tokenizing dataset...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )

    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val"
    )

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"✅ Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_loader, val_loader


def create_optimizer(model, config):
    """Create Muon or NOLAH optimizer."""
    if config.get('nolah_enabled', False):
        print("🧪 Using NOLAH-modified Muon optimizer")
        # Import NOLAH variant
        from src.optim.muon_nolah import MuonNOLAH
        optimizer = MuonNOLAH(
            model.parameters(),
            lr=config['lr'],
            gate_type=config.get('nolah_gate_type', 'tanh'),
            scale_factor=config.get('nolah_scale_factor', 0.95),
        )
    else:
        print("⚙️  Using baseline Muon optimizer")
        optimizer = Muon(
            model.parameters(),
            lr=config['lr'],
        )

    return optimizer


def evaluate(model, val_loader, device):
    """Run validation."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

            # Limit validation to 50 batches for speed
            if num_batches >= 50:
                break

    avg_loss = total_loss / num_batches
    model.train()
    return avg_loss


def train(config):
    """Main training loop."""
    print("=" * 60)
    print(f"🚀 Starting training: {config['run_name']}")
    print(f"📊 Config: {json.dumps(config, indent=2)}")
    print("=" * 60)

    # Setup
    setup_wandb(config)
    model, tokenizer = load_model_and_tokenizer(config)
    train_loader, val_loader = load_data(config, tokenizer)
    optimizer = create_optimizer(model, config)

    device = next(model.parameters()).device
    model.train()

    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    output_dir = Path(config['output_dir']) / config['run_name']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Running median tracking (window of 10 steps)
    from collections import deque
    import statistics
    loss_window = deque(maxlen=10)

    # Warmup scheduler
    def get_lr_scale(step):
        if step < config['warmup_steps']:
            return step / config['warmup_steps']
        return 1.0

    train_iter = iter(train_loader)

    pbar = tqdm(total=config['max_steps'], desc="Training")

    while global_step < config['max_steps']:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update with warmup
        lr_scale = get_lr_scale(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['lr'] * lr_scale

        optimizer.step()
        optimizer.zero_grad()

        # Track loss for running median
        current_loss = loss.item()
        loss_window.append(current_loss)

        # Calculate running median
        median_loss = statistics.median(loss_window)

        # Log
        wandb.log({
            'train_loss': current_loss,
            'train_loss_median': median_loss,
            'lr': config['lr'] * lr_scale,
            'step': global_step
        })

        pbar.set_postfix({'loss': f"{current_loss:.4f}", 'median': f"{median_loss:.4f}", 'lr': f"{config['lr'] * lr_scale:.2e}"})
        pbar.update(1)

        global_step += 1

        # Evaluate
        if global_step % config['eval_steps'] == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"\n📊 Step {global_step}: Val Loss = {val_loss:.4f}")

            wandb.log({
                'val_loss': val_loss,
                'step': global_step
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "best_model"
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                print(f"💾 Saved best model (val_loss={val_loss:.4f})")

        # Checkpoint
        if global_step % config['save_steps'] == 0:
            checkpoint_path = output_dir / f"checkpoint-{global_step}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"💾 Saved checkpoint at step {global_step}")

    # Final save
    final_path = output_dir / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save metadata
    metadata = {
        'run_name': config['run_name'],
        'final_step': global_step,
        'best_val_loss': best_val_loss,
        'config': config
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print(f"✅ Training complete!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Results saved to: {output_dir}")
    print("=" * 60)

    wandb.finish()


if __name__ == "__main__":
    # CONFIG is injected by muon.py during upload
    if 'CONFIG' not in globals():
        print("❌ ERROR: CONFIG not found. This script must be launched via muon.py")
        exit(1)

    train(CONFIG)
