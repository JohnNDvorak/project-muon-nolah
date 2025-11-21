"""
Distributed training script for Muon/NOLAH from-scratch pretraining.

This script supports:
- Multi-GPU training with PyTorch DDP
- Random weight initialization (from-scratch pretraining)
- Streaming dataset support for large-scale training
- Enhanced metrics logging (gradient norms, loss variance)

Launch with: torchrun --nproc_per_node=4 src/train_ddp.py
"""

import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# CONFIG will be injected here by upload_script()
# CONFIG = {...}

# Import optimizer (will be in same directory on RunPod)
import sys
sys.path.insert(0, '/workspace')
from src.optim.muon import Muon


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')

    # Get rank info
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    # Set device
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def setup_wandb(config, rank):
    """Initialize WandB (only on rank 0)."""
    if rank != 0:
        return

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


def load_model_and_tokenizer(config, local_rank):
    """Load model and tokenizer with optional random initialization."""
    if local_rank == 0:
        print(f"📦 Loading model: {config['model_name']}")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # Check if we should initialize from scratch or use pretrained
    use_pretrained = config.get('use_pretrained', True)

    if use_pretrained:
        if local_rank == 0:
            print("🔄 Loading pretrained weights...")
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.bfloat16,
        )
    else:
        if local_rank == 0:
            print("🎲 Initializing random weights (from scratch)...")
        # Load config and initialize with random weights
        model_config = AutoConfig.from_pretrained(config['model_name'])
        model = AutoModelForCausalLM.from_config(
            model_config,
            torch_dtype=torch.bfloat16,
        )
        # Ensure proper initialization
        model.init_weights()

    # Move model to GPU
    model = model.to(local_rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    if local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Model loaded: {num_params:.1f}M parameters")

    return model, tokenizer


def load_data(config, tokenizer, rank, world_size):
    """Load and prepare dataset with distributed sampling."""
    if rank == 0:
        print(f"📚 Loading dataset: {config['dataset_name']}")

    # Determine if we should stream or load fully
    num_examples = config.get('num_train_examples', 10000)
    use_streaming = config.get('use_streaming', False)

    if use_streaming:
        # Streaming mode for very large datasets
        if rank == 0:
            print(f"🌊 Using streaming mode for {num_examples} examples...")

        dataset = load_dataset(
            config['dataset_name'],
            config['dataset_config'],
            split="train",
            streaming=True
        )

        # Take first num_examples
        dataset = dataset.take(num_examples)

        # Convert to iterable dataset
        from datasets import IterableDataset
        # This is already iterable, we'll handle it differently

    else:
        # Regular mode - load into memory
        if rank == 0:
            print(f"📦 Loading {num_examples} examples into memory...")

        dataset = load_dataset(
            config['dataset_name'],
            config['dataset_config'],
            split=f"train[:{num_examples}]"
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

    if rank == 0:
        print("🔄 Tokenizing dataset...")

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train" if rank == 0 else None
    )

    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val" if rank == 0 else None
    )

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.get('seed', 42)
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],  # Per-GPU batch size
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    if rank == 0:
        global_batch_size = config['batch_size'] * world_size
        print(f"✅ Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"📊 Global batch size: {global_batch_size} ({config['batch_size']} per GPU × {world_size} GPUs)")

    return train_loader, val_loader, train_sampler


def create_optimizer(model, config):
    """Create Muon, NOLAH, OrthoNoise, or Isotropic optimizer."""
    if config.get('orthonoise_enabled', False):
        print("🔬 Using OrthoNoise optimizer (Method #1)")
        from src.optim.muon_orthonoise import MuonOrthoNoise
        optimizer = MuonOrthoNoise(
            model.parameters(),
            lr=config['lr'],
            alpha=config.get('orthonoise_alpha', 1e-2),
            annealing=config.get('orthonoise_annealing', True),
            adaptive=config.get('orthonoise_adaptive', True),
            rank_threshold_ratio=config.get('orthonoise_rank_threshold', 0.5),
        )
    elif config.get('isotropic_enabled', False):
        print("🎲 Using Isotropic noise optimizer (control)")
        from src.optim.muon_isotropic import MuonIsotropic
        optimizer = MuonIsotropic(
            model.parameters(),
            lr=config['lr'],
            alpha=config.get('isotropic_alpha', 1e-2),
            annealing=config.get('isotropic_annealing', True),
            adaptive=config.get('isotropic_adaptive', True),
            rank_threshold_ratio=config.get('isotropic_rank_threshold', 0.5),
        )
    elif config.get('nolah_enabled', False):
        print("🧪 Using NOLAH-modified Muon optimizer")
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


def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def evaluate(model, val_loader, device, rank):
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

    # Average loss across all processes
    avg_loss = total_loss / num_batches

    # Gather losses from all ranks
    loss_tensor = torch.tensor([avg_loss], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    avg_loss = loss_tensor.item()

    model.train()
    return avg_loss


def train(config):
    """Main distributed training loop."""
    # Setup distributed
    local_rank, rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print(f"🚀 Starting training: {config['run_name']}")
        print(f"📊 Config: {json.dumps(config, indent=2)}")
        print(f"🌐 Distributed: {world_size} GPUs")
        print("=" * 60)

    # Setup
    setup_wandb(config, rank)
    model, tokenizer = load_model_and_tokenizer(config, local_rank)
    train_loader, val_loader, train_sampler = load_data(config, tokenizer, rank, world_size)
    optimizer = create_optimizer(model, config)

    device = torch.device(f'cuda:{local_rank}')
    model.train()

    # Training loop
    global_step = 0
    best_val_loss = float('inf')

    if rank == 0:
        output_dir = Path(config['output_dir']) / config['run_name']
        output_dir.mkdir(parents=True, exist_ok=True)

    # Running median tracking (window of 10 steps)
    from collections import deque
    import statistics
    loss_window = deque(maxlen=10)
    grad_norm_window = deque(maxlen=100)  # Track gradient norms

    # Warmup scheduler
    def get_lr_scale(step):
        if step < config['warmup_steps']:
            return step / config['warmup_steps']
        return 1.0

    train_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm(total=config['max_steps'], desc="Training")

    while global_step < config['max_steps']:
        # Set epoch for sampler (for proper shuffling)
        epoch = global_step // len(train_loader)
        train_sampler.set_epoch(epoch)

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

        # Compute gradient norm before optimizer step
        grad_norm = compute_gradient_norm(model)
        grad_norm_window.append(grad_norm)

        # Update with warmup
        lr_scale = get_lr_scale(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['lr'] * lr_scale

        optimizer.step()
        optimizer.zero_grad()

        # Track loss for running median
        current_loss = loss.item()
        loss_window.append(current_loss)

        # Calculate running median and variance
        median_loss = statistics.median(loss_window)
        if len(loss_window) > 1:
            loss_variance = statistics.variance(loss_window)
        else:
            loss_variance = 0.0

        # Log (only rank 0)
        if rank == 0:
            log_dict = {
                'train_loss': current_loss,
                'train_loss_median': median_loss,
                'train_loss_variance': loss_variance,
                'gradient_norm': grad_norm,
                'gradient_norm_mean': np.mean(list(grad_norm_window)) if grad_norm_window else 0,
                'lr': config['lr'] * lr_scale,
                'step': global_step
            }

            # Log OrthoNoise/Isotropic specific metrics
            if hasattr(optimizer, 'get_noise_stats'):
                noise_stats = optimizer.get_noise_stats()
                log_dict['noise_added_count'] = noise_stats['noise_added_count']
                log_dict['noise_skipped_count'] = noise_stats['noise_skipped_count']
                # Reset stats after logging to get per-step counts
                if global_step % 10 == 0:  # Reset every 10 steps
                    optimizer.reset_noise_stats()

            wandb.log(log_dict)

            pbar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'median': f"{median_loss:.4f}",
                'grad_norm': f"{grad_norm:.2f}",
                'lr': f"{config['lr'] * lr_scale:.2e}"
            })
            pbar.update(1)

        global_step += 1

        # Evaluate
        if global_step % config['eval_steps'] == 0:
            val_loss = evaluate(model, val_loader, device, rank)

            if rank == 0:
                print(f"\n📊 Step {global_step}: Val Loss = {val_loss:.4f}")

                wandb.log({
                    'val_loss': val_loss,
                    'step': global_step
                })

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = output_dir / "best_model"
                    model.module.save_pretrained(best_path)
                    tokenizer.save_pretrained(best_path)
                    print(f"💾 Saved best model (val_loss={val_loss:.4f})")

        # Checkpoint
        if global_step % config['save_steps'] == 0 and rank == 0:
            checkpoint_path = output_dir / f"checkpoint-{global_step}"
            model.module.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"💾 Saved checkpoint at step {global_step}")

    # Final save (only rank 0)
    if rank == 0:
        final_path = output_dir / "final_model"
        model.module.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)

        # Save metadata
        metadata = {
            'run_name': config['run_name'],
            'final_step': global_step,
            'best_val_loss': best_val_loss,
            'world_size': world_size,
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

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    # CONFIG is injected by muon.py during upload
    if 'CONFIG' not in globals():
        print("❌ ERROR: CONFIG not found. This script must be launched via muon.py")
        exit(1)

    train(CONFIG)
