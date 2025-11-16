"""
RunPod SSH utilities for remote training execution.

Handles:
- Script upload via base64 encoding (avoids shell quoting issues)
- Training launch with environment injection
- Status monitoring (GPU, process, logs)
- Results download via rsync
"""

import os
import subprocess
import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional


def get_ssh_config() -> Dict[str, str]:
    """Load SSH configuration from environment variables."""
    config = {
        "pod_ip": os.getenv("POD_IP"),
        "ssh_port": os.getenv("SSH_PORT"),
        "ssh_key": os.getenv("SSH_KEY_PATH", os.path.expanduser("~/.ssh/id_ed25519_runpod")),
        "wandb_key": os.getenv("WANDB_API_KEY"),
    }

    missing = [k for k, v in config.items() if not v]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}. Check secrets/.env")

    return config


def build_ssh_command(config: Dict[str, str], command: str) -> list:
    """Build SSH command with proper authentication."""
    return [
        "ssh",
        "-i", config["ssh_key"],
        "-p", config["ssh_port"],
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"root@{config['pod_ip']}",
        command
    ]


def upload_script(script_path: str, config_dict: Dict[str, Any], dry_run: bool = False) -> None:
    """
    Upload training script to RunPod with embedded configuration.

    Uses base64 encoding to avoid shell escaping issues.
    """
    ssh_config = get_ssh_config()

    # Read the training script
    with open(script_path, 'r') as f:
        script_content = f.read()

    # Inject configuration at the top of the script
    config_json = json.dumps(config_dict, indent=2)
    modified_script = f"""# Auto-generated configuration
CONFIG = {config_json}

{script_content}
"""

    if dry_run:
        print("=" * 60)
        print("DRY RUN - Would upload the following script:")
        print("=" * 60)
        print(modified_script[:500] + "\n...\n")
        print(f"Total script length: {len(modified_script)} bytes")
        print("=" * 60)
        return

    # Encode script as base64
    encoded = base64.b64encode(modified_script.encode()).decode()

    # Upload via SSH
    print(f"📤 Uploading {script_path} to RunPod...")

    upload_cmd = build_ssh_command(
        ssh_config,
        f'echo "{encoded}" | base64 -d > /workspace/train.py'
    )

    result = subprocess.run(upload_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Upload failed: {result.stderr}")
        raise RuntimeError("Failed to upload script")

    print("✅ Script uploaded successfully")


def launch_training(script_path: str, config: Dict[str, Any], dry_run: bool = False) -> None:
    """
    Launch training on RunPod.

    Steps:
    1. Upload script with embedded config
    2. Start training in background with nohup
    3. Return WandB link
    """
    ssh_config = get_ssh_config()

    # Upload script first
    upload_script(script_path, config, dry_run)

    if dry_run:
        print("\nDRY RUN - Would execute:")
        print(f"  python /workspace/train.py")
        print(f"\nWith config:")
        print(json.dumps(config, indent=2))
        return

    # Launch training
    print(f"🚀 Starting training (run: {config['run_name']})...")

    launch_cmd = build_ssh_command(
        ssh_config,
        f"export WANDB_API_KEY={ssh_config['wandb_key']} && "
        f"nohup python /workspace/train.py > /workspace/training.log 2>&1 &"
    )

    result = subprocess.run(launch_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Launch failed: {result.stderr}")
        raise RuntimeError("Failed to launch training")

    print("✅ Training started!")
    print(f"\n📊 WandB: https://wandb.ai/{config['wandb_entity']}/{config['wandb_project']}")
    print(f"📝 Check logs: python muon.py logs")
    print(f"📈 Check status: python muon.py status")


def check_status(show_gpu: bool = False, verbose: bool = False) -> None:
    """Check RunPod training status."""
    ssh_config = get_ssh_config()

    # Check if training process is running
    ps_cmd = build_ssh_command(
        ssh_config,
        "ps aux | grep 'python /workspace/train.py' | grep -v grep"
    )

    result = subprocess.run(ps_cmd, capture_output=True, text=True)

    if result.stdout.strip():
        print("🟢 Training is RUNNING")
        if verbose:
            print(f"\nProcess info:\n{result.stdout}")
    else:
        print("🔴 Training is NOT RUNNING")

    # Show GPU status if requested
    if show_gpu:
        gpu_cmd = build_ssh_command(ssh_config, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")

        result = subprocess.run(gpu_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"\n🎮 GPU Status:")
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                util = parts[0].strip()
                mem_used = parts[1].strip()
                mem_total = parts[2].strip()
                print(f"  Utilization: {util}%")
                print(f"  Memory: {mem_used}MB / {mem_total}MB")

    # Show last few lines of log
    tail_cmd = build_ssh_command(ssh_config, "tail -n 5 /workspace/training.log 2>/dev/null")
    result = subprocess.run(tail_cmd, capture_output=True, text=True)

    if result.stdout.strip():
        print(f"\n📜 Last log lines:")
        print(result.stdout)


def get_logs(tail: int = 50, follow: bool = False) -> None:
    """Fetch training logs from RunPod."""
    ssh_config = get_ssh_config()

    if follow:
        # Follow logs in real-time
        follow_cmd = build_ssh_command(ssh_config, "tail -f /workspace/training.log")
        subprocess.run(follow_cmd)
    else:
        # Get last N lines
        tail_cmd = build_ssh_command(ssh_config, f"tail -n {tail} /workspace/training.log")
        result = subprocess.run(tail_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Failed to fetch logs: {result.stderr}")


def download_results(run_name: str, include_checkpoints: bool = True) -> None:
    """
    Download results from RunPod to local results/ directory.

    Args:
        run_name: Name of the run (e.g., 'baseline_v4')
        include_checkpoints: If False, skip large .pth/.safetensors files
    """
    ssh_config = get_ssh_config()
    local_dir = Path("results") / run_name
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_path = f"/workspace/results/{run_name}/"

    # Build rsync command
    rsync_cmd = [
        "rsync",
        "-avz",
        "-e", f"ssh -i {ssh_config['ssh_key']} -p {ssh_config['ssh_port']} -o StrictHostKeyChecking=no",
    ]

    # Exclude large files if requested
    if not include_checkpoints:
        rsync_cmd.extend([
            "--exclude", "*.pth",
            "--exclude", "*.safetensors",
            "--exclude", "*.bin",
        ])

    rsync_cmd.extend([
        f"root@{ssh_config['pod_ip']}:{remote_path}",
        str(local_dir) + "/"
    ])

    print(f"⬇️  Downloading from {remote_path}...")
    if not include_checkpoints:
        print("   (Skipping large checkpoint files)")

    result = subprocess.run(rsync_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Downloaded to {local_dir}")

        # Show what was downloaded
        files = list(local_dir.rglob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"\n📁 Downloaded {len(files)} items ({total_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"❌ Download failed: {result.stderr}")
        raise RuntimeError("Failed to download results")
