#!/usr/bin/env python3
"""
Muon-NOLAH CLI - Unified interface for training and managing experiments.

Usage:
    python muon.py baseline --steps 500
    python muon.py nolah --gate tanh --steps 100
    python muon.py status
    python muon.py logs --tail 50
    python muon.py download --run baseline_v4
    python muon.py commit "Completed baseline run"
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_baseline(args):
    """Launch baseline Muon training."""
    from utils.runpod_ssh import launch_training

    print(f"🚀 Launching baseline training ({args.steps} steps)...")

    config = {
        "run_name": f"granite_baseline_{args.name}" if args.name else "granite_baseline",
        "model_name": os.getenv("MODEL_NAME", "ibm-granite/granite-4.0-h-350m-base"),
        "dataset_name": os.getenv("DATASET_NAME", "HuggingFaceFW/fineweb-edu"),
        "dataset_config": os.getenv("DATASET_CONFIG", "sample-10BT"),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "max_steps": int(args.steps),
        "warmup_steps": int(args.warmup),
        "eval_steps": int(args.eval_steps),
        "save_steps": int(args.save_steps),
        "nolah_enabled": False,
        "output_dir": "/workspace/results",
        "wandb_project": os.getenv("WANDB_PROJECT", "granite-muon-nolah"),
        "wandb_entity": os.getenv("WANDB_ENTITY", "fishhooks1-independent-researcher")
    }

    launch_training(
        script_path="src/train.py",
        config=config,
        dry_run=args.dry_run
    )


def cmd_nolah(args):
    """Launch NOLAH-modified Muon training."""
    from utils.runpod_ssh import launch_training

    print(f"🧪 Launching NOLAH training (gate={args.gate}, scale={args.scale}, {args.steps} steps)...")

    config = {
        "run_name": f"granite_nolah_{args.name}" if args.name else "granite_nolah",
        "model_name": os.getenv("MODEL_NAME", "ibm-granite/granite-4.0-h-350m-base"),
        "dataset_name": os.getenv("DATASET_NAME", "HuggingFaceFW/fineweb-edu"),
        "dataset_config": os.getenv("DATASET_CONFIG", "sample-10BT"),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "max_steps": int(args.steps),
        "warmup_steps": int(args.warmup),
        "eval_steps": int(args.eval_steps),
        "save_steps": int(args.save_steps),
        "nolah_enabled": True,
        "nolah_gate_type": args.gate,
        "nolah_scale_factor": float(args.scale),
        "output_dir": "/workspace/results",
        "wandb_project": os.getenv("WANDB_PROJECT", "granite-muon-nolah"),
        "wandb_entity": os.getenv("WANDB_ENTITY", "fishhooks1-independent-researcher")
    }

    launch_training(
        script_path="src/train.py",
        config=config,
        dry_run=args.dry_run
    )


def cmd_status(args):
    """Check RunPod training status."""
    from utils.runpod_ssh import check_status

    print("📊 Checking RunPod status...\n")
    check_status(show_gpu=args.gpu, verbose=args.verbose)


def cmd_logs(args):
    """View training logs."""
    from utils.runpod_ssh import get_logs

    print(f"📜 Fetching logs (tail={args.tail})...\n")
    get_logs(tail=args.tail, follow=args.follow)


def cmd_download(args):
    """Download results from RunPod."""
    from utils.runpod_ssh import download_results

    print(f"⬇️  Downloading results for run: {args.run}...")
    download_results(run_name=args.run, include_checkpoints=not args.no_checkpoints)


def cmd_commit(args):
    """Commit results to git."""
    import subprocess

    print(f"💾 Preparing commit: {args.message}")

    # Add metadata files (not large checkpoints)
    subprocess.run(["git", "add", "results/**/metadata.json"], check=False)
    subprocess.run(["git", "add", "results/**/training_log.txt"], check=False)
    subprocess.run(["git", "add", "src/", "config/", "scripts/"], check=True)

    # Show what will be committed
    result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
    print("\n📝 Changes to commit:")
    print(result.stdout)

    # Commit
    subprocess.run(["git", "commit", "-m", args.message], check=True)
    print("✅ Committed locally")

    # Push if requested
    if args.push:
        response = input("\n🚀 Push to GitHub? [y/N] ")
        if response.lower() == 'y':
            subprocess.run(["git", "push"], check=True)
            print("✅ Pushed to GitHub")


def cmd_setup(args):
    """Setup Google Drive symlink and verify environment."""
    import subprocess

    print("🔧 Setting up Project Muon-NOLAH...\n")

    # Check for .env file
    env_file = Path("secrets/.env")
    if not env_file.exists():
        print("⚠️  secrets/.env not found. Creating from template...")
        template = Path("secrets/.env.template")
        with open(template) as f:
            content = f.read()
        with open(env_file, 'w') as f:
            f.write(content)
        print("📝 Created secrets/.env - please fill in your API keys!")
        return

    # Check Google Drive
    if args.drive_path:
        drive_path = Path(args.drive_path).expanduser()
        results_path = Path("results")

        if not results_path.exists() or not results_path.is_symlink():
            print(f"🔗 Creating symlink: results -> {drive_path}")
            drive_path.mkdir(parents=True, exist_ok=True)

            if results_path.exists():
                print(f"⚠️  Moving existing results/ to {drive_path}")
                subprocess.run(["mv", "results", str(drive_path / "backup")], check=True)

            results_path.symlink_to(drive_path)
            print("✅ Google Drive symlink created")
        else:
            print(f"✅ Results already linked to: {results_path.resolve()}")

    print("\n✅ Setup complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Muon-NOLAH CLI - Train and benchmark NOLAH optimizer modifications",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Baseline command
    baseline = subparsers.add_parser("baseline", help="Launch baseline Muon training")
    baseline.add_argument("--steps", type=int, default=500, help="Training steps (default: 500)")
    baseline.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    baseline.add_argument("--batch-size", type=int, default=16, help="Batch size")
    baseline.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    baseline.add_argument("--eval-steps", type=int, default=50, help="Evaluation interval")
    baseline.add_argument("--save-steps", type=int, default=100, help="Checkpoint interval")
    baseline.add_argument("--name", type=str, help="Run name suffix")
    baseline.add_argument("--dry-run", action="store_true", help="Show config without launching")
    baseline.set_defaults(func=cmd_baseline)

    # NOLAH command
    nolah = subparsers.add_parser("nolah", help="Launch NOLAH training")
    nolah.add_argument("--gate", choices=["tanh", "sigmoid", "relu"], default="tanh", help="Gate function")
    nolah.add_argument("--scale", type=float, default=0.95, help="Momentum scale factor")
    nolah.add_argument("--steps", type=int, default=100, help="Training steps (default: 100)")
    nolah.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    nolah.add_argument("--batch-size", type=int, default=16, help="Batch size")
    nolah.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    nolah.add_argument("--eval-steps", type=int, default=50, help="Evaluation interval")
    nolah.add_argument("--save-steps", type=int, default=100, help="Checkpoint interval")
    nolah.add_argument("--name", type=str, help="Run name suffix")
    nolah.add_argument("--dry-run", action="store_true", help="Show config without launching")
    nolah.set_defaults(func=cmd_nolah)

    # Status command
    status = subparsers.add_parser("status", help="Check RunPod training status")
    status.add_argument("--gpu", action="store_true", help="Show GPU utilization")
    status.add_argument("--verbose", action="store_true", help="Verbose output")
    status.set_defaults(func=cmd_status)

    # Logs command
    logs = subparsers.add_parser("logs", help="View training logs")
    logs.add_argument("--tail", type=int, default=50, help="Number of lines to show")
    logs.add_argument("--follow", action="store_true", help="Follow log output")
    logs.set_defaults(func=cmd_logs)

    # Download command
    download = subparsers.add_parser("download", help="Download results from RunPod")
    download.add_argument("--run", required=True, help="Run name to download")
    download.add_argument("--no-checkpoints", action="store_true", help="Skip large checkpoint files")
    download.set_defaults(func=cmd_download)

    # Commit command
    commit = subparsers.add_parser("commit", help="Commit results to git")
    commit.add_argument("message", help="Commit message")
    commit.add_argument("--push", action="store_true", help="Push to GitHub after commit")
    commit.set_defaults(func=cmd_commit)

    # Setup command
    setup = subparsers.add_parser("setup", help="Setup environment and Google Drive")
    setup.add_argument("--drive-path", help="Google Drive path for results (e.g., ~/Google Drive/muon-nolah-results)")
    setup.set_defaults(func=cmd_setup)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv("secrets/.env")

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
