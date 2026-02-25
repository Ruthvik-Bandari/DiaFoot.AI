"""DiaFoot.AI v2 — Training Entry Point.

Usage:
    python scripts/train.py --config configs/training/baseline.yaml
    python scripts/train.py --config configs/training/multitask.yaml --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="DiaFoot.AI v2 Training")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--compile", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--distributed", type=str, default=None, choices=["ddp", "fsdp", None])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tags", type=str, default=None, help="Comma-separated W&B tags")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    return parser.parse_args()


def main() -> None:
    """Run training pipeline."""
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    print("DiaFoot.AI v2 Training")
    print(f"  Config: {config_path} | Device: {args.device} | Seed: {args.seed}")

    # TODO: Phase 3 (Commits 13-18)
    print("Training pipeline scaffold ready. Implementation in Phase 3.")


if __name__ == "__main__":
    main()
