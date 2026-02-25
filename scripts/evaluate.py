"""DiaFoot.AI v2 — Evaluation Entry Point."""

from __future__ import annotations

import argparse


def main() -> None:
    """Run comprehensive evaluation on trained model."""
    parser = argparse.ArgumentParser(description="DiaFoot.AI v2 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tta", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--save-predictions", type=str, default="false")
    args = parser.parse_args()

    print(f"Evaluation | Checkpoint: {args.checkpoint} | TTA: {args.tta}")
    # TODO: Phase 4 implementation


if __name__ == "__main__":
    main()
