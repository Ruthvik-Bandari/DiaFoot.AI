"""DiaFoot.AI v2 — Triage Classifier Training."""

from __future__ import annotations

import argparse


def main() -> None:
    """Train the triage classifier (Healthy vs Non-DFU vs DFU)."""
    parser = argparse.ArgumentParser(description="Train triage classifier")
    parser.add_argument("--config", type=str, default="configs/model/classifier.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Triage Classifier Training | Config: {args.config}")
    # TODO: Phase 2-3 implementation


if __name__ == "__main__":
    main()
