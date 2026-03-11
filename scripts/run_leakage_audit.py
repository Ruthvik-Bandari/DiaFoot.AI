"""DiaFoot.AI v2 — Split Leakage Audit CLI.

Usage:
    python scripts/run_leakage_audit.py
    python scripts/run_leakage_audit.py --splits-dir data/splits \
        --output data/metadata/leakage_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.leakage_audit import audit_samples_for_leakage, load_split_csv


def main() -> None:
    """Run leakage audit on train/val/test CSV splits."""
    parser = argparse.ArgumentParser(description="Split leakage audit")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--output", type=str, default="data/metadata/leakage_report.json")
    parser.add_argument("--near-threshold", type=int, default=6)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    splits_dir = Path(args.splits_dir)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

    train = load_split_csv(train_csv)
    val = load_split_csv(val_csv)
    test = load_split_csv(test_csv)

    report = audit_samples_for_leakage(
        train_samples=train,
        val_samples=val,
        test_samples=test,
        near_duplicate_threshold=args.near_threshold,
        max_near_duplicate_pairs=args.max_examples,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("Split Leakage Audit")
    print("=" * 60)
    print(f"Train/Val/Test: {report['counts']}")
    print(f"Path overlap: {report['path_overlap']}")
    print(f"Canonical overlap: {report['canonical_overlap']}")
    print(f"Content overlap: {report['content_overlap']}")
    print(f"Near-duplicates: {report['near_duplicates']['counts']}")
    print(f"Any leakage signal: {report['has_any_leakage']}")
    print(f"Report written: {output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
