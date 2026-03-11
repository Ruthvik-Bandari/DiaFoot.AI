"""Summarize 5-fold cross-validation JSON outputs.

Usage:
    python scripts/summarize_cv_results.py --results-dir results --folds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CV fold metrics")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/cv_summary.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for fold in range(args.folds):
        p = results_dir / f"cv_fold{fold}.json"
        if not p.exists():
            continue
        with open(p) as f:
            rows.append(json.load(f))

    if not rows:
        raise FileNotFoundError("No cv_fold*.json files found")

    dice = np.array([float(r.get("dice", 0.0)) for r in rows], dtype=float)
    iou = np.array([float(r.get("iou", 0.0)) for r in rows], dtype=float)

    summary = {
        "num_folds": len(rows),
        "dice": {
            "mean": float(dice.mean()),
            "std": float(dice.std(ddof=1)) if len(dice) > 1 else 0.0,
            "min": float(dice.min()),
            "max": float(dice.max()),
        },
        "iou": {
            "mean": float(iou.mean()),
            "std": float(iou.std(ddof=1)) if len(iou) > 1 else 0.0,
            "min": float(iou.min()),
            "max": float(iou.max()),
        },
        "folds": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("CV summary written:", out)
    print("Dice mean±std:", f"{summary['dice']['mean']:.4f} ± {summary['dice']['std']:.4f}")
    print("IoU  mean±std:", f"{summary['iou']['mean']:.4f} ± {summary['iou']['std']:.4f}")


if __name__ == "__main__":
    main()
