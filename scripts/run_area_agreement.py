"""DiaFoot.AI v2 — Evaluate agreement between model and manual wound area measurements.

Expected CSV columns:
- pred_area_mm2
- manual_area_mm2
Optional:
- size_group (small|medium|large)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.evaluation.area_agreement import compute_area_agreement


def main() -> None:
    parser = argparse.ArgumentParser(description="Area agreement analysis")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/area_agreement.json")
    args = parser.parse_args()

    rows = []
    with open(args.input_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    pred = np.array([float(r["pred_area_mm2"]) for r in rows], dtype=float)
    manual = np.array([float(r["manual_area_mm2"]) for r in rows], dtype=float)

    report = {
        "overall": compute_area_agreement(pred, manual),
    }

    # Optional subgroup analysis by size_group
    if rows and "size_group" in rows[0]:
        by_group: dict[str, dict] = {}
        groups = sorted(set(r.get("size_group", "unknown") for r in rows))
        for g in groups:
            idx = [i for i, r in enumerate(rows) if r.get("size_group", "unknown") == g]
            if not idx:
                continue
            by_group[g] = compute_area_agreement(pred[idx], manual[idx])
        report["by_size_group"] = by_group

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Area agreement report saved to: {out}")  # noqa: T201


if __name__ == "__main__":
    main()
