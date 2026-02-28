"""DiaFoot.AI v2 — ITA Skin Tone Analysis.

Phase 1, Commit 6.

Usage:
    python scripts/run_ita_analysis.py
    python scripts/run_ita_analysis.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ita_analysis import run_ita_analysis


def main() -> None:
    """Run ITA analysis across all datasets."""
    parser = argparse.ArgumentParser(description="ITA Skin Tone Analysis")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/metadata")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    report = run_ita_analysis(args.data_root, args.output_dir)

    print(f"\n{'=' * 60}")
    print("ITA Skin Tone Analysis — Summary")
    print(f"{'=' * 60}")

    for ds_name, ds_report in report.get("datasets", {}).items():
        total = ds_report.get("total_analyzed", 0)
        stats = ds_report.get("ita_stats", {})
        dist = ds_report.get("category_distribution", {})

        print(f"\n  {ds_name} ({total} images):")
        if stats:
            print(
                f"    ITA: mean={stats['mean']:.1f}, "
                f"median={stats['median']:.1f}, "
                f"std={stats['std']:.1f}, "
                f"range=[{stats['min']:.1f}, {stats['max']:.1f}]"
            )
        if dist:
            print("    Distribution:")
            for cat, count in dist.items():
                pct = count / total * 100 if total > 0 else 0
                bar = "█" * int(pct / 2)
                print(f"      {cat:15s}: {count:5d} ({pct:5.1f}%) {bar}")

    print("\n  Files: data/metadata/ita_scores.csv, data/metadata/ita_report.json")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
