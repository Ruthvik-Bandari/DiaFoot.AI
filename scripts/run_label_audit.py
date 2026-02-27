"""DiaFoot.AI v2 — Label Quality Audit + Wagner Grade Setup.

Phase 1, Commit 5.

Usage:
    python scripts/run_label_audit.py
    python scripts/run_label_audit.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.wagner_labeling import run_label_audit


def main() -> None:
    """Run label quality audit across all datasets."""
    parser = argparse.ArgumentParser(description="Label Quality Audit + Wagner Grade Setup")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/metadata")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    report = run_label_audit(args.data_root, args.output_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Label Quality Audit — Summary")
    print(f"{'=' * 60}")

    for ds_name, ds_report in report.get("datasets", {}).items():
        total = ds_report.get("total_masks", 0)
        valid = ds_report.get("valid", 0)
        invalid = ds_report.get("invalid", 0)
        empty = ds_report.get("empty_masks", 0)
        coverage = ds_report.get("coverage_stats", {})

        print(f"\n  {ds_name}:")
        print(f"    Masks: {total} (valid: {valid}, invalid: {invalid}, empty: {empty})")
        if coverage:
            print(
                f"    Coverage: mean={coverage.get('mean_pct', 0):.1f}%, "
                f"median={coverage.get('median_pct', 0):.1f}%, "
                f"range=[{coverage.get('min_pct', 0):.1f}%, {coverage.get('max_pct', 0):.1f}%]"
            )
        issues = ds_report.get("issues", {})
        if issues:
            for issue_type, info in issues.items():
                print(f"    Issue: {issue_type} — {info['count']} masks")

    if "wagner_grade_entries" in report:
        print(f"\n  Wagner grade CSV: {report['wagner_grade_entries']} DFU images to annotate")
        print("    → data/metadata/wagner_grades.csv (fill in grades manually)")

    print("\n  Full report: data/metadata/label_issues.json")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
