"""DiaFoot.AI v2 — Run Data Cleaning / Quality Audit.

Phase 1, Commit 2: Execute CleanVision audit on downloaded datasets.

Usage:
    # Audit all DFU datasets
    python scripts/run_cleaning.py

    # Audit a specific directory
    python scripts/run_cleaning.py --data-dir data/raw/dfu/fuseg --name fuseg

    # Use custom thresholds
    python scripts/run_cleaning.py --config configs/data/cleaning.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cleaning import DataQualityAuditor, audit_all_dfu_datasets


def main() -> None:
    """Run data quality audit."""
    parser = argparse.ArgumentParser(description="DiaFoot.AI v2 — Data Quality Audit")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Specific image directory to audit (default: audit all under data/raw/dfu)",
    )
    parser.add_argument("--name", type=str, default=None, help="Dataset name for the report")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/cleaning.yaml",
        help="Cleaning config YAML path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/metadata",
        help="Directory for output reports",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_cleaning")

    if args.data_dir:
        # Audit a single directory
        data_path = Path(args.data_dir)
        if not data_path.exists():
            logger.error("Directory not found: %s", data_path)
            sys.exit(1)

        from src.data.cleaning import AuditConfig

        config_path = Path(args.config)
        config = AuditConfig.from_yaml(config_path) if config_path.exists() else AuditConfig()

        auditor = DataQualityAuditor(
            image_dir=data_path,
            config=config,
            dataset_name=args.name or data_path.name,
        )
        auditor.run_audit()
        auditor.print_summary()

        output_path = Path(args.output_dir) / f"quality_report_{auditor.dataset_name}.json"
        auditor.save_report(output_path)
        print(f"Report saved: {output_path}")

    else:
        # Audit all DFU datasets
        logger.info("Running audit on all datasets in data/raw/dfu/")
        reports = audit_all_dfu_datasets(
            data_root="data/raw/dfu",
            config_path=args.config,
            output_dir=args.output_dir,
        )
        print(f"\nAudited {len(reports)} datasets. Reports saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
