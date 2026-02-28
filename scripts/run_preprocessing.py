"""DiaFoot.AI v2 — Preprocessing Pipeline + Stratified Splits.

Phase 1, Commit 7.

Usage:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --skip-splits
    python scripts/run_preprocessing.py --target-size 256
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_stratified_splits
from src.data.preprocessing import preprocess_dataset


def main() -> None:
    """Run full preprocessing pipeline and create splits."""
    parser = argparse.ArgumentParser(description="Preprocessing + Splits")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("preprocessing")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    target = (args.target_size, args.target_size)

    # Define dataset mappings: (category, image_subdir, mask_subdir)
    datasets = [
        (
            "dfu",
            data_root / "dfu" / "fuseg" / "train" / "images",
            data_root / "dfu" / "fuseg" / "train" / "labels",
        ),
        (
            "dfu",
            data_root / "dfu" / "fuseg" / "validation" / "images",
            data_root / "dfu" / "fuseg" / "validation" / "labels",
        ),
        ("healthy", data_root / "healthy" / "images", data_root / "healthy" / "masks"),
        ("non_dfu", data_root / "non_dfu" / "images", data_root / "non_dfu" / "masks"),
    ]

    total_stats: dict[str, dict[str, int]] = {}

    for category, img_dir, mask_dir in datasets:
        if not img_dir.exists():
            logger.warning("Skipping %s: %s not found", category, img_dir)
            continue

        out_img = output_dir / category / "images"
        out_mask = output_dir / category / "masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)

        logger.info("Preprocessing %s from %s...", category, img_dir)
        result = preprocess_dataset(
            image_dir=img_dir,
            mask_dir=mask_dir if mask_dir.exists() else None,
            output_image_dir=out_img,
            output_mask_dir=out_mask,
            target_size=target,
            apply_clahe_flag=not args.no_clahe,
        )

        ds_key = f"{category}_{img_dir.parent.name}"
        total_stats[ds_key] = result
        logger.info("  %s: %d success, %d failed", ds_key, result["success"], result["failed"])

    # Print preprocessing summary
    print(f"\n{'=' * 60}")
    print("Preprocessing — Summary")
    print(f"{'=' * 60}")
    grand_total = 0
    for ds, counts in total_stats.items():
        print(f"  {ds}: {counts['success']} processed, {counts['failed']} failed")
        grand_total += counts["success"]
    print(f"\n  Total processed: {grand_total}")
    print(f"  Output: {output_dir}")

    # Create stratified splits
    if not args.skip_splits:
        logger.info("Creating stratified splits...")
        split_stats = create_stratified_splits(
            processed_dir=output_dir,
            splits_dir=args.splits_dir,
        )

        print(f"\n{'=' * 60}")
        print("Stratified Splits — Summary")
        print(f"{'=' * 60}")
        print(f"  Train: {split_stats.get('train', 0)}")
        print(f"  Val:   {split_stats.get('val', 0)}")
        print(f"  Test:  {split_stats.get('test', 0)}")

        for split_name in ["train", "val", "test"]:
            class_dist = split_stats.get("class_distribution", {}).get(split_name, {})
            if class_dist:
                print(f"\n  {split_name} class distribution:")
                for cls, count in sorted(class_dist.items()):
                    print(f"    {cls}: {count}")

        print(f"\n  Split files: {args.splits_dir}/")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
