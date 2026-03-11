"""DiaFoot.AI v2 — Healthy Foot Data Collection Pipeline.

Phase 1, Commit 3: Download, organize, validate, and audit healthy foot images.

Usage:
    # Full pipeline: organize from downloaded sources + validate + audit
    python scripts/collect_healthy_feet.py

    # Just create empty masks for already-collected images
    python scripts/collect_healthy_feet.py --masks-only

    # Specify source directories
    python scripts/collect_healthy_feet.py \
        --kaggle-dir /path/to/kaggle/dfu \
        --mendeley-dir /path/to/mendeley/dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.healthy_feet import (
    create_empty_masks,
    organize_kaggle_dfu_normal,
    organize_mendeley_normal,
    run_quality_audit_on_healthy,
    validate_healthy_images,
)


def main() -> None:
    """Run healthy foot data collection pipeline."""
    parser = argparse.ArgumentParser(description="Collect & curate healthy foot images")
    parser.add_argument(
        "--kaggle-dir", type=str, default=None, help="Path to extracted Kaggle DFU dataset"
    )
    parser.add_argument(
        "--mendeley-dir", type=str, default=None, help="Path to extracted Mendeley dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw/healthy", help="Output directory"
    )
    parser.add_argument("--masks-only", action="store_true", help="Only create empty masks")
    parser.add_argument("--skip-audit", action="store_true", help="Skip CleanVision audit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("collect_healthy")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)

    total_collected = 0

    if not args.masks_only:
        # ── Step 1: Organize from Kaggle ────────────────────────────────
        if args.kaggle_dir:
            logger.info("Organizing Kaggle normal foot images...")
            n = organize_kaggle_dfu_normal(args.kaggle_dir, images_dir)
            total_collected += n
            logger.info("  Kaggle: %d images", n)
        else:
            logger.info(
                "Kaggle source not provided. To use it:\n"
                "  1. Download from: https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu\n"
                "  2. Extract the zip\n"
                "  3. Re-run with: --kaggle-dir /path/to/extracted/folder"
            )

        # ── Step 2: Organize from Mendeley ──────────────────────────────
        if args.mendeley_dir:
            logger.info("Organizing Mendeley normal foot images...")
            n = organize_mendeley_normal(args.mendeley_dir, images_dir)
            total_collected += n
            logger.info("  Mendeley: %d images", n)
        else:
            logger.info(
                "Mendeley source not provided. To use it:\n"
                "  1. Download from: https://data.mendeley.com/datasets/hsj38fwnvr/3\n"
                "  2. Extract the zip\n"
                "  3. Re-run with: --mendeley-dir /path/to/extracted/folder"
            )

    # ── Step 3: Validate ────────────────────────────────────────────────
    existing_images = list(images_dir.glob("*"))
    image_count = len([f for f in existing_images if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    if image_count == 0:
        logger.warning("No healthy foot images found in %s", images_dir)
        logger.info(
            "\nTo collect healthy foot data, download from these sources:\n"
            "  1. Kaggle: https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu\n"
            "     (Contains ~543 normal foot images)\n"
            "  2. Mendeley: https://data.mendeley.com/datasets/hsj38fwnvr/3\n"
            "     (Contains normal feet images, CC BY 4.0)\n"
            "\nThen re-run with --kaggle-dir and/or --mendeley-dir flags."
        )
        return

    logger.info("Validating %d healthy foot images...", image_count)
    validation = validate_healthy_images(images_dir)
    logger.info(
        "Validation results: %d valid, %d corrupt, %d too_small, %d grayscale",
        len(validation["valid"]),
        len(validation["corrupt"]),
        len(validation["too_small"]),
        len(validation["grayscale"]),
    )

    # ── Step 4: Create empty masks ──────────────────────────────────────
    logger.info("Creating empty (all-zero) segmentation masks...")
    mask_count = create_empty_masks(images_dir, masks_dir)
    logger.info("Created %d empty masks", mask_count)

    # ── Step 5: Quality audit ───────────────────────────────────────────
    if not args.skip_audit:
        logger.info("Running CleanVision quality audit...")
        run_quality_audit_on_healthy(
            images_dir,
            output_path="data/metadata/quality_report_healthy.json",
        )

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("Healthy Foot Data Collection — Summary")
    print(f"{'═' * 60}")
    print(f"Images:      {image_count} in {images_dir}")
    print(f"Masks:       {mask_count} in {masks_dir}")
    print(f"Valid:       {len(validation['valid'])}")
    print(f"Corrupt:     {len(validation['corrupt'])}")
    print(f"Too small:   {len(validation['too_small'])}")
    print(f"Grayscale:   {len(validation['grayscale'])}")
    if image_count < 2000:
        print(f"\n⚠  Target is 2,000-5,000 healthy images. Currently at {image_count}.")
        print("   Additional sources needed (web scraping, clinical partnerships).")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
