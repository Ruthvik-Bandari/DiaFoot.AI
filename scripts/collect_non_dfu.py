"""DiaFoot.AI v2 — Non-DFU Foot Condition Data Collection.

Phase 1, Commit 4: Collect hard negative examples (wounds that are NOT DFU).

Sources:
    1. Mendeley wound dataset (wound_main + wound_mask) — 2,686 general wound images
    2. DermNet foot subset (optional) — callus, corn, fungal, eczema, warts

Usage:
    # After copying Mendeley wounds to data/raw/non_dfu/:
    python scripts/collect_non_dfu.py --audit

    # With DermNet dataset (optional):
    python scripts/collect_non_dfu.py --dermnet-dir /path/to/dermnet --audit
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cleaning import DataQualityAuditor

# DermNet categories relevant as non-DFU foot conditions
DERMNET_FOOT_CATEGORIES = [
    "Cellulitis",
    "Eczema Photos",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections",
]


def collect_from_dermnet(
    dermnet_dir: Path,
    output_dir: Path,
    categories: list[str] | None = None,
    max_per_category: int = 200,
) -> int:
    """Extract foot-relevant skin conditions from DermNet dataset.

    Args:
        dermnet_dir: Path to extracted DermNet dataset (train/ folder).
        output_dir: Where to copy selected images.
        categories: List of category folder names to include.
        max_per_category: Max images per category to avoid imbalance.

    Returns:
        Number of images copied.
    """
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = categories or DERMNET_FOOT_CATEGORIES

    # DermNet has train/ and test/ folders, each with category subdirectories
    count = 0
    for split in ["train", "test"]:
        split_dir = dermnet_dir / split
        if not split_dir.exists():
            continue

        for cat_dir in sorted(split_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            # Check if this category matches any of our target categories
            matched = any(target.lower() in cat_dir.name.lower() for target in categories)
            if not matched:
                continue

            cat_images = sorted(
                f for f in cat_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )[:max_per_category]

            for img_path in cat_images:
                safe_cat = cat_dir.name.replace(" ", "_")[:30]
                dest = output_dir / f"dermnet_{safe_cat}_{count:04d}{img_path.suffix.lower()}"
                shutil.copy2(img_path, dest)
                count += 1

            logger.info("  %s: %d images", cat_dir.name, len(cat_images))

    logger.info("Copied %d images from DermNet to %s", count, output_dir)
    return count


def validate_non_dfu(image_dir: Path, mask_dir: Path) -> dict[str, int]:
    """Validate that images and masks are paired correctly.

    Returns:
        Dict with counts of matched, unmatched_images, unmatched_masks.
    """
    from PIL import Image

    img_exts = {".jpg", ".jpeg", ".png"}
    images = {f.stem: f for f in image_dir.iterdir() if f.suffix.lower() in img_exts}
    masks = {f.stem: f for f in mask_dir.iterdir() if f.suffix.lower() in img_exts}

    # Match by stem (wound_main-0001 <-> wound_mask-0001)
    # Handle naming: wound_main-XXXX <-> wound_mask-XXXX
    img_ids = {s.replace("wound_main", ""): s for s in images}
    mask_ids = {s.replace("wound_mask", ""): s for s in masks}

    matched = set(img_ids.keys()) & set(mask_ids.keys())
    unmatched_img = set(img_ids.keys()) - set(mask_ids.keys())
    unmatched_mask = set(mask_ids.keys()) - set(img_ids.keys())

    # Also validate a sample of images can be opened
    corrupt = 0
    for stem in list(matched)[:100]:
        try:
            Image.open(images[img_ids[stem]]).verify()
        except Exception:
            corrupt += 1

    return {
        "matched_pairs": len(matched),
        "unmatched_images": len(unmatched_img),
        "unmatched_masks": len(unmatched_mask),
        "corrupt_sample": corrupt,
    }


def main() -> None:
    """Run non-DFU data collection pipeline."""
    parser = argparse.ArgumentParser(description="Collect non-DFU foot condition images")
    parser.add_argument("--dermnet-dir", type=str, default=None, help="Path to DermNet dataset")
    parser.add_argument("--output-dir", type=str, default="data/raw/non_dfu")
    parser.add_argument("--audit", action="store_true", help="Run CleanVision audit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("collect_non_dfu")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"

    # Count existing images
    img_count = 0
    if images_dir.exists():
        img_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        logger.info("Found %d existing images in %s", img_count, images_dir)

    # Optionally add DermNet images
    if args.dermnet_dir:
        dermnet_path = Path(args.dermnet_dir)
        if dermnet_path.exists():
            dermnet_images_dir = images_dir / "dermnet"
            n = collect_from_dermnet(dermnet_path, dermnet_images_dir)
            logger.info("Added %d DermNet images", n)
        else:
            logger.error("DermNet directory not found: %s", dermnet_path)

    # Validate image-mask pairs
    if images_dir.exists() and masks_dir.exists():
        logger.info("Validating image-mask pairs...")
        validation = validate_non_dfu(images_dir, masks_dir)
        logger.info("Validation: %s", validation)

    # Recount
    if images_dir.exists():
        img_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    mask_count = (
        len(list(masks_dir.glob("*.jpg")) + list(masks_dir.glob("*.png")))
        if masks_dir.exists()
        else 0
    )

    # Run audit
    if args.audit and images_dir.exists() and img_count > 0:
        logger.info("Running CleanVision audit on non-DFU images...")
        auditor = DataQualityAuditor(images_dir, dataset_name="non_dfu")
        auditor.run_audit()
        auditor.print_summary()
        auditor.save_report("data/metadata/quality_report_non_dfu.json")

    # Summary
    print(f"\n{'=' * 60}")
    print("Non-DFU Foot Conditions — Summary")
    print(f"{'=' * 60}")
    print(f"Images: {img_count} in {images_dir}")
    print(f"Masks:  {mask_count} in {masks_dir}")
    if img_count < 2000:
        print(f"\n⚠  Target is 2,000-5,000. Currently at {img_count}.")
    else:
        print("\n✓  Target of 2,000+ met!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
