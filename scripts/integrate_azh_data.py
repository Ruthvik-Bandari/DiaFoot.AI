"""DiaFoot.AI v2 — Clean & Integrate AZH GitHub Wound Data.

Integrates the AZH wound care center dataset (1,109 images) into
the existing processed data pipeline. Handles:
  1. Image integrity check
  2. Mask validation (binary, alignment, coverage)
  3. Deduplication against existing FUSeg data
  4. Preprocessing (resize 512x512, CLAHE, mask binarization)
  5. Adds to processed/dfu/ for segmentation training

Usage:
    python scripts/integrate_azh_data.py --verbose
    python scripts/integrate_azh_data.py --dry-run --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("azh_integrate")

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SIZE = 512
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
MASK_THRESHOLD = 127
MIN_IMAGE_SIZE = 64


# ── Preprocessing ────────────────────────────────────────────────────────────
def resize_with_padding(
    img: np.ndarray,
    target_size: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize preserving aspect ratio, pad with black."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    if len(img.shape) == 3:
        canvas = np.zeros((target_size, target_size, img.shape[2]), dtype=img.dtype)
    else:
        canvas = np.zeros((target_size, target_size), dtype=img.dtype)

    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Resize to 512x512 + CLAHE on L channel."""
    img = resize_with_padding(img, TARGET_SIZE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    l_enhanced = clahe.apply(l_ch)
    enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Resize to 512x512 (nearest) + binarize."""
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = resize_with_padding(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


# ── Validation ───────────────────────────────────────────────────────────────
def validate_pair(img_path: Path, mask_path: Path) -> dict:
    """Validate a single image-mask pair. Returns issue dict."""
    issues = []

    # Image integrity
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return {"valid": False, "issues": ["image_unreadable"]}
        h, w = img.shape[:2]
        if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
            issues.append(f"too_small:{w}x{h}")
    except Exception as e:
        return {"valid": False, "issues": [f"image_error:{e}"]}

    # Mask integrity
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return {"valid": False, "issues": ["mask_unreadable"]}
        mh, mw = mask.shape[:2]
    except Exception as e:
        return {"valid": False, "issues": [f"mask_error:{e}"]}

    # Dimension check
    if (h, w) != (mh, mw):
        issues.append(f"size_mismatch:img={w}x{h},mask={mw}x{mh}")

    # Mask content check
    unique = np.unique(mask)
    if len(unique) == 1 and unique[0] == 0:
        issues.append("all_zero_mask")
    elif len(unique) == 1:
        issues.append("all_nonzero_mask")

    # Coverage
    binary = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask
    coverage = binary.sum() / binary.size
    if coverage > 0.80:
        issues.append(f"high_coverage:{coverage:.1%}")
    elif 0 < coverage < 0.001:
        issues.append(f"tiny_wound:{coverage:.4%}")

    return {
        "valid": "all_nonzero_mask" not in " ".join(issues),
        "issues": issues,
        "size": f"{w}x{h}",
        "coverage": round(float(coverage), 4),
    }


# ── Deduplication ────────────────────────────────────────────────────────────
def compute_dhash(img_path: Path, hash_size: int = 16) -> str | None:
    """Compute difference hash for near-duplicate detection."""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        resized = cv2.resize(img, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return "".join(str(int(b)) for b in diff.flatten())
    except Exception:
        return None


def build_existing_hashes(processed_dfu_dir: Path) -> set[str]:
    """Build hash set from existing processed DFU images."""
    img_dir = processed_dfu_dir / "images"
    if not img_dir.exists():
        return set()

    hashes = set()
    for img_path in img_dir.glob("*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            h = compute_dhash(img_path)
            if h:
                hashes.add(h)

    logger.info("Built hash index: %d existing DFU images", len(hashes))
    return hashes


def is_duplicate(img_path: Path, existing_hashes: set[str], threshold: int = 5) -> bool:
    """Check if image is a near-duplicate of existing data."""
    h = compute_dhash(img_path)
    if h is None:
        return False

    # Exact match
    if h in existing_hashes:
        return True

    # Near-duplicate (hamming distance ≤ threshold)
    for eh in existing_hashes:
        dist = sum(c1 != c2 for c1, c2 in zip(h, eh, strict=False))
        if dist <= threshold:
            return True

    return False


# ── Main pipeline ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Clean & integrate AZH GitHub data")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip dedup check (faster)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir)
    azh_dir = data_dir / "raw" / "azh_github"
    processed_dfu = data_dir / "processed" / "dfu"
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if not azh_dir.exists():
        logger.error("AZH directory not found: %s", azh_dir)
        return

    # ── Step 1: Discover image-mask pairs ────────────────────────────────
    print("═" * 60)
    print("  Step 1: Discovering image-mask pairs")
    print("═" * 60)

    pairs = []
    for split in ["train", "test"]:
        img_dir = azh_dir / split / "images"
        mask_dir = azh_dir / split / "labels"

        if not img_dir.exists():
            logger.warning("Missing: %s", img_dir)
            continue

        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                pairs.append((img_path, mask_path, split))
            else:
                logger.debug("Orphan (no mask): %s", img_path.name)

    logger.info("Found %d image-mask pairs", len(pairs))

    # ── Step 2: Validate all pairs ───────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Step 2: Validating image-mask pairs")
    print("═" * 60)

    valid_pairs = []
    issues_summary: dict[str, int] = {}
    coverages = []

    for img_path, mask_path, split in pairs:
        result = validate_pair(img_path, mask_path)

        for issue in result["issues"]:
            key = issue.split(":")[0]
            issues_summary[key] = issues_summary.get(key, 0) + 1

        if result["valid"]:
            valid_pairs.append((img_path, mask_path, split))
            if "coverage" in result:
                coverages.append(result["coverage"])
        else:
            logger.debug("Invalid: %s — %s", img_path.name, result["issues"])

    logger.info("Valid: %d / %d", len(valid_pairs), len(pairs))
    if issues_summary:
        logger.info("Issues: %s", issues_summary)
    if coverages:
        cov = np.array(coverages)
        logger.info(
            "Coverage: mean=%.1f%%, median=%.1f%%, range=[%.2f%%, %.1f%%]",
            cov.mean() * 100,
            np.median(cov) * 100,
            cov.min() * 100,
            cov.max() * 100,
        )

    # ── Step 3: Deduplication against existing FUSeg ─────────────────────
    if not args.skip_dedup:
        print("\n" + "═" * 60)
        print("  Step 3: Deduplication check against existing DFU data")
        print("═" * 60)

        existing_hashes = build_existing_hashes(processed_dfu)
        unique_pairs = []
        dup_count = 0

        for img_path, mask_path, split in valid_pairs:
            if is_duplicate(img_path, existing_hashes):
                dup_count += 1
                logger.debug("Duplicate: %s", img_path.name)
            else:
                unique_pairs.append((img_path, mask_path, split))

        logger.info(
            "Dedup: %d unique, %d duplicates removed",
            len(unique_pairs),
            dup_count,
        )
        valid_pairs = unique_pairs
    else:
        logger.info("Skipping deduplication (--skip-dedup)")

    # ── Step 4: Preprocess and save ──────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Step 4: Preprocessing and saving to processed/dfu/")
    print("═" * 60)

    img_out = processed_dfu / "images"
    mask_out = processed_dfu / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    saved = 0
    errors = 0

    for img_path, mask_path, split in valid_pairs:
        try:
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                errors += 1
                continue

            img_proc = preprocess_image(img)
            mask_proc = preprocess_mask(mask)

            # Save with source prefix to avoid collisions
            out_name = f"azh_{split}_{img_path.stem}.png"

            if not args.dry_run:
                cv2.imwrite(str(img_out / out_name), img_proc)
                cv2.imwrite(str(mask_out / out_name), mask_proc)

            saved += 1
        except Exception as e:
            logger.warning("Error processing %s: %s", img_path.name, e)
            errors += 1

    logger.info("Saved: %d, Errors: %d", saved, errors)

    # ── Step 5: Report ───────────────────────────────────────────────────
    # Count final state
    final_img_count = len(list(img_out.glob("*"))) if not args.dry_run else "N/A (dry run)"
    final_mask_count = len(list(mask_out.glob("*"))) if not args.dry_run else "N/A (dry run)"

    report = {
        "source": "azh_github",
        "total_pairs_found": len(pairs),
        "valid_pairs": len(valid_pairs) + (dup_count if not args.skip_dedup else 0),
        "duplicates_removed": dup_count if not args.skip_dedup else "skipped",
        "saved": saved,
        "errors": errors,
        "issues_summary": issues_summary,
        "coverage_stats": {
            "mean": round(float(np.mean(coverages)), 4) if coverages else None,
            "median": round(float(np.median(coverages)), 4) if coverages else None,
            "min": round(float(np.min(coverages)), 6) if coverages else None,
            "max": round(float(np.max(coverages)), 4) if coverages else None,
        },
        "final_dfu_images": final_img_count,
        "final_dfu_masks": final_mask_count,
    }

    report_path = metadata_dir / "azh_integration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "═" * 60)
    print("  AZH INTEGRATION COMPLETE")
    print("═" * 60)
    print(f"  Pairs found:       {len(pairs)}")
    print(f"  Valid:             {len(valid_pairs) + (dup_count if not args.skip_dedup else 0)}")
    if not args.skip_dedup:
        print(f"  Duplicates:        {dup_count}")
    print(f"  Saved to DFU:      {saved}")
    print(f"  Errors:            {errors}")
    if not args.dry_run:
        print(f"  Total DFU images:  {final_img_count}")
        print(f"  Total DFU masks:   {final_mask_count}")
    print(f"  Report: {report_path}")
    print("═" * 60)


if __name__ == "__main__":
    main()
