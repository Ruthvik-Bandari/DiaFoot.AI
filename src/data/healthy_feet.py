"""DiaFoot.AI v2 — Healthy Foot Data Collection & Processing.

Phase 1, Commit 3: Collect and curate healthy foot images (negative class).

Sources:
    1. Kaggle DFU dataset (laithjj/diabetic-foot-ulcer-dfu) — 543 normal foot patches
    2. Mendeley Lower Limb & Feet Wound Dataset (hsj38fwnvr) — normal feet images, CC BY 4.0
    3. Manual curation from web (optional, requires human review)

All healthy images get empty (all-zero) segmentation masks.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def create_empty_masks(
    image_dir: str | Path,
    mask_dir: str | Path,
    image_size: tuple[int, int] = (512, 512),
) -> int:
    """Create all-zero segmentation masks for healthy foot images.

    For healthy feet, the ground truth mask is entirely background (no wound).

    Args:
        image_dir: Directory containing healthy foot images.
        mask_dir: Output directory for empty masks.
        image_size: If set, resize mask to this size. If None, match source image.

    Returns:
        Number of masks created.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_exts]

    count = 0
    for img_path in sorted(images):
        try:
            if image_size:
                w, h = image_size
            else:
                with Image.open(img_path) as img:
                    w, h = img.size

            # Create all-zero mask (no wound)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask_path = mask_dir / f"{img_path.stem}.png"
            Image.fromarray(mask).save(mask_path)
            count += 1
        except Exception as e:
            logger.warning("Failed to create mask for %s: %s", img_path.name, e)

    logger.info("Created %d empty masks in %s", count, mask_dir)
    return count


def organize_kaggle_dfu_normal(
    kaggle_dir: str | Path,
    output_dir: str | Path,
) -> int:
    """Extract normal/healthy foot images from Kaggle DFU dataset.

    The Kaggle dataset (laithjj/diabetic-foot-ulcer-dfu) has two folders:
    - Normal (healthy feet) — ~543 images
    - Abnormal (DFU) — ~512 images

    We only take the Normal folder.

    Args:
        kaggle_dir: Path to extracted Kaggle dataset root.
        output_dir: Where to copy healthy images.

    Returns:
        Number of images copied.
    """
    kaggle_dir = Path(kaggle_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The Kaggle dataset may have various structures
    normal_candidates = [
        kaggle_dir / "Normal",
        kaggle_dir / "normal",
        kaggle_dir / "Normal Skin" / "Normal Skin",
        kaggle_dir / "Original Images" / "Normal",
    ]

    normal_dir = None
    for candidate in normal_candidates:
        if candidate.exists():
            normal_dir = candidate
            break

    if normal_dir is None:
        # Fallback: search for any directory with "normal" in name
        for d in kaggle_dir.rglob("*"):
            if d.is_dir() and "normal" in d.name.lower():
                normal_dir = d
                break

    if normal_dir is None:
        logger.error("Could not find Normal folder in %s", kaggle_dir)
        logger.info("Directory contents: %s", list(kaggle_dir.iterdir()))
        return 0

    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    images = [f for f in normal_dir.rglob("*") if f.suffix.lower() in image_exts]

    count = 0
    for img_path in sorted(images):
        dest = output_dir / f"kaggle_normal_{count:04d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dest)
        count += 1

    logger.info("Copied %d normal images from Kaggle to %s", count, output_dir)
    return count


def organize_mendeley_normal(
    mendeley_dir: str | Path,
    output_dir: str | Path,
) -> int:
    """Extract normal/healthy foot images from Mendeley dataset.

    Mendeley dataset (hsj38fwnvr): Lower Limb and Feet Wound Image Dataset
    Contains a "Normal" or "normal_feet" folder with healthy images.

    Args:
        mendeley_dir: Path to extracted Mendeley dataset root.
        output_dir: Where to copy healthy images.

    Returns:
        Number of images copied.
    """
    mendeley_dir = Path(mendeley_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for normal/healthy subdirectory
    normal_dir = None
    for d in mendeley_dir.rglob("*"):
        if d.is_dir() and any(
            keyword in d.name.lower() for keyword in ["normal", "healthy", "control"]
        ):
            normal_dir = d
            break

    if normal_dir is None:
        logger.error("Could not find Normal/Healthy folder in %s", mendeley_dir)
        return 0

    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    images = [f for f in normal_dir.rglob("*") if f.suffix.lower() in image_exts]

    count = 0
    for img_path in sorted(images):
        dest = output_dir / f"mendeley_normal_{count:04d}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dest)
        count += 1

    logger.info("Copied %d normal images from Mendeley to %s", count, output_dir)
    return count


def validate_healthy_images(
    image_dir: str | Path,
    min_size: int = 64,
) -> dict[str, list[str]]:
    """Basic validation of healthy foot images.

    Checks:
    - Image can be opened without error
    - Minimum resolution met
    - Is RGB (not grayscale)

    Args:
        image_dir: Directory to validate.
        min_size: Minimum dimension in pixels.

    Returns:
        Dict with "valid", "corrupt", "too_small", "grayscale" lists of filenames.
    """
    image_dir = Path(image_dir)
    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    results: dict[str, list[str]] = {"valid": [], "corrupt": [], "too_small": [], "grayscale": []}

    for f in sorted(image_dir.iterdir()):
        if f.suffix.lower() not in image_exts:
            continue
        try:
            with Image.open(f) as img:
                w, h = img.size
                if w < min_size or h < min_size:
                    results["too_small"].append(f.name)
                elif img.mode != "RGB":
                    results["grayscale"].append(f.name)
                else:
                    results["valid"].append(f.name)
        except Exception:
            results["corrupt"].append(f.name)

    logger.info(
        "Validation: %d valid, %d corrupt, %d too_small, %d grayscale",
        len(results["valid"]),
        len(results["corrupt"]),
        len(results["too_small"]),
        len(results["grayscale"]),
    )
    return results


def run_quality_audit_on_healthy(image_dir: str | Path, output_path: str | Path) -> None:
    """Run CleanVision audit on the healthy foot images."""
    from src.data.cleaning import DataQualityAuditor

    auditor = DataQualityAuditor(image_dir, dataset_name="healthy_feet")
    auditor.run_audit()
    auditor.print_summary()
    auditor.save_report(output_path)
