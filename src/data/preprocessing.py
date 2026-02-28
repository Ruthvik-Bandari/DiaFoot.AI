"""DiaFoot.AI v2 — Preprocessing Pipeline.

Phase 1, Commit 7: Standardize all images and masks for training.

Pipeline:
    1. Resize to target size (512x512) with aspect ratio preservation (pad)
    2. CLAHE adaptive histogram equalization
    3. Binarize masks (threshold > 127 -> 255)
    4. Normalize (ImageNet stats for pretrained encoders)
"""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# ImageNet normalization stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_with_padding(
    image: np.ndarray,
    target_size: tuple[int, int] = (512, 512),
    pad_value: int = 0,
) -> np.ndarray:
    """Resize image preserving aspect ratio with padding.

    Args:
        image: Input image (H, W, C) or (H, W).
        target_size: Target (width, height).
        pad_value: Padding pixel value (0 for masks, 0 for images).

    Returns:
        Resized and padded image.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # Compute scale to fit within target
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    if len(image.shape) == 2:
        # Mask: use nearest neighbor
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Pad to target size
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    return padded


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE adaptive histogram equalization.

    Applied to the L channel in LAB color space to preserve color.

    Args:
        image: BGR image (as loaded by cv2).
        clip_limit: CLAHE clip limit.
        grid_size: CLAHE tile grid size.

    Returns:
        Enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def binarize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Binarize a mask image.

    Converts any grayscale mask to clean binary (0 or 255).
    Handles JPG compression artifacts in non-DFU masks.

    Args:
        mask: Grayscale mask array.
        threshold: Pixel value threshold.

    Returns:
        Binary mask (0 or 255).
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary


def preprocess_image(
    image_path: Path,
    output_path: Path,
    target_size: tuple[int, int] = (512, 512),
    apply_clahe_flag: bool = True,
) -> bool:
    """Preprocess a single image.

    Args:
        image_path: Input image path.
        output_path: Output path for preprocessed image.
        target_size: Target dimensions.
        apply_clahe_flag: Whether to apply CLAHE.

    Returns:
        True if successful.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Failed to read: %s", image_path)
            return False

        # Resize with padding
        img = resize_with_padding(img, target_size, pad_value=0)

        # CLAHE
        if apply_clahe_flag:
            img = apply_clahe(img)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        return True

    except Exception as e:
        logger.warning("Preprocessing failed for %s: %s", image_path, e)
        return False


def preprocess_mask(
    mask_path: Path,
    output_path: Path,
    target_size: tuple[int, int] = (512, 512),
) -> bool:
    """Preprocess a single mask.

    Args:
        mask_path: Input mask path.
        output_path: Output path.
        target_size: Target dimensions.

    Returns:
        True if successful.
    """
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning("Failed to read mask: %s", mask_path)
            return False

        # Binarize (fixes JPG compression artifacts)
        mask = binarize_mask(mask)

        # Resize with padding (nearest neighbor for masks)
        mask = resize_with_padding(mask, target_size, pad_value=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), mask)
        return True

    except Exception as e:
        logger.warning("Mask preprocessing failed for %s: %s", mask_path, e)
        return False


def preprocess_dataset(
    image_dir: Path,
    mask_dir: Path | None,
    output_image_dir: Path,
    output_mask_dir: Path | None,
    target_size: tuple[int, int] = (512, 512),
    apply_clahe_flag: bool = True,
) -> dict[str, int]:
    """Preprocess an entire dataset directory.

    Args:
        image_dir: Input image directory.
        mask_dir: Input mask directory (None for healthy images).
        output_image_dir: Output directory for images.
        output_mask_dir: Output directory for masks.
        target_size: Target image size.
        apply_clahe_flag: Whether to apply CLAHE.

    Returns:
        Dict with success/failure counts.
    """
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)

    success, failed = 0, 0
    for img_path in images:
        out_img = output_image_dir / f"{img_path.stem}.png"
        ok = preprocess_image(img_path, out_img, target_size, apply_clahe_flag)

        if ok and mask_dir and output_mask_dir:
            # Find matching mask
            mask_found = False
            for ext in [".png", ".jpg", ".jpeg"]:
                # Handle naming conventions
                for prefix in [
                    img_path.stem,
                    img_path.stem.replace("wound_main", "wound_mask"),
                ]:
                    mask_path = mask_dir / f"{prefix}{ext}"
                    if mask_path.exists():
                        out_mask = output_mask_dir / f"{img_path.stem}.png"
                        preprocess_mask(mask_path, out_mask, target_size)
                        mask_found = True
                        break
                if mask_found:
                    break

            if not mask_found and output_mask_dir:
                # Create empty mask
                empty = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
                out_mask = output_mask_dir / f"{img_path.stem}.png"
                out_mask.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_mask), empty)

        if ok:
            success += 1
        else:
            failed += 1

    return {"success": success, "failed": failed}
