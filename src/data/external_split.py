"""DiaFoot.AI v2 — Shared helpers for external-split builder scripts.

Home for the image/extension helpers shared by
``scripts/build_external_segmentation_split.py`` and
``scripts/build_external_split_from_patches.py``. Extracted so a single
implementation exists instead of copy-pasted duplicates.

NOTE: ``resize_with_padding`` here is intentionally distinct from
``src.data.preprocessing.resize_with_padding``. This version uses
``round()`` for the resized dimensions and a caller-chosen interpolation,
matching the numerics the external-split builders already used to generate
``data/processed_external*``. The preprocessing variant uses ``int()`` and
auto-selects interpolation, so swapping the two would perturb already-built
external artifacts. Keep them separate unless the external splits are rebuilt.
"""

from __future__ import annotations

import cv2
import numpy as np

# Image file extensions the external-split builders accept.
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resize_with_padding(
    img: np.ndarray, target: int = 512, interp: int = cv2.INTER_AREA
) -> np.ndarray:
    """Resize an image to a ``target`` square, preserving aspect ratio with zero padding.

    Args:
        img: Input image, either ``(H, W)`` grayscale or ``(H, W, C)``.
        target: Output side length in pixels (square ``target`` x ``target``).
        interp: OpenCV interpolation flag (e.g. ``cv2.INTER_AREA`` for images,
            ``cv2.INTER_NEAREST`` for masks).

    Returns:
        A ``target`` x ``target`` image (matching the input's channel count and
        dtype), with the resized content centered and the border zero-padded.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size")
    scale = target / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    if img.ndim == 2:
        canvas = np.zeros((target, target), dtype=img.dtype)
    else:
        canvas = np.zeros((target, target, img.shape[2]), dtype=img.dtype)

    yo, xo = (target - nh) // 2, (target - nw) // 2
    canvas[yo : yo + nh, xo : xo + nw] = resized
    return canvas
