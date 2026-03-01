"""DiaFoot.AI v2 — Boundary Refinement Post-Processing.

Phase 2, Commit 12: Morphological operations and connected component
filtering to clean up segmentation predictions.
"""

from __future__ import annotations

import cv2
import numpy as np


def morphological_smooth(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """Smooth mask boundaries using morphological operations.

    Applies closing (fill small holes) then opening (remove small noise).

    Args:
        mask: Binary mask (H, W), values 0 or 255.
        kernel_size: Size of morphological kernel.
        iterations: Number of iterations.

    Returns:
        Smoothed binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Close: fill small holes in wound region
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    # Open: remove small noise outside wound
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened


def connected_component_filter(
    mask: np.ndarray,
    min_area: int = 100,
    keep_largest_n: int | None = None,
) -> np.ndarray:
    """Filter small connected components from mask.

    Removes noise by keeping only components above a minimum area,
    and optionally only the N largest.

    Args:
        mask: Binary mask (H, W).
        min_area: Minimum component area in pixels.
        keep_largest_n: If set, keep only this many largest components.

    Returns:
        Filtered binary mask.
    """
    # Ensure binary
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    # stats[:, cv2.CC_STAT_AREA] gives area of each component
    # Label 0 is background
    output = np.zeros_like(binary)

    if num_labels <= 1:
        return output

    # Get areas (skip background at index 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    label_indices = np.arange(1, num_labels)

    # Filter by minimum area
    valid = areas >= min_area
    label_indices = label_indices[valid]
    areas = areas[valid]

    # Keep only largest N
    if keep_largest_n and len(label_indices) > keep_largest_n:
        top_indices = np.argsort(areas)[-keep_largest_n:]
        label_indices = label_indices[top_indices]

    # Build output mask
    for label_idx in label_indices:
        output[labels == label_idx] = 255

    return output


def refine_prediction(
    mask: np.ndarray,
    smooth_kernel: int = 5,
    min_component_area: int = 100,
    keep_largest_n: int | None = 3,
) -> np.ndarray:
    """Full boundary refinement pipeline.

    Args:
        mask: Raw binary prediction mask (H, W).
        smooth_kernel: Morphological kernel size.
        min_component_area: Min area to keep.
        keep_largest_n: Keep only N largest components.

    Returns:
        Refined binary mask.
    """
    # Ensure uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255

    # Step 1: Morphological smoothing
    smoothed = morphological_smooth(mask, kernel_size=smooth_kernel)

    # Step 2: Connected component filtering
    filtered = connected_component_filter(
        smoothed,
        min_area=min_component_area,
        keep_largest_n=keep_largest_n,
    )

    return filtered
