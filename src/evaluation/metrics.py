"""DiaFoot.AI v2 — Segmentation Metrics.

Phase 4, Commit 20: Dice, IoU, HD95, NSD, ASSD + clinical metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient.

    Args:
        pred: Binary prediction mask (H, W).
        target: Binary ground truth mask (H, W).
        smooth: Smoothing to avoid division by zero.

    Returns:
        Dice score between 0 and 1.
    """
    pred_flat = pred.astype(bool).flatten()
    target_flat = target.astype(bool).flatten()
    intersection = (pred_flat & target_flat).sum()
    return float((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union (Jaccard Index).

    Args:
        pred: Binary prediction mask (H, W).
        target: Binary ground truth mask (H, W).
        smooth: Smoothing factor.

    Returns:
        IoU score between 0 and 1.
    """
    pred_flat = pred.astype(bool).flatten()
    target_flat = target.astype(bool).flatten()
    intersection = (pred_flat & target_flat).sum()
    union = (pred_flat | target_flat).sum()
    return float((intersection + smooth) / (union + smooth))


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff Distance.

    Measures the boundary quality of segmentation.

    Args:
        pred: Binary prediction mask (H, W).
        target: Binary ground truth mask (H, W).

    Returns:
        HD95 in pixels. Lower is better.
    """
    from scipy.ndimage import distance_transform_edt

    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    # Handle edge cases
    if not pred_bool.any() and not target_bool.any():
        return 0.0
    if not pred_bool.any() or not target_bool.any():
        return float(max(pred.shape))

    # Distance from pred boundary to nearest target boundary
    pred_boundary = pred_bool ^ _erode(pred_bool)
    target_boundary = target_bool ^ _erode(target_bool)

    if not pred_boundary.any() or not target_boundary.any():
        return float(max(pred.shape))

    dt_target = distance_transform_edt(~target_boundary)
    dt_pred = distance_transform_edt(~pred_boundary)

    dist_pred_to_target = dt_target[pred_boundary]
    dist_target_to_pred = dt_pred[target_boundary]

    all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
    return float(np.percentile(all_distances, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Simple erosion by 1 pixel."""
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, iterations=1)


def surface_dice(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance_mm: float = 2.0,
    pixel_spacing: float = 1.0,
) -> float:
    """Compute Normalized Surface Dice (NSD).

    Measures what fraction of boundary points are within tolerance distance.

    Args:
        pred: Binary prediction mask.
        target: Binary ground truth mask.
        tolerance_mm: Tolerance in mm.
        pixel_spacing: mm per pixel.

    Returns:
        NSD score between 0 and 1.
    """
    from scipy.ndimage import distance_transform_edt

    tolerance_px = tolerance_mm / pixel_spacing
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    if not pred_bool.any() and not target_bool.any():
        return 1.0
    if not pred_bool.any() or not target_bool.any():
        return 0.0

    pred_boundary = pred_bool ^ _erode(pred_bool)
    target_boundary = target_bool ^ _erode(target_bool)

    if not pred_boundary.any() or not target_boundary.any():
        return 0.0

    dt_target = distance_transform_edt(~target_boundary)
    dt_pred = distance_transform_edt(~pred_boundary)

    pred_within = (dt_target[pred_boundary] <= tolerance_px).sum()
    target_within = (dt_pred[target_boundary] <= tolerance_px).sum()

    total_boundary = pred_boundary.sum() + target_boundary.sum()
    return float((pred_within + target_within) / max(1, total_boundary))


def wound_area_mm2(
    mask: np.ndarray,
    pixel_spacing_mm: float = 0.5,
) -> float:
    """Estimate wound area in mm squared.

    Args:
        mask: Binary wound mask.
        pixel_spacing_mm: Physical size of one pixel in mm.

    Returns:
        Wound area in mm squared.
    """
    wound_pixels = mask.astype(bool).sum()
    return float(wound_pixels * pixel_spacing_mm * pixel_spacing_mm)


def compute_segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    pixel_spacing_mm: float = 0.5,
) -> dict[str, float]:
    """Compute all segmentation metrics for a single image.

    Args:
        pred: Binary prediction (H, W).
        target: Binary ground truth (H, W).
        pixel_spacing_mm: Physical pixel size.

    Returns:
        Dict with all metrics.
    """
    metrics: dict[str, float] = {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
    }

    # Only compute boundary metrics if both masks have content
    if pred.astype(bool).any() and target.astype(bool).any():
        metrics["hd95"] = hausdorff_distance_95(pred, target)
        metrics["nsd_2mm"] = surface_dice(
            pred, target, tolerance_mm=2.0, pixel_spacing=pixel_spacing_mm
        )
        metrics["nsd_5mm"] = surface_dice(
            pred, target, tolerance_mm=5.0, pixel_spacing=pixel_spacing_mm
        )
    else:
        metrics["hd95"] = 0.0 if not target.astype(bool).any() else float(max(pred.shape))
        metrics["nsd_2mm"] = 1.0 if not target.astype(bool).any() else 0.0
        metrics["nsd_5mm"] = 1.0 if not target.astype(bool).any() else 0.0

    # Clinical metrics
    metrics["wound_area_mm2"] = wound_area_mm2(pred, pixel_spacing_mm)
    metrics["wound_area_gt_mm2"] = wound_area_mm2(target, pixel_spacing_mm)

    return metrics


def aggregate_metrics(
    all_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    """Aggregate per-image metrics into summary statistics.

    Args:
        all_metrics: List of per-image metric dicts.

    Returns:
        Dict with mean, std, median for each metric.
    """
    if not all_metrics:
        return {}

    keys = all_metrics[0].keys()
    summary: dict[str, Any] = {}

    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return summary


def print_segmentation_report(summary: dict[str, Any]) -> None:
    """Print formatted segmentation results."""
    print(f"\n{'=' * 60}")  # noqa: T201
    print("Segmentation Results")  # noqa: T201
    print(f"{'=' * 60}")  # noqa: T201
    for metric, stats in summary.items():
        if isinstance(stats, dict) and "mean" in stats:
            print(  # noqa: T201
                f"  {metric:20s}: {stats['mean']:.4f} "
                f"(+/- {stats['std']:.4f}) "
                f"[{stats['min']:.4f}, {stats['max']:.4f}]"
            )
    print(f"{'=' * 60}\n")  # noqa: T201
