"""DiaFoot.AI v2 — Subgroup analysis with bootstrap confidence intervals."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from src.evaluation.external_validation import bootstrap_ci

if TYPE_CHECKING:
    import numpy as np


def classification_subgroup_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: list[str],
    min_count: int = 10,
) -> dict[str, Any]:
    """Compute subgroup classification accuracy with bootstrap CIs."""
    bucket: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        bucket[g].append(i)

    per_group: dict[str, Any] = {}
    for g, idxs in bucket.items():
        if len(idxs) < min_count:
            continue
        correct = (y_true[idxs] == y_pred[idxs]).astype(float)
        ci = bootstrap_ci(correct)
        per_group[g] = {
            "count": len(idxs),
            "accuracy": ci["mean"],
            "accuracy_ci95": ci,
        }

    values = [v["accuracy"] for v in per_group.values()]
    gap = float(max(values) - min(values)) if len(values) >= 2 else 0.0
    return {
        "per_group": per_group,
        "accuracy_gap": gap,
        "bias_concern": gap > 0.05,
    }


def segmentation_subgroup_report(
    dice_values: np.ndarray,
    iou_values: np.ndarray,
    groups: list[str],
    min_count: int = 10,
) -> dict[str, Any]:
    """Compute subgroup segmentation metrics with bootstrap CIs."""
    bucket: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        bucket[g].append(i)

    per_group: dict[str, Any] = {}
    for g, idxs in bucket.items():
        if len(idxs) < min_count:
            continue
        dice_ci = bootstrap_ci(dice_values[idxs])
        iou_ci = bootstrap_ci(iou_values[idxs])
        per_group[g] = {
            "count": len(idxs),
            "dice": dice_ci["mean"],
            "dice_ci95": dice_ci,
            "iou": iou_ci["mean"],
            "iou_ci95": iou_ci,
        }

    dice_list = [v["dice"] for v in per_group.values()]
    iou_list = [v["iou"] for v in per_group.values()]
    return {
        "per_group": per_group,
        "dice_gap": float(max(dice_list) - min(dice_list)) if len(dice_list) >= 2 else 0.0,
        "iou_gap": float(max(iou_list) - min(iou_list)) if len(iou_list) >= 2 else 0.0,
        "bias_concern": (
            (float(max(dice_list) - min(dice_list)) > 0.05 if len(dice_list) >= 2 else False)
            or (float(max(iou_list) - min(iou_list)) > 0.05 if len(iou_list) >= 2 else False)
        ),
    }
