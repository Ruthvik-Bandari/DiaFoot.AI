"""DiaFoot.AI v2 — Failure atlas helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def classify_segmentation_failure(
    pred: np.ndarray,
    target: np.ndarray,
    dice: float,
) -> str:
    """Assign a simple failure category for segmentation output."""
    pred_any = bool(pred.astype(bool).any())
    gt_any = bool(target.astype(bool).any())

    if not gt_any and pred_any:
        return "false_positive_empty_gt"
    if gt_any and not pred_any:
        return "missed_lesion_false_negative"
    if gt_any and pred_any and dice < 0.4:
        return "poor_overlap"
    if gt_any and pred_any and dice < 0.7:
        return "boundary_error"
    return "acceptable"


def summarize_failure_types(types: list[str]) -> dict[str, Any]:
    """Summarize counts per failure type."""
    out: dict[str, int] = {}
    for t in types:
        out[t] = out.get(t, 0) + 1
    return {
        "total": len(types),
        "counts": out,
    }
