"""DiaFoot.AI v2 — ITA-Stratified Fairness Audit.

Phase 5, Commit 26: Evaluate model performance across skin tone groups.

Reports ALL metrics per ITA category for both classification and segmentation.
Flags bias concerns when max-min gap exceeds 5%.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ITA_CATEGORIES = ["Very Light", "Light", "Intermediate", "Tan", "Brown", "Dark"]


def load_ita_mapping(ita_csv: str | Path) -> dict[str, str]:
    """Load filename -> ITA category mapping."""
    mapping: dict[str, str] = {}
    csv_path = Path(ita_csv)
    if not csv_path.exists():
        return mapping
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["filename"]] = row.get("category", "Unknown")
    return mapping


def stratified_classification_audit(
    filenames: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ita_mapping: dict[str, str],
) -> dict[str, Any]:
    """Compute classification metrics stratified by ITA category.

    Args:
        filenames: List of image filenames.
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).
        ita_mapping: Filename -> ITA category.

    Returns:
        Dict with per-ITA-group metrics and fairness gaps.
    """
    # Group by ITA category
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"true": [], "pred": []})

    for fname, true, pred in zip(filenames, y_true, y_pred, strict=False):
        ita_cat = ita_mapping.get(fname, "Unknown")
        groups[ita_cat]["true"].append(int(true))
        groups[ita_cat]["pred"].append(int(pred))

    per_group: dict[str, dict[str, float]] = {}
    for cat in ITA_CATEGORIES:
        if cat not in groups or len(groups[cat]["true"]) == 0:
            continue
        true_arr = np.array(groups[cat]["true"])
        pred_arr = np.array(groups[cat]["pred"])
        acc = float((true_arr == pred_arr).mean())
        per_group[cat] = {
            "accuracy": acc,
            "count": len(true_arr),
        }

    # Compute fairness gap
    accuracies = [v["accuracy"] for v in per_group.values()]
    gap = max(accuracies) - min(accuracies) if len(accuracies) >= 2 else 0.0

    return {
        "per_ita_group": per_group,
        "fairness_gap_accuracy": gap,
        "bias_concern": gap > 0.05,
    }


def stratified_segmentation_audit(
    filenames: list[str],
    metrics_per_image: list[dict[str, float]],
    ita_mapping: dict[str, str],
) -> dict[str, Any]:
    """Compute segmentation metrics stratified by ITA category.

    Args:
        filenames: List of image filenames.
        metrics_per_image: Per-image metric dicts (dice, iou, etc.).
        ita_mapping: Filename -> ITA category.

    Returns:
        Dict with per-ITA-group segmentation metrics and fairness gaps.
    """
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)

    for fname, metrics in zip(filenames, metrics_per_image, strict=False):
        ita_cat = ita_mapping.get(fname, "Unknown")
        groups[ita_cat].append(metrics)

    per_group: dict[str, dict[str, float]] = {}
    key_metrics = ["dice", "iou", "hd95", "nsd_2mm", "nsd_5mm"]

    for cat in ITA_CATEGORIES:
        if cat not in groups or len(groups[cat]) == 0:
            continue
        group_metrics: dict[str, float] = {"count": float(len(groups[cat]))}
        for key in key_metrics:
            values = [m[key] for m in groups[cat] if key in m]
            if values:
                group_metrics[f"{key}_mean"] = float(np.mean(values))
                group_metrics[f"{key}_std"] = float(np.std(values))
        per_group[cat] = group_metrics

    # Compute fairness gaps for key metrics
    gaps: dict[str, float] = {}
    for key in ["dice", "iou"]:
        values = [v[f"{key}_mean"] for v in per_group.values() if f"{key}_mean" in v]
        if len(values) >= 2:
            gaps[f"{key}_gap"] = max(values) - min(values)

    bias_concern = any(g > 0.05 for g in gaps.values())

    return {
        "per_ita_group": per_group,
        "fairness_gaps": gaps,
        "bias_concern": bias_concern,
    }


def run_fairness_audit(
    classification_results: dict[str, Any] | None = None,
    segmentation_results: dict[str, Any] | None = None,
    ita_csv: str | Path = "data/metadata/ita_scores.csv",
) -> dict[str, Any]:
    """Run complete fairness audit.

    Args:
        classification_results: Dict with filenames, y_true, y_pred.
        segmentation_results: Dict with filenames, metrics_per_image.
        ita_csv: Path to ITA scores CSV.

    Returns:
        Combined fairness report.
    """
    ita_mapping = load_ita_mapping(ita_csv)
    report: dict[str, Any] = {}

    if classification_results:
        report["classification"] = stratified_classification_audit(
            classification_results["filenames"],
            classification_results["y_true"],
            classification_results["y_pred"],
            ita_mapping,
        )

    if segmentation_results:
        report["segmentation"] = stratified_segmentation_audit(
            segmentation_results["filenames"],
            segmentation_results["metrics_per_image"],
            ita_mapping,
        )

    return report


def print_fairness_report(report: dict[str, Any]) -> None:
    """Print formatted fairness audit results."""
    print(f"\n{'=' * 70}")  # noqa: T201
    print("ITA-Stratified Fairness Audit")  # noqa: T201
    print(f"{'=' * 70}")  # noqa: T201

    if "classification" in report:
        cls = report["classification"]
        print("\n  Classification by Skin Tone:")  # noqa: T201
        for cat, metrics in cls["per_ita_group"].items():
            print(  # noqa: T201
                f"    {cat:15s}: acc={metrics['accuracy']:.4f} (n={metrics['count']})"
            )
        gap = cls["fairness_gap_accuracy"]
        flag = " !! BIAS CONCERN" if cls["bias_concern"] else " (OK)"
        print(f"  Fairness gap: {gap:.4f}{flag}")  # noqa: T201

    if "segmentation" in report:
        seg = report["segmentation"]
        print("\n  Segmentation by Skin Tone:")  # noqa: T201
        for cat, metrics in seg["per_ita_group"].items():
            dice = metrics.get("dice_mean", 0)
            iou = metrics.get("iou_mean", 0)
            n = int(metrics.get("count", 0))
            print(  # noqa: T201
                f"    {cat:15s}: dice={dice:.4f} iou={iou:.4f} (n={n})"
            )
        for metric, gap in seg["fairness_gaps"].items():
            flag = " !! BIAS CONCERN" if gap > 0.05 else " (OK)"
            print(f"  {metric}: {gap:.4f}{flag}")  # noqa: T201

    print(f"{'=' * 70}\n")  # noqa: T201
