"""DiaFoot.AI v2 — Classification Metrics.

Phase 4, Commit 19: Comprehensive classification evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np  # noqa: TC002
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES = ["Healthy", "Non-DFU", "DFU"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).
        y_prob: Prediction probabilities (N, C) for AUROC.
        class_names: Class name mapping.

    Returns:
        Dict with all metrics.
    """
    class_names = class_names or CLASS_NAMES
    num_classes = len(class_names)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        binary_true = (y_true == i).astype(int)
        binary_pred = (y_pred == i).astype(int)
        per_class[name] = {
            "precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
            "recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
            "f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
            "support": int(binary_true.sum()),
        }
    metrics["per_class"] = per_class

    # Sensitivity (recall for DFU class — most critical)
    dfu_idx = class_names.index("DFU") if "DFU" in class_names else num_classes - 1
    metrics["dfu_sensitivity"] = per_class[class_names[dfu_idx]]["recall"]

    # Specificity (healthy correctly identified)
    if "Healthy" in class_names:
        healthy_idx = class_names.index("Healthy")
        healthy_true = y_true == healthy_idx
        healthy_pred = y_pred == healthy_idx
        tn = int(((~healthy_true) & (~healthy_pred)).sum())
        fp = int(((~healthy_true) & healthy_pred).sum())
        metrics["healthy_specificity"] = tn / max(1, tn + fp)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    # AUROC (if probabilities provided)
    if y_prob is not None and y_prob.shape[1] == num_classes:
        try:
            metrics["auroc_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
            # Per-class AUROC
            for i, name in enumerate(class_names):
                binary_true = (y_true == i).astype(int)
                if binary_true.sum() > 0 and binary_true.sum() < len(binary_true):
                    per_class[name]["auroc"] = float(roc_auc_score(binary_true, y_prob[:, i]))
        except ValueError:
            pass

    # Classification report string
    metrics["report"] = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0,
    )

    return metrics


def print_classification_report(metrics: dict[str, Any]) -> None:
    """Print formatted classification results."""
    print(f"\n{'=' * 60}")  # noqa: T201
    print("Classification Results")  # noqa: T201
    print(f"{'=' * 60}")  # noqa: T201
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")  # noqa: T201
    print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")  # noqa: T201
    print(f"  F1 (weighted):   {metrics['f1_weighted']:.4f}")  # noqa: T201
    print(f"  DFU Sensitivity: {metrics['dfu_sensitivity']:.4f}")  # noqa: T201
    if "auroc_macro" in metrics:
        print(f"  AUROC (macro):   {metrics['auroc_macro']:.4f}")  # noqa: T201
    print(f"\n{metrics['report']}")  # noqa: T201
    print(f"{'=' * 60}\n")  # noqa: T201
