"""DiaFoot.AI v2 — Calibration Analysis.

Phase 5, Commit 25: ECE, reliability diagrams, temperature scaling.

A well-calibrated model's confidence matches its accuracy:
if the model says 80% confident, it should be correct 80% of the time.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 2D logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def multiclass_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute multiclass Brier score.

    Lower is better.
    """
    n, c = probs.shape
    one_hot = np.zeros((n, c), dtype=np.float64)
    one_hot[np.arange(n), labels.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15,
) -> tuple[float, dict[str, Any]]:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and
    actual accuracy, weighted by the number of samples in each bin.

    Args:
        confidences: Model confidence scores (N,), values in [0, 1].
        accuracies: Whether predictions were correct (N,), binary.
        num_bins: Number of confidence bins.

    Returns:
        Tuple of (ECE value, bin details for reliability diagram).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_details: dict[str, list[float]] = {
        "bin_centers": [],
        "bin_accuracies": [],
        "bin_confidences": [],
        "bin_counts": [],
    }

    n_total = len(confidences)

    for i in range(num_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        count = in_bin.sum()

        bin_details["bin_centers"].append(float((lower + upper) / 2))
        bin_details["bin_counts"].append(int(count))

        if count > 0:
            bin_acc = float(accuracies[in_bin].mean())
            bin_conf = float(confidences[in_bin].mean())
            ece += (count / n_total) * abs(bin_acc - bin_conf)
            bin_details["bin_accuracies"].append(bin_acc)
            bin_details["bin_confidences"].append(bin_conf)
        else:
            bin_details["bin_accuracies"].append(0.0)
            bin_details["bin_confidences"].append(0.0)

    return float(ece), bin_details


def temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 100,
) -> float:
    """Find optimal temperature for calibration via grid search.

    Temperature scaling divides logits by T before softmax,
    reducing overconfidence when T > 1.

    Args:
        logits: Raw model logits (N, C).
        labels: Ground truth class indices (N,).
        lr: Learning rate (unused, grid search instead).
        max_iter: Max iterations (unused).

    Returns:
        Optimal temperature value.
    """
    best_t = 1.0
    best_ece = float("inf")

    for t in np.arange(0.1, 5.1, 0.1):
        scaled = logits / t
        probs = _softmax(scaled)

        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == labels).astype(float)

        ece, _ = expected_calibration_error(confidences, correct)
        if ece < best_ece:
            best_ece = ece
            best_t = float(t)

    logger.info(
        "Temperature scaling: selected T=%.2f with ECE=%.4f",
        best_t,
        best_ece,
    )
    return best_t


def tune_defer_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
    min_coverage: float = 0.6,
) -> dict[str, Any]:
    """Tune defer threshold for abstention policy.

    Policy:
      - defer if max_confidence < threshold
      - evaluate quality only on covered (non-deferred) predictions

    Objective:
      - maximize covered accuracy subject to coverage >= min_coverage
      - tie-break by larger coverage
    """
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.95, 27)

    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    sweep: list[dict[str, float]] = []
    best: dict[str, float] | None = None

    for thr in thresholds:
        keep = confidences >= float(thr)
        coverage = float(keep.mean())
        deferred = int((~keep).sum())
        kept = int(keep.sum())

        if kept > 0:
            acc_kept = float((preds[keep] == labels[keep]).mean())
        else:
            acc_kept = 0.0

        row = {
            "threshold": float(thr),
            "coverage": coverage,
            "defer_rate": 1.0 - coverage,
            "num_deferred": float(deferred),
            "num_kept": float(kept),
            "accuracy_kept": acc_kept,
        }
        sweep.append(row)

        if coverage < min_coverage:
            continue
        if best is None:
            best = row
            continue
        if row["accuracy_kept"] > best["accuracy_kept"] + 1e-12:
            best = row
        elif abs(row["accuracy_kept"] - best["accuracy_kept"]) <= 1e-12:
            if row["coverage"] > best["coverage"]:
                best = row

    # If no threshold satisfies min coverage, fall back to highest coverage.
    if best is None:
        best = max(sweep, key=lambda r: (r["coverage"], r["accuracy_kept"]))

    return {
        "min_coverage": float(min_coverage),
        "recommended_threshold": float(best["threshold"]),
        "recommended": best,
        "sweep": sweep,
    }


def compute_segmentation_calibration(
    pred_probs: np.ndarray,
    ground_truth: np.ndarray,
    num_bins: int = 10,
) -> tuple[float, dict[str, Any]]:
    """Compute calibration for segmentation (pixel-level).

    Args:
        pred_probs: Predicted probability map (H, W), values in [0, 1].
        ground_truth: Binary ground truth mask (H, W).
        num_bins: Number of bins.

    Returns:
        Tuple of (pixel-ECE, bin details).
    """
    confidences = np.maximum(pred_probs, 1 - pred_probs).flatten()
    predictions = (pred_probs > 0.5).astype(float).flatten()
    gt_flat = ground_truth.astype(float).flatten()
    correct = (predictions == gt_flat).astype(float)

    return expected_calibration_error(confidences, correct, num_bins)


def compute_calibration_report(
    classification_logits: np.ndarray | None = None,
    classification_labels: np.ndarray | None = None,
    seg_pred_probs: list[np.ndarray] | None = None,
    seg_ground_truths: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute full calibration report.

    Args:
        classification_logits: Raw classifier logits (N, C).
        classification_labels: Ground truth labels (N,).
        seg_pred_probs: List of segmentation probability maps.
        seg_ground_truths: List of ground truth masks.

    Returns:
        Calibration report dict.
    """
    report: dict[str, Any] = {}

    # Classification calibration
    if classification_logits is not None and classification_labels is not None:
        # Before temperature scaling
        probs = _softmax(classification_logits)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == classification_labels).astype(float)

        ece_before, bins_before = expected_calibration_error(confidences, correct)
        brier_before = multiclass_brier_score(probs, classification_labels)

        # Temperature scaling
        opt_t = temperature_scaling(classification_logits, classification_labels)

        # After temperature scaling
        scaled = classification_logits / opt_t
        probs_after = _softmax(scaled)
        conf_after = probs_after.max(axis=1)
        pred_after = probs_after.argmax(axis=1)
        correct_after = (pred_after == classification_labels).astype(float)

        ece_after, bins_after = expected_calibration_error(conf_after, correct_after)
        brier_after = multiclass_brier_score(probs_after, classification_labels)
        defer_tuning = tune_defer_threshold(probs_after, classification_labels)

        report["classification"] = {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "optimal_temperature": opt_t,
            "defer_tuning": defer_tuning,
            "bins_before": bins_before,
            "bins_after": bins_after,
        }

    # Segmentation calibration
    if seg_pred_probs and seg_ground_truths:
        all_ece = []
        for pred, gt in zip(seg_pred_probs, seg_ground_truths, strict=False):
            ece, _ = compute_segmentation_calibration(pred, gt)
            all_ece.append(ece)

        report["segmentation"] = {
            "pixel_ece_mean": float(np.mean(all_ece)),
            "pixel_ece_std": float(np.std(all_ece)),
            "pixel_ece_median": float(np.median(all_ece)),
        }

    return report


def print_calibration_report(report: dict[str, Any]) -> None:
    """Print formatted calibration results."""
    print(f"\n{'=' * 60}")  # noqa: T201
    print("Calibration Analysis")  # noqa: T201
    print(f"{'=' * 60}")  # noqa: T201

    if "classification" in report:
        cls = report["classification"]
        print("  Classification:")  # noqa: T201
        print(f"    ECE (before):     {cls['ece_before']:.4f}")  # noqa: T201
        print(f"    ECE (after):      {cls['ece_after']:.4f}")  # noqa: T201
        print(f"    Brier (before):   {cls['brier_before']:.4f}")  # noqa: T201
        print(f"    Brier (after):    {cls['brier_after']:.4f}")  # noqa: T201
        print(f"    Temperature:      {cls['optimal_temperature']:.2f}")  # noqa: T201
        defer = cls.get("defer_tuning", {})
        if defer:
            rec = defer.get("recommended", {})
            print(  # noqa: T201
                "    Defer threshold: "
                f"{defer.get('recommended_threshold', 0):.2f} "
                f"(coverage={rec.get('coverage', 0):.2%}, "
                f"acc_kept={rec.get('accuracy_kept', 0):.2%})"
            )

    if "segmentation" in report:
        seg = report["segmentation"]
        print("  Segmentation (pixel-level):")  # noqa: T201
        print(f"    ECE mean:   {seg['pixel_ece_mean']:.4f}")  # noqa: T201
        print(f"    ECE std:    {seg['pixel_ece_std']:.4f}")  # noqa: T201

    print(f"{'=' * 60}\n")  # noqa: T201
