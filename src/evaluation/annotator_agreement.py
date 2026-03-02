"""DiaFoot.AI v2 — Inter-Annotator Agreement Analysis.

Phase 4, Commit 21: Compare model predictions against human annotations.

Computes:
- STAPLE consensus from multiple annotations (if available)
- Per-annotator Dice scores
- Model vs human ceiling analysis
- Fleiss' kappa for classification reliability
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_pairwise_dice(
    masks: list[np.ndarray],
) -> np.ndarray:
    """Compute pairwise Dice scores between multiple annotations.

    Args:
        masks: List of K binary masks (H, W) for the same image.

    Returns:
        (K, K) matrix of pairwise Dice scores.
    """
    k = len(masks)
    dice_matrix = np.ones((k, k))

    for i in range(k):
        for j in range(i + 1, k):
            mi = masks[i].astype(bool).flatten()
            mj = masks[j].astype(bool).flatten()
            intersection = (mi & mj).sum()
            total = mi.sum() + mj.sum()
            dice = (2.0 * intersection) / max(total, 1)
            dice_matrix[i, j] = dice
            dice_matrix[j, i] = dice

    return dice_matrix


def compute_majority_vote(masks: list[np.ndarray]) -> np.ndarray:
    """Compute majority vote consensus from multiple annotations.

    Simple alternative to STAPLE when SimpleITK is unavailable.

    Args:
        masks: List of K binary masks (H, W).

    Returns:
        Consensus mask where pixel=1 if majority of annotators agree.
    """
    stacked = np.stack([m.astype(float) for m in masks], axis=0)
    agreement = stacked.mean(axis=0)
    threshold = 0.5
    return (agreement >= threshold).astype(np.uint8)


def staple_consensus(masks: list[np.ndarray]) -> np.ndarray:
    """Compute STAPLE consensus from multiple annotations.

    STAPLE (Simultaneous Truth and Performance Level Estimation)
    estimates the true segmentation from multiple noisy annotations.

    Falls back to majority vote if SimpleITK is not available.

    Args:
        masks: List of K binary masks (H, W).

    Returns:
        STAPLE consensus mask.
    """
    try:
        import SimpleITK as sitk  # noqa: N813

        sitk_images = [sitk.GetImageFromArray(m.astype(np.uint8)) for m in masks]
        staple_filter = sitk.STAPLEImageFilter()
        staple_filter.SetForegroundValue(1)
        result = staple_filter.Execute(sitk_images)
        consensus = sitk.GetArrayFromImage(result)
        return (consensus >= 0.5).astype(np.uint8)

    except ImportError:
        logger.warning("SimpleITK not available, falling back to majority vote")
        return compute_majority_vote(masks)


def model_vs_human_ceiling(
    model_pred: np.ndarray,
    annotator_masks: list[np.ndarray],
) -> dict[str, Any]:
    """Compare model performance against inter-annotator variability.

    Establishes whether the model performs within human-level range.

    Args:
        model_pred: Model's predicted mask (H, W).
        annotator_masks: List of K annotator masks (H, W).

    Returns:
        Dict with model vs annotator comparison metrics.
    """
    from src.evaluation.metrics import dice_score, iou_score

    # Pairwise annotator agreement
    annotator_dice = compute_pairwise_dice(annotator_masks)
    # Extract upper triangle (excluding diagonal)
    k = len(annotator_masks)
    upper_indices = np.triu_indices(k, k=1)
    pairwise_scores = annotator_dice[upper_indices]

    inter_annotator_dice = float(np.mean(pairwise_scores)) if len(pairwise_scores) > 0 else 0.0

    # Model vs each annotator
    model_vs_annotator_dice = [dice_score(model_pred, ann) for ann in annotator_masks]
    model_vs_annotator_iou = [iou_score(model_pred, ann) for ann in annotator_masks]

    # Consensus
    consensus = staple_consensus(annotator_masks)
    model_vs_consensus_dice = dice_score(model_pred, consensus)
    model_vs_consensus_iou = iou_score(model_pred, consensus)

    return {
        "inter_annotator_dice": {
            "mean": inter_annotator_dice,
            "std": float(np.std(pairwise_scores)) if len(pairwise_scores) > 0 else 0.0,
            "min": float(np.min(pairwise_scores)) if len(pairwise_scores) > 0 else 0.0,
            "max": float(np.max(pairwise_scores)) if len(pairwise_scores) > 0 else 0.0,
        },
        "model_vs_annotators_dice": {
            "mean": float(np.mean(model_vs_annotator_dice)),
            "std": float(np.std(model_vs_annotator_dice)),
            "per_annotator": model_vs_annotator_dice,
        },
        "model_vs_annotators_iou": {
            "mean": float(np.mean(model_vs_annotator_iou)),
            "per_annotator": model_vs_annotator_iou,
        },
        "model_vs_consensus_dice": model_vs_consensus_dice,
        "model_vs_consensus_iou": model_vs_consensus_iou,
        "model_reaches_human_level": (
            float(np.mean(model_vs_annotator_dice)) >= inter_annotator_dice * 0.95
        ),
    }


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Compute Fleiss' kappa for classification reliability.

    Args:
        ratings: (N, K) matrix where N=subjects, K=categories.
                 Each entry is the number of raters who assigned
                 that category to that subject.

    Returns:
        Fleiss' kappa score (-1 to 1, >0.6 is substantial agreement).
    """
    n_subjects, _n_categories = ratings.shape
    n_raters = ratings.sum(axis=1)[0]  # Assume same number of raters

    # Proportion of assignments to each category
    p_j = ratings.sum(axis=0) / (n_subjects * n_raters)

    # Per-subject agreement
    p_i = (ratings**2).sum(axis=1) - n_raters
    p_i = p_i / (n_raters * (n_raters - 1))

    p_bar = p_i.mean()
    p_e = (p_j**2).sum()

    if abs(1 - p_e) < 1e-10:
        return 1.0

    return float((p_bar - p_e) / (1 - p_e))


def print_agreement_report(results: dict[str, Any]) -> None:
    """Print formatted agreement analysis."""
    print(f"\n{'=' * 60}")  # noqa: T201
    print("Inter-Annotator Agreement Analysis")  # noqa: T201
    print(f"{'=' * 60}")  # noqa: T201

    iad = results["inter_annotator_dice"]
    print(f"  Inter-annotator Dice: {iad['mean']:.4f} (+/- {iad['std']:.4f})")  # noqa: T201

    mad = results["model_vs_annotators_dice"]
    print(f"  Model vs annotators:  {mad['mean']:.4f} (+/- {mad['std']:.4f})")  # noqa: T201

    print(f"  Model vs consensus Dice: {results['model_vs_consensus_dice']:.4f}")  # noqa: T201
    print(f"  Model vs consensus IoU:  {results['model_vs_consensus_iou']:.4f}")  # noqa: T201

    if results["model_reaches_human_level"]:
        print("  Model reaches human-level performance (within 95%)")  # noqa: T201
    else:
        ratio = mad["mean"] / max(iad["mean"], 1e-6) * 100
        print(f"  Model at {ratio:.1f}% of inter-annotator agreement")  # noqa: T201

    print(f"{'=' * 60}\n")  # noqa: T201
