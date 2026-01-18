"""
Evaluation Module
==================

This module contains evaluation utilities:
    - Segmentation metrics (IoU, Dice, Hausdorff)
    - Classification metrics (Accuracy, F1, AUC)
    - Visualization utilities
"""

from .metrics import (
    SegmentationMetrics,
    ClassificationMetrics,
    compute_iou,
    compute_dice,
    compute_hausdorff_distance,
    compute_boundary_iou,
    evaluate_segmentation_model,
    print_evaluation_results,
)

__all__ = [
    "SegmentationMetrics",
    "ClassificationMetrics",
    "compute_iou",
    "compute_dice",
    "compute_hausdorff_distance",
    "compute_boundary_iou",
    "evaluate_segmentation_model",
    "print_evaluation_results",
]
