"""Tests for subgroup analysis utilities."""

from __future__ import annotations

import numpy as np

from src.evaluation.subgroup_analysis import (
    classification_subgroup_report,
    segmentation_subgroup_report,
)


def test_classification_subgroup_report() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    groups = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]

    rep = classification_subgroup_report(y_true, y_pred, groups, min_count=2)
    assert "per_group" in rep
    assert "a" in rep["per_group"]
    assert "b" in rep["per_group"]


def test_segmentation_subgroup_report() -> None:
    dice = np.array([0.8, 0.7, 0.9, 0.6, 0.5, 0.85, 0.75, 0.65])
    iou = np.array([0.7, 0.6, 0.8, 0.5, 0.4, 0.75, 0.65, 0.55])
    groups = ["x", "x", "x", "x", "y", "y", "y", "y"]

    rep = segmentation_subgroup_report(dice, iou, groups, min_count=2)
    assert "per_group" in rep
    assert "x" in rep["per_group"]
    assert "y" in rep["per_group"]
