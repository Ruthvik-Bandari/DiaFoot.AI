"""Tests for failure atlas utilities."""

from __future__ import annotations

import numpy as np

from src.evaluation.failure_atlas import classify_segmentation_failure, summarize_failure_types


def test_classify_segmentation_failure_false_positive() -> None:
    pred = np.ones((16, 16), dtype=np.uint8)
    gt = np.zeros((16, 16), dtype=np.uint8)
    assert classify_segmentation_failure(pred, gt, dice=0.0) == "false_positive_empty_gt"


def test_classify_segmentation_failure_missed_lesion() -> None:
    pred = np.zeros((16, 16), dtype=np.uint8)
    gt = np.ones((16, 16), dtype=np.uint8)
    assert classify_segmentation_failure(pred, gt, dice=0.0) == "missed_lesion_false_negative"


def test_summarize_failure_types() -> None:
    s = summarize_failure_types(["a", "a", "b"])
    assert s["total"] == 3
    assert s["counts"]["a"] == 2
