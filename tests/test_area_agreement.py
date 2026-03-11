"""Tests for area agreement metrics."""

from __future__ import annotations

import numpy as np

from src.evaluation.area_agreement import compute_area_agreement


def test_compute_area_agreement() -> None:
    pred = np.array([100.0, 120.0, 80.0])
    gt = np.array([110.0, 115.0, 85.0])
    rep = compute_area_agreement(pred, gt)
    assert rep["n"] == 3
    assert rep["mae_mm2"] >= 0
    assert -1.0 <= rep["pearson_r"] <= 1.0
