"""Tests for external validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.external_validation import bootstrap_ci, compute_drop_report


def test_bootstrap_ci_outputs_bounds() -> None:
    values = np.array([0.7, 0.8, 0.9, 1.0], dtype=float)
    ci = bootstrap_ci(values, n_bootstrap=200, seed=0)
    assert 0.0 <= ci["mean"] <= 1.0
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_compute_drop_report() -> None:
    rep = compute_drop_report(
        internal={"accuracy": 0.9},
        external={"accuracy": 0.75},
        keys=["accuracy"],
    )
    assert "accuracy" in rep
    assert rep["accuracy"]["absolute_drop"] == pytest.approx(0.15)
    assert rep["accuracy"]["relative_drop"] > 0
