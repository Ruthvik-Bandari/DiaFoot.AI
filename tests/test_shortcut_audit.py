"""Tests for shortcut-learning audit utilities."""

from __future__ import annotations

import numpy as np

from src.evaluation.shortcut_audit import (
    keep_center_only,
    perturb_border_noise,
    summarize_shortcut_shift,
)


def test_perturb_border_noise_changes_border() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    pert = perturb_border_noise(image, border_fraction=0.2, seed=1)
    assert pert.shape == image.shape
    assert not np.array_equal(pert[:8, :, :], image[:8, :, :])


def test_keep_center_only_zeros_border() -> None:
    image = np.ones((64, 64, 3), dtype=np.uint8) * 255
    out = keep_center_only(image, border_fraction=0.25)
    assert out.shape == image.shape
    assert out[0, 0, 0] == 0
    assert out[32, 32, 0] == 255


def test_summarize_shortcut_shift() -> None:
    labels = np.array([0, 1, 2, 2])
    baseline_pred = np.array([0, 1, 2, 1])
    baseline_conf = np.array([0.9, 0.8, 0.85, 0.7])
    perturbed_pred = np.array([0, 0, 2, 0])
    perturbed_conf = np.array([0.7, 0.5, 0.8, 0.4])

    rep = summarize_shortcut_shift(
        labels,
        baseline_pred,
        baseline_conf,
        perturbed_pred,
        perturbed_conf,
    )
    assert rep["n"] == 4
    assert rep["baseline_accuracy"] >= rep["perturbed_accuracy"]
