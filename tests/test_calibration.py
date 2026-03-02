"""DiaFoot.AI v2 — Calibration Tests (Phase 5, Commit 25)."""

import numpy as np

from src.evaluation.calibration import (
    compute_segmentation_calibration,
    expected_calibration_error,
    temperature_scaling,
)


class TestECE:
    def test_perfect_calibration(self) -> None:
        confidences = np.array([0.9, 0.9, 0.1, 0.1])
        accuracies = np.array([1.0, 1.0, 0.0, 0.0])
        ece, _ = expected_calibration_error(confidences, accuracies)
        assert ece < 0.15

    def test_poor_calibration(self) -> None:
        confidences = np.array([0.99, 0.99, 0.99, 0.99])
        accuracies = np.array([1.0, 0.0, 0.0, 0.0])
        ece, _ = expected_calibration_error(confidences, accuracies)
        assert ece > 0.3

    def test_bin_details(self) -> None:
        confidences = np.random.rand(100)
        accuracies = (np.random.rand(100) > 0.5).astype(float)
        _, bins = expected_calibration_error(confidences, accuracies)
        assert "bin_centers" in bins
        assert len(bins["bin_centers"]) == 15


class TestTemperatureScaling:
    def test_returns_positive(self) -> None:
        logits = np.random.randn(50, 3)
        labels = np.random.randint(0, 3, 50)
        t = temperature_scaling(logits, labels)
        assert t > 0

    def test_overconfident_model(self) -> None:
        logits = np.random.randn(100, 3) * 10
        labels = np.random.randint(0, 3, 100)
        t = temperature_scaling(logits, labels)
        assert t >= 1.0


class TestSegmentationCalibration:
    def test_returns_ece(self) -> None:
        pred = np.random.rand(64, 64)
        gt = (np.random.rand(64, 64) > 0.5).astype(np.uint8)
        ece, bins = compute_segmentation_calibration(pred, gt)
        assert 0 <= ece <= 1
        assert "bin_centers" in bins
