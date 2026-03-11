"""DiaFoot.AI v2 — Calibration Tests (Phase 5, Commit 25)."""

import numpy as np

from src.evaluation.calibration import (
    compute_calibration_report,
    compute_segmentation_calibration,
    expected_calibration_error,
    multiclass_brier_score,
    temperature_scaling,
    tune_defer_threshold,
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


class TestDeferThresholdTuning:
    def test_returns_recommended_threshold(self) -> None:
        probs = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.55, 0.40, 0.05],
                [0.34, 0.33, 0.33],
                [0.1, 0.8, 0.1],
            ]
        )
        labels = np.array([0, 0, 2, 1])
        report = tune_defer_threshold(probs, labels, min_coverage=0.5)
        assert "recommended_threshold" in report
        assert 0.0 <= report["recommended_threshold"] <= 1.0
        assert len(report["sweep"]) > 0


class TestBrierAndReport:
    def test_multiclass_brier_score_range(self) -> None:
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        labels = np.array([0, 2])
        brier = multiclass_brier_score(probs, labels)
        assert 0.0 <= brier <= 2.0

    def test_calibration_report_contains_defer_tuning(self) -> None:
        logits = np.random.randn(30, 3)
        labels = np.random.randint(0, 3, 30)
        report = compute_calibration_report(
            classification_logits=logits,
            classification_labels=labels,
        )
        assert "classification" in report
        cls = report["classification"]
        assert "brier_before" in cls
        assert "brier_after" in cls
        assert "defer_tuning" in cls


class TestSegmentationCalibration:
    def test_returns_ece(self) -> None:
        pred = np.random.rand(64, 64)
        gt = (np.random.rand(64, 64) > 0.5).astype(np.uint8)
        ece, bins = compute_segmentation_calibration(pred, gt)
        assert 0 <= ece <= 1
        assert "bin_centers" in bins
