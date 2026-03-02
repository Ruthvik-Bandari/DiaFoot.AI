"""DiaFoot.AI v2 — Fairness Tests (Phase 5, Commit 26)."""

import numpy as np

from src.evaluation.fairness import (
    stratified_classification_audit,
    stratified_segmentation_audit,
)


class TestClassificationFairness:
    def test_equal_performance(self) -> None:
        filenames = ["a.png", "b.png", "c.png", "d.png"]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        ita_map = {
            "a.png": "Light",
            "b.png": "Light",
            "c.png": "Dark",
            "d.png": "Dark",
        }
        result = stratified_classification_audit(filenames, y_true, y_pred, ita_map)
        assert result["fairness_gap_accuracy"] == 0.0
        assert not result["bias_concern"]

    def test_biased_performance(self) -> None:
        filenames = ["a.png", "b.png", "c.png", "d.png"]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])  # Wrong for Dark group
        ita_map = {
            "a.png": "Light",
            "b.png": "Light",
            "c.png": "Dark",
            "d.png": "Dark",
        }
        result = stratified_classification_audit(filenames, y_true, y_pred, ita_map)
        assert result["fairness_gap_accuracy"] == 1.0
        assert result["bias_concern"]


class TestSegmentationFairness:
    def test_stratified_metrics(self) -> None:
        filenames = ["a.png", "b.png"]
        metrics = [
            {"dice": 0.9, "iou": 0.85},
            {"dice": 0.7, "iou": 0.6},
        ]
        ita_map = {"a.png": "Light", "b.png": "Dark"}
        result = stratified_segmentation_audit(filenames, metrics, ita_map)
        assert "Light" in result["per_ita_group"]
        assert "Dark" in result["per_ita_group"]
        assert abs(result["fairness_gaps"]["dice_gap"] - 0.2) < 1e-6
        assert result["bias_concern"]
