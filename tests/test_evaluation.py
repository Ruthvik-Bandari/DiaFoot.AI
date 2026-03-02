"""DiaFoot.AI v2 — Evaluation Tests (Phase 4)."""

import numpy as np
import pytest

from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.metrics import (
    aggregate_metrics,
    compute_segmentation_metrics,
    dice_score,
    hausdorff_distance_95,
    iou_score,
    surface_dice,
    wound_area_mm2,
)


class TestDiceScore:
    def test_perfect_overlap(self) -> None:
        mask = np.ones((64, 64), dtype=np.uint8)
        assert dice_score(mask, mask) > 0.99

    def test_no_overlap(self) -> None:
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[:32, :] = 1
        target = np.zeros((64, 64), dtype=np.uint8)
        target[32:, :] = 1
        assert dice_score(pred, target) < 0.01

    def test_partial_overlap(self) -> None:
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[10:50, 10:50] = 1
        target = np.zeros((64, 64), dtype=np.uint8)
        target[20:60, 20:60] = 1
        score = dice_score(pred, target)
        assert 0.3 < score < 0.8


class TestIoUScore:
    def test_perfect(self) -> None:
        mask = np.ones((64, 64), dtype=np.uint8)
        assert iou_score(mask, mask) > 0.99

    def test_empty_both(self) -> None:
        empty = np.zeros((64, 64), dtype=np.uint8)
        assert iou_score(empty, empty) > 0.99  # Both empty = perfect


class TestHausdorffDistance:
    def test_perfect(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        assert hausdorff_distance_95(mask, mask) < 1.0

    def test_shifted(self) -> None:
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[20:40, 20:40] = 1
        target = np.zeros((64, 64), dtype=np.uint8)
        target[25:45, 25:45] = 1
        hd = hausdorff_distance_95(pred, target)
        assert hd > 0


class TestSurfaceDice:
    def test_perfect(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        assert surface_dice(mask, mask) > 0.99

    def test_empty_both(self) -> None:
        empty = np.zeros((64, 64), dtype=np.uint8)
        assert surface_dice(empty, empty) == 1.0


class TestWoundArea:
    def test_known_area(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 1  # 100 pixels
        area = wound_area_mm2(mask, pixel_spacing_mm=1.0)
        assert area == 100.0


class TestComputeSegmentationMetrics:
    def test_returns_all_keys(self) -> None:
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[20:40, 20:40] = 1
        target = np.zeros((64, 64), dtype=np.uint8)
        target[22:42, 22:42] = 1
        m = compute_segmentation_metrics(pred, target)
        assert "dice" in m
        assert "iou" in m
        assert "hd95" in m
        assert "nsd_2mm" in m
        assert "wound_area_mm2" in m


class TestAggregateMetrics:
    def test_aggregation(self) -> None:
        metrics_list = [
            {"dice": 0.8, "iou": 0.7},
            {"dice": 0.9, "iou": 0.85},
        ]
        summary = aggregate_metrics(metrics_list)
        assert summary["dice"]["mean"] == pytest.approx(0.85)
        assert summary["iou"]["mean"] == pytest.approx(0.775)


class TestClassificationMetrics:
    def test_perfect_classification(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        m = compute_classification_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0

    def test_with_probabilities(self) -> None:
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        y_prob = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
        m = compute_classification_metrics(y_true, y_pred, y_prob)
        assert "auroc_macro" in m
