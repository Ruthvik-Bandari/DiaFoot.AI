"""DiaFoot.AI v2 — Uncertainty Tests (Phase 5, Commit 24)."""

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.uncertainty import (
    compute_uncertainty_metrics,
    conformal_wound_area,
    enable_mc_dropout,
    mc_dropout_predict,
)


class DropoutSegModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x))


class TestMCDropout:
    def test_enable_dropout(self) -> None:
        model = DropoutSegModel()
        model.eval()
        assert not model.dropout.training
        enable_mc_dropout(model)
        assert model.dropout.training

    def test_mc_predict_shapes(self) -> None:
        model = DropoutSegModel()
        image = torch.randn(1, 3, 32, 32)
        mean_pred, uncertainty, samples = mc_dropout_predict(
            model,
            image,
            num_samples=5,
        )
        assert mean_pred.shape == (32, 32)
        assert uncertainty.shape == (32, 32)
        assert samples.shape == (5, 32, 32)

    def test_uncertainty_non_negative(self) -> None:
        model = DropoutSegModel()
        image = torch.randn(1, 3, 32, 32)
        _, uncertainty, _ = mc_dropout_predict(
            model,
            image,
            num_samples=5,
        )
        assert (uncertainty >= 0).all()


class TestConformalPrediction:
    def test_coverage_interval(self) -> None:
        cal_pred = np.array([100, 200, 150, 180, 120])
        cal_gt = np.array([110, 190, 160, 170, 130])
        lower, upper = conformal_wound_area(cal_pred, cal_gt, 150.0, alpha=0.1)
        assert lower < 150.0
        assert upper > 150.0
        assert lower >= 0

    def test_zero_area(self) -> None:
        cal_pred = np.array([10, 20, 15])
        cal_gt = np.array([12, 18, 16])
        lower, _upper = conformal_wound_area(cal_pred, cal_gt, 0.0, alpha=0.1)
        assert lower == 0.0


class TestUncertaintyMetrics:
    def test_basic_metrics(self) -> None:
        mean_pred = np.random.rand(64, 64)
        uncertainty = np.random.rand(64, 64) * 0.1
        metrics = compute_uncertainty_metrics(mean_pred, uncertainty)
        assert "mean_uncertainty" in metrics
        assert "max_uncertainty" in metrics
        assert metrics["mean_uncertainty"] >= 0

    def test_with_ground_truth(self) -> None:
        mean_pred = np.zeros((64, 64))
        mean_pred[20:40, 20:40] = 0.9
        uncertainty = np.random.rand(64, 64) * 0.1
        gt = np.zeros((64, 64))
        gt[22:42, 22:42] = 1
        metrics = compute_uncertainty_metrics(mean_pred, uncertainty, gt)
        assert "uncertainty_error_correlation" in metrics
