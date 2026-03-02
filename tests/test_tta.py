"""DiaFoot.AI v2 — TTA Tests (Phase 4, Commit 23)."""

import pytest
import torch
import torch.nn as nn

from src.inference.tta import (
    compute_tta_improvement,
    tta_predict_classification,
    tta_predict_segmentation,
)


class SimpleSegModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])


class SimpleClsModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.shape[0], 3)


class TestTTASegmentation:
    def test_output_shapes(self) -> None:
        model = SimpleSegModel()
        image = torch.randn(1, 3, 64, 64)
        mean_pred, uncertainty = tta_predict_segmentation(
            model,
            image,
            num_augmentations=4,
        )
        assert mean_pred.shape == (64, 64)
        assert uncertainty.shape == (64, 64)

    def test_uncertainty_non_negative(self) -> None:
        model = SimpleSegModel()
        image = torch.randn(1, 3, 64, 64)
        _, uncertainty = tta_predict_segmentation(
            model,
            image,
            num_augmentations=4,
        )
        assert (uncertainty >= 0).all()


class TestTTAClassification:
    def test_output_shapes(self) -> None:
        model = SimpleClsModel()
        image = torch.randn(1, 3, 64, 64)
        probs, uncertainty = tta_predict_classification(
            model,
            image,
            num_augmentations=4,
        )
        assert probs.shape == (3,)
        assert 0 <= uncertainty <= 1


class TestTTAImprovement:
    def test_computes_diff(self) -> None:
        base = {"dice": 0.80, "iou": 0.70}
        tta = {"dice": 0.85, "iou": 0.74}
        result = compute_tta_improvement(base, tta)
        assert result["dice_abs_improvement"] == pytest.approx(0.05)
