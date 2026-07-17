"""DiaFoot.AI v2 — TTA Tests (Phase 4, Commit 23)."""

import pytest
import torch
import torch.nn as nn

from src.inference.tta import (
    _d4_transforms,
    compute_tta_improvement,
    tta_predict_classification,
    tta_predict_segmentation,
)


class TestD4Transforms:
    def test_eight_distinct_transforms(self) -> None:
        # num_augmentations=8 must yield 8 genuinely distinct views. The old
        # rot180 == hvflip (torch.flip(x,[2,3])), so requesting 8 silently gave 7
        # with one view double-counted, biasing the average and uncertainty.
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        pairs = _d4_transforms(8)
        assert len(pairs) == 8
        outs = [fwd(x) for fwd, _ in pairs]
        for i in range(len(outs)):
            for j in range(i + 1, len(outs)):
                assert not torch.equal(outs[i], outs[j]), f"transforms {i} and {j} identical"

    def test_inverse_restores_orientation(self) -> None:
        # Each inverse must undo its forward so spatial predictions realign
        # before averaging — including non-square inputs, where transpose/rot90
        # change H<->W and the inverse must restore the original shape.
        for shape in [(1, 2, 5, 5), (1, 2, 4, 6)]:
            x = torch.randn(*shape)
            for fwd, inv in _d4_transforms(8):
                restored = inv(fwd(x))
                assert restored.shape == x.shape
                torch.testing.assert_close(restored, x)


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
