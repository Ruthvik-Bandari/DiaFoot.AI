"""DiaFoot.AI v2 — Inference Pipeline Tests."""

import numpy as np
import torch
import torch.nn as nn

from src.inference.pipeline import InferencePipeline


class _FixedClassifier(nn.Module):
    """Classifier stub that returns fixed logits for any input.

    Logits are set to log(probs) so that softmax recovers ``probs`` exactly
    (probs must sum to 1).
    """

    def __init__(self, probs: list[float]) -> None:
        super().__init__()
        self.logits = torch.log(torch.tensor([probs], dtype=torch.float32))

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        return self.logits


class _WoundSegmenter(nn.Module):
    """Segmenter stub that predicts a wound over the whole frame."""

    def __init__(self, size: int = 64) -> None:
        super().__init__()
        self.size = size

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        # Large positive logits -> sigmoid ~ 1 -> every pixel above threshold.
        return torch.full((1, 1, self.size, self.size), 10.0)


def _dummy_image() -> np.ndarray:
    return np.full((64, 64, 3), 128, dtype=np.uint8)


class TestDisagreementEscalation:
    """Segmenter-finds-wound vs classifier must always flag for review."""

    def test_healthy_prediction_with_segmented_wound_defers(self) -> None:
        # Healthy is argmax (conf 0.70, not confident enough to early-exit),
        # dfu_prob 0.15 >= fallback (0.10) so segmentation runs, and it finds
        # a wound. The classifier said "Healthy" but a wound was segmented:
        # this disagreement must be flagged, not silently returned as Healthy.
        pipeline = InferencePipeline(
            classifier=_FixedClassifier([0.70, 0.15, 0.15]),
            segmenter=_WoundSegmenter(),
            input_size=64,
        )
        result = pipeline.predict(_dummy_image())

        assert result.has_wound is True
        assert result.defer_to_clinician is True
        assert result.defer_reason == "segmentation_classifier_disagreement"
        assert result.classification != "Healthy"

    def test_nondfu_prediction_with_segmented_wound_still_defers(self) -> None:
        # Regression guard for existing behavior: Non-DFU + wound already deferred.
        pipeline = InferencePipeline(
            classifier=_FixedClassifier([0.15, 0.70, 0.15]),
            segmenter=_WoundSegmenter(),
            input_size=64,
        )
        result = pipeline.predict(_dummy_image())

        assert result.has_wound is True
        assert result.defer_to_clinician is True
        assert result.defer_reason == "segmentation_classifier_disagreement"

    def test_confident_healthy_early_exits_no_defer(self) -> None:
        # Guard against over-triggering: a confident Healthy (>= 0.95) early-exits
        # before segmentation and must stay Healthy with no review flag.
        pipeline = InferencePipeline(
            classifier=_FixedClassifier([0.99, 0.005, 0.005]),
            segmenter=_WoundSegmenter(),
            input_size=64,
        )
        result = pipeline.predict(_dummy_image())

        assert result.has_wound is False
        assert result.defer_to_clinician is False
        assert result.classification == "Healthy"

    def test_borderline_healthy_low_dfu_prob_segments_and_defers(self) -> None:
        # Healthy argmax, confidence 0.85 (< 0.95, so no early exit) but dfu_prob
        # 0.05 < fallback (0.10): before the gating fix segmentation never ran,
        # so a real wound was missed with no defer. A non-confident Healthy must
        # now be segmented, and the detected wound triggers escalation.
        pipeline = InferencePipeline(
            classifier=_FixedClassifier([0.85, 0.10, 0.05]),
            segmenter=_WoundSegmenter(),
            input_size=64,
        )
        result = pipeline.predict(_dummy_image())

        assert result.has_wound is True
        assert result.defer_to_clinician is True
        assert result.defer_reason == "segmentation_classifier_disagreement"
        assert result.classification != "Healthy"


class TestPreprocess:
    def test_2d_grayscale_image_does_not_crash(self) -> None:
        # A 2-D grayscale image (H, W) with no channel axis must be handled,
        # not crash with IndexError on image.shape[2].
        pipeline = InferencePipeline(
            classifier=_FixedClassifier([0.5, 0.3, 0.2]),
            segmenter=_WoundSegmenter(),
            input_size=64,
        )
        gray = np.full((32, 48), 128, dtype=np.uint8)
        tensor = pipeline.preprocess(gray)
        assert tuple(tensor.shape) == (1, 3, 64, 64)
