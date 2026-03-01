"""DiaFoot.AI v2 — Classifier & Dataset Tests (Phase 2, Commit 8)."""

import pytest
import torch

from src.models.classifier import TriageClassifier


class TestTriageClassifier:
    """Test the triage classifier architecture."""

    def test_output_shape(self) -> None:
        model = TriageClassifier(
            backbone="tf_efficientnetv2_s",  # Smaller for testing
            num_classes=3,
            pretrained=False,
        )
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
        assert out.shape == (2, 3)

    def test_predict_with_confidence(self) -> None:
        model = TriageClassifier(
            backbone="tf_efficientnetv2_s",
            num_classes=3,
            pretrained=False,
        )
        x = torch.randn(2, 3, 512, 512)
        predicted, confidence = model.predict_with_confidence(x)
        assert predicted.shape == (2,)
        assert confidence.shape == (2,)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_different_input_sizes(self) -> None:
        model = TriageClassifier(
            backbone="tf_efficientnetv2_s",
            num_classes=3,
            pretrained=False,
        )
        for size in [224, 384, 512]:
            x = torch.randn(1, 3, size, size)
            out = model(x)
            assert out.shape == (1, 3)

    @pytest.mark.slow
    def test_pretrained_loads(self) -> None:
        model = TriageClassifier(
            backbone="tf_efficientnetv2_m",
            pretrained=True,
        )
        assert model.encoder.num_features > 0
