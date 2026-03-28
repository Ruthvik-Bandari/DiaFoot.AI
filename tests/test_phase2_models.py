"""DiaFoot.AI v2 — Phase 2 Model Tests (DINOv2 + Legacy)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.boundary_refine import (
    connected_component_filter,
    morphological_smooth,
    refine_prediction,
)
from src.models.fusegnet import FUSegNet
from src.models.nnunet_wrapper import NNUNetConfig, generate_dataset_json


class TestNNUNetWrapper:
    def test_config_defaults(self) -> None:
        config = NNUNetConfig()
        assert config.num_classes == 4
        assert "3" in config.class_names

    def test_generate_dataset_json(self, tmp_path: pytest.TempPathFactory) -> None:
        config = NNUNetConfig()
        path = generate_dataset_json(config, tmp_path, num_training=100)  # type: ignore[arg-type]
        assert path.exists()
        import json

        with open(path) as f:
            data = json.load(f)
        assert data["numTraining"] == 100
        assert "labels" in data


class TestDINOv2Classifier:
    @pytest.mark.slow
    def test_output_shape(self) -> None:
        from src.models.dinov2_classifier import DINOv2Classifier

        model = DINOv2Classifier(
            backbone="dinov2_vits14",
            num_classes=3,
            freeze_backbone=True,
        )
        x = torch.randn(1, 3, 518, 518)
        out = model(x)
        assert out.shape == (1, 3)

    @pytest.mark.slow
    def test_predict_with_confidence(self) -> None:
        from src.models.dinov2_classifier import DINOv2Classifier

        model = DINOv2Classifier(
            backbone="dinov2_vits14",
            num_classes=3,
            freeze_backbone=True,
        )
        x = torch.randn(1, 3, 518, 518)
        predicted, confidence = model.predict_with_confidence(x)
        assert predicted.shape == (1,)
        assert confidence.shape == (1,)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()


class TestDINOv2Segmenter:
    @pytest.mark.slow
    def test_output_shape(self) -> None:
        from src.models.dinov2_segmenter import DINOv2Segmenter

        model = DINOv2Segmenter(
            backbone="dinov2_vits14",
            num_classes=1,
            freeze_backbone=True,
        )
        x = torch.randn(1, 3, 518, 518)
        out = model(x)
        assert out.shape == (1, 1, 518, 518)


class TestFUSegNet:
    def test_output_shape(self) -> None:
        model = FUSegNet(
            encoder_name="resnet18",
            encoder_weights=None,
            classes=1,
        )
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_multiclass(self) -> None:
        model = FUSegNet(
            encoder_name="resnet18",
            encoder_weights=None,
            classes=4,
        )
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == (1, 4, 256, 256)


class TestBoundaryRefinement:
    def test_morphological_smooth(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        mask[45:55, 45:55] = 0  # hole
        result = morphological_smooth(mask, kernel_size=11, iterations=2)
        # Hole should be filled
        assert result[50, 50] == 255

    def test_connected_component_filter(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255  # Small (100px)
        mask[50:90, 50:90] = 255  # Large (1600px)
        result = connected_component_filter(mask, min_area=200)
        assert result[15, 15] == 0  # Small removed
        assert result[70, 70] == 255  # Large kept

    def test_refine_prediction(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        result = refine_prediction(mask)
        assert result.shape == (100, 100)
        assert result[50, 50] == 255
