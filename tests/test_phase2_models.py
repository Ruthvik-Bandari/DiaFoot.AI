"""DiaFoot.AI v2 — Phase 2 Model Tests (Commits 10-12)."""

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
from src.models.medsam2_finetune import (
    LoRAConfig,
    LoRALinear,
    apply_lora_to_model,
    mask_to_bbox,
)
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


class TestLoRA:
    def test_lora_linear_shape(self) -> None:
        original = nn.Linear(256, 128)
        config = LoRAConfig(rank=8)
        lora = LoRALinear(original, config)
        x = torch.randn(4, 256)
        out = lora(x)
        assert out.shape == (4, 128)

    def test_lora_preserves_output_initially(self) -> None:
        original = nn.Linear(64, 32)
        config = LoRAConfig(rank=4)
        lora = LoRALinear(original, config)
        x = torch.randn(2, 64)
        # B is initialized to zeros, so LoRA output should equal original
        orig_out = original(x)
        lora_out = lora(x)
        torch.testing.assert_close(orig_out, lora_out, atol=1e-6, rtol=1e-6)

    def test_apply_lora_to_model(self) -> None:
        _model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        # Rename modules to match target
        model_with_names = nn.Module()
        model_with_names.q_proj = nn.Linear(64, 32)
        model_with_names.relu = nn.ReLU()
        model_with_names.v_proj = nn.Linear(32, 16)

        config = LoRAConfig(rank=4, target_modules=("q_proj", "v_proj"))
        _adapted, num_params = apply_lora_to_model(model_with_names, config)
        assert num_params > 0

    def test_mask_to_bbox(self) -> None:
        mask = torch.zeros(64, 64)
        mask[10:30, 20:50] = 1
        bbox = mask_to_bbox(mask, padding=5)
        assert bbox.shape == (1, 4)
        assert bbox[0, 0] < 20  # x1 with padding
        assert bbox[0, 2] > 49  # x2 with padding

    def test_mask_to_bbox_empty(self) -> None:
        mask = torch.zeros(64, 64)
        bbox = mask_to_bbox(mask)
        assert bbox.shape == (1, 4)


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
