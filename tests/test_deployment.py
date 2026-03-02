"""DiaFoot.AI v2 — Deployment Tests (Phase 6)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.inference.onnx_export import export_to_onnx
from src.inference.pipeline import InferencePipeline, PipelineResult


class DummyClassifier(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.tensor([[0.1, 0.1, 0.8]] * b)


class DummySegmenter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]) * 2.0


class TestONNXExport:
    def test_export_creates_file(self, tmp_path: pytest.TempPathFactory) -> None:
        pytest.importorskip("onnxscript")
        model = nn.Sequential(nn.Conv2d(3, 1, 1))
        path = export_to_onnx(model, tmp_path / "test.onnx", input_shape=(1, 3, 64, 64))  # type: ignore[arg-type]
        assert path.exists()
        assert path.stat().st_size > 0


class TestInferencePipeline:
    def test_classify_dfu(self) -> None:
        pipe = InferencePipeline(
            classifier=DummyClassifier(),
            segmenter=DummySegmenter(),
            device="cpu",
        )
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = pipe.predict(image)
        assert result.classification == "DFU"
        assert result.has_wound
        assert result.wound_area_px > 0

    def test_early_exit_healthy(self) -> None:
        class HealthyClassifier(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b = x.shape[0]
                return torch.tensor([[10.0, -10.0, -10.0]] * b)

        pipe = InferencePipeline(
            classifier=HealthyClassifier(),
            segmenter=DummySegmenter(),
            device="cpu",
        )
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = pipe.predict(image)
        assert result.classification == "Healthy"
        assert not result.has_wound
        assert result.segmentation_mask is None

    def test_result_fields(self) -> None:
        result = PipelineResult()
        assert result.classification == "Unknown"
        assert result.wound_area_mm2 == 0.0
