"""DiaFoot.AI v2 — Deployment Tests (Phase 6)."""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.deploy.app import assess_image_quality, resolve_runtime_thresholds
from src.inference.onnx_export import export_to_onnx
from src.inference.pipeline import InferencePipeline, PipelineResult


class DummyClassifier(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.tensor([[-2.0, -2.0, 4.0]] * b)


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
        assert result.defer_to_clinician is False
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
        assert result.defer_to_clinician is False
        assert result.wound_area_mm2 == 0.0

    def test_defer_on_low_confidence(self) -> None:
        class UncertainClassifier(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b = x.shape[0]
                return torch.tensor([[0.0, 0.0, 0.0]] * b)

        pipe = InferencePipeline(
            classifier=UncertainClassifier(),
            segmenter=DummySegmenter(),
            device="cpu",
            defer_threshold=0.9,
        )
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = pipe.predict(image)
        assert result.defer_to_clinician is True
        assert result.defer_reason == "low_classification_confidence"


class TestDeploymentThresholdConfig:
    def test_resolve_thresholds_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DIAFOOT_CONFIDENCE_THRESHOLD", "0.88")
        monkeypatch.setenv("DIAFOOT_DEFER_THRESHOLD", "0.72")

        confidence, defer, source = resolve_runtime_thresholds()
        assert confidence == pytest.approx(0.88)
        assert defer == pytest.approx(0.72)
        assert source == "env"

    def test_resolve_defer_threshold_from_calibration(
        self,
        tmp_path: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cal_path = tmp_path / "classification_calibration.json"
        payload = {
            "classification": {
                "defer_tuning": {
                    "recommended_threshold": 0.67,
                }
            }
        }
        cal_path.write_text(json.dumps(payload))

        monkeypatch.delenv("DIAFOOT_DEFER_THRESHOLD", raising=False)
        monkeypatch.setenv("DIAFOOT_CALIBRATION_PATH", str(cal_path))

        _confidence, defer, source = resolve_runtime_thresholds()
        assert defer == pytest.approx(0.67)
        assert source.startswith("calibration:")


class TestImageQualityAssessment:
    def test_flags_low_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DIAFOOT_MIN_IMAGE_SIDE", "300")
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        quality = assess_image_quality(image)
        assert quality["quality_passed"] is False
        assert "low_resolution" in quality["quality_flags"]

    def test_flags_too_dark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DIAFOOT_BRIGHTNESS_MIN", "30")
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        quality = assess_image_quality(image)
        assert quality["quality_passed"] is False
        assert "too_dark" in quality["quality_flags"]

    def test_quality_passes_for_reasonable_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DIAFOOT_MIN_IMAGE_SIDE", "256")
        monkeypatch.setenv("DIAFOOT_BLUR_VARIANCE_THRESHOLD", "5")
        monkeypatch.setenv("DIAFOOT_BRIGHTNESS_MIN", "20")
        monkeypatch.setenv("DIAFOOT_BRIGHTNESS_MAX", "235")

        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        quality = assess_image_quality(image)
        assert quality["quality_passed"] is True
        assert quality["quality_flags"] == []
