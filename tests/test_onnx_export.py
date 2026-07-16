"""Unit tests for ONNX export and validation helpers."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import src.inference.onnx_export as onnx_export


class _IdentityModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _DictOutputModel(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"seg_logits": x}


class TestExportToONNX:
    def test_export_uses_dynamo_and_dynamic_shapes(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        called: dict[str, object] = {}

        def _fake_export(*args: object, **kwargs: object) -> None:
            called["args"] = args
            called["kwargs"] = kwargs
            output_path = Path(args[2])
            output_path.write_bytes(b"onnx")

        monkeypatch.setattr(onnx_export.torch.onnx, "export", _fake_export)

        model = _IdentityModel()
        out = onnx_export.export_to_onnx(model, tmp_path / "model.onnx", dynamic_batch=True)

        assert out.exists()
        kwargs = called["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs.get("dynamo") is True
        assert "dynamic_shapes" in kwargs


class TestValidateONNX:
    def test_validate_skips_when_onnxruntime_missing(
        self,
        monkeypatch,
    ) -> None:
        real_import = builtins.__import__

        def _import(name: str, *args: object, **kwargs: object) -> object:
            if name == "onnxruntime":
                raise ImportError("onnxruntime unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import)

        ok = onnx_export.validate_onnx(_IdentityModel(), "dummy.onnx", input_shape=(1, 3, 8, 8))
        assert ok is True

    def test_validate_returns_true_on_matching_outputs(
        self,
        monkeypatch,
    ) -> None:
        fixed = torch.ones(1, 3, 8, 8)
        monkeypatch.setattr(onnx_export.torch, "randn", lambda *shape: fixed.clone())

        class _Session:
            def __init__(self, _path: str) -> None:
                self.path = _path

            def run(self, _unused: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
                return [feed["input"]]

        class _ORT:
            InferenceSession = _Session

        monkeypatch.setitem(sys.modules, "onnxruntime", _ORT())

        ok = onnx_export.validate_onnx(_IdentityModel(), "dummy.onnx", input_shape=(1, 3, 8, 8))
        assert ok is True

    def test_validate_returns_false_on_mismatch(
        self,
        monkeypatch,
    ) -> None:
        fixed = torch.ones(1, 3, 8, 8)
        monkeypatch.setattr(onnx_export.torch, "randn", lambda *shape: fixed.clone())

        class _Session:
            def __init__(self, _path: str) -> None:
                self.path = _path

            def run(self, _unused: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
                return [np.zeros_like(feed["input"])]

        class _ORT:
            InferenceSession = _Session

        monkeypatch.setitem(sys.modules, "onnxruntime", _ORT())

        ok = onnx_export.validate_onnx(_IdentityModel(), "dummy.onnx", input_shape=(1, 3, 8, 8))
        assert ok is False

    def test_validate_handles_dict_output(self, monkeypatch) -> None:
        fixed = torch.ones(1, 3, 8, 8)
        monkeypatch.setattr(onnx_export.torch, "randn", lambda *shape: fixed.clone())

        class _Session:
            def __init__(self, _path: str) -> None:
                self.path = _path

            def run(self, _unused: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
                return [feed["input"]]

        class _ORT:
            InferenceSession = _Session

        monkeypatch.setitem(sys.modules, "onnxruntime", _ORT())

        ok = onnx_export.validate_onnx(_DictOutputModel(), "dummy.onnx", input_shape=(1, 3, 8, 8))
        assert ok is True


class TestOpsetMismatchWarning:
    """Verify export_to_onnx warns loudly when the produced opset differs
    from the requested opset_version (Finding 2: opset mismatch is silent).
    """

    @staticmethod
    def _fake_export_with_opset(actual_opset: int):
        """Build a fake torch.onnx.export that writes a real ONNX file
        declaring `actual_opset` as its ai.onnx opset, regardless of the
        requested opset_version.
        """
        import onnx

        def _fake_export(*args: object, **kwargs: object) -> None:
            output_path = Path(args[2])
            model_proto = onnx.helper.make_model(
                onnx.helper.make_graph([], "g", [], []),
                opset_imports=[onnx.helper.make_opsetid("", actual_opset)],
            )
            onnx.save(model_proto, str(output_path))

        return _fake_export

    def test_warns_when_actual_opset_differs_from_requested(
        self,
        tmp_path: Path,
        monkeypatch,
        caplog,
    ) -> None:
        monkeypatch.setattr(onnx_export.torch.onnx, "export", self._fake_export_with_opset(18))

        model = _IdentityModel()
        with caplog.at_level("WARNING", logger=onnx_export.logger.name):
            onnx_export.export_to_onnx(
                model, tmp_path / "model.onnx", opset_version=17, dynamic_batch=False
            )

        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) == 1
        message = warning_records[0].message
        assert "17" in message
        assert "18" in message

    def test_no_warning_when_actual_opset_matches_requested(
        self,
        tmp_path: Path,
        monkeypatch,
        caplog,
    ) -> None:
        monkeypatch.setattr(onnx_export.torch.onnx, "export", self._fake_export_with_opset(17))

        model = _IdentityModel()
        with caplog.at_level("WARNING", logger=onnx_export.logger.name):
            onnx_export.export_to_onnx(
                model, tmp_path / "model.onnx", opset_version=17, dynamic_batch=False
            )

        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warning_records == []


class TestExportRoundTrip:
    """Real (non-mocked) export + onnxruntime round-trip tests.

    These exercise the actual production models end-to-end: dynamo export
    with dynamic batch, then onnxruntime inference at batch 1 and batch 2,
    compared numerically against PyTorch eval-mode output. This is the
    coverage that pure-mock tests above cannot provide.
    """

    @pytest.mark.slow
    @pytest.mark.integration
    def test_segmenter_dynamic_batch_round_trip(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxscript")
        import onnxruntime as ort

        from src.models.dinov2_segmenter import DINOv2Segmenter

        try:
            model = DINOv2Segmenter(backbone="dinov2_vits14", num_classes=1, freeze_backbone=True)
        except Exception:  # torch.hub.load can raise many error types
            pytest.skip("DINOv2 backbone unavailable")

        model.eval()

        onnx_path = onnx_export.export_to_onnx(
            model,
            tmp_path / "segmenter.onnx",
            input_shape=(1, 3, 518, 518),
            dynamic_batch=True,
        )
        assert onnx_path.exists()

        session = ort.InferenceSession(str(onnx_path))

        for batch_size in (1, 2):
            x = torch.randn(batch_size, 3, 518, 518)
            with torch.no_grad():
                pt_out = model(x).numpy()

            assert pt_out.shape == (batch_size, 1, 518, 518)

            ort_out = session.run(None, {"input": x.numpy()})[0]

            max_diff = float(np.max(np.abs(pt_out - ort_out)))
            assert np.allclose(pt_out, ort_out, atol=1e-3), (
                f"segmenter ONNX/PyTorch mismatch at batch_size={batch_size}: "
                f"max abs diff={max_diff}"
            )

    @pytest.mark.slow
    @pytest.mark.integration
    def test_classifier_dynamic_batch_round_trip(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxscript")
        import onnxruntime as ort

        from src.models.dinov2_classifier import DINOv2Classifier

        try:
            model = DINOv2Classifier(backbone="dinov2_vits14", num_classes=3, freeze_backbone=True)
        except Exception:  # torch.hub.load can raise many error types
            pytest.skip("DINOv2 backbone unavailable")

        model.eval()

        onnx_path = onnx_export.export_to_onnx(
            model,
            tmp_path / "classifier.onnx",
            input_shape=(1, 3, 518, 518),
            dynamic_batch=True,
        )
        assert onnx_path.exists()

        session = ort.InferenceSession(str(onnx_path))

        for batch_size in (1, 2):
            x = torch.randn(batch_size, 3, 518, 518)
            with torch.no_grad():
                pt_out = model(x).numpy()

            assert pt_out.shape == (batch_size, 3)

            ort_out = session.run(None, {"input": x.numpy()})[0]

            max_diff = float(np.max(np.abs(pt_out - ort_out)))
            assert np.allclose(pt_out, ort_out, atol=1e-3), (
                f"classifier ONNX/PyTorch mismatch at batch_size={batch_size}: "
                f"max abs diff={max_diff}"
            )
