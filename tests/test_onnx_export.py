"""Unit tests for ONNX export and validation helpers."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import numpy as np
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
