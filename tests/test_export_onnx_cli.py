"""Integration test for the CLI export path (Finding 3).

Proves that `scripts/export_onnx.py`'s own load+export path — the code
actually run by `python scripts/export_onnx.py --model dinov2 ...` — produces
a segmenter ONNX model that works at batch size > 1. Before the CLI was
refactored to delegate to `src.inference.onnx_export.export_to_onnx`, the
script's local `dynamic_axes`-based exporter baked batch-dependent Resize
scale factors, and onnxruntime rejected batch 2 with a `ScalesValidation`
error. This test exercises the CLI's own `_load_model` + `export_to_onnx`
call so a regression back to the old exporter is caught here, not just in
`src/inference/onnx_export.py`'s own tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

# `scripts` has no __init__.py, but the CLI's own `sys.path.insert` (and
# tests/conftest.py) put the repo root on sys.path, so `scripts.export_onnx`
# resolves as an implicit namespace package import — no package marker file
# needed.
from scripts import export_onnx

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.slow
@pytest.mark.integration
def test_cli_segmenter_export_dynamic_batch_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    import onnxruntime as ort

    from src.models.dinov2_segmenter import DINOv2Segmenter

    try:
        model = DINOv2Segmenter(backbone="dinov2_vits14", num_classes=1, freeze_backbone=True)
    except Exception:  # torch.hub.load can raise many error types
        pytest.skip("DINOv2 backbone unavailable")

    model.eval()

    ckpt_path = tmp_path / "segmenter.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    loaded_model = export_onnx._load_model(str(ckpt_path), "dinov2", "dinov2_vits14")
    loaded_model.eval()

    onnx_path = tmp_path / "segmenter.onnx"
    export_onnx.export_to_onnx(
        loaded_model,
        onnx_path,
        input_shape=(1, 3, 518, 518),
        dynamic_batch=True,
    )
    assert onnx_path.exists()

    session = ort.InferenceSession(str(onnx_path))

    for batch_size in (1, 2):
        x = torch.randn(batch_size, 3, 518, 518)
        with torch.no_grad():
            pt_out = loaded_model(x).numpy()

        # Must not raise (this is what regresses to `ScalesValidation` with
        # the old dynamic_axes-based exporter at batch_size=2).
        ort_out = session.run(None, {"input": x.numpy()})[0]

        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        assert np.allclose(pt_out, ort_out, atol=1e-3), (
            f"CLI-exported segmenter ONNX/PyTorch mismatch at "
            f"batch_size={batch_size}: max abs diff={max_diff}"
        )
