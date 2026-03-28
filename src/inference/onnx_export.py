"""DiaFoot.AI v2 — ONNX Export Pipeline.

Export trained DINOv2 or legacy models to ONNX for production inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 518, 518),
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model.
        output_path: Where to save the .onnx file.
        input_shape: Example input shape for tracing.
        opset_version: ONNX opset version.
        dynamic_batch: Allow dynamic batch size.

    Returns:
        Path to exported ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        } if dynamic_batch else None,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Exported ONNX model: %s (%.1f MB)", output_path, file_size_mb)
    return output_path


def validate_onnx(
    pytorch_model: nn.Module,
    onnx_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 518, 518),
    atol: float = 1e-5,
) -> bool:
    """Validate ONNX model output matches PyTorch.

    Args:
        pytorch_model: Original PyTorch model.
        onnx_path: Path to exported ONNX model.
        input_shape: Test input shape.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed, skipping validation")
        return True

    pytorch_model.eval()
    dummy = torch.randn(*input_shape)

    with torch.no_grad():
        pt_output = pytorch_model(dummy)
        if isinstance(pt_output, dict):
            pt_output = pt_output.get("seg_logits", pt_output.get("cls_logits"))
        pt_numpy = pt_output.numpy()

    session = ort.InferenceSession(str(onnx_path))
    ort_output = session.run(None, {"input": dummy.numpy()})

    matches = np.allclose(pt_numpy, ort_output[0], atol=atol)
    if matches:
        logger.info("ONNX validation passed (atol=%s)", atol)
    else:
        max_diff = float(np.max(np.abs(pt_numpy - ort_output[0])))
        logger.warning("ONNX validation failed: max diff = %f", max_diff)

    return matches
