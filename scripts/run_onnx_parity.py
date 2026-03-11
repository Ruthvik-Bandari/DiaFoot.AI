"""DiaFoot.AI v2 — ONNX parity benchmark (PyTorch vs ONNXRuntime)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_val_transforms
from src.data.torch_dataset import DFUDataset
from src.models.unetpp import build_unetpp


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX parity benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--split-csv", type=str, default="data/splits/test.csv")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="results/onnx_parity_report.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("onnx_parity")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for parity benchmark") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_unetpp(encoder_name="efficientnet-b4", encoder_weights=None, classes=1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    ds = DFUDataset(args.split_csv, transform=get_val_transforms(), return_metadata=True)

    providers = ["CPUExecutionProvider"]
    if ort.get_device().upper() == "GPU":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(args.onnx, providers=providers)

    max_n = min(len(ds), args.max_samples)

    mae_values: list[float] = []
    max_abs_values: list[float] = []
    mask_agreement: list[float] = []
    pt_times_ms: list[float] = []
    onnx_times_ms: list[float] = []

    for i in range(max_n):
        sample = ds[i]
        image = sample["image"].unsqueeze(0)

        # PyTorch forward
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_logits = model(image.to(device)).cpu().numpy()
        pt_times_ms.append((time.perf_counter() - t0) * 1000)

        # ONNX forward
        t1 = time.perf_counter()
        onnx_logits = session.run(None, {"input": image.numpy()})[0]
        onnx_times_ms.append((time.perf_counter() - t1) * 1000)

        diff = np.abs(pt_logits - onnx_logits)
        mae_values.append(float(diff.mean()))
        max_abs_values.append(float(diff.max()))

        pt_mask = (1 / (1 + np.exp(-pt_logits)) > args.threshold).astype(np.uint8)
        onnx_mask = (1 / (1 + np.exp(-onnx_logits)) > args.threshold).astype(np.uint8)
        agree = float((pt_mask == onnx_mask).mean())
        mask_agreement.append(agree)

    report = {
        "num_samples": max_n,
        "mae": {
            "mean": float(np.mean(mae_values)) if mae_values else 0.0,
            "max": float(np.max(mae_values)) if mae_values else 0.0,
        },
        "max_abs_diff": {
            "mean": float(np.mean(max_abs_values)) if max_abs_values else 0.0,
            "max": float(np.max(max_abs_values)) if max_abs_values else 0.0,
        },
        "mask_agreement": {
            "mean": float(np.mean(mask_agreement)) if mask_agreement else 0.0,
            "min": float(np.min(mask_agreement)) if mask_agreement else 0.0,
        },
        "latency_ms": {
            "pytorch_mean": float(np.mean(pt_times_ms)) if pt_times_ms else 0.0,
            "onnx_mean": float(np.mean(onnx_times_ms)) if onnx_times_ms else 0.0,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("ONNX parity report saved to %s", out)
    logger.info("mask agreement mean: %.6f", report["mask_agreement"]["mean"])


if __name__ == "__main__":
    main()
