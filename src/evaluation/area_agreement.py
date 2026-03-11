"""DiaFoot.AI v2 — Wound area agreement metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_area_agreement(pred_area: np.ndarray, manual_area: np.ndarray) -> dict[str, Any]:
    """Compute agreement metrics between predicted and manual wound areas."""
    pred = pred_area.astype(float)
    gt = manual_area.astype(float)
    err = pred - gt
    abs_err = np.abs(err)

    mae = float(abs_err.mean()) if len(abs_err) else 0.0
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else 0.0
    mape = float((abs_err / np.maximum(gt, 1e-6)).mean()) if len(abs_err) else 0.0
    bias = float(err.mean()) if len(err) else 0.0

    # Correlation-based agreement proxy
    corr = float(np.corrcoef(pred, gt)[0, 1]) if len(pred) > 1 else 0.0

    return {
        "n": int(len(pred)),
        "mae_mm2": mae,
        "rmse_mm2": rmse,
        "mape": mape,
        "bias_mm2": bias,
        "pearson_r": corr,
    }
