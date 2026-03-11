"""DiaFoot.AI v2 — Classifier shortcut-learning audit utilities."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def perturb_border_noise(
    image: np.ndarray,
    border_fraction: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """Replace border pixels with noise while preserving center region."""
    out = image.copy()
    h, w = out.shape[:2]
    by = max(1, int(h * border_fraction))
    bx = max(1, int(w * border_fraction))

    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=out.shape, dtype=np.uint8)

    out[:by, :] = noise[:by, :]
    out[h - by :, :] = noise[h - by :, :]
    out[:, :bx] = noise[:, :bx]
    out[:, w - bx :] = noise[:, w - bx :]
    return out


def keep_center_only(
    image: np.ndarray,
    border_fraction: float = 0.15,
) -> np.ndarray:
    """Zero out border area and keep center untouched."""
    out = np.zeros_like(image)
    h, w = image.shape[:2]
    by = max(1, int(h * border_fraction))
    bx = max(1, int(w * border_fraction))
    out[by : h - by, bx : w - bx] = image[by : h - by, bx : w - bx]
    return out


def blur_background(
    image: np.ndarray,
    border_fraction: float = 0.15,
) -> np.ndarray:
    """Blur border while keeping center region unchanged."""
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    out = blurred.copy()
    h, w = image.shape[:2]
    by = max(1, int(h * border_fraction))
    bx = max(1, int(w * border_fraction))
    out[by : h - by, bx : w - bx] = image[by : h - by, bx : w - bx]
    return out


def summarize_shortcut_shift(
    labels: np.ndarray,
    baseline_pred: np.ndarray,
    baseline_conf: np.ndarray,
    perturbed_pred: np.ndarray,
    perturbed_conf: np.ndarray,
) -> dict[str, Any]:
    """Summarize prediction and confidence shifts under perturbation."""
    labels = labels.astype(int)
    baseline_pred = baseline_pred.astype(int)
    perturbed_pred = perturbed_pred.astype(int)

    baseline_acc = float((baseline_pred == labels).mean()) if len(labels) > 0 else 0.0
    perturbed_acc = float((perturbed_pred == labels).mean()) if len(labels) > 0 else 0.0
    pred_consistency = float((baseline_pred == perturbed_pred).mean()) if len(labels) > 0 else 0.0

    conf_drop = baseline_conf - perturbed_conf
    return {
        "n": len(labels),
        "baseline_accuracy": baseline_acc,
        "perturbed_accuracy": perturbed_acc,
        "accuracy_drop": baseline_acc - perturbed_acc,
        "prediction_consistency": pred_consistency,
        "confidence_drop_mean": float(conf_drop.mean()) if len(conf_drop) > 0 else 0.0,
        "confidence_drop_median": float(np.median(conf_drop)) if len(conf_drop) > 0 else 0.0,
        "confidence_drop_p90": float(np.percentile(conf_drop, 90)) if len(conf_drop) > 0 else 0.0,
    }
