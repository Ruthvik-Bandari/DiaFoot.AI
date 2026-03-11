"""DiaFoot.AI v2 — Robustness Testing Suite.

Phase 4, Commit 22: Test model performance under synthetic degradations.

Degradation types:
    - Gaussian blur (simulates out-of-focus)
    - Gaussian noise (simulates sensor noise)
    - Brightness shift (simulates lighting variation)
    - Contrast reduction (simulates poor exposure)
    - JPEG compression (simulates storage artifacts)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_gaussian_blur(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    """Apply Gaussian blur at varying severity levels."""
    kernel_sizes = [3, 5, 7, 11, 15]
    k = kernel_sizes[min(severity - 1, len(kernel_sizes) - 1)]
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_gaussian_noise(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    """Apply Gaussian noise at varying severity levels."""
    std_devs = [5, 10, 20, 35, 50]
    std = std_devs[min(severity - 1, len(std_devs) - 1)]
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def apply_brightness_shift(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    """Apply brightness shift (darker or brighter)."""
    shifts = [-30, -50, -70, 50, 70]
    shift = shifts[min(severity - 1, len(shifts) - 1)]
    adjusted = np.clip(image.astype(np.int16) + shift, 0, 255)
    return adjusted.astype(np.uint8)


def apply_contrast_reduction(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    """Reduce image contrast."""
    factors = [0.8, 0.6, 0.4, 0.3, 0.2]
    factor = factors[min(severity - 1, len(factors) - 1)]
    mean = image.mean()
    adjusted = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255)
    return adjusted.astype(np.uint8)


def apply_jpeg_compression(
    image: np.ndarray,
    severity: int = 1,
) -> np.ndarray:
    """Apply JPEG compression artifacts."""
    qualities = [80, 60, 40, 20, 10]
    quality = qualities[min(severity - 1, len(qualities) - 1)]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# Registry of all degradations
DEGRADATIONS = {
    "gaussian_blur": apply_gaussian_blur,
    "gaussian_noise": apply_gaussian_noise,
    "brightness_shift": apply_brightness_shift,
    "contrast_reduction": apply_contrast_reduction,
    "jpeg_compression": apply_jpeg_compression,
}


def run_robustness_test(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    predict_fn: object,
    metric_fn: object,
    degradations: list[str] | None = None,
    severity_levels: list[int] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """Run robustness test across degradations and severity levels.

    Args:
        images: List of test images (H, W, 3).
        masks: List of ground truth masks (H, W).
        predict_fn: Function that takes image -> predicted mask.
        metric_fn: Function that takes (pred, target) -> dict of metrics.
        degradations: Which degradations to test (None = all).
        severity_levels: Which severity levels (default: 1-5).

    Returns:
        Nested dict: {degradation: {severity: {metric: value}}}.
    """
    degradations = degradations or list(DEGRADATIONS.keys())
    severity_levels = severity_levels or [1, 2, 3, 4, 5]

    results: dict[str, dict[int, dict[str, float]]] = {}

    # Baseline (no degradation)
    baseline_metrics: list[dict[str, float]] = []
    for img, mask in zip(images, masks, strict=False):
        pred = predict_fn(img)  # type: ignore[operator]
        baseline_metrics.append(metric_fn(pred, mask))  # type: ignore[operator]

    results["none"] = {
        0: _average_metrics(baseline_metrics),
    }

    # Each degradation x severity
    for deg_name in degradations:
        if deg_name not in DEGRADATIONS:
            logger.warning("Unknown degradation: %s", deg_name)
            continue

        deg_fn = DEGRADATIONS[deg_name]
        results[deg_name] = {}

        for severity in severity_levels:
            severity_metrics: list[dict[str, float]] = []
            for img, mask in zip(images, masks, strict=False):
                degraded = deg_fn(img, severity=severity)
                pred = predict_fn(degraded)  # type: ignore[operator]
                severity_metrics.append(metric_fn(pred, mask))  # type: ignore[operator]

            results[deg_name][severity] = _average_metrics(severity_metrics)

        logger.info("Completed robustness test: %s", deg_name)

    return results


def _average_metrics(
    metrics_list: list[dict[str, float]],
) -> dict[str, float]:
    """Average a list of metric dicts."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list if k in m])) for k in keys}


def print_robustness_report(
    results: dict[str, dict[int, dict[str, float]]],
    key_metric: str = "dice",
) -> None:
    """Print formatted robustness results."""
    print(f"\n{'=' * 70}")  # noqa: T201
    print(f"Robustness Test Results (metric: {key_metric})")  # noqa: T201
    print(f"{'=' * 70}")  # noqa: T201

    # Header
    severities = [1, 2, 3, 4, 5]
    header = f"{'Degradation':25s} | {'Base':6s}"
    for s in severities:
        header += f" | {'Sev' + str(s):6s}"
    print(header)  # noqa: T201
    print("-" * 70)  # noqa: T201

    for deg_name, sev_results in results.items():
        base = sev_results.get(0, {}).get(key_metric, 0)
        row = f"{deg_name:25s} | {base:6.4f}"
        for s in severities:
            val = sev_results.get(s, {}).get(key_metric, 0)
            drop = base - val if base > 0 else 0
            marker = " !" if drop > 0.1 else ""
            row += f" | {val:5.3f}{marker}"
        print(row)  # noqa: T201

    print(f"{'=' * 70}")  # noqa: T201
    print("  ! = >10% drop from baseline")  # noqa: T201
    print(f"{'=' * 70}\n")  # noqa: T201
