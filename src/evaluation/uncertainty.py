"""DiaFoot.AI v2 — Uncertainty Quantification.

Phase 5, Commit 24: Pixel-wise uncertainty via MC Dropout and ensemble variance.

Methods:
    - MC Dropout: Run multiple forward passes with dropout enabled
    - Ensemble: Use variance across multiple trained models
    - Conformal prediction: Statistical coverage guarantees for wound area
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC Dropout.

    Keeps BatchNorm in eval mode but switches Dropout to train mode.

    Args:
        model: The model to modify in-place.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout | nn.Dropout2d):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device | str = "cpu",
    num_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MC Dropout inference for uncertainty estimation.

    Performs multiple stochastic forward passes with dropout enabled,
    then computes mean prediction and pixel-wise uncertainty.

    Args:
        model: Segmentation model with dropout layers.
        image: Input image (1, C, H, W) or (C, H, W).
        device: Computation device.
        num_samples: Number of stochastic forward passes.

    Returns:
        Tuple of (mean_prediction, uncertainty_map, all_samples).
        mean_prediction: (H, W) averaged probability map.
        uncertainty_map: (H, W) standard deviation per pixel.
        all_samples: (num_samples, H, W) individual predictions.
    """
    model.eval()
    enable_mc_dropout(model)

    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    samples: list[np.ndarray] = []

    with torch.no_grad():
        for _ in range(num_samples):
            output = model(image)
            if isinstance(output, dict):
                output = output.get("seg_logits", output)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            samples.append(prob)

    stacked = np.stack(samples, axis=0)
    mean_pred = stacked.mean(axis=0)
    uncertainty = stacked.std(axis=0)

    return mean_pred, uncertainty, stacked


def ensemble_predict(
    models: list[nn.Module],
    image: torch.Tensor,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run ensemble inference for uncertainty estimation.

    Uses disagreement between independently trained models as
    uncertainty measure.

    Args:
        models: List of trained segmentation models.
        image: Input image (1, C, H, W) or (C, H, W).
        device: Device.

    Returns:
        Tuple of (mean_prediction, uncertainty_map).
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    predictions: list[np.ndarray] = []

    for model in models:
        model.eval()
        model.to(device)
        with torch.no_grad():
            output = model(image)
            if isinstance(output, dict):
                output = output.get("seg_logits", output)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            predictions.append(prob)

    stacked = np.stack(predictions, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def conformal_wound_area(
    calibration_areas: np.ndarray,
    calibration_gt_areas: np.ndarray,
    test_pred_area: float,
    alpha: float = 0.1,
) -> tuple[float, float]:
    """Compute conformal prediction interval for wound area.

    Provides a statistical guarantee that the true wound area falls
    within the interval with probability >= (1 - alpha).

    Args:
        calibration_areas: Predicted areas on calibration set.
        calibration_gt_areas: Ground truth areas on calibration set.
        test_pred_area: Predicted area for test image.
        alpha: Significance level (0.1 = 90% coverage).

    Returns:
        Tuple of (lower_bound, upper_bound) in same units as input.
    """
    residuals = np.abs(calibration_areas - calibration_gt_areas)
    n = len(residuals)
    quantile_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
    quantile_idx = min(quantile_idx, n - 1)
    sorted_residuals = np.sort(residuals)
    q_hat = sorted_residuals[quantile_idx]

    lower = max(0.0, test_pred_area - q_hat)
    upper = test_pred_area + q_hat

    return float(lower), float(upper)


def compute_uncertainty_metrics(
    mean_pred: np.ndarray,
    uncertainty: np.ndarray,
    ground_truth: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute summary statistics for uncertainty map.

    Args:
        mean_pred: Mean predicted probability map (H, W).
        uncertainty: Uncertainty map (H, W).
        ground_truth: Optional ground truth for calibration analysis.

    Returns:
        Dict with uncertainty statistics.
    """
    binary_pred = (mean_pred > 0.5).astype(bool)

    metrics: dict[str, float] = {
        "mean_uncertainty": float(uncertainty.mean()),
        "max_uncertainty": float(uncertainty.max()),
        "median_uncertainty": float(np.median(uncertainty)),
    }

    # Uncertainty at wound boundaries (most clinically relevant)
    if binary_pred.any():
        from scipy.ndimage import binary_erosion

        boundary = binary_pred ^ binary_erosion(binary_pred, iterations=1)
        if boundary.any():
            metrics["boundary_uncertainty_mean"] = float(uncertainty[boundary].mean())
            metrics["boundary_uncertainty_max"] = float(uncertainty[boundary].max())

    # Uncertainty inside vs outside wound
    if binary_pred.any() and (~binary_pred).any():
        metrics["wound_uncertainty"] = float(uncertainty[binary_pred].mean())
        metrics["background_uncertainty"] = float(uncertainty[~binary_pred].mean())

    # Correlation with errors (if ground truth available)
    if ground_truth is not None:
        errors = (binary_pred != ground_truth.astype(bool)).astype(float)
        if errors.std() > 0 and uncertainty.std() > 0:
            correlation = np.corrcoef(errors.flatten(), uncertainty.flatten())[0, 1]
            metrics["uncertainty_error_correlation"] = float(correlation)

    return metrics


def print_uncertainty_report(metrics: dict[str, float]) -> None:
    """Print formatted uncertainty analysis."""
    print(f"\n{'=' * 60}")  # noqa: T201
    print("Uncertainty Analysis")  # noqa: T201
    print(f"{'=' * 60}")  # noqa: T201
    for key, value in metrics.items():
        print(f"  {key:35s}: {value:.4f}")  # noqa: T201
    print(f"{'=' * 60}\n")  # noqa: T201
