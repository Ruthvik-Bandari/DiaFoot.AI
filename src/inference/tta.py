"""DiaFoot.AI v2 — Test-Time Augmentation.

Phase 4, Commit 23: TTA for improved predictions + uncertainty proxy.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def tta_predict_segmentation(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device | str = "cpu",
    num_augmentations: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TTA for segmentation: apply flips/rotations, average predictions."""
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    predictions = []

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    def hflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [3])

    def vflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [2])

    def hvflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [2, 3])

    augmentations: list[tuple] = [
        (identity, identity),
        (hflip, hflip),
        (vflip, vflip),
        (hvflip, hvflip),
    ]

    if num_augmentations >= 8:

        def rot90(x: torch.Tensor) -> torch.Tensor:
            return torch.rot90(x, 1, [2, 3])

        def rot90_inv(x: torch.Tensor) -> torch.Tensor:
            return torch.rot90(x, -1, [2, 3])

        def rot180(x: torch.Tensor) -> torch.Tensor:
            return torch.rot90(x, 2, [2, 3])

        def rot270(x: torch.Tensor) -> torch.Tensor:
            return torch.rot90(x, 3, [2, 3])

        def rot270_inv(x: torch.Tensor) -> torch.Tensor:
            return torch.rot90(x, -3, [2, 3])

        augmentations.extend(
            [
                (rot90, rot90_inv),
                (rot180, rot180),
                (rot270, rot270_inv),
            ]
        )

    with torch.no_grad():
        for fwd_fn, inv_fn in augmentations[:num_augmentations]:
            augmented = fwd_fn(image)
            output = model(augmented)
            if isinstance(output, dict):
                output = output.get("seg_logits", output)
            prob = torch.sigmoid(output)
            restored = inv_fn(prob)
            predictions.append(restored.squeeze().cpu().numpy())

    stacked = np.stack(predictions, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def tta_predict_classification(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device | str = "cpu",
    num_augmentations: int = 8,
) -> tuple[np.ndarray, float]:
    """Run TTA for classification: average probs, compute entropy."""
    model.eval()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    all_probs = []

    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.flip(x, [2, 3]),
    ]
    if num_augmentations >= 8:
        transforms.extend(
            [
                lambda x: torch.rot90(x, 1, [2, 3]),
                lambda x: torch.rot90(x, 2, [2, 3]),
                lambda x: torch.rot90(x, 3, [2, 3]),
            ]
        )

    with torch.no_grad():
        for aug_fn in transforms[:num_augmentations]:
            logits = model(aug_fn(image))
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.squeeze().cpu().numpy())

    stacked = np.stack(all_probs, axis=0)
    mean_probs = stacked.mean(axis=0)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
    max_entropy = np.log(len(mean_probs))
    uncertainty = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    return mean_probs, uncertainty


def compute_tta_improvement(
    base_metrics: dict[str, float],
    tta_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute improvement from TTA over base predictions."""
    improvements: dict[str, float] = {}
    for key in base_metrics:
        if key in tta_metrics:
            diff = tta_metrics[key] - base_metrics[key]
            rel = diff / max(abs(base_metrics[key]), 1e-8) * 100
            improvements[f"{key}_abs_improvement"] = diff
            improvements[f"{key}_rel_improvement_pct"] = rel
    return improvements
