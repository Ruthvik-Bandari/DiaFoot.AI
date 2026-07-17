"""DiaFoot.AI v2 — Test-Time Augmentation.

Phase 4, Commit 23: TTA for improved predictions + uncertainty proxy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _d4_transforms(
    num_augmentations: int,
) -> list[tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]]:
    """Return (forward, inverse) transform pairs for D4-symmetry TTA.

    Provides up to 8 distinct dihedral-group-of-the-square views: identity, three
    rotations, two axis reflections, and two diagonal reflections. Each inverse
    undoes its forward transform so spatial predictions realign before averaging
    (reflections are their own inverse; rot90/rot270 invert to rot270/rot90).

    Note rot180 is the same operation as flipping both axes (hvflip), so it is
    represented once — requesting 8 augmentations yields 8 genuinely distinct
    views rather than a duplicate.
    """

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    def hflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [3])

    def vflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [2])

    def hvflip(x: torch.Tensor) -> torch.Tensor:  # equivalent to rot180
        return torch.flip(x, [2, 3])

    def rot90(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, 1, [2, 3])

    def rot90_inv(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, -1, [2, 3])

    def rot270(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, 3, [2, 3])

    def rot270_inv(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, -3, [2, 3])

    def transpose(x: torch.Tensor) -> torch.Tensor:  # main-diagonal reflection
        return x.transpose(2, 3)

    def anti_transpose(x: torch.Tensor) -> torch.Tensor:  # anti-diagonal reflection
        return torch.flip(x.transpose(2, 3), [2, 3])

    pairs: list[
        tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]
    ] = [
        (identity, identity),
        (hflip, hflip),
        (vflip, vflip),
        (hvflip, hvflip),
        (rot90, rot90_inv),
        (rot270, rot270_inv),
        (transpose, transpose),
        (anti_transpose, anti_transpose),
    ]
    n = max(1, min(num_augmentations, len(pairs)))
    return pairs[:n]


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

    with torch.no_grad():
        for fwd_fn, inv_fn in _d4_transforms(num_augmentations):
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

    with torch.no_grad():
        # Classification is orientation-invariant, so the inverse is unused.
        for fwd_fn, _inv_fn in _d4_transforms(num_augmentations):
            logits = model(fwd_fn(image))
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
