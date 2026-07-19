"""DiaFoot.AI v2 — Shared segmentation evaluation loop.

Canonical home for the "run a segmentation model over a test loader and
collect per-image metrics" loop that was previously copy-pasted between
``scripts/evaluate.py`` (``evaluate_segmentation``) and
``scripts/evaluate_all.py`` (``evaluate_checkpoint``). Extracted so a single
implementation exists instead of duplicated inference loops.
"""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation.metrics import compute_segmentation_metrics


def run_segmentation_eval(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[list[dict], list[int]]:
    """Run a segmentation model over a loader and collect per-image metrics.

    Handles both plain-tensor model outputs and multi-task models that return
    a dict (the segmentation logits are read from the ``seg_logits`` key).
    Predictions are thresholded at ``sigmoid > 0.5``, matching the historical
    behaviour of both callers.

    Args:
        model: Segmentation model. Moved to ``device`` and set to eval mode.
        test_loader: DataLoader yielding batches with ``image``, ``mask`` and
            ``label`` keys (``return_metadata=True`` on ``DFUDataset``).
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).

    Returns:
        A ``(metrics, labels)`` tuple of equal length, where ``metrics[i]`` is
        the :func:`compute_segmentation_metrics` result for image ``i`` and
        ``labels[i]`` is its integer class label. Callers slice these lists to
        build per-class subsets (e.g. DFU-only where ``label == 2``).
    """
    model = model.to(device).eval()

    metrics: list[dict] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            batch_labels = batch["label"].numpy()

            logits = model(images)
            if isinstance(logits, dict):
                logits = logits.get("seg_logits", logits)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

            for i in range(len(images)):
                metrics.append(compute_segmentation_metrics(preds[i], masks[i]))
                labels.append(int(batch_labels[i]))

    return metrics, labels
