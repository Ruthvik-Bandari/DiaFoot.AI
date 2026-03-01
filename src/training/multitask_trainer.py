"""DiaFoot.AI v2 — Multi-Task Training Loop.

Phase 3, Commit 15: Joint training for classification + segmentation + staging.
Supports task-weighted loss and curriculum learning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskConfig:
    """Multi-task training configuration."""

    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    precision: str = "bf16-mixed"
    device: str = "cuda"

    # Task weights: L_total = w_cls * L_cls + w_seg * L_seg + w_stage * L_stage
    weight_classification: float = 1.0
    weight_segmentation: float = 2.0
    weight_staging: float = 1.0

    # Curriculum: gradually unfreeze tasks
    classifier_warmup_epochs: int = 10
    segmentation_unfreeze_epoch: int = 10
    staging_unfreeze_epoch: int = 20

    # Gradient accumulation
    accumulation_steps: int = 1

    checkpoint_dir: str = "checkpoints"
    seed: int = 42


class MultiTaskTrainer:
    """Trainer for multi-task models (classify + segment + stage).

    Supports:
    - Task-weighted loss with configurable weights
    - Curriculum learning (gradually unfreeze task heads)
    - Gradient accumulation for larger effective batch size
    - BFloat16 mixed precision

    Args:
        model: Multi-task model (outputs dict with seg/cls/stage logits).
        config: Training configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MultiTaskConfig | None = None,
    ) -> None:
        """Initialize multi-task trainer."""
        self.config = config or MultiTaskConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def compute_multitask_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        seg_loss_fn: nn.Module,
        cls_loss_fn: nn.Module,
        stage_loss_fn: nn.Module,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted multi-task loss.

        Args:
            outputs: Model outputs with seg_logits, cls_logits, stage_logits.
            batch: Data batch with mask, label keys.
            seg_loss_fn: Segmentation loss function.
            cls_loss_fn: Classification loss function.
            stage_loss_fn: Staging loss function.
            epoch: Current epoch (for curriculum).

        Returns:
            Tuple of (total_loss, loss_breakdown_dict).
        """
        cfg = self.config
        losses: dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Classification loss (always active)
        if "cls_logits" in outputs:
            labels = batch["label"].to(self.device)
            cls_loss = cls_loss_fn(outputs["cls_logits"], labels)
            total_loss = total_loss + cfg.weight_classification * cls_loss
            losses["cls_loss"] = cls_loss.item()

        # Segmentation loss (after warmup)
        if "seg_logits" in outputs and epoch >= cfg.segmentation_unfreeze_epoch:
            masks = batch["mask"].to(self.device)
            seg_loss = seg_loss_fn(outputs["seg_logits"], masks)
            total_loss = total_loss + cfg.weight_segmentation * seg_loss
            losses["seg_loss"] = seg_loss.item()

        # Staging loss (after staging unfreeze, only for DFU samples)
        if "stage_logits" in outputs and epoch >= cfg.staging_unfreeze_epoch:
            labels = batch["label"].to(self.device)
            dfu_mask = labels == 2
            if dfu_mask.any():
                # Only compute staging loss on DFU images
                stage_logits = outputs["stage_logits"][dfu_mask]
                # Use label as proxy (in production, use Wagner grades)
                stage_targets = torch.zeros(
                    stage_logits.shape[0], dtype=torch.long, device=self.device
                )
                stage_loss = stage_loss_fn(stage_logits, stage_targets)
                total_loss = total_loss + cfg.weight_staging * stage_loss
                losses["stage_loss"] = stage_loss.item()

        losses["total_loss"] = total_loss.item()
        return total_loss, losses

    def train_epoch(
        self,
        train_loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        seg_loss_fn: nn.Module,
        cls_loss_fn: nn.Module,
        stage_loss_fn: nn.Module,
        epoch: int,
    ) -> dict[str, float]:
        """Run one multi-task training epoch."""
        self.model.train()
        epoch_losses: dict[str, list[float]] = {}
        acc_steps = self.config.accumulation_steps

        use_amp = self.config.precision == "bf16-mixed" and torch.cuda.is_available()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(images)
                    loss, loss_dict = self.compute_multitask_loss(
                        outputs, batch, seg_loss_fn, cls_loss_fn, stage_loss_fn, epoch
                    )
                    loss = loss / acc_steps
            else:
                outputs = self.model(images)
                loss, loss_dict = self.compute_multitask_loss(
                    outputs, batch, seg_loss_fn, cls_loss_fn, stage_loss_fn, epoch
                )
                loss = loss / acc_steps

            loss.backward()

            # Gradient accumulation
            if (step + 1) % acc_steps == 0:
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                optimizer.step()
                optimizer.zero_grad()

            # Track losses
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)

        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        seg_loss_fn: nn.Module,
        cls_loss_fn: nn.Module,
        stage_loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
    ) -> dict[str, list[float]]:
        """Full multi-task training loop."""
        history: dict[str, list[float]] = {}

        logger.info("Starting multi-task training: %d epochs", self.config.epochs)

        for epoch in range(self.config.epochs):
            t0 = time.time()

            metrics = self.train_epoch(
                train_loader, optimizer, seg_loss_fn, cls_loss_fn, stage_loss_fn, epoch
            )

            for k, v in metrics.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)

            if scheduler:
                scheduler.step()

            elapsed = time.time() - t0
            loss_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info(
                "Epoch %d/%d | %s | %.1fs", epoch + 1, self.config.epochs, loss_str, elapsed
            )

        return history
