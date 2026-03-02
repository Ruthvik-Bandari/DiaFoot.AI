"""DiaFoot.AI v2 — Single-Task Training Loop.

Includes: BFloat16, gradient clipping, checkpointing, early stopping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader  # noqa: TC002

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    precision: str = "bf16-mixed"
    compile_model: bool = False
    ema_enabled: bool = False
    ema_decay: float = 0.999
    checkpoint_dir: str = "checkpoints"
    monitor_metric: str = "val/dice"
    monitor_mode: str = "max"
    save_top_k: int = 3
    device: str = "cuda"
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4


class Trainer:
    """Single-task trainer with early stopping."""

    def __init__(self, model: nn.Module, config: TrainConfig | None = None) -> None:
        """Initialize trainer."""
        self.config = config or TrainConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.best_metric = float("-inf") if self.config.monitor_mode == "max" else float("inf")
        self.patience_counter = 0

    def _get_targets(self, batch: dict[str, Any]) -> torch.Tensor:
        """Get correct targets based on task type."""
        if self.config.monitor_metric == "val/accuracy":
            return batch["label"].to(self.device)
        return batch["mask"].to(self.device)

    def _is_better(self, current: float) -> bool:
        """Check if current metric is better than best."""
        if self.config.monitor_mode == "max":
            return current > self.best_metric + self.config.early_stopping_min_delta
        return current < self.best_metric - self.config.early_stopping_min_delta

    def train_epoch(
        self,
        train_loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        epoch: int,
    ) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        use_amp = self.config.precision == "bf16-mixed" and torch.cuda.is_available()

        for batch in train_loader:
            images = batch["image"].to(self.device)
            targets = self._get_targets(batch)
            optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs.get("seg_logits", outputs.get("cls_logits"))
                    loss = loss_fn(outputs, targets)
            else:
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs.get("seg_logits", outputs.get("cls_logits"))
                loss = loss_fn(outputs, targets)

            loss.backward()
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return {"train/loss": total_loss / max(1, num_batches)}

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader[Any],
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # For Dice computation
        dice_scores: list[float] = []

        for batch in val_loader:
            images = batch["image"].to(self.device)
            targets = self._get_targets(batch)
            outputs = self.model(images)
            if isinstance(outputs, dict):
                outputs = outputs.get("seg_logits", outputs.get("cls_logits"))
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

            # Classification accuracy
            if targets.dim() == 1:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            else:
                # Compute batch Dice for segmentation
                pred_mask = (torch.sigmoid(outputs) > 0.5).float()
                if pred_mask.dim() == 4:
                    pred_mask = pred_mask.squeeze(1)
                target_f = targets.float()
                for i in range(pred_mask.shape[0]):
                    p = pred_mask[i].flatten()
                    t = target_f[i].flatten()
                    inter = (p * t).sum()
                    dice = (2.0 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
                    dice_scores.append(dice.item())

        metrics: dict[str, float] = {
            "val/loss": total_loss / max(1, num_batches),
        }
        if total > 0:
            metrics["val/accuracy"] = correct / total
        if dice_scores:
            metrics["val/dice"] = sum(dice_scores) / len(dice_scores)

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        optimizer: torch.optim.Optimizer,
    ) -> Path | None:
        """Save checkpoint if metric improved."""
        current = metrics.get(self.config.monitor_metric, 0.0)

        if not self._is_better(current):
            self.patience_counter += 1
            return None

        self.best_metric = current
        self.patience_counter = 0
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"best_epoch{epoch:03d}_{current:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info("Saved checkpoint: %s (metric=%.4f)", path, current)
        return path

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None = None,
    ) -> dict[str, list[float]]:
        """Full training loop with early stopping."""
        history: dict[str, list[float]] = {}
        patience = self.config.early_stopping_patience

        logger.info(
            "Starting training: %d epochs, device=%s, precision=%s, patience=%d",
            self.config.epochs,
            self.device,
            self.config.precision,
            patience,
        )

        for epoch in range(self.config.epochs):
            t0 = time.time()
            if hasattr(loss_fn, "set_epoch"):
                loss_fn.set_epoch(epoch)

            train_metrics = self.train_epoch(train_loader, optimizer, loss_fn, epoch)
            val_metrics = self.validate(val_loader, loss_fn)
            metrics = {**train_metrics, **val_metrics}

            for k, v in metrics.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)

            self.save_checkpoint(epoch, metrics, optimizer)

            if scheduler is not None and hasattr(scheduler, "step"):
                scheduler.step()

            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            parts = [
                f"Epoch {epoch + 1}/{self.config.epochs}",
                f"loss={train_metrics['train/loss']:.4f}",
                f"val_loss={val_metrics['val/loss']:.4f}",
            ]
            if "val/dice" in val_metrics:
                parts.append(f"dice={val_metrics['val/dice']:.4f}")
            if "val/accuracy" in val_metrics:
                parts.append(f"acc={val_metrics['val/accuracy']:.4f}")
            parts.append(f"lr={lr:.2e}")
            parts.append(f"{elapsed:.1f}s")
            logger.info(" | ".join(parts))

            # Early stopping check
            if self.patience_counter >= patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1,
                    patience,
                )
                break

        logger.info("Training complete. Best metric: %.4f", self.best_metric)
        return history
