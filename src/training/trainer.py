"""
Trainer Module
===============

Production-grade training loops for wound segmentation and classification.
Features: Mixed precision, EMA, gradient accumulation, differential LR.

Author: Ruthvik
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import DiceBCELoss, FocalLoss, create_segmentation_loss
from .callbacks import (
    Callback, CallbackList, EarlyStopping, 
    ModelCheckpoint, LearningRateLogger, MetricsLogger
)
from .optimizers import get_optimizer, get_scheduler, ExponentialMovingAverage


class Trainer:
    """
    Base trainer class with modern training features.
    
    Features:
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation for effective large batches
    - Exponential Moving Average (EMA)
    - Differential learning rates
    - Comprehensive callbacks
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.train_config = config.get("training", {})
        
        # Device setup
        if device is None:
            device = self._get_device()
        self.device = device
        
        # Model
        self.model = model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        self.criterion = criterion or self._create_criterion()
        
        # Optimizer
        self.optimizer = optimizer or get_optimizer(model, config)
        
        # Scheduler
        self.scheduler = scheduler or get_scheduler(
            self.optimizer, config, len(train_loader)
        )
        
        # Mixed precision
        self.use_amp = self.train_config.get("mixed_precision", {}).get("enabled", True)
        self.scaler = GradScaler() if self.use_amp and self.device.type == "cuda" else None
        
        # Gradient accumulation
        self.accumulation_steps = self.train_config.get("accumulation_steps", 1)
        
        # EMA
        self.use_ema = self.train_config.get("ema", {}).get("enabled", True)
        self.ema = None
        if self.use_ema:
            decay = self.train_config.get("ema", {}).get("decay", 0.999)
            self.ema = ExponentialMovingAverage(model, decay=decay, device=self.device)
        
        # Gradient clipping
        self.grad_clip = self.train_config.get("gradient_clip", {})
        self.clip_grad = self.grad_clip.get("enabled", True)
        self.max_grad_norm = self.grad_clip.get("max_norm", 1.0)
        
        # Callbacks
        self.callbacks = CallbackList(callbacks or [])
        self._setup_default_callbacks()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.stop_training = False
        
        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
    
    def _get_device(self) -> torch.device:
        """Determine best available device."""
        device_name = self.train_config.get("device", "auto")
        
        if device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        
        return torch.device(device_name)
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function from config."""
        return create_segmentation_loss(self.config)
    
    def _setup_default_callbacks(self):
        """Add default callbacks if not present."""
        callback_types = [type(cb) for cb in self.callbacks.callbacks]
        
        # Early stopping
        if EarlyStopping not in callback_types:
            es_config = self.train_config.get("early_stopping", {})
            if es_config.get("enabled", True):
                self.callbacks.add(EarlyStopping(
                    monitor=es_config.get("monitor", "val_loss"),
                    patience=es_config.get("patience", 20),
                    min_delta=es_config.get("min_delta", 0.001),
                    mode=es_config.get("mode", "min"),
                ))
        
        # Model checkpoint
        if ModelCheckpoint not in callback_types:
            ckpt_config = self.train_config.get("checkpoint", {})
            checkpoint_dir = Path(self.config.get("paths", {}).get("checkpoints", "models/checkpoints"))
            self.callbacks.add(ModelCheckpoint(
                checkpoint_dir=str(checkpoint_dir),
                monitor=ckpt_config.get("monitor", "val_loss"),
                mode=ckpt_config.get("mode", "min"),
                save_best_only=ckpt_config.get("save_best", True),
                save_last=ckpt_config.get("save_last", True),
            ))
        
        # Learning rate logger
        if LearningRateLogger not in callback_types:
            self.callbacks.add(LearningRateLogger())
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )
        
        for batch_idx, batch in enumerate(pbar):
            self.callbacks.on_batch_begin(self, batch_idx)
            
            # Move data to device
            images = batch["image"].to(self.device)
            targets = batch["mask"].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.clip_grad:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update EMA
                    if self.ema is not None:
                        self.ema.update()
            else:
                # Standard precision
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.ema is not None:
                        self.ema.update()
            
            # Update metrics
            total_loss += loss.item() * self.accumulation_steps
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item() * self.accumulation_steps:.4f}"})
            
            self.callbacks.on_batch_end(
                self, batch_idx, {"batch_loss": loss.item() * self.accumulation_steps}
            )
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        # Use EMA weights for validation if available
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        num_batches = len(self.val_loader)
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch["image"].to(self.device)
            targets = batch["mask"].to(self.device)
            
            if self.use_amp and self.device.type == "cuda":
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Compute metrics
            preds = torch.sigmoid(outputs) > 0.5
            iou, dice = self._compute_metrics(preds, targets)
            total_iou += iou
            total_dice += dice
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore()
        
        return {
            "val_loss": total_loss / num_batches,
            "val_iou": total_iou / num_batches,
            "val_dice": total_dice / num_batches,
        }
    
    def _compute_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-6,
    ) -> tuple:
        """Compute IoU and Dice metrics."""
        preds = preds.float()
        targets = targets.float()
        
        # Flatten
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        # Intersection and union
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum() - intersection
        
        # IoU
        iou = (intersection + smooth) / (union + smooth)
        
        # Dice
        dice = (2 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
        
        return iou.item(), dice.item()
    
    def fit(
        self,
        epochs: Optional[int] = None,
        start_epoch: int = 0,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs (uses config if None)
            start_epoch: Starting epoch for resumed training
        
        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.train_config.get("epochs", 100)
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Accumulation steps: {self.accumulation_steps}")
        print(f"Effective batch size: {self.train_loader.batch_size * self.accumulation_steps}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"EMA: {self.use_ema}")
        print(f"{'='*60}\n")
        
        self.callbacks.on_train_begin(self)
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            self.callbacks.on_epoch_begin(self, epoch)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update history
            for key, value in epoch_metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_metrics.get("val_loss", train_metrics["train_loss"]))
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            self._print_epoch_summary(epoch, epochs, epoch_metrics)
            
            # Callbacks
            self.callbacks.on_epoch_end(self, epoch, epoch_metrics)
            
            # Check for early stopping
            if self.stop_training:
                print(f"\nTraining stopped at epoch {epoch + 1}")
                break
        
        self.callbacks.on_train_end(self)
        
        return self.history
    
    def _print_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        metrics: Dict[str, float],
    ):
        """Print epoch summary."""
        summary = f"Epoch {epoch + 1}/{total_epochs}"
        for key, value in metrics.items():
            summary += f" | {key}: {value:.4f}"
        print(summary)
    
    def save_checkpoint(self, path: Union[str, Path], extra: Optional[Dict] = None):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        
        if extra:
            checkpoint.update(extra)
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        if "ema_state_dict" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.history = checkpoint.get("history", self.history)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


class SegmentationTrainer(Trainer):
    """
    Specialized trainer for wound segmentation.
    Adds segmentation-specific metrics and loss functions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs,
    ):
        # Default to DiceBCE loss for segmentation
        if "criterion" not in kwargs:
            kwargs["criterion"] = DiceBCELoss()
        
        super().__init__(model, config, train_loader, val_loader, **kwargs)
        
        # Additional segmentation-specific settings
        self.threshold = config.get("inference", {}).get("confidence_threshold", 0.5)
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate with comprehensive segmentation metrics."""
        if self.val_loader is None:
            return {}
        
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_precision = 0.0
        total_recall = 0.0
        num_batches = len(self.val_loader)
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch["image"].to(self.device)
            targets = batch["mask"].to(self.device)
            
            if self.use_amp and self.device.type == "cuda":
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Compute metrics
            preds = (torch.sigmoid(outputs) > self.threshold).float()
            metrics = self._compute_seg_metrics(preds, targets)
            
            total_iou += metrics["iou"]
            total_dice += metrics["dice"]
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
        
        if self.ema is not None:
            self.ema.restore()
        
        return {
            "val_loss": total_loss / num_batches,
            "val_iou": total_iou / num_batches,
            "val_dice": total_dice / num_batches,
            "val_precision": total_precision / num_batches,
            "val_recall": total_recall / num_batches,
        }
    
    def _compute_seg_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-6,
    ) -> Dict[str, float]:
        """Compute comprehensive segmentation metrics."""
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        # True positives, false positives, false negatives
        tp = (preds_flat * targets_flat).sum()
        fp = (preds_flat * (1 - targets_flat)).sum()
        fn = ((1 - preds_flat) * targets_flat).sum()
        
        # Metrics
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        
        intersection = tp
        union = tp + fp + fn
        
        iou = (intersection + smooth) / (union + smooth)
        dice = (2 * intersection + smooth) / (2 * intersection + fp + fn + smooth)
        
        return {
            "iou": iou.item(),
            "dice": dice.item(),
            "precision": precision.item(),
            "recall": recall.item(),
        }


class ClassificationTrainer(Trainer):
    """
    Specialized trainer for tissue/infection classification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_classes: int = 6,
        **kwargs,
    ):
        self.num_classes = num_classes
        
        # Default to CrossEntropy for classification
        if "criterion" not in kwargs:
            from .losses import LabelSmoothingCrossEntropy
            smoothing = config.get("training", {}).get("loss", {}).get(
                "classification", {}
            ).get("label_smoothing", 0.1)
            kwargs["criterion"] = LabelSmoothingCrossEntropy(smoothing=smoothing)
        
        super().__init__(model, config, train_loader, val_loader, **kwargs)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )
        
        for batch_idx, batch in enumerate(pbar):
            self.callbacks.on_batch_begin(self, batch_idx)
            
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            if self.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.clip_grad:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.ema is not None:
                        self.ema.update()
            else:
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.ema is not None:
                        self.ema.update()
            
            # Accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            total_loss += loss.item() * self.accumulation_steps
            self.global_step += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item() * self.accumulation_steps:.4f}",
                "acc": f"{100. * correct / total:.2f}%"
            })
            
            self.callbacks.on_batch_end(
                self, batch_idx, {"batch_loss": loss.item() * self.accumulation_steps}
            )
        
        return {
            "train_loss": total_loss / num_batches,
            "train_acc": correct / total,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # For per-class metrics
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            if self.use_amp and self.device.type == "cuda":
                with autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += predicted[i].eq(labels[i]).item()
                class_total[label] += 1
        
        if self.ema is not None:
            self.ema.restore()
        
        metrics = {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": correct / total,
        }
        
        # Per-class accuracy
        for i in range(self.num_classes):
            if class_total[i] > 0:
                metrics[f"val_acc_class_{i}"] = class_correct[i] / class_total[i]
        
        return metrics
