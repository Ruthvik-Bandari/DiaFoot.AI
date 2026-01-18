"""
Training Callbacks Module
==========================

Callbacks for training monitoring, early stopping, and checkpointing.

Author: Ruthvik
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer: Any) -> None:
        pass
    
    def on_train_end(self, trainer: Any) -> None:
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        pass
    
    def on_batch_begin(self, trainer: Any, batch_idx: int) -> None:
        pass
    
    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        logs: Dict[str, float]
    ) -> None:
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.
    Stops training when a monitored metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            restore_best_weights: Whether to restore weights from best epoch
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_weights = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == "min":
            return current < (self.best_value - self.min_delta)
        else:
            return current > (self.best_value + self.min_delta)
    
    def on_train_begin(self, trainer: Any) -> None:
        self.wait = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.stop_training = False
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self._is_improvement(current):
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone()
                    for k, v in trainer.model.state_dict().items()
                }
            
            if self.verbose:
                print(f"  EarlyStopping: {self.monitor} improved to {current:.6f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"  EarlyStopping: No improvement for {self.wait}/{self.patience} epochs")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                trainer.stop_training = True
                
                if self.verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch}")
    
    def on_train_end(self, trainer: Any) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"Restored best weights from epoch {self.best_epoch}")


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints during training.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
        save_every_n_epochs: Optional[int] = None,
        filename_format: str = "checkpoint_epoch{epoch:03d}_{monitor:.4f}.pt",
        verbose: bool = True,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max'
            save_best_only: Only save when monitored metric improves
            save_last: Always save last epoch
            save_every_n_epochs: Save every N epochs
            filename_format: Format string for checkpoint filenames
            verbose: Whether to print messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.filename_format = filename_format
        self.verbose = verbose
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_path = None
        
        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value
        else:
            return current > self.best_value
    
    def _save_checkpoint(
        self,
        trainer: Any,
        epoch: int,
        logs: Dict[str, float],
        filename: str,
    ) -> str:
        """Save a checkpoint."""
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metrics": logs,
            "config": getattr(trainer, "config", {}),
        }
        
        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()
        
        if hasattr(trainer, "scaler") and trainer.scaler is not None:
            checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        return str(filepath)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        current = logs.get(self.monitor, 0)
        
        # Check if this is the best model
        if self._is_improvement(current):
            self.best_value = current
            
            # Save best model
            best_filename = f"best_model.pt"
            self.best_path = self._save_checkpoint(trainer, epoch, logs, best_filename)
            
            if self.verbose:
                print(f"  Checkpoint: Saved best model ({self.monitor}={current:.4f})")
        
        # Save periodic checkpoints
        if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
            filename = self.filename_format.format(epoch=epoch, monitor=current)
            self._save_checkpoint(trainer, epoch, logs, filename)
            
            if self.verbose:
                print(f"  Checkpoint: Saved periodic checkpoint")
        
        # Save last model
        if self.save_last:
            self._save_checkpoint(trainer, epoch, logs, "last_model.pt")


class LearningRateLogger(Callback):
    """
    Callback to log learning rate during training.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.lr_history = []
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        if hasattr(trainer, "optimizer"):
            lr = trainer.optimizer.param_groups[0]["lr"]
            self.lr_history.append(lr)
            logs["learning_rate"] = lr
            
            if self.verbose:
                print(f"  Learning rate: {lr:.2e}")


class MetricsLogger(Callback):
    """
    Callback to log and save training metrics.
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.metrics_history: Dict[str, List[float]] = {}
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        for metric_name, value in logs.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(value)
        
        # Save metrics to JSON
        metrics_path = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def on_train_end(self, trainer: Any) -> None:
        # Save final summary
        summary_path = self.log_dir / f"{self.experiment_name}_summary.json"
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_epochs": len(self.metrics_history.get("train_loss", [])),
            "best_metrics": {
                metric: {
                    "min": min(values),
                    "max": max(values),
                    "final": values[-1] if values else None,
                }
                for metric, values in self.metrics_history.items()
            },
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


class WandBLogger(Callback):
    """
    Weights & Biases logging callback.
    """
    
    def __init__(
        self,
        project: str = "diafootai",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_model: bool = True,
    ):
        self.project = project
        self.name = name
        self.config = config
        self.log_model = log_model
        self.run = None
    
    def on_train_begin(self, trainer: Any) -> None:
        try:
            import wandb
            
            self.run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
            )
            
            # Log model architecture
            if hasattr(trainer, "model"):
                wandb.watch(trainer.model, log_freq=100)
        except ImportError:
            print("wandb not installed. Skipping W&B logging.")
            self.run = None
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        if self.run is not None:
            import wandb
            wandb.log(logs, step=epoch)
    
    def on_train_end(self, trainer: Any) -> None:
        if self.run is not None:
            import wandb
            
            if self.log_model and hasattr(trainer, "model"):
                # Save model artifact
                artifact = wandb.Artifact("model", type="model")
                model_path = Path("temp_model.pt")
                torch.save(trainer.model.state_dict(), model_path)
                artifact.add_file(str(model_path))
                self.run.log_artifact(artifact)
                model_path.unlink()
            
            wandb.finish()


class ProgressCallback(Callback):
    """
    Rich progress bar callback for training.
    """
    
    def __init__(self, total_epochs: int, total_batches: int):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.progress = None
        self.epoch_task = None
        self.batch_task = None
    
    def on_train_begin(self, trainer: Any) -> None:
        try:
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
            
            self.progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            )
            self.progress.start()
            self.epoch_task = self.progress.add_task(
                "Training", total=self.total_epochs
            )
        except ImportError:
            self.progress = None
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        if self.progress is not None:
            self.batch_task = self.progress.add_task(
                f"Epoch {epoch + 1}/{self.total_epochs}",
                total=self.total_batches,
            )
    
    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        logs: Dict[str, float]
    ) -> None:
        if self.progress is not None and self.batch_task is not None:
            self.progress.update(self.batch_task, advance=1)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        if self.progress is not None:
            self.progress.update(self.epoch_task, advance=1)
            if self.batch_task is not None:
                self.progress.remove_task(self.batch_task)
    
    def on_train_end(self, trainer: Any) -> None:
        if self.progress is not None:
            self.progress.stop()


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs)
    
    def on_batch_begin(self, trainer: Any, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx)
    
    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        logs: Dict[str, float]
    ) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, logs)
