"""
Optimizers and Schedulers Module
==================================

Factory functions for creating optimizers and learning rate schedulers.

Author: Ruthvik
"""

import math
from typing import Dict, Any, Optional, Iterator, List
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
    differential_lr: bool = False,
    encoder_lr_factor: float = 0.1,
) -> Optimizer:
    """
    Create optimizer from config.
    
    Args:
        model: Model to optimize
        config: Training configuration
        differential_lr: Use different LR for encoder vs decoder
        encoder_lr_factor: Factor to multiply encoder LR by (if differential_lr)
    
    Returns:
        Configured optimizer
    """
    opt_config = config.get("training", {}).get("optimizer", {})
    
    name = opt_config.get("name", "adamw").lower()
    lr = opt_config.get("lr", 1e-4)
    weight_decay = opt_config.get("weight_decay", 1e-4)
    
    # Setup parameter groups
    if differential_lr and hasattr(model, "get_encoder_params"):
        param_groups = [
            {
                "params": model.get_encoder_params(),
                "lr": lr * encoder_lr_factor,
                "name": "encoder",
            },
            {
                "params": model.get_decoder_params(),
                "lr": lr,
                "name": "decoder",
            },
        ]
    else:
        param_groups = [{"params": model.parameters()}]
    
    # Create optimizer
    if name == "adamw":
        betas = tuple(opt_config.get("betas", [0.9, 0.999]))
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif name == "adam":
        betas = tuple(opt_config.get("betas", [0.9, 0.999]))
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif name == "sgd":
        momentum = opt_config.get("momentum", 0.9)
        nesterov = opt_config.get("nesterov", True)
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif name == "lion":
        # Lion optimizer (newer, efficient)
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
            )
        except ImportError:
            print("lion-pytorch not installed, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
            )
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizer


def get_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any],
    steps_per_epoch: Optional[int] = None,
) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer to schedule
        config: Training configuration
        steps_per_epoch: Number of steps per epoch (for OneCycleLR)
    
    Returns:
        Configured scheduler or None
    """
    sched_config = config.get("training", {}).get("scheduler", {})
    
    name = sched_config.get("name", "cosine_annealing_warm_restarts").lower()
    
    if name == "none" or name is None:
        return None
    
    epochs = config.get("training", {}).get("epochs", 100)
    
    if name == "cosine_annealing_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_config.get("T_0", 10),
            T_mult=sched_config.get("T_mult", 2),
            eta_min=sched_config.get("eta_min", 1e-7),
        )
    
    elif name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=sched_config.get("eta_min", 1e-7),
        )
    
    elif name == "one_cycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        
        max_lr = sched_config.get("max_lr", optimizer.param_groups[0]["lr"] * 10)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=sched_config.get("pct_start", 0.3),
            anneal_strategy=sched_config.get("anneal_strategy", "cos"),
        )
    
    elif name == "step":
        step_size = sched_config.get("step_size", 30)
        gamma = sched_config.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    
    elif name == "multi_step":
        milestones = sched_config.get("milestones", [30, 60, 90])
        gamma = sched_config.get("gamma", 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    
    elif name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_config.get("mode", "min"),
            factor=sched_config.get("factor", 0.1),
            patience=sched_config.get("patience", 10),
            min_lr=sched_config.get("min_lr", 1e-7),
            verbose=True,
        )
    
    elif name == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=sched_config.get("warmup_epochs", 5),
            total_epochs=epochs,
            min_lr=sched_config.get("eta_min", 1e-7),
        )
    
    elif name == "polynomial":
        return PolynomialLRDecay(
            optimizer,
            total_epochs=epochs,
            power=sched_config.get("power", 0.9),
            min_lr=sched_config.get("min_lr", 1e-7),
        )
    
    else:
        print(f"Unknown scheduler: {name}, using CosineAnnealingWarmRestarts")
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
        )


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    Standard in modern training pipelines.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class PolynomialLRDecay(_LRScheduler):
    """
    Polynomial learning rate decay.
    Used in DeepLab and other segmentation models.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        power: float = 0.9,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch >= self.total_epochs:
            return [self.min_lr for _ in self.base_lrs]
        
        decay_factor = (1 - self.last_epoch / self.total_epochs) ** self.power
        return [
            self.min_lr + (base_lr - self.min_lr) * decay_factor
            for base_lr in self.base_lrs
        ]


class ExponentialMovingAverage:
    """
    Exponential Moving Average of model parameters.
    Improves model generalization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.decay = decay
        self.device = device
        
        # Store shadow parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)
    
    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self) -> None:
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self) -> None:
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state for saving."""
        return {
            "shadow": self.shadow,
            "decay": self.decay,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load EMA state."""
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)
