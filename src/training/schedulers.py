"""DiaFoot.AI v2 — Learning Rate Schedulers.

Phase 3, Commit 13: Cosine annealing with linear warmup.
"""

from __future__ import annotations

import math

from torch import Tensor  # noqa: TC002
from torch.optim import Optimizer  # noqa: TC002
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithWarmup(_LRScheduler):
    """Cosine annealing scheduler with linear warmup.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs.
        max_epochs: Total training epochs.
        eta_min: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        eta_min: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        """Initialize scheduler."""
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup. Use (last_epoch + 1) so epoch 0 starts at a non-zero
            # fraction of base_lr and reaches base_lr at the final warmup epoch:
            # last_epoch starts at 0 (step() runs once in __init__) and Trainer
            # trains before stepping, so `last_epoch / warmup_epochs` would waste
            # epoch 0 at LR=0.
            alpha = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]

        # Cosine decay. Clamp progress to [0, 1]: once training reaches or passes
        # max_epochs, hold at progress=1 (LR = eta_min). Without the clamp,
        # cos(pi * progress) turns back upward for progress > 1 and the LR
        # rebounds, undoing the anneal if training runs past max_epochs.
        progress = min(
            1.0,
            (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs),
        )
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs
        ]
