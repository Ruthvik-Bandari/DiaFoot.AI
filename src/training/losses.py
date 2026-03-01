"""DiaFoot.AI v2 — Segmentation Loss Functions.

Phase 3, Commit 13: Compound losses for wound segmentation.

Loss strategies:
    1. Dice + CE (baseline warm-start)
    2. Dice + Boundary Loss (primary — with alpha ramp)
    3. Focal Tversky Loss (v1 reproduction)
    4. Unified Focal Loss (ablation)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class DiceLoss(nn.Module):
    """Soft Dice loss for binary/multi-class segmentation.

    Args:
        smooth: Smoothing factor to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        """Initialize Dice loss."""
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            pred: Predicted logits (B, C, H, W) or (B, 1, H, W).
            target: Ground truth (B, H, W) as long tensor.

        Returns:
            Scalar Dice loss (1 - Dice coefficient).
        """
        if pred.shape[1] == 1:
            pred_soft = torch.sigmoid(pred).squeeze(1)
            target_flat = target.float()
        else:
            pred_soft = torch.softmax(pred, dim=1)
            target_flat = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2).float()
            # Average across classes
            intersection = (pred_soft * target_flat).sum(dim=(0, 2, 3))
            union = pred_soft.sum(dim=(0, 2, 3)) + target_flat.sum(dim=(0, 2, 3))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice.mean()

        intersection = (pred_soft * target_flat).sum()
        union = pred_soft.sum() + target_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class DiceCELoss(nn.Module):
    """Combined Dice + Cross Entropy loss.

    Args:
        dice_weight: Weight for Dice component.
        ce_weight: Weight for CE component.
    """

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        """Initialize Dice + CE loss."""
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        dice_loss = self.dice(pred, target)
        # CE expects (B, C, H, W) pred and (B, H, W) target
        if pred.shape[1] == 1:
            ce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(1), target.float())
        else:
            ce_loss = self.ce(pred, target)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss (v1 approach, for comparison).

    Addresses class imbalance by down-weighting easy examples.

    Args:
        alpha: Weight for false positives.
        beta: Weight for false negatives.
        gamma: Focal parameter (higher = more focus on hard examples).
        smooth: Smoothing factor.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1.0,
    ) -> None:
        """Initialize Focal Tversky loss."""
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal Tversky loss."""
        pred_soft = (
            torch.sigmoid(pred.squeeze(1))
            if pred.shape[1] == 1
            else torch.softmax(pred, dim=1)[:, 1]
        )
        target_flat = target.float()

        tp = (pred_soft * target_flat).sum()
        fp = ((1 - target_flat) * pred_soft).sum()
        fn = (target_flat * (1 - pred_soft)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky) ** self.gamma


class DiceBoundaryLoss(nn.Module):
    """Dice + Boundary Loss with warm-start alpha ramp.

    Strategy: Start with Dice+CE, gradually introduce Boundary Loss.
    - Epochs 1-warmup: Dice + CE only (alpha=0)
    - Epochs warmup-max: Linear ramp of alpha for Boundary Loss

    Args:
        warmup_epoch: Epoch to start introducing boundary loss.
        max_epoch: Epoch at which alpha reaches max.
        alpha_max: Maximum boundary loss weight.
    """

    def __init__(
        self,
        warmup_epoch: int = 30,
        max_epoch: int = 100,
        alpha_max: float = 1.0,
    ) -> None:
        """Initialize Dice + Boundary loss."""
        super().__init__()
        self.dice_ce = DiceCELoss()
        self.warmup_epoch = warmup_epoch
        self.max_epoch = max_epoch
        self.alpha_max = alpha_max
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for alpha scheduling."""
        self._current_epoch = epoch

    def _get_alpha(self) -> float:
        """Compute current boundary loss weight."""
        if self._current_epoch < self.warmup_epoch:
            return 0.0
        progress = (self._current_epoch - self.warmup_epoch) / max(
            1, self.max_epoch - self.warmup_epoch
        )
        return min(self.alpha_max, progress * self.alpha_max)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss with alpha-ramped boundary component."""
        dice_ce_loss = self.dice_ce(pred, target)
        alpha = self._get_alpha()

        if alpha > 0:
            # Simple boundary approximation: loss on edge pixels
            # Full boundary loss from MONAI would be used in production
            if target.dim() == 3:
                # Compute edges using morphological gradient
                target_float = target.float().unsqueeze(1)
                kernel_size = 3
                padding = kernel_size // 2
                dilated = F.max_pool2d(target_float, kernel_size, stride=1, padding=padding)
                eroded = -F.max_pool2d(-target_float, kernel_size, stride=1, padding=padding)
                boundary = (dilated - eroded).squeeze(1)

                pred_sig = (
                    torch.sigmoid(pred.squeeze(1))
                    if pred.shape[1] == 1
                    else torch.softmax(pred, dim=1)[:, 1]
                )
                boundary_loss = F.binary_cross_entropy(
                    pred_sig * boundary, target.float() * boundary, reduction="mean"
                )
                return (1 - alpha) * dice_ce_loss + alpha * boundary_loss

        return dice_ce_loss
