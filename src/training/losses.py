"""
Loss Functions Module
======================

State-of-the-art loss functions for wound segmentation and classification.
Includes Dice, Focal, Lovász, Tversky, and boundary-aware losses.

Author: Ruthvik
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    Handles class imbalance well.
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        square: bool = False,
    ):
        super().__init__()
        self.smooth = smooth
        self.square = square
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C, H, W) - after sigmoid
            target: Targets (B, C, H, W) or (B, 1, H, W)
        """
        # Apply sigmoid if not already applied
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Compute Dice
        intersection = (pred_flat * target_flat).sum(dim=1)
        
        if self.square:
            union = (pred_flat ** 2).sum(dim=1) + (target_flat ** 2).sum(dim=1)
        else:
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy Loss.
    Best default for medical image segmentation.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # BCE on logits
        bce = self.bce_loss(pred, target)
        
        # Dice on sigmoid(pred)
        dice = self.dice_loss(torch.sigmoid(pred), target)
        
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        For binary classification/segmentation.
        pred: logits (not sigmoid)
        target: binary labels
        """
        # Sigmoid
        p = torch.sigmoid(pred)
        
        # Compute focal weight
        ce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        p_t = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice with adjustable FP/FN weights.
    Good for imbalanced datasets where false negatives are more important.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,  # Weight for false positives
        beta: float = 0.7,   # Weight for false negatives
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - combines Tversky with focal weighting.
    Excellent for highly imbalanced segmentation.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,  # Focal parameter
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for sharper wound edges.
    Computes distance transform and weighs boundary pixels more.
    """
    
    def __init__(
        self,
        theta0: float = 3.0,
        theta: float = 5.0,
    ):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dist_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted segmentation
            target: Ground truth
            dist_map: Pre-computed distance transform (optional)
        """
        pred = torch.sigmoid(pred)
        
        if dist_map is None:
            # Compute simple boundary weight based on target edges
            # (For proper distance transform, compute offline)
            kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
            
            boundary = F.conv2d(target, kernel, padding=1).abs()
            boundary = (boundary > 0).float()
            
            # Weight boundary pixels more
            weight_map = 1 + self.theta * boundary
        else:
            weight_map = 1 + self.theta * torch.exp(-dist_map / self.theta0)
        
        # Weighted BCE
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        weighted_bce = (bce * weight_map).mean()
        
        return weighted_bce


class LovaszHingeLoss(nn.Module):
    """
    Lovász-Hinge Loss for binary segmentation.
    Directly optimizes IoU/Jaccard metric.
    """
    
    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred.view(-1) if not self.per_image else pred
        target = target.view(-1) if not self.per_image else target
        
        if self.per_image:
            loss = 0
            for p, t in zip(pred, target):
                loss += self._lovasz_hinge_flat(p.view(-1), t.view(-1))
            return loss / pred.size(0)
        else:
            return self._lovasz_hinge_flat(pred.view(-1), target.view(-1))
    
    def _lovasz_hinge_flat(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Lovász hinge loss for flattened tensors."""
        signs = 2 * target - 1
        errors = 1 - pred * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = target[perm]
        
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        
        return loss
    
    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute Lovász gradient."""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1 - intersection / union
        
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        
        return jaccard


class CombinedLoss(nn.Module):
    """
    Combine multiple losses with configurable weights.
    """
    
    def __init__(
        self,
        losses: List[Tuple[nn.Module, float]],
    ):
        """
        Args:
            losses: List of (loss_fn, weight) tuples
        """
        super().__init__()
        
        self.losses = nn.ModuleList([loss for loss, _ in losses])
        self.weights = [weight for _, weight in losses]
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with label smoothing for classification.
    Improves model calibration and generalization.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_target = torch.full_like(pred, self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        # Cross entropy
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(smooth_target * log_probs).sum(dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def create_segmentation_loss(config: dict) -> nn.Module:
    """Create segmentation loss from config."""
    loss_config = config.get("training", {}).get("loss", {}).get("segmentation", {})
    
    loss_name = loss_config.get("name", "dice_bce")
    
    if loss_name == "dice":
        return DiceLoss()
    elif loss_name == "dice_bce":
        return DiceBCELoss(
            dice_weight=loss_config.get("dice_weight", 0.5),
            bce_weight=loss_config.get("bce_weight", 0.5),
        )
    elif loss_name == "focal":
        return FocalLoss(
            alpha=loss_config.get("alpha", 0.25),
            gamma=loss_config.get("gamma", 2.0),
        )
    elif loss_name == "tversky":
        return TverskyLoss(
            alpha=loss_config.get("alpha", 0.3),
            beta=loss_config.get("beta", 0.7),
        )
    elif loss_name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=loss_config.get("alpha", 0.3),
            beta=loss_config.get("beta", 0.7),
            gamma=loss_config.get("gamma", 0.75),
        )
    elif loss_name == "combined":
        # DiceBCE + Boundary
        return CombinedLoss([
            (DiceBCELoss(), 0.7),
            (BoundaryLoss(), 0.3),
        ])
    else:
        return DiceBCELoss()


def create_classification_loss(config: dict) -> nn.Module:
    """Create classification loss from config."""
    loss_config = config.get("training", {}).get("loss", {}).get("classification", {})
    
    loss_name = loss_config.get("name", "cross_entropy")
    smoothing = loss_config.get("label_smoothing", 0.1)
    
    if loss_name == "cross_entropy" and smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "focal":
        return FocalLoss(
            alpha=loss_config.get("alpha", 0.25),
            gamma=loss_config.get("gamma", 2.0),
        )
    else:
        return nn.CrossEntropyLoss()
