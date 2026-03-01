"""DiaFoot.AI v2 — Classification Loss Functions.

Phase 3, Commit 13: Losses for triage classifier and Wagner staging.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class FocalLoss(nn.Module):
    """Focal Loss for classification with class imbalance.

    Down-weights well-classified examples, focuses on hard ones.

    Args:
        alpha: Class weighting (None for uniform).
        gamma: Focusing parameter (0 = standard CE, 2 = typical).
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> None:
        """Initialize Focal Loss."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Logits (B, C).
            target: Class indices (B,).

        Returns:
            Scalar focal loss.
        """
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(pred.device)
            at = alpha.gather(0, target)
            focal_loss = at * focal_loss

        return focal_loss.mean()


class LabelSmoothingCE(nn.Module):
    """Cross Entropy with Label Smoothing for Wagner staging.

    Soft targets for ambiguous grades (e.g., Grade 1 vs 2 boundary).

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing).
        num_classes: Number of classes.
    """

    def __init__(self, smoothing: float = 0.1, num_classes: int = 6) -> None:
        """Initialize label smoothing CE."""
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed CE loss."""
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)

        # Create soft targets
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1.0)
        soft_targets = one_hot * confidence + (1 - one_hot) * smooth_val

        log_probs = F.log_softmax(pred, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1)
        return loss.mean()
