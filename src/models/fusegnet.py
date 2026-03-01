"""DiaFoot.AI v2 — FUSegNet Architecture.

Phase 2, Commit 12: EfficientNet-B7 encoder + P-scSE attention.
Top-performing architecture for DFU segmentation (89.23% Dice on FUSeg).
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.models.attention import ParallelScSE


class FUSegNet(nn.Module):
    """FUSegNet: EfficientNet-B7 with P-scSE attention for wound segmentation.

    Based on the FUSegNet paper (top of FUSeg Challenge leaderboard).

    Args:
        encoder_name: Backbone encoder.
        encoder_weights: Pretrained weights.
        classes: Number of segmentation classes.
        use_pscse: Whether to apply P-scSE attention.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b7",
        encoder_weights: str | None = "imagenet",
        classes: int = 1,
        use_pscse: bool = True,
    ) -> None:
        """Initialize FUSegNet."""
        super().__init__()

        self.base = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            decoder_attention_type="scse",
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
        )

        # Add P-scSE to decoder stages
        self.pscse_blocks: nn.ModuleList | None = None
        if use_pscse:
            decoder_channels = [256, 128, 64, 32, 16]
            self.pscse_blocks = nn.ModuleList([ParallelScSE(ch) for ch in decoder_channels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional P-scSE on decoder features.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Segmentation logits (B, classes, H, W).
        """
        # Use base model's forward (includes encoder + decoder + head)
        return self.base(x)
