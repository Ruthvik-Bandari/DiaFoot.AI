"""DiaFoot.AI v2 — Multi-Task U-Net++ (Classify + Segment + Stage).

Phase 2, Commit 9: Shared encoder with three task heads.

Architecture:
    Shared EfficientNet encoder
    ├── Classification head (GAP -> FC -> 3 classes)
    ├── Segmentation decoder (U-Net++ -> wound mask)
    └── Staging head (bottleneck features -> Wagner grade 0-5)
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.models.attention import ParallelScSE


class MultiTaskUNetPP(nn.Module):
    """Multi-task U-Net++ with shared encoder.

    Three output heads:
    1. Classification: Healthy | Non-DFU | DFU (from encoder features)
    2. Segmentation: Binary wound mask (from decoder)
    3. Staging: Wagner grade 0-5 (from bottleneck features)

    Args:
        encoder_name: Backbone name for SMP.
        encoder_weights: Pretrained weights.
        seg_classes: Number of segmentation output channels.
        cls_classes: Number of classification classes.
        stage_classes: Number of Wagner grades.
        dropout: Dropout rate for classification heads.
        use_pscse: Use P-scSE attention in decoder.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str | None = "imagenet",
        seg_classes: int = 1,
        cls_classes: int = 3,
        stage_classes: int = 6,
        dropout: float = 0.3,
        use_pscse: bool = True,
    ) -> None:
        """Initialize multi-task U-Net++."""
        super().__init__()

        # Build base U-Net++
        self.base = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=seg_classes,
            decoder_attention_type="scse",
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
        )

        # Get encoder output dimension
        dummy = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            features = self.base.encoder(dummy)
        encoder_dim = features[-1].shape[1]

        # P-scSE attention on bottleneck
        self.pscse: nn.Module | None = None
        if use_pscse:
            self.pscse = ParallelScSE(encoder_dim)

        # Classification head (from encoder bottleneck)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(encoder_dim, cls_classes),
        )

        # Staging head (from encoder bottleneck, for DFU images only)
        self.stage_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(encoder_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, stage_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning all three task outputs.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Dict with keys:
                - seg_logits: (B, seg_classes, H, W) segmentation logits
                - cls_logits: (B, cls_classes) classification logits
                - stage_logits: (B, stage_classes) staging logits
        """
        # Shared encoder
        features = self.base.encoder(x)
        bottleneck = features[-1]  # Deepest feature map

        # Apply P-scSE to bottleneck
        bottleneck_attn = self.pscse(bottleneck) if self.pscse is not None else bottleneck

        # Segmentation (full decoder)
        seg_logits = self.base.decoder(*features)
        seg_logits = self.base.segmentation_head(seg_logits)

        # Classification (from bottleneck)
        cls_logits = self.cls_head(bottleneck_attn)

        # Staging (from bottleneck)
        stage_logits = self.stage_head(bottleneck_attn)

        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
            "stage_logits": stage_logits,
        }
