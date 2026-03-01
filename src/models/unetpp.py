"""DiaFoot.AI v2 — U-Net++ via Segmentation Models PyTorch.

Phase 2, Commit 9: Baseline single-task segmentation model.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn  # noqa: TC002


def build_unetpp(
    encoder_name: str = "efficientnet-b4",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    decoder_attention_type: str | None = "scse",
    deep_supervision: bool = True,
) -> nn.Module:
    """Build a U-Net++ model via SMP.

    Args:
        encoder_name: Encoder backbone name (from timm/SMP).
        encoder_weights: Pretrained weights ('imagenet' or None).
        in_channels: Number of input channels.
        classes: Number of output segmentation classes.
        decoder_attention_type: Attention type ('scse' or None).
        deep_supervision: Enable deep supervision for better gradients.

    Returns:
        SMP UnetPlusPlus model.
    """
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        decoder_attention_type=decoder_attention_type,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
    )
