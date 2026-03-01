"""DiaFoot.AI v2 — Attention Modules.

Phase 2, Commit 9: P-scSE (Parallel Spatial-Channel Squeeze & Excitation).
Based on FUSegNet paper — proven effective for wound boundary detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelSE(nn.Module):
    """Channel Squeeze & Excitation block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize channel SE with reduction ratio."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpatialSE(nn.Module):
    """Spatial Squeeze & Excitation block."""

    def __init__(self, channels: int) -> None:
        """Initialize spatial SE."""
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention."""
        w = self.sigmoid(self.conv(x))
        return x * w


class ParallelScSE(nn.Module):
    """Parallel Spatial-Channel Squeeze & Excitation (P-scSE).

    Combines channel and spatial attention via element-wise max
    instead of addition (as proposed in FUSegNet).

    Args:
        channels: Number of input channels.
        reduction: Channel SE reduction ratio.
        use_max: If True, use max fusion. If False, use additive fusion.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        use_max: bool = True,
    ) -> None:
        """Initialize P-scSE with channel and spatial attention."""
        super().__init__()
        self.channel_se = ChannelSE(channels, reduction)
        self.spatial_se = SpatialSE(channels)
        self.use_max = use_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parallel spatial-channel attention."""
        ch_out = self.channel_se(x)
        sp_out = self.spatial_se(x)
        if self.use_max:
            return torch.max(ch_out, sp_out)
        return ch_out + sp_out
