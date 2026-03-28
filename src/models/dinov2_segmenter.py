"""DiaFoot.AI v2 — DINOv2 Segmentation Model.

Transfer learning with Meta's DINOv2 as the encoder backbone for wound
segmentation. Uses DINOv2 frozen patch tokens + lightweight UPerNet-style
decoder for dense prediction.

Architecture:
    DINOv2 ViT backbone (frozen + optional LoRA)
    → Multi-scale patch token features (extracted from layers 3, 6, 9, 12)
    → UPerNet-style FPN decoder with lateral connections
    → 1×1 conv → Binary wound mask

DINOv2's self-supervised features capture rich semantic structure (wound
boundaries, texture, morphology) without learning dataset-specific shortcuts.
"""

from __future__ import annotations

import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# DINOv2 backbone configs: (embed_dim, num_heads, num_layers)
DINOV2_CONFIGS = {
    "dinov2_vits14": {"embed_dim": 384, "num_layers": 12, "patch_size": 14},
    "dinov2_vitb14": {"embed_dim": 768, "num_layers": 12, "patch_size": 14},
    "dinov2_vitl14": {"embed_dim": 1024, "num_layers": 24, "patch_size": 14},
}


class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PPM(nn.Module):
    """Pyramid Pooling Module (from PSPNet) for global context."""

    def __init__(self, in_ch: int, out_ch: int, pool_sizes: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        for size in pool_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.bottleneck = ConvBNReLU(in_ch + out_ch * len(pool_sizes), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feats = [x]
        for stage in self.stages:
            feats.append(F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(feats, dim=1))


class UPerNetDecoder(nn.Module):
    """UPerNet-style decoder for multi-scale ViT features.

    Takes features from multiple transformer layers, projects them to a common
    channel dimension, applies FPN-style lateral connections, and produces
    a dense prediction map.

    Args:
        embed_dim: ViT embedding dimension.
        decoder_dim: Internal decoder channel dimension.
        num_classes: Number of output segmentation classes (1 for binary).
    """

    def __init__(
        self,
        embed_dim: int,
        decoder_dim: int = 256,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        # Lateral projections (one per feature level)
        self.lateral_convs = nn.ModuleList([
            ConvBNReLU(embed_dim, decoder_dim, kernel_size=1)
            for _ in range(4)
        ])

        # FPN smoothing convolutions
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(decoder_dim, decoder_dim)
            for _ in range(4)
        ])

        # Pyramid pooling on deepest features
        self.ppm = PPM(embed_dim, decoder_dim)

        # Fusion after concatenating all FPN levels
        self.fusion = ConvBNReLU(decoder_dim * 5, decoder_dim)

        # Final segmentation head
        self.seg_head = nn.Sequential(
            ConvBNReLU(decoder_dim, decoder_dim),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(
        self,
        multi_scale_features: list[torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            multi_scale_features: List of 4 feature maps [layer_3, layer_6, layer_9, layer_12],
                                  each of shape (B, H_feat, W_feat, embed_dim).
            target_size: (H, W) output spatial dimensions.

        Returns:
            Segmentation logits (B, num_classes, H, W).
        """
        # Reshape from (B, H*W, C) → (B, C, H, W) if needed
        feat_maps = []
        for feat in multi_scale_features:
            if feat.dim() == 3:
                b, hw, c = feat.shape
                h = w = int(hw ** 0.5)
                feat = feat.reshape(b, h, w, c).permute(0, 3, 1, 2)
            feat_maps.append(feat)

        # PPM on deepest feature
        ppm_out = self.ppm(feat_maps[-1])

        # FPN: lateral connections + top-down pathway
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feat_maps)]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=(h, w), mode="bilinear", align_corners=False
            )

        # FPN smoothing
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Upsample all to same resolution (largest feature map)
        target_feat_size = fpn_outs[0].shape[2:]
        upsampled = [fpn_outs[0]]
        for fpn_out in fpn_outs[1:]:
            upsampled.append(
                F.interpolate(fpn_out, size=target_feat_size, mode="bilinear", align_corners=False)
            )
        upsampled.append(
            F.interpolate(ppm_out, size=target_feat_size, mode="bilinear", align_corners=False)
        )

        # Fuse all levels
        fused = self.fusion(torch.cat(upsampled, dim=1))

        # Segmentation head
        seg_logits = self.seg_head(fused)

        # Upsample to input resolution
        seg_logits = F.interpolate(seg_logits, size=target_size, mode="bilinear", align_corners=False)

        return seg_logits


class DINOv2Segmenter(nn.Module):
    """DINOv2-based segmentation model for wound segmentation.

    Uses DINOv2 ViT as a frozen feature extractor with multi-scale feature
    extraction from intermediate transformer layers, fed into a UPerNet decoder.

    Args:
        backbone: DINOv2 model name.
        num_classes: Output segmentation classes (1 for binary wound mask).
        decoder_dim: Decoder channel dimension.
        freeze_backbone: Freeze backbone parameters.
        use_lora: Apply LoRA adapters to backbone attention layers.
        lora_rank: LoRA rank.
        lora_alpha: LoRA scaling factor.
        feature_layers: Which transformer layers to extract features from.
    """

    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        num_classes: int = 1,
        decoder_dim: int = 256,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        feature_layers: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize DINOv2 segmenter."""
        super().__init__()

        if backbone not in DINOV2_CONFIGS:
            msg = f"Unsupported backbone: {backbone}. Choose from {list(DINOV2_CONFIGS)}"
            raise ValueError(msg)

        config = DINOV2_CONFIGS[backbone]
        self.embed_dim = config["embed_dim"]
        self.patch_size = config["patch_size"]
        num_layers = config["num_layers"]
        self.backbone_name = backbone

        # Default feature extraction layers (evenly spaced through the network)
        if feature_layers is None:
            if num_layers == 12:
                self.feature_layers = (2, 5, 8, 11)  # layers 3, 6, 9, 12 (0-indexed)
            else:  # 24 layers (ViT-L)
                self.feature_layers = (5, 11, 17, 23)
        else:
            self.feature_layers = feature_layers

        # Load DINOv2 backbone
        self.encoder = torch.hub.load("facebookresearch/dinov2", backbone)
        logger.info("Loaded DINOv2 backbone: %s (embed_dim=%d)", backbone, self.embed_dim)

        # Register hooks for intermediate feature extraction
        self._features: dict[int, torch.Tensor] = {}
        self._register_hooks()

        # Freeze backbone
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen — only decoder is trainable")

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha)

        # UPerNet decoder
        self.decoder = UPerNetDecoder(
            embed_dim=self.embed_dim,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )

        self._log_param_counts()

    def _register_hooks(self) -> None:
        """Register forward hooks on target transformer blocks."""
        for layer_idx in self.feature_layers:
            block = self.encoder.blocks[layer_idx]
            block.register_forward_hook(
                partial(self._hook_fn, layer_idx=layer_idx)
            )

    def _hook_fn(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Store intermediate features from transformer blocks."""
        self._features[layer_idx] = output

    def _apply_lora(self, rank: int, alpha: int) -> None:
        """Apply LoRA adapters to attention qkv layers in the backbone."""
        scaling = alpha / rank
        lora_count = 0

        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Linear) and ("qkv" in name or "q_proj" in name or "v_proj" in name):
                in_features = module.in_features
                out_features = module.out_features

                lora_a = nn.Parameter(torch.zeros(in_features, rank))
                lora_b = nn.Parameter(torch.zeros(rank, out_features))
                nn.init.kaiming_uniform_(lora_a)
                nn.init.zeros_(lora_b)

                module.lora_a = lora_a
                module.lora_b = lora_b
                module.lora_scaling = scaling

                original_forward = module.forward

                def make_lora_forward(orig_fwd, la, lb, s):
                    def lora_forward(x):
                        return orig_fwd(x) + (x @ la @ lb) * s
                    return lora_forward

                module.forward = make_lora_forward(original_forward, lora_a, lora_b, scaling)

                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

                lora_count += 1

        logger.info("Applied LoRA (rank=%d, alpha=%d) to %d layers", rank, alpha, lora_count)

    def _log_param_counts(self) -> None:
        """Log parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        logger.info(
            "DINOv2Segmenter: %d total params, %d trainable (%.1f%%), %d frozen",
            total,
            trainable,
            trainable / total * 100 if total > 0 else 0,
            frozen,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Segmentation logits (B, num_classes, H, W).
        """
        B, _, H, W = x.shape

        # Clear stored features
        self._features.clear()

        # Forward through DINOv2 (triggers hooks)
        _ = self.encoder.forward_features(x)

        # Collect multi-scale features from hooked layers
        # Each feature is (B, num_patches, embed_dim) — strip [CLS] token
        multi_scale = []
        for layer_idx in self.feature_layers:
            feat = self._features[layer_idx]
            # Remove CLS token (first token)
            if feat.shape[1] > (H // self.patch_size) * (W // self.patch_size):
                feat = feat[:, 1:, :]
            multi_scale.append(feat)

        # Decode
        seg_logits = self.decoder(multi_scale, target_size=(H, W))

        return seg_logits


def build_dinov2_segmenter(
    backbone: str = "dinov2_vitb14",
    num_classes: int = 1,
    decoder_dim: int = 256,
    freeze_backbone: bool = True,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
) -> DINOv2Segmenter:
    """Build a DINOv2 segmentation model.

    Convenience factory function matching the build_unetpp() pattern.

    Args:
        backbone: DINOv2 model name.
        num_classes: Number of output classes (1 for binary wound mask).
        decoder_dim: Decoder feature dimension.
        freeze_backbone: If True, freeze encoder weights.
        use_lora: If True, add LoRA adapters to backbone.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling.

    Returns:
        DINOv2Segmenter model.
    """
    return DINOv2Segmenter(
        backbone=backbone,
        num_classes=num_classes,
        decoder_dim=decoder_dim,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
