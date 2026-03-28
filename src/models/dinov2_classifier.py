"""DiaFoot.AI v2 — DINOv2 Triage Classifier.

Transfer learning with Meta's DINOv2 self-supervised ViT backbone.
DINOv2 features are domain-agnostic (not camera/dataset-specific), which
addresses the shortcut-learning problem seen with the EfficientNet classifier.

Architecture:
    DINOv2 ViT backbone (frozen or LoRA fine-tuned)
    → [CLS] token (768-dim for ViT-B/14)
    → LayerNorm → Dropout → Linear(768, 256) → GELU → Dropout → Linear(256, 3)

Supported backbones:
    - dinov2_vits14: 21M params, fastest (good for prototyping)
    - dinov2_vitb14: 86M params, best accuracy/speed balance (recommended)
    - dinov2_vitl14: 304M params, highest accuracy (needs more VRAM)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# DINOv2 backbone → feature dimension mapping
DINOV2_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


class DINOv2Classifier(nn.Module):
    """Three-class triage classifier using DINOv2 backbone.

    Args:
        backbone: DINOv2 model name (dinov2_vits14, dinov2_vitb14, dinov2_vitl14).
        num_classes: Number of output classes (default: 3).
        dropout: Dropout rate in classification head.
        freeze_backbone: If True, freeze all backbone parameters.
        use_lora: If True, apply LoRA adapters to attention q/v projections.
        lora_rank: LoRA rank (only used if use_lora=True).
        lora_alpha: LoRA scaling factor (only used if use_lora=True).
    """

    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        num_classes: int = 3,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ) -> None:
        """Initialize DINOv2 classifier."""
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        if backbone not in DINOV2_EMBED_DIMS:
            msg = f"Unsupported DINOv2 backbone: {backbone}. Choose from {list(DINOV2_EMBED_DIMS)}"
            raise ValueError(msg)

        embed_dim = DINOV2_EMBED_DIMS[backbone]

        # Load DINOv2 via torch.hub
        self.encoder = torch.hub.load("facebookresearch/dinov2", backbone)
        logger.info("Loaded DINOv2 backbone: %s (embed_dim=%d)", backbone, embed_dim)

        # Freeze backbone
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen — only classification head is trainable")

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha)

        # Classification head: LayerNorm → Dropout → FC → GELU → Dropout → FC
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._log_param_counts()

    def _apply_lora(self, rank: int, alpha: int) -> None:
        """Apply LoRA adapters to attention q_proj and v_proj layers."""
        scaling = alpha / rank
        lora_count = 0

        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Linear) and ("qkv" in name or "q_proj" in name or "v_proj" in name):
                # For DINOv2, attention uses qkv fused linear — apply LoRA to it
                in_features = module.in_features
                out_features = module.out_features

                lora_a = nn.Parameter(torch.zeros(in_features, rank))
                lora_b = nn.Parameter(torch.zeros(rank, out_features))
                nn.init.kaiming_uniform_(lora_a)
                nn.init.zeros_(lora_b)

                # Store LoRA params as buffers on the module
                module.lora_a = lora_a
                module.lora_b = lora_b
                module.lora_scaling = scaling

                # Patch forward to include LoRA
                original_forward = module.forward

                def make_lora_forward(orig_fwd, la, lb, s):
                    def lora_forward(x):
                        return orig_fwd(x) + (x @ la @ lb) * s
                    return lora_forward

                module.forward = make_lora_forward(original_forward, lora_a, lora_b, scaling)

                # Keep original weights frozen, LoRA params trainable
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

                lora_count += 1

        logger.info("Applied LoRA (rank=%d, alpha=%d) to %d layers", rank, alpha, lora_count)

    def _log_param_counts(self) -> None:
        """Log parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "DINOv2Classifier: %d total params, %d trainable (%.1f%%)",
            total,
            trainable,
            trainable / total * 100 if total > 0 else 0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W). DINOv2 expects 518x518 but
               handles other sizes via interpolated position embeddings.

        Returns:
            Logits (B, num_classes). Apply softmax for probabilities.
        """
        # DINOv2 forward returns [CLS] token embedding
        features = self.encoder(x)  # (B, embed_dim)
        logits = self.head(features)  # (B, num_classes)
        return logits

    def predict_with_confidence(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict class with confidence scores.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Tuple of (predicted_classes, confidence_scores).
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        return predicted, confidence


# Class label mapping (same as original)
CLASS_NAMES = {0: "Healthy", 1: "Non-DFU", 2: "DFU"}
