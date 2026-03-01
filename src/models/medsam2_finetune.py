"""DiaFoot.AI v2 — MedSAM2 with LoRA Fine-Tuning.

Phase 2, Commit 11: Adapter-based fine-tuning of MedSAM2.

Strategy:
    - Freeze MedSAM2 image encoder (Hiera ViT)
    - Apply LoRA (rank=8) to q_proj and v_proj in attention layers
    - Fully fine-tune the mask decoder
    - Auto-generate bounding box prompts from ground truth masks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA adapter configuration.

    Args:
        rank: LoRA rank (lower = fewer params, 8-16 typical).
        alpha: Scaling factor (usually 2x rank).
        dropout: Dropout on LoRA layers.
        target_modules: Which attention modules to apply LoRA to.
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    """LoRA adapter for a linear layer.

    Adds low-rank decomposition: W' = W + (alpha/rank) * B @ A
    where A is (in, rank) and B is (rank, out).

    Args:
        original: The original linear layer to adapt.
        config: LoRA configuration.
    """

    def __init__(self, original: nn.Linear, config: LoRAConfig) -> None:
        """Initialize LoRA adapter around original linear layer."""
        super().__init__()
        self.original = original
        self.config = config

        in_features = original.in_features
        out_features = original.out_features

        # LoRA matrices
        self.lora_a = nn.Parameter(torch.zeros(in_features, config.rank))
        self.lora_b = nn.Parameter(torch.zeros(config.rank, out_features))
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(p=config.dropout)

        # Initialize A with Kaiming, B with zeros (so initial output = original)
        nn.init.kaiming_uniform_(self.lora_a)
        nn.init.zeros_(self.lora_b)

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with LoRA adaptation."""
        original_out = self.original(x)
        lora_out = self.dropout(x) @ self.lora_a @ self.lora_b * self.scaling
        return original_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig | None = None,
) -> tuple[nn.Module, int]:
    """Apply LoRA adapters to target modules in a model.

    Searches for linear layers matching target_modules names and
    wraps them with LoRA adapters.

    Args:
        model: The model to adapt.
        config: LoRA configuration.

    Returns:
        Tuple of (adapted model, number of LoRA parameters added).
    """
    config = config or LoRAConfig()
    lora_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module name matches any target
            if any(target in name for target in config.target_modules):
                # Find parent module and attribute name
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    attr_name = name

                # Replace with LoRA-wrapped version
                lora_layer = LoRALinear(module, config)
                setattr(parent, attr_name, lora_layer)

                added = config.rank * (module.in_features + module.out_features)
                lora_params += added
                logger.debug("Applied LoRA to %s (+%d params)", name, added)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "LoRA applied: %d total params, %d trainable (%.1f%%)",
        total_params,
        trainable,
        trainable / total_params * 100 if total_params > 0 else 0,
    )

    return model, lora_params


def mask_to_bbox(mask: torch.Tensor, padding: int = 10) -> torch.Tensor:
    """Generate bounding box prompt from ground truth mask.

    Used as automatic prompt for SAM-style models.

    Args:
        mask: Binary mask (H, W) or (B, H, W).
        padding: Pixels to pad around the bounding box.

    Returns:
        Bounding box as (x1, y1, x2, y2) tensor.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    bboxes = []
    for m in mask:
        nonzero = torch.nonzero(m, as_tuple=True)
        if len(nonzero[0]) == 0:
            # Empty mask — return full image bbox
            h, w = m.shape
            bboxes.append(torch.tensor([0, 0, w, h], dtype=torch.float32))
        else:
            y_min, y_max = nonzero[0].min(), nonzero[0].max()
            x_min, x_max = nonzero[1].min(), nonzero[1].max()
            # Add padding
            h, w = m.shape
            x1 = max(0, x_min.item() - padding)
            y1 = max(0, y_min.item() - padding)
            x2 = min(w, x_max.item() + padding)
            y2 = min(h, y_max.item() + padding)
            bboxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))

    return torch.stack(bboxes)
