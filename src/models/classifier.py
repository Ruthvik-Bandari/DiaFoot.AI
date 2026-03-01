"""DiaFoot.AI v2 — Triage Classifier.

Phase 2, Commit 8: Gateway classifier for the multi-task pipeline.
Classifies foot images into: Healthy | Non-DFU Condition | DFU

Architecture: EfficientNet-V2-M backbone (ImageNet pretrained)
              → Global Average Pooling → Dropout → FC → 3 classes

If classifier predicts "Healthy" with >95% confidence, segmentation is skipped.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class TriageClassifier(nn.Module):
    """Three-class triage classifier for foot images.

    Args:
        backbone: timm model name for the encoder.
        num_classes: Number of output classes (default: 3).
        dropout: Dropout rate before final FC layer.
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_m",
        num_classes: int = 3,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        """Initialize classifier with pretrained backbone."""
        super().__init__()
        self.num_classes = num_classes

        # Load backbone with global pooling, no classification head
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove default head
            global_pool="avg",
        )
        encoder_dim = self.encoder.num_features

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(encoder_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Logits (B, num_classes). Apply softmax for probabilities.
        """
        features = self.encoder(x)  # (B, encoder_dim)
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


# Class label mapping
CLASS_NAMES = {0: "Healthy", 1: "Non-DFU", 2: "DFU"}
