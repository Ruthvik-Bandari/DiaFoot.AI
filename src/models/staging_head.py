"""DiaFoot.AI v2 — Wagner Grade Staging Head.

Phase 2, Commit 9: Classifies wound severity into Wagner grades 0-5.
Can be attached to segmentation features or used standalone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

WAGNER_GRADES = {
    0: "Pre-ulcerative / at-risk",
    1: "Superficial ulcer",
    2: "Deep ulcer (tendon/bone)",
    3: "Deep with abscess",
    4: "Partial gangrene",
    5: "Extensive gangrene",
}


class WagnerStagingHead(nn.Module):
    """Wagner grade classification head.

    Takes pooled features and predicts Wagner grade 0-5.

    Args:
        in_features: Input feature dimension.
        num_grades: Number of Wagner grades (default 6: grades 0-5).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_features: int = 1792,
        num_grades: int = 6,
        dropout: float = 0.3,
    ) -> None:
        """Initialize staging head."""
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_grades),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict Wagner grade from pooled features.

        Args:
            features: (B, in_features) pooled feature vector.

        Returns:
            (B, num_grades) logits.
        """
        return self.head(features)
