"""DiaFoot.AI v2 — Exponential Moving Average.

Phase 3, Commit 13: EMA for training stabilization.
Use EMA weights for evaluation (typically better than final weights).
"""

from __future__ import annotations

import copy

import torch.nn as nn  # noqa: TC002  # noqa: TCH002


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights updated as:
        shadow = decay * shadow + (1 - decay) * model_params

    Args:
        model: The model to track.
        decay: EMA decay rate (0.999 typical).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        """Initialize EMA with a copy of model parameters."""
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _update_fn(ema_v: float, model_v: float, decay: float) -> float:
        return decay * ema_v + (1.0 - decay) * model_v

    def update(self, model: nn.Module) -> None:
        """Update shadow weights from current model."""
        with_no_grad = dict(self.shadow.named_parameters())
        for name, param in model.named_parameters():
            if name in with_no_grad:
                with_no_grad[name].data.copy_(
                    self.decay * with_no_grad[name].data + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Copy shadow weights into model (for evaluation)."""
        model_params = dict(model.named_parameters())
        for name, param in self.shadow.named_parameters():
            if name in model_params:
                model_params[name].data.copy_(param.data)

    def get_shadow_model(self) -> nn.Module:
        """Return the shadow model for inference."""
        return self.shadow
