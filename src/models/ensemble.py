"""
Ensemble Models Module
=======================

Ensemble strategies for combining multiple models.
Includes voting, averaging, and learned fusion.

Author: Ruthvik
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union
import copy


class EnsembleModel(nn.Module):
    """
    General-purpose ensemble wrapper for multiple models.
    Supports different fusion strategies.
    """
    
    FUSION_METHODS = ["average", "weighted", "voting", "learned", "max"]
    
    def __init__(
        self,
        models: List[nn.Module],
        fusion: str = "average",
        weights: Optional[List[float]] = None,
        freeze_base_models: bool = True,
    ):
        """
        Args:
            models: List of models to ensemble
            fusion: Fusion strategy ('average', 'weighted', 'voting', 'learned', 'max')
            weights: Optional weights for weighted averaging
            freeze_base_models: Whether to freeze base model parameters
        """
        super().__init__()
        
        assert fusion in self.FUSION_METHODS, f"Unknown fusion: {fusion}"
        assert len(models) > 0, "Need at least one model"
        
        self.models = nn.ModuleList(models)
        self.fusion = fusion
        self.n_models = len(models)
        
        # Freeze base models if requested
        if freeze_base_models:
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False
        
        # Setup fusion mechanism
        if fusion == "weighted":
            if weights is None:
                weights = [1.0 / self.n_models] * self.n_models
            assert len(weights) == self.n_models
            self.register_buffer(
                "weights",
                torch.tensor(weights, dtype=torch.float32)
            )
        
        elif fusion == "learned":
            # Learnable attention weights
            self.attention = nn.Sequential(
                nn.Linear(self.n_models, self.n_models),
                nn.Softmax(dim=-1),
            )
            # Initialize close to uniform
            nn.init.eye_(self.attention[0].weight)
            nn.init.zeros_(self.attention[0].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models and fuse predictions.
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.set_grad_enabled(self.training and not self._frozen(model)):
                pred = model(x)
                predictions.append(pred)
        
        # Stack predictions: (n_models, batch, ...)
        stacked = torch.stack(predictions, dim=0)
        
        # Apply fusion
        if self.fusion == "average":
            return stacked.mean(dim=0)
        
        elif self.fusion == "max":
            return stacked.max(dim=0)[0]
        
        elif self.fusion == "weighted":
            # Weighted average
            weights = self.weights.view(-1, 1, *([1] * (stacked.dim() - 2)))
            return (stacked * weights).sum(dim=0)
        
        elif self.fusion == "voting":
            # Hard voting (for classification)
            votes = stacked.argmax(dim=-1)  # (n_models, batch)
            # Count votes for each class
            batch_size = x.shape[0]
            num_classes = stacked.shape[-1]
            vote_counts = torch.zeros(batch_size, num_classes, device=x.device)
            for i in range(self.n_models):
                vote_counts.scatter_add_(
                    1,
                    votes[i].unsqueeze(1),
                    torch.ones_like(votes[i]).unsqueeze(1).float()
                )
            return vote_counts
        
        elif self.fusion == "learned":
            # Use attention mechanism to weight predictions
            batch_size = stacked.shape[1]
            
            # Get attention weights based on prediction confidence
            if stacked.dim() == 3:  # Classification: (n_models, batch, classes)
                # Use max softmax probability as confidence
                confidences = F.softmax(stacked, dim=-1).max(dim=-1)[0]  # (n_models, batch)
                confidences = confidences.transpose(0, 1)  # (batch, n_models)
                weights = self.attention(confidences)  # (batch, n_models)
                weights = weights.transpose(0, 1).unsqueeze(-1)  # (n_models, batch, 1)
                return (stacked * weights).sum(dim=0)
            else:
                # For segmentation: simple learned weights
                weights = self.attention(
                    torch.ones(batch_size, self.n_models, device=x.device)
                )
                weights = weights.transpose(0, 1)
                for _ in range(stacked.dim() - 2):
                    weights = weights.unsqueeze(-1)
                return (stacked * weights).sum(dim=0)
        
        return stacked.mean(dim=0)  # Default fallback
    
    def _frozen(self, model: nn.Module) -> bool:
        """Check if model is frozen."""
        return not any(p.requires_grad for p in model.parameters())
    
    def get_individual_predictions(
        self,
        x: torch.Tensor
    ) -> List[torch.Tensor]:
        """Get predictions from each individual model."""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                predictions.append(model(x))
        return predictions


class TestTimeAugmentationEnsemble(nn.Module):
    """
    Test-Time Augmentation (TTA) ensemble.
    Applies different augmentations and averages predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        transforms: Optional[List[str]] = None,
    ):
        """
        Args:
            model: Base model
            transforms: List of transforms to apply
                       Options: 'hflip', 'vflip', 'rot90', 'rot180', 'rot270'
        """
        super().__init__()
        
        self.model = model
        self.transforms = transforms or [
            "original",
            "hflip",
            "vflip",
            "rot90",
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with TTA.
        """
        predictions = []
        
        for transform in self.transforms:
            # Apply transform
            transformed = self._apply_transform(x, transform)
            
            # Get prediction
            with torch.no_grad():
                pred = self.model(transformed)
            
            # Reverse transform on prediction
            pred = self._reverse_transform(pred, transform)
            predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
    
    def _apply_transform(self, x: torch.Tensor, transform: str) -> torch.Tensor:
        """Apply spatial transform to input."""
        if transform == "original":
            return x
        elif transform == "hflip":
            return torch.flip(x, dims=[-1])
        elif transform == "vflip":
            return torch.flip(x, dims=[-2])
        elif transform == "rot90":
            return torch.rot90(x, k=1, dims=[-2, -1])
        elif transform == "rot180":
            return torch.rot90(x, k=2, dims=[-2, -1])
        elif transform == "rot270":
            return torch.rot90(x, k=3, dims=[-2, -1])
        else:
            return x
    
    def _reverse_transform(self, x: torch.Tensor, transform: str) -> torch.Tensor:
        """Reverse spatial transform on output."""
        if transform == "original":
            return x
        elif transform == "hflip":
            return torch.flip(x, dims=[-1])
        elif transform == "vflip":
            return torch.flip(x, dims=[-2])
        elif transform == "rot90":
            return torch.rot90(x, k=-1, dims=[-2, -1])
        elif transform == "rot180":
            return torch.rot90(x, k=-2, dims=[-2, -1])
        elif transform == "rot270":
            return torch.rot90(x, k=-3, dims=[-2, -1])
        else:
            return x


class SnapshotEnsemble(nn.Module):
    """
    Snapshot ensemble from cyclic learning rate training.
    Uses multiple checkpoints from the same training run.
    """
    
    def __init__(
        self,
        model_class: type,
        checkpoint_paths: List[str],
        model_kwargs: Optional[Dict[str, Any]] = None,
        fusion: str = "average",
    ):
        """
        Args:
            model_class: Class of the model to instantiate
            checkpoint_paths: Paths to model checkpoints
            model_kwargs: Arguments to pass to model constructor
            fusion: Fusion strategy
        """
        super().__init__()
        
        model_kwargs = model_kwargs or {}
        
        # Load models from checkpoints
        models = []
        for path in checkpoint_paths:
            model = model_class(**model_kwargs)
            state_dict = torch.load(path, map_location="cpu")
            
            # Handle different checkpoint formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        
        self.ensemble = EnsembleModel(
            models=models,
            fusion=fusion,
            freeze_base_models=True,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ensemble(x)


def create_ensemble(
    models: List[nn.Module],
    config: Dict[str, Any],
) -> EnsembleModel:
    """
    Create ensemble from config.
    
    Args:
        models: List of models to ensemble
        config: Ensemble configuration
    
    Returns:
        Configured ensemble model
    """
    fusion = config.get("fusion", "average")
    weights = config.get("weights", None)
    freeze = config.get("freeze_base_models", True)
    
    return EnsembleModel(
        models=models,
        fusion=fusion,
        weights=weights,
        freeze_base_models=freeze,
    )
