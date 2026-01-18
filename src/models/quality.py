"""
Image Quality Assessment Module
================================

Lightweight model for checking image quality before processing.
Ensures input images are suitable for wound analysis.

Author: Ruthvik
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Any, Tuple, Optional


class ImageQualityChecker(nn.Module):
    """
    Lightweight model to assess image quality.
    
    Quality Classes:
    - 0: Acceptable (good quality)
    - 1: Blurry (motion blur or out of focus)
    - 2: Poor lighting (too dark or overexposed)
    - 3: Bad angle (wound not visible or partially occluded)
    """
    
    QUALITY_CLASSES = [
        "acceptable",
        "blurry",
        "poor_lighting",
        "bad_angle",
    ]
    
    def __init__(
        self,
        encoder_name: str = "mobilenetv3_small_100",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Lightweight backbone for fast inference
        self.backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        
        self.feature_dim = self.backbone.num_features
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        
        # Additional head for quality score (0-100)
        self.quality_scorer = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_score: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Dictionary with:
            - 'logits': Quality class logits
            - 'quality_score': Quality score (0-100) if return_score=True
        """
        features = self.backbone(x)
        
        outputs = {
            "logits": self.classifier(features),
        }
        
        if return_score:
            outputs["quality_score"] = self.quality_scorer(features) * 100
        
        return outputs
    
    def check_quality(
        self,
        x: torch.Tensor,
        threshold: float = 50.0,
    ) -> Tuple[bool, str, float]:
        """
        Check if image quality is acceptable.
        
        Args:
            x: Input image tensor (B, C, H, W)
            threshold: Minimum quality score to accept
        
        Returns:
            Tuple of (is_acceptable, issue_description, quality_score)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            quality_score = outputs["quality_score"].item()
            predicted_class = outputs["logits"].argmax(dim=1).item()
            
            is_acceptable = (
                predicted_class == 0 and quality_score >= threshold
            )
            
            issue = self.QUALITY_CLASSES[predicted_class]
            
            return is_acceptable, issue, quality_score


class QualityGatedModel(nn.Module):
    """
    Wrapper that gates predictions based on image quality.
    Only processes images that pass quality check.
    """
    
    def __init__(
        self,
        main_model: nn.Module,
        quality_checker: Optional[ImageQualityChecker] = None,
        quality_threshold: float = 50.0,
    ):
        super().__init__()
        
        self.main_model = main_model
        self.quality_checker = quality_checker or ImageQualityChecker()
        self.quality_threshold = quality_threshold
    
    def forward(
        self,
        x: torch.Tensor,
        skip_quality_check: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with quality gating.
        
        Returns:
            Dictionary with predictions and quality info
        """
        batch_size = x.shape[0]
        
        outputs = {
            "predictions": None,
            "quality_passed": torch.zeros(batch_size, dtype=torch.bool),
            "quality_scores": torch.zeros(batch_size),
            "quality_issues": [""] * batch_size,
        }
        
        if skip_quality_check:
            outputs["predictions"] = self.main_model(x)
            outputs["quality_passed"] = torch.ones(batch_size, dtype=torch.bool)
            return outputs
        
        # Check quality for each image
        with torch.no_grad():
            quality_outputs = self.quality_checker(x)
            quality_scores = quality_outputs["quality_score"].squeeze()
            quality_classes = quality_outputs["logits"].argmax(dim=1)
        
        # Determine which images pass quality check
        acceptable_mask = (quality_classes == 0) & (quality_scores >= self.quality_threshold)
        
        outputs["quality_scores"] = quality_scores
        outputs["quality_passed"] = acceptable_mask
        
        for i in range(batch_size):
            outputs["quality_issues"][i] = ImageQualityChecker.QUALITY_CLASSES[
                quality_classes[i].item()
            ]
        
        # Only process acceptable images
        if acceptable_mask.any():
            acceptable_images = x[acceptable_mask]
            predictions = self.main_model(acceptable_images)
            
            # Reconstruct full batch output
            if isinstance(predictions, torch.Tensor):
                full_predictions = torch.zeros(
                    batch_size, *predictions.shape[1:],
                    device=predictions.device,
                    dtype=predictions.dtype,
                )
                full_predictions[acceptable_mask] = predictions
                outputs["predictions"] = full_predictions
            else:
                outputs["predictions"] = predictions
        
        return outputs
