"""
Classification Models Module
=============================

Models for tissue classification and infection detection.
Includes modern architectures with attention mechanisms.

Author: Ruthvik
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import timm


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.channel_mlp(self.channel_avg_pool(x))
        max_out = self.channel_mlp(self.channel_max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class TissueClassifier(nn.Module):
    """
    Multi-class tissue classification model.
    
    Classifies wound tissue into categories:
    - Granulation (healthy healing)
    - Slough (needs debridement)
    - Necrotic (dead tissue)
    - Epithelial (new skin)
    - Periwound (surrounding skin)
    """
    
    ENCODERS = {
        "efficientnet_b3": "efficientnet_b3",
        "efficientnet_b4": "efficientnet_b4",
        "efficientnetv2_s": "tf_efficientnetv2_s",
        "efficientnetv2_m": "tf_efficientnetv2_m",
        "convnext_tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
        "vit_small": "vit_small_patch16_224",
        "vit_base": "vit_base_patch16_224",
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "swin_small": "swin_small_patch4_window7_224",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
    }
    
    def __init__(
        self,
        encoder_name: str = "efficientnet_b3",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Get timm model name
        if encoder_name in self.ENCODERS:
            timm_name = self.ENCODERS[encoder_name]
        else:
            timm_name = encoder_name
        
        # Create backbone
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool="",  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            if len(features.shape) == 4:
                self.feature_dim = features.shape[1]
            else:
                self.feature_dim = features.shape[-1]
        
        # Attention module
        if use_attention:
            self.attention = CBAM(self.feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_features: Whether to return intermediate features
        
        Returns:
            logits: Classification logits
            features: Optional feature tensor
        """
        # Extract features
        features = self.backbone(x)
        
        # Handle different output shapes (ViT vs CNN)
        if len(features.shape) == 3:
            # Transformer: (B, N, D) -> use class token or mean
            features = features.mean(dim=1)
            logits = self.classifier(features)
        else:
            # CNN: (B, C, H, W)
            if self.use_attention:
                features = self.attention(features)
            
            pooled = self.global_pool(features).flatten(1)
            logits = self.classifier(pooled)
        
        if return_features:
            return logits, pooled if len(features.shape) != 3 else features
        return logits
    
    def get_cam(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get Class Activation Map for visualization.
        
        Args:
            x: Input image
            class_idx: Target class (uses predicted class if None)
        
        Returns:
            CAM heatmap
        """
        features = self.backbone(x)
        
        if len(features.shape) == 3:
            # Can't do CAM for transformers easily
            return torch.zeros_like(x[:, 0:1])
        
        if self.use_attention:
            features = self.attention(features)
        
        pooled = self.global_pool(features).flatten(1)
        logits = self.classifier(pooled)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        
        # Get weights of the final classifier layer
        weights = self.classifier[-1].weight[class_idx]  # (num_classes, 512)
        
        # This is a simplified CAM - for proper Grad-CAM, use hooks
        # Here we just weight the features
        cam = (features * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear")
        
        return cam


class InfectionDetector(nn.Module):
    """
    Binary classification model for infection detection.
    
    Uses attention mechanisms to focus on infection indicators:
    - Redness/erythema
    - Swelling
    - Discharge/pus
    - Discoloration
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout: float = 0.4,
        use_attention: bool = True,
    ):
        super().__init__()
        
        # Create backbone
        self.backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 384, 384)  # Larger input for infection detection
            features = self.backbone(dummy)
            if len(features.shape) == 4:
                self.feature_dim = features.shape[1]
            else:
                self.feature_dim = features.shape[-1]
        
        # Multi-scale feature aggregation
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(self.feature_dim)
        
        # Global pooling with both avg and max
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier (binary)
        pooled_dim = self.feature_dim * 2  # avg + max pooling
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # Binary: infected vs not infected
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        if len(features.shape) == 3:
            # Transformer output
            features = features.mean(dim=1)
            logits = self.classifier(features)
        else:
            if self.use_attention:
                features = self.attention(features)
            
            # Dual pooling for richer representation
            avg_pooled = self.avg_pool(features).flatten(1)
            max_pooled = self.max_pool(features).flatten(1)
            pooled = torch.cat([avg_pooled, max_pooled], dim=1)
            
            logits = self.classifier(pooled)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get infection probability."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]  # Probability of infection


class MultiTaskClassifier(nn.Module):
    """
    Multi-task model for joint tissue and infection classification.
    Shares encoder, uses task-specific heads.
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet_b4",
        num_tissue_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Shared backbone
        self.backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Shared feature transform
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        self.tissue_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_tissue_classes),
        )
        
        self.infection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 2),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        task: str = "both",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            task: 'tissue', 'infection', or 'both'
        
        Returns:
            Dictionary with task predictions
        """
        features = self.backbone(x)
        shared = self.shared_fc(features)
        
        outputs = {}
        
        if task in ["tissue", "both"]:
            outputs["tissue"] = self.tissue_head(shared)
        
        if task in ["infection", "both"]:
            outputs["infection"] = self.infection_head(shared)
        
        return outputs


class WoundSeverityClassifier(nn.Module):
    """
    Wound severity classification (Wagner scale for diabetic foot).
    
    Wagner Grades:
    - Grade 0: No ulcer, high-risk foot
    - Grade 1: Superficial ulcer
    - Grade 2: Deep ulcer to tendon/bone
    - Grade 3: Deep with abscess/osteomyelitis
    - Grade 4: Partial gangrene
    - Grade 5: Extensive gangrene
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        
        self.feature_dim = self.backbone.num_features
        
        # Use ordinal regression approach for severity
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),  # 6 Wagner grades
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def predict_grade(self, x: torch.Tensor) -> torch.Tensor:
        """Predict Wagner grade (0-5)."""
        logits = self.forward(x)
        return logits.argmax(dim=1)


def create_tissue_classifier(config: Dict[str, Any]) -> nn.Module:
    """Create tissue classifier from config."""
    clf_config = config.get("models", {}).get("tissue_classifier", {})
    
    return TissueClassifier(
        encoder_name=clf_config.get("architecture", "efficientnet_b3"),
        num_classes=clf_config.get("num_classes", 6),
        pretrained=clf_config.get("pretrained", True),
        dropout=clf_config.get("dropout", 0.3),
        use_attention=True,
    )


def create_infection_detector(config: Dict[str, Any]) -> nn.Module:
    """Create infection detector from config."""
    det_config = config.get("models", {}).get("infection_detector", {})
    
    return InfectionDetector(
        encoder_name=det_config.get("architecture", "resnet50"),
        pretrained=det_config.get("pretrained", True),
        dropout=det_config.get("dropout", 0.4),
        use_attention=det_config.get("attention", True),
    )
