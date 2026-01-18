"""
Segmentation Models Module
===========================

State-of-the-art segmentation architectures for wound boundary detection.

Author: Ruthvik
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union
import segmentation_models_pytorch as smp


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # Ensure reduced channels is at least 1
        reduced_channels = max(channels // reduction, 1)
        
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BoundaryRefinementModule(nn.Module):
    """Boundary refinement for sharper wound edges."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        mid_channels = max(in_channels, 8)
        
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(x)


class UNetPlusPlus(nn.Module):
    """U-Net++ with modern encoders."""
    
    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with modern encoders."""
    
    def __init__(
        self,
        encoder_name: str = "resnet101",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SegmentationModel(nn.Module):
    """
    Factory class for creating segmentation models.
    """
    
    ARCHITECTURES = {
        "unetplusplus": smp.UnetPlusPlus,
        "unet": smp.Unet,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "pan": smp.PAN,
        "manet": smp.MAnet,
        "linknet": smp.Linknet,
    }
    
    def __init__(
        self,
        architecture: str = "unetplusplus",
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
        activation: Optional[str] = None,
        use_attention: bool = False,  # Disabled by default for stability
    ):
        super().__init__()
        
        self.architecture = architecture.lower()
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        
        if self.architecture not in self.ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Create base model
        model_class = self.ARCHITECTURES[self.architecture]
        self.model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
        )
        
        # Only use attention if channels > 1
        self.use_attention = use_attention and num_classes > 1
        if self.use_attention:
            self.attention = CBAM(num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masks = self.model(x)
        
        if self.use_attention:
            masks = self.attention(masks)
        
        return masks
    
    def get_encoder_params(self):
        """Get encoder parameters for differential learning rates."""
        return self.model.encoder.parameters()
    
    def get_decoder_params(self):
        """Get decoder parameters."""
        params = list(self.model.decoder.parameters())
        params += list(self.model.segmentation_head.parameters())
        return params


class EnsembleSegmentation(nn.Module):
    """Ensemble of multiple segmentation models."""
    
    def __init__(self, models: list, fusion: str = "average"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.fusion = fusion
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = [model(x) for model in self.models]
        stacked = torch.stack(predictions, dim=0)
        
        if self.fusion == "average":
            return stacked.mean(dim=0)
        elif self.fusion == "max":
            return stacked.max(dim=0)[0]
        return stacked.mean(dim=0)


def create_segmentation_model(config: Dict[str, Any]) -> nn.Module:
    """Create segmentation model from config."""
    seg_config = config.get("models", {}).get("segmentation", {})
    
    return SegmentationModel(
        architecture=seg_config.get("architecture", "unetplusplus"),
        encoder_name=seg_config.get("encoder", "efficientnet-b4"),
        encoder_weights=seg_config.get("encoder_weights", "imagenet"),
        in_channels=seg_config.get("in_channels", 3),
        num_classes=seg_config.get("num_classes", 1),
        use_attention=False,
    )


def unetpp_efficientnet_b4(num_classes: int = 1) -> nn.Module:
    """U-Net++ with EfficientNet-B4 encoder."""
    return SegmentationModel(
        architecture="unetplusplus",
        encoder_name="efficientnet-b4",
        num_classes=num_classes,
    )


def deeplabv3plus_resnet101(num_classes: int = 1) -> nn.Module:
    """DeepLabV3+ with ResNet-101 encoder."""
    return SegmentationModel(
        architecture="deeplabv3plus",
        encoder_name="resnet101",
        num_classes=num_classes,
    )


def unet_mobilenet(num_classes: int = 1) -> nn.Module:
    """Lightweight U-Net with MobileNetV2."""
    return SegmentationModel(
        architecture="unet",
        encoder_name="mobilenet_v2",
        num_classes=num_classes,
    )
