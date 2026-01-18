"""
Models Module
==============

This module contains all neural network architectures:
    - Segmentation models (U-Net++, DeepLabV3+)
    - Classification models (EfficientNet, ViT)
    - Infection detection models
    - Image quality assessment models
"""

from .segmentation import (
    UNetPlusPlus, 
    DeepLabV3Plus, 
    SegmentationModel,
    CBAM,
    SEBlock,
    BoundaryRefinementModule,
    EnsembleSegmentation,
    create_segmentation_model,
    unetpp_efficientnet_b4,
    deeplabv3plus_resnet101,
    unet_mobilenet,
)
from .classification import (
    TissueClassifier, 
    InfectionDetector,
    MultiTaskClassifier,
    WoundSeverityClassifier,
    create_tissue_classifier,
    create_infection_detector,
)
from .quality import (
    ImageQualityChecker,
    QualityGatedModel,
)
from .ensemble import (
    EnsembleModel,
    TestTimeAugmentationEnsemble,
    SnapshotEnsemble,
    create_ensemble,
)

__all__ = [
    # Segmentation
    "UNetPlusPlus",
    "DeepLabV3Plus",
    "SegmentationModel",
    "CBAM",
    "SEBlock",
    "BoundaryRefinementModule",
    "EnsembleSegmentation",
    "create_segmentation_model",
    "unetpp_efficientnet_b4",
    "deeplabv3plus_resnet101",
    "unet_mobilenet",
    # Classification
    "TissueClassifier",
    "InfectionDetector",
    "MultiTaskClassifier",
    "WoundSeverityClassifier",
    "create_tissue_classifier",
    "create_infection_detector",
    # Quality
    "ImageQualityChecker",
    "QualityGatedModel",
    # Ensemble
    "EnsembleModel",
    "TestTimeAugmentationEnsemble",
    "SnapshotEnsemble",
    "create_ensemble",
]
