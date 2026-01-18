"""
Data Module
============

This module handles all data-related operations:
    - Dataset classes for loading wound images
    - Data augmentation pipelines
    - Preprocessing utilities
    - Data splits and sampling
"""

from .dataset import (
    WoundDataset, 
    WoundSegmentationDataset, 
    WoundClassificationDataset,
    FUSeg2021Dataset,
    AZHWoundDataset,
    MultiTaskWoundDataset,
)
from .augmentation import (
    get_training_augmentation, 
    get_validation_augmentation, 
    get_tta_augmentation,
    get_augmentation_from_config,
    MixUpCutMix,
)
from .preprocessing import (
    preprocess_image, 
    normalize_image, 
    denormalize_image,
    resize_with_padding,
    remove_padding,
    assess_image_quality,
    enhance_wound_image,
    extract_wound_roi,
    compute_wound_metrics,
)
from .splits import (
    create_data_splits, 
    get_data_loaders,
    load_splits,
    create_kfold_splits,
    create_segmentation_loaders,
    create_fuseg_loaders,
)

__all__ = [
    # Datasets
    "WoundDataset",
    "WoundSegmentationDataset",
    "WoundClassificationDataset",
    "FUSeg2021Dataset",
    "AZHWoundDataset",
    "MultiTaskWoundDataset",
    # Augmentation
    "get_training_augmentation",
    "get_validation_augmentation",
    "get_tta_augmentation",
    "get_augmentation_from_config",
    "MixUpCutMix",
    # Preprocessing
    "preprocess_image",
    "normalize_image",
    "denormalize_image",
    "resize_with_padding",
    "remove_padding",
    "assess_image_quality",
    "enhance_wound_image",
    "extract_wound_roi",
    "compute_wound_metrics",
    # Splits
    "create_data_splits",
    "get_data_loaders",
    "load_splits",
    "create_kfold_splits",
    "create_segmentation_loaders",
    "create_fuseg_loaders",
]
