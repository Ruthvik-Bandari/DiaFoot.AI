"""
Data Augmentation Module
=========================

State-of-the-art augmentation pipelines for diabetic foot wound images.
Includes skin tone diversity augmentation for Fitzpatrick scale fairness.

Author: Ruthvik
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import Optional, Dict, Any, List, Tuple


def get_training_augmentation(
    image_size: int = 512,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    config: Optional[Dict[str, Any]] = None
) -> A.Compose:
    """
    Get comprehensive training augmentation pipeline.
    
    Includes:
    - Geometric transforms (flip, rotate, scale, elastic)
    - Color transforms (brightness, contrast, hue, saturation)
    - Skin tone diversity augmentation
    - Quality simulation (noise, blur, compression)
    - CutOut for regularization
    """
    transforms = [
        # Resize with aspect ratio preservation
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        ),
        
        # === Geometric Transforms ===
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7
        ),
        A.OneOf([
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                p=1.0
            ),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1.0),
        ], p=0.3),
        
        # === Skin Tone Diversity (Fitzpatrick Scale) ===
        # Critical for ensuring model works across all skin tones
        A.OneOf([
            # Simulate lighter skin tones (Fitzpatrick I-II)
            A.Sequential([
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.3),
                    contrast_limit=(-0.1, 0.1),
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=(-20, 0),
                    val_shift_limit=(10, 30),
                    p=1.0
                ),
            ]),
            # Simulate medium skin tones (Fitzpatrick III-IV)
            A.Sequential([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=(-0.1, 0.1),
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=(-10, 10),
                    val_shift_limit=(-10, 10),
                    p=1.0
                ),
            ]),
            # Simulate darker skin tones (Fitzpatrick V-VI)
            A.Sequential([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, -0.1),
                    contrast_limit=(0.1, 0.2),
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=(0, 20),
                    val_shift_limit=(-30, -10),
                    p=1.0
                ),
            ]),
        ], p=0.5),
        
        # === Standard Color Augmentations ===
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.5),
        
        # === Quality/Noise Simulation ===
        # Simulates real-world smartphone capture conditions
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        
        # === Regularization ===
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            mask_fill_value=0,
            p=0.3
        ),
        
        # === Normalize and Convert ===
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    
    return A.Compose(transforms)


def get_validation_augmentation(
    image_size: int = 512,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> A.Compose:
    """
    Get validation augmentation pipeline (minimal transforms).
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_tta_augmentation(
    image_size: int = 512,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> List[A.Compose]:
    """
    Get Test Time Augmentation (TTA) transforms.
    Returns list of transforms for ensemble prediction.
    """
    base_transform = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        ),
    ]
    
    normalize = [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    
    tta_transforms = [
        # Original
        A.Compose(base_transform + normalize),
        # Horizontal flip
        A.Compose(base_transform + [A.HorizontalFlip(p=1.0)] + normalize),
        # Vertical flip
        A.Compose(base_transform + [A.VerticalFlip(p=1.0)] + normalize),
        # Rotate 90
        A.Compose(base_transform + [A.Rotate(limit=(90, 90), p=1.0)] + normalize),
        # Rotate 180
        A.Compose(base_transform + [A.Rotate(limit=(180, 180), p=1.0)] + normalize),
        # Rotate 270
        A.Compose(base_transform + [A.Rotate(limit=(270, 270), p=1.0)] + normalize),
        # Slight scale up
        A.Compose(base_transform + [A.RandomScale(scale_limit=(0.1, 0.1), p=1.0)] + normalize),
        # Slight scale down
        A.Compose(base_transform + [A.RandomScale(scale_limit=(-0.1, -0.1), p=1.0)] + normalize),
    ]
    
    return tta_transforms


class MixUpCutMix:
    """
    MixUp and CutMix augmentation for classification/segmentation.
    Improves model generalization and calibration.
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
    
    def __call__(
        self,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Apply MixUp or CutMix augmentation.
        
        Returns:
            images: Augmented images
            masks: Augmented masks (if provided)
            labels: Augmented labels (if provided)
            lam: Lambda mixing coefficient
        """
        batch_size = images.shape[0]
        
        # Decide which augmentation to apply
        rand_val = np.random.random()
        
        if rand_val < self.mixup_prob:
            # Apply MixUp
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            rand_index = np.random.permutation(batch_size)
            
            images = lam * images + (1 - lam) * images[rand_index]
            
            if masks is not None:
                masks = lam * masks + (1 - lam) * masks[rand_index]
            
            if labels is not None:
                labels = (labels, labels[rand_index])
                
        elif rand_val < self.mixup_prob + self.cutmix_prob:
            # Apply CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            rand_index = np.random.permutation(batch_size)
            
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.shape, lam)
            
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            if masks is not None:
                masks[:, bbx1:bbx2, bby1:bby2] = masks[rand_index, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda based on actual area
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.shape[-2] * images.shape[-1]))
            
            if labels is not None:
                labels = (labels, labels[rand_index])
        else:
            # No augmentation
            lam = 1.0
            if labels is not None:
                labels = (labels, labels)
        
        return images, masks, labels, lam
    
    def _rand_bbox(self, size: Tuple, lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


def get_augmentation_from_config(config: Dict[str, Any], mode: str = "train") -> A.Compose:
    """
    Build augmentation pipeline from config dictionary.
    
    Args:
        config: Configuration dictionary from YAML
        mode: One of "train", "val", or "test"
    """
    aug_config = config.get("augmentation", {})
    dataset_config = config.get("dataset", {}).get("image", {})
    
    image_size = dataset_config.get("input_size", 512)
    mean = dataset_config.get("mean", [0.485, 0.456, 0.406])
    std = dataset_config.get("std", [0.229, 0.224, 0.225])
    
    if mode == "train":
        return get_training_augmentation(image_size, mean, std, aug_config.get("train"))
    elif mode == "val":
        return get_validation_augmentation(image_size, mean, std)
    else:
        return get_validation_augmentation(image_size, mean, std)
