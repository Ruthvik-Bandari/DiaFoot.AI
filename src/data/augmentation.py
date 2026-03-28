"""DiaFoot.AI v2 — Albumentations Augmentation Pipeline.

Phase 2, Commit 8-9: Mask-aware augmentation for training.
Includes wound-specific transforms and ITA-calibrated color augmentation.
"""

from __future__ import annotations

import albumentations as A  # noqa: N812
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 518) -> A.Compose:
    """Get training augmentation pipeline.

    Mask-aware transforms that apply identically to image and mask.
    Includes geometric, color, and wound-specific distortions.

    Args:
        image_size: Target image size (518 for DINOv2, 512 for legacy).

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose(
        [
            # Resize to target size (518 for DINOv2 patch alignment)
            A.Resize(image_size, image_size),
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=0,
                p=0.5,
            ),
            # Wound-specific distortions
            A.ElasticTransform(alpha=50, sigma=10, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=0.2),
            # Color augmentation (calibrated for clinical images)
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.5,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # Normalize (ImageNet stats for pretrained encoders)
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
    )


def get_val_transforms(image_size: int = 518) -> A.Compose:
    """Get validation/test transforms (no augmentation, just normalize).

    Args:
        image_size: Target image size (518 for DINOv2, 512 for legacy).

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
    )
