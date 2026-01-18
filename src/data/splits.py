"""
Data Splits Module
===================

Utilities for creating train/validation/test splits.
Supports stratified splitting and cross-validation.

Author: Ruthvik
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import defaultdict
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .dataset import (
    WoundSegmentationDataset,
    WoundClassificationDataset,
    FUSeg2021Dataset,
)
from .augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
)


def create_data_splits(
    dataset_path: Union[str, Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by: Optional[str] = None,
    save_splits: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[int]]:
    """
    Create train/validation/test splits for a dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility
        stratify_by: Column/attribute to stratify by (for classification)
        save_splits: Whether to save splits to JSON
        output_path: Path to save splits JSON
    
    Returns:
        Dictionary with 'train', 'val', 'test' indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    dataset_path = Path(dataset_path)
    
    # Find all images
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        image_files.extend(dataset_path.rglob(f"*{ext}"))
        image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    n_samples = len(image_files)
    indices = list(range(n_samples))
    
    np.random.seed(seed)
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_idx, val_test_idx = train_test_split(
        indices,
        test_size=val_test_ratio,
        random_state=seed,
    )
    
    # Second split: val vs test
    val_size = val_ratio / val_test_ratio
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=(1 - val_size),
        random_state=seed,
    )
    
    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    
    # Save splits
    if save_splits:
        if output_path is None:
            output_path = dataset_path / "splits.json"
        
        splits_data = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
            "metadata": {
                "n_samples": n_samples,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "seed": seed,
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(splits_data, f, indent=2)
        
        print(f"Splits saved to: {output_path}")
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_idx)} samples ({len(train_idx)/n_samples*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} samples ({len(val_idx)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/n_samples*100:.1f}%)")
    
    return splits


def load_splits(splits_path: Union[str, Path]) -> Dict[str, List[int]]:
    """Load previously saved splits from JSON."""
    with open(splits_path) as f:
        data = json.load(f)
    
    return {
        "train": data["train"],
        "val": data["val"],
        "test": data["test"],
    }


def create_kfold_splits(
    n_samples: int,
    n_folds: int = 5,
    seed: int = 42,
    stratify_labels: Optional[List[int]] = None,
) -> List[Dict[str, List[int]]]:
    """
    Create K-fold cross-validation splits.
    
    Args:
        n_samples: Total number of samples
        n_folds: Number of folds
        seed: Random seed
        stratify_labels: Labels for stratified splitting
    
    Returns:
        List of dictionaries with 'train' and 'val' indices for each fold
    """
    indices = np.arange(n_samples)
    
    if stratify_labels is not None:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_iter = kfold.split(indices, stratify_labels)
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_iter = kfold.split(indices)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits_iter):
        folds.append({
            "fold": fold_idx,
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
        })
    
    return folds


def get_data_loaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: Optional[torch.utils.data.Dataset] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    weighted_sampling: bool = False,
    sample_weights: Optional[torch.Tensor] = None,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for CUDA
        weighted_sampling: Whether to use weighted random sampling
        sample_weights: Per-sample weights for weighted sampling
    
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders
    """
    # Training sampler
    train_sampler = None
    train_shuffle = True
    
    if weighted_sampling and sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
    
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }
    
    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    
    return loaders


def create_segmentation_loaders(
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict[str, DataLoader], Dict[str, List[int]]]:
    """
    Create data loaders for segmentation task.
    
    Returns:
        Tuple of (loaders dict, splits dict)
    """
    # Create full dataset to get all pairs
    full_dataset = WoundSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=None,
    )
    
    n_samples = len(full_dataset)
    indices = list(range(n_samples))
    
    # Create splits
    np.random.seed(seed)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - train_ratio), random_state=seed
    )
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_size), random_state=seed
    )
    
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    
    # Create augmentations
    train_transform = get_training_augmentation(image_size)
    val_transform = get_validation_augmentation(image_size)
    
    # Create datasets for each split
    train_dataset = WoundSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=train_transform,
    )
    val_dataset = WoundSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=val_transform,
    )
    test_dataset = WoundSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=val_transform,
    )
    
    # Create subset datasets
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)
    
    # Create loaders
    loaders = {
        "train": DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    
    print(f"Created segmentation loaders:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    
    return loaders, splits


def create_fuseg_loaders(
    fuseg_root: Union[str, Path],
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Create data loaders specifically for FUSeg 2021 dataset.
    Uses the official train/validation/test splits.
    """
    fuseg_root = Path(fuseg_root)
    
    train_transform = get_training_augmentation(image_size)
    val_transform = get_validation_augmentation(image_size)
    
    # FUSeg has predefined splits
    loaders = {}
    
    for split in ["train", "validation", "test"]:
        transform = train_transform if split == "train" else val_transform
        try:
            dataset = FUSeg2021Dataset(
                root_dir=fuseg_root,
                split=split,
                transform=transform,
            )
            loaders[split if split != "validation" else "val"] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == "train"),
            )
            print(f"  {split}: {len(dataset)} samples")
        except Exception as e:
            print(f"  {split}: Not found ({e})")
    
    return loaders
