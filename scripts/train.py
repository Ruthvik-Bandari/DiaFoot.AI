#!/usr/bin/env python3
"""
DiaFootAI Training Script
==========================

Main entry point for training wound segmentation models.
Supports multiple architectures, datasets, and training configurations.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --resume models/checkpoints/last_model.pt

Author: Ruthvik
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.utils.config import load_config
from src.utils.device import get_device, print_device_info
from src.data.dataset import WoundSegmentationDataset, FUSeg2021Dataset
from src.data.augmentation import get_training_augmentation, get_validation_augmentation
from src.data.splits import create_data_splits
from src.models.segmentation import SegmentationModel, unetpp_efficientnet_b4
from src.training.trainer import SegmentationTrainer
from src.training.losses import DiceBCELoss, FocalTverskyLoss, CombinedLoss
from src.training.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateLogger, 
    MetricsLogger, WandBLogger
)
from src.evaluation.metrics import evaluate_segmentation_model, print_evaluation_results


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiaFootAI models")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory from config"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training"
    )
    
    return parser.parse_args()


def setup_experiment(config: dict, args) -> str:
    """Setup experiment directory and logging."""
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.get("models", {}).get("segmentation", {}).get("architecture", "unetpp")
        encoder = config.get("models", {}).get("segmentation", {}).get("encoder", "effnet_b4")
        exp_name = f"{model_name}_{encoder}_{timestamp}"
    
    # Create experiment directories
    exp_dir = Path(config.get("paths", {}).get("outputs", "outputs")) / exp_name
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment: {exp_name}")
    print(f"Output directory: {exp_dir}")
    
    return str(exp_dir)


def create_model(config: dict) -> nn.Module:
    """Create model from configuration."""
    seg_config = config.get("models", {}).get("segmentation", {})
    
    model = SegmentationModel(
        architecture=seg_config.get("architecture", "unetplusplus"),
        encoder_name=seg_config.get("encoder", "efficientnet-b4"),
        encoder_weights=seg_config.get("encoder_weights", "imagenet"),
        in_channels=seg_config.get("in_channels", 3),
        num_classes=seg_config.get("num_classes", 1),
        use_attention=True,
    )
    
    return model


def create_dataloaders(config: dict, args) -> tuple:
    """Create train/val/test dataloaders."""
    # Get paths
    data_dir = args.data_dir or config.get("paths", {}).get("data_root", "data")
    data_dir = Path(data_dir)
    
    # Dataset config
    dataset_config = config.get("dataset", {})
    image_size = dataset_config.get("image", {}).get("input_size", 512)
    
    # Check for FUSeg dataset structure
    fuseg_path = data_dir / "raw" / "fuseg" / "wound-segmentation" / "data" / "Foot Ulcer Segmentation Challenge"
    
    if fuseg_path.exists():
        print(f"Found FUSeg dataset at: {fuseg_path}")
        
        train_transform = get_training_augmentation(image_size)
        val_transform = get_validation_augmentation(image_size)
        
        train_dataset = FUSeg2021Dataset(
            root_dir=fuseg_path,
            split="train",
            transform=train_transform,
        )
        
        val_dataset = FUSeg2021Dataset(
            root_dir=fuseg_path,
            split="validation",
            transform=val_transform,
        )
        
        # Try to load test set
        try:
            test_dataset = FUSeg2021Dataset(
                root_dir=fuseg_path,
                split="test",
                transform=val_transform,
            )
        except:
            test_dataset = None
            print("Test set not found, using validation set for testing")
    else:
        # Generic dataset loading
        images_dir = data_dir / "processed" / "segmentation" / "images"
        masks_dir = data_dir / "processed" / "segmentation" / "masks"
        
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {images_dir}\n"
                f"Please run: python scripts/download_datasets.py --all"
            )
        
        print(f"Loading dataset from: {images_dir}")
        
        # Create full dataset
        full_dataset = WoundSegmentationDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            transform=None,
        )
        
        # Create splits
        n_samples = len(full_dataset)
        indices = list(range(n_samples))
        
        train_ratio = dataset_config.get("train_ratio", 0.7)
        val_ratio = dataset_config.get("val_ratio", 0.15)
        
        np.random.seed(dataset_config.get("seed", 42))
        np.random.shuffle(indices)
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create augmented datasets
        train_transform = get_training_augmentation(image_size)
        val_transform = get_validation_augmentation(image_size)
        
        train_dataset = WoundSegmentationDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            transform=train_transform,
        )
        train_dataset = Subset(train_dataset, train_indices)
        
        val_dataset = WoundSegmentationDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            transform=val_transform,
        )
        val_dataset = Subset(val_dataset, val_indices)
        
        test_dataset = WoundSegmentationDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            transform=val_transform,
        )
        test_dataset = Subset(test_dataset, test_indices)
    
    # Create dataloaders
    train_config = config.get("training", {})
    batch_size = args.batch_size or train_config.get("batch_size", 8)
    num_workers = train_config.get("num_workers", 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    if test_dataset:
        print(f"  Test:  {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def create_criterion(config: dict) -> nn.Module:
    """Create loss function from config."""
    loss_config = config.get("training", {}).get("loss", {}).get("segmentation", {})
    loss_name = loss_config.get("name", "dice_bce")
    
    if loss_name == "dice_bce":
        return DiceBCELoss(
            dice_weight=loss_config.get("dice_weight", 0.5),
            bce_weight=loss_config.get("bce_weight", 0.5),
        )
    elif loss_name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=loss_config.get("alpha", 0.3),
            beta=loss_config.get("beta", 0.7),
            gamma=loss_config.get("gamma", 0.75),
        )
    elif loss_name == "combined":
        from src.training.losses import BoundaryLoss
        return CombinedLoss([
            (DiceBCELoss(), 0.7),
            (BoundaryLoss(), 0.3),
        ])
    else:
        return DiceBCELoss()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["optimizer"]["lr"] = args.lr
    if args.no_amp:
        config["training"]["mixed_precision"]["enabled"] = False
    
    # Setup experiment
    exp_dir = setup_experiment(config, args)
    
    # Device setup
    device = get_device(config.get("training", {}).get("device", "auto"))
    print_device_info(device)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(config, args)
    
    # Create loss function
    criterion = create_criterion(config)
    print(f"Loss function: {criterion.__class__.__name__}")
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_iou",
            patience=config.get("training", {}).get("early_stopping", {}).get("patience", 20),
            mode="max",
        ),
        ModelCheckpoint(
            checkpoint_dir=f"{exp_dir}/checkpoints",
            monitor="val_iou",
            mode="max",
        ),
        LearningRateLogger(),
        MetricsLogger(log_dir=f"{exp_dir}/logs"),
    ]
    
    if args.wandb:
        callbacks.append(WandBLogger(
            project="diafootai",
            name=Path(exp_dir).name,
            config=config,
        ))
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        callbacks=callbacks,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    history = trainer.fit(
        epochs=config.get("training", {}).get("epochs", 100),
    )
    
    # Evaluate on test set
    if test_loader is not None:
        print("\nEvaluating on test set...")
        test_results = evaluate_segmentation_model(
            model=model,
            dataloader=test_loader,
            device=device,
            threshold=config.get("inference", {}).get("confidence_threshold", 0.5),
        )
        print_evaluation_results(test_results, "Test Results")
        
        # Save test results
        results_path = Path(exp_dir) / "test_results.json"
        test_results_serializable = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in test_results.items()
            if k != "confusion_matrix"
        }
        with open(results_path, "w") as f:
            json.dump(test_results_serializable, f, indent=2)
    
    # Save final model
    final_model_path = Path(exp_dir) / "checkpoints" / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    print("\nTraining complete!")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
