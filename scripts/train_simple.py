#!/usr/bin/env python3
"""
Simplified Training Script for DiaFootAI
Optimized for class-imbalanced wound segmentation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.data.dataset import FUSeg2021Dataset
from src.data.augmentation import get_training_augmentation, get_validation_augmentation
from src.models.segmentation import SegmentationModel


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss - better for imbalanced data."""
    
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # BCE Loss
        bce = F.binary_cross_entropy(pred, target, reduction='mean')
        
        # Dice Loss
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return self.dice_weight * dice + self.bce_weight * bce


def compute_metrics(pred, target, threshold=0.5):
    """Compute IoU and Dice."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    smooth = 1e-5
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (pred_binary.sum() + target.sum() + smooth)
    
    return iou.item(), dice.item()


def train():
    # Settings
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 1e-4
    IMAGE_SIZE = 512
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {DEVICE}")
    
    # Data
    data_root = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge")
    
    train_dataset = FUSeg2021Dataset(
        root_dir=str(data_root),
        split="train",
        transform=get_training_augmentation(IMAGE_SIZE)
    )
    
    val_dataset = FUSeg2021Dataset(
        root_dir=str(data_root),
        split="validation",
        transform=get_validation_augmentation(IMAGE_SIZE)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = SegmentationModel(
        architecture="unetplusplus",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        num_classes=1,
    ).to(DEVICE)
    
    # Loss - more Dice weight for imbalanced data
    criterion = DiceBCELoss(dice_weight=0.7, bce_weight=0.3)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training
    best_iou = 0
    output_dir = Path("outputs/fuseg_simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                iou, dice = compute_metrics(outputs, masks)
                val_iou += iou
                val_dice += dice
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_iou={val_iou:.4f} | val_dice={val_dice:.4f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
            }, output_dir / "best_model.pt")
            print(f"  ✅ Saved best model (IoU: {val_iou:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")
    
    print(f"\nTraining complete! Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    train()
