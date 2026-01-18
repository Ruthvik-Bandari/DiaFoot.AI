#!/usr/bin/env python3
"""
Advanced Training Script for DiaFootAI
=======================================

Includes:
- Focal Tversky Loss (better for imbalanced data)
- Exponential Moving Average (EMA)
- Differential Learning Rate
- Cosine Annealing with Warm Restarts

Author: Ruthvik
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
import json

from src.data.dataset import FUSeg2021Dataset
from src.data.augmentation import get_training_augmentation, get_validation_augmentation
from src.models.segmentation import SegmentationModel


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow((1 - tversky), self.gamma)


class CombinedLoss(nn.Module):
    def __init__(self, tversky_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.focal_tversky = FocalTverskyLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return self.tversky_weight * self.focal_tversky(pred, target) + self.bce_weight * self.bce(pred, target)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def compute_metrics(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    smooth = 1e-5
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()
    
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    return {"iou": iou.item(), "dice": dice.item()}


def train():
    BATCH_SIZE = 8
    EPOCHS = 150
    LR = 3e-4
    MIN_LR = 1e-6
    IMAGE_SIZE = 512
    PATIENCE = 25
    EMA_DECAY = 0.999
    
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() 
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    )
    
    print("=" * 60)
    print("DiaFootAI Advanced Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Loss: Focal Tversky + BCE")
    print(f"EMA Decay: {EMA_DECAY}")
    print(f"Differential LR: Encoder {LR*0.1:.0e}, Decoder {LR:.0e}")
    print("=" * 60)
    
    data_root = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge")
    
    train_dataset = FUSeg2021Dataset(str(data_root), "train", get_training_augmentation(IMAGE_SIZE))
    val_dataset = FUSeg2021Dataset(str(data_root), "validation", get_validation_augmentation(IMAGE_SIZE))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    model = SegmentationModel(
        architecture="unetplusplus",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        num_classes=1,
    ).to(DEVICE)
    
    criterion = CombinedLoss()
    
    encoder_params = list(model.model.encoder.parameters())
    decoder_params = list(model.model.decoder.parameters()) + list(model.model.segmentation_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": LR * 0.1},
        {"params": decoder_params, "lr": LR},
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=MIN_LR)
    ema = EMA(model, decay=EMA_DECAY)
    
    output_dir = Path("outputs/fuseg_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_iou = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    
    for epoch in range(EPOCHS):
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        ema.apply_shadow()
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
                metrics = compute_metrics(outputs, masks)
                val_iou += metrics["iou"]
                val_dice += metrics["dice"]
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        ema.restore()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        
        lr = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"val_iou: {val_iou:.4f} | val_dice: {val_dice:.4f} | lr: {lr:.2e}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            ema.apply_shadow()
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
            }, output_dir / "best_model.pt")
            ema.restore()
            print(f"  ✅ Saved best model (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print("=" * 60)
    
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    train()
