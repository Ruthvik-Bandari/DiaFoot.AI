#!/usr/bin/env python3
"""
Advanced Training with Focal Tversky WARMUP
- Epochs 1-10: Use Dice+BCE (gets model to reasonable state)
- Epochs 11+: Gradually introduce Focal Tversky
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FUSeg2021Dataset
from src.data.augmentation import get_training_augmentation, get_validation_augmentation
from src.models.segmentation import SegmentationModel


class DiceBCELoss(nn.Module):
    """Standard loss - works from scratch"""
    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        bce = F.binary_cross_entropy(pred_sig, target, reduction="mean")
        smooth = 1e-5
        inter = (pred_sig.view(-1) * target.view(-1)).sum()
        dice = 1 - (2 * inter + smooth) / (pred_sig.view(-1).sum() + target.view(-1).sum() + smooth)
        return 0.7 * dice + 0.3 * bce


class FocalTverskyLoss(nn.Module):
    """Focal Tversky - only works after warmup"""
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight (higher = focus on recall)
        self.gamma = gamma  # Focal parameter
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        smooth = 1e-5
        
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        
        # Focal component - but clamped to prevent instability
        focal = torch.pow(torch.clamp(1 - tversky, min=0.01), self.gamma)
        
        return focal


class WarmupFocalTverskyLoss(nn.Module):
    """
    Combines Dice+BCE with Focal Tversky using warmup schedule.
    
    - Epochs 1-10: 100% Dice+BCE, 0% Focal Tversky
    - Epochs 11-30: Gradually increase Focal Tversky
    - Epochs 31+: 30% Dice+BCE, 70% Focal Tversky
    """
    def __init__(self):
        super().__init__()
        self.dice_bce = DiceBCELoss()
        self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, pred, target):
        dice_bce_loss = self.dice_bce(pred, target)
        
        if self.current_epoch < 10:
            # Pure Dice+BCE for first 10 epochs
            return dice_bce_loss
        elif self.current_epoch < 30:
            # Gradually introduce Focal Tversky
            ft_weight = (self.current_epoch - 10) / 20 * 0.7  # 0 to 0.7
            ft_loss = self.focal_tversky(pred, target)
            return (1 - ft_weight) * dice_bce_loss + ft_weight * ft_loss
        else:
            # Full Focal Tversky blend
            ft_loss = self.focal_tversky(pred, target)
            return 0.3 * dice_bce_loss + 0.7 * ft_loss


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data
    
    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]
    
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]


def train():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 60)
    print("DiaFootAI Advanced Training - Focal Tversky with Warmup")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print("Loss Schedule:")
    print("  Epochs 1-10:  100% Dice+BCE (warmup)")
    print("  Epochs 11-30: Gradual transition to Focal Tversky")
    print("  Epochs 31+:   70% Focal Tversky + 30% Dice+BCE")
    print("EMA: Enabled (0.999)")
    print("Differential LR: Encoder 1e-5, Decoder 1e-4")
    print("=" * 60)
    
    data_root = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge")
    train_ds = FUSeg2021Dataset(str(data_root), "train", get_training_augmentation(512))
    val_ds = FUSeg2021Dataset(str(data_root), "validation", get_validation_augmentation(512))
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    model = SegmentationModel(
        architecture="unetplusplus",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
    ).to(DEVICE)
    
    criterion = WarmupFocalTverskyLoss()
    
    enc_params = list(model.model.encoder.parameters())
    dec_params = list(model.model.decoder.parameters()) + list(model.model.segmentation_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": 1e-5},
        {"params": dec_params, "lr": 1e-4}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    ema = EMA(model)
    
    output_dir = Path("outputs/fuseg_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_iou, patience = 0, 0
    
    for epoch in range(100):
        criterion.set_epoch(epoch)
        
        # Determine current loss mode
        if epoch < 10:
            loss_mode = "Dice+BCE"
        elif epoch < 30:
            ft_pct = int((epoch - 10) / 20 * 70)
            loss_mode = f"FT:{ft_pct}%"
        else:
            loss_mode = "FT:70%"
        
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/100 [{loss_mode}]"):
            images, masks = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()
        
        ema.apply(model)
        model.eval()
        val_iou, val_dice, n = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch["image"].to(DEVICE), batch["mask"].to(DEVICE)
                pred = torch.sigmoid(model(images))
                pred_bin = (pred > 0.5).float()
                smooth = 1e-5
                inter = (pred_bin * masks).sum()
                union = pred_bin.sum() + masks.sum() - inter
                val_iou += ((inter + smooth) / (union + smooth)).item()
                val_dice += ((2 * inter + smooth) / (pred_bin.sum() + masks.sum() + smooth)).item()
                n += 1
        val_iou /= n
        val_dice /= n
        ema.restore(model)
        
        lr = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch+1} | loss: {train_loss:.4f} | val_iou: {val_iou:.4f} | val_dice: {val_dice:.4f} | lr: {lr:.2e}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            patience = 0
            ema.apply(model)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_iou": val_iou, "val_dice": val_dice}, output_dir / "best_model.pt")
            ema.restore(model)
            print(f"  ✅ Saved best (IoU: {best_iou:.4f})")
        else:
            patience += 1
            if patience >= 25:
                print("Early stopping!")
                break
    
    print(f"\nDone! Best IoU: {best_iou:.4f}")
    print(f"Model: {output_dir}/best_model.pt")


if __name__ == "__main__":
    train()
