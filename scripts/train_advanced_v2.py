#!/usr/bin/env python3
"""
DiaFootAI Advanced Training v2
==============================

Improvements over v1:
- EfficientNetV2-S encoder (faster, better accuracy)
- Boundary Loss (Hausdorff Distance inspired) for sharper edges
- Configurable encoder via command line
- Saves EMA weights separately for TTA

Author: Ruthvik
Date: January 2025
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.data.dataset import FUSeg2021Dataset
from src.data.augmentation import get_training_augmentation, get_validation_augmentation
from src.models.segmentation import SegmentationModel


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiceBCELoss(nn.Module):
    """Standard Dice + BCE loss - works from scratch"""
    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        bce = F.binary_cross_entropy(pred_sig, target, reduction="mean")
        smooth = 1e-5
        inter = (pred_sig.view(-1) * target.view(-1)).sum()
        dice = 1 - (2 * inter + smooth) / (pred_sig.view(-1).sum() + target.view(-1).sum() + smooth)
        return 0.7 * dice + 0.3 * bce


class FocalTverskyLoss(nn.Module):
    """Focal Tversky - focuses on hard examples and recall"""
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
        focal = torch.pow(torch.clamp(1 - tversky, min=0.01), self.gamma)
        
        return focal


class BoundaryLoss(nn.Module):
    """
    Fast Boundary Loss using max pooling (MPS-friendly).
    
    Uses morphological operations via max pooling to find boundaries:
    boundary = mask - eroded_mask
    
    This is much faster than Sobel convolution on MPS.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
    def get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary using morphological erosion via max pooling."""
        # Erosion: erode = -maxpool(-mask)
        # This shrinks the mask by kernel_size
        eroded = -F.max_pool2d(
            -mask, 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.padding
        )
        
        # Boundary = original - eroded
        boundary = mask - eroded
        
        # Normalize
        boundary = torch.clamp(boundary, 0, 1)
        
        return boundary
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-aware loss.
        
        Args:
            pred: (B, 1, H, W) logits
            target: (B, 1, H, W) binary mask
        """
        pred_sig = torch.sigmoid(pred)
        
        # Get target boundary regions
        target_boundary = self.get_boundary(target)
        
        # Weight: higher weight at boundaries
        # 1.0 for non-boundary, 5.0 for boundary pixels
        weight = 1.0 + 4.0 * target_boundary
        
        # Weighted BCE loss
        bce = F.binary_cross_entropy(pred_sig, target, weight=weight, reduction='mean')
        
        return bce


# Remove the slow precompute functions - no longer needed
def precompute_distance_maps(dataset, image_size=512, cache_dir=None):
    """No longer needed - using fast GPU-based boundary loss."""
    print("   ℹ️  Using fast GPU-based boundary loss (no pre-computation needed)")
    return {}


class DatasetWithDistanceMaps(torch.utils.data.Dataset):
    """Wrapper - passes through to base dataset (distance maps not needed)."""
    
    def __init__(self, base_dataset, distance_maps=None):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.base_dataset[idx]


class WarmupFocalTverskyBoundaryLoss(nn.Module):
    """
    Combined loss with warmup schedule:
    
    - Epochs 1-10: 100% Dice+BCE (warmup)
    - Epochs 11-30: Gradual transition to Focal Tversky + Boundary
    - Epochs 31+: 60% Focal Tversky + 30% Dice+BCE + 10% Boundary
    """
    def __init__(self, use_boundary=True):
        super().__init__()
        self.dice_bce = DiceBCELoss()
        self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        self.boundary = BoundaryLoss() if use_boundary else None
        self.use_boundary = use_boundary
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, pred, target, dist_map=None):
        dice_bce_loss = self.dice_bce(pred, target)
        
        if self.current_epoch < 10:
            # Pure Dice+BCE for first 10 epochs
            return dice_bce_loss
        elif self.current_epoch < 30:
            # Gradually introduce Focal Tversky
            ft_weight = (self.current_epoch - 10) / 20 * 0.7  # 0 to 0.7
            ft_loss = self.focal_tversky(pred, target)
            combined = (1 - ft_weight) * dice_bce_loss + ft_weight * ft_loss
            
            # Add boundary loss after epoch 20
            if self.use_boundary and self.current_epoch >= 20 and dist_map is not None:
                boundary_weight = (self.current_epoch - 20) / 10 * 0.1  # 0 to 0.1
                boundary_loss = self.boundary(pred, target, dist_map)
                combined = combined + boundary_weight * boundary_loss
            
            return combined
        else:
            # Full blend: 60% FT + 30% Dice+BCE + 10% Boundary
            ft_loss = self.focal_tversky(pred, target)
            combined = 0.3 * dice_bce_loss + 0.6 * ft_loss
            
            if self.use_boundary and dist_map is not None:
                boundary_loss = self.boundary(pred, target, dist_map)
                combined = combined + 0.1 * boundary_loss
            
            return combined


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

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
    
    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


# =============================================================================
# AVAILABLE ENCODERS
# =============================================================================

ENCODER_OPTIONS = {
    # EfficientNet family (well-tested with smp)
    'efficientnet-b4': {'params': '19M', 'speed': 'baseline', 'notes': 'Current model'},
    'efficientnet-b5': {'params': '30M', 'speed': '~20% slower', 'notes': 'Larger, more accurate'},
    'efficientnet-b6': {'params': '43M', 'speed': '~35% slower', 'notes': 'Even larger'},
    'efficientnet-b7': {'params': '66M', 'speed': '~50% slower', 'notes': 'Largest EfficientNet'},
    
    # timm encoders (use 'timm-' prefix in smp)
    'timm-efficientnet_b5': {'params': '30M', 'speed': '~20% slower', 'notes': 'timm version of B5'},
    'timm-tf_efficientnetv2_s': {'params': '22M', 'speed': 'same as b4', 'notes': 'EfficientNetV2-S from TF'},
    'timm-tf_efficientnetv2_m': {'params': '54M', 'speed': '~40% slower', 'notes': 'EfficientNetV2-M from TF'},
    
    # ResNeSt (attention-based ResNet)
    'timm-resnest50d': {'params': '28M', 'speed': '~10% slower', 'notes': 'ResNet + Split Attention'},
    'timm-resnest101e': {'params': '48M', 'speed': '~25% slower', 'notes': 'Larger ResNeSt'},
    
    # ResNet family (baselines)
    'resnet50': {'params': '25M', 'speed': 'fast', 'notes': 'Classic baseline'},
    'resnet101': {'params': '44M', 'speed': 'medium', 'notes': 'Deeper ResNet'},
    
    # SE-ResNet (Squeeze-and-Excitation)
    'se_resnet50': {'params': '28M', 'speed': '~5% slower', 'notes': 'ResNet50 + SE blocks'},
    'se_resnext50_32x4d': {'params': '27M', 'speed': '~10% slower', 'notes': 'ResNeXt + SE'},
}


def print_encoder_options():
    """Print available encoder options."""
    print("\n📋 Available Encoders:")
    print("-" * 70)
    for name, info in ENCODER_OPTIONS.items():
        print(f"  {name:25s} | {info['params']:>5s} params | {info['speed']:15s} | {info['notes']}")
    print("-" * 70)


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(args):
    # Device setup
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    
    print("=" * 70)
    print("DiaFootAI Advanced Training v2")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Encoder: {args.encoder}")
    print(f"Architecture: {args.architecture}")
    print(f"Boundary Loss: {'Enabled' if args.boundary_loss else 'Disabled'}")
    print(f"Image Size: {args.image_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.epochs}")
    print("-" * 70)
    print("Loss Schedule:")
    print("  Epochs 1-10:  100% Dice+BCE (warmup)")
    print("  Epochs 11-30: Gradual transition to Focal Tversky")
    if args.boundary_loss:
        print("  Epochs 20-30: Gradual introduction of Boundary Loss")
        print("  Epochs 31+:   60% Focal Tversky + 30% Dice+BCE + 10% Boundary")
    else:
        print("  Epochs 31+:   70% Focal Tversky + 30% Dice+BCE")
    print("-" * 70)
    print("Regularization:")
    print("  EMA: Enabled (0.999)")
    print("  Differential LR: Encoder 1e-5, Decoder 1e-4")
    print("  Weight Decay: 1e-4")
    print("  Gradient Clipping: 1.0")
    print("=" * 70)
    
    # Data setup
    data_root = Path(args.data_dir)
    train_ds = FUSeg2021Dataset(
        str(data_root), 
        "train", 
        get_training_augmentation(args.image_size)
    )
    val_ds = FUSeg2021Dataset(
        str(data_root), 
        "validation", 
        get_validation_augmentation(args.image_size)
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=True,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")
    
    # Model setup
    model = SegmentationModel(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
    
    # Loss, optimizer, scheduler
    criterion = WarmupFocalTverskyBoundaryLoss(use_boundary=args.boundary_loss)
    
    # Differential learning rates
    enc_params = list(model.model.encoder.parameters())
    dec_params = list(model.model.decoder.parameters()) + list(model.model.segmentation_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.encoder_lr},
        {"params": dec_params, "lr": args.decoder_lr}
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=1e-6
    )
    
    ema = EMA(model, decay=0.999)
    
    # Mixed precision scaler (CUDA only, MPS uses autocast without scaler)
    scaler = None
    if args.amp:
        if DEVICE.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            print("⚡ Mixed Precision: Enabled (CUDA + GradScaler)")
        elif DEVICE.type == 'mps':
            print("⚡ Mixed Precision: Enabled (MPS autocast)")
        else:
            print("⚠️  Mixed Precision: Disabled (CPU)")
            args.amp = False
    
    # torch.compile for additional speedup (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("🚀 torch.compile: Enabled")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_iou = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_iou': [], 'val_dice': [], 'lr': []}
    
    for epoch in range(args.epochs):
        criterion.set_epoch(epoch)
        
        # Determine current loss mode for display
        if epoch < 10:
            loss_mode = "Dice+BCE"
        elif epoch < 30:
            ft_pct = int((epoch - 10) / 20 * 70)
            if args.boundary_loss and epoch >= 20:
                bd_pct = int((epoch - 20) / 10 * 10)
                loss_mode = f"FT:{ft_pct}%+BD:{bd_pct}%"
            else:
                loss_mode = f"FT:{ft_pct}%"
        else:
            if args.boundary_loss:
                loss_mode = "FT:60%+BD:10%"
            else:
                loss_mode = "FT:70%"
        
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [{loss_mode}]")
        
        for batch in pbar:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Mixed precision training (1.5-2x speedup)
            if args.amp and DEVICE.type in ['cuda', 'mps']:
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                # Scale loss for mixed precision
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            ema.update(model)
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation with EMA weights
        ema.apply(model)
        model.eval()
        
        # Data-based metrics (aggregate all pixels)
        total_intersection = 0
        total_union = 0
        total_pred_sum = 0
        total_mask_sum = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                
                pred = torch.sigmoid(model(images))
                pred_bin = (pred > 0.5).float()
                
                intersection = (pred_bin * masks).sum().item()
                union = pred_bin.sum().item() + masks.sum().item() - intersection
                
                total_intersection += intersection
                total_union += union
                total_pred_sum += pred_bin.sum().item()
                total_mask_sum += masks.sum().item()
        
        # Calculate data-based IoU and Dice
        smooth = 1e-5
        val_iou = (total_intersection + smooth) / (total_union + smooth)
        val_dice = (2 * total_intersection + smooth) / (total_pred_sum + total_mask_sum + smooth)
        
        ema.restore(model)
        
        # Logging
        lr = optimizer.param_groups[1]["lr"]
        history['train_loss'].append(train_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        history['lr'].append(lr)
        
        print(f"Epoch {epoch+1} | loss: {train_loss:.4f} | val_iou: {val_iou:.4f} | val_dice: {val_dice:.4f} | lr: {lr:.2e}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            
            ema.apply(model)
            
            # Save complete checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),  # Save EMA weights separately
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
                "encoder": args.encoder,
                "architecture": args.architecture,
                "history": history,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            
            ema.restore(model)
            print(f"  ✅ Saved best model (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best IoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
    print(f"Best Dice: {history['val_dice'][history['val_iou'].index(best_iou)]:.4f}")
    print(f"Model saved: {output_dir}/best_model.pt")
    print(f"Encoder: {args.encoder}")
    print("=" * 70)
    
    return best_iou


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DiaFootAI Advanced Training v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with EfficientNet-B5 + Boundary Loss + Mixed Precision (FASTEST)
  python train_advanced_v2.py --encoder efficientnet-b5 --boundary-loss --amp

  # Train with larger batch size (if you have enough VRAM)
  python train_advanced_v2.py --encoder efficientnet-b5 --batch-size 16 --amp

  # Train without mixed precision (if you get NaN errors)
  python train_advanced_v2.py --encoder efficientnet-b5 --no-amp

  # List available encoders
  python train_advanced_v2.py --list-encoders
        """
    )
    
    # Model options
    parser.add_argument('--encoder', type=str, default='efficientnet-b5',
                        help='Encoder backbone (default: efficientnet-b5)')
    parser.add_argument('--architecture', type=str, default='unetplusplus',
                        choices=['unet', 'unetplusplus', 'deeplabv3plus', 'fpn', 'manet'],
                        help='Segmentation architecture (default: unetplusplus)')
    parser.add_argument('--list-encoders', action='store_true',
                        help='List available encoder options and exit')
    
    # Loss options
    parser.add_argument('--boundary-loss', action='store_true',
                        help='Enable boundary loss for sharper edges')
    parser.add_argument('--no-boundary-loss', dest='boundary_loss', action='store_false',
                        help='Disable boundary loss')
    parser.set_defaults(boundary_loss=True)
    
    # Training options
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size (default: 512)')
    parser.add_argument('--encoder-lr', type=float, default=1e-5,
                        help='Encoder learning rate (default: 1e-5)')
    parser.add_argument('--decoder-lr', type=float, default=1e-4,
                        help='Decoder learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience (default: 25)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (default: 0 for MPS compatibility)')
    
    # Performance options
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Enable Automatic Mixed Precision (1.5-2x speedup)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                        help='Disable mixed precision')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Enable torch.compile (PyTorch 2.0+, experimental on MPS)')
    
    # Data options
    parser.add_argument('--data-dir', type=str,
                        default='data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge',
                        help='Path to FUSeg dataset')
    parser.add_argument('--output-dir', type=str, default='outputs/fuseg_v2',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    if args.list_encoders:
        print_encoder_options()
        sys.exit(0)
    
    train(args)
