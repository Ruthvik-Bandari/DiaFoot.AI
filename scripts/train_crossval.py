#!/usr/bin/env python3
"""
DiaFootAI 5-Fold Cross-Validation Training
===========================================

Trains 5 models on different data splits to:
1. Prove model robustness (not overfitting to one split)
2. Get reliable performance estimates
3. Create ensemble of 5 models for better predictions

This is the gold standard for medical image analysis papers.

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
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from src.data.dataset import FUSeg2021Dataset
from src.data.augmentation import get_training_augmentation, get_validation_augmentation
from src.models.segmentation import SegmentationModel


# =============================================================================
# LOSS FUNCTIONS (same as train_advanced_v2.py)
# =============================================================================

class DiceBCELoss(nn.Module):
    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        bce = F.binary_cross_entropy(pred_sig, target, reduction="mean")
        smooth = 1e-5
        inter = (pred_sig.view(-1) * target.view(-1)).sum()
        dice = 1 - (2 * inter + smooth) / (pred_sig.view(-1).sum() + target.view(-1).sum() + smooth)
        return 0.7 * dice + 0.3 * bce


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
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
    """Fast GPU-based Boundary Loss using Sobel edge detection."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def get_boundary(self, mask):
        mask = mask.float()
        edge_x = F.conv2d(mask, self.sobel_x, padding=1)
        edge_y = F.conv2d(mask, self.sobel_y, padding=1)
        boundary = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        boundary = boundary / (boundary.max() + 1e-8)
        return boundary
    
    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        pred_boundary = self.get_boundary(pred_sig)
        target_boundary = self.get_boundary(target)
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        boundary_weight = target_boundary + 0.1
        weighted_bce = F.binary_cross_entropy(pred_sig, target, weight=boundary_weight, reduction='mean')
        return 0.5 * boundary_loss + 0.5 * weighted_bce


class CombinedLoss(nn.Module):
    """Combined loss with warmup schedule."""
    def __init__(self, use_boundary=True):
        super().__init__()
        self.dice_bce = DiceBCELoss()
        self.focal_tversky = FocalTverskyLoss()
        self.boundary = BoundaryLoss() if use_boundary else None
        self.use_boundary = use_boundary
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, pred, target):
        dice_bce_loss = self.dice_bce(pred, target)
        
        if self.current_epoch < 10:
            return dice_bce_loss
        elif self.current_epoch < 30:
            ft_weight = (self.current_epoch - 10) / 20 * 0.7
            ft_loss = self.focal_tversky(pred, target)
            combined = (1 - ft_weight) * dice_bce_loss + ft_weight * ft_loss
            
            if self.use_boundary and self.current_epoch >= 20:
                boundary_weight = (self.current_epoch - 20) / 10 * 0.1
                combined = combined + boundary_weight * self.boundary(pred, target)
            
            return combined
        else:
            ft_loss = self.focal_tversky(pred, target)
            combined = 0.3 * dice_bce_loss + 0.6 * ft_loss
            if self.use_boundary:
                combined = combined + 0.1 * self.boundary(pred, target)
            return combined


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


# =============================================================================
# SINGLE FOLD TRAINING
# =============================================================================

def train_fold(
    fold: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    full_dataset: FUSeg2021Dataset,
    args,
    device: torch.device
) -> dict:
    """Train a single fold and return metrics."""
    
    print(f"\n{'='*70}")
    print(f"FOLD {fold + 1}/5")
    print(f"{'='*70}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # Create data subsets
    # We need separate datasets with different transforms
    train_ds_full = FUSeg2021Dataset(
        args.data_dir,
        "train",
        get_training_augmentation(args.image_size)
    )
    val_ds_full = FUSeg2021Dataset(
        args.data_dir,
        "train",  # Same base data, different transform
        get_validation_augmentation(args.image_size)
    )
    
    train_subset = Subset(train_ds_full, train_indices)
    val_subset = Subset(val_ds_full, val_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    model = SegmentationModel(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
    ).to(device)
    
    # Loss, optimizer, scheduler
    criterion = CombinedLoss(use_boundary=args.boundary_loss).to(device)
    
    enc_params = list(model.model.encoder.parameters())
    dec_params = list(model.model.decoder.parameters()) + list(model.model.segmentation_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.encoder_lr},
        {"params": dec_params, "lr": args.decoder_lr}
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    ema = EMA(model, decay=0.999)
    
    # Training loop
    best_iou = 0
    patience_counter = 0
    fold_history = {'train_loss': [], 'val_iou': [], 'val_dice': []}
    
    for epoch in range(args.epochs):
        criterion.set_epoch(epoch)
        
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            optimizer.zero_grad()
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
        
        # Validation
        ema.apply(model)
        model.eval()
        
        total_intersection = 0
        total_union = 0
        total_pred_sum = 0
        total_mask_sum = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                pred = torch.sigmoid(model(images))
                pred_bin = (pred > 0.5).float()
                
                intersection = (pred_bin * masks).sum().item()
                union = pred_bin.sum().item() + masks.sum().item() - intersection
                
                total_intersection += intersection
                total_union += union
                total_pred_sum += pred_bin.sum().item()
                total_mask_sum += masks.sum().item()
        
        smooth = 1e-5
        val_iou = (total_intersection + smooth) / (total_union + smooth)
        val_dice = (2 * total_intersection + smooth) / (total_pred_sum + total_mask_sum + smooth)
        
        ema.restore(model)
        
        # Logging
        fold_history['train_loss'].append(train_loss)
        fold_history['val_iou'].append(val_iou)
        fold_history['val_dice'].append(val_dice)
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            best_dice = val_dice
            patience_counter = 0
            
            ema.apply(model)
            checkpoint = {
                "fold": fold,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_iou": val_iou,
                "val_dice": val_dice,
                "encoder": args.encoder,
                "architecture": args.architecture,
            }
            save_path = Path(args.output_dir) / f"fold_{fold+1}_best.pt"
            torch.save(checkpoint, save_path)
            ema.restore(model)
            print(f"Epoch {epoch+1} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f} | ✅ Best!")
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
            
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n📊 Fold {fold+1} Best: IoU={best_iou:.4f}, Dice={best_dice:.4f}")
    
    return {
        'fold': fold + 1,
        'best_iou': best_iou,
        'best_dice': best_dice,
        'history': fold_history,
        'model_path': str(save_path)
    }


# =============================================================================
# MAIN CROSS-VALIDATION
# =============================================================================

def run_cross_validation(args):
    """Run 5-fold cross-validation."""
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print("=" * 70)
    print("DiaFootAI 5-Fold Cross-Validation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}")
    print(f"Architecture: {args.architecture}")
    print(f"Boundary Loss: {'Enabled' if args.boundary_loss else 'Disabled'}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full dataset to get indices
    full_dataset = FUSeg2021Dataset(
        args.data_dir,
        "train",
        get_validation_augmentation(args.image_size)
    )
    
    n_samples = len(full_dataset)
    print(f"\nTotal samples: {n_samples}")
    print(f"Samples per fold: ~{n_samples // 5}")
    
    # Setup K-Fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Track results
    fold_results = []
    
    # Run each fold
    start_time = datetime.now()
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(n_samples))):
        fold_result = train_fold(
            fold=fold,
            train_indices=train_idx,
            val_indices=val_idx,
            full_dataset=full_dataset,
            args=args,
            device=device
        )
        fold_results.append(fold_result)
        
        # Save intermediate results
        with open(output_dir / "cv_results.json", 'w') as f:
            json.dump({
                'completed_folds': len(fold_results),
                'results': fold_results,
                'config': vars(args)
            }, f, indent=2)
    
    total_time = datetime.now() - start_time
    
    # Final summary
    ious = [r['best_iou'] for r in fold_results]
    dices = [r['best_dice'] for r in fold_results]
    
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 70)
    print("\n📊 Results per Fold:")
    print("-" * 40)
    for r in fold_results:
        print(f"  Fold {r['fold']}: IoU={r['best_iou']:.4f}, Dice={r['best_dice']:.4f}")
    print("-" * 40)
    print(f"\n🎯 Mean IoU:  {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"🎯 Mean Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"\n⏱️  Total Time: {total_time}")
    print(f"📁 Models saved in: {output_dir}")
    print("=" * 70)
    
    # Save final results
    final_results = {
        'mean_iou': float(np.mean(ious)),
        'std_iou': float(np.std(ious)),
        'mean_dice': float(np.mean(dices)),
        'std_dice': float(np.std(dices)),
        'fold_results': fold_results,
        'total_time_seconds': total_time.total_seconds(),
        'config': vars(args)
    }
    
    with open(output_dir / "cv_final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}/cv_final_results.json")
    
    return final_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DiaFootAI 5-Fold Cross-Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 5-fold CV with EfficientNet-B5
  python train_crossval.py --encoder efficientnet-b5 --boundary-loss

  # Run faster with fewer epochs per fold (for testing)
  python train_crossval.py --encoder efficientnet-b4 --epochs 30

  # Run on Windows with RTX (much faster!)
  python train_crossval.py --encoder efficientnet-b5 --num-workers 4
        """
    )
    
    # Model options
    parser.add_argument('--encoder', type=str, default='efficientnet-b5',
                        help='Encoder backbone (default: efficientnet-b5)')
    parser.add_argument('--architecture', type=str, default='unetplusplus',
                        help='Segmentation architecture (default: unetplusplus)')
    
    # Loss options
    parser.add_argument('--boundary-loss', action='store_true', default=True,
                        help='Enable boundary loss')
    parser.add_argument('--no-boundary-loss', dest='boundary_loss', action='store_false')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=80,
                        help='Max epochs per fold (default: 80)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Image size (default: 512)')
    parser.add_argument('--encoder-lr', type=float, default=1e-5,
                        help='Encoder learning rate')
    parser.add_argument('--decoder-lr', type=float, default=1e-4,
                        help='Decoder learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (default: 0)')
    
    # Data options
    parser.add_argument('--data-dir', type=str,
                        default='data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge',
                        help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='outputs/crossval',
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_cross_validation(args)
