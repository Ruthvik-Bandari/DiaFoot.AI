#!/usr/bin/env python3
"""
DiaFootAI Ensemble Inference
============================

Combines multiple models (cross-validation folds) for improved predictions.
Optionally integrates Test Time Augmentation (TTA) for maximum accuracy.

Expected improvement: +1-2% IoU over single model

Author: Ruthvik
Date: January 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


@dataclass
class EnsembleConfig:
    """Configuration for ensemble inference."""
    image_size: int = 512
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    threshold: float = 0.5
    merge_strategy: str = 'mean'  # 'mean', 'max', 'weighted'


def load_model(
    checkpoint_path: str,
    encoder: str = 'efficientnet-b4',
    architecture: str = 'unetplusplus',
    device: str = 'cpu'
) -> nn.Module:
    """Load a trained model from checkpoint."""
    
    if architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
    elif architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


class TTAPredictor:
    """Test Time Augmentation predictor."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> List[Tuple[str, callable, callable]]:
        """Build TTA transforms and their inverses."""
        transforms = [
            ('original', lambda x: x, lambda x: x),
            ('hflip', lambda x: np.fliplr(x).copy(), lambda x: np.fliplr(x).copy()),
            ('vflip', lambda x: np.flipud(x).copy(), lambda x: np.flipud(x).copy()),
            ('rot90', lambda x: np.rot90(x, 1).copy(), lambda x: np.rot90(x, -1).copy()),
            ('rot180', lambda x: np.rot90(x, 2).copy(), lambda x: np.rot90(x, 2).copy()),
            ('rot270', lambda x: np.rot90(x, 3).copy(), lambda x: np.rot90(x, 1).copy()),
            ('hflip_rot90', 
             lambda x: np.rot90(np.fliplr(x), 1).copy(), 
             lambda x: np.fliplr(np.rot90(x, -1)).copy()),
            ('vflip_rot90',
             lambda x: np.rot90(np.flipud(x), 1).copy(),
             lambda x: np.flipud(np.rot90(x, -1)).copy()),
        ]
        return transforms
    
    def predict_with_tta(
        self, 
        model: nn.Module, 
        image: np.ndarray,
        device: str
    ) -> np.ndarray:
        """Run prediction with TTA."""
        config = self.config
        predictions = []
        
        preprocess = A.Compose([
            A.LongestMaxSize(max_size=config.image_size),
            A.PadIfNeeded(
                min_height=config.image_size,
                min_width=config.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0
            ),
            A.Normalize(mean=config.mean, std=config.std),
            ToTensorV2()
        ])
        
        for name, forward_fn, inverse_fn in self.transforms:
            # Apply forward transform
            aug_image = forward_fn(image)
            
            # Preprocess
            transformed = preprocess(image=aug_image)
            tensor = transformed['image'].unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                pred = model(tensor)
                pred = torch.sigmoid(pred)
            
            # Convert to numpy
            pred_np = pred.squeeze().cpu().numpy()
            
            # Resize to original padded size if needed
            if pred_np.shape[0] != config.image_size or pred_np.shape[1] != config.image_size:
                pred_np = cv2.resize(pred_np, (config.image_size, config.image_size))
            
            # Apply inverse transform
            pred_np = inverse_fn(pred_np)
            predictions.append(pred_np)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred


class EnsembleInference:
    """
    Ensemble inference combining multiple models.
    
    Supports:
    - Multiple cross-validation fold models
    - Test Time Augmentation
    - Various merge strategies
    """
    
    def __init__(
        self,
        checkpoint_paths: List[str],
        encoder: str = 'efficientnet-b4',
        architecture: str = 'unetplusplus',
        device: Optional[str] = None,
        use_tta: bool = True,
        config: Optional[EnsembleConfig] = None
    ):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        self.config = config or EnsembleConfig()
        self.use_tta = use_tta
        
        # Load all models
        print(f"\n📦 Loading {len(checkpoint_paths)} models...")
        self.models = []
        for i, path in enumerate(checkpoint_paths):
            model = load_model(path, encoder, architecture, device)
            self.models.append(model)
            print(f"   ✅ Model {i+1}: {Path(path).parent.name}/{Path(path).name}")
        
        # TTA predictor
        if use_tta:
            self.tta = TTAPredictor(self.config)
            print(f"   ✅ TTA enabled ({len(self.tta.transforms)} transforms)")
        
        # Preprocessing for non-TTA
        self.preprocess = A.Compose([
            A.LongestMaxSize(max_size=self.config.image_size),
            A.PadIfNeeded(
                min_height=self.config.image_size,
                min_width=self.config.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0
            ),
            A.Normalize(mean=self.config.mean, std=self.config.std),
            ToTensorV2()
        ])
    
    def predict_single(self, image: np.ndarray) -> np.ndarray:
        """Predict on a single image using ensemble."""
        all_predictions = []
        
        for model in self.models:
            if self.use_tta:
                pred = self.tta.predict_with_tta(model, image, self.device)
            else:
                # Standard prediction
                transformed = self.preprocess(image=image)
                tensor = transformed['image'].unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    pred = model(tensor)
                    pred = torch.sigmoid(pred)
                
                pred = pred.squeeze().cpu().numpy()
            
            all_predictions.append(pred)
        
        # Merge predictions
        if self.config.merge_strategy == 'mean':
            ensemble_pred = np.mean(all_predictions, axis=0)
        elif self.config.merge_strategy == 'max':
            ensemble_pred = np.max(all_predictions, axis=0)
        else:
            ensemble_pred = np.mean(all_predictions, axis=0)
        
        return ensemble_pred
    
    def predict_binary(self, image: np.ndarray) -> np.ndarray:
        """Predict binary mask."""
        prob = self.predict_single(image)
        return (prob > self.config.threshold).astype(np.uint8)


def compute_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute segmentation metrics."""
    if pred.max() <= 1.0 and pred.min() >= 0.0:
        pred_binary = (pred > threshold).astype(np.float32)
    else:
        pred_binary = pred.astype(np.float32)
    
    target = target.astype(np.float32)
    if target.max() > 1:
        target = target / 255.0
    
    pred_flat = pred_binary.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (pred_flat.sum() + target_flat.sum() + 1e-7)
    
    tp = intersection
    fp = pred_flat.sum() - intersection
    fn = target_flat.sum() - intersection
    
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall)
    }


def evaluate_ensemble(
    ensemble: EnsembleInference,
    image_dir: Path,
    mask_dir: Path,
    config: EnsembleConfig
) -> Dict:
    """Evaluate ensemble on validation set."""
    
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    
    # Accumulators for data-based metrics
    total_intersection = 0.0
    total_union = 0.0
    total_pred = 0.0
    total_target = 0.0
    
    # Per-image metrics
    image_metrics = []
    
    print(f"\n🔬 Evaluating ensemble on {len(image_files)} images...")
    
    for img_path in tqdm(image_files, desc="Evaluating"):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = mask_dir / img_path.stem.replace('.png', '.jpg')
        if not mask_path.exists():
            mask_path = mask_dir / (img_path.stem + '.png')
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Resize mask to match prediction size
        mask_resized = cv2.resize(mask, (config.image_size, config.image_size))
        mask_binary = (mask_resized > 127).astype(np.float32)
        
        # Predict
        pred_prob = ensemble.predict_single(image)
        pred_binary = (pred_prob > config.threshold).astype(np.float32)
        
        # Accumulate for data-based metrics
        intersection = (pred_binary * mask_binary).sum()
        union = pred_binary.sum() + mask_binary.sum() - intersection
        
        total_intersection += intersection
        total_union += union
        total_pred += pred_binary.sum()
        total_target += mask_binary.sum()
        
        # Per-image metrics
        metrics = compute_metrics(pred_binary, mask_binary)
        image_metrics.append(metrics)
    
    # Data-based metrics (matches training evaluation)
    data_iou = total_intersection / (total_union + 1e-7)
    data_dice = (2 * total_intersection) / (total_pred + total_target + 1e-7)
    
    # Image-based metrics (per-image average)
    avg_iou = np.mean([m['iou'] for m in image_metrics])
    avg_dice = np.mean([m['dice'] for m in image_metrics])
    avg_precision = np.mean([m['precision'] for m in image_metrics])
    avg_recall = np.mean([m['recall'] for m in image_metrics])
    
    return {
        'data_based': {
            'iou': float(data_iou),
            'dice': float(data_dice)
        },
        'image_based': {
            'iou': float(avg_iou),
            'dice': float(avg_dice),
            'precision': float(avg_precision),
            'recall': float(avg_recall)
        },
        'num_images': len(image_metrics)
    }


def main():
    parser = argparse.ArgumentParser(description='DiaFootAI Ensemble Inference')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Paths to model checkpoints')
    parser.add_argument('--encoder', type=str, default='efficientnet-b4',
                        help='Encoder architecture')
    parser.add_argument('--architecture', type=str, default='unetplusplus',
                        help='Model architecture')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--use-tta', action='store_true', default=True,
                        help='Use Test Time Augmentation')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable Test Time Augmentation')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binary threshold')
    parser.add_argument('--output', type=str, default='outputs/ensemble_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Determine TTA setting
    use_tta = args.use_tta and not args.no_tta
    
    # Setup paths
    data_dir = Path(args.data_dir)
    image_dir = data_dir / 'validation' / 'images'
    mask_dir = data_dir / 'validation' / 'labels'
    
    if not image_dir.exists():
        image_dir = data_dir / 'images'
        mask_dir = data_dir / 'labels'
    
    print("=" * 70)
    print("ENSEMBLE INFERENCE")
    print("=" * 70)
    print(f"Models: {len(args.checkpoints)}")
    print(f"Encoder: {args.encoder}")
    print(f"TTA: {'Enabled' if use_tta else 'Disabled'}")
    print(f"Data: {data_dir}")
    print("=" * 70)
    
    # Config
    config = EnsembleConfig(threshold=args.threshold)
    
    # Create ensemble
    ensemble = EnsembleInference(
        checkpoint_paths=args.checkpoints,
        encoder=args.encoder,
        architecture=args.architecture,
        use_tta=use_tta,
        config=config
    )
    
    # Evaluate
    results = evaluate_ensemble(ensemble, image_dir, mask_dir, config)
    
    # Print results
    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)
    
    print(f"\n📊 DATA-BASED METRICS:")
    print(f"   IoU:  {results['data_based']['iou']:.4f} ({results['data_based']['iou']*100:.2f}%)")
    print(f"   Dice: {results['data_based']['dice']:.4f} ({results['data_based']['dice']*100:.2f}%)")
    
    print(f"\n📈 IMAGE-BASED METRICS:")
    print(f"   IoU:       {results['image_based']['iou']:.4f} ({results['image_based']['iou']*100:.2f}%)")
    print(f"   Dice:      {results['image_based']['dice']:.4f} ({results['image_based']['dice']*100:.2f}%)")
    print(f"   Precision: {results['image_based']['precision']:.4f}")
    print(f"   Recall:    {results['image_based']['recall']:.4f}")
    
    print("=" * 70)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'checkpoints': args.checkpoints,
                'encoder': args.encoder,
                'use_tta': use_tta,
                'threshold': args.threshold
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
