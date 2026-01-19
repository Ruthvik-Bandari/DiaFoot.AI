"""
Test Time Augmentation (TTA) Inference Module
=============================================

Production-grade TTA inference for diabetic foot ulcer segmentation.
Used by FUSeg 2021 challenge winners to boost IoU by 1-2%.

Features:
- 8 TTA transforms (original + flips + rotations + scales)
- Multiple merge strategies (mean, max, weighted)
- Proper inverse transforms for mask alignment
- Batch processing for efficiency
- Detailed metrics comparison (with/without TTA)

Author: Ruthvik
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Union
from dataclasses import dataclass
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class TTAConfig:
    """Configuration for Test Time Augmentation."""
    image_size: int = 512
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    # TTA options
    use_hflip: bool = True
    use_vflip: bool = True
    use_rotate90: bool = True
    use_rotate180: bool = True
    use_rotate270: bool = True
    use_scale_up: bool = True      # 1.1x scale
    use_scale_down: bool = True    # 0.9x scale
    use_multiscale: bool = False   # Additional scales (1.2x, 0.8x)
    
    # Merge strategy: 'mean', 'max', 'weighted', 'geometric'
    merge_strategy: str = 'mean'
    
    # For weighted merge - weights for each transform
    # [original, hflip, vflip, rot90, rot180, rot270, scale_up, scale_down]
    weights: Optional[List[float]] = None
    
    # Threshold for binary prediction
    threshold: float = 0.5
    
    def get_num_transforms(self) -> int:
        """Count number of active TTA transforms."""
        count = 1  # Original always included
        if self.use_hflip: count += 1
        if self.use_vflip: count += 1
        if self.use_rotate90: count += 1
        if self.use_rotate180: count += 1
        if self.use_rotate270: count += 1
        if self.use_scale_up: count += 1
        if self.use_scale_down: count += 1
        if self.use_multiscale: count += 2
        return count


class TTATransforms:
    """
    Manages TTA transformations and their inverses.
    
    Each transform has a corresponding inverse to align predictions
    back to the original image orientation.
    """
    
    def __init__(self, config: TTAConfig):
        self.config = config
        self.image_size = config.image_size
        self.mean = np.array(config.mean)
        self.std = np.array(config.std)
        
        # Build base preprocessing (resize + pad)
        self.preprocess = A.Compose([
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0
            ),
        ])
        
        self.normalize = A.Compose([
            A.Normalize(mean=config.mean, std=config.std),
            ToTensorV2(),
        ])
    
    def get_transforms(self) -> List[Tuple[str, Callable, Callable]]:
        """
        Get list of (name, forward_transform, inverse_transform) tuples.
        
        Returns:
            List of tuples: (transform_name, forward_fn, inverse_fn)
        """
        transforms = []
        
        # Original - identity transform
        transforms.append((
            "original",
            lambda x: x,
            lambda x: x
        ))
        
        if self.config.use_hflip:
            transforms.append((
                "hflip",
                lambda x: np.fliplr(x).copy(),
                lambda x: np.fliplr(x).copy()
            ))
        
        if self.config.use_vflip:
            transforms.append((
                "vflip",
                lambda x: np.flipud(x).copy(),
                lambda x: np.flipud(x).copy()
            ))
        
        if self.config.use_rotate90:
            transforms.append((
                "rotate90",
                lambda x: np.rot90(x, k=1).copy(),
                lambda x: np.rot90(x, k=-1).copy()
            ))
        
        if self.config.use_rotate180:
            transforms.append((
                "rotate180",
                lambda x: np.rot90(x, k=2).copy(),
                lambda x: np.rot90(x, k=-2).copy()
            ))
        
        if self.config.use_rotate270:
            transforms.append((
                "rotate270",
                lambda x: np.rot90(x, k=3).copy(),
                lambda x: np.rot90(x, k=-3).copy()
            ))
        
        if self.config.use_scale_up:
            transforms.append((
                "scale_up",
                lambda x: self._scale(x, 1.1),
                lambda x: x  # No inverse needed - resize to original handles it
            ))
        
        if self.config.use_scale_down:
            transforms.append((
                "scale_down",
                lambda x: self._scale(x, 0.9),
                lambda x: x  # No inverse needed - resize to original handles it
            ))
        
        if self.config.use_multiscale:
            transforms.append((
                "scale_1.2x",
                lambda x: self._scale(x, 1.2),
                lambda x: x  # No inverse needed
            ))
            transforms.append((
                "scale_0.8x",
                lambda x: self._scale(x, 0.8),
                lambda x: x  # No inverse needed
            ))
        
        return transforms
    
    def _scale(self, img: np.ndarray, scale: float) -> np.ndarray:
        """Scale image by factor."""
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if len(img.shape) == 3:
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def apply_forward(
        self, 
        image: np.ndarray, 
        transform_fn: Callable
    ) -> torch.Tensor:
        """
        Apply forward transform and convert to tensor.
        
        Args:
            image: Input image (H, W, C) uint8
            transform_fn: Transform function
        
        Returns:
            Preprocessed tensor (C, H, W)
        """
        # Apply geometric transform
        transformed = transform_fn(image)
        
        # Preprocess (resize + pad)
        preprocessed = self.preprocess(image=transformed)['image']
        
        # Normalize and convert to tensor
        normalized = self.normalize(image=preprocessed)['image']
        
        return normalized
    
    def apply_inverse(
        self,
        prediction: np.ndarray,
        inverse_fn: Callable,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Apply inverse transform to prediction mask.
        
        Args:
            prediction: Model output (H, W) float32 - typically 512x512
            inverse_fn: Inverse transform function
            original_shape: Original image shape (H, W)
        
        Returns:
            Aligned prediction (H, W)
        """
        # Apply inverse geometric transform FIRST (on square prediction)
        # This handles rotations correctly before resizing to potentially non-square
        aligned = inverse_fn(prediction)
        
        # Then resize to original shape
        h, w = original_shape
        resized = cv2.resize(aligned, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return resized


class TTAPredictor:
    """
    Test Time Augmentation predictor for segmentation models.
    
    Usage:
        predictor = TTAPredictor(model, config)
        prediction = predictor.predict(image)
        
        # Or with mask for evaluation
        iou = predictor.evaluate(image, mask)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TTAConfig] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model: Trained segmentation model
            config: TTA configuration
            device: Device to run inference on
        """
        self.model = model
        self.config = config or TTAConfig()
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tta_transforms = TTATransforms(config)
        self.transforms = self.tta_transforms.get_transforms()
        
        # Set up weights for weighted merge
        if self.config.weights is None:
            # Default: equal weights
            self.weights = [1.0] * len(self.transforms)
        else:
            self.weights = self.config.weights
        
        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        return_probs: bool = False,
        return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Dict]:
        """
        Predict segmentation mask with TTA.
        
        Args:
            image: Input image (H, W, C) uint8
            return_probs: If True, return probability map instead of binary
            return_all: If True, return all individual predictions
        
        Returns:
            Binary mask or probability map (H, W)
            Optionally returns dict with all predictions if return_all=True
        """
        original_shape = image.shape[:2]  # (H, W)
        predictions = []
        
        for name, forward_fn, inverse_fn in self.transforms:
            # Apply forward transform and preprocess
            tensor = self.tta_transforms.apply_forward(image, forward_fn)
            
            # Add batch dimension and move to device
            batch = tensor.unsqueeze(0).to(self.device)
            
            # Model inference
            output = self.model(batch)
            
            # Handle different output formats
            if isinstance(output, dict):
                output = output.get('out', output.get('mask', list(output.values())[0]))
            
            # Convert to probability
            if output.shape[1] > 1:  # Multi-class
                prob = F.softmax(output, dim=1)[:, 1]  # Take foreground class
            else:
                prob = torch.sigmoid(output)
            
            # Convert to numpy
            prob_np = prob.squeeze().cpu().detach().numpy()
            
            # Apply inverse transform to align with original
            aligned = self.tta_transforms.apply_inverse(prob_np, inverse_fn, original_shape)
            predictions.append(aligned)
        
        # Merge predictions
        merged = self._merge_predictions(predictions)
        
        if return_all:
            return {
                'merged': merged if return_probs else (merged > self.config.threshold).astype(np.uint8),
                'merged_prob': merged,
                'individual': predictions,
                'transform_names': [t[0] for t in self.transforms]
            }
        
        if return_probs:
            return merged
        else:
            return (merged > self.config.threshold).astype(np.uint8)
    
    def _merge_predictions(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Merge multiple predictions using configured strategy."""
        preds = np.stack(predictions, axis=0)  # (N, H, W)
        
        if self.config.merge_strategy == 'mean':
            return np.mean(preds, axis=0)
        
        elif self.config.merge_strategy == 'max':
            return np.max(preds, axis=0)
        
        elif self.config.merge_strategy == 'weighted':
            weights = np.array(self.weights).reshape(-1, 1, 1)
            return np.sum(preds * weights, axis=0)
        
        elif self.config.merge_strategy == 'geometric':
            # Geometric mean - good for probabilities
            epsilon = 1e-7
            preds = np.clip(preds, epsilon, 1 - epsilon)
            log_preds = np.log(preds)
            return np.exp(np.mean(log_preds, axis=0))
        
        else:
            raise ValueError(f"Unknown merge strategy: {self.config.merge_strategy}")
    
    def predict_single(self, image: np.ndarray) -> np.ndarray:
        """Predict without TTA (single forward pass)."""
        original_shape = image.shape[:2]
        
        # Use only the original transform
        tensor = self.tta_transforms.apply_forward(image, lambda x: x)
        batch = tensor.unsqueeze(0).to(self.device)
        
        output = self.model(batch)
        if isinstance(output, dict):
            output = output.get('out', output.get('mask', list(output.values())[0]))
        
        if output.shape[1] > 1:
            prob = F.softmax(output, dim=1)[:, 1]
        else:
            prob = torch.sigmoid(output)
        
        prob_np = prob.squeeze().cpu().detach().numpy()
        
        # Resize to original shape
        h, w = original_shape
        resized = cv2.resize(prob_np, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return (resized > self.config.threshold).astype(np.uint8)
    
    def predict_at_model_size(self, image: np.ndarray) -> np.ndarray:
        """
        Predict without TTA, keeping output at model resolution (512x512).
        This matches training evaluation exactly.
        """
        # Use only the original transform
        tensor = self.tta_transforms.apply_forward(image, lambda x: x)
        batch = tensor.unsqueeze(0).to(self.device)
        
        output = self.model(batch)
        if isinstance(output, dict):
            output = output.get('out', output.get('mask', list(output.values())[0]))
        
        if output.shape[1] > 1:
            prob = F.softmax(output, dim=1)[:, 1]
        else:
            prob = torch.sigmoid(output)
        
        prob_np = prob.squeeze().cpu().detach().numpy()
        
        # Keep at model resolution - don't resize
        return (prob_np > self.config.threshold).astype(np.uint8)
    
    def predict_at_model_size_tta(self, image: np.ndarray) -> np.ndarray:
        """
        Predict with TTA, keeping output at model resolution (512x512).
        This matches training evaluation exactly.
        """
        predictions = []
        
        for name, forward_fn, inverse_fn in self.transforms:
            # Apply forward transform and preprocess
            tensor = self.tta_transforms.apply_forward(image, forward_fn)
            
            # Add batch dimension and move to device
            batch = tensor.unsqueeze(0).to(self.device)
            
            # Model inference
            output = self.model(batch)
            
            # Handle different output formats
            if isinstance(output, dict):
                output = output.get('out', output.get('mask', list(output.values())[0]))
            
            # Convert to probability
            if output.shape[1] > 1:  # Multi-class
                prob = F.softmax(output, dim=1)[:, 1]
            else:
                prob = torch.sigmoid(output)
            
            # Convert to numpy - keep at 512x512
            prob_np = prob.squeeze().cpu().detach().numpy()
            
            # Apply inverse geometric transform (stays at 512x512)
            aligned = inverse_fn(prob_np)
            predictions.append(aligned)
        
        # Merge predictions
        merged = self._merge_predictions(predictions)
        
        return (merged > self.config.threshold).astype(np.uint8)


def calculate_metrics(
    pred: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate segmentation metrics.
    
    Args:
        pred: Predicted binary mask
        mask: Ground truth binary mask
    
    Returns:
        Dictionary with IoU, Dice, Precision, Recall
    """
    pred = pred.astype(bool)
    mask = mask.astype(bool)
    
    intersection = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    pred_sum = pred.sum()
    mask_sum = mask.sum()
    
    # IoU (Jaccard Index)
    iou = intersection / (union + 1e-7)
    
    # Dice coefficient
    dice = (2 * intersection) / (pred_sum + mask_sum + 1e-7)
    
    # Precision
    precision = intersection / (pred_sum + 1e-7)
    
    # Recall
    recall = intersection / (mask_sum + 1e-7)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall
    }


def evaluate_with_tta(
    model: nn.Module,
    image_paths: List[Path],
    mask_paths: List[Path],
    config: Optional[TTAConfig] = None,
    device: Optional[str] = None,
    show_progress: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model with and without TTA on a dataset.
    
    Args:
        model: Trained segmentation model
        image_paths: List of image file paths
        mask_paths: List of corresponding mask file paths
        config: TTA configuration
        device: Device for inference
        show_progress: Show progress bar
    
    Returns:
        Dictionary with metrics for 'single' and 'tta' predictions
    """
    if config is None:
        config = TTAConfig()
    
    predictor = TTAPredictor(model, config, device)
    
    # Image-based metrics (per-image, then average)
    metrics_single = {'iou': [], 'dice': [], 'precision': [], 'recall': []}
    metrics_tta = {'iou': [], 'dice': [], 'precision': [], 'recall': []}
    
    # Data-based metrics (aggregate pixels across all images) - matches training!
    total_intersection_single = 0
    total_union_single = 0
    total_intersection_tta = 0
    total_union_tta = 0
    total_pred_single = 0
    total_pred_tta = 0
    total_mask = 0
    
    # Create preprocessing for mask to match training
    mask_preprocess = A.Compose([
        A.LongestMaxSize(max_size=config.image_size),
        A.PadIfNeeded(
            min_height=config.image_size,
            min_width=config.image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0
        ),
    ])
    
    iterator = zip(image_paths, mask_paths)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Evaluating")
    
    for img_path, mask_path in iterator:
        # Load image and mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Preprocess to 512x512
        processed = mask_preprocess(image=image, mask=mask)
        mask_processed = processed['mask']
        
        # Binarize mask (handles both 0/1 and 0/255)
        mask_binary = (mask_processed > 127).astype(np.uint8)
        
        # Predict without TTA - get 512x512 output
        pred_single = predictor.predict_at_model_size(image)
        
        # Predict with TTA - get 512x512 output  
        pred_tta = predictor.predict_at_model_size_tta(image)
        
        # Image-based metrics
        m_single = calculate_metrics(pred_single, mask_binary)
        m_tta = calculate_metrics(pred_tta, mask_binary)
        for k, v in m_single.items():
            metrics_single[k].append(v)
        for k, v in m_tta.items():
            metrics_tta[k].append(v)
        
        # Data-based aggregation (matches training)
        pred_single_bool = pred_single.astype(bool)
        pred_tta_bool = pred_tta.astype(bool)
        mask_bool = mask_binary.astype(bool)
        
        total_intersection_single += np.logical_and(pred_single_bool, mask_bool).sum()
        total_union_single += np.logical_or(pred_single_bool, mask_bool).sum()
        total_intersection_tta += np.logical_and(pred_tta_bool, mask_bool).sum()
        total_union_tta += np.logical_or(pred_tta_bool, mask_bool).sum()
        total_pred_single += pred_single_bool.sum()
        total_pred_tta += pred_tta_bool.sum()
        total_mask += mask_bool.sum()
    
    # Compute data-based metrics (THIS MATCHES TRAINING!)
    data_iou_single = total_intersection_single / (total_union_single + 1e-7)
    data_dice_single = (2 * total_intersection_single) / (total_pred_single + total_mask + 1e-7)
    data_iou_tta = total_intersection_tta / (total_union_tta + 1e-7)
    data_dice_tta = (2 * total_intersection_tta) / (total_pred_tta + total_mask + 1e-7)
    
    # Compute image-based mean metrics
    results = {
        'single': {k: np.mean(v) for k, v in metrics_single.items()},
        'tta': {k: np.mean(v) for k, v in metrics_tta.items()},
        'single_data_based': {'iou': data_iou_single, 'dice': data_dice_single},
        'tta_data_based': {'iou': data_iou_tta, 'dice': data_dice_tta},
        'improvement': {},
        'improvement_data_based': {}
    }
    
    # Calculate improvements (image-based)
    for metric in ['iou', 'dice', 'precision', 'recall']:
        diff = results['tta'][metric] - results['single'][metric]
        pct = (diff / results['single'][metric]) * 100 if results['single'][metric] > 0 else 0
        results['improvement'][metric] = {
            'absolute': diff,
            'relative_pct': pct
        }
    
    # Calculate improvements (data-based)
    for metric in ['iou', 'dice']:
        diff = results['tta_data_based'][metric] - results['single_data_based'][metric]
        pct = (diff / results['single_data_based'][metric]) * 100 if results['single_data_based'][metric] > 0 else 0
        results['improvement_data_based'][metric] = {
            'absolute': diff,
            'relative_pct': pct
        }
    
    return results


def load_model_for_tta(
    checkpoint_path: str,
    architecture: str = 'unetplusplus',
    encoder: str = 'efficientnet-b4',
    device: Optional[str] = None
) -> nn.Module:
    """
    Load trained model from checkpoint for TTA inference.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        architecture: Model architecture name
        encoder: Encoder backbone name
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    import segmentation_models_pytorch as smp
    
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Create model architecture
    arch_map = {
        'unet': smp.Unet,
        'unetplusplus': smp.UnetPlusPlus,
        'deeplabv3': smp.DeepLabV3,
        'deeplabv3plus': smp.DeepLabV3Plus,
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
        'manet': smp.MAnet,
    }
    
    arch_class = arch_map.get(architecture.lower())
    if arch_class is None:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Available: {list(arch_map.keys())}")
    
    model = arch_class(
        encoder_name=encoder,
        encoder_weights=None,  # Will load from checkpoint
        in_channels=3,
        classes=1,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats - PREFER EMA WEIGHTS
    print(f"   Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        print("   ✅ Using EMA weights (better performance)")
    elif 'ema' in checkpoint:
        # Some formats store EMA differently
        state_dict = checkpoint['ema']
        print("   ✅ Using EMA weights (better performance)")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("   ⚠️  Using model_state_dict (EMA not found)")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("   ⚠️  Using state_dict (EMA not found)")
    else:
        state_dict = checkpoint
        print("   ⚠️  Using raw checkpoint (no nested dict)")
    
    # Remove 'module.' prefix if present (from DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Remove 'model.' prefix if present (from custom wrapper classes)
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        print("Removed 'model.' prefix from state dict keys")
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    return model


def print_tta_results(results: Dict) -> None:
    """Pretty print TTA evaluation results."""
    print("\n" + "=" * 70)
    print("TEST TIME AUGMENTATION RESULTS")
    print("=" * 70)
    
    # Data-based metrics (matches training!)
    print("\n📊 DATA-BASED METRICS (matches training evaluation):")
    print("-" * 50)
    print("Without TTA:")
    print(f"   IoU:  {results['single_data_based']['iou']:.4f} ({results['single_data_based']['iou']*100:.2f}%)")
    print(f"   Dice: {results['single_data_based']['dice']:.4f} ({results['single_data_based']['dice']*100:.2f}%)")
    print("With TTA:")
    print(f"   IoU:  {results['tta_data_based']['iou']:.4f} ({results['tta_data_based']['iou']*100:.2f}%)")
    print(f"   Dice: {results['tta_data_based']['dice']:.4f} ({results['tta_data_based']['dice']*100:.2f}%)")
    print("Improvement:")
    for metric in ['iou', 'dice']:
        abs_imp = results['improvement_data_based'][metric]['absolute']
        rel_imp = results['improvement_data_based'][metric]['relative_pct']
        sign = '+' if abs_imp >= 0 else ''
        print(f"   {metric.upper():5s}: {sign}{abs_imp:.4f} ({sign}{rel_imp:.2f}%)")
    
    # Image-based metrics
    print("\n📈 IMAGE-BASED METRICS (per-image average):")
    print("-" * 50)
    print("Without TTA:")
    print(f"   IoU:       {results['single']['iou']:.4f} ({results['single']['iou']*100:.2f}%)")
    print(f"   Dice:      {results['single']['dice']:.4f} ({results['single']['dice']*100:.2f}%)")
    print(f"   Precision: {results['single']['precision']:.4f}")
    print(f"   Recall:    {results['single']['recall']:.4f}")
    print("With TTA:")
    print(f"   IoU:       {results['tta']['iou']:.4f} ({results['tta']['iou']*100:.2f}%)")
    print(f"   Dice:      {results['tta']['dice']:.4f} ({results['tta']['dice']*100:.2f}%)")
    print(f"   Precision: {results['tta']['precision']:.4f}")
    print(f"   Recall:    {results['tta']['recall']:.4f}")
    
    print("=" * 70 + "\n")


# =============================================================================
# MAIN: Run TTA evaluation on DiaFootAI model
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="TTA Inference for DiaFootAI")
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default='outputs/fuseg_advanced/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge',
        help='Path to FUSeg dataset'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        choices=['train', 'validation', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--architecture', '-a',
        type=str,
        default='unetplusplus',
        help='Model architecture'
    )
    parser.add_argument(
        '--encoder', '-e',
        type=str,
        default='efficientnet-b4',
        help='Encoder backbone'
    )
    parser.add_argument(
        '--merge-strategy', '-m',
        type=str,
        default='mean',
        choices=['mean', 'max', 'weighted', 'geometric'],
        help='TTA merge strategy'
    )
    parser.add_argument(
        '--image-size', '-s',
        type=int,
        default=512,
        help='Image size for inference'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Threshold for binary prediction'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    image_dir = data_dir / args.split / 'images'
    mask_dir = data_dir / args.split / 'labels'
    
    # Check if paths exist
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        print("Trying alternative path structure...")
        # Try alternative structure
        if args.split == 'validation':
            image_dir = data_dir / 'validation' / 'images'
            mask_dir = data_dir / 'validation' / 'labels'
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Cannot find image directory: {image_dir}")
    
    print(f"\n📁 Data directory: {data_dir}")
    print(f"📁 Image directory: {image_dir}")
    print(f"📁 Mask directory: {mask_dir}")
    
    # Get image/mask pairs
    image_paths = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    print(f"📊 Found {len(image_paths)} images")
    
    # Map images to masks
    mask_paths = []
    for img_path in image_paths:
        # Try different mask naming conventions
        mask_name = img_path.stem + '.png'
        mask_path = mask_dir / mask_name
        if not mask_path.exists():
            mask_name = img_path.stem + '_mask.png'
            mask_path = mask_dir / mask_name
        if not mask_path.exists():
            mask_name = img_path.name  # Same name
            mask_path = mask_dir / mask_name
        mask_paths.append(mask_path)
    
    # Check first few masks exist
    missing_masks = [p for p in mask_paths[:5] if not p.exists()]
    if missing_masks:
        print(f"⚠️  Warning: Some masks not found: {missing_masks[:3]}")
    
    # Load model
    print(f"\n🔄 Loading model from: {args.checkpoint}")
    model = load_model_for_tta(
        checkpoint_path=args.checkpoint,
        architecture=args.architecture,
        encoder=args.encoder
    )
    print("✅ Model loaded successfully")
    
    # Configure TTA
    tta_config = TTAConfig(
        image_size=args.image_size,
        merge_strategy=args.merge_strategy,
        threshold=args.threshold,
        use_hflip=True,
        use_vflip=True,
        use_rotate90=True,
        use_rotate180=True,
        use_rotate270=True,
        use_scale_up=True,
        use_scale_down=True,
        use_multiscale=False,  # Keep at 8 transforms for speed
    )
    
    print(f"\n⚙️  TTA Configuration:")
    print(f"   Transforms: {tta_config.get_num_transforms()}")
    print(f"   Merge strategy: {tta_config.merge_strategy}")
    print(f"   Threshold: {tta_config.threshold}")
    
    # Run evaluation
    print("\n🔬 Running TTA evaluation...")
    results = evaluate_with_tta(
        model=model,
        image_paths=image_paths,
        mask_paths=mask_paths,
        config=tta_config,
        show_progress=True
    )
    
    # Print results
    print_tta_results(results)
    
    # Save results
    import json
    results_serializable = {
        'single_data_based': {k: float(v) for k, v in results['single_data_based'].items()},
        'tta_data_based': {k: float(v) for k, v in results['tta_data_based'].items()},
        'single_image_based': results['single'],
        'tta_image_based': results['tta'],
        'improvement_data_based': {
            k: {'absolute': float(v['absolute']), 'relative_pct': float(v['relative_pct'])}
            for k, v in results['improvement_data_based'].items()
        },
        'improvement_image_based': {
            k: {'absolute': float(v['absolute']), 'relative_pct': float(v['relative_pct'])}
            for k, v in results['improvement'].items()
        },
        'config': {
            'checkpoint': args.checkpoint,
            'architecture': args.architecture,
            'encoder': args.encoder,
            'merge_strategy': args.merge_strategy,
            'num_transforms': tta_config.get_num_transforms(),
            'num_images': len(image_paths),
        }
    }
    
    output_path = Path('outputs/tta_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"💾 Results saved to: {output_path}")
