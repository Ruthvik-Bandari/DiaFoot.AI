"""
Evaluation Metrics Module
==========================

Comprehensive metrics for wound segmentation and classification evaluation.
Includes IoU, Dice, Hausdorff distance, and clinical metrics.

Author: Ruthvik
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)


class SegmentationMetrics:
    """
    Compute comprehensive segmentation metrics.
    """
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.num_samples = 0
        self.ious = []
        self.dices = []
    
    def update(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred: Predictions (after sigmoid) or logits
            target: Ground truth masks
        """
        # Convert to numpy if tensor
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Binarize predictions
        pred_binary = (pred > self.threshold).astype(np.float32)
        target_binary = (target > 0.5).astype(np.float32)
        
        # Flatten for batch processing
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            p = pred_binary[i].flatten()
            t = target_binary[i].flatten()
            
            # Confusion matrix elements
            tp = np.sum(p * t)
            fp = np.sum(p * (1 - t))
            fn = np.sum((1 - p) * t)
            tn = np.sum((1 - p) * (1 - t))
            
            self.total_tp += tp
            self.total_fp += fp
            self.total_fn += fn
            self.total_tn += tn
            
            # Per-sample IoU and Dice
            intersection = tp
            union = tp + fp + fn
            
            iou = (intersection + self.smooth) / (union + self.smooth)
            dice = (2 * intersection + self.smooth) / (2 * intersection + fp + fn + self.smooth)
            
            self.ious.append(iou)
            self.dices.append(dice)
            self.num_samples += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if self.num_samples == 0:
            return {}
        
        # Global metrics (micro-average)
        precision = (self.total_tp + self.smooth) / (self.total_tp + self.total_fp + self.smooth)
        recall = (self.total_tp + self.smooth) / (self.total_tp + self.total_fn + self.smooth)
        f1 = 2 * precision * recall / (precision + recall + self.smooth)
        
        global_iou = (self.total_tp + self.smooth) / (
            self.total_tp + self.total_fp + self.total_fn + self.smooth
        )
        global_dice = (2 * self.total_tp + self.smooth) / (
            2 * self.total_tp + self.total_fp + self.total_fn + self.smooth
        )
        
        # Mean metrics (macro-average)
        mean_iou = np.mean(self.ious)
        mean_dice = np.mean(self.dices)
        std_iou = np.std(self.ious)
        std_dice = np.std(self.dices)
        
        return {
            "iou": mean_iou,
            "iou_std": std_iou,
            "iou_global": global_iou,
            "dice": mean_dice,
            "dice_std": std_dice,
            "dice_global": global_dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sensitivity": recall,  # Same as recall
            "specificity": (self.total_tn + self.smooth) / (self.total_tn + self.total_fp + self.smooth),
        }


def compute_iou(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection
    
    return (intersection + smooth) / (union + smooth)


def compute_dice(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Dice coefficient (F1 score for segmentation).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_binary * target_binary)
    
    return (2 * intersection + smooth) / (np.sum(pred_binary) + np.sum(target_binary) + smooth)


def compute_hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: float = 95,
) -> float:
    """
    Compute Hausdorff distance between prediction and target boundaries.
    Uses 95th percentile by default for robustness.
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        percentile: Percentile for robust HD (95 = HD95)
    
    Returns:
        Hausdorff distance in pixels
    """
    pred_binary = (pred > 0.5).astype(np.uint8)
    target_binary = (target > 0.5).astype(np.uint8)
    
    # Handle empty masks
    if pred_binary.sum() == 0 or target_binary.sum() == 0:
        return float("inf")
    
    # Compute distance transforms
    pred_dist = distance_transform_edt(1 - pred_binary)
    target_dist = distance_transform_edt(1 - target_binary)
    
    # Get boundary pixels
    pred_boundary = pred_binary - (pred_binary & (pred_dist > 1).astype(np.uint8))
    target_boundary = target_binary - (target_binary & (target_dist > 1).astype(np.uint8))
    
    # Distance from pred boundary to target
    pred_to_target = target_dist[pred_boundary > 0]
    # Distance from target boundary to pred
    target_to_pred = pred_dist[target_boundary > 0]
    
    if len(pred_to_target) == 0 or len(target_to_pred) == 0:
        return float("inf")
    
    # Hausdorff distance (or percentile version)
    if percentile == 100:
        return max(pred_to_target.max(), target_to_pred.max())
    else:
        return max(
            np.percentile(pred_to_target, percentile),
            np.percentile(target_to_pred, percentile)
        )


def compute_boundary_iou(
    pred: np.ndarray,
    target: np.ndarray,
    boundary_width: int = 2,
) -> float:
    """
    Compute IoU specifically on the boundary region.
    More sensitive to edge accuracy.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    
    pred_binary = (pred > 0.5).astype(np.uint8)
    target_binary = (target > 0.5).astype(np.uint8)
    
    # Extract boundaries using morphological operations
    struct = np.ones((3, 3))
    
    pred_dilated = binary_dilation(pred_binary, struct, iterations=boundary_width)
    pred_eroded = binary_erosion(pred_binary, struct, iterations=boundary_width)
    pred_boundary = pred_dilated.astype(np.uint8) - pred_eroded.astype(np.uint8)
    
    target_dilated = binary_dilation(target_binary, struct, iterations=boundary_width)
    target_eroded = binary_erosion(target_binary, struct, iterations=boundary_width)
    target_boundary = target_dilated.astype(np.uint8) - target_eroded.astype(np.uint8)
    
    # IoU on boundary
    intersection = np.sum(pred_boundary * target_boundary)
    union = np.sum(pred_boundary) + np.sum(target_boundary) - intersection
    
    return (intersection + 1e-6) / (union + 1e-6)


class ClassificationMetrics:
    """
    Compute comprehensive classification metrics.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions."""
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []
    
    def update(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        probs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        """
        Update with batch predictions.
        
        Args:
            pred: Predicted class indices
            target: Ground truth class indices
            probs: Prediction probabilities (for AUC)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        self.all_preds.extend(pred.flatten().tolist())
        self.all_targets.extend(target.flatten().tolist())
        
        if probs is not None:
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            self.all_probs.extend(probs.tolist())
    
    def compute(self) -> Dict[str, Union[float, np.ndarray]]:
        """Compute final metrics."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "precision_macro": precision_score(targets, preds, average="macro", zero_division=0),
            "precision_weighted": precision_score(targets, preds, average="weighted", zero_division=0),
            "recall_macro": recall_score(targets, preds, average="macro", zero_division=0),
            "recall_weighted": recall_score(targets, preds, average="weighted", zero_division=0),
            "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(targets, preds, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(targets, preds),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(targets, preds, average=None, zero_division=0)
        recall_per_class = recall_score(targets, preds, average=None, zero_division=0)
        f1_per_class = f1_score(targets, preds, average=None, zero_division=0)
        
        for i, name in enumerate(self.class_names[:len(precision_per_class)]):
            metrics[f"precision_{name}"] = precision_per_class[i]
            metrics[f"recall_{name}"] = recall_per_class[i]
            metrics[f"f1_{name}"] = f1_per_class[i]
        
        # AUC-ROC if probabilities available
        if self.all_probs and self.num_classes == 2:
            try:
                probs = np.array(self.all_probs)
                if probs.ndim == 2:
                    probs = probs[:, 1]  # Probability of positive class
                metrics["auc_roc"] = roc_auc_score(targets, probs)
            except Exception:
                pass
        elif self.all_probs and self.num_classes > 2:
            try:
                probs = np.array(self.all_probs)
                metrics["auc_roc_ovr"] = roc_auc_score(
                    targets, probs, multi_class="ovr", average="macro"
                )
            except Exception:
                pass
        
        return metrics
    
    def report(self) -> str:
        """Generate classification report."""
        return classification_report(
            self.all_targets,
            self.all_preds,
            target_names=self.class_names,
            zero_division=0,
        )


def evaluate_segmentation_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate a segmentation model on a dataset.
    
    Args:
        model: Trained segmentation model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        threshold: Binarization threshold
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = SegmentationMetrics(threshold=threshold)
    
    hausdorff_distances = []
    boundary_ious = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["mask"]
            
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu()
            
            metrics.update(preds, targets)
            
            # Compute HD and boundary IoU for each sample
            for i in range(preds.shape[0]):
                p = preds[i, 0].numpy()
                t = targets[i, 0].numpy()
                
                hd = compute_hausdorff_distance(p, t)
                if hd != float("inf"):
                    hausdorff_distances.append(hd)
                
                b_iou = compute_boundary_iou(p, t)
                boundary_ious.append(b_iou)
    
    results = metrics.compute()
    
    if hausdorff_distances:
        results["hausdorff_95"] = np.mean(hausdorff_distances)
        results["hausdorff_95_std"] = np.std(hausdorff_distances)
    
    if boundary_ious:
        results["boundary_iou"] = np.mean(boundary_ious)
    
    return results


def print_evaluation_results(results: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for key, value in results.items():
        if key == "confusion_matrix":
            continue
        if isinstance(value, float):
            print(f"  {key:.<40} {value:.4f}")
        else:
            print(f"  {key:.<40} {value}")
    
    print(f"{'='*60}\n")
