#!/usr/bin/env python3
"""
Optimized Production Pipeline for DiaFootAI
============================================

Best configuration found through testing:
- CLAHE preprocessing: ON
- TTA: Flips only (horizontal + vertical)
- Post-processing: Min region size = 50

Performance:
- Base model IoU: 0.8493 (validation during training)
- With optimizations: +2.5% improvement on test images

Author: Ruthvik
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.segmentation import SegmentationModel


class OptimizedWoundPipeline:
    """
    Production-ready wound segmentation pipeline.
    
    Usage:
        pipeline = OptimizedWoundPipeline("outputs/fuseg_simple/best_model.pt")
        result = pipeline.predict(image)
        mask = result["mask"]
        visualization = pipeline.visualize(image, result)
    """
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        # Device setup
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Settings (optimized through testing)
        self.image_size = 512
        self.threshold = 0.5
        self.min_region_size = 50  # Optimized: less aggressive
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        print(f"✅ Optimized Pipeline Ready")
        print(f"   Device: {self.device}")
        print(f"   Model IoU: {self.model_info['val_iou']:.4f}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        model = SegmentationModel(
            architecture="unetplusplus",
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            num_classes=1,
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        
        self.model_info = {
            "epoch": checkpoint.get("epoch", "unknown"),
            "val_iou": checkpoint.get("val_iou", 0),
            "val_dice": checkpoint.get("val_dice", 0),
        }
        return model
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model."""
        resized = cv2.resize(image, (self.image_size, self.image_size))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)
    
    @torch.no_grad()
    def _predict_single(self, image: np.ndarray) -> np.ndarray:
        """Single prediction."""
        tensor = self._preprocess(image)
        output = self.model(tensor)
        return torch.sigmoid(output).squeeze().cpu().numpy()
    
    @torch.no_grad()
    def _predict_with_tta(self, image: np.ndarray) -> np.ndarray:
        """Prediction with TTA (flips only - optimized)."""
        predictions = []
        
        # Original
        predictions.append(self._predict_single(image))
        
        # Horizontal flip
        flipped_h = np.fliplr(image).copy()
        pred_h = self._predict_single(flipped_h)
        predictions.append(np.fliplr(pred_h))
        
        # Vertical flip
        flipped_v = np.flipud(image).copy()
        pred_v = self._predict_single(flipped_v)
        predictions.append(np.flipud(pred_v))
        
        # Average
        return np.mean(predictions, axis=0)
    
    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Clean up prediction mask."""
        mask = mask.astype(np.uint8)
        
        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_region_size:
                cleaned[labels == i] = 1
        
        # Fill holes
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(cleaned)
        cv2.drawContours(filled, contours, -1, 1, -1)
        
        # Smooth boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smoothed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        
        return smoothed
    
    def predict(self, image: np.ndarray, use_tta: bool = True) -> Dict:
        """
        Run full prediction pipeline.
        
        Args:
            image: RGB image (H, W, 3), uint8
            use_tta: Whether to use Test Time Augmentation
        
        Returns:
            Dictionary with mask, probability_map, and statistics
        """
        original_size = (image.shape[1], image.shape[0])
        
        # 1. CLAHE preprocessing
        enhanced = self._apply_clahe(image)
        
        # 2. Model prediction
        if use_tta:
            prob_map = self._predict_with_tta(enhanced)
        else:
            prob_map = self._predict_single(enhanced)
        
        # 3. Resize to original
        prob_map_full = cv2.resize(prob_map, original_size)
        
        # 4. Threshold
        mask = (prob_map_full > self.threshold).astype(np.uint8)
        
        # 5. Post-processing
        mask = self._postprocess(mask)
        
        # Calculate stats
        wound_pixels = mask.sum()
        total_pixels = mask.size
        
        return {
            "mask": mask,
            "probability_map": prob_map_full,
            "wound_percentage": wound_pixels / total_pixels * 100,
            "wound_pixels": int(wound_pixels),
            "confidence": float(prob_map_full[mask > 0].mean()) if wound_pixels > 0 else 0.0,
        }
    
    def visualize(self, image: np.ndarray, result: Dict, alpha: float = 0.4) -> np.ndarray:
        """Create visualization."""
        mask = result["mask"]
        prob_map = result["probability_map"]
        h, w = image.shape[:2]
        
        # Overlay
        overlay = image.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red for wound
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        # Contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)
        
        # Heatmap
        heatmap = (prob_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Combine: Original | Overlay | Heatmap
        canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
        canvas[:, :w] = image
        canvas[:, w:w*2] = blended
        canvas[:, w*2:] = heatmap
        
        return canvas
    
    def predict_and_save(self, image_path: str, output_path: str) -> Dict:
        """Predict on image file and save visualization."""
        from PIL import Image as PILImage
        
        image = np.array(PILImage.open(image_path).convert("RGB"))
        result = self.predict(image)
        vis = self.visualize(image, result)
        
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        return result


def load_pipeline(model_path: str = "outputs/fuseg_simple/best_model.pt") -> OptimizedWoundPipeline:
    """Load the optimized pipeline."""
    return OptimizedWoundPipeline(model_path)
