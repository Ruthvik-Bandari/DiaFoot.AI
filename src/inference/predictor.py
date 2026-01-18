"""
Inference Predictor Module
===========================

Production inference for wound segmentation models.

Author: Ruthvik
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import cv2


class WoundPredictor:
    """
    Production predictor for wound segmentation.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path: Union[str, Path]) -> nn.Module:
        """Load model from checkpoint."""
        from src.models.segmentation import SegmentationModel
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint.get("config", {})
        seg_config = config.get("models", {}).get("segmentation", {})
        
        model = SegmentationModel(
            architecture=seg_config.get("architecture", "unetplusplus"),
            encoder_name=seg_config.get("encoder", "efficientnet-b4"),
            num_classes=seg_config.get("num_classes", 1),
        )
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def preprocess(self, image: np.ndarray, size: int = 512) -> torch.Tensor:
        """Preprocess image for inference."""
        # Resize
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        y_off = (size - new_h) // 2
        x_off = (size - new_w) // 2
        padded[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        # Normalize
        tensor = padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        tensor = (tensor - mean) / std
        
        # To tensor
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict wound segmentation.
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            Dictionary with mask, probability map, and metadata
        """
        original_shape = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        output = self.model(input_tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Resize back to original
        prob_map = cv2.resize(prob_map, (original_shape[1], original_shape[0]))
        
        # Binary mask
        binary_mask = (prob_map > self.threshold).astype(np.uint8)
        
        # Compute wound area
        wound_pixels = binary_mask.sum()
        total_pixels = binary_mask.size
        wound_percentage = wound_pixels / total_pixels * 100
        
        return {
            "mask": binary_mask,
            "probability_map": prob_map,
            "wound_percentage": wound_percentage,
            "wound_pixels": int(wound_pixels),
            "confidence": float(prob_map[binary_mask > 0].mean()) if wound_pixels > 0 else 0.0,
        }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Predict on multiple images."""
        return [self.predict(img) for img in images]


def load_predictor(model_path: str, device: Optional[str] = None) -> WoundPredictor:
    """Convenience function to load predictor."""
    dev = torch.device(device) if device else None
    return WoundPredictor(model_path, device=dev)
