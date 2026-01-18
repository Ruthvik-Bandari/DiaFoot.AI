#!/usr/bin/env python3
"""
Enhanced Inference Pipeline for DiaFootAI
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


class CLAHEPreprocessor:
    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_size = tile_size
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_size, self.tile_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class PostProcessor:
    def __init__(self, min_size: int = 100, fill_holes: bool = True):
        self.min_size = min_size
        self.fill_holes = fill_holes
    
    def apply(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.uint8)
        mask = self._remove_small_regions(mask)
        if self.fill_holes:
            mask = self._fill_holes(mask)
        mask = self._smooth_boundaries(mask)
        return mask
    
    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_size:
                cleaned[labels == i] = 1
        return cleaned
    
    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 1, -1)
        return filled
    
    def _smooth_boundaries(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        return smoothed


class TestTimeAugmentation:
    def __init__(self, use_flip: bool = True, use_rotate: bool = True):
        self.use_flip = use_flip
        self.use_rotate = use_rotate
    
    def augment(self, image: np.ndarray) -> list:
        augmented = [image]
        if self.use_flip:
            augmented.append(np.fliplr(image).copy())
            augmented.append(np.flipud(image).copy())
        if self.use_rotate:
            augmented.append(np.rot90(image, k=1).copy())
            augmented.append(np.rot90(image, k=3).copy())
        return augmented
    
    def deaugment(self, predictions: list) -> np.ndarray:
        deaugmented = [predictions[0]]
        idx = 1
        if self.use_flip:
            deaugmented.append(np.fliplr(predictions[idx]))
            idx += 1
            deaugmented.append(np.flipud(predictions[idx]))
            idx += 1
        if self.use_rotate:
            deaugmented.append(np.rot90(predictions[idx], k=-1))
            idx += 1
            deaugmented.append(np.rot90(predictions[idx], k=-3))
            idx += 1
        return np.mean(deaugmented, axis=0)


class EnhancedWoundSegmentationPipeline:
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        use_clahe: bool = True,
        use_tta: bool = True,
        use_postprocessing: bool = True,
        threshold: float = 0.5,
        image_size: int = 512,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        
        self.model = self._load_model(model_path)
        self.use_clahe = use_clahe
        self.use_tta = use_tta
        self.use_postprocessing = use_postprocessing
        self.threshold = threshold
        self.image_size = image_size
        
        if use_clahe:
            self.clahe = CLAHEPreprocessor(clip_limit=2.0, tile_size=8)
        if use_tta:
            self.tta = TestTimeAugmentation(use_flip=True, use_rotate=True)
        if use_postprocessing:
            self.postprocessor = PostProcessor(min_size=100, fill_holes=True)
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        print(f"✅ Enhanced Pipeline Initialized")
        print(f"   Device: {self.device}")
        print(f"   CLAHE: {'✓' if use_clahe else '✗'}")
        print(f"   TTA: {'✓' if use_tta else '✗'}")
        print(f"   Post-processing: {'✓' if use_postprocessing else '✗'}")
    
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
        return model
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(image, (self.image_size, self.image_size))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)
    
    @torch.no_grad()
    def _predict_single(self, image: np.ndarray) -> np.ndarray:
        input_tensor = self._preprocess(image)
        output = self.model(input_tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        return prob_map
    
    @torch.no_grad()
    def _predict_with_tta(self, image: np.ndarray) -> np.ndarray:
        augmented_images = self.tta.augment(image)
        predictions = []
        for aug_img in augmented_images:
            pred = self._predict_single(aug_img)
            predictions.append(pred)
        return self.tta.deaugment(predictions)
    
    def predict(self, image: np.ndarray) -> Dict:
        original_size = (image.shape[1], image.shape[0])
        
        if self.use_clahe:
            processed = self.clahe.apply(image)
        else:
            processed = image
        
        if self.use_tta:
            prob_map = self._predict_with_tta(processed)
        else:
            prob_map = self._predict_single(processed)
        
        prob_map_full = cv2.resize(prob_map, original_size)
        mask = (prob_map_full > self.threshold).astype(np.uint8)
        
        if self.use_postprocessing:
            mask = self.postprocessor.apply(mask)
        
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
        mask = result["mask"]
        prob_map = result["probability_map"]
        h, w = image.shape[:2]
        
        overlay = image.copy()
        overlay[mask > 0] = [255, 0, 0]
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)
        
        heatmap = (prob_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
        canvas[:, :w] = image
        canvas[:, w:w*2] = blended
        canvas[:, w*2:] = heatmap
        
        return canvas


def create_pipeline(
    model_path: str = "outputs/fuseg_simple/best_model.pt",
    use_clahe: bool = True,
    use_tta: bool = True,
    use_postprocessing: bool = True,
) -> EnhancedWoundSegmentationPipeline:
    return EnhancedWoundSegmentationPipeline(
        model_path=model_path,
        use_clahe=use_clahe,
        use_tta=use_tta,
        use_postprocessing=use_postprocessing,
    )
