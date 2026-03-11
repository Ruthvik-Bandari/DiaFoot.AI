"""DiaFoot.AI v2 — Full Multi-Task Inference Pipeline.

Phase 6, Commit 30: End-to-end pipeline from image to clinical output.

Pipeline: Image -> Preprocess -> Classify -> (if DFU) Segment -> Post-process -> Results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete inference result for a single image."""

    classification: str = "Unknown"
    classification_confidence: float = 0.0
    classification_probs: dict[str, float] = field(default_factory=dict)
    defer_to_clinician: bool = False
    defer_reason: str = ""
    has_wound: bool = False
    segmentation_mask: np.ndarray | None = None
    wound_area_px: int = 0
    wound_area_mm2: float = 0.0
    wound_coverage_pct: float = 0.0
    uncertainty_map: np.ndarray | None = None
    mean_uncertainty: float = 0.0


CLASS_NAMES = {0: "Healthy", 1: "Non-DFU", 2: "DFU"}


class InferencePipeline:
    """Multi-task inference pipeline.

    Runs triage classification first. If DFU detected, runs segmentation.
    Supports early exit for healthy feet (skip segmentation).

    Args:
        classifier: Triage classifier model.
        segmenter: Segmentation model (optional).
        device: Computation device.
        confidence_threshold: Min confidence to trust classification.
        seg_threshold: Threshold for binarizing segmentation output.
        pixel_spacing_mm: Physical pixel size for area calculation.
    """

    def __init__(
        self,
        classifier: nn.Module,
        segmenter: nn.Module | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.95,
        defer_threshold: float = 0.60,
        seg_threshold: float = 0.5,
        pixel_spacing_mm: float = 0.5,
    ) -> None:
        """Initialize inference pipeline."""
        self.device = torch.device(device)
        self.classifier = classifier.to(self.device).eval()
        self.segmenter = segmenter
        if self.segmenter:
            self.segmenter = self.segmenter.to(self.device).eval()
        self.confidence_threshold = confidence_threshold
        self.defer_threshold = defer_threshold
        self.seg_threshold = seg_threshold
        self.pixel_spacing_mm = pixel_spacing_mm

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference.

        Args:
            image: BGR or RGB image (H, W, 3).

        Returns:
            Normalized tensor (1, 3, 512, 512).
        """
        if image.shape[2] == 3 and len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img = image

        img = cv2.resize(img, (512, 512))
        img = img.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return tensor

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> PipelineResult:
        """Run full inference pipeline on a single image.

        Args:
            image: Input image (H, W, 3) BGR or RGB.

        Returns:
            PipelineResult with all outputs.
        """
        result = PipelineResult()
        tensor = self.preprocess(image).to(self.device)

        # Step 1: Classification
        cls_logits = self.classifier(tensor)
        cls_probs = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()

        pred_class = int(cls_probs.argmax())
        confidence = float(cls_probs.max())

        result.classification = CLASS_NAMES.get(pred_class, "Unknown")
        result.classification_confidence = confidence
        result.classification_probs = {
            CLASS_NAMES[i]: float(cls_probs[i]) for i in range(len(cls_probs))
        }

        # Defer path: low confidence classification should be reviewed manually.
        if confidence < self.defer_threshold:
            result.defer_to_clinician = True
            result.defer_reason = "low_classification_confidence"
            return result

        # Early exit: if Healthy with high confidence, skip segmentation
        if pred_class == 0 and confidence >= self.confidence_threshold:
            result.has_wound = False
            return result

        # Step 2: Segmentation (if model available and wound detected)
        if pred_class == 2 and self.segmenter is None:
            result.defer_to_clinician = True
            result.defer_reason = "segmenter_unavailable"
            return result

        if self.segmenter and pred_class == 2:
            seg_logits = self.segmenter(tensor)
            if isinstance(seg_logits, dict):
                seg_logits = seg_logits.get("seg_logits", seg_logits)

            seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            seg_mask = (seg_prob > self.seg_threshold).astype(np.uint8)

            result.has_wound = bool(seg_mask.any())
            result.segmentation_mask = seg_mask
            result.wound_area_px = int(seg_mask.sum())
            result.wound_area_mm2 = float(
                result.wound_area_px * self.pixel_spacing_mm * self.pixel_spacing_mm
            )
            total_pixels = seg_mask.shape[0] * seg_mask.shape[1]
            result.wound_coverage_pct = result.wound_area_px / total_pixels * 100

        if pred_class == 1:
            result.has_wound = False

        return result

    def predict_batch(
        self,
        images: list[np.ndarray],
    ) -> list[PipelineResult]:
        """Run inference on multiple images.

        Args:
            images: List of input images.

        Returns:
            List of PipelineResults.
        """
        return [self.predict(img) for img in images]
