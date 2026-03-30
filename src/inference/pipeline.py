"""DiaFoot.AI v2 — Full Multi-Task Inference Pipeline.

Pipeline: Image -> Preprocess -> Classify (DINOv2) -> (if DFU) Segment (DINOv2) -> Post-process -> Results
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
    diagnostics: dict[str, object] = field(default_factory=dict)


CLASS_NAMES = {0: "Healthy", 1: "Non-DFU", 2: "DFU"}


class InferencePipeline:
    """Multi-task inference pipeline.

    Runs triage classification first. If DFU detected, runs segmentation.
    Supports early exit for healthy feet (skip segmentation).

    Args:
        classifier: Triage classifier model (DINOv2Classifier or TriageClassifier).
        segmenter: Segmentation model (DINOv2Segmenter or U-Net++, optional).
        device: Computation device.
        confidence_threshold: Min confidence to trust classification.
        seg_threshold: Threshold for binarizing segmentation output.
        pixel_spacing_mm: Physical pixel size for area calculation.
        input_size: Model input resolution (518 for DINOv2, 512 for legacy).
    """

    def __init__(
        self,
        classifier: nn.Module,
        segmenter: nn.Module | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.95,
        defer_threshold: float = 0.60,
        dfu_seg_fallback_prob: float = 0.10,
        dfu_promotion_threshold: float = 0.04,
        seg_threshold: float = 0.5,
        pixel_spacing_mm: float = 0.5,
        input_size: int = 518,
    ) -> None:
        """Initialize inference pipeline."""
        self.device = torch.device(device)
        self.classifier = classifier.to(self.device).eval()
        self.segmenter = segmenter
        if self.segmenter:
            self.segmenter = self.segmenter.to(self.device).eval()
        self.confidence_threshold = confidence_threshold
        self.defer_threshold = defer_threshold
        self.dfu_seg_fallback_prob = dfu_seg_fallback_prob
        self.dfu_promotion_threshold = dfu_promotion_threshold
        self.seg_threshold = seg_threshold
        self.pixel_spacing_mm = pixel_spacing_mm
        self.input_size = input_size

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference.

        Args:
            image: BGR or RGB image (H, W, 3).

        Returns:
            Normalized tensor (1, 3, input_size, input_size).
        """
        if image.shape[2] == 3 and len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img = image

        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        # ImageNet normalization (same as DINOv2 pretraining)
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
        diag: dict[str, object] = {
            "segmentation_ran": False,
            "segmentation_skip_reason": None,
            "seg_threshold_used": self.seg_threshold,
            "confidence_threshold_used": self.confidence_threshold,
            "defer_threshold_used": self.defer_threshold,
        }
        result.diagnostics = diag
        tensor = self.preprocess(image).to(self.device)

        # Step 1: Classification
        cls_logits = self.classifier(tensor)
        cls_probs = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()

        pred_class = int(cls_probs.argmax())
        confidence = float(cls_probs.max())
        dfu_prob = float(cls_probs[2]) if len(cls_probs) > 2 else 0.0

        result.classification = CLASS_NAMES.get(pred_class, "Unknown")
        result.classification_confidence = confidence
        result.classification_probs = {
            CLASS_NAMES[i]: float(cls_probs[i]) for i in range(len(cls_probs))
        }

        diag["classifier_pred_class"] = pred_class
        diag["classifier_confidence"] = confidence
        diag["classifier_dfu_prob"] = dfu_prob
        logger.info(
            "Classification: class=%d (%s), confidence=%.4f, dfu_prob=%.4f",
            pred_class, result.classification, confidence, dfu_prob,
        )

        # Defer path: flag low-confidence predictions for manual review,
        # but continue to segmentation so wound metrics are still computed.
        if confidence < self.defer_threshold:
            result.defer_to_clinician = True
            result.defer_reason = "low_classification_confidence"
            logger.info(
                "Defer flagged: confidence %.4f < defer_threshold %.4f — continuing to segmentation",
                confidence, self.defer_threshold,
            )

        # Early exit: if Healthy with high confidence, skip segmentation
        if pred_class == 0 and confidence >= self.confidence_threshold:
            result.has_wound = False
            diag["segmentation_skip_reason"] = "healthy_early_exit"
            logger.info(
                "Healthy early exit: confidence %.4f >= threshold %.4f — skipping segmentation",
                confidence, self.confidence_threshold,
            )
            return result

        # Step 2: Segmentation
        # Run segmentation for DFU predictions, and also as fallback when triage
        # is uncertain or DFU probability is non-trivial.
        cond_dfu = pred_class == 2
        cond_nondfu_low = pred_class == 1 and confidence < self.confidence_threshold
        cond_fallback = dfu_prob >= self.dfu_seg_fallback_prob
        should_run_seg = cond_dfu or cond_nondfu_low or cond_fallback

        logger.info(
            "Segmentation decision: should_run_seg=%s (is_dfu=%s, nondfu_low_conf=%s, dfu_fallback=%s)",
            should_run_seg, cond_dfu, cond_nondfu_low, cond_fallback,
        )

        if should_run_seg and self.segmenter is None:
            result.defer_to_clinician = True
            result.defer_reason = "segmenter_unavailable"
            diag["segmentation_skip_reason"] = "segmenter_unavailable"
            logger.warning("Segmenter unavailable — cannot run segmentation")
            return result

        if not should_run_seg:
            diag["segmentation_skip_reason"] = "should_run_seg_false"
            logger.info("Segmentation skipped: none of the trigger conditions met")

        if self.segmenter and should_run_seg:
            diag["segmentation_ran"] = True
            diag["segmentation_skip_reason"] = None
            seg_logits = self.segmenter(tensor)
            if isinstance(seg_logits, dict):
                seg_logits = seg_logits.get("seg_logits", seg_logits)

            seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            seg_mask = (seg_prob > self.seg_threshold).astype(np.uint8)

            diag["seg_prob_min"] = float(seg_prob.min())
            diag["seg_prob_max"] = float(seg_prob.max())
            diag["seg_prob_mean"] = float(seg_prob.mean())
            diag["pixels_above_threshold"] = int(seg_mask.sum())
            diag["total_pixels"] = int(seg_prob.shape[0] * seg_prob.shape[1])
            logger.info(
                "Segmentation stats: prob min=%.4f, max=%.4f, mean=%.4f, "
                "pixels_above_threshold=%d / %d (threshold=%.2f)",
                seg_prob.min(), seg_prob.max(), seg_prob.mean(),
                seg_mask.sum(), seg_prob.size, self.seg_threshold,
            )

            result.has_wound = bool(seg_mask.any())
            result.segmentation_mask = seg_mask
            result.wound_area_px = int(seg_mask.sum())
            result.wound_area_mm2 = float(
                result.wound_area_px * self.pixel_spacing_mm * self.pixel_spacing_mm
            )
            total_pixels = seg_mask.shape[0] * seg_mask.shape[1]
            result.wound_coverage_pct = result.wound_area_px / total_pixels * 100

            logger.info(
                "Wound metrics: has_wound=%s, area_px=%d, area_mm2=%.2f, coverage=%.2f%%",
                result.has_wound, result.wound_area_px, result.wound_area_mm2, result.wound_coverage_pct,
            )

            # If segmentation finds wound but classifier says Non-DFU,
            # escalate to manual review instead of silently returning no wound.
            if result.has_wound and pred_class == 1:
                if dfu_prob >= self.dfu_promotion_threshold:
                    result.classification = "DFU"
                    # Keep confidence consistent with displayed class.
                    result.classification_confidence = dfu_prob
                result.defer_to_clinician = True
                result.defer_reason = "segmentation_classifier_disagreement"
                logger.info("Classifier/segmenter disagreement: Non-DFU + wound detected")

        if pred_class == 1 and not result.has_wound:
            # If no disagreement was found, keep Non-DFU with no wound.
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
