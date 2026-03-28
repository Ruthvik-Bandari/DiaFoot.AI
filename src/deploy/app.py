"""DiaFoot.AI v2 — FastAPI REST Service.

Endpoints:
    POST /predict — Upload image, get full pipeline result
    GET /health — Service health check
    GET /model/info — Model metadata
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage
from pydantic import BaseModel

from src.deploy.middleware import (
    MaxContentLengthMiddleware,
    RateLimitMiddleware,
    validate_upload_metadata,
)
from src.deploy.schemas import DriftFeatures, PredictionLogEvent
from src.inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.95
DEFAULT_DEFER_THRESHOLD = 0.60
DEFAULT_CALIBRATION_PATH = "results/classification_calibration.json"
DEFAULT_CLASSIFIER_CKPT = "checkpoints/dinov2_classifier/best_epoch009_0.9785.pt"
DEFAULT_SEGMENTER_CKPT = "checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt"
DEFAULT_DFU_SEG_FALLBACK_PROB = 0.10
DEFAULT_DFU_PROMOTION_THRESHOLD = 0.04
DEFAULT_MIN_IMAGE_SIDE = 256
DEFAULT_BLUR_VARIANCE_THRESHOLD = 30.0
DEFAULT_BRIGHTNESS_MIN = 20.0
DEFAULT_BRIGHTNESS_MAX = 235.0
DEFAULT_BACKBONE = "dinov2_vitb14"


def _getenv(*names: str, default: str | None = None) -> str | None:
    """Return first non-empty environment value among aliases."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


def _load_state_dict_from_checkpoint(path: Path) -> dict[str, Any]:
    """Load PyTorch checkpoint and return a state-dict payload."""
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
        if isinstance(state, dict):
            return state
    if isinstance(payload, dict):
        return payload
    msg = f"Unsupported checkpoint format: {path}"
    raise RuntimeError(msg)


def _clamp_probability(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _parse_probability(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return _clamp_probability(float(raw))
    except ValueError:
        logger.warning("Invalid probability value '%s'; using default %.2f", raw, default)
        return default


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer value '%s'; using default %d", raw, default)
        return default


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float value '%s'; using default %.2f", raw, default)
        return default


def assess_image_quality(image: np.ndarray) -> dict[str, Any]:
    """Assess image quality and return quality flags for manual review."""
    min_side = _parse_int(os.getenv("DIAFOOT_MIN_IMAGE_SIDE"), DEFAULT_MIN_IMAGE_SIDE)
    blur_thr = _parse_float(
        os.getenv("DIAFOOT_BLUR_VARIANCE_THRESHOLD"),
        DEFAULT_BLUR_VARIANCE_THRESHOLD,
    )
    brightness_min = _parse_float(os.getenv("DIAFOOT_BRIGHTNESS_MIN"), DEFAULT_BRIGHTNESS_MIN)
    brightness_max = _parse_float(os.getenv("DIAFOOT_BRIGHTNESS_MAX"), DEFAULT_BRIGHTNESS_MAX)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())

    flags: list[str] = []
    if min(h, w) < min_side:
        flags.append("low_resolution")
    if blur_variance < blur_thr:
        flags.append("blurry")
    if brightness_mean < brightness_min:
        flags.append("too_dark")
    if brightness_mean > brightness_max:
        flags.append("too_bright")

    return {
        "quality_passed": len(flags) == 0,
        "quality_flags": flags,
        "blur_variance": blur_variance,
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "width": int(w),
        "height": int(h),
    }


def _maybe_log_prediction(
    classification: str,
    classification_confidence: float,
    defer_to_clinician: bool,
    defer_reason: str,
    quality_flags: list[str],
    has_wound: bool,
    wound_area_mm2: float,
    quality: dict[str, Any],
) -> None:
    """Write structured prediction event to JSONL if enabled."""
    if not _prediction_log_path:
        return

    event = PredictionLogEvent(
        timestamp_utc=PredictionLogEvent.now_timestamp(),
        classification=classification,
        classification_confidence=classification_confidence,
        defer_to_clinician=defer_to_clinician,
        defer_reason=defer_reason,
        quality_flags=quality_flags,
        has_wound=has_wound,
        wound_area_mm2=wound_area_mm2,
        drift=DriftFeatures(
            brightness_mean=float(quality.get("brightness_mean", 0.0)),
            brightness_std=float(quality.get("brightness_std", 0.0)),
            blur_variance=float(quality.get("blur_variance", 0.0)),
            width=int(quality.get("width", 0)),
            height=int(quality.get("height", 0)),
        ),
    )

    try:
        log_path = Path(_prediction_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(event.model_dump_json() + "\n")
    except Exception:
        logger.exception("Failed to write prediction log event")


def _read_calibrated_defer_threshold(path: str | Path) -> float | None:
    """Read recommended defer threshold from calibration output JSON."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        with open(p) as f:
            payload = json.load(f)

        cls = payload.get("classification", {})
        defer = cls.get("defer_tuning", {})
        recommended = defer.get("recommended_threshold")
        if recommended is None:
            return None
        return _clamp_probability(float(recommended))
    except Exception:
        logger.warning("Unable to load calibrated defer threshold from %s", path)
        return None


def resolve_runtime_thresholds() -> tuple[float, float, str]:
    """Resolve confidence/defer thresholds from env and calibration artifact.

    Priority for defer threshold:
      1) DIAFOOT_DEFER_THRESHOLD env
      2) recommended threshold from calibration JSON
      3) default constant
    """
    confidence_threshold = _parse_probability(
        _getenv("DIAFOOT_CONFIDENCE_THRESHOLD"),
        DEFAULT_CONFIDENCE_THRESHOLD,
    )

    defer_raw = _getenv("DIAFOOT_DEFER_THRESHOLD")
    if defer_raw is not None:
        return confidence_threshold, _parse_probability(defer_raw, DEFAULT_DEFER_THRESHOLD), "env"

    calibration_path = _getenv("DIAFOOT_CALIBRATION_PATH", default=DEFAULT_CALIBRATION_PATH)
    calibrated = _read_calibrated_defer_threshold(calibration_path)
    if calibrated is not None:
        return confidence_threshold, calibrated, f"calibration:{calibration_path}"

    return confidence_threshold, DEFAULT_DEFER_THRESHOLD, "default"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize pipeline when service starts."""
    global _pipeline
    try:
        _pipeline = _build_pipeline_from_checkpoints()
        if _pipeline is None:
            logger.warning("Pipeline not initialized")
        else:
            logger.info("Pipeline initialized successfully")
    except Exception:
        logger.exception("Failed to initialize pipeline")
        _pipeline = None
    yield


app = FastAPI(
    title="DiaFoot.AI v2",
    description="Diabetic Foot Ulcer Detection, Classification & Segmentation (DINOv2)",
    version="2.1.0",
    lifespan=lifespan,
)

_cors_origins_raw = _getenv("DIAFOOT_CORS_ORIGINS", "CORS_ORIGINS", default="*")
_cors_origins = [o.strip() for o in (_cors_origins_raw or "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    """API response for a prediction."""

    classification: str
    classification_confidence: float
    classification_probs: dict[str, float]
    defer_to_clinician: bool
    defer_reason: str
    quality_flags: list[str]
    has_wound: bool
    wound_area_mm2: float
    wound_coverage_pct: float
    inference_time_ms: float
    segmentation_mask_base64: str | None = None


class HealthResponse(BaseModel):
    """API response for health check."""

    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """API response for model information."""

    classifier: str
    segmenter: str
    input_size: list[int]
    num_classes: int
    confidence_threshold: float
    defer_threshold: float
    defer_threshold_source: str
    max_image_size_mb: float
    rate_limit_rpm: int
    version: str


# Global pipeline (initialized on startup)
_pipeline: Any = None
_confidence_threshold, _defer_threshold, _defer_threshold_source = resolve_runtime_thresholds()
_max_image_size_mb = _parse_float(os.getenv("DIAFOOT_MAX_IMAGE_SIZE_MB"), 20.0)
_rate_limit_rpm = _parse_int(os.getenv("DIAFOOT_RATE_LIMIT_RPM"), 100)
_max_image_size_bytes = int(_max_image_size_mb * 1024 * 1024)
_prediction_log_path = os.getenv("DIAFOOT_PREDICTION_LOG", "")

app.add_middleware(
    MaxContentLengthMiddleware,
    max_content_length=_max_image_size_bytes,
    path_prefix="/predict",
)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=_rate_limit_rpm,
    window_seconds=60,
    path_prefix="/predict",
)


def _resolve_device() -> str:
    requested = _getenv("DIAFOOT_DEVICE", default="cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _build_pipeline_from_checkpoints() -> InferencePipeline | None:
    """Load DINOv2 classifier/segmenter checkpoints and construct inference pipeline."""
    classifier_ckpt = Path(
        _getenv(
            "DIAFOOT_CLASSIFIER_CKPT",
            "DIAFOOT_CLASSIFIER_CHECKPOINT",
            default=DEFAULT_CLASSIFIER_CKPT,
        )
    )
    segmenter_ckpt = Path(
        _getenv(
            "DIAFOOT_SEGMENTER_CKPT",
            "DIAFOOT_SEGMENTER_CHECKPOINT",
            default=DEFAULT_SEGMENTER_CKPT,
        )
    )
    backbone = _getenv("DIAFOOT_BACKBONE", default=DEFAULT_BACKBONE)
    device = _resolve_device()
    dfu_seg_fallback_prob = _parse_probability(
        _getenv("DIAFOOT_DFU_SEG_FALLBACK_PROB"),
        DEFAULT_DFU_SEG_FALLBACK_PROB,
    )
    dfu_promotion_threshold = _parse_probability(
        _getenv("DIAFOOT_DFU_PROMOTION_THRESHOLD"),
        DEFAULT_DFU_PROMOTION_THRESHOLD,
    )

    if not classifier_ckpt.exists():
        logger.warning("Classifier checkpoint not found: %s", classifier_ckpt)
        return None

    # Load DINOv2 classifier
    from src.models.dinov2_classifier import DINOv2Classifier

    classifier = DINOv2Classifier(
        backbone=backbone, num_classes=3, freeze_backbone=True, dropout=0.3
    )
    cls_state = _load_state_dict_from_checkpoint(classifier_ckpt)
    classifier.load_state_dict(cls_state)

    # Load DINOv2 segmenter
    segmenter = None
    if segmenter_ckpt.exists():
        from src.models.dinov2_segmenter import DINOv2Segmenter

        segmenter = DINOv2Segmenter(
            backbone=backbone, num_classes=1, freeze_backbone=True
        )
        seg_state = _load_state_dict_from_checkpoint(segmenter_ckpt)
        segmenter.load_state_dict(seg_state)
    else:
        logger.warning("Segmenter checkpoint not found: %s", segmenter_ckpt)

    return InferencePipeline(
        classifier=classifier,
        segmenter=segmenter,
        device=device,
        confidence_threshold=_confidence_threshold,
        defer_threshold=_defer_threshold,
        dfu_seg_fallback_prob=dfu_seg_fallback_prob,
        dfu_promotion_threshold=dfu_promotion_threshold,
        input_size=518,
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check service health."""
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        version="2.1.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Get model metadata."""
    backbone = _getenv("DIAFOOT_BACKBONE", default=DEFAULT_BACKBONE)
    segmenter_name = f"DINOv2 ({backbone}) + UPerNet" if _pipeline is not None else "Unavailable"
    return ModelInfoResponse(
        classifier=f"DINOv2 ({backbone}) 3-class triage",
        segmenter=segmenter_name,
        input_size=[518, 518],
        num_classes=3,
        confidence_threshold=_confidence_threshold,
        defer_threshold=_defer_threshold,
        defer_threshold_source=_defer_threshold_source,
        max_image_size_mb=_max_image_size_mb,
        rate_limit_rpm=_rate_limit_rpm,
        version="2.1.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File()) -> PredictionResponse:  # noqa: B008
    """Run full inference pipeline on uploaded image.

    Args:
        file: Uploaded image file (JPEG, PNG).

    Returns:
        Full prediction result.
    """
    t0 = time.time()

    # Read image
    contents = await file.read()
    ok_upload, upload_reason = validate_upload_metadata(
        content_type=file.content_type,
        payload_size_bytes=len(contents),
        max_payload_bytes=_max_image_size_bytes,
    )
    if not ok_upload:
        _maybe_log_prediction(
            classification="Manual Review Required",
            classification_confidence=0.0,
            defer_to_clinician=True,
            defer_reason=upload_reason,
            quality_flags=[],
            has_wound=False,
            wound_area_mm2=0.0,
            quality={
                "brightness_mean": 0.0,
                "brightness_std": 0.0,
                "blur_variance": 0.0,
                "width": 0,
                "height": 0,
            },
        )
        return PredictionResponse(
            classification="Manual Review Required",
            classification_confidence=0.0,
            classification_probs={},
            defer_to_clinician=True,
            defer_reason=upload_reason,
            quality_flags=[],
            has_wound=False,
            wound_area_mm2=0.0,
            wound_coverage_pct=0.0,
            inference_time_ms=(time.time() - t0) * 1000,
        )

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        _maybe_log_prediction(
            classification="Error",
            classification_confidence=0.0,
            defer_to_clinician=True,
            defer_reason="invalid_image",
            quality_flags=[],
            has_wound=False,
            wound_area_mm2=0.0,
            quality={
                "brightness_mean": 0.0,
                "brightness_std": 0.0,
                "blur_variance": 0.0,
                "width": 0,
                "height": 0,
            },
        )
        return PredictionResponse(
            classification="Error",
            classification_confidence=0.0,
            classification_probs={},
            defer_to_clinician=True,
            defer_reason="invalid_image",
            quality_flags=[],
            has_wound=False,
            wound_area_mm2=0.0,
            wound_coverage_pct=0.0,
            inference_time_ms=0.0,
        )

    quality = assess_image_quality(image)
    quality_flags = quality["quality_flags"]
    if not quality["quality_passed"]:
        _maybe_log_prediction(
            classification="Manual Review Required",
            classification_confidence=0.0,
            defer_to_clinician=True,
            defer_reason="low_image_quality",
            quality_flags=quality_flags,
            has_wound=False,
            wound_area_mm2=0.0,
            quality=quality,
        )
        return PredictionResponse(
            classification="Manual Review Required",
            classification_confidence=0.0,
            classification_probs={},
            defer_to_clinician=True,
            defer_reason="low_image_quality",
            quality_flags=quality_flags,
            has_wound=False,
            wound_area_mm2=0.0,
            wound_coverage_pct=0.0,
            inference_time_ms=(time.time() - t0) * 1000,
        )

    if _pipeline is None:
        _maybe_log_prediction(
            classification="Model not loaded",
            classification_confidence=0.0,
            defer_to_clinician=True,
            defer_reason="model_not_loaded",
            quality_flags=quality_flags,
            has_wound=False,
            wound_area_mm2=0.0,
            quality=quality,
        )
        return PredictionResponse(
            classification="Model not loaded",
            classification_confidence=0.0,
            classification_probs={},
            defer_to_clinician=True,
            defer_reason="model_not_loaded",
            quality_flags=quality_flags,
            has_wound=False,
            wound_area_mm2=0.0,
            wound_coverage_pct=0.0,
            inference_time_ms=0.0,
        )

    result = _pipeline.predict(image)
    elapsed = (time.time() - t0) * 1000

    _maybe_log_prediction(
        classification=result.classification,
        classification_confidence=result.classification_confidence,
        defer_to_clinician=result.defer_to_clinician,
        defer_reason=result.defer_reason,
        quality_flags=quality_flags,
        has_wound=result.has_wound,
        wound_area_mm2=result.wound_area_mm2,
        quality=quality,
    )

    # Encode segmentation mask as base64 PNG
    mask_b64: str | None = None
    if result.segmentation_mask is not None and result.has_wound:
        mask_img = PILImage.fromarray((result.segmentation_mask * 255).astype("uint8"))
        buf = BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return PredictionResponse(
        classification=result.classification,
        classification_confidence=result.classification_confidence,
        classification_probs=result.classification_probs,
        defer_to_clinician=result.defer_to_clinician,
        defer_reason=result.defer_reason,
        quality_flags=quality_flags,
        has_wound=result.has_wound,
        wound_area_mm2=result.wound_area_mm2,
        wound_coverage_pct=result.wound_coverage_pct,
        inference_time_ms=elapsed,
        segmentation_mask_base64=mask_b64,
    )
