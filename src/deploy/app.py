"""DiaFoot.AI v2 — FastAPI REST Service.

Phase 6, Commit 31: REST API for multi-task inference.

Endpoints:
    POST /predict — Upload image, get full pipeline result
    GET /health — Service health check
    GET /model/info — Model metadata
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DiaFoot.AI v2",
    description="Diabetic Foot Ulcer Detection, Classification & Segmentation",
    version="2.0.0",
)


class PredictionResponse(BaseModel):
    """API response for a prediction."""

    classification: str
    classification_confidence: float
    classification_probs: dict[str, float]
    has_wound: bool
    wound_area_mm2: float
    wound_coverage_pct: float
    inference_time_ms: float


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
    version: str


# Global pipeline (initialized on startup)
_pipeline: Any = None


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check service health."""
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        version="2.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Get model metadata."""
    return ModelInfoResponse(
        classifier="EfficientNet-V2-M (3-class triage)",
        segmenter="U-Net++ / EfficientNet-B4",
        input_size=[512, 512],
        num_classes=3,
        version="2.0.0",
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
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return PredictionResponse(
            classification="Error",
            classification_confidence=0.0,
            classification_probs={},
            has_wound=False,
            wound_area_mm2=0.0,
            wound_coverage_pct=0.0,
            inference_time_ms=0.0,
        )

    if _pipeline is None:
        return PredictionResponse(
            classification="Model not loaded",
            classification_confidence=0.0,
            classification_probs={},
            has_wound=False,
            wound_area_mm2=0.0,
            wound_coverage_pct=0.0,
            inference_time_ms=0.0,
        )

    result = _pipeline.predict(image)
    elapsed = (time.time() - t0) * 1000

    return PredictionResponse(
        classification=result.classification,
        classification_confidence=result.classification_confidence,
        classification_probs=result.classification_probs,
        has_wound=result.has_wound,
        wound_area_mm2=result.wound_area_mm2,
        wound_coverage_pct=result.wound_coverage_pct,
        inference_time_ms=elapsed,
    )
