"""DiaFoot.AI v2 — Deployment schemas for monitoring and drift hooks."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel


class DriftFeatures(BaseModel):
    """Lightweight input-distribution features for drift monitoring."""

    brightness_mean: float
    brightness_std: float
    blur_variance: float
    width: int
    height: int


class PredictionLogEvent(BaseModel):
    """Structured prediction log event for offline monitoring."""

    timestamp_utc: str
    classification: str
    classification_confidence: float
    defer_to_clinician: bool
    defer_reason: str
    quality_flags: list[str]
    has_wound: bool
    wound_area_mm2: float
    drift: DriftFeatures

    @classmethod
    def now_timestamp(cls) -> str:
        """Return current UTC timestamp."""
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
