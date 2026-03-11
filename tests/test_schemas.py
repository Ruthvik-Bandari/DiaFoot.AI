"""Tests for deploy schemas."""

from __future__ import annotations

from src.deploy.schemas import DriftFeatures, PredictionLogEvent


def test_prediction_log_event_schema() -> None:
    event = PredictionLogEvent(
        timestamp_utc=PredictionLogEvent.now_timestamp(),
        classification="DFU",
        classification_confidence=0.92,
        defer_to_clinician=False,
        defer_reason="",
        quality_flags=[],
        has_wound=True,
        wound_area_mm2=123.4,
        drift=DriftFeatures(
            brightness_mean=110.0,
            brightness_std=25.0,
            blur_variance=220.0,
            width=512,
            height=512,
        ),
    )
    payload = event.model_dump()
    assert payload["classification"] == "DFU"
    assert payload["drift"]["width"] == 512
