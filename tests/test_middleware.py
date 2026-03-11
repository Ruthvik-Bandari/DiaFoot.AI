"""Middleware tests for API payload/rate guardrails."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.deploy.middleware import (
    MaxContentLengthMiddleware,
    RateLimitMiddleware,
    validate_upload_metadata,
)


def test_validate_upload_metadata_content_type() -> None:
    ok, reason = validate_upload_metadata(
        content_type="text/plain",
        payload_size_bytes=128,
        max_payload_bytes=1024,
    )
    assert ok is False
    assert reason == "invalid_content_type"


def test_validate_upload_metadata_size() -> None:
    ok, reason = validate_upload_metadata(
        content_type="image/png",
        payload_size_bytes=4096,
        max_payload_bytes=1024,
    )
    assert ok is False
    assert reason == "payload_too_large"


def test_max_content_length_middleware_blocks_large_request() -> None:
    app = FastAPI()
    app.add_middleware(MaxContentLengthMiddleware, max_content_length=10, path_prefix="/predict")

    @app.post("/predict")
    async def predict() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    resp = client.post("/predict", content=b"x" * 64)
    assert resp.status_code == 413


def test_rate_limit_middleware_blocks_after_limit() -> None:
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=2,
        window_seconds=60,
        path_prefix="/predict",
    )

    @app.post("/predict")
    async def predict() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    assert client.post("/predict").status_code == 200
    assert client.post("/predict").status_code == 200
    blocked = client.post("/predict")
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers
