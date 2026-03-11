"""DiaFoot.AI v2 — API middleware utilities.

Includes:
- Request payload size limiting
- In-memory per-client rate limiting
- Upload content-type validation helpers
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


def is_allowed_content_type(content_type: str | None) -> bool:
    """Return True if content type is an allowed image type."""
    if content_type is None:
        return False
    return content_type.lower().split(";")[0].strip() in ALLOWED_IMAGE_CONTENT_TYPES


def validate_upload_metadata(
    content_type: str | None,
    payload_size_bytes: int,
    max_payload_bytes: int,
) -> tuple[bool, str]:
    """Validate upload metadata and return `(ok, reason)`.

    Reasons when invalid:
    - `invalid_content_type`
    - `payload_too_large`
    """
    if not is_allowed_content_type(content_type):
        return False, "invalid_content_type"
    if payload_size_bytes > max_payload_bytes:
        return False, "payload_too_large"
    return True, ""


class MaxContentLengthMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds configured maximum."""

    def __init__(
        self,
        app: Any,
        max_content_length: int,
        path_prefix: str = "/predict",
    ) -> None:
        super().__init__(app)
        self.max_content_length = max_content_length
        self.path_prefix = path_prefix

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.url.path.startswith(self.path_prefix):
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > self.max_content_length:
                        return JSONResponse(
                            status_code=413,
                            content={
                                "detail": "Request payload too large",
                                "max_bytes": self.max_content_length,
                            },
                        )
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length header"},
                    )
        return await call_next(request)


class _InMemoryRateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, int]:
        """Return `(allowed, retry_after_seconds)` for a key."""
        now = time.time()
        events = self._events[key]

        # Drop stale timestamps.
        while events and (now - events[0]) > self.window_seconds:
            events.popleft()

        if len(events) >= self.max_requests:
            retry_after = max(1, int(self.window_seconds - (now - events[0])))
            return False, retry_after

        events.append(now)
        return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply in-memory per-client rate limiting to matching paths."""

    def __init__(
        self,
        app: Any,
        max_requests: int,
        window_seconds: int = 60,
        path_prefix: str = "/predict",
    ) -> None:
        super().__init__(app)
        self.path_prefix = path_prefix
        self.limiter = _InMemoryRateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.url.path.startswith(self.path_prefix):
            client_host = request.client.host if request.client else "unknown"
            allowed, retry_after = self.limiter.allow(client_host)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after_seconds": retry_after,
                    },
                )
        return await call_next(request)
