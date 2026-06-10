"""Production middleware for the effGen API server.

Attached to an existing FastAPI app via :func:`install_production_middleware`.
Provides:

- Request ID injection (``X-Request-ID``)
- CORS configuration
- Response compression (gzip)
- Request/response validation is handled by FastAPI/Pydantic by default
- Graceful shutdown hooks (SIGTERM/SIGINT draining)
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import uuid
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_rate_limit(rate_limit_per_minute: int | None) -> int:
    """Resolve the requests-per-minute limit (param overrides EFFGEN_RATE_LIMIT).

    Returns ``0`` (disabled) when neither source provides a positive integer.
    """
    if rate_limit_per_minute is None:
        raw = (os.getenv("EFFGEN_RATE_LIMIT", "") or "").strip()
        if not raw:
            return 0
        try:
            rate_limit_per_minute = int(raw)
        except ValueError:
            logger.warning(
                "EFFGEN_RATE_LIMIT=%r is not an integer — request rate limiting disabled",
                raw,
            )
            return 0
    return max(0, int(rate_limit_per_minute))


def install_production_middleware(
    app: Any,
    *,
    cors_origins: Iterable[str] | None = None,
    dev_mode: bool = False,
    allow_credentials: bool | None = None,
    enable_gzip: bool = True,
    gzip_min_size: int = 500,
    enable_request_id: bool = True,
    shutdown_timeout: float = 10.0,
    rate_limit_per_minute: int | None = None,
) -> None:
    """Install production-grade middleware on a FastAPI ``app``.

    Safe to call even if FastAPI/Starlette is not importable (no-op).

    CORS is fail-closed: when ``cors_origins`` is not provided, cross-origin
    access is **disabled** in production (no CORS middleware is installed) and
    only opened to ``*`` in dev mode. A wildcard origin is never combined with
    ``allow_credentials=True`` (that combination is both insecure and rejected
    by browsers), so credentials are enabled only for an explicit origin list.
    """
    try:
        from starlette.middleware.cors import CORSMiddleware
        from starlette.middleware.gzip import GZipMiddleware
    except Exception:  # pragma: no cover
        logger.warning("fastapi/starlette not available — middleware skipped")
        return

    # 1. CORS
    origins = list(cors_origins or [])
    if not origins and dev_mode:
        origins = ["*"]
    wildcard = "*" in origins
    if allow_credentials is None:
        # Credentials only for an explicit, non-wildcard origin allow-list.
        allow_credentials = bool(origins) and not wildcard
    if wildcard and allow_credentials:
        logger.warning(
            "CORS: refusing to combine wildcard origin with credentials; "
            "disabling allow_credentials."
        )
        allow_credentials = False

    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )
    else:
        logger.info(
            "CORS: no cross-origin access configured (set EFFGEN_CORS_ORIGINS "
            "or pass cors_origins to enable)."
        )

    # 2. gzip compression
    if enable_gzip:
        app.add_middleware(GZipMiddleware, minimum_size=gzip_min_size)

    # 3. Request ID injection. Implemented as a *pure-ASGI* middleware (not the
    # @app.middleware("http") / BaseHTTPMiddleware form) because
    # BaseHTTPMiddleware buffers the whole response body before forwarding it,
    # which breaks server-sent-event streaming (the /v1 SSE responses) — it
    # would hang or raise "No response returned." Pure-ASGI middleware forwards
    # each chunk as it is produced, so streaming stays truly incremental.
    if enable_request_id:
        app.add_middleware(RequestIDMiddleware)  # type: ignore[arg-type]

    # 4. Per-client request rate limiting (opt-in via EFFGEN_RATE_LIMIT).
    # Added last so it wraps as the outermost layer: a request flood is rejected
    # cheaply (per-IP, before auth/route work). Disabled unless a positive limit
    # is configured.
    limit = _resolve_rate_limit(rate_limit_per_minute)
    if limit > 0:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=limit)  # type: ignore[arg-type]
        logger.info("Request rate limiting enabled: %d req/min per client", limit)

    # 5. Graceful shutdown
    _install_graceful_shutdown(app, shutdown_timeout)


class RequestIDMiddleware:
    """Pure-ASGI middleware that stamps every request/response with an id.

    Reads an inbound ``X-Request-ID`` (or generates one), exposes it on
    ``scope["state"]["request_id"]``, and echoes it on the response. Unlike a
    ``BaseHTTPMiddleware`` it never buffers the response body, so streaming
    (SSE) responses are forwarded chunk-by-chunk.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        inbound = headers.get(b"x-request-id")
        req_id = inbound.decode("latin-1", errors="replace") if inbound else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = req_id
        req_id_bytes = req_id.encode("latin-1", errors="replace")

        async def _send(message: dict) -> None:
            if message.get("type") == "http.response.start":
                hdrs = [
                    (k, v) for k, v in message.get("headers", [])
                    if k.lower() != b"x-request-id"
                ]
                hdrs.append((b"x-request-id", req_id_bytes))
                message = {**message, "headers": hdrs}
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except Exception:
            logger.exception("request %s failed", req_id)
            raise


# Health/liveness/readiness probes are exempt from rate limiting so that a
# frequently-polling load balancer or K8s probe can never throttle real traffic.
_RATE_LIMIT_EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/health", "/healthz", "/livez", "/readyz", "/ready"}
)


class RateLimitMiddleware:
    """Pure-ASGI fixed-window per-client request limiter.

    Honours the ``EFFGEN_RATE_LIMIT`` knob (requests/minute). Keyed by client IP
    so a flood is rejected before auth/route work, mirroring the per-IP limit the
    Cloudflare worker enforces at the edge. The window is per-process (each
    uvicorn worker keeps its own counters); behind multiple workers the effective
    limit is ``requests_per_minute × workers``. Probe paths are exempt and
    ``OPTIONS`` preflight requests are never counted. On breach it returns a
    redacted ``429`` with a ``Retry-After`` header — no buffering, so SSE
    responses on the allowed path stay incremental.
    """

    def __init__(self, app: Any, *, requests_per_minute: int, window_seconds: int = 60) -> None:
        self.app = app
        self.limit = max(1, int(requests_per_minute))
        self.window = max(1, int(window_seconds))
        # client-ip -> [window_start_epoch, count]
        self._buckets: dict[str, list[float]] = {}

    def _client_ip(self, scope: dict) -> str:
        # Honour a single X-Forwarded-For hop when present (reverse-proxy case),
        # else fall back to the socket peer. Never trust beyond the first hop.
        for k, v in scope.get("headers", []):
            if k == b"x-forwarded-for":
                first = v.decode("latin-1", errors="replace").split(",")[0].strip()
                if first:
                    return first
        client = scope.get("client")
        return client[0] if client else "unknown"

    def _check(self, ip: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        bucket = self._buckets.get(ip)
        if bucket is None or now - bucket[0] >= self.window:
            self._buckets[ip] = [now, 1]
            # Opportunistically prune stale buckets to bound memory.
            if len(self._buckets) > 4096:
                cutoff = now - self.window
                self._buckets = {
                    k: b for k, b in self._buckets.items() if b[0] >= cutoff
                }
            return True, 0
        if bucket[1] < self.limit:
            bucket[1] += 1
            return True, 0
        return False, max(1, int(self.window - (now - bucket[0])))

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in _RATE_LIMIT_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        allowed, retry_after = self._check(self._client_ip(scope))
        if allowed:
            await self.app(scope, receive, send)
            return

        import json as _json

        body = _json.dumps(
            {"detail": "Rate limit exceeded. Please retry later."}
        ).encode()
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", str(retry_after).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def _install_graceful_shutdown(app: Any, timeout: float) -> None:
    """Drain in-flight requests before exiting on SIGTERM/SIGINT."""
    inflight: set = set()
    shutting_down = {"value": False}

    class _InflightMiddleware:
        """Pure-ASGI in-flight tracker (no response buffering — SSE-safe)."""

        def __init__(self, asgi_app: Any) -> None:
            self.app = asgi_app

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope.get("type") != "http":
                await self.app(scope, receive, send)
                return
            if shutting_down["value"]:
                import json as _json

                body = _json.dumps({"error": "server is shutting down"}).encode()
                await send({
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                })
                await send({"type": "http.response.body", "body": body})
                return
            task = asyncio.current_task()
            if task is not None:
                inflight.add(task)
            try:
                await self.app(scope, receive, send)
            finally:
                if task is not None:
                    inflight.discard(task)

    app.add_middleware(_InflightMiddleware)  # type: ignore[arg-type]

    @app.on_event("shutdown")
    async def _drain() -> None:
        shutting_down["value"] = True
        deadline = asyncio.get_event_loop().time() + timeout
        while inflight and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.1)
        if inflight:
            logger.warning(
                "shutdown: %d requests still in-flight after %.1fs",
                len(inflight),
                timeout,
            )

    # Best-effort signal hookup; uvicorn handles SIGTERM itself, but when
    # embedded elsewhere this ensures flags still flip.
    try:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig, lambda: shutting_down.__setitem__("value", True)
                )
            except (NotImplementedError, RuntimeError):
                pass
    except RuntimeError:
        pass
