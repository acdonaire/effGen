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
import signal
import uuid
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


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

    # 4. Graceful shutdown
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
