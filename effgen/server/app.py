"""effGen API Server — main FastAPI application with auth/RBAC/audit wired in.

Usage::

    # Production (requires EFFGEN_OIDC_ISSUER + EFFGEN_OIDC_CLIENT_ID)
    uvicorn effgen.server.app:create_app --factory --host 0.0.0.0 --port 8080

    # Dev mode (auth disabled, loud warning)
    EFFGEN_DEV_MODE=1 uvicorn effgen.server.app:create_app --factory ...

Environment variables
---------------------
EFFGEN_DEV_MODE          — "1" disables auth (warn only, never silently)
EFFGEN_OIDC_ISSUER       — OIDC issuer URL
EFFGEN_OIDC_CLIENT_ID    — JWT audience claim
EFFGEN_OIDC_JWKS_URI     — JWKS endpoint (auto-discovered if omitted)
EFFGEN_RBAC_POLICY_FILE  — path to JSON role policy file
EFFGEN_AUDIT_DIR         — override audit log directory
EFFGEN_METRICS_AUTH      — "1" to require auth on /metrics
"""
from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

# Known provider prefixes — stable fallback when the dynamic ProviderRegistry
# is unavailable or has been reset (e.g. in test teardowns).
_KNOWN_PROVIDERS: frozenset[str] = frozenset(
    {
        "cerebras",
        "openai",
        "anthropic",
        "gemini",
        "groq",
        "together",
        "fireworks",
        "replicate",
        "hf",
    }
)

# FastAPI Request must be importable at module level so route type annotations
# resolve correctly when `from __future__ import annotations` is active.
try:
    from fastapi import Request as _FastAPIRequest
except ImportError:  # pragma: no cover
    _FastAPIRequest = None  # type: ignore[assignment,misc]


def _server_version() -> str:
    """Return the effGen package version without paying for a heavy import.

    Reads the installed distribution metadata first (cheap; does not import the
    ``effgen`` package). Falls back to the package attribute, then a sentinel.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("effgen")
        except PackageNotFoundError:
            pass
    except Exception:  # noqa: BLE001 - importlib.metadata always present on 3.11
        pass
    try:
        from effgen import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "0.0.0"


def _truthy_env(name: str) -> bool:
    """Return True when env var *name* is set to a truthy value."""
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    dev_mode: bool | None = None,
    oidc_issuer: str | None = None,
    oidc_client_id: str | None = None,
    oidc_jwks_uri: str | None = None,
    api_key: str | None = None,
    metrics_auth: bool = False,
    public_metrics: bool | None = None,
    public_dashboard: bool | None = None,
    cors_origins: list[str] | None = None,
    rate_limit_per_minute: int | None = None,
    runner: Any = None,
) -> Any:
    """Create and return the FastAPI application.

    All parameters fall back to environment variables when ``None``.

    Parameters
    ----------
    runner:
        Optional callable ``runner(prompt, *, model, tools, stream, ...)`` used
        by the OpenAI-compatible ``/v1`` endpoints. When ``None`` a default
        agent-backed runner is constructed lazily on first use.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise ImportError("fastapi is required: pip install 'effgen[server]'") from exc

    _dev = dev_mode if dev_mode is not None else (os.getenv("EFFGEN_DEV_MODE", "0") == "1")

    # Observability endpoints are protected by default. They become public only
    # in dev mode or when an operator explicitly opts in (constructor flag or
    # EFFGEN_PUBLIC_METRICS / EFFGEN_PUBLIC_DASHBOARD). The legacy
    # ``metrics_auth`` flag (and EFFGEN_METRICS_AUTH) still force metrics auth on.
    if public_metrics is None:
        public_metrics = _dev or _truthy_env("EFFGEN_PUBLIC_METRICS")
    if metrics_auth or _truthy_env("EFFGEN_METRICS_AUTH"):
        public_metrics = False
    if public_dashboard is None:
        public_dashboard = _dev or _truthy_env("EFFGEN_PUBLIC_DASHBOARD")

    app = FastAPI(
        title="effGen API",
        description="effGen — Efficient Agent Framework API",
        version=_server_version(),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ------------------------------------------------------------------
    # Middleware stack. ``add_middleware`` is LIFO, so the *last* added wraps
    # as the outermost layer. We want, outer → inner:
    #
    #   production (CORS/gzip/request-id)  →  audit  →  auth  →  RBAC/budget  →  route
    #
    # Auth must run before RBAC/budget so request.state.user is populated, and
    # RBAC/budget sits innermost (just outside the route) so its request-body
    # replay survives the production BaseHTTPMiddleware layers above it.
    # ------------------------------------------------------------------

    # 1. RBAC + budget enforcement (innermost; reads body for /v1 endpoints)
    app.add_middleware(RBACBudgetMiddleware)  # type: ignore[arg-type]

    # 2. Auth middleware (validates JWT, populates request.state.user)
    from effgen.server.auth import AuthMiddleware

    app.add_middleware(
        AuthMiddleware,  # type: ignore[arg-type]
        issuer=oidc_issuer,
        client_id=oidc_client_id,
        jwks_uri=oidc_jwks_uri,
        api_key=api_key,
        public_metrics=public_metrics,
        public_dashboard=public_dashboard,
        dev_mode=_dev,
    )

    # 3. Audit middleware (logs every request; reads user from state set by auth)
    from effgen.server.audit import AuditMiddleware

    app.add_middleware(AuditMiddleware)  # type: ignore[arg-type]

    # 4. Production middleware (CORS, gzip, request-ID) — outermost.
    # CORS is fail-closed: outside dev mode, cross-origin access is disabled
    # unless ``cors_origins`` (or EFFGEN_CORS_ORIGINS) is explicitly configured.
    if cors_origins is None:
        _env_origins = os.getenv("EFFGEN_CORS_ORIGINS", "").strip()
        if _env_origins:
            cors_origins = [o.strip() for o in _env_origins.split(",") if o.strip()]
    try:
        from effgen.api.middleware import install_production_middleware

        install_production_middleware(
            app,
            cors_origins=cors_origins,
            dev_mode=_dev,
            rate_limit_per_minute=rate_limit_per_minute,
        )
    except ImportError:
        logger.warning("Production middleware not available")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", tags=["ops"])
    @app.get("/healthz", tags=["ops"])
    @app.get("/livez", tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness probe — always public.

        ``/health`` is the canonical name; ``/healthz`` and ``/livez`` are
        aliases for the conventional Kubernetes liveness probe path. A 200 here
        means the process is up and serving requests.
        """
        return {"status": "ok", "version": _server_version()}

    @app.get("/ready", tags=["ops"])
    @app.get("/readyz", tags=["ops"])
    async def ready() -> dict[str, str]:
        """Readiness probe — always public.

        Returns 200 once the application is constructed and able to accept
        traffic. K8s readiness probes conventionally use ``/readyz``; ``/ready``
        is provided as an alias. Kept lightweight (no per-request model load) so
        a load balancer can poll it cheaply.
        """
        return {"status": "ready", "version": _server_version()}

    @app.get("/metrics", tags=["ops"])
    async def metrics(request: _FastAPIRequest) -> Any:  # type: ignore[valid-type]
        """Prometheus metrics (requires auth by default; opt out via --public-metrics)."""
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
            from starlette.responses import Response

            payload = generate_latest().decode("utf-8", errors="replace")
            try:
                from effgen.observability.metrics import export_metrics

                payload += "\n" + export_metrics()
            except Exception:  # noqa: BLE001
                pass
            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
        except ImportError:
            return JSONResponse({"detail": "prometheus_client not installed"}, status_code=503)

    @app.get("/whoami", tags=["auth"])
    async def whoami(request: _FastAPIRequest) -> dict[str, Any]:  # type: ignore[valid-type]
        """Return the current principal's identity (requires auth in non-dev mode)."""
        user = getattr(request.state, "user", None)
        if user is None:
            return {"principal": "anonymous"}
        return {
            "sub": user.sub,
            "iss": user.iss,
            "roles": user.roles,
            "email": user.email,
        }

    @app.get("/rbac/policy", tags=["auth"])
    async def rbac_policy(request: _FastAPIRequest) -> dict[str, Any]:  # type: ignore[valid-type]
        """Return the effective RBAC policy for the current principal."""
        from effgen.server.rbac import PolicyDenied, resolve_policy

        user = getattr(request.state, "user", None)
        roles: list[str] = getattr(user, "roles", []) if user else []
        try:
            policy = resolve_policy(roles)
        except PolicyDenied as exc:
            return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)
        return {
            "roles": [r.name for r in policy.roles],
            "allowed_tools": sorted(policy.allowed_tools) or ["*"],
            "allowed_models": sorted(policy.allowed_models) or ["*"],
            "max_cost_per_day": policy.max_cost_per_day,
        }

    @app.get("/rbac/roles", tags=["auth"])
    async def rbac_roles() -> list[dict[str, Any]]:
        """List all defined RBAC roles."""
        from effgen.server.rbac import list_roles

        return [r.to_dict() for r in list_roles()]

    # Import and mount the OpenAI-compatible + embeddings routers, with RBAC
    # and per-principal cost-cap enforcement layered on top.
    _mount_existing_routers(app, runner=runner)

    # Mount the local dashboard (static SPA + data.json + SSE spans).
    _mount_dashboard(app)

    return app


# ---------------------------------------------------------------------------
# Model pool — reuse loaded models across requests (Audit-2 #25)
# ---------------------------------------------------------------------------
#
# Constructing a fresh Agent per request is cheap, but each Agent used to spin
# up its own ModelLoader and *reload the model* every call — catastrophic for
# local GPU models and wasteful for cloud adapters. We instead pool the loaded
# *model* (the heavy object) keyed by its resolved id, bounded by an LRU, and
# build a lightweight, fresh-memory Agent around the pooled model per request.
# A fresh Agent per call keeps the endpoint stateless (no conversation bleed
# between unrelated API requests) while paying the model-load cost only once.

_MODEL_POOL: "OrderedDict[str, Any]" = OrderedDict()
_MODEL_POOL_LOCK = threading.Lock()
_LOADING_LOCKS: dict[str, Any] = {}


def _model_pool_max() -> int:
    try:
        return max(1, int(os.getenv("EFFGEN_MODEL_POOL_SIZE", "4")))
    except ValueError:
        return 4


def _get_pooled_model(resolved_model: str) -> Any:
    """Return a loaded model for *resolved_model*, reusing a pooled instance.

    Thread-safe with a bounded LRU. The (potentially slow) load happens outside
    the pool lock, serialized per-model so two concurrent first requests don't
    both load the same model.
    """
    with _MODEL_POOL_LOCK:
        model = _MODEL_POOL.get(resolved_model)
        if model is not None:
            _MODEL_POOL.move_to_end(resolved_model)
            return model
        load_lock = _LOADING_LOCKS.setdefault(resolved_model, threading.Lock())

    with load_lock:
        # Re-check: another thread may have loaded it while we waited.
        with _MODEL_POOL_LOCK:
            model = _MODEL_POOL.get(resolved_model)
            if model is not None:
                _MODEL_POOL.move_to_end(resolved_model)
                return model

        from effgen.models.model_loader import load_model as _load_model

        model = _load_model(resolved_model)

        evicted: list[Any] = []
        with _MODEL_POOL_LOCK:
            _MODEL_POOL[resolved_model] = model
            _MODEL_POOL.move_to_end(resolved_model)
            while len(_MODEL_POOL) > _model_pool_max():
                _, old = _MODEL_POOL.popitem(last=False)
                evicted.append(old)
            _LOADING_LOCKS.pop(resolved_model, None)

    for old in evicted:
        for closer in ("close", "unload", "shutdown"):
            fn = getattr(old, closer, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    logger.debug("Error closing evicted model", exc_info=True)
                break
    return model


def _extract_usage(response: Any) -> tuple[int | None, int | None]:
    """Pull (prompt_tokens, completion_tokens) from an AgentResponse.

    Prefers provider/tokenizer counts the agent recorded in metadata; falls
    back to the aggregate ``tokens_used`` for completion tokens.
    """
    meta = getattr(response, "metadata", None) or {}

    def _int_or_none(v: Any) -> int | None:
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    prompt = _int_or_none(meta.get("input_tokens", meta.get("prompt_tokens")))
    completion = _int_or_none(meta.get("output_tokens", meta.get("completion_tokens")))
    if completion is None:
        completion = _int_or_none(getattr(response, "tokens_used", None)) or None
    return prompt, completion


def _build_default_runner() -> Any:
    """Construct an agent-backed runner for the OpenAI-compatible endpoints.

    Returns a callable ``runner(prompt, *, model, tools, stream, **kw)`` that
    drives an :class:`~effgen.core.agent.Agent` around a *pooled* model. For
    ``stream=True`` it returns a lazy token generator (so the route can begin
    flushing SSE immediately); otherwise it returns a
    :class:`~effgen.api.openai_compat.RunnerResult` carrying real usage and the
    resolved model id.
    """
    def _runner(
        prompt: str,
        *,
        model: str,
        tools: Any = None,
        stream: bool = False,
        temperature: float | None = None,
        **_: Any,
    ) -> Any:
        from effgen.api.openai_compat import RunnerResult
        from effgen.core.agent import Agent, AgentConfig

        resolved_model = _normalize_model_id(model)
        pooled_model = _get_pooled_model(resolved_model)
        resolved_tools = _resolve_tools(tools)
        config = AgentConfig(
            name="api",
            model=pooled_model,  # pre-loaded instance → no per-request reload
            tools=resolved_tools,
            temperature=temperature if temperature is not None else 0.7,
            require_model=True,
        )
        agent = Agent(config)

        if stream:
            # Close the ephemeral agent once the stream is exhausted. ``close()``
            # releases the agent's own resources (memory handles, circuit
            # breakers) but NOT the pooled model, which is shared and stays
            # loaded — and it silences the "garbage-collected without close()"
            # warning that would otherwise fire per streamed request.
            def _stream_then_close() -> Any:
                try:
                    yield from agent.stream(prompt)
                finally:
                    agent.close()

            return _stream_then_close()

        try:
            response = agent.run(prompt)
            prompt_tokens, completion_tokens = _extract_usage(response)
            return RunnerResult(
                text=getattr(response, "output", "") or "",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                resolved_model=resolved_model,
                finish_reason="stop" if getattr(response, "success", True) else "error",
            )
        finally:
            agent.close()

    return _runner


def _normalize_model_id(model: str) -> str:
    """Normalize an OpenAI-style ``provider/model`` id for effGen's loader.

    OpenAI-compatible clients send ``"cerebras/llama3.1-8b"`` (slash), but
    effGen's ``ModelLoader`` routes API providers via the ``"provider:model"``
    (colon) prefix. Without this, a slash id falls through to the local
    Transformers path and fails with "not a valid model identifier". Rewrite
    the *provider* separator to a colon only when the prefix is a known
    provider, leaving bare ids and HF ``org/repo`` ids untouched.
    """
    if not isinstance(model, str) or "/" not in model:
        return model
    prefix, _, rest = model.partition("/")
    try:
        from effgen.models.registry import ProviderRegistry

        providers = ProviderRegistry.list_providers()
        if not providers:
            # Registry may have been reset (e.g. by test teardowns).
            # Fall back to the static provider set rather than attempting a
            # re-import that could cause circular-import or side-effect issues.
            providers = _KNOWN_PROVIDERS  # type: ignore[assignment]

        if prefix in providers:
            return f"{prefix}:{rest}"
    except Exception:  # noqa: BLE001 - registry optional; fall back to static set
        if prefix in _KNOWN_PROVIDERS:
            return f"{prefix}:{rest}"
    return model


def _resolve_tools(tools: Any) -> list[Any]:
    """Resolve OpenAI-style tool specs / bare names into effGen tool objects.

    The OpenAI chat schema sends tools as ``{"type": "function", "function":
    {"name": ...}}`` dicts; effGen's :class:`Agent` expects tool *instances*.
    Names are resolved against the built-in tool registry; anything that
    cannot be resolved is skipped (RBAC has already authorized the names).
    """
    if not tools:
        return []
    import asyncio

    from effgen.tools.registry import ToolRegistry

    registry = ToolRegistry()
    try:
        registry.discover_builtin_tools()
    except Exception:  # noqa: BLE001
        pass

    def _get(name: str) -> Any:
        """Resolve one tool name, working whether or not a loop is running."""
        coro = registry.get_tool(name)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # A loop is already running (FastAPI route): run in a worker thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()

    resolved: list[Any] = []
    for tool in tools:
        if not isinstance(tool, str) and not isinstance(tool, dict):
            # Already a tool object.
            resolved.append(tool)
            continue
        name = _tool_name(tool)
        try:
            obj = _get(name)
            if obj is not None:
                resolved.append(obj)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not resolve tool %r: %s", name, exc)
    return resolved


def _tool_name(tool: Any) -> str:
    """Extract a tool name from an OpenAI-style tool spec or a bare string."""
    if isinstance(tool, dict):
        return tool.get("function", {}).get("name") or tool.get("name") or "tool"
    return str(tool)


class RBACBudgetMiddleware:
    """Pure-ASGI middleware enforcing RBAC tool/model access + daily cost cap.

    Applies only to model-invoking endpoints (``/v1/chat/completions`` and
    ``/v1/completions``). It runs *after* :class:`AuthMiddleware` (so
    ``scope["state"]["user"]`` is set) and *just outside* the route, so its
    request-body replay survives the production ``BaseHTTPMiddleware`` layers.

    Rejections:
      * 403 ``role X does not permit tool Y`` — disallowed tool,
      * 403 ``role X does not permit model Y`` — disallowed model,
      * 429 ``BudgetExceeded`` — daily cost cap already met.

    Budget handling is reserve-then-reconcile: a per-call estimate
    (``EFFGEN_PER_CALL_COST_USD``, default ``0.01``) is *reserved* before the
    route runs (rejecting an over-cap principal with 429) and committed only if
    the call succeeds. Failed calls (HTTP >= 400 or an exception) release the
    reservation and are **not** charged.

    Request bodies are bounded by ``EFFGEN_MAX_BODY_BYTES`` (default 10 MiB)
    before buffering; oversized bodies are rejected with 413. Bodies are read
    only for the enforced ``/v1`` routes — all other paths pass straight
    through without touching the body.
    """

    _ENFORCED_PATHS = ("/v1/chat/completions", "/v1/completions")

    def __init__(self, app: Any) -> None:
        self.app = app
        self.per_call_cost = float(os.getenv("EFFGEN_PER_CALL_COST_USD", "0.01"))
        self.max_body_bytes = int(os.getenv("EFFGEN_MAX_BODY_BYTES", str(10 * 1024 * 1024)))

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        # Skip body reads entirely for routes that don't need RBAC/budget.
        if scope.get("type") != "http" or scope.get("path") not in self._ENFORCED_PATHS:
            await self.app(scope, receive, send)
            return

        from effgen.server import budget as _budget
        from effgen.server.rbac import PolicyDenied, resolve_policy

        user = scope.get("state", {}).get("user")
        principal = getattr(user, "sub", "anonymous") if user else "anonymous"
        roles: list[str] = list(getattr(user, "roles", []) if user else [])
        try:
            policy = resolve_policy(roles)
        except PolicyDenied as exc:
            await _reject_json(send, exc.status_code, str(exc))
            return
        primary_role = roles[0] if roles else (policy.roles[0].name if policy.roles else "anonymous")

        # Reject early on an oversized declared Content-Length.
        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_body_bytes:
                    await _reject_json(send, 413, "Request body too large")
                    return
            except ValueError:
                pass

        # Buffer the request body (bounded), then replay it to the route.
        chunks: list[bytes] = []
        total = 0
        more = True
        while more:
            msg = await receive()
            chunk = msg.get("body", b"")
            total += len(chunk)
            if total > self.max_body_bytes:
                await _reject_json(send, 413, "Request body too large")
                return
            chunks.append(chunk)
            more = msg.get("more_body", False)
        raw = b"".join(chunks)

        # Replay the buffered body to the route once. A StreamingResponse runs
        # a disconnect-listener concurrently with the body generator, looping on
        # receive(); if we kept handing it the request body it would spin
        # forever and starve the SSE generator (the /v1 streaming hang). After
        # the single replay we *park* — exactly as a real ASGI server's
        # receive() blocks until the client actually sends something — so the
        # disconnect-listener waits quietly and is cancelled when the response
        # finishes, instead of busy-looping or being told the client vanished.
        import asyncio as _asyncio

        _replay_state = {"sent": False}
        _parked = _asyncio.Event()

        async def _replay() -> dict[str, Any]:
            if not _replay_state["sent"]:
                _replay_state["sent"] = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await _parked.wait()  # cancelled when the route's response completes
            return {"type": "http.disconnect"}

        body: Any = {}
        if raw:
            try:
                import json as _json

                body = _json.loads(raw)
            except Exception:  # noqa: BLE001
                body = {}
        model = body.get("model") if isinstance(body, dict) else None
        tools = body.get("tools") if isinstance(body, dict) else None

        if model and not policy.allows_model(str(model)):
            await _reject_json(
                send, 403, f"role {primary_role} does not permit model {model}"
            )
            return

        for tool in tools or []:
            tname = _tool_name(tool)
            if not policy.allows_tool(tname):
                await _reject_json(
                    send, 403, f"role {primary_role} does not permit tool {tname}"
                )
                return

        # Reserve budget before the call; commit only on success.
        try:
            token = _budget.reserve(
                principal, self.per_call_cost, cap=policy.max_cost_per_day
            )
        except _budget.BudgetExceeded as exc:
            await _reject_json(send, exc.status_code, str(exc))
            return

        status_holder: dict[str, int] = {"status": 500}

        async def _send_wrapper(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status_holder["status"] = int(message.get("status", 200))
            await send(message)

        committed = False
        try:
            await self.app(scope, _replay, _send_wrapper)
            if status_holder["status"] < 400:
                _budget.reconcile(principal, token)  # charge the reserved estimate
            else:
                _budget.release(principal, token)  # failed call → no charge
            committed = True
        finally:
            if not committed:
                # The route raised before we could settle the reservation.
                _budget.release(principal, token)


async def _reject_json(send: Any, status: int, detail: str) -> None:
    """Emit a JSON error response from an ASGI middleware."""
    import json as _json

    payload = _json.dumps({"detail": detail}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": payload})


def _mount_existing_routers(app: Any, *, runner: Any = None) -> None:
    """Mount the OpenAI-compat + embeddings routers."""
    try:
        from effgen.api.openai_compat import create_openai_router

        _runner = runner or _build_default_runner()
        router = create_openai_router(_runner)
        app.include_router(router)
        logger.info("Mounted OpenAI-compat router at /v1 (RBAC + budget enforced)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAI-compat router not mounted: %s", exc)

    try:
        from effgen.api.embeddings import create_embeddings_router

        router = create_embeddings_router()
        # Router already has /v1 in path; mount at root
        app.include_router(router)
        logger.info("Mounted embeddings router")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Embeddings router not mounted: %s", exc)


def _mount_dashboard(app: Any) -> None:
    """Mount the local dashboard SPA at /dashboard.

    Serves:
    - ``GET /dashboard``           — the SPA index.html
    - ``GET /dashboard/data.json`` — live metrics + run data as JSON
    - ``GET /dashboard/spans``     — SSE stream of recent span events
    - ``GET /dashboard/{path}``    — other static assets (JS, CSS)
    """
    try:
        from pathlib import Path as _Path

        from fastapi import APIRouter
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

        router = APIRouter(prefix="/dashboard", tags=["dashboard"])

        static_dir = _Path(__file__).parent.parent / "dashboard" / "static"

        # ---- /dashboard → index.html ----------------------------------------
        @router.get("", include_in_schema=False)
        @router.get("/", include_in_schema=False)
        async def dashboard_index() -> Any:
            """Serve the dashboard SPA."""
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(str(index), media_type="text/html")
            return JSONResponse({"detail": "dashboard not found"}, status_code=404)

        # ---- /dashboard/data.json -------------------------------------------
        @router.get("/data.json", include_in_schema=False)
        async def dashboard_data() -> Any:
            """Live metrics + recent runs as JSON consumed by the SPA."""
            return JSONResponse(_build_dashboard_data())

        # ---- /dashboard/spans (SSE) ----------------------------------------
        @router.get("/spans", include_in_schema=False)
        async def dashboard_spans(request: _FastAPIRequest) -> Any:  # type: ignore[valid-type]
            """Server-sent events stream of recent trace spans.

            Emits all currently-buffered spans, then heartbeats while the
            client stays connected. The loop terminates as soon as the client
            disconnects (so it never leaks a worker thread) and is also bounded
            by ``EFFGEN_DASHBOARD_SSE_MAX_SECONDS`` (default 3600s) as a
            backstop — the SPA simply reconnects via ``EventSource``.
            """
            import asyncio
            import json as _json
            import time as _time

            max_seconds = float(os.getenv("EFFGEN_DASHBOARD_SSE_MAX_SECONDS", "3600"))
            heartbeat_s = float(os.getenv("EFFGEN_DASHBOARD_SSE_HEARTBEAT_SECONDS", "5"))

            async def _generate():  # type: ignore[return]
                # Stream any buffered spans first.
                for span in _get_recent_spans():
                    yield f"data: {_json.dumps(span)}\n\n"
                # Heartbeat until the client disconnects or the cap is hit.
                deadline = _time.monotonic() + max_seconds
                while _time.monotonic() < deadline:
                    if await request.is_disconnected():
                        break
                    await asyncio.sleep(heartbeat_s)
                    yield ": heartbeat\n\n"

            return StreamingResponse(
                _generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # ---- /dashboard/{file} (static assets) ------------------------------
        @router.get("/{asset_path:path}", include_in_schema=False)
        async def dashboard_static(asset_path: str) -> Any:
            """Serve static assets (app.js, style.css, …)."""
            asset = static_dir / asset_path
            if asset.exists() and asset.is_file():
                return FileResponse(str(asset))
            return JSONResponse({"detail": "not found"}, status_code=404)

        app.include_router(router)
        logger.info("Mounted dashboard at /dashboard")

    except Exception as exc:  # noqa: BLE001
        logger.warning("Dashboard not mounted: %s", exc)


def _build_dashboard_data() -> dict[str, Any]:
    """Assemble the JSON payload served at /dashboard/data.json."""
    import time

    raw_metrics, samples = _collect_dashboard_metrics()

    # --- Derive summary metrics from raw ---
    model_call_count = _sum_samples(samples, "effgen_model_call_latency_seconds_count")
    model_call_errors = _sum_samples(
        samples,
        "effgen_model_call_latency_seconds_count",
        lambda labels: labels.get("outcome") not in (None, "", "ok"),
    )
    legacy_requests = _sum_samples(samples, "effgen_requests_total")
    legacy_errors = _sum_samples(samples, "effgen_errors_total")

    total_requests = model_call_count or legacy_requests
    total_errors = model_call_errors or legacy_errors

    latency_sum = _sum_samples(samples, "effgen_model_call_latency_seconds_sum")
    latency_count = model_call_count
    if not latency_count:
        latency_sum = _sum_samples(samples, "effgen_response_latency_seconds_sum")
        latency_count = _sum_samples(samples, "effgen_response_latency_seconds_count")
    avg_latency_s = (latency_sum / latency_count) if latency_count else None

    # Token / cost estimates
    model_tokens = _sum_samples(samples, "effgen_tokens_total")
    legacy_tokens = _sum_samples(samples, "effgen_tokens_used_total")
    total_tokens = model_tokens or legacy_tokens
    # $0.01 per request as rough estimate (matches budget middleware default)
    PER_CALL_COST = 0.01
    daily_cost_usd = total_requests * PER_CALL_COST

    # --- SLO burn rates (simplified: burn = current_rate / target_rate) ---
    LATENCY_THRESHOLD = 2.0  # seconds — p99 target
    ERROR_RATE_TARGET = 0.01  # 1% errors allowed
    error_rate = (total_errors / total_requests) if total_requests > 0 else 0.0
    availability = 1.0 - error_rate

    slo: dict[str, float] = {
        "p99_latency_burn": (avg_latency_s / LATENCY_THRESHOLD) if avg_latency_s else 0.0,
        "error_rate_burn": error_rate / ERROR_RATE_TARGET if ERROR_RATE_TARGET > 0 else 0.0,
        "availability": availability,
    }

    # --- Recent agent runs (from in-memory ring buffer if available) ---
    recent_runs = _get_recent_runs()

    # --- Recent spans ---
    recent_spans = _get_recent_spans()

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": {
            "total_requests": int(total_requests),
            "total_errors": int(total_errors),
            "avg_latency_s": round(avg_latency_s, 4) if avg_latency_s is not None else None,
            "total_tokens": int(total_tokens),
            "daily_cost_usd": round(daily_cost_usd, 6),
        },
        # ``slo``/``recent_spans`` are the canonical keys consumed by the SPA;
        # ``slos``/``spans`` are documented aliases so external consumers can use
        # either spelling.
        "slo": slo,
        "slos": slo,
        "recent_runs": recent_runs,
        "recent_spans": recent_spans[:20],
        "spans": recent_spans[:20],
        "prompt_templates": _get_prompt_templates(),
        "raw_metrics": dict(sorted(raw_metrics.items())),
    }


_PROM_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


def _collect_dashboard_metrics() -> tuple[dict[str, float], list[tuple[str, dict[str, str], float]]]:
    """Collect prometheus_client and effGen-native metric samples."""
    raw_metrics: dict[str, float] = {}
    samples: list[tuple[str, dict[str, str], float]] = []

    try:
        from prometheus_client import REGISTRY

        for metric in REGISTRY.collect():
            for sample in metric.samples:
                labels = {str(k): str(v) for k, v in dict(sample.labels).items()}
                value = float(sample.value)
                samples.append((sample.name, labels, value))
                raw_metrics[_metric_key(sample.name, labels)] = value
    except Exception:  # noqa: BLE001
        pass

    try:
        from effgen.observability.metrics import export_metrics

        for name, labels, value in _parse_prometheus_text(export_metrics()):
            samples.append((name, labels, value))
            raw_metrics[_metric_key(name, labels)] = value
    except Exception:  # noqa: BLE001
        pass

    return raw_metrics, samples


def _parse_prometheus_text(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse simple Prometheus text-format sample lines."""
    parsed: list[tuple[str, dict[str, str], float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE_RE.match(line)
        if not match:
            continue
        parsed.append(
            (
                match.group("name"),
                _parse_prometheus_labels(match.group("labels") or ""),
                float(match.group("value")),
            )
        )
    return parsed


def _parse_prometheus_labels(label_text: str) -> dict[str, str]:
    """Parse a Prometheus label set emitted by effGen's metric exporter."""
    labels: dict[str, str] = {}
    if not label_text:
        return labels
    for part in label_text.split(","):
        key, sep, value = part.partition("=")
        if sep:
            labels[key.strip()] = value.strip().strip('"')
    return labels


def _metric_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"


def _sum_samples(
    samples: list[tuple[str, dict[str, str], float]],
    name: str,
    predicate: Any = None,
) -> float:
    total = 0.0
    for sample_name, labels, value in samples:
        if sample_name != name:
            continue
        if predicate is not None and not predicate(labels):
            continue
        total += value
    return total


def _get_prompt_templates() -> list[dict[str, str]]:
    """Expose prompt-library entries for lightweight editor integrations."""
    try:
        from effgen.prompts.library import registry

        templates: list[dict[str, str]] = []
        for prompt in registry.all():
            templates.append(
                {
                    "name": prompt.name,
                    "description": prompt.description,
                    "template": f'registry.get("{prompt.name}").render(...)',
                    "category": prompt.domain,
                }
            )
        return templates
    except Exception:  # noqa: BLE001
        return []


def _get_recent_runs() -> list[dict[str, Any]]:
    """Return up to 50 recent agent runs from the in-memory run log."""
    try:
        from effgen.observability.run_log import get_recent_runs as _runs

        return _runs(limit=50)
    except Exception:  # noqa: BLE001
        return []


def _get_recent_spans() -> list[dict[str, Any]]:
    """Return up to 100 recent trace spans from the in-memory span buffer."""
    try:
        from effgen.observability.tracing import get_recent_spans as _spans

        return _spans(limit=100)
    except Exception:  # noqa: BLE001
        return []
