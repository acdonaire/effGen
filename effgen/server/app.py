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
import math
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

# Machine-readable ``code`` for the errors the framework itself raises, so a
# client can branch on a stable string instead of parsing the message.
_HTTP_ERROR_CODES: dict[int, str] = {
    404: "not_found",
    405: "method_not_allowed",
}

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
        except PackageNotFoundError:  # no install metadata (source/editable run); use the fallback version
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
    public_playground: bool | None = None,
    cors_origins: list[str] | None = None,
    rate_limit_per_minute: int | None = None,
    trust_proxy: bool | None = None,
    runner: Any = None,
    extra_models: Any = None,
) -> Any:
    """Create and return the FastAPI application.

    All parameters fall back to environment variables when ``None``.

    Parameters
    ----------
    runner:
        Optional callable ``runner(prompt, *, model, tools, stream, ...)`` used
        by the OpenAI-compatible ``/v1`` endpoints. When ``None`` a default
        agent-backed runner is constructed lazily on first use.
    extra_models:
        Optional callable returning model ids to list in ``GET /v1/models``
        alongside the legacy aliases. When ``None`` this defaults to the ids
        the default runner has actually served a successful response for.
    trust_proxy:
        Whether the per-IP rate limiter should trust the first
        ``X-Forwarded-For`` hop as the client IP. Defaults to ``False`` (the
        raw socket peer is used); enable only when the deployment sits behind
        a reverse proxy that sets/overwrites this header, since any direct
        caller can otherwise set it to bypass the limit.
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
    # The playground drives real, billed model calls, so its local-view opt-in
    # is a flag of its own (not shared with the dashboard's) — an operator must
    # knowingly authorize spend from a page anyone with the URL can reach.
    if public_playground is None:
        public_playground = _dev or _truthy_env("EFFGEN_PUBLIC_PLAYGROUND")

    app = FastAPI(
        title="effGen API",
        description="effGen — Efficient Agent Framework API",
        version=_server_version(),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Every error this app returns — validation (422), a raised HTTPException
    # (including the router's own 404/405), and an unhandled exception (500) —
    # is rendered with the same `{"error": {message, type, param, code}}`
    # envelope the model routes use, so a client branches on `err.type` /
    # `err.code` uniformly instead of switching between that and FastAPI's
    # `{"detail": ...}`.
    try:
        from fastapi.exceptions import RequestValidationError
        from starlette.exceptions import HTTPException as _StarletteHTTPException

        from effgen.api.openai_compat import _classify_http, error_envelope

        @app.exception_handler(RequestValidationError)
        async def _validation_handler(request: Any, exc: RequestValidationError) -> Any:
            errs = exc.errors()
            try:
                first = errs[0]
                loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
                msg = f"{loc}: {first.get('msg')}" if loc else str(first.get("msg"))
            except Exception:  # noqa: BLE001 - fall back to a generic message
                msg = "invalid request body"
            return JSONResponse(
                status_code=422,
                content=error_envelope(422, msg, code="invalid_request_body", redact=False),
            )

        @app.exception_handler(_StarletteHTTPException)
        async def _http_exception_handler(request: Any, exc: Any) -> Any:
            detail = exc.detail if isinstance(exc.detail, str) else "request failed"
            # A 404/405 from the router is about the URL, not about a model, so
            # it is typed as an invalid request. Response headers the exception
            # carries are part of the HTTP contract (``Allow`` on a 405,
            # ``WWW-Authenticate`` on a 401) and survive the re-render.
            return JSONResponse(
                status_code=exc.status_code,
                content=error_envelope(
                    exc.status_code,
                    detail,
                    code=_HTTP_ERROR_CODES.get(exc.status_code),
                    error_type=(
                        "invalid_request_error" if exc.status_code in (404, 405) else None
                    ),
                ),
                headers=getattr(exc, "headers", None),
            )

        @app.exception_handler(Exception)
        async def _unhandled_handler(request: Any, exc: Exception) -> Any:
            logger.exception("Unhandled error serving %s", getattr(request, "url", ""))
            status, _err_type, code = _classify_http(exc)
            if status < 500:
                status, code = 500, None
            # The message is redacted before it leaves the process so an
            # upstream error carrying a key does not reach the client.
            return JSONResponse(
                status_code=status,
                content=error_envelope(status, str(exc) or exc.__class__.__name__, code=code),
            )
    except Exception:  # noqa: BLE001 - error handlers are best-effort
        logger.debug("Could not install error handlers", exc_info=True)

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

    # 1b. Body-size cap for body-accepting routes RBAC/budget doesn't cover
    # (e.g. /v1/embeddings, which needs no RBAC/budget enforcement but must
    # still bound how much a client can make the server buffer).
    app.add_middleware(MaxBodySizeMiddleware)  # type: ignore[arg-type]

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
        public_playground=public_playground,
        dev_mode=_dev,
    )

    # Record the resolved playground settings so the playground bootstrap route
    # can decide whether to expose a local-view session key and report whether
    # spend is authorized without re-reading the environment.
    app.state.public_playground = bool(public_playground)
    app.state.dev_mode = bool(_dev)
    app.state.api_key = api_key if api_key is not None else os.getenv("EFFGEN_API_KEY", "")

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
            trust_proxy=trust_proxy,
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
        """Prometheus metrics (requires auth by default; EFFGEN_PUBLIC_METRICS=1 opens it)."""
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
            from starlette.responses import Response

            payload = generate_latest().decode("utf-8", errors="replace")
            try:
                from effgen.observability.metrics import export_metrics

                payload += "\n" + export_metrics()
            except Exception:  # noqa: BLE001 - metrics export is optional; omit it on failure
                pass
            return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
        except ImportError:
            from effgen.api.openai_compat import error_envelope

            return JSONResponse(
                error_envelope(
                    503,
                    "Metrics are unavailable: prometheus_client is not installed. "
                    "Install it with `pip install 'effgen[server]'`.",
                    code="metrics_unavailable",
                    redact=False,
                ),
                status_code=503,
            )

    @app.get("/slo", tags=["ops"])
    async def slo_status() -> dict[str, Any]:
        """SLO burn-rate status for every registered SLO (public, like /health).

        Lists objectives this process registered via
        ``effgen.observability.slo.get_tracker().register(...)`` and recorded
        events against with ``tracker.record(name, ok=...)``. It does not
        compute latency or availability from request metrics, so it stays empty
        on a server that has served traffic but registered no objective; the
        measured percentiles and availability for that traffic are in the
        ``slo`` block of ``/dashboard/data.json``. An empty list carries a
        ``detail`` note saying so.
        """
        from effgen.observability.slo import EMPTY_SLO_DETAIL, get_tracker

        statuses = get_tracker().all_statuses()
        payload: dict[str, Any] = {"slos": statuses}
        if not statuses:
            payload["detail"] = EMPTY_SLO_DETAIL
        return payload

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
        from effgen.api.openai_compat import error_envelope
        from effgen.server.rbac import PolicyDenied, resolve_policy

        user = getattr(request.state, "user", None)
        roles: list[str] = getattr(user, "roles", []) if user else []
        try:
            policy = resolve_policy(roles)
        except PolicyDenied as exc:
            # Same envelope the RBAC middleware returns when it denies a model
            # call, so both denials read identically to a client.
            return JSONResponse(
                error_envelope(exc.status_code, str(exc), redact=False),
                status_code=exc.status_code,
            )
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
    _mount_existing_routers(app, runner=runner, extra_models=extra_models)

    # Mount the local dashboard (static SPA + data.json + SSE spans).
    _mount_dashboard(app)

    # Mount the in-browser playground (static SPA + bootstrap config).
    _mount_playground(app)

    return app


# ---------------------------------------------------------------------------
# Model pool — reuse loaded models across requests
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

# Ids that produced at least one successful response this run. This drives the
# ``GET /v1/models`` listing so it advertises only models the server has really
# served — not every id that merely loaded a cloud adapter (a nonexistent
# ``provider:model`` can construct an adapter and only 404 on the actual call).
_SERVED_MODEL_IDS: "OrderedDict[str, None]" = OrderedDict()
_SERVED_MODEL_LOCK = threading.Lock()
_SERVED_MODEL_MAX = 64


def _record_served_model(resolved_model: str) -> None:
    """Record that *resolved_model* returned a successful response."""
    if not resolved_model:
        return
    with _SERVED_MODEL_LOCK:
        _SERVED_MODEL_IDS[resolved_model] = None
        _SERVED_MODEL_IDS.move_to_end(resolved_model)
        while len(_SERVED_MODEL_IDS) > _SERVED_MODEL_MAX:
            _SERVED_MODEL_IDS.popitem(last=False)


def _served_model_ids() -> list[str]:
    """Return the ids that have served a successful response this run."""
    with _SERVED_MODEL_LOCK:
        return list(_SERVED_MODEL_IDS.keys())


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


def _extract_cost(response: Any) -> float | None:
    """Pull per-call ``cost_usd`` from an AgentResponse, if the model is priced.

    effGen records the dollar cost of a run in ``metadata["cost_usd"]`` (from the
    provider's pricing table). Returns ``None`` for unpriced/local models so the
    response can omit cost rather than report a misleading zero.
    """
    meta = getattr(response, "metadata", None) or {}
    raw = meta.get("cost_usd")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


_TRACE_ARG_MAX = 200
_TRACE_RESULT_MAX = 200


def _extract_tool_trace(response: Any) -> list[dict[str, Any]]:
    """Summarize the tools a run executed, in call order.

    The agent records each tool call as a start/complete (or failed) pair of
    events in ``response.execution_trace``. This pairs them into a compact
    per-tool summary — ``{tool, args, result_summary, ok, duration_ms}`` — so a
    caller can render what the agent actually did. Returns an empty list when no
    tool ran (a direct model answer).
    """
    trace = getattr(response, "execution_trace", None) or []
    if not isinstance(trace, list):
        return []
    steps: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def _truncate(value: Any, limit: int) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= limit else text[: limit - 1] + "…"

    for event in trace:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        data = event.get("data") or {}
        if etype == "tool_call_start":
            if pending is not None:
                steps.append(pending)  # a start with no matching completion
            pending = {
                "tool": data.get("tool_name") or "tool",
                "args": _truncate(data.get("tool_input"), _TRACE_ARG_MAX),
                "result_summary": None,
                "ok": None,
                "duration_ms": None,
                "_start_ts": event.get("timestamp"),
            }
        elif etype in ("tool_call_complete", "tool_call_failed"):
            step = pending or {
                "tool": data.get("tool_name") or "tool",
                "args": _truncate(data.get("input"), _TRACE_ARG_MAX),
                "_start_ts": None,
            }
            pending = None
            ok = etype == "tool_call_complete"
            if ok:
                step["result_summary"] = _truncate(data.get("result"), _TRACE_RESULT_MAX)
            else:
                step["result_summary"] = _truncate(data.get("error"), _TRACE_RESULT_MAX)
            step["ok"] = ok
            start_ts = step.pop("_start_ts", None)
            end_ts = event.get("timestamp")
            if isinstance(start_ts, int | float) and isinstance(end_ts, int | float):
                step["duration_ms"] = round(max(0.0, (end_ts - start_ts) * 1000), 1)
            step.setdefault("result_summary", None)
            step.setdefault("ok", ok)
            step.setdefault("duration_ms", None)
            steps.append(step)
    if pending is not None:
        pending.pop("_start_ts", None)
        steps.append(pending)
    return steps


class _StreamWithUsage:
    """Token iterator that publishes the run's usage once it is exhausted.

    The OpenAI-compatible route reads ``usage`` after the last token to fill the
    ``stream_options.include_usage`` chunk, so a streamed request reports the
    same token counts and cost a non-streamed one does. Iterating closes the
    ephemeral agent (releasing its memory handles and circuit breakers, but not
    the pooled model, which is shared and stays loaded).
    """

    def __init__(self, agent: Any, prompt: str, resolved_model: str) -> None:
        self._agent = agent
        self._prompt = prompt
        self._resolved_model = resolved_model
        self.usage: dict[str, Any] | None = None

    def __iter__(self) -> Any:
        served = False
        try:
            for chunk in self._agent.stream(self._prompt):
                served = True
                yield chunk
            self.usage = self._agent.last_stream_usage
        finally:
            self._agent.close()
            if served:
                _record_served_model(self._resolved_model)


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
            # Name the served model in the agent's identity so a server-driven
            # run is distinguishable in traces and run history instead of every
            # request reading as the same "api" agent.
            name=f"api:{resolved_model}",
            model=pooled_model,  # pre-loaded instance → no per-request reload
            tools=resolved_tools,
            temperature=temperature if temperature is not None else 0.7,
            require_model=True,
            # A generation failure must reach the route as a typed exception so
            # it maps to a real HTTP status (404/502/503/...) via the same
            # taxonomy every other runner failure uses, instead of returning
            # HTTP 200 with the error text as the answer. ``stream()`` already
            # raises on failure unconditionally, so this only changes run().
            raise_on_error=True,
        )
        agent = Agent(config)

        if stream:
            # Close the ephemeral agent once the stream is exhausted. ``close()``
            # releases the agent's own resources (memory handles, circuit
            # breakers) but NOT the pooled model, which is shared and stays
            # loaded — and it silences the "garbage-collected without close()"
            # warning that would otherwise fire per streamed request.
            return _StreamWithUsage(agent, prompt, resolved_model)

        try:
            # A failure raises here (raise_on_error=True above), so a response
            # that reaches this point is always a success.
            response = agent.run(prompt)
            _record_served_model(resolved_model)
            prompt_tokens, completion_tokens = _extract_usage(response)
            tool_trace = _extract_tool_trace(response)
            run_meta = getattr(response, "metadata", None) or {}
            # Surface which tools ran (and how many) so a caller can render the
            # step trace. Lives in the runner's ``metadata`` and is carried into
            # the response's non-standard ``effgen`` object by the OpenAI-compat
            # layer; standard OpenAI clients ignore unknown keys.
            extra_meta: dict[str, Any] = {
                "tool_calls": int(getattr(response, "tool_calls", 0) or 0),
            }
            if tool_trace:
                extra_meta["trace"] = tool_trace
            run_id = run_meta.get("run_id")
            if run_id:
                extra_meta["run_id"] = run_id
            return RunnerResult(
                text=getattr(response, "output", "") or "",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                resolved_model=resolved_model,
                cost_usd=_extract_cost(response),
                finish_reason="stop",
                metadata=extra_meta,
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
    Names are resolved against the built-in tool registry (RBAC has already
    authorized the names). A requested tool that the server does not host is
    **not** silently dropped — that would leave a client expecting OpenAI
    function-calling with prose and no ``tool_calls``. Instead the unhosted
    names are collected and surfaced as an :class:`UnknownToolError`, which the
    route turns into a clear ``400``.
    """
    if not tools:
        return []
    import asyncio

    from effgen.api.openai_compat import UnknownToolError
    from effgen.tools.registry import ToolRegistry

    registry = ToolRegistry()
    try:
        registry.discover_builtin_tools()
    except Exception:  # noqa: BLE001 - tool discovery is best-effort at startup; server still serves
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
    unresolved: list[str] = []
    for tool in tools:
        if not isinstance(tool, str) and not isinstance(tool, dict):
            # Already a tool object.
            resolved.append(tool)
            continue
        name = _tool_name(tool)
        try:
            obj = _get(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not resolve tool %r: %s", name, exc)
            obj = None
        if obj is not None:
            resolved.append(obj)
        else:
            unresolved.append(name)
    if unresolved:
        raise UnknownToolError(unresolved)
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

        rejected, raw = await _enforce_max_body_size(scope, receive, send, self.max_body_bytes)
        if rejected:
            return

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
    """Emit a JSON error response from an ASGI middleware.

    Uses the shared OpenAI error envelope so RBAC/budget rejections (403/413/429)
    carry the same ``{"error": {message, type, param, code}}`` shape as model
    errors, letting a client branch on ``err.type``/``err.code`` uniformly.
    """
    import json as _json

    from effgen.api.openai_compat import error_envelope

    payload = _json.dumps(error_envelope(status, detail)).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": payload})


async def _enforce_max_body_size(
    scope: Any, receive: Any, send: Any, max_body_bytes: int
) -> tuple[bool, bytes]:
    """Reject an oversized request body; otherwise buffer and return it.

    Checks the declared ``Content-Length`` first (cheap, no buffering), then
    streams the body while counting bytes so an unbounded/chunked body without
    a declared length is still capped. Returns ``(rejected, raw_body)`` — when
    ``rejected`` is ``True`` a 413 has already been sent and the caller must
    not read further from ``receive`` or write to ``send``.
    """
    headers = dict(scope.get("headers", []))
    declared = headers.get(b"content-length")
    if declared is not None:
        try:
            if int(declared) > max_body_bytes:
                await _reject_json(send, 413, "Request body too large")
                return True, b""
        except ValueError:  # non-integer Content-Length; the streamed check still applies
            pass

    chunks: list[bytes] = []
    total = 0
    more = True
    while more:
        msg = await receive()
        chunk = msg.get("body", b"")
        total += len(chunk)
        if total > max_body_bytes:
            await _reject_json(send, 413, "Request body too large")
            return True, b""
        chunks.append(chunk)
        more = msg.get("more_body", False)
    return False, b"".join(chunks)


class MaxBodySizeMiddleware:
    """Pure-ASGI middleware enforcing ``EFFGEN_MAX_BODY_BYTES`` on routes that
    accept a body but are not already covered by :class:`RBACBudgetMiddleware`
    (which enforces the same cap for ``/v1/chat/completions`` and
    ``/v1/completions`` as part of its RBAC/budget replay). Add a path here
    whenever a new body-accepting route is mounted that doesn't need RBAC or
    budget enforcement, so the cap is never an allowlist of just the model
    endpoints.

    Buffers the body (bounded) and replays it once to the route, mirroring
    ``RBACBudgetMiddleware``'s SSE-safe replay-then-park pattern so a
    streaming response downstream isn't starved by a spinning ``receive()``.
    """

    _ENFORCED_PATHS = ("/v1/embeddings",)

    def __init__(self, app: Any) -> None:
        self.app = app
        self.max_body_bytes = int(os.getenv("EFFGEN_MAX_BODY_BYTES", str(10 * 1024 * 1024)))

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") not in self._ENFORCED_PATHS:
            await self.app(scope, receive, send)
            return

        rejected, raw = await _enforce_max_body_size(scope, receive, send, self.max_body_bytes)
        if rejected:
            return

        import asyncio as _asyncio

        _replay_state = {"sent": False}
        _parked = _asyncio.Event()

        async def _replay() -> dict[str, Any]:
            if not _replay_state["sent"]:
                _replay_state["sent"] = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await _parked.wait()  # cancelled when the route's response completes
            return {"type": "http.disconnect"}

        await self.app(scope, _replay, send)


def _mount_existing_routers(
    app: Any, *, runner: Any = None, extra_models: Any = None
) -> None:
    """Mount the OpenAI-compat + embeddings routers."""
    try:
        from effgen.api.openai_compat import create_openai_router

        _runner = runner or _build_default_runner()
        router = create_openai_router(
            _runner, extra_models=extra_models or _served_model_ids
        )
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


def _asset_not_found(name: str) -> Any:
    """Return the shared 404 error envelope for a missing web asset.

    The dashboard and playground answer a missing file with the same
    ``{"error": {message, type, param, code}}`` shape as the ``/v1`` routes, so
    a caller never has to handle two error formats from one server.
    """
    from fastapi.responses import JSONResponse

    from effgen.api.openai_compat import error_envelope

    label = name[:100]
    return JSONResponse(
        error_envelope(
            404,
            f"{label} not found",
            code="not_found",
            error_type="invalid_request_error",
            redact=False,
        ),
        status_code=404,
    )


def _resolve_web_asset(asset_path: str, *roots: Any) -> Any:
    """Resolve a static asset request against the given directories in order.

    The first directory holding the file wins, so a surface can override a
    shared asset with its own copy. The resolved path must stay inside the
    directory it was found in; a request that escapes it (``..`` segments, an
    absolute path, a symlink out of the tree) resolves to ``None``.
    """
    from pathlib import Path as _Path

    if not asset_path or asset_path.startswith("/"):
        return None
    for root in roots:
        base = _Path(root).resolve()
        try:
            candidate = (base / asset_path).resolve()
            candidate.relative_to(base)
        except (ValueError, OSError):
            continue
        if candidate.is_file():
            return candidate
    return None


def _mount_dashboard(app: Any) -> None:
    """Mount the local dashboard SPA at /dashboard.

    Serves:
    - ``GET /dashboard``           — the SPA index.html
    - ``GET /dashboard/data.json`` — live metrics + run data as JSON
    - ``GET /dashboard/spans``     — SSE stream of recent span events
    - ``GET /dashboard/{path}``    — other static assets (JS, CSS, and the
      files shared with the playground)
    """
    try:
        from pathlib import Path as _Path

        from fastapi import APIRouter
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

        router = APIRouter(prefix="/dashboard", tags=["dashboard"])

        static_dir = _Path(__file__).parent.parent / "dashboard" / "static"
        # Assets both web surfaces load (command palette, keyboard layer).
        shared_dir = _Path(__file__).parent.parent / "webui" / "static"

        # ---- /dashboard → index.html ----------------------------------------
        @router.get("", include_in_schema=False)
        @router.get("/", include_in_schema=False)
        async def dashboard_index() -> Any:
            """Serve the dashboard SPA."""
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(str(index), media_type="text/html")
            return _asset_not_found("dashboard")

        # ---- /dashboard/data.json -------------------------------------------
        @router.get("/data.json", include_in_schema=False)
        async def dashboard_data() -> Any:
            """Live metrics + recent runs as JSON consumed by the SPA."""
            return JSONResponse(_build_dashboard_data())

        # ---- /dashboard/catalog.json ----------------------------------------
        @router.get("/catalog.json", include_in_schema=False)
        async def dashboard_catalog(provider: str | None = None) -> Any:
            """Model catalog (pricing, capabilities, provenance) for the SPA.

            Serves the same payload as ``GET /v1/models/catalog`` but under the
            dashboard's own access rule, so the catalog view follows the
            dashboard's local-viewing story (``EFFGEN_PUBLIC_DASHBOARD``) rather
            than requiring a separately-supplied API key.
            """
            from effgen.api.openai_compat import build_model_catalog

            return JSONResponse(build_model_catalog(provider))

        # ---- /dashboard/history.json ----------------------------------------
        @router.get("/history.json", include_in_schema=False)
        async def dashboard_history(
            limit: int = 50,
            status: str | None = None,
            search: str | None = None,
            run_id: str | None = None,
        ) -> Any:
            """Stored run history and saved sessions for the History view.

            Reads the same run store as ``effgen runs`` and the same session
            files as ``effgen sessions``, so the dashboard shows runs recorded
            by the CLI and by other processes, not only this server's traffic.
            Served under the dashboard's own access rule.
            """
            return JSONResponse(_build_dashboard_history(
                limit=limit, status=status, search=search, run_id=run_id,
            ))

        # ---- /dashboard/topology.json ---------------------------------------
        @router.get("/topology.json", include_in_schema=False)
        async def dashboard_topology(limit: int = 6) -> Any:
            """Recent team and workflow executions as node-link graphs.

            Built from the durable run store plus the buffered spans, so a team
            or workflow run from a script or the CLI is included, not only work
            done inside this server process. Served under the dashboard's own
            access rule.
            """
            from effgen.observability.topology import build_topology

            return JSONResponse(build_topology(limit=max(1, min(int(limit or 6), 20))))

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
            """Serve static assets (app.js, style.css, the shared webui files)."""
            asset = _resolve_web_asset(asset_path, static_dir, shared_dir)
            if asset is not None:
                return FileResponse(str(asset))
            return _asset_not_found(asset_path)

        app.include_router(router)
        logger.info("Mounted dashboard at /dashboard")

    except Exception as exc:  # noqa: BLE001
        logger.warning("Dashboard not mounted: %s", exc)


def _mount_playground(app: Any) -> None:
    """Mount the in-browser playground SPA at /playground.

    Serves:
    - ``GET /playground``            — the SPA index.html
    - ``GET /playground/bootstrap``  — presets, tool options, defaults, and (in
      local-view mode) a session key, as JSON
    - ``GET /playground/{path}``     — other static assets (JS, CSS, and the
      files shared with the dashboard)

    The Run button drives the existing ``POST /v1/chat/completions`` endpoint;
    the playground adds no new model-execution path.
    """
    try:
        from pathlib import Path as _Path

        from fastapi import APIRouter
        from fastapi.responses import FileResponse, JSONResponse

        router = APIRouter(prefix="/playground", tags=["playground"])
        static_dir = _Path(__file__).parent.parent / "playground" / "static"
        shared_dir = _Path(__file__).parent.parent / "webui" / "static"

        @router.get("", include_in_schema=False)
        @router.get("/", include_in_schema=False)
        async def playground_index() -> Any:
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(str(index), media_type="text/html")
            return _asset_not_found("playground")

        @router.get("/bootstrap", include_in_schema=False)
        async def playground_bootstrap(request: _FastAPIRequest) -> Any:  # type: ignore[valid-type]
            """Config the page needs to render the pickers and (optionally) run."""
            return JSONResponse(_build_playground_bootstrap(request))

        @router.get("/{asset_path:path}", include_in_schema=False)
        async def playground_static(asset_path: str) -> Any:
            asset = _resolve_web_asset(asset_path, static_dir, shared_dir)
            if asset is not None:
                return FileResponse(str(asset))
            return _asset_not_found(asset_path)

        app.include_router(router)
        logger.info("Mounted playground at /playground")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Playground not mounted: %s", exc)


# Tool names offered as one-click attachments in the playground. Kept to
# credential-free, side-effect-free tools so a first run works with no setup;
# a preset the user selects may pre-fill its own tools on top of these.
_PLAYGROUND_SAFE_TOOLS: tuple[str, ...] = ("calculator", "wikipedia")


def _build_playground_bootstrap(request: Any) -> dict[str, Any]:
    """Assemble the JSON served at /playground/bootstrap.

    Carries the presets (name + system prompt + tools, applied client-side), the
    always-safe tool options, sensible defaults, and — only when the operator
    opted into a public playground and a static key is configured — a session
    key so a local-view demo can Run without pasting one. When spend is not
    authorized locally the key is ``None`` and the page prompts for one.
    """
    app_state = getattr(request, "app", None)
    state = getattr(app_state, "state", None)
    public_playground = bool(getattr(state, "public_playground", False))
    dev_mode = bool(getattr(state, "dev_mode", False))
    configured_key = getattr(state, "api_key", "") or ""

    # Only hand the browser a key when the operator has authorized local spend
    # (public playground) AND a static key exists. In dev mode auth is bypassed,
    # so no key is needed for Run to work.
    session_key = configured_key if (public_playground and configured_key) else None

    presets: list[dict[str, Any]] = []
    try:
        from effgen.presets.registry import list_presets as _list_presets

        for name in _list_presets():
            try:
                from effgen.presets.registry import get_preset as _get_preset

                cfg = _get_preset(name)
            except Exception:  # noqa: BLE001 - skip a preset that won't load
                continue
            presets.append(
                {
                    "name": getattr(cfg, "name", name),
                    "description": getattr(cfg, "description", ""),
                    "system_prompt": getattr(cfg, "system_prompt", "") or "",
                    "tools": list(getattr(cfg, "tool_names", []) or []),
                    "temperature": getattr(cfg, "temperature", None),
                }
            )
    except Exception:  # noqa: BLE001 - presets are optional; page still works without them
        presets = []

    from effgen.api.openai_compat import default_model_id

    return {
        "version": _server_version(),
        "presets": presets,
        "tools": list(_PLAYGROUND_SAFE_TOOLS),
        "default_model": default_model_id(),
        "spend_authorized": bool(session_key) or dev_mode,
        "session_key": session_key,
        "dev_mode": dev_mode,
        # Client-side spend guardrails the page applies by default.
        "defaults": {"max_tokens": 512, "temperature": 0.7, "stream": True},
        "catalog_url": "/v1/models/catalog",
    }


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

    # Token count
    model_tokens = _sum_samples(samples, "effgen_tokens_total")
    legacy_tokens = _sum_samples(samples, "effgen_tokens_used_total")
    total_tokens = model_tokens or legacy_tokens

    # --- Recent agent runs (from in-memory ring buffer if available) ---
    recent_runs = _get_recent_runs()

    # Cost is the sum of the real per-run ``cost_usd`` recorded for each run;
    # runs on unpriced models (or failed before pricing) contribute nothing and
    # are counted separately so the figure is never inflated by a flat estimate.
    priced_cost = 0.0
    priced_runs = 0
    unpriced_runs = 0
    for run in recent_runs:
        cost = run.get("cost_usd")
        if isinstance(cost, int | float):
            priced_cost += float(cost)
            priced_runs += 1
        else:
            unpriced_runs += 1
    session_cost_usd = round(priced_cost, 6) if priced_runs else None

    # --- Latency percentiles from the model-call histogram buckets ---
    percentiles = _latency_percentiles(samples)

    # --- SLO burn rates (burn = current / target; p99 burn uses the true p99) ---
    LATENCY_THRESHOLD = 2.0  # seconds — p99 target
    ERROR_RATE_TARGET = 0.01  # 1% errors allowed
    error_rate = (total_errors / total_requests) if total_requests > 0 else 0.0
    availability = 1.0 - error_rate
    p99_latency_s = percentiles.get("p99")

    slo: dict[str, float] = {
        # Burn rate now derives from the true p99 (falls back to the mean only
        # when no histogram buckets have been recorded yet).
        "p99_latency_burn": (
            (p99_latency_s / LATENCY_THRESHOLD)
            if p99_latency_s is not None
            else (avg_latency_s / LATENCY_THRESHOLD) if avg_latency_s else 0.0
        ),
        "error_rate_burn": error_rate / ERROR_RATE_TARGET if ERROR_RATE_TARGET > 0 else 0.0,
        "availability": availability,
        "latency_threshold_s": LATENCY_THRESHOLD,
        "p50_latency_s": percentiles.get("p50"),
        "p95_latency_s": percentiles.get("p95"),
        "p99_latency_s": p99_latency_s,
    }

    # --- HTTP responses by status code (from the request counter) ---
    by_status, http_client_errors, http_server_errors = _http_status_breakdown(samples)

    # --- Per-model / per-provider breakdown ---
    by_model = _model_breakdown(samples, recent_runs)

    # --- Recent spans ---
    recent_spans = _get_recent_spans()

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": _effgen_version(),
        "metrics": {
            "total_requests": int(total_requests),
            "total_errors": int(total_errors),
            "avg_latency_s": round(avg_latency_s, 4) if avg_latency_s is not None else None,
            "total_tokens": int(total_tokens),
            # Real session cost (sum of per-run ``cost_usd``); ``None`` when no
            # run had a priced model. ``daily_cost_usd`` mirrors it for consumers
            # of the older key name — both are the true summed cost, not an
            # estimate.
            "cost_usd": session_cost_usd,
            "daily_cost_usd": session_cost_usd,
            "priced_runs": priced_runs,
            "unpriced_runs": unpriced_runs,
            "http_client_errors": http_client_errors,
            "http_server_errors": http_server_errors,
        },
        # ``slo``/``recent_spans`` are the canonical keys consumed by the SPA;
        # ``slos``/``spans`` are documented aliases so external consumers can use
        # either spelling.
        "slo": slo,
        "slos": slo,
        "by_model": by_model,
        "by_status": by_status,
        "recent_runs": recent_runs,
        "recent_spans": recent_spans[:20],
        "spans": recent_spans[:20],
        "prompt_templates": _get_prompt_templates(),
        "raw_metrics": dict(sorted(raw_metrics.items())),
    }


def _effgen_version() -> str:
    """Return the running effGen version string (best-effort)."""
    try:
        from effgen import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return ""


def _histogram_quantile(bounds: list[tuple[float, float]], quantile: float) -> float | None:
    """Estimate a quantile from cumulative histogram buckets.

    ``bounds`` is a list of ``(upper_bound, cumulative_count)`` pairs sorted by
    upper bound ascending (the final bound is ``+Inf``). Linear interpolation
    within the matching bucket mirrors Prometheus' ``histogram_quantile``.
    Returns ``None`` when there is no observed count.
    """
    if not bounds:
        return None
    total = bounds[-1][1]
    if total <= 0:
        return None
    rank = quantile * total
    prev_bound = 0.0
    prev_count = 0.0
    for upper, cum in bounds:
        if cum >= rank:
            if upper == math.inf:
                return prev_bound if prev_bound > 0 else None
            if cum == prev_count:
                return upper
            frac = (rank - prev_count) / (cum - prev_count)
            return prev_bound + frac * (upper - prev_bound)
        prev_bound, prev_count = upper, cum
    return prev_bound


def _bucket_bounds(
    samples: list[tuple[str, dict[str, str], float]],
    predicate: Any = None,
) -> list[tuple[float, float]]:
    """Aggregate ``model_call_latency`` histogram buckets into sorted bounds."""
    acc: dict[float, float] = {}
    for name, labels, value in samples:
        if name != "effgen_model_call_latency_seconds_bucket":
            continue
        if predicate is not None and not predicate(labels):
            continue
        le_raw = labels.get("le", "")
        try:
            le = math.inf if le_raw in ("+Inf", "Inf", "inf") else float(le_raw)
        except (TypeError, ValueError):
            continue
        acc[le] = acc.get(le, 0.0) + value
    return sorted(acc.items(), key=lambda kv: kv[0])


def _latency_percentiles(
    samples: list[tuple[str, dict[str, str], float]],
) -> dict[str, float | None]:
    """Compute p50/p95/p99 latency (seconds) across all model-call buckets."""
    bounds = _bucket_bounds(samples)
    out: dict[str, float | None] = {}
    for label, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        val = _histogram_quantile(bounds, q)
        out[label] = round(val, 4) if val is not None else None
    return out


def _http_status_breakdown(
    samples: list[tuple[str, dict[str, str], float]],
) -> tuple[dict[str, int], int, int]:
    """Return ({status: count}, 4xx total, 5xx total) from the request counter."""
    by_status: dict[str, int] = {}
    client_errors = 0
    server_errors = 0
    for name, labels, value in samples:
        if name != "effgen_http_requests_total":
            continue
        status = str(labels.get("status", "")).strip()
        if not status:
            continue
        by_status[status] = by_status.get(status, 0) + int(value)
        if status.startswith("4"):
            client_errors += int(value)
        elif status.startswith("5"):
            server_errors += int(value)
    return dict(sorted(by_status.items())), client_errors, server_errors


def _model_breakdown(
    samples: list[tuple[str, dict[str, str], float]],
    recent_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate calls, error rate, p95 latency, tokens and cost per model."""
    # Cost per model comes from the real per-run cost ledger.
    cost_by_model: dict[str, float] = {}
    for run in recent_runs:
        cost = run.get("cost_usd")
        if isinstance(cost, int | float):
            model = str(run.get("model", "")).split(":")[-1]
            cost_by_model[model] = cost_by_model.get(model, 0.0) + float(cost)

    agg: dict[tuple[str, str], dict[str, Any]] = {}

    def _row(model: str, provider: str) -> dict[str, Any]:
        key = (model, provider)
        if key not in agg:
            agg[key] = {
                "model": model,
                "provider": provider,
                "calls": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        return agg[key]

    for name, labels, value in samples:
        if name != "effgen_model_call_latency_seconds_count":
            continue
        model = labels.get("model", "unknown")
        provider = labels.get("provider", "")
        row = _row(model, provider)
        row["calls"] += int(value)
        if labels.get("outcome") not in (None, "", "ok"):
            row["errors"] += int(value)

    for name, labels, value in samples:
        if name != "effgen_tokens_total":
            continue
        model = labels.get("model", "unknown")
        provider = labels.get("provider", "")
        row = _row(model, provider)
        if labels.get("kind") == "input":
            row["input_tokens"] += int(value)
        elif labels.get("kind") == "output":
            row["output_tokens"] += int(value)

    rows: list[dict[str, Any]] = []
    for (model, _provider), row in agg.items():
        p95 = _histogram_quantile(
            _bucket_bounds(samples, lambda lb, _m=model: lb.get("model") == _m),
            0.95,
        )
        calls = row["calls"]
        rows.append(
            {
                **row,
                "error_rate": round(row["errors"] / calls, 4) if calls else 0.0,
                "p95_latency_s": round(p95, 4) if p95 is not None else None,
                "cost_usd": round(cost_by_model[model], 6) if model in cost_by_model else None,
            }
        )
    rows.sort(key=lambda r: r["calls"], reverse=True)
    return rows


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
    except Exception:  # noqa: BLE001 - best-effort metrics scrape; return what parsed
        pass

    try:
        from effgen.observability.metrics import export_metrics

        for name, labels, value in _parse_prometheus_text(export_metrics()):
            samples.append((name, labels, value))
            raw_metrics[_metric_key(name, labels)] = value
    except Exception:  # noqa: BLE001 - best-effort metrics scrape; return what parsed
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


def _build_dashboard_history(
    *,
    limit: int = 50,
    status: str | None = None,
    search: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the payload served at /dashboard/history.json."""
    limit = max(1, min(int(limit or 50), 500))
    payload: dict[str, Any] = {"runs": [], "sessions": [], "run": None}
    try:
        from effgen.observability import run_log

        if status == "failed":
            status = "error"
        payload["runs"] = run_log.read_runs(limit=limit, status=status, search=search)
        payload["runs_dir"] = str(run_log.history_dir())
        payload["persisted"] = run_log.history_enabled()
        if run_id:
            payload["run"] = run_log.get_run(run_id)
    except Exception:  # noqa: BLE001 - an empty history is a valid view
        logger.debug("Dashboard history: run store unavailable", exc_info=True)
    try:
        from effgen.core.session import SessionManager

        manager = SessionManager()
        sessions, unreadable = manager.scan()
        payload["sessions"] = sessions[:limit]
        payload["unreadable_sessions"] = unreadable
        payload["sessions_dir"] = str(manager.sessions_dir)
    except Exception:  # noqa: BLE001
        logger.debug("Dashboard history: session store unavailable", exc_info=True)
    return payload


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
