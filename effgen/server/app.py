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
from typing import Any

logger = logging.getLogger(__name__)

# FastAPI Request must be importable at module level so route type annotations
# resolve correctly when `from __future__ import annotations` is active.
try:
    from fastapi import Request as _FastAPIRequest
except ImportError:  # pragma: no cover
    _FastAPIRequest = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    dev_mode: bool | None = None,
    oidc_issuer: str | None = None,
    oidc_client_id: str | None = None,
    oidc_jwks_uri: str | None = None,
    metrics_auth: bool = False,
    cors_origins: list[str] | None = None,
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

    app = FastAPI(
        title="effGen API",
        description="effGen — Efficient Agent Framework API",
        version="0.2.10",
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
        metrics_auth=metrics_auth,
    )

    # 3. Audit middleware (logs every request; reads user from state set by auth)
    from effgen.server.audit import AuditMiddleware

    app.add_middleware(AuditMiddleware)  # type: ignore[arg-type]

    # 4. Production middleware (CORS, gzip, request-ID) — outermost
    try:
        from effgen.api.middleware import install_production_middleware

        install_production_middleware(app, cors_origins=cors_origins or ["*"])
    except ImportError:
        logger.warning("Production middleware not available")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        """Health probe — always public."""
        return {"status": "ok", "version": "0.2.10"}

    @app.get("/metrics", tags=["ops"])
    async def metrics(request: _FastAPIRequest) -> Any:  # type: ignore[valid-type]
        """Prometheus metrics (public by default; set EFFGEN_METRICS_AUTH=1 to protect)."""
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
            from starlette.responses import Response

            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
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
        from effgen.server.rbac import resolve_policy

        user = getattr(request.state, "user", None)
        roles: list[str] = getattr(user, "roles", []) if user else []
        policy = resolve_policy(roles)
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

    return app


def _build_default_runner() -> Any:
    """Construct an agent-backed runner for the OpenAI-compatible endpoints.

    Returns a callable ``runner(prompt, *, model, tools, stream, **kw)`` that
    drives an :class:`~effgen.core.agent.Agent`. The agent (and its model) is
    created lazily per call from the requested model id so the server can serve
    any provider the host environment is configured for.
    """
    def _runner(prompt: str, *, model: str, tools: Any = None, stream: bool = False, **_: Any) -> str:
        from effgen.core.agent import Agent, AgentConfig

        resolved_tools = _resolve_tools(tools)
        config = AgentConfig(
            name="api",
            model=_normalize_model_id(model),
            tools=resolved_tools,
            require_model=True,
        )
        agent = Agent(config)
        response = agent.run(prompt)
        return getattr(response, "output", str(response))

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

        if prefix in ProviderRegistry.list_providers():
            return f"{prefix}:{rest}"
    except Exception:  # noqa: BLE001 - registry optional; fall back to original
        pass
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

    On a permitted request a small per-call estimate (``EFFGEN_PER_CALL_COST_USD``,
    default ``0.01``) is charged against the principal's daily budget.
    """

    _ENFORCED_PATHS = ("/v1/chat/completions", "/v1/completions")

    def __init__(self, app: Any) -> None:
        self.app = app
        self.per_call_cost = float(os.getenv("EFFGEN_PER_CALL_COST_USD", "0.01"))

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") not in self._ENFORCED_PATHS:
            await self.app(scope, receive, send)
            return

        from effgen.server import budget as _budget
        from effgen.server.rbac import resolve_policy

        user = scope.get("state", {}).get("user")
        principal = getattr(user, "sub", "anonymous") if user else "anonymous"
        roles: list[str] = list(getattr(user, "roles", []) if user else [])
        policy = resolve_policy(roles)
        primary_role = roles[0] if roles else (policy.roles[0].name if policy.roles else "anonymous")

        # Buffer the full request body, then replay it to the route.
        chunks: list[bytes] = []
        more = True
        while more:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        raw = b"".join(chunks)

        async def _replay() -> dict[str, Any]:
            return {"type": "http.request", "body": raw, "more_body": False}

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

        try:
            _budget.charge(principal, self.per_call_cost, cap=policy.max_cost_per_day)
        except _budget.BudgetExceeded as exc:
            await _reject_json(send, exc.status_code, str(exc))
            return

        await self.app(scope, _replay, send)


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
