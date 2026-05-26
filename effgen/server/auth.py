"""OIDC/JWT authentication for the effGen API server.

Bearer JWT validation on every non-public endpoint.
Configurable via environment variables:

  EFFGEN_OIDC_ISSUER       — OIDC issuer URL (e.g. https://accounts.google.com)
  EFFGEN_OIDC_CLIENT_ID    — Audience claim that JWTs must carry
  EFFGEN_OIDC_JWKS_URI     — JWKS endpoint (auto-discovered if omitted)
  EFFGEN_DEV_MODE          — set to "1" to disable auth (loud warning)

Public endpoints (no JWT required by default):
  /health
  /metrics   (can be protected via EFFGEN_METRICS_AUTH=1)
"""
from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEV_MODE_WARNED = False  # module-level flag so we warn only once per process


def _is_dev_mode() -> bool:
    return os.getenv("EFFGEN_DEV_MODE", "0").strip() == "1"


def _warn_dev_mode() -> None:
    global _DEV_MODE_WARNED
    if not _DEV_MODE_WARNED:
        warnings.warn(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  ⚠  EFFGEN_DEV_MODE=1: AUTHENTICATION IS DISABLED  ⚠       ║\n"
            "║  Never run with EFFGEN_DEV_MODE=1 in production!            ║\n"
            "╚══════════════════════════════════════════════════════════════╝",
            stacklevel=3,
        )
        logger.critical(
            "EFFGEN_DEV_MODE=1 — JWT authentication is DISABLED. "
            "Do NOT use this in production."
        )
        _DEV_MODE_WARNED = True


# ---------------------------------------------------------------------------
# JWKS cache
# ---------------------------------------------------------------------------


@dataclass
class _JWKSCache:
    """Simple in-memory JWKS cache with TTL."""

    keys: dict[str, Any] = field(default_factory=dict)
    fetched_at: float = 0.0
    ttl: float = 3600.0  # 1 hour


_jwks_cache = _JWKSCache()


def _fetch_jwks(jwks_uri: str) -> dict[str, Any]:
    """Fetch JWKS from the IdP.  Returns a dict keyed by ``kid``."""
    now = time.time()
    if _jwks_cache.keys and (now - _jwks_cache.fetched_at) < _jwks_cache.ttl:
        return _jwks_cache.keys

    try:
        import httpx

        resp = httpx.get(jwks_uri, timeout=10.0)
        resp.raise_for_status()
        raw: dict[str, Any] = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise AuthError(f"Failed to fetch JWKS from {jwks_uri}: {exc}") from exc

    keys: dict[str, Any] = {}
    for jwk in raw.get("keys", []):
        kid = jwk.get("kid", "__default__")
        keys[kid] = jwk

    _jwks_cache.keys = keys
    _jwks_cache.fetched_at = now
    return keys


def _discover_jwks_uri(issuer: str) -> str:
    """OIDC discovery: GET {issuer}/.well-known/openid-configuration."""
    discovery_url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        import httpx

        resp = httpx.get(discovery_url, timeout=10.0)
        resp.raise_for_status()
        cfg: dict[str, Any] = resp.json()
        return cfg["jwks_uri"]
    except Exception as exc:  # noqa: BLE001
        raise AuthError(
            f"OIDC discovery failed for issuer '{issuer}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Raised when a JWT cannot be validated."""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TokenPayload:
    """Decoded and verified JWT claims."""

    sub: str
    iss: str
    aud: str | list[str]
    exp: int
    iat: int
    roles: list[str] = field(default_factory=list)
    email: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_claims(cls, claims: dict[str, Any]) -> "TokenPayload":
        aud = claims.get("aud", "")
        roles_raw = claims.get("roles", claims.get("scope", ""))
        if isinstance(roles_raw, str):
            roles = [r for r in roles_raw.split() if r]
        else:
            roles = list(roles_raw)

        known = {"sub", "iss", "aud", "exp", "iat", "roles", "scope", "email"}
        extra = {k: v for k, v in claims.items() if k not in known}

        return cls(
            sub=claims.get("sub", ""),
            iss=claims.get("iss", ""),
            aud=aud,
            exp=int(claims.get("exp", 0)),
            iat=int(claims.get("iat", 0)),
            roles=roles,
            email=claims.get("email", ""),
            extra=extra,
        )


def verify_jwt(
    token: str,
    *,
    issuer: str | None = None,
    client_id: str | None = None,
    jwks_uri: str | None = None,
) -> TokenPayload:
    """Validate a Bearer JWT and return its decoded payload.

    Parameters
    ----------
    token:
        Raw JWT string (without ``Bearer `` prefix).
    issuer:
        Expected ``iss`` claim value.  Falls back to ``EFFGEN_OIDC_ISSUER``.
    client_id:
        Expected ``aud`` claim value.  Falls back to ``EFFGEN_OIDC_CLIENT_ID``.
    jwks_uri:
        JWKS endpoint.  Falls back to ``EFFGEN_OIDC_JWKS_URI`` or OIDC
        discovery via ``issuer``.
    """
    issuer = issuer or os.getenv("EFFGEN_OIDC_ISSUER", "")
    client_id = client_id or os.getenv("EFFGEN_OIDC_CLIENT_ID", "")
    jwks_uri = jwks_uri or os.getenv("EFFGEN_OIDC_JWKS_URI", "")

    try:
        import jwt as pyjwt
        from jwt.algorithms import RSAAlgorithm
    except ImportError as exc:
        raise AuthError("PyJWT is required for JWT validation") from exc

    # Decode header to find key id
    try:
        header = pyjwt.get_unverified_header(token)
    except Exception as exc:
        raise AuthError(f"Malformed JWT header: {exc}") from exc

    if jwks_uri:
        # Fetch public key from JWKS
        kid = header.get("kid", "__default__")
        jwks = _fetch_jwks(jwks_uri)
        jwk = jwks.get(kid) or next(iter(jwks.values()), None)
        if jwk is None:
            raise AuthError(f"No JWK found for kid={kid!r}")

        try:
            public_key = RSAAlgorithm.from_jwk(json.dumps(jwk))
        except Exception as exc:
            raise AuthError(f"Cannot load public key from JWK: {exc}") from exc

        options: dict[str, Any] = {"verify_aud": bool(client_id)}
        decode_kwargs: dict[str, Any] = {
            "key": public_key,
            "algorithms": ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
            "options": options,
        }
        if client_id:
            decode_kwargs["audience"] = client_id
        if issuer:
            decode_kwargs["issuer"] = issuer

        try:
            claims: dict[str, Any] = pyjwt.decode(token, **decode_kwargs)
        except pyjwt.ExpiredSignatureError as exc:
            raise AuthError("JWT has expired") from exc
        except pyjwt.InvalidAudienceError as exc:
            raise AuthError(f"JWT audience mismatch (expected {client_id!r})") from exc
        except pyjwt.InvalidIssuerError as exc:
            raise AuthError(f"JWT issuer mismatch (expected {issuer!r})") from exc
        except pyjwt.PyJWTError as exc:
            raise AuthError(f"JWT validation failed: {exc}") from exc

    elif issuer:
        # Discover JWKS from OIDC configuration
        discovered_jwks_uri = _discover_jwks_uri(issuer)
        return verify_jwt(
            token,
            issuer=issuer,
            client_id=client_id,
            jwks_uri=discovered_jwks_uri,
        )
    else:
        # No JWKS URI and no issuer — decode without signature verification
        # (only allowed in dev mode; callers should check _is_dev_mode() first)
        logger.warning(
            "No JWKS URI or issuer configured — JWT decoded without signature verification"
        )
        try:
            claims = pyjwt.decode(
                token,
                options={"verify_signature": False, "verify_aud": False},
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "HS256"],
            )
        except pyjwt.PyJWTError as exc:
            raise AuthError(f"JWT decode failed: {exc}") from exc

    return TokenPayload.from_claims(claims)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

# Public endpoints that never require auth
_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/health",
    "/healthz",
    "/ready",
    "/livez",
})


def get_auth_dependency():  # noqa: ANN201
    """Return a FastAPI Depends-compatible callable for JWT auth.

    Usage::

        from fastapi import Depends
        from effgen.server.auth import get_auth_dependency

        AuthDep = Annotated[TokenPayload, Depends(get_auth_dependency())]

        @router.get("/secure")
        async def secure(user: AuthDep):
            return {"sub": user.sub}
    """
    try:
        from fastapi import Request
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fastapi is required for get_auth_dependency()") from exc

    _bearer = HTTPBearer(auto_error=False)

    async def _auth_dep(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = None,
    ) -> TokenPayload | None:
        # Public paths are exempt
        if request.url.path in _PUBLIC_PATHS:
            return None

        # Dev-mode bypass
        if _is_dev_mode():
            _warn_dev_mode()
            return TokenPayload(
                sub="dev-user",
                iss="dev",
                aud="dev",
                exp=int(time.time()) + 86400,
                iat=int(time.time()),
                roles=["admin"],
                email="dev@localhost",
            )

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        raw_token = auth_header.removeprefix("Bearer ").strip()
        try:
            payload = verify_jwt(raw_token)
        except AuthError as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        return payload

    return _auth_dep


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------


class AuthMiddleware:
    """ASGI middleware that validates Bearer JWTs on every non-public request.

    Adds ``request.state.user`` (a :class:`TokenPayload`) for downstream
    handlers to consume.  Rejected requests receive a JSON 401.
    """

    def __init__(
        self,
        app: Any,
        *,
        issuer: str | None = None,
        client_id: str | None = None,
        jwks_uri: str | None = None,
        public_paths: frozenset[str] | None = None,
        metrics_auth: bool = False,
    ) -> None:
        self.app = app
        self.issuer = issuer or os.getenv("EFFGEN_OIDC_ISSUER", "")
        self.client_id = client_id or os.getenv("EFFGEN_OIDC_CLIENT_ID", "")
        self.jwks_uri = jwks_uri or os.getenv("EFFGEN_OIDC_JWKS_URI", "")
        self.metrics_auth = metrics_auth or os.getenv("EFFGEN_METRICS_AUTH", "0") == "1"
        _extra = set(public_paths or set())
        _default_public = set(_PUBLIC_PATHS)
        if not self.metrics_auth:
            _default_public.add("/metrics")
        self.public_paths: frozenset[str] = frozenset(_default_public | _extra)

        if _is_dev_mode():
            _warn_dev_mode()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "/")

        if path in self.public_paths or _is_dev_mode():
            if _is_dev_mode():
                scope.setdefault("state", {})["user"] = TokenPayload(
                    sub="dev-user",
                    iss="dev",
                    aud="dev",
                    exp=int(time.time()) + 86400,
                    iat=int(time.time()),
                    roles=["admin"],
                    email="dev@localhost",
                )
            await self.app(scope, receive, send)
            return

        # Extract Authorization header
        headers = dict(scope.get("headers", []))
        auth_bytes: bytes = headers.get(b"authorization", b"")
        auth_str = auth_bytes.decode("latin-1", errors="replace")

        if not auth_str.startswith("Bearer "):
            await self._reject(send, 401, "Missing or invalid Authorization header")
            return

        raw_token = auth_str.removeprefix("Bearer ").strip()
        try:
            payload = verify_jwt(
                raw_token,
                issuer=self.issuer or None,
                client_id=self.client_id or None,
                jwks_uri=self.jwks_uri or None,
            )
        except AuthError as exc:
            await self._reject(send, exc.status_code, str(exc))
            return

        scope.setdefault("state", {})["user"] = payload
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(send: Any, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
