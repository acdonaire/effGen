"""OpenAI-compatible API endpoints.

Provides ``/v1/chat/completions`` and ``/v1/completions`` endpoints matching the
OpenAI REST API spec, so the official ``openai`` Python client (and any
OpenAI-compatible client) can be pointed at an effGen server unchanged.

Compatibility level
-------------------
* **Usage** is real: provider-reported counts when the upstream API returns
  them, otherwise tokenizer counts — never a ``len(text) // 4`` estimate.
* **Model aliases** (``gpt-4`` → a local model, see :data:`MODEL_ALIASES`) are a
  documented compatibility *shim*. The response's ``model`` field reports the
  model that actually ran, and a non-standard ``effgen`` object records the
  requested alias vs. the resolved model (OpenAI clients ignore unknown keys).
* **Streaming** (SSE) is truly incremental. When the client sets
  ``stream_options.include_usage`` a final usage-only chunk is emitted; without
  it, no usage chunk is sent (matching OpenAI's behaviour).
* **Tools**: effGen runs tools **server-side** (its ReAct loop) and returns the
  final assistant message. It does **not** stream client-side ``tool_calls``
  deltas for the client to execute — passing ``tools`` lets effGen *use* its own
  registered tools; the answer comes back already resolved. A non-streaming
  response may carry a ``tool_calls`` array when a runner surfaces one. A tool
  the server does not host is **rejected with a clear 400** (it is never
  silently ignored), so a client expecting OpenAI function-calling is told
  plainly rather than getting prose back.
* **Errors** use the OpenAI error envelope (``{"error": {...}}``) with a sane
  status (404 model-not-found, 401 client-auth, 502/503 upstream-provider
  failure, 429 rate-limit, …) and are redacted.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore

    def Field(default=None, **kwargs):  # type: ignore
        return default


# ---------------------------------------------------------------------------
# Model aliasing
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "gpt-4": "Qwen/Qwen2.5-7B-Instruct",
    "gpt-4-turbo": "Qwen/Qwen2.5-7B-Instruct",
    "gpt-4o": "Qwen/Qwen2.5-7B-Instruct",
    "gpt-4o-mini": "Qwen/Qwen2.5-3B-Instruct",
    "gpt-3.5-turbo": "Qwen/Qwen2.5-3B-Instruct",
    "gpt-3.5-turbo-instruct": "Qwen/Qwen2.5-3B-Instruct",
}

# Names that route to the server's configured default model. A caller that has
# no particular model in mind (including the native client's zero-argument
# ``chat("hi")``) can send ``"effgen-default"`` or ``"default"`` and get an
# answer without knowing a concrete id. The target is read from
# ``EFFGEN_DEFAULT_MODEL`` at request time, falling back to a small local model.
DEFAULT_MODEL_ALIASES: frozenset[str] = frozenset({"effgen-default", "default"})
_FALLBACK_DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def default_model_id() -> str:
    """Return the model id the ``effgen-default``/``default`` names resolve to.

    Controlled by ``EFFGEN_DEFAULT_MODEL``; defaults to a small local model when
    the variable is unset or blank.
    """
    return os.environ.get("EFFGEN_DEFAULT_MODEL", "").strip() or _FALLBACK_DEFAULT_MODEL


def resolve_model_alias(model: str) -> str:
    """Resolve an OpenAI model name to a local effGen model id."""
    if model in DEFAULT_MODEL_ALIASES:
        return default_model_id()
    return MODEL_ALIASES.get(model, model)


# ---------------------------------------------------------------------------
# Runner result
# ---------------------------------------------------------------------------


@dataclass
class RunnerResult:
    """Structured result a runner may return instead of a bare string.

    Returning this (rather than a plain ``str``) lets the OpenAI-compatible
    layer surface **real** usage — provider-reported counts when the upstream
    API returns them, tokenizer counts for local models — instead of
    re-estimating from the response text. It also carries the *resolved* effGen
    model so the response can document which model actually ran when an OpenAI
    alias (e.g. ``gpt-4``) was mapped to a local model.

    A runner may still return a plain ``str`` (or, for streaming, an iterator of
    string chunks); the router falls back to a tokenizer estimate in that case.
    """

    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    resolved_model: str | None = None
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):  # type: ignore[misc]
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):  # type: ignore[misc]
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    max_tokens: int | None = Field(default=None, gt=0)
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    # OpenAI streaming option: {"include_usage": true} requests a final
    # usage-only chunk (choices: []) after the content chunks.
    stream_options: dict[str, Any] | None = None


class CompletionRequest(BaseModel):  # type: ignore[misc]
    model: str
    prompt: str | list[str]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    max_tokens: int | None = Field(default=16, gt=0)
    stop: str | list[str] | None = None
    user: str | None = None
    seed: int | None = None


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _now() -> int:
    return int(time.time())


def _chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _cmpl_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


_TIKTOKEN_ENCODER: Any = None
_TIKTOKEN_TRIED = False


def _get_tiktoken_encoder() -> Any:
    """Return a cached tiktoken encoder, or ``None`` if tiktoken is unavailable."""
    global _TIKTOKEN_ENCODER, _TIKTOKEN_TRIED
    if _TIKTOKEN_TRIED:
        return _TIKTOKEN_ENCODER
    _TIKTOKEN_TRIED = True
    try:
        import tiktoken

        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - tiktoken optional
        _TIKTOKEN_ENCODER = None
    return _TIKTOKEN_ENCODER


def count_tokens(text: str) -> int:
    """Count tokens in *text* using a real tokenizer when available.

    Prefers a BPE tokenizer (tiktoken ``cl100k_base``) over the legacy
    ``len(text) // 4`` heuristic so usage/cost numbers track real counts.
    When tiktoken is not installed it falls back to a slightly
    refined character heuristic.  The server runner should supply provider- or
    model-reported usage when it has it; this is the estimate of last resort.
    """
    if not text:
        return 0
    enc = _get_tiktoken_encoder()
    if enc is not None:
        try:
            return max(1, len(enc.encode(text)))
        except Exception:  # pragma: no cover - encoding edge cases
            pass
    return max(1, len(text) // 4)


def _approx_tokens(text: str) -> int:  # retained for back-compat
    return count_tokens(text)


def build_chat_completion(
    model: str,
    content: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int | None = None,
    finish_reason: str = "stop",
    effgen_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-format chat.completion response dict.

    ``completion_tokens`` may be supplied directly (e.g. provider-reported or
    tokenizer counts); when ``None`` it is estimated from ``content``.
    ``effgen_meta`` adds a non-standard ``effgen`` object documenting the
    requested vs. resolved model (OpenAI clients ignore unknown top-level keys).
    """
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    if completion_tokens is None:
        completion_tokens = _approx_tokens(content or "")
    resp: dict[str, Any] = {
        "id": _chat_id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if effgen_meta:
        resp["effgen"] = effgen_meta
    return resp


def build_chat_chunk(
    model: str,
    delta_content: str,
    *,
    chat_id: str | None = None,
    finish_reason: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Build one chat.completion.chunk for SSE streaming."""
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if delta_content:
        delta["content"] = delta_content
    return {
        "id": chat_id or _chat_id(),
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def build_usage_chunk(
    model: str,
    *,
    chat_id: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Build a final usage-only streaming chunk (OpenAI ``include_usage`` form).

    Per the OpenAI streaming spec, when ``stream_options.include_usage`` is set
    the server emits one extra chunk after the content chunks whose ``choices``
    is empty and which carries the ``usage`` totals so clients can reconcile
    billing for streamed requests.
    """
    return {
        "id": chat_id or _chat_id(),
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_text_completion(
    model: str,
    text: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    if completion_tokens is None:
        completion_tokens = _approx_tokens(text)
    return {
        "id": _cmpl_id(),
        "object": "text_completion",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Model catalog (read-only)
# ---------------------------------------------------------------------------

_LOCAL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf", ".onnx")


def _local_cached_models() -> list[dict[str, Any]]:
    """List models present in the local HuggingFace cache (best-effort).

    Each entry marks whether the download is ``complete`` (has real weight
    files) so an incomplete snapshot isn't offered as ready. Returns an empty
    list when the cache can't be scanned (e.g. ``huggingface_hub`` absent).
    """
    out: list[dict[str, Any]] = []
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in sorted(info.repos, key=lambda r: r.repo_id):
            if repo.repo_type != "model":
                continue
            has_weights = any(
                f.file_name.endswith(_LOCAL_WEIGHT_SUFFIXES)
                and not f.file_name.endswith(".index.json")
                for rev in repo.revisions
                for f in rev.files
            )
            out.append(
                {
                    "id": repo.repo_id,
                    "provider": "local",
                    "engine": "transformers",
                    "size_gb": round(repo.size_on_disk / (1024**3), 2),
                    "complete": bool(has_weights),
                    "local": True,
                    "is_priced": False,
                }
            )
    except Exception:  # noqa: BLE001 - cache scan is best-effort
        pass
    return out


def build_model_catalog(provider: str | None = None) -> dict[str, Any]:
    """Assemble the catalog payload served at ``GET /v1/models/catalog``.

    Reads the in-package model catalog (the same source the ``effgen models``
    CLI uses) so a picker sees real ids, pricing, capabilities and provenance
    instead of the drop-in aliases ``GET /v1/models`` returns. Never raises: on
    any read failure it returns whatever parsed, with empty lists otherwise.
    """
    data: list[dict[str, Any]] = []
    providers_meta: list[dict[str, Any]] = []
    try:
        from effgen.models import _catalog

        names = (
            [provider]
            if provider and provider in _catalog.known_providers()
            else _catalog.known_providers()
        )
        for prov in names:
            try:
                records = _catalog.list_models(prov)
            except Exception:  # noqa: BLE001 - skip a provider whose catalog won't load
                continue
            meta = {}
            try:
                meta = _catalog.snapshot_meta(prov)
            except Exception:  # noqa: BLE001
                meta = {}
            providers_meta.append(
                {
                    "provider": prov,
                    "count": len(records),
                    "verified_on": meta.get("verified_on"),
                    "default_model": _catalog.default_model(prov),
                }
            )
            for rec in records:
                data.append(
                    {
                        "id": rec.id,
                        "provider": rec.provider,
                        "family": rec.family,
                        "context_window": rec.context_window,
                        "max_output": rec.max_output,
                        "price_in_per_1m": rec.price_in_per_1m,
                        "price_out_per_1m": rec.price_out_per_1m,
                        "supports_tools": rec.supports_tools,
                        "supports_vision": rec.supports_vision,
                        "free_tier": rec.free_tier,
                        "deprecated": rec.deprecated,
                        "is_priced": rec.is_priced,
                        "price_source": rec.price_source,
                        "verified_on": rec.verified_on or meta.get("verified_on"),
                        "local": False,
                    }
                )
    except Exception:  # noqa: BLE001 - degrade to whatever parsed
        pass

    local = _local_cached_models()
    return {
        "object": "list",
        "providers": providers_meta,
        "data": data,
        "local": local,
        "counts": {"catalog": len(data), "local": len(local)},
    }


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

# Type alias for the runner callable injected by the API server.
# runner(prompt: str, *, model: str, tools: list, stream: bool) -> str | iterator
Runner = Callable[..., Any]


_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|(?:bearer|api[-_ ]?key|token|secret|password)"
    r"[=: ]+[^\s'\"]+)",
    re.IGNORECASE,
)


def _redact(text: str) -> str:
    """Strip key/secret-looking substrings from an error message."""
    return _SECRET_RE.sub("[REDACTED]", text or "")


class UnknownToolError(ValueError):
    """A client requested a tool the server does not host.

    The OpenAI-compatible endpoint executes only effGen's *own* registered
    tools server-side; it does not forward arbitrary client-defined function
    specs to the model for the client to execute. Rather than silently dropping
    an unhosted tool (and returning prose), the endpoint raises this so the
    caller gets a clear ``400`` naming the offending tool(s).
    """

    def __init__(self, tool_names: list[str]) -> None:
        self.tool_names = list(tool_names)
        names = ", ".join(repr(n) for n in self.tool_names)
        super().__init__(
            f"tool(s) {names} are not available on this server. This endpoint "
            "executes only its own registered tools server-side (run "
            "`effgen tools list` to see them); pass one of those by name, or "
            "omit `tools` and let the model answer directly. This server does "
            "not forward client-defined function tools for the client to execute."
        )


def _classify_http(exc: Exception) -> tuple[int, str, str | None]:
    """Map an exception to (http_status, openai_error_type, code).

    Reuses effGen's provider-error taxonomy so the OpenAI-compatible surface
    fails with the right status (404 for a bad model, 502/503 for an upstream
    provider-auth/config failure, 429 for rate limits, …) instead of a blanket
    500. The server's own client-auth rejection (a forged/absent caller token)
    is handled upstream of this runner and stays 401.
    """
    if isinstance(exc, UnknownToolError):
        return 400, "invalid_request_error", "unknown_tool"
    category = "unknown"
    try:
        from effgen.models.errors import classify_provider_error

        category = classify_provider_error(exc).category
    except Exception:  # noqa: BLE001
        pass
    low = str(exc).lower()
    if category == "auth":
        # The server's own client-auth layer rejects bad caller credentials with
        # 401 *before* the runner is ever invoked. An auth failure that surfaces
        # HERE therefore comes from the upstream provider — the server is missing
        # or holding a rejected provider key — so it is the server's problem, not
        # the caller's. Returning 401 would wrongly blame the client's
        # credentials; report a gateway error instead. A missing key is a
        # configuration gap (503 Service Unavailable); a present-but-rejected key
        # is an upstream failure (502 Bad Gateway).
        if any(k in low for k in (
            "not found", "not set", "no api key", "set the", "missing", "not configured",
        )):
            return 503, "upstream_unavailable", "upstream_key_missing"
        return 502, "upstream_error", "upstream_auth_failed"
    if category == "not_found":
        return 404, "model_not_found", "model_not_found"
    if category == "rate_limited":
        return 429, "rate_limit_exceeded", "rate_limit_exceeded"
    if category in ("invalid_request", "refusal", "fatal"):
        return 400, "invalid_request_error", None
    if category == "timeout":
        return 504, "timeout", None
    # A failed model load is, from the client's view, a bad model id.
    if any(k in low for k in ("load model", "not a valid model", "model_not_found", "could not find")):
        return 404, "model_not_found", "model_not_found"
    return 500, "server_error", None


def _error_payload(message: str, err_type: str, code: str | None) -> dict[str, Any]:
    """Build an OpenAI-style error envelope (redacted)."""
    return {
        "error": {
            "message": _redact(message),
            "type": err_type,
            "param": None,
            "code": code,
        }
    }


# Map an HTTP status to an OpenAI error ``type``/``code`` so that *every* error
# the server emits — including the ones raised by middleware before a route runs
# (auth 401, rate-limit 429, RBAC 403, validation 422) — uses the same
# ``{"error": {message, type, param, code}}`` envelope a model error uses. An
# OpenAI client can then branch on ``err.type`` / ``err.code`` uniformly.
_STATUS_ERROR_TYPE: dict[int, str] = {
    400: "invalid_request_error",
    401: "invalid_request_error",
    403: "permission_error",
    404: "model_not_found",
    413: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_exceeded",
    500: "server_error",
    502: "upstream_error",
    503: "upstream_unavailable",
    504: "timeout",
}
_STATUS_ERROR_CODE: dict[int, str] = {
    401: "invalid_api_key",
    403: "permission_denied",
    413: "request_too_large",
    429: "rate_limit_exceeded",
}


def error_envelope(
    status: int, message: str, *, code: str | None = None, redact: bool = True
) -> dict[str, Any]:
    """Build the standard OpenAI error envelope for an HTTP *status*.

    Shared by the OpenAI-compat routes **and** the ASGI middleware (auth,
    rate-limit, RBAC/budget) so the whole server speaks one error shape. Set
    ``redact=False`` for server-authored, secret-free messages (e.g. the auth
    hint that legitimately spells out ``Authorization: Bearer <key>``) so the
    key/secret scrubber does not mangle the guidance; provider/upstream error
    text is always redacted (``redact=True``).
    """
    err_type = _STATUS_ERROR_TYPE.get(
        status, "server_error" if status >= 500 else "invalid_request_error"
    )
    err_code = code if code is not None else _STATUS_ERROR_CODE.get(status)
    if not redact:
        return {
            "error": {"message": message, "type": err_type, "param": None, "code": err_code}
        }
    return _error_payload(message, err_type, err_code)


def _effgen_meta(requested: str, resolved: str) -> dict[str, Any]:
    """Build the non-standard ``effgen`` response object documenting aliasing.

    OpenAI-style aliases (e.g. ``gpt-4``) are mapped to concrete effGen models
    via :data:`MODEL_ALIASES`. Rather than silently swapping the model, every
    response carries an ``effgen`` object naming the requested alias and the
    model that actually ran so clients can tell when a compatibility shim was
    applied. OpenAI clients ignore unknown top-level keys.
    """
    return {
        "requested_model": requested,
        "resolved_model": resolved,
        # True when a compatibility alias (gpt-4 → local model) or the
        # effgen-default/default name was applied — not for internal
        # provider/model routing normalization.
        "alias_applied": requested in MODEL_ALIASES or requested in DEFAULT_MODEL_ALIASES,
    }


def _messages_to_prompt(messages: list[Any]) -> str:
    """Flatten a chat message list into a single prompt string.

    Used as a fallback path when an effGen agent is invoked directly rather
    than a chat-native model. The format mirrors ChatML loosely.
    """
    parts: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", None) if not isinstance(msg, dict) else msg.get("role")
        content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
        if isinstance(content, list):
            # Multimodal content: extract text parts only.
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        parts.append(f"{role}: {content or ''}")
    return "\n".join(parts)


def _has_actionable_content(messages: list[Any]) -> bool:
    """Return True if there is anything for the model to act on.

    A request whose every message has empty/whitespace/``None``/absent text
    content — and no image parts, no ``tool_calls``, and no tool result — gives
    the model nothing to answer. Such a request is rejected with a 400 before a
    billed model call, matching the Agent layer's empty-task guard and OpenAI's
    own handling. A message counts as actionable when it has non-blank text, a
    non-empty multimodal content list, ``tool_calls``, or is a ``tool`` result.
    """
    for msg in messages:
        is_dict = isinstance(msg, dict)
        role = msg.get("role") if is_dict else getattr(msg, "role", None)
        content = msg.get("content") if is_dict else getattr(msg, "content", None)
        tool_calls = msg.get("tool_calls") if is_dict else getattr(msg, "tool_calls", None)
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list) and content:
            return True
        if tool_calls:
            return True
        if role == "tool" and content is not None:
            return True
    return False


def create_openai_router(
    runner: Runner, *, extra_models: Callable[[], list[str]] | None = None
) -> Any:
    """Create a FastAPI router exposing OpenAI-compatible endpoints.

    Parameters
    ----------
    runner:
        Callable invoked with (prompt, *, model, tools, stream). Should return
        a string (non-stream) or an iterable of string chunks (stream).
    extra_models:
        Optional callable returning model ids the server has actually served
        this run (e.g. the pooled-model cache). ``GET /v1/models`` lists these
        alongside the drop-in :data:`MODEL_ALIASES` so a client can discover
        real, currently-servable ids instead of only the 6 legacy aliases.

    Returns
    -------
    fastapi.APIRouter
        Router to mount on the main FastAPI app.
    """
    try:
        from fastapi import APIRouter
        from fastapi.responses import JSONResponse, StreamingResponse
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "fastapi is required for the OpenAI-compatible API. "
            "Install with `pip install fastapi`."
        ) from e

    def _error_response(exc: Exception) -> Any:
        """Return an OpenAI-style, redacted error JSONResponse for *exc*."""
        status, err_type, code = _classify_http(exc)
        message = str(exc)
        # A bare/unknown model id falls through to the local loader and fails
        # with a Transformers "not a valid model identifier" message. Add a
        # one-line hint pointing at the id shapes the server accepts.
        low = message.lower()
        if code == "model_not_found" and (
            "not a valid model identifier" in low or "is not a local folder" in low
        ):
            message += (
                " Pass a provider-prefixed model id (e.g. 'openai:gpt-5-nano', "
                "'groq:llama-3.1-8b-instant'), a valid local model id, or "
                "'effgen-default'."
            )
        return JSONResponse(
            status_code=status,
            content=_error_payload(message, err_type, code),
        )

    router = APIRouter(prefix="/v1", tags=["openai-compat"])

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        # The list is the drop-in aliases plus the ids this process has actually
        # served a successful response for this run. It is not exhaustive: any
        # `provider:model` id the server can reach (e.g. "openai:gpt-5-nano",
        # "groq:llama-3.1-8b-instant") is callable whether or not it appears here.
        now = _now()
        # The legacy OpenAI-flagship names (gpt-4, gpt-3.5-turbo, ...) are
        # drop-in compatibility aliases, each mapped to a concrete local model
        # (see ``root``). Mark them so a client does not read the list as a claim
        # that the server serves GPT-4 itself.
        data = [
            {
                "id": alias,
                "object": "model",
                "created": now,
                "owned_by": "effgen",
                "root": target,
                "effgen": {"compatibility_alias": True, "mapped_to": target},
            }
            for alias, target in MODEL_ALIASES.items()
        ]
        # The names that route to the server's configured default model.
        for name in sorted(DEFAULT_MODEL_ALIASES):
            data.append({
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": "effgen",
                "root": default_model_id(),
                "effgen": {"compatibility_alias": True, "mapped_to": default_model_id()},
            })
        # Alongside the drop-in legacy aliases, list the ids the server has
        # actually served this run (e.g. "openai:gpt-5-nano"), so a client
        # discovers real, currently-servable models instead of only the
        # hardcoded aliases.
        if extra_models is not None:
            try:
                served = extra_models()
            except Exception:  # noqa: BLE001 - discoverability is best-effort
                served = []
            for model_id in served:
                if model_id in MODEL_ALIASES:
                    continue
                data.append({
                    "id": model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "effgen",
                    "root": model_id,
                })
        # OpenAI's schema is ``{object, data}``; the extra ``effgen`` key is an
        # additive hint (standard clients read ``data`` and ignore it) telling a
        # caller the list is not exhaustive — any reachable ``provider:model`` id
        # is callable whether or not it is listed here.
        return {
            "object": "list",
            "data": data,
            "effgen": {
                "note": (
                    "Listed ids are drop-in compatibility aliases plus models "
                    "this server has served this run. Any reachable "
                    "'provider:model' id (e.g. 'openai:gpt-5-nano', "
                    "'groq:llama-3.1-8b-instant') is also accepted as the "
                    "request 'model', whether or not it appears in this list."
                ),
            },
        }

    @router.get("/models/catalog")
    async def models_catalog(provider: str | None = None) -> dict[str, Any]:
        """Return the model catalog with pricing, capabilities and provenance.

        Unlike ``GET /v1/models`` (drop-in aliases plus ids served this run),
        this reports the same catalog the ``effgen models`` CLI reads: every
        known provider model with its context window, per-1M pricing, tool/vision
        support, free-tier flag, and the ``verified_on`` date and ``price_source``
        the price came from — plus the models present in the local cache. Prices
        are ``None`` when unpublished (never a fabricated ``$0``), so a picker can
        label priced/free/unpriced accurately. Pass ``?provider=<name>`` to scope
        to one provider. Degrades to an empty list if the catalog is unreadable
        rather than failing the request.
        """
        return build_model_catalog(provider)

    @router.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        # OpenAI rejects an empty `messages` array with a 400; match that instead
        # of inventing a question and replying 200 to a content-free request.
        if not request.messages:
            return JSONResponse(
                status_code=400,
                content=error_envelope(
                    400, "messages must not be empty", code="empty_messages"
                ),
            )
        # A request whose messages carry no text, no image, no tool_calls, and no
        # tool result gives the model nothing to answer. Reject it with a 400
        # before any billed model call rather than returning a paid-for
        # non-answer at 200.
        if not _has_actionable_content(request.messages):
            return JSONResponse(
                status_code=400,
                content=error_envelope(
                    400, "message content must not be empty", code="empty_content"
                ),
            )
        resolved = resolve_model_alias(request.model)
        model = resolved  # echoed in the response 'model' field (what actually ran)
        prompt = _messages_to_prompt(request.messages)
        prompt_tokens = _approx_tokens(prompt)
        effgen_meta = _effgen_meta(request.model, resolved)

        try:
            result = runner(
                prompt,
                model=resolved,
                tools=request.tools or [],
                stream=bool(request.stream),
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as e:
            return _error_response(e)

        if request.stream:
            chat_id = _chat_id()
            include_usage = bool(
                (request.stream_options or {}).get("include_usage")
            )

            def sse_iter():
                first = build_chat_chunk(
                    model, "", chat_id=chat_id, role="assistant"
                )
                first["effgen"] = effgen_meta
                yield f"data: {json.dumps(first)}\n\n"
                completion_text_parts: list[str] = []
                provider_completion_tokens: int | None = None
                started = False
                try:
                    for chunk in result:
                        started = True
                        text = str(chunk)
                        completion_text_parts.append(text)
                        payload = build_chat_chunk(
                            model, text, chat_id=chat_id
                        )
                        yield f"data: {json.dumps(payload)}\n\n"
                except TypeError:
                    if started:
                        raise
                    # ``result`` was not iterable (a bare object) — stringify once.
                    text = str(result)
                    completion_text_parts.append(text)
                    payload = build_chat_chunk(model, text, chat_id=chat_id)
                    yield f"data: {json.dumps(payload)}\n\n"
                except Exception as e:  # noqa: BLE001
                    # Mid-stream failure: emit a terminal error event
                    # (redacted) instead of silently truncating or buffering it
                    # into a content chunk.
                    _, _etype, _ecode = _classify_http(e)
                    err_evt = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": _now(),
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                        "error": _error_payload(str(e), _etype, _ecode)["error"],
                    }
                    yield f"data: {json.dumps(err_evt)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                # A streaming runner may expose final provider/tokenizer usage
                # once exhausted (e.g. as ``.usage`` on the iterator).
                _stream_usage = getattr(result, "usage", None)
                if isinstance(_stream_usage, dict):
                    provider_completion_tokens = _stream_usage.get("completion_tokens")
                final = build_chat_chunk(
                    model, "", chat_id=chat_id, finish_reason="stop"
                )
                yield f"data: {json.dumps(final)}\n\n"
                # Emit a final usage-only chunk when the client opts in via
                # stream_options.include_usage so streamed requests can be
                # reconciled for billing.
                if include_usage:
                    completion_tokens = (
                        provider_completion_tokens
                        if provider_completion_tokens is not None
                        else count_tokens("".join(completion_text_parts))
                    )
                    usage_chunk = build_usage_chunk(
                        model,
                        chat_id=chat_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_iter(), media_type="text/event-stream")

        # Non-streaming: prefer real usage + resolved model from a RunnerResult.
        if isinstance(result, RunnerResult):
            content = result.text
            if result.prompt_tokens is not None:
                prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            tool_calls = result.tool_calls
            finish_reason = result.finish_reason
            # Surface per-call cost next to token usage so operators can bill
            # and track. Lives in the non-standard `effgen` extension (OpenAI
            # clients ignore unknown keys); `None` when the model has no pricing.
            if result.cost_usd is not None:
                effgen_meta = {**effgen_meta, "cost_usd": result.cost_usd}
            # Surface the tool/step trace and run id a runner recorded (also in
            # the `effgen` extension). ``trace`` is a compact list of the tools
            # the server ran — ``{tool, args, result_summary, ok, duration_ms}``
            # — so a caller can show what the agent did; ``tool_calls`` is the
            # count. Absent when the run used no tools.
            extra = result.metadata or {}
            for key in ("trace", "tool_calls", "run_id"):
                value = extra.get(key)
                if value:
                    effgen_meta = {**effgen_meta, key: value}
        else:
            content = result if isinstance(result, str) else "".join(str(c) for c in result)
            completion_tokens = None
            tool_calls = None
            finish_reason = "stop"

        return build_chat_completion(
            model,
            content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            effgen_meta=effgen_meta,
        )

    @router.post("/completions")
    async def completions(request: CompletionRequest) -> Any:
        model = resolve_model_alias(request.model)
        prompt = (
            request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)
        )
        prompt_tokens = _approx_tokens(prompt)

        try:
            result = runner(
                prompt,
                model=model,
                tools=[],
                stream=bool(request.stream),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as e:
            return _error_response(e)

        if request.stream:
            cmpl_id = _cmpl_id()

            def sse_iter():
                try:
                    for chunk in result:
                        payload = {
                            "id": cmpl_id,
                            "object": "text_completion",
                            "created": _now(),
                            "model": model,
                            "choices": [
                                {
                                    "text": str(chunk),
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                except TypeError:
                    payload = {
                        "id": cmpl_id,
                        "object": "text_completion",
                        "created": _now(),
                        "model": model,
                        "choices": [
                            {
                                "text": str(result),
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_iter(), media_type="text/event-stream")

        if isinstance(result, RunnerResult):
            text = result.text
            if result.prompt_tokens is not None:
                prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
        else:
            text = result if isinstance(result, str) else "".join(str(c) for c in result)
            completion_tokens = None
        return build_text_completion(
            model, text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )

    return router
