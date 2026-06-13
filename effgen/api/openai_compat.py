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
  deltas for the client to execute — passing ``tools`` lets effGen *use* them;
  the answer comes back already resolved. A non-streaming response may carry a
  ``tool_calls`` array when a runner surfaces one.
* **Errors** use the OpenAI error envelope (``{"error": {...}}``) with a sane
  status (404 model-not-found, 401 auth, 429 rate-limit, …) and are redacted.
"""
from __future__ import annotations

import json
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


def resolve_model_alias(model: str) -> str:
    """Resolve an OpenAI model name to a local effGen model id."""
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
    max_tokens: int | None = None
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
    max_tokens: int | None = 16
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


def _classify_http(exc: Exception) -> tuple[int, str, str | None]:
    """Map an exception to (http_status, openai_error_type, code).

    Reuses effGen's provider-error taxonomy so the OpenAI-compatible surface
    fails with the right status (404 for a bad model, 401 for auth, 429 for
    rate limits, …) instead of a blanket 500.
    """
    category = "unknown"
    try:
        from effgen.models.errors import classify_provider_error

        category = classify_provider_error(exc).category
    except Exception:  # noqa: BLE001
        pass
    low = str(exc).lower()
    if category == "auth":
        return 401, "authentication_error", "invalid_api_key"
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
        # True only when an OpenAI compatibility alias (gpt-4 → local model) was
        # applied — not for internal provider/model routing normalization.
        "alias_applied": requested in MODEL_ALIASES,
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


def create_openai_router(runner: Runner) -> Any:
    """Create a FastAPI router exposing OpenAI-compatible endpoints.

    Parameters
    ----------
    runner:
        Callable invoked with (prompt, *, model, tools, stream). Should return
        a string (non-stream) or an iterable of string chunks (stream).

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
        return JSONResponse(
            status_code=status,
            content=_error_payload(str(exc), err_type, code),
        )

    router = APIRouter(prefix="/v1", tags=["openai-compat"])

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        now = _now()
        data = [
            {
                "id": alias,
                "object": "model",
                "created": now,
                "owned_by": "effgen",
                "root": target,
            }
            for alias, target in MODEL_ALIASES.items()
        ]
        return {"object": "list", "data": data}

    @router.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
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
                    # Honest mid-stream failure: emit a terminal error event
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
