"""Response and usage builders for the OpenAI-compatible API.

Builds the ``chat.completion``, ``chat.completion.chunk``, usage-only chunk and
``text_completion`` payloads, and counts tokens with a real tokenizer when one
is available. Every name is re-exported from ``effgen.api.openai_compat``;
import from there.
"""
from __future__ import annotations

import time
import uuid
from typing import Any


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
    effgen_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a final usage-only streaming chunk (OpenAI ``include_usage`` form).

    Per the OpenAI streaming spec, when ``stream_options.include_usage`` is set
    the server emits one extra chunk after the content chunks whose ``choices``
    is empty and which carries the ``usage`` totals so clients can reconcile
    billing for streamed requests. ``effgen_meta`` adds the same non-standard
    ``effgen`` object the non-streaming response carries — including
    ``cost_usd`` — so a client tallying spend reads one number from the server
    on both paths instead of re-deriving a price from the catalog itself.
    """
    chunk: dict[str, Any] = {
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
    if effgen_meta:
        chunk["effgen"] = effgen_meta
    return chunk


def build_text_completion(
    model: str,
    text: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build a legacy ``text_completion`` response envelope for *text*."""
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
