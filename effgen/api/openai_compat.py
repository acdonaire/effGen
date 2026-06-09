"""OpenAI-compatible API endpoints.

Provides /v1/chat/completions and /v1/completions endpoints matching the
OpenAI REST API spec, so the official `openai` Python client (and any
OpenAI-compatible client) can be pointed at an effGen server unchanged.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
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
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build an OpenAI-format chat.completion response dict."""
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    completion_tokens = _approx_tokens(content or "")
    return {
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
    billing for streamed requests (Audit-2 #47).
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
    finish_reason: str = "stop",
) -> dict[str, Any]:
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
        from fastapi import APIRouter, HTTPException
        from fastapi.responses import StreamingResponse
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "fastapi is required for the OpenAI-compatible API. "
            "Install with `pip install fastapi`."
        ) from e

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
        model = resolve_model_alias(request.model)
        prompt = _messages_to_prompt(request.messages)
        prompt_tokens = _approx_tokens(prompt)

        try:
            result = runner(
                prompt,
                model=model,
                tools=request.tools or [],
                stream=bool(request.stream),
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        if request.stream:
            chat_id = _chat_id()
            include_usage = bool(
                (request.stream_options or {}).get("include_usage")
            )

            def sse_iter():
                first = build_chat_chunk(
                    model, "", chat_id=chat_id, role="assistant"
                )
                yield f"data: {json.dumps(first)}\n\n"
                completion_text_parts: list[str] = []
                try:
                    for chunk in result:
                        text = str(chunk)
                        completion_text_parts.append(text)
                        payload = build_chat_chunk(
                            model, text, chat_id=chat_id
                        )
                        yield f"data: {json.dumps(payload)}\n\n"
                except TypeError:
                    text = str(result)
                    completion_text_parts.append(text)
                    payload = build_chat_chunk(model, text, chat_id=chat_id)
                    yield f"data: {json.dumps(payload)}\n\n"
                final = build_chat_chunk(
                    model, "", chat_id=chat_id, finish_reason="stop"
                )
                yield f"data: {json.dumps(final)}\n\n"
                # Emit a final usage-only chunk when the client opts in via
                # stream_options.include_usage so streamed requests can be
                # reconciled for billing (Audit-2 #47).
                if include_usage:
                    completion_tokens = count_tokens("".join(completion_text_parts))
                    usage_chunk = build_usage_chunk(
                        model,
                        chat_id=chat_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_iter(), media_type="text/event-stream")

        content = result if isinstance(result, str) else "".join(str(c) for c in result)
        return build_chat_completion(
            model, content, prompt_tokens=prompt_tokens
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
            raise HTTPException(status_code=500, detail=str(e))

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

        text = result if isinstance(result, str) else "".join(str(c) for c in result)
        return build_text_completion(model, text, prompt_tokens=prompt_tokens)

    return router
