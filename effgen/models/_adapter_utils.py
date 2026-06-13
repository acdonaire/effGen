"""Shared helpers for provider adapter consistency.

Provider SDKs disagree on how they report two things that effGen treats as a
uniform contract:

* **finish reason** — OpenAI-style adapters return ``"stop"``/``"length"``/
  ``"tool_calls"``; Anthropic returns ``"end_turn"``/``"max_tokens"``/
  ``"tool_use"``; Gemini returns an enum whose ``str()`` is ``"FinishReason.STOP"``.
  :func:`normalize_finish_reason` maps all of them to one canonical set so
  downstream code (the agent loop, cost/observability) sees the same tokens
  regardless of provider.

* **error reporting** — adapters historically raised bare
  ``RuntimeError(f"... failed: {exc}")`` with the unredacted SDK message and no
  machine-readable context. :func:`build_error_context` and
  :func:`provider_runtime_error` produce a consistent, **redacted** error whose
  ``.error_context`` carries ``{provider, model, request_type, retry_status,
  remediation}``.

These are internal helpers (no public API surface change).
"""

from __future__ import annotations

from typing import Any

from .errors import (
    RETRY_NON_RETRYABLE,
    RETRY_RATE_LIMITED,
    RETRY_WILL_RETRY,
    classify_provider_error,
    error_context_dict,
)

# Canonical finish reasons. Keep this set small and OpenAI-flavoured because
# that is what the agent loop and the bulk of adapters already emit.
FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_TOOL_CALLS = "tool_calls"
FINISH_CONTENT_FILTER = "content_filter"
FINISH_ERROR = "error"
FINISH_UNKNOWN = "unknown"

CANONICAL_FINISH_REASONS = frozenset(
    {
        FINISH_STOP,
        FINISH_LENGTH,
        FINISH_TOOL_CALLS,
        FINISH_CONTENT_FILTER,
        FINISH_ERROR,
        FINISH_UNKNOWN,
    }
)

# Raw provider value -> canonical. Keys are lowercased and stripped of any
# ``finishreason.`` / ``stopreason.`` enum prefix before lookup.
_FINISH_REASON_MAP: dict[str, str] = {
    # OpenAI / OpenAI-compatible (groq, cerebras, together, fireworks, hf)
    "stop": FINISH_STOP,
    "length": FINISH_LENGTH,
    "tool_calls": FINISH_TOOL_CALLS,
    "function_call": FINISH_TOOL_CALLS,
    "content_filter": FINISH_CONTENT_FILTER,
    "eos": FINISH_STOP,
    "eos_token": FINISH_STOP,
    "complete": FINISH_STOP,
    "completed": FINISH_STOP,
    # Anthropic stop_reason
    "end_turn": FINISH_STOP,
    "stop_sequence": FINISH_STOP,
    "max_tokens": FINISH_LENGTH,
    "tool_use": FINISH_TOOL_CALLS,
    "pause_turn": FINISH_STOP,
    "refusal": FINISH_CONTENT_FILTER,
    "model_context_window_exceeded": FINISH_LENGTH,
    # Gemini FinishReason (str(enum) -> "finishreason.stop")
    "max_tokens_reached": FINISH_LENGTH,
    "safety": FINISH_CONTENT_FILTER,
    "recitation": FINISH_CONTENT_FILTER,
    "blocklist": FINISH_CONTENT_FILTER,
    "prohibited_content": FINISH_CONTENT_FILTER,
    "spii": FINISH_CONTENT_FILTER,
    "image_safety": FINISH_CONTENT_FILTER,
    "malformed_function_call": FINISH_TOOL_CALLS,
    "unexpected_tool_call": FINISH_TOOL_CALLS,
    "other": FINISH_UNKNOWN,
    "finish_reason_unspecified": FINISH_UNKNOWN,
    "unspecified": FINISH_UNKNOWN,
    # error / cancellation markers used internally
    "error": FINISH_ERROR,
}

# Numeric Gemini FinishReason enum values (google-genai), in case a bare int
# leaks through instead of the named enum.
_GEMINI_FINISH_INT_MAP: dict[int, str] = {
    0: FINISH_UNKNOWN,  # FINISH_REASON_UNSPECIFIED
    1: FINISH_STOP,  # STOP
    2: FINISH_LENGTH,  # MAX_TOKENS
    3: FINISH_CONTENT_FILTER,  # SAFETY
    4: FINISH_CONTENT_FILTER,  # RECITATION
    5: FINISH_UNKNOWN,  # OTHER
    6: FINISH_CONTENT_FILTER,  # BLOCKLIST
    7: FINISH_CONTENT_FILTER,  # PROHIBITED_CONTENT
    8: FINISH_CONTENT_FILTER,  # SPII
    9: FINISH_TOOL_CALLS,  # MALFORMED_FUNCTION_CALL
    10: FINISH_CONTENT_FILTER,  # IMAGE_SAFETY
}


def normalize_finish_reason(raw: Any, *, default: str = FINISH_STOP) -> str:
    """Map any provider finish/stop reason to a canonical lowercase token.

    The canonical set is :data:`CANONICAL_FINISH_REASONS`. Unknown but
    non-empty values are returned lowercased+stripped (so nothing is silently
    lost), while ``None``/empty falls back to ``default`` (a completed
    generation almost always means a normal stop).

    Args:
        raw: The provider's raw finish reason (str, enum, int, or None).
        default: Canonical value to use when ``raw`` is missing/empty.

    Returns:
        A canonical finish-reason string.
    """
    if raw is None:
        return default

    # Enum instances expose ``.name`` (e.g. google-genai FinishReason.STOP).
    name = getattr(raw, "name", None)
    if isinstance(name, str) and name:
        key = name
    elif isinstance(raw, bool):  # guard: bool is an int subclass
        return default
    elif isinstance(raw, int):
        return _GEMINI_FINISH_INT_MAP.get(raw, FINISH_UNKNOWN)
    else:
        key = str(raw)

    key = key.strip().lower()
    if not key:
        return default

    # Strip enum-repr prefixes like "finishreason." / "stopreason.".
    if "." in key:
        key = key.rsplit(".", 1)[-1]

    # A value that was only punctuation/prefix (e.g. "." or "stopreason.")
    # collapses to empty here; never surface an empty finish reason.
    if not key:
        return default

    return _FINISH_REASON_MAP.get(key, key)


# ---------------------------------------------------------------------------
# Structured, redacted provider errors
# ---------------------------------------------------------------------------

def build_error_context(
    provider: str,
    model: str,
    request_type: str,
    exc: Exception,
) -> dict[str, str]:
    """Build the structured ``error_context`` for a provider failure.

    Classifies *exc* and returns ``{provider, model, request_type,
    retry_status, remediation, category}``. The fields are static/derived (no
    secret material), so the dict itself is always safe to log or surface. The
    retry-status + remediation text come from the single source of truth in
    :mod:`effgen.models.errors`.
    """
    cls = classify_provider_error(exc)
    return error_context_dict(provider, model, request_type, cls.category)


def provider_runtime_error(
    provider: str,
    model: str,
    request_type: str,
    exc: Exception,
    *,
    message: str | None = None,
) -> RuntimeError:
    """Build a consistent, **redacted** :class:`RuntimeError` for a provider failure.

    The returned error carries an ``.error_context`` attribute (the dict from
    :func:`build_error_context`) and a message of the form::

        "<provider> <request_type> failed [<retry_status>]: <redacted cause>. <remediation>"

    The underlying SDK message is run through the process redactor so no API
    keys/secrets leak into logs or user-facing output. Callers should
    ``raise provider_runtime_error(...) from exc`` to preserve the traceback.
    """
    # Local import keeps the module import cost low and avoids a hard
    # dependency cycle at import time.
    from ..observability.redact import get_redactor

    ctx = build_error_context(provider, model, request_type, exc)
    redactor = get_redactor()
    cause = redactor.scrub(str(exc)) if str(exc) else exc.__class__.__name__
    head = redactor.scrub(message) if message else f"{provider} {request_type} failed"

    remediation = ctx["remediation"]
    # On a 404 / model_not_found, append the live "did you mean…/available
    # now…" hint so the user sees real alternatives instead of a raw provider
    # 404 — regardless of which adapter path (plain or tool-calling) failed.
    if ctx["category"] == "not_found" and provider and model:
        try:
            from ._catalog import suggest_for_missing

            hint = suggest_for_missing(provider, model)
            if hint:
                remediation = remediation + hint
        except Exception:  # pragma: no cover - suggestion is best-effort
            pass

    err = RuntimeError(
        f"{head} [{ctx['retry_status']}]: {cause}. {remediation}"
    )
    err.error_context = ctx  # type: ignore[attr-defined]
    return err


def attach_error_context(
    err: Exception,
    provider: str,
    model: str,
    request_type: str,
    *,
    source: Exception | None = None,
) -> Exception:
    """Attach a structured ``.error_context`` to an already-typed effGen error.

    Used where an adapter raises a specific typed error (auth/not-found/etc.)
    but we still want the uniform machine-readable context on it. ``source``
    is the original exception used for classification (defaults to ``err``).
    """
    if not hasattr(err, "error_context"):
        err.error_context = build_error_context(  # type: ignore[attr-defined]
            provider, model, request_type, source or err
        )
    return err


__all__ = [
    "CANONICAL_FINISH_REASONS",
    "normalize_finish_reason",
    "build_error_context",
    "provider_runtime_error",
    "attach_error_context",
    "RETRY_WILL_RETRY",
    "RETRY_RATE_LIMITED",
    "RETRY_NON_RETRYABLE",
]
