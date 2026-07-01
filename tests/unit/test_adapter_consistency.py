"""Provider-adapter consistency contracts.

These tests pin the uniform behaviour every adapter must share regardless of
the provider SDK's native quirks:

* ``finish_reason`` is normalized to one canonical set (OpenAI returns
  ``"stop"``, Anthropic ``"end_turn"``, Gemini ``FinishReason.STOP`` — all must
  collapse to ``"stop"``).
* provider failures carry a structured, **redacted** ``error_context`` with
  ``{provider, model, request_type, retry_status, remediation, category}``.

The finish-reason cases below are *golden fixtures* of the raw values each
provider SDK actually emits, captured so a provider SDK change is caught here
instead of silently leaking a non-canonical token downstream.
"""

from __future__ import annotations

import pytest

from effgen.models._adapter_utils import (
    CANONICAL_FINISH_REASONS,
    build_error_context,
    normalize_finish_reason,
    provider_runtime_error,
)
from effgen.models.errors import (
    RETRY_NON_RETRYABLE,
    RETRY_RATE_LIMITED,
    RETRY_WILL_RETRY,
    InvalidRequestError,
    ModelAuthError,
    ModelNotFoundError,
    ModelTimeoutError,
    ProviderTransientError,
    error_context_dict,
)


class _Enum:
    """Mimics google-genai FinishReason: str(enum) == 'FinishReason.NAME'."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:  # pragma: no cover - exercised indirectly
        return f"FinishReason.{self.name}"


# (raw_value, expected_canonical) — golden fixtures per provider SDK.
FINISH_REASON_CASES = [
    # OpenAI / OpenAI-compatible (openai, groq, cerebras, together, fireworks, hf)
    ("stop", "stop"),
    ("length", "length"),
    ("tool_calls", "tool_calls"),
    ("function_call", "tool_calls"),
    ("content_filter", "content_filter"),
    ("eos_token", "stop"),  # together/fireworks-ism
    # Anthropic stop_reason
    ("end_turn", "stop"),
    ("max_tokens", "length"),
    ("tool_use", "tool_calls"),
    ("stop_sequence", "stop"),
    ("refusal", "content_filter"),
    # Gemini FinishReason (str(enum) form and .name form)
    (_Enum("STOP"), "stop"),
    (_Enum("MAX_TOKENS"), "length"),
    (_Enum("SAFETY"), "content_filter"),
    (_Enum("MALFORMED_FUNCTION_CALL"), "tool_calls"),
    ("FinishReason.STOP", "stop"),
    ("FinishReason.RECITATION", "content_filter"),
    # Gemini numeric enum fallback
    (1, "stop"),
    (2, "length"),
    (3, "content_filter"),
    # local transformers engine
    # missing / unknown
    (None, "stop"),
    ("", "stop"),
]


@pytest.mark.parametrize("raw,expected", FINISH_REASON_CASES)
def test_normalize_finish_reason(raw, expected):
    assert normalize_finish_reason(raw) == expected


@pytest.mark.parametrize("raw,expected", FINISH_REASON_CASES)
def test_normalized_value_is_canonical(raw, expected):
    assert normalize_finish_reason(raw) in CANONICAL_FINISH_REASONS


def test_unknown_finish_reason_is_preserved_lowercased_not_dropped():
    # A novel value we don't map is returned lowercased (never silently lost).
    assert normalize_finish_reason("SOME_NEW_REASON") == "some_new_reason"


def test_bool_is_not_treated_as_int_enum():
    # bool is an int subclass; must not map True->1->"stop" by accident.
    assert normalize_finish_reason(True) == "stop"  # falls back to default


# ---------------------------------------------------------------------------
# Structured, redacted error context
# ---------------------------------------------------------------------------

_REQUIRED_CONTEXT_KEYS = {
    "provider",
    "model",
    "request_type",
    "retry_status",
    "remediation",
    "category",
}


def test_error_context_dict_shape():
    ctx = error_context_dict("openai", "gpt-5-nano", "generate", "auth")
    assert set(ctx) == _REQUIRED_CONTEXT_KEYS
    assert ctx["provider"] == "openai"
    assert ctx["model"] == "gpt-5-nano"
    assert ctx["request_type"] == "generate"
    assert ctx["retry_status"] == RETRY_NON_RETRYABLE
    assert "API key" in ctx["remediation"]


@pytest.mark.parametrize(
    "exc,expected_status",
    [
        (ModelAuthError("openai"), RETRY_NON_RETRYABLE),
        (ModelNotFoundError("openai", "x"), RETRY_NON_RETRYABLE),
        (InvalidRequestError("openai"), RETRY_NON_RETRYABLE),
        (ProviderTransientError("openai", "x", 503), RETRY_WILL_RETRY),
        (ModelTimeoutError("replicate", "x"), RETRY_WILL_RETRY),
    ],
)
def test_build_error_context_classifies_retry_status(exc, expected_status):
    ctx = build_error_context("openai", "x", "generate", exc)
    assert ctx["retry_status"] == expected_status
    assert set(ctx) == _REQUIRED_CONTEXT_KEYS


def test_rate_limit_status():
    err = RuntimeError("HTTP 429 rate limit exceeded")
    ctx = build_error_context("groq", "x", "generate", err)
    assert ctx["retry_status"] == RETRY_RATE_LIMITED


@pytest.mark.parametrize(
    "exc",
    [
        ModelAuthError("openai", "m"),
        ModelNotFoundError("openai", "m"),
        ProviderTransientError("openai", "m", 500),
        ModelTimeoutError("replicate", "m"),
        InvalidRequestError("openai", "m"),
    ],
)
def test_typed_errors_carry_error_context(exc):
    ctx = getattr(exc, "error_context", None)
    assert isinstance(ctx, dict)
    assert set(ctx) == _REQUIRED_CONTEXT_KEYS
    assert ctx["model"] == "m"


def test_provider_runtime_error_is_redacted():
    # A secret leaking into the SDK message must be scrubbed from the wrapped
    # RuntimeError and never reach logs / users.
    secret = "sk-abcdEFGH0123456789abcdEFGH0123456789abcdEF12"
    src = RuntimeError(f"connection error with key {secret}")
    err = provider_runtime_error("openai", "gpt-5-nano", "generate", src)
    assert secret not in str(err)
    assert err.error_context["provider"] == "openai"
    assert err.error_context["request_type"] == "generate"


def test_provider_runtime_error_keeps_classification():
    src = RuntimeError("invalid api key provided")
    err = provider_runtime_error("openai", "m", "generate", src)
    assert err.error_context["category"] == "auth"
    assert err.error_context["retry_status"] == RETRY_NON_RETRYABLE


# ---------------------------------------------------------------------------
# Cost metadata: one canonical per-call key, no duplicate alias
# ---------------------------------------------------------------------------

# ``cost_usd`` (this call's cost) and ``total_cost`` (the adapter's cumulative
# session cost — a genuinely different number) are both legitimate. A bare
# ``"cost"`` key was a third, always-equal-to-cost_usd duplicate with no
# consumer of its own; guard against it creeping back into a metadata dict.
_COST_METADATA_ADAPTER_MODULES = [
    "effgen.models.openai_adapter",
    "effgen.models.anthropic_adapter",
    "effgen.models.gemini_adapter",
]


@pytest.mark.parametrize("module_name", _COST_METADATA_ADAPTER_MODULES)
def test_adapter_source_has_no_redundant_cost_key(module_name):
    import importlib
    import inspect
    import re

    source = inspect.getsource(importlib.import_module(module_name))
    # Matches a literal `"cost": <expr>,` dict entry — not `"cost_usd"` or
    # `"total_cost"`, both of which are intentionally kept.
    assert not re.search(r'"cost"\s*:\s*\w', source), (
        f"{module_name} reintroduced a bare 'cost' metadata key; "
        "cost_usd is the canonical per-call key."
    )
