"""Unit tests for GeminiAdapter helpers (no live API calls)."""

from __future__ import annotations

import time

import pytest

from effgen.models import gemini_models
from effgen.models.gemini_adapter import GeminiAdapter

# ---------------------------------------------------------------------------
# Schema sanitization
# ---------------------------------------------------------------------------

def test_sanitize_schema_strips_unsupported_fields():
    raw = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "expr",
                "minLength": 1,
                "maxLength": 1024,
            },
            "precision": {
                "type": "integer",
                "minimum": 0,
                "maximum": 15,
                "default": 2,
            },
        },
        "required": ["expression"],
        "additionalProperties": False,  # also unsupported by Gemini
    }
    cleaned = GeminiAdapter._sanitize_schema(raw)
    assert cleaned == {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "expr"},
            "precision": {"type": "integer"},
        },
        "required": ["expression"],
    }


def test_sanitize_schema_preserves_property_names():
    """`properties` keys are user-defined names, not schema fields — keep them."""
    raw = {
        "type": "object",
        "properties": {
            "minLength": {"type": "string"},  # property literally named "minLength"
            "type": {"type": "string"},
        },
        "required": ["minLength"],
    }
    cleaned = GeminiAdapter._sanitize_schema(raw)
    assert "minLength" in cleaned["properties"]
    assert "type" in cleaned["properties"]
    assert cleaned["required"] == ["minLength"]


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

def test_convert_tools_to_genai_openai_format():
    pytest.importorskip("google.genai")
    tools = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "math",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "minLength": 1}},
                "required": ["expression"],
            },
        },
    }]
    converted = GeminiAdapter._convert_tools_to_genai(tools)
    assert len(converted) == 1
    fds = converted[0].function_declarations
    assert len(fds) == 1
    assert fds[0].name == "calculator"


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def test_suggested_retry_delay_parses_message_tail():
    class FakeExc(Exception):
        pass

    exc = FakeExc("429 quota exceeded. Please retry in 21.5s")
    assert GeminiAdapter._suggested_retry_delay(exc) == pytest.approx(21.5)


def test_suggested_retry_delay_returns_none_when_absent():
    assert GeminiAdapter._suggested_retry_delay(RuntimeError("boom")) is None


def test_generate_with_retry_honors_retry_delay(monkeypatch):
    """The retry helper should sleep the suggested delay and then succeed."""
    pytest.importorskip("google.api_core.exceptions")
    from google.api_core import exceptions as gax_exc

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.MAX_RATE_LIMIT_RETRIES = 3
    adapter.model_name = "gemini-3.1-flash-lite"

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                exc = gax_exc.ResourceExhausted("Please retry in 7s")
                raise exc
            return "ok"

    class FakeClient:
        models = FakeModels()

    adapter.client = FakeClient()
    out = adapter._generate_with_retry(contents="hi", gen_config=None)
    assert out == "ok"
    assert calls["n"] == 2
    assert sleeps and sleeps[0] >= 7.0  # honored suggested delay


def test_generate_with_retry_gives_up_after_max(monkeypatch):
    pytest.importorskip("google.api_core.exceptions")
    from google.api_core import exceptions as gax_exc

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.MAX_RATE_LIMIT_RETRIES = 2
    adapter.model_name = "gemini-3.1-flash-lite"
    monkeypatch.setattr(time, "sleep", lambda s: None)

    class FakeModels:
        def generate_content(self, **kwargs):
            raise gax_exc.ResourceExhausted("Please retry in 1s")

    class FakeClient:
        models = FakeModels()

    adapter.client = FakeClient()
    with pytest.raises(gax_exc.ResourceExhausted):
        adapter._generate_with_retry(contents="hi", gen_config=None)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_3_1_flash_lite():
    info = gemini_models.model_info("gemini-3.1-flash-lite")
    assert info["canonical_id"] == "gemini-3.1-flash-lite"
    assert info["free_tier"] is True
    assert info["rpm"] == 15
    assert info["supports_native_tools"] is True


def test_registry_free_tier_excludes_pro_preview():
    free = set(gemini_models.free_tier_models())
    assert "gemini-3.1-flash-lite" in free
    assert "gemini-3-pro-preview" not in free
    assert "gemma-3-27b" in free


def test_registry_recommended_models_filters_by_tier():
    free = gemini_models.recommended_models(tier="free")
    assert all(m["tier"] == "free" for m in free)
    assert any(m["id"] == "gemini-3.1-flash-lite" for m in free)


def test_registry_unknown_model_raises():
    with pytest.raises(KeyError):
        gemini_models.model_info("gemini-does-not-exist")
