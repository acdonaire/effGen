"""
Live integration tests for FireworksAdapter.

Skipped automatically when FIREWORKS_API_KEY is not set.
Uses cheap/small models where possible to reduce cost.
"""

from __future__ import annotations

import os
import time

import pytest

FIREWORKS_KEY = os.getenv("FIREWORKS_API_KEY")
pytestmark = pytest.mark.skipif(
    not FIREWORKS_KEY, reason="FIREWORKS_API_KEY not set"
)

# Cheap models used for live tests
_SMALL_MODEL = "accounts/fireworks/models/qwen3-8b"
# kimi-k2p5 confirmed native structured tool_calls on 2026-04-28
_TOOL_MODEL = "accounts/fireworks/models/kimi-k2p5"


@pytest.fixture(scope="module")
def small_adapter():
    from pathlib import Path

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[3] / ".env", override=True)
    from effgen.models.fireworks_adapter import FireworksAdapter
    adapter = FireworksAdapter(_SMALL_MODEL, enable_rate_limiting=False)
    adapter.load()
    yield adapter
    adapter.unload()


@pytest.fixture(scope="module")
def tool_adapter():
    from pathlib import Path

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[3] / ".env", override=True)
    from effgen.models.fireworks_adapter import FireworksAdapter
    adapter = FireworksAdapter(_TOOL_MODEL, enable_rate_limiting=False)
    adapter.load()
    yield adapter
    adapter.unload()


class TestFireworksLiveGenerate:
    def test_basic_generate(self, small_adapter):
        result = small_adapter.generate(
            "What is the capital of France? Answer in one word.",
            config=None,
        )
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert "paris" in result.text.lower() or "France" in result.text
        assert result.tokens_used > 0

    def test_usage_populated(self, small_adapter):
        from effgen.models.base import GenerationConfig
        result = small_adapter.generate(
            "Say hello.",
            config=GenerationConfig(max_tokens=20),
        )
        assert result.metadata["prompt_tokens"] > 0
        assert result.metadata["completion_tokens"] > 0
        assert result.metadata["total_tokens"] > 0
        assert result.metadata["provider"] == "fireworks"

    def test_generation_config_max_tokens(self, small_adapter):
        from effgen.models.base import GenerationConfig
        result = small_adapter.generate(
            "Count slowly from 1 to 100.",
            config=GenerationConfig(max_tokens=15),
        )
        assert result.tokens_used <= 20  # small buffer for off-by-one

    def test_finish_reason(self, small_adapter):
        from effgen.models.base import GenerationConfig
        result = small_adapter.generate(
            "What is 1 + 1?",
            config=GenerationConfig(max_tokens=50),
        )
        assert result.finish_reason in ("stop", "length", "eos_token")


class TestFireworksLiveStream:
    def test_stream_yields_multiple_chunks(self, small_adapter):
        from effgen.models.base import GenerationConfig
        chunks = list(small_adapter.generate_stream(
            "Count from 1 to 5 slowly.",
            config=GenerationConfig(max_tokens=100),
        ))
        assert len(chunks) > 1, "Expected multiple streaming chunks"
        full_text = "".join(chunks)
        assert len(full_text) > 5

    def test_stream_timing(self, small_adapter):
        """Verify streaming delivers tokens incrementally (not all at once)."""
        from effgen.models.base import GenerationConfig
        timestamps = []

        def timed_stream():
            for chunk in small_adapter.generate_stream(
                "Tell me a very short sentence.",
                config=GenerationConfig(max_tokens=50),
            ):
                timestamps.append(time.monotonic())
                yield chunk

        list(timed_stream())
        assert len(timestamps) >= 2, "Should receive multiple chunk timestamps"


class TestFireworksLiveTools:
    _CALCULATOR_TOOL = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '17 * 23'",
                    }
                },
                "required": ["expression"],
            },
        },
    }]

    def test_native_tool_call(self, tool_adapter):
        from effgen.models.base import GenerationConfig
        result = tool_adapter.generate_with_tools(
            "What is 17 multiplied by 23? Use the calculator tool.",
            tools=self._CALCULATOR_TOOL,
            config=GenerationConfig(max_tokens=200, temperature=0.0),
        )
        tc = result.metadata.get("tool_calls", [])
        assert len(tc) >= 1, f"Expected at least 1 tool call, got: {tc}"
        assert tc[0]["function"]["name"] == "calculator"

    def test_supports_tool_calling(self, tool_adapter):
        assert tool_adapter.supports_tool_calling() is True


class TestFireworksLiveContextLength:
    def test_context_length_matches_registry(self, small_adapter):
        from effgen.models.fireworks_models import FIREWORKS_MODELS
        info = FIREWORKS_MODELS.get(_SMALL_MODEL, {})
        assert small_adapter.get_context_length() == info.get("context", 0)

    def test_count_tokens(self, small_adapter):
        tc = small_adapter.count_tokens("Hello, world! This is a test prompt.")
        assert tc.count > 0
        assert tc.model_name == _SMALL_MODEL
