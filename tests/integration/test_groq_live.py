"""Integration tests for GroqAdapter — skipped if GROQ_API_KEY absent, real calls otherwise."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)


def _has_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: GROQ_API_KEY not set")
class TestGroqLive:
    def test_generate_llama31_8b(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("llama-3.1-8b-instant")
        adapter.load()
        try:
            result = adapter.generate("Respond with exactly: GROQ_OK", config=None)
            assert result.text, "Expected non-empty response"
            assert result.tokens_used > 0
            assert result.metadata["provider"] == "groq"
        finally:
            adapter.unload()

    def test_generate_llama33_70b(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("llama-3.3-70b-versatile")
        adapter.load()
        try:
            result = adapter.generate("What is 2 + 2? Answer with just the number.")
            assert "4" in result.text
        finally:
            adapter.unload()

    def test_load_model_via_provider(self):
        from effgen.models import load_model

        model = load_model("llama-3.1-8b-instant", provider="groq")
        try:
            result = model.generate("Say hello in one word")
            assert result.text
        finally:
            model.unload()

    def test_generate_stream_yields_chunks(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("llama-3.1-8b-instant")
        adapter.load()
        try:
            chunks = list(adapter.generate_stream("Count from 1 to 3 briefly."))
            assert len(chunks) >= 1, "Expected at least one streaming chunk"
            full_text = "".join(chunks)
            assert len(full_text) > 0
        finally:
            adapter.unload()

    def test_native_tool_calling(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("llama-3.3-70b-versatile")
        adapter.load()
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "Math expression"}},
                    "required": ["expression"],
                },
            },
        }]
        try:
            result = adapter.generate_with_tools("What is 17 * 23?", tools)
            assert len(result.metadata["tool_calls"]) >= 1
            tc = result.metadata["tool_calls"][0]
            assert tc["function"]["name"] == "calculator"
        finally:
            adapter.unload()

    def test_usage_populated(self):
        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("llama-3.1-8b-instant")
        adapter.load()
        try:
            result = adapter.generate("Hi")
            assert result.metadata["prompt_tokens"] > 0
            assert result.metadata["completion_tokens"] > 0
            assert result.metadata["total_tokens"] > 0
        finally:
            adapter.unload()
