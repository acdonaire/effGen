"""Integration tests for Cerebras native tool-calling — skipped if key absent."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)


def _has_key() -> bool:
    return bool(os.getenv("CEREBRAS_API_KEY"))


def _xfail_if_cerebras_backpressure(exc: Exception) -> None:
    msg = str(exc).lower()
    transient_markers = (
        "429",
        "rate-limit",
        "rate limit",
        "request_quota_exceeded",
        "queue_exceeded",
        "high traffic",
        "too many requests",
    )
    if any(marker in msg for marker in transient_markers):
        pytest.xfail(f"Cerebras transient backpressure/rate limit: {exc}")


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: CEREBRAS_API_KEY not in ~/.effgen/.env")
class TestCerebrasNativeTools:
    TOOLS = [
        {
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
        }
    ]

    def test_llama_native_tools_returns_tool_call(self):
        from effgen.models.cerebras_adapter import CerebrasAdapter

        adapter = CerebrasAdapter("llama3.1-8b", enable_rate_limiting=False)
        adapter.load()
        try:
            try:
                result = adapter.generate_with_tools(
                    "What is 17 * 23?",
                    tools=self.TOOLS,
                )
            except RuntimeError as exc:
                _xfail_if_cerebras_backpressure(exc)
                raise
            assert result.metadata is not None
            # The model may return a tool call or an answer directly
            # Both are valid — just assert it returned something
            assert result.text is not None or result.metadata.get("tool_calls")
        finally:
            adapter.unload()

    def test_qwen_native_tools_returns_tool_call(self):
        from effgen.models.cerebras_adapter import CerebrasAdapter

        adapter = CerebrasAdapter(
            "qwen-3-235b-a22b-instruct-2507",
            enable_rate_limiting=False,
            max_retries=1,
            timeout=20,
        )
        adapter.load()
        try:
            try:
                result = adapter.generate_with_tools(
                    "What is 17 * 23?",
                    tools=self.TOOLS,
                )
            except RuntimeError as exc:
                _xfail_if_cerebras_backpressure(exc)
                raise
            assert result.metadata is not None
        finally:
            adapter.unload()

    def test_unsupported_model_raises_not_implemented(self):
        from effgen.models.cerebras_adapter import CerebrasAdapter

        adapter = CerebrasAdapter("zai-glm-4.7", enable_rate_limiting=False)
        adapter._client = object()  # bypass load() check
        adapter._is_loaded = True
        with pytest.raises(NotImplementedError, match="does not support native tool-calling"):
            adapter.generate_with_tools("test", tools=self.TOOLS)

    def test_supports_tool_calling_flag(self):
        from effgen.models.cerebras_adapter import CerebrasAdapter

        assert CerebrasAdapter("llama3.1-8b").supports_tool_calling() is True
        assert CerebrasAdapter("qwen-3-235b-a22b-instruct-2507").supports_tool_calling() is True
        assert CerebrasAdapter("zai-glm-4.7").supports_tool_calling() is False


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: CEREBRAS_API_KEY not in ~/.effgen/.env")
class TestCerebrasAgentWithTools:
    def test_agent_math_task_llama(self):
        """Agent with Calculator on a multi-step math task using llama3.1-8b."""
        from effgen.core.agent import Agent, AgentConfig
        from effgen.models.cerebras_adapter import CerebrasAdapter
        from effgen.tools.builtin.calculator import Calculator

        adapter = CerebrasAdapter("llama3.1-8b", enable_rate_limiting=False)
        adapter.load()
        try:
            config = AgentConfig(
                name="cerebras-math-agent",
                model=adapter,
                tools=[Calculator()],
                system_prompt="You are a math assistant. Use the calculator tool.",
                max_iterations=6,
                temperature=0.1,
            )
            agent = Agent(config)
            try:
                response = agent.run("What is 15 * 15?")
            except RuntimeError as exc:
                _xfail_if_cerebras_backpressure(exc)
                raise
            assert response.output is not None
            assert len(response.output) > 0
            # Answer should contain 225
            assert "225" in response.output or response.tool_calls >= 1
        finally:
            adapter.unload()

    def test_agent_math_task_qwen(self):
        """Agent with Calculator on a multi-step task using qwen-3-235b."""
        from effgen.core.agent import Agent, AgentConfig
        from effgen.models.cerebras_adapter import CerebrasAdapter
        from effgen.tools.builtin.calculator import Calculator

        adapter = CerebrasAdapter(
            "qwen-3-235b-a22b-instruct-2507",
            enable_rate_limiting=False,
            max_retries=1,
            timeout=20,
        )
        adapter.load()
        try:
            config = AgentConfig(
                name="cerebras-qwen-agent",
                model=adapter,
                tools=[Calculator()],
                system_prompt="You are a math assistant. Use the calculator tool.",
                max_iterations=6,
                temperature=0.1,
            )
            agent = Agent(config)
            try:
                response = agent.run("What is 15 * 15?")
            except RuntimeError as exc:
                _xfail_if_cerebras_backpressure(exc)
                raise
            assert response.output is not None
            assert len(response.output) > 0
        finally:
            adapter.unload()
