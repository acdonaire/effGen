"""
Live integration tests for Gemini native tools.

Requires GOOGLE_API_KEY.  All tests are skipped automatically if the key is absent.
Run with a real Gemini model (gemini-3.1-flash-lite per project rules).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load from project .env first, then ~/.effgen/.env
load_dotenv(Path(__file__).parents[2] / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SKIP = pytest.mark.skipif(
    not GOOGLE_API_KEY,
    reason="GOOGLE_API_KEY not available — skipping live Gemini native tools tests",
)

MODEL = "gemini-3.1-flash-lite-preview"


@SKIP
def test_google_search_tool_live():
    """GoogleSearchTool returns a non-empty answer about a current event."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.tools.builtin.gemini_native import GoogleSearchTool

    model = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
    model.load()

    agent = Agent(AgentConfig(
        name="gemini-search-agent",
        model=model,
        tools=[GoogleSearchTool()],
    ))

    response = agent.run("What is the latest major AI model released by Google in 2025?")
    assert response.output.strip(), "Expected non-empty answer"
    print("GoogleSearchTool response:", response.output[:300])
    model.unload()


@SKIP
def test_url_context_tool_live():
    """GeminiUrlContextTool can summarize a URL."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.tools.builtin.gemini_native import GeminiUrlContextTool

    model = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
    model.load()

    agent = Agent(AgentConfig(
        name="gemini-url-agent",
        model=model,
        tools=[GeminiUrlContextTool()],
    ))

    response = agent.run(
        "Please summarize what this page is about: https://en.wikipedia.org/wiki/Python_(programming_language)"
    )
    assert response.output.strip(), "Expected non-empty summary"
    assert len(response.output) > 50, "Summary is too short"
    print("GeminiUrlContextTool response:", response.output[:300])
    model.unload()


@SKIP
def test_code_execution_tool_live():
    """GeminiCodeExecutionTool can compute 100! exactly."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.tools.builtin.gemini_native import GeminiCodeExecutionTool

    model = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
    model.load()

    agent = Agent(AgentConfig(
        name="gemini-code-agent",
        model=model,
        tools=[GeminiCodeExecutionTool()],
    ))

    response = agent.run("Compute 100! (100 factorial) exactly.")
    # 100! starts with 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
    FACTORIAL_100 = "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000"
    assert FACTORIAL_100 in response.output, (
        f"Expected 100! value in response, got: {response.output[:200]}"
    )
    print("GeminiCodeExecutionTool 100! response:", response.output[:200])
    model.unload()


@SKIP
def test_parallel_function_calls_live():
    """Parallel function calling: Agent + Calculator on two expressions in one turn."""
    from effgen.core.agent import Agent, AgentConfig
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.tools.builtin.calculator import Calculator

    model = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
    model.load()

    agent = Agent(AgentConfig(
        name="gemini-parallel-agent",
        model=model,
        tools=[Calculator()],
    ))

    # The question asks for two independent calculations — model should use two calls
    response = agent.run("What is 2+3? Also, what is 7*8? Give me both answers.")
    assert response.output.strip(), "Expected non-empty answer"
    # Both results should appear somewhere in the final answer
    assert "5" in response.output, "Expected 2+3=5 in answer"
    assert "56" in response.output, "Expected 7*8=56 in answer"
    print("Parallel function calls response:", response.output[:300])
    model.unload()
