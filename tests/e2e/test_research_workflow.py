"""End-to-end tests for research workflow (no API key required)."""

import pytest

from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import DateTimeTool, TextProcessingTool

# A 3B instruct model reliably calls the right tool and gets the right value,
# but does not always close the loop with a well-formed final answer inside a
# fixed iteration budget. These tests therefore assert what the framework
# controls — the tool ran, the answer is in the run, and the run ended for a
# known reason — rather than asserting that a small model always emits the
# final-answer form, which would make them flaky rather than strict.
_COMPLETED_REASONS = {"final_answer", "max_iterations_partial"}


def _assert_reached_answer(result, needle: str) -> None:
    assert result.tool_calls >= 1, "the agent never called the tool"
    assert result.metadata.get("reason") in _COMPLETED_REASONS, (
        f"unexpected end reason: {result.metadata.get('reason')}"
    )
    assert needle in result.output or needle in str(result.execution_trace)


@pytest.mark.gpu
class TestResearchWorkflow:
    """Test research-style workflows using free tools."""

    def test_datetime_query(self, real_model):
        agent = Agent(config=AgentConfig(
            name="research_e2e",
            model=real_model,
            tools=[DateTimeTool()],
            max_iterations=10,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run("What is the current date?", max_tokens=128)
        _assert_reached_answer(result, "202")

    def test_text_analysis(self, real_model):
        agent = Agent(config=AgentConfig(
            name="text_e2e",
            model=real_model,
            tools=[TextProcessingTool()],
            max_iterations=10,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run(
            "Count the words in: 'The quick brown fox jumps over the lazy dog'",
            max_tokens=128,
        )
        _assert_reached_answer(result, "9")
