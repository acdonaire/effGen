"""End-to-end tests for research workflow (no API key required)."""

import pytest

from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import DateTimeTool, TextProcessingTool


@pytest.mark.gpu
class TestResearchWorkflow:
    """Test research-style workflows using free tools."""

    def test_datetime_query(self, real_model):
        agent = Agent(config=AgentConfig(
            name="research_e2e",
            model=real_model,
            tools=[DateTimeTool()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run("What is the current date?", max_tokens=128)
        assert result.success
        assert "202" in result.output or "202" in str(result.execution_trace)

    def test_text_analysis(self, real_model):
        agent = Agent(config=AgentConfig(
            name="text_e2e",
            model=real_model,
            tools=[TextProcessingTool()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run(
            "Count the words in: 'The quick brown fox jumps over the lazy dog'",
            max_tokens=128,
        )
        assert result.success
        assert "9" in result.output or "9" in str(result.execution_trace)
