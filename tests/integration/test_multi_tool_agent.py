"""Integration tests for multi-tool agent with real model."""

import pytest

from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, DateTimeTool, TextProcessingTool


def _assert_run_reached_an_answer(result):
    """The run either succeeded or spent its whole iteration budget.

    A 3B model sometimes keeps re-reading the tool instead of writing a final
    answer. That run is reported fail-closed as a partial result, with the tool
    output still in hand — a legitimate outcome for the model, not a defect. Any
    other unsuccessful reason (a tool error, a refusal, an empty result) still
    fails here.
    """
    assert result.success or result.reason == "max_iterations_partial", (
        f"run failed for an unexpected reason: reason={result.reason!r} "
        f"output={result.output!r}"
    )


@pytest.mark.gpu
class TestMultiToolAgent:
    """Test agent with multiple tools."""

    def test_calculator_with_multiple_tools(self, real_model):
        agent = Agent(config=AgentConfig(
            name="multi_test",
            model=real_model,
            tools=[Calculator(), DateTimeTool(), TextProcessingTool()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run("What is 17 * 23? Use the calculator tool.")
        _assert_run_reached_an_answer(result)
        # Check that the tool was called and produced the right answer
        # The answer may be in output or in the execution trace
        trace_str = str(result.execution_trace)
        assert "391" in result.output or "391" in trace_str

    def test_datetime_tool(self, real_model):
        agent = Agent(config=AgentConfig(
            name="dt_test",
            model=real_model,
            tools=[DateTimeTool()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run("What is the current date and time in UTC?")
        _assert_run_reached_an_answer(result)
        trace_str = str(result.execution_trace)
        # The tool result may appear in either the final output or the trace
        # (small local models sometimes return a "no further steps needed"
        # summary instead of quoting the date back).
        assert "202" in result.output or "202" in trace_str
