"""Unit tests for the Agent class."""

from effgen.core.agent import Agent, AgentConfig, AgentMode, AgentResponse
from tests.fixtures.mock_models import MockToolCallingModel


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_config(self, mock_model):
        config = AgentConfig(name="test", model=mock_model)
        assert config.name == "test"
        assert config.max_iterations == 10
        assert config.temperature == 0.7
        assert config.enable_sub_agents is True
        assert config.enable_memory is True
        assert config.enable_streaming is False
        assert config.tools == []
        assert config.enable_fallback is True

    def test_custom_config(self, mock_model):
        config = AgentConfig(
            name="custom",
            model=mock_model,
            max_iterations=3,
            temperature=0.1,
            enable_sub_agents=False,
            enable_memory=False,
        )
        assert config.max_iterations == 3
        assert config.temperature == 0.1
        assert config.enable_sub_agents is False

    def test_memory_config_defaults(self, mock_model):
        config = AgentConfig(name="test", model=mock_model)
        assert "short_term_max_tokens" in config.memory_config
        assert "long_term_backend" in config.memory_config


class TestAgentInit:
    """Tests for Agent initialization."""

    def test_basic_init(self, mock_model):
        config = AgentConfig(name="test", model=mock_model, enable_memory=False, enable_sub_agents=False)
        agent = Agent(config=config)
        assert agent.name == "test"
        assert agent.model is mock_model

    def test_init_with_tools(self, mock_model, calculator, datetime_tool):
        config = AgentConfig(
            name="test",
            model=mock_model,
            tools=[calculator, datetime_tool],
            enable_memory=False,
            enable_sub_agents=False,
        )
        agent = Agent(config=config)
        assert "calculator" in agent.tools
        assert "datetime" in agent.tools

    def test_init_with_string_model_fails_gracefully(self):
        config = AgentConfig(
            name="test",
            model="nonexistent/model",
            require_model=False,
            enable_memory=False,
            enable_sub_agents=False,
        )
        agent = Agent(config=config)
        assert agent.model is None


class TestAgentRun:
    """Tests for Agent.run() method."""

    def test_simple_run(self, basic_agent):
        result = basic_agent.run("What is the answer?")
        assert isinstance(result, AgentResponse)
        assert result.success is True
        assert result.output is not None
        assert len(result.output) > 0

    def test_run_with_tool(self, tool_agent):
        result = tool_agent.run("What is 2 + 2?")
        assert isinstance(result, AgentResponse)
        assert result.success is True

    def test_simple_calculator_task_returns_tool_result_directly(self, calculator):
        model = MockToolCallingModel([
            {
                "thought": "I need to calculate this.",
                "action": "calculator",
                "action_input": '{"expression": "15 * 23"}',
            },
            {
                "thought": "I should do another calculation.",
                "action": "calculator",
                "action_input": '{"expression": "345 + 50"}',
            },
        ])
        agent = Agent(config=AgentConfig(
            name="direct-calc",
            model=model,
            tools=[calculator],
            enable_memory=False,
            enable_sub_agents=False,
        ))

        result = agent.run("What is 15 * 23?")

        assert result.success is True
        assert result.output == "345"
        assert result.tool_calls == 1
        assert model.call_count == 1

    def test_response_has_metadata(self, basic_agent):
        result = basic_agent.run("test")
        assert hasattr(result, "iterations")
        assert hasattr(result, "tool_calls")
        assert hasattr(result, "execution_time")
        assert result.execution_time >= 0

    def test_response_to_dict(self, basic_agent):
        result = basic_agent.run("test")
        d = result.to_dict()
        assert "output" in d
        assert "success" in d
        assert "mode" in d
        assert "iterations" in d


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_default_response(self):
        resp = AgentResponse(output="hello")
        assert resp.output == "hello"
        assert resp.success is True
        assert resp.mode == AgentMode.SINGLE
        assert resp.iterations == 0
        assert resp.tool_calls == 0

    def test_response_to_dict(self):
        resp = AgentResponse(output="test", iterations=3, tool_calls=1)
        d = resp.to_dict()
        assert d["output"] == "test"
        assert d["iterations"] == 3
        assert d["tool_calls"] == 1
