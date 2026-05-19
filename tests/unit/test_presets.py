"""Tests for preset registry and create_agent factory."""


import pytest

from effgen.presets.registry import (
    PresetConfig,
    create_agent,
    get_preset,
    list_presets,
)
from tests.fixtures.mock_models import MockModel

EXPECTED_PRESETS = {"math", "research", "coding", "general", "rag", "minimal", "media"}


class TestListPresets:
    """Test list_presets()."""

    def test_returns_all_presets(self):
        presets = list_presets()
        assert set(presets.keys()) == EXPECTED_PRESETS

    def test_returns_descriptions(self):
        presets = list_presets()
        for _name, desc in presets.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestGetPreset:
    """Test get_preset()."""

    def test_valid_preset(self):
        cfg = get_preset("math")
        assert isinstance(cfg, PresetConfig)
        assert cfg.name == "math"

    def test_invalid_preset_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("invalid_preset_name")

    def test_math_has_calculator(self):
        cfg = get_preset("math")
        assert "calculator" in cfg.tool_names

    def test_minimal_has_no_tools(self):
        cfg = get_preset("minimal")
        assert cfg.tool_names == []


class TestCreateAgent:
    """Test create_agent() factory."""

    def _mock_model(self):
        return MockModel(responses=["Thought: done\nFinal Answer: ok"])

    def test_create_math_agent(self):
        model = self._mock_model()
        agent = create_agent("math", model)
        assert agent is not None
        assert agent.config.name == "math-agent"

    def test_create_minimal_agent(self):
        model = self._mock_model()
        agent = create_agent("minimal", model)
        assert agent.config.tools == []

    def test_invalid_preset_raises(self):
        model = self._mock_model()
        with pytest.raises(KeyError, match="Unknown preset"):
            create_agent("invalid_preset_name", model)

    def test_custom_agent_name(self):
        model = self._mock_model()
        agent = create_agent("minimal", model, agent_name="custom-name")
        assert agent.config.name == "custom-name"

    def test_custom_max_iterations(self):
        model = self._mock_model()
        agent = create_agent("minimal", model, max_iterations=99)
        assert agent.config.max_iterations == 99

    def test_create_all_presets(self):
        """Verify all presets can be created without error."""
        model = self._mock_model()
        for name in EXPECTED_PRESETS:
            agent = create_agent(name, model)
            assert agent is not None
