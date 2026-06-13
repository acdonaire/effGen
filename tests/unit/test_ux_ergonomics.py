"""UX/usability contracts (no live calls).

Covers the user-facing ergonomics fixed in the 0.3.0 polish work:
result ``__str__``/aliases, the ``@tool`` / ``Tool.from_function`` helper, typed
preset errors with fuzzy suggestions, zero-config guidance, model-less-agent
construction failure, and the centralized "did you mean" model suggestion.
"""

import asyncio
from typing import Literal

import pytest


# --------------------------------------------------------------------------- #
# Result ergonomics
# --------------------------------------------------------------------------- #
def test_agent_response_str_is_the_answer():
    from effgen.core.agent import AgentResponse

    r = AgentResponse(output="42.5", success=True)
    assert str(r) == "42.5"
    # repr still carries the structured detail.
    assert "AgentResponse(" in repr(r)
    assert "success=True" in repr(r)


def test_agent_response_text_content_aliases():
    from effgen.core.agent import AgentResponse

    r = AgentResponse(output="hello")
    assert r.text == "hello"
    assert r.content == "hello"
    assert r.text == r.output == r.content


def test_agent_response_str_handles_none_output():
    from effgen.core.agent import AgentResponse

    r = AgentResponse(output=None)  # type: ignore[arg-type]
    assert str(r) == ""


# --------------------------------------------------------------------------- #
# @tool / Tool.from_function
# --------------------------------------------------------------------------- #
def test_tool_decorator_derives_schema_from_signature_and_docstring():
    from effgen import FunctionTool, tool

    @tool
    def word_count(text: str) -> int:
        """Count the words in a piece of text.

        Args:
            text: The text to count words in.
        """
        return len(text.split())

    assert isinstance(word_count, FunctionTool)
    assert word_count.name == "word_count"
    assert word_count.description == "Count the words in a piece of text."
    schema = word_count.metadata.to_json_schema()
    props = schema["parameters"]["properties"]
    assert props["text"]["type"] == "string"
    assert props["text"]["description"] == "The text to count words in."
    assert schema["parameters"]["required"] == ["text"]


def test_tool_decorator_executes_sync_function():
    from effgen import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    result = asyncio.run(add.execute(a=2, b=3))
    assert result.success is True
    assert result.output == 5


def test_tool_decorator_executes_async_function():
    from effgen import tool

    @tool
    async def double(x: int) -> int:
        """Double x."""
        await asyncio.sleep(0)
        return x * 2

    result = asyncio.run(double.execute(x=21))
    assert result.success is True
    assert result.output == 42


def test_tool_decorator_optional_and_literal_params():
    from effgen import tool

    @tool(name="fmt", category="computation")
    def fmt(value: int, style: Literal["hex", "dec"] = "dec", label: str | None = None) -> str:
        """Format a number.

        Args:
            value: the number to format.
            style: output style.
            label: an optional label.
        """
        return label or str(value)

    schema = fmt.metadata.to_json_schema()
    props = schema["parameters"]["properties"]
    # required = only the parameter without a default
    assert schema["parameters"]["required"] == ["value"]
    assert props["style"]["enum"] == ["hex", "dec"]
    assert props["label"]["type"] == "string"  # Optional[str] -> string


def test_tool_validation_reports_missing_required():
    from effgen import tool

    @tool
    def needs_text(text: str) -> str:
        """Echo text."""
        return text

    result = asyncio.run(needs_text.execute())
    assert result.success is False
    assert "required" in (result.error or "").lower()


def test_from_function_equivalent_to_decorator():
    from effgen import Tool

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    t = Tool.from_function(add)
    assert t.name == "add"
    assert asyncio.run(t.execute(a=4, b=5)).output == 9


def test_tool_rejects_lambda_without_name():
    from effgen import Tool

    with pytest.raises(ValueError, match="name="):
        Tool.from_function(lambda x: x)


def test_tool_usable_in_agent_config():
    """A decorated tool drops straight into AgentConfig(tools=[...])."""
    from effgen import tool
    from effgen.core.agent import AgentConfig

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    cfg = AgentConfig(name="t", model=None, tools=[ping], require_model=False)
    assert "ping" in {t.name for t in cfg.tools}


# --------------------------------------------------------------------------- #
# Preset errors + zero-config guidance
# --------------------------------------------------------------------------- #
def test_unknown_preset_is_typed_with_fuzzy_suggestion():
    from effgen.presets import UnknownPresetError, get_preset

    with pytest.raises(UnknownPresetError) as ei:
        get_preset("maths")
    msg = str(ei.value)
    assert "Unknown preset 'maths'" in msg
    assert "Did you mean 'math'?" in msg
    # Clean message (no KeyError repr-quoting), but still compatible.
    assert not msg.startswith('"')
    assert isinstance(ei.value, ValueError | KeyError)


def test_create_agent_without_model_gives_guidance():
    from effgen.presets import create_agent

    with pytest.raises(ValueError) as ei:
        create_agent("math")
    msg = str(ei.value)
    assert "effgen models list" in msg
    assert "gpt-5-nano" in msg


def test_create_agent_env_default_model(monkeypatch):
    from effgen.presets.registry import _resolve_default_model

    monkeypatch.setenv("EFFGEN_DEFAULT_MODEL", "my-default-model")
    assert _resolve_default_model("math") == "my-default-model"


def test_create_agent_docstring_lists_all_presets():
    from effgen.presets import create_agent, list_presets

    doc = create_agent.__doc__ or ""
    for name in list_presets():
        assert name in doc, f"preset {name!r} missing from create_agent docstring"
    # steers newcomers
    assert "kitchen sink" in doc


# --------------------------------------------------------------------------- #
# Model-less agent fails at construction
# --------------------------------------------------------------------------- #
def test_modelless_agent_raises_at_construction():
    from effgen.core.agent import Agent, AgentConfig

    with pytest.raises(ValueError, match="without a model"):
        Agent(AgentConfig(name="x", model=None))  # require_model defaults True


def test_modelless_agent_allowed_when_require_model_false():
    from effgen.core.agent import Agent, AgentConfig

    agent = Agent(AgentConfig(name="x", model=None, require_model=False))
    assert agent.model is None


# --------------------------------------------------------------------------- #
# Centralized "did you mean" model suggestion
# --------------------------------------------------------------------------- #
def test_not_found_error_appends_live_suggestion():
    from effgen.models._adapter_utils import provider_runtime_error

    class _Fake404(Exception):
        status_code = 404

        def __str__(self) -> str:
            return "model_not_found: gpt-5-nanoo does not exist"

    err = provider_runtime_error("openai", "gpt-5-nanoo", "generate", _Fake404())
    text = str(err)
    assert "Did you mean" in text
    assert "gpt-5-nano" in text
    assert err.error_context["category"] == "not_found"
