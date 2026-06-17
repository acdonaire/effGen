"""Unit tests for the interactive chat REPL (``effgen chat``).

These exercise the pure logic — model↔tool compatibility filtering, the
model/tool-aware prompt, slash-command routing, the event-aware trace
formatter, and session save/load file mechanics — without any live model
calls. The live streaming/animation/Ctrl-C behavior is verified end-to-end with
real models.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from effgen.cli import chat as C
from effgen.cli import progress as P
from effgen.cli._main import filter_incompatible_tools

# ---------------------------------------------------------------------------
# filter_incompatible_tools — the F2 fix (chat must filter like run)
# ---------------------------------------------------------------------------


class _NamedTool:
    """Minimal stand-in for a tool with a ``name`` and a class identity."""

    def __init__(self, name: str):
        self.name = name


class AnthropicNativeBashTool(_NamedTool):
    pass


class OpenAINativeWebSearchTool(_NamedTool):
    pass


def test_filter_drops_anthropic_native_for_non_claude():
    tools = [_NamedTool("calculator"), AnthropicNativeBashTool("anthropic_bash")]
    kept, skipped = filter_incompatible_tools(tools, "gpt-5-nano")
    names = [t.name for t in kept]
    assert "calculator" in names
    assert "anthropic_bash" not in names
    assert skipped and skipped[0][0] == "anthropic_bash"


def test_filter_keeps_anthropic_native_for_claude():
    tools = [AnthropicNativeBashTool("anthropic_bash")]
    kept, skipped = filter_incompatible_tools(tools, "claude-sonnet-4-6")
    assert [t.name for t in kept] == ["anthropic_bash"]
    assert skipped == []


def test_filter_drops_openai_native_for_non_openai():
    tools = [OpenAINativeWebSearchTool("web_search")]
    kept, _ = filter_incompatible_tools(tools, "Qwen/Qwen2.5-1.5B-Instruct")
    assert kept == []
    kept2, _ = filter_incompatible_tools(tools, "gpt-5-nano")
    assert len(kept2) == 1


def test_filter_warn_callback_invoked():
    seen: list[str] = []
    tools = [AnthropicNativeBashTool("anthropic_bash")]
    filter_incompatible_tools(tools, "gpt-5-nano", warn=seen.append)
    assert any("anthropic_bash" in m for m in seen)


def test_filter_on_real_default_chat_tools():
    """A real registry's first tools must filter cleanly for a generic model."""
    from effgen import get_tool_registry

    reg = get_tool_registry()
    reg.discover_builtin_tools()
    names = reg.list_tools()[:5]  # the old chat loaded exactly these → crashed
    tools = []
    for n in names:
        try:
            tools.append(reg.get_tool_sync(n))
        except Exception:  # noqa: BLE001
            pass
    kept, _ = filter_incompatible_tools(tools, "gpt-5-nano")
    # No kept tool should be an Anthropic-native one for a gpt model.
    assert all("anthropic" not in type(t).__name__.lower() for t in kept)


# ---------------------------------------------------------------------------
# execution_trace_lines — event-aware (not thought/action/observation)
# ---------------------------------------------------------------------------


def test_execution_trace_lines_renders_events():
    trace = [
        {"type": "task_start", "message": "Starting task: foo", "data": {}},
        {"type": "reasoning_step", "message": "Iteration 1: Reasoning...", "data": {}},
        {
            "type": "tool_call_start",
            "message": "Calling tool: calculator",
            "data": {"tool_name": "calculator", "tool_input": '{"expression": "2+2"}'},
        },
        {
            "type": "tool_call_complete",
            "message": "Tool calculator completed",
            "data": {"result": "4"},
        },
    ]
    lines = P.execution_trace_lines(trace)
    text = "\n".join(t for _, t in lines)
    assert "Reasoning" in text
    assert "calculator" in text
    assert "2+2" in text
    assert "4" in text


def test_execution_trace_lines_handles_empty():
    assert P.execution_trace_lines(None) == []
    assert P.execution_trace_lines([]) == []


def test_execution_trace_lines_tool_failure():
    trace = [{"type": "tool_call_failed", "message": "boom", "data": {"error": "bad input"}}]
    lines = P.execution_trace_lines(trace)
    assert any("bad input" in t for _, t in lines)


# ---------------------------------------------------------------------------
# ChatREPL prompt / dispatch logic (no agent build)
# ---------------------------------------------------------------------------


class _FakeCLI:
    """Captures printed output and exposes the surface ChatREPL relies on."""

    def __init__(self):
        self.console = None
        self.messages: list[str] = []

        class _Reg:
            def discover_builtin_tools(self):
                return None

            def list_tools(self):
                return ["calculator", "datetime", "web_search"]

            def get_tool_sync(self, name):
                return _NamedTool(name)

        self.tool_registry = _Reg()

    def _animate(self, args):
        return False

    def print(self, *a, **k):
        self.messages.append(" ".join(str(x) for x in a))

    def print_header(self, t):
        self.messages.append(str(t))

    def print_success(self, t):
        self.messages.append("OK:" + str(t))

    def print_warning(self, t):
        self.messages.append("WARN:" + str(t))

    def print_error(self, t):
        self.messages.append("ERR:" + str(t))


def _make_repl(**arg_overrides: Any) -> tuple[C.ChatREPL, _FakeCLI]:
    args = SimpleNamespace(
        model="gpt-5-nano",
        provider=None,
        _provider=None,
        preset=None,
        temperature=None,
        no_sub_agents=False,
        quiet=False,
        verbose=False,
        no_animation=True,
    )
    for k, v in arg_overrides.items():
        setattr(args, k, v)
    cli = _FakeCLI()
    repl = C.ChatREPL(cli, args)
    return repl, cli


def test_prompt_str_shows_model_and_tools():
    repl, _ = _make_repl()
    # No agent yet -> 0 tools, just the model label.
    assert repl._prompt_str() == "gpt-5-nano › "
    # Fake an agent with two tools.
    repl.agent = SimpleNamespace(tools={"calculator": 1, "datetime": 1})
    assert repl._prompt_str() == "gpt-5-nano · 2 tools › "


def test_prompt_str_includes_preset():
    repl, _ = _make_repl(preset="math")
    repl.agent = SimpleNamespace(tools={"calculator": 1})
    assert repl._prompt_str() == "math · gpt-5-nano · 1 tool › "


def test_dispatch_exit_and_help_and_unknown():
    repl, cli = _make_repl()
    assert repl._dispatch("exit") == "exit"
    assert repl._dispatch("/quit") == "exit"
    assert repl._dispatch("/help") == "handled"
    assert repl._dispatch("/bogus") == "handled"
    assert any("Unknown command" in m for m in cli.messages)
    # A plain message is not a command.
    assert repl._dispatch("hello there") is None


def test_dispatch_cost_reads_session_counters():
    repl, cli = _make_repl()
    repl.agent = SimpleNamespace(reset_memory=lambda: None)
    repl.session_tokens = 1234
    repl.session_cost = 0.0456
    repl.turns = 3
    assert repl._dispatch("/cost") == "handled"
    joined = "\n".join(cli.messages)
    assert "1,234" in joined
    assert "0.0456" in joined


def test_default_tools_empty_for_clean_streaming():
    repl, _ = _make_repl()
    assert repl.tool_names == []


# ---------------------------------------------------------------------------
# save / load file mechanics
# ---------------------------------------------------------------------------


class _FakeMemory:
    def __init__(self):
        from effgen.memory.short_term import MessageRole

        self._msgs = [
            SimpleNamespace(role=MessageRole.USER, content="hi"),
            SimpleNamespace(role=MessageRole.ASSISTANT, content="hello"),
        ]
        self.added: list[tuple[str, str]] = []

    def get_messages(self, n=None):
        return self._msgs

    def add_user_message(self, c):
        self.added.append(("user", c))

    def add_assistant_message(self, c):
        self.added.append(("assistant", c))


def test_save_and_dump_history(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "eh"))
    repl, cli = _make_repl()
    repl.agent = SimpleNamespace(short_term_memory=_FakeMemory())
    repl._cmd_save("mysession")
    saved = list((tmp_path / "eh" / "history").glob("chat_mysession.json"))
    assert saved, "save should write a chat_*.json file"
    data = json.loads(saved[0].read_text())
    assert data["model"] == "gpt-5-nano"
    assert {"role": "user", "content": "hi"} in data["messages"]


def test_banner_hints_resume_when_saved_sessions_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "eh"))
    # No saved sessions yet -> no resume hint.
    repl, cli = _make_repl()
    repl._banner()
    assert not any("Resume:" in m for m in cli.messages)
    # Save one, then a fresh banner should offer to resume it by name.
    repl.agent = SimpleNamespace(short_term_memory=_FakeMemory())
    repl._cmd_save("prior")
    repl2, cli2 = _make_repl()
    repl2._banner()
    joined = "\n".join(cli2.messages)
    assert "Resume:" in joined
    assert "prior" in joined


def test_history_dir_respects_effgen_home(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "home"))
    d = C._history_dir()
    assert str(tmp_path / "home") in str(d)
    assert d.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
