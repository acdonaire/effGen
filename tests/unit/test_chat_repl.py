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


def test_tools_flag_attaches_named_tools():
    repl, _ = _make_repl(tools=["calculator", "datetime"])
    assert repl.tool_names == ["calculator", "datetime"]


def test_tools_flag_unknown_name_warns_with_hint():
    repl, cli = _make_repl(tools=["calculatorr"])
    assert "calculatorr" not in repl.tool_names
    assert any("No tool named 'calculatorr'" in m for m in cli.messages)
    assert any("Did you mean: calculator" in m for m in cli.messages)


def test_tools_flag_merges_with_preset_tools():
    repl, _ = _make_repl(preset="math", tools=["datetime"])
    # Preset tools first, then the extra --tools, deduplicated.
    assert repl.tool_names == ["calculator", "python_repl", "datetime"]


def test_non_interactive_session_marks_not_interactive():
    # Under pytest stdin/stdout are not ttys, so the "Thinking…" placeholder
    # is suppressed for a piped session.
    repl, _ = _make_repl()
    assert repl.interactive is False


# ---------------------------------------------------------------------------
# --preset attaches the preset's tools (mirrors `effgen run --preset`)
# ---------------------------------------------------------------------------


def test_preset_attaches_its_tools():
    repl, _ = _make_repl(preset="math")
    assert repl.tool_names == ["calculator", "python_repl"]
    assert repl._preset_temperature == 0.3
    assert "calculator" in repl._preset_system_prompt


def test_preset_minimal_stays_tool_free():
    repl, _ = _make_repl(preset="minimal")
    assert repl.tool_names == []


def test_unknown_preset_falls_back_to_label_only():
    repl, _ = _make_repl(preset="not-a-real-preset")
    assert repl.tool_names == []
    assert repl._preset_system_prompt is None


def test_build_agent_uses_preset_system_prompt_and_temperature(monkeypatch):
    import effgen as _effgen

    captured: dict[str, Any] = {}

    class _FakeConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeAgent:
        def __init__(self, config, session_id=None):
            self.config = config

    monkeypatch.setattr(_effgen, "AgentConfig", _FakeConfig)
    monkeypatch.setattr(_effgen, "Agent", _FakeAgent)

    repl, _ = _make_repl(preset="math")
    repl._build_agent()

    assert captured["system_prompt"] == repl._preset_system_prompt
    assert captured["temperature"] == 0.3
    assert [t.name for t in captured["tools"]] == ["calculator"]


def test_build_agent_explicit_system_prompt_overrides_preset(monkeypatch):
    import effgen as _effgen

    captured: dict[str, Any] = {}

    class _FakeConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeAgent:
        def __init__(self, config, session_id=None):
            self.config = config

    monkeypatch.setattr(_effgen, "AgentConfig", _FakeConfig)
    monkeypatch.setattr(_effgen, "Agent", _FakeAgent)

    repl, _ = _make_repl(preset="math", system_prompt="You are a pirate.")
    repl._build_agent()

    assert captured["system_prompt"] == "You are a pirate."
    assert captured["temperature"] == 0.3


def test_build_agent_explicit_temperature_overrides_preset(monkeypatch):
    import effgen as _effgen

    captured: dict[str, Any] = {}

    class _FakeConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeAgent:
        def __init__(self, config, session_id=None):
            self.config = config

    monkeypatch.setattr(_effgen, "AgentConfig", _FakeConfig)
    monkeypatch.setattr(_effgen, "Agent", _FakeAgent)

    repl, _ = _make_repl(preset="math", temperature=0.9)
    repl._build_agent()

    assert captured["temperature"] == 0.9


# ---------------------------------------------------------------------------
# --guardrails reaches AgentConfig and shows in the banner
# ---------------------------------------------------------------------------


def test_build_agent_forwards_guardrails(monkeypatch):
    import effgen as _effgen

    captured: dict[str, Any] = {}

    class _FakeConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeAgent:
        def __init__(self, config, session_id=None):
            self.config = config

    monkeypatch.setattr(_effgen, "AgentConfig", _FakeConfig)
    monkeypatch.setattr(_effgen, "Agent", _FakeAgent)

    repl, _ = _make_repl(guardrails="phi")
    repl._build_agent()

    assert captured["guardrails"] == "phi"


def test_build_agent_no_guardrails_flag_defaults_to_none(monkeypatch):
    import effgen as _effgen

    captured: dict[str, Any] = {}

    class _FakeConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeAgent:
        def __init__(self, config, session_id=None):
            self.config = config

    monkeypatch.setattr(_effgen, "AgentConfig", _FakeConfig)
    monkeypatch.setattr(_effgen, "Agent", _FakeAgent)

    repl, _ = _make_repl()
    repl._build_agent()

    assert captured["guardrails"] is None


def test_banner_shows_guardrails_when_set():
    repl, cli = _make_repl(guardrails="strict")
    repl._banner()
    assert "Guardrails: strict" in " ".join(cli.messages)


def test_banner_omits_guardrails_line_when_unset():
    repl, cli = _make_repl()
    repl._banner()
    assert "Guardrails:" not in " ".join(cli.messages)


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


# ---------------------------------------------------------------------------
# persistent session (effgen chat --session-id / --resume)
# ---------------------------------------------------------------------------


def test_session_id_picked_up_from_args():
    repl, _ = _make_repl(session_id="cust-7788")
    assert repl.session_id == "cust-7788"
    # Absent arg -> no session (back-compat).
    repl2, _ = _make_repl()
    assert repl2.session_id is None


def test_persist_session_turn_writes_to_store(tmp_path, monkeypatch):
    # A streamed (plain) turn is persisted to the same session store that
    # `effgen run --session-id` / `effgen sessions` use, so it can be resumed.
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path / "sess"))
    from effgen.core.session import Session

    repl, _ = _make_repl(session_id="cust-1")
    session = Session.load_or_create("cust-1")
    repl.agent = SimpleNamespace(session=session)
    repl._persist_session_turn("my order is ORD-7788", "Noted, ORD-7788.")

    # A brand-new load of the same id recalls the turn.
    reloaded = Session.load_or_create("cust-1")
    contents = [m["content"] for m in reloaded.messages]
    assert "my order is ORD-7788" in contents
    assert "Noted, ORD-7788." in contents


def test_persist_session_turn_noop_without_session():
    # No attached session -> nothing happens, no crash.
    repl, _ = _make_repl()
    repl.agent = SimpleNamespace(session=None)
    repl._persist_session_turn("hi", "there")  # must not raise


def test_banner_shows_session_resume_count(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "eh"))
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path / "sess"))
    from effgen.core.session import Session

    s = Session.load_or_create("cust-9")
    s.add_user_message("hello")
    s.add_assistant_message("hi")
    s.save()

    repl, cli = _make_repl(session_id="cust-9")
    repl._banner()
    joined = "\n".join(cli.messages)
    assert "Session: cust-9" in joined
    assert "resuming 2 prior message" in joined

    # A fresh id is announced as new.
    repl2, cli2 = _make_repl(session_id="brand-new")
    repl2._banner()
    assert any("new (will be saved)" in m for m in cli2.messages)


# ---------------------------------------------------------------------------
# /model provider stickiness — a swap must not keep a stale provider
# ---------------------------------------------------------------------------

def test_resolve_swap_target_known_provider_prefix_pins_provider():
    repl, _ = _make_repl(provider="openai", _provider="openai")
    provider, model_id = repl._resolve_swap_target("groq:llama-3.1-8b-instant")
    assert provider == "groq"
    assert model_id == "llama-3.1-8b-instant"


def test_resolve_swap_target_bare_id_clears_sticky_provider():
    repl, _ = _make_repl(provider="openai", _provider="openai")
    provider, model_id = repl._resolve_swap_target("llama-3.1-8b-instant")
    assert provider is None
    assert model_id == "llama-3.1-8b-instant"


def test_resolve_swap_target_unknown_prefix_clears_provider_id_unchanged():
    # An unknown prefix (typo or local-engine prefix) is left for the model
    # loader's own routing/validation to handle, exactly as at startup.
    repl, _ = _make_repl(provider="openai", _provider="openai")
    provider, model_id = repl._resolve_swap_target("transformers:Qwen/Qwen2.5-1.5B-Instruct")
    assert provider is None
    assert model_id == "transformers:Qwen/Qwen2.5-1.5B-Instruct"


def test_cmd_model_swap_updates_provider_and_does_not_stick():
    repl, cli = _make_repl(provider="openai", _provider="openai")
    assert repl.provider == "openai"
    calls = []

    def _fake_rebuild():
        calls.append((repl.model_id, repl.provider))

    repl._rebuild = _fake_rebuild
    repl._cmd_model("groq:llama-3.1-8b-instant")

    assert repl.provider == "groq"
    assert repl.model_id == "llama-3.1-8b-instant"
    assert calls == [("llama-3.1-8b-instant", "groq")]
    assert any("Switched model" in m for m in cli.messages)


def test_cmd_model_failed_swap_restores_old_model_and_provider():
    repl, cli = _make_repl(provider="openai", _provider="openai")

    def _fake_rebuild_fails():
        raise RuntimeError("Unknown provider or engine prefix 'grok'")

    repl._rebuild = _fake_rebuild_fails
    repl._teach_model_error = lambda e: None
    repl._cmd_model("grok:llama-3.1-8b-instant")

    # Never claims success for a swap that can't be resolved.
    assert repl.model_id == "gpt-5-nano"
    assert repl.provider == "openai"
    assert not any("Switched model" in m for m in cli.messages)
    assert any("Could not switch" in m for m in cli.messages)


def test_cmd_model_failed_swap_suppresses_library_error_log_by_default(caplog):
    import logging

    repl, cli = _make_repl(provider="openai", _provider="openai")

    def _fake_rebuild_fails():
        logging.getLogger("effgen.core.agent").error(
            "Failed to load model 'grok:llama-3.1-8b-instant': boom"
        )
        raise RuntimeError("Unknown provider or engine prefix 'grok'")

    repl._rebuild = _fake_rebuild_fails
    repl._teach_model_error = lambda e: None
    with caplog.at_level(logging.ERROR, logger="effgen"):
        repl._cmd_model("grok:llama-3.1-8b-instant")

    # The REPL's own styled failure message is shown ...
    assert any("Could not switch" in m for m in cli.messages)
    # ... but the library's raw ERROR log line does not leak into the
    # transcript alongside it.
    assert not any(r.levelno == logging.ERROR for r in caplog.records)


def test_cmd_model_failed_swap_verbose_still_shows_library_error_log(caplog):
    import logging

    repl, cli = _make_repl(provider="openai", _provider="openai", verbose=True)

    def _fake_rebuild_fails():
        logging.getLogger("effgen.core.agent").error(
            "Failed to load model 'grok:llama-3.1-8b-instant': boom"
        )
        raise RuntimeError("Unknown provider or engine prefix 'grok'")

    repl._rebuild = _fake_rebuild_fails
    repl._teach_model_error = lambda e: None
    with caplog.at_level(logging.ERROR, logger="effgen"):
        repl._cmd_model("grok:llama-3.1-8b-instant")

    assert any("Could not switch" in m for m in cli.messages)
    assert any(r.levelno == logging.ERROR for r in caplog.records)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
