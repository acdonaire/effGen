"""Terminal-presentation regressions for the CLI polish pass.

Covers: bare group commands printing help (not an ``Unknown X command: None``
dead end), batch output-format validation before any model call, the duplicate
library ERROR line being quieted at default verbosity, unified error panels,
compact tool-call trace args, uniform ``--json``, batch positional input, chat
``--tools``, the non-interactive chat placeholder, and the shared table
renderer's TTY-vs-piped behavior. No live model calls here.
"""
from __future__ import annotations

import io
import json
import logging
import sys

import pytest

from effgen.cli import _main
from effgen.cli.progress import format_tool_call
from effgen.ui.tables import console_is_interactive, render_table


# --------------------------------------------------------------------------- #
# bare group commands print help, never `Unknown X command: None`
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group", ["config", "tools", "models", "examples"])
def test_bare_group_command_prints_help_not_none(group, capsys):
    parser = _main.create_parser()
    args = parser.parse_args([group])
    cli = _main.CLIInterface()
    cli.console = None
    if group in ("config", "tools", "models", "examples"):
        handler = {
            "config": cli.config_commands,
            "tools": cli.tools_commands,
            "models": cli.models_commands,
            "examples": cli.examples_commands,
        }[group]
        code = handler(args)
    out = capsys.readouterr().out
    assert code == 0
    assert "None" not in out
    assert f"{group} [-h]" in out  # the group's own usage line


def test_bare_sessions_and_workflow_print_help(capsys):
    parser = _main.create_parser()
    cli = _main.CLIInterface()
    cli.console = None
    assert _main._handle_sessions_command(parser.parse_args(["sessions"]), cli) == 0
    assert "sessions [-h]" in capsys.readouterr().out
    assert _main._handle_workflow_command(parser.parse_args(["workflow"]), cli) == 0
    assert "workflow [-h]" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# batch validates -o before any model call
# --------------------------------------------------------------------------- #
def test_batch_bad_output_format_fails_before_model(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    src.write_text('{"query": "hi"}\n', encoding="utf-8")
    parser = _main.create_parser()
    args = parser.parse_args(["batch", str(src), "-o", str(tmp_path / "out.bad")])
    cli = _main.CLIInterface()
    cli.console = None
    code = _main._handle_batch_command(args, cli)
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert code == 1
    # Names the valid formats and never reaches the writer's message.
    assert ".jsonl" in text and ".csv" in text and ".json" in text


def test_batch_bad_output_format_json_error_object(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    src.write_text('{"query": "hi"}\n', encoding="utf-8")
    parser = _main.create_parser()
    args = parser.parse_args(["batch", str(src), "-o", "x.xyz", "--json", "-q"])
    cli = _main.CLIInterface()
    cli.console = None
    code = _main._handle_batch_command(args, cli)
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["success"] is False
    assert "Unsupported --output format" in payload["error"]["message"]


# --------------------------------------------------------------------------- #
# batch positional input + chat --tools
# --------------------------------------------------------------------------- #
def test_batch_accepts_positional_input():
    parser = _main.create_parser()
    args = parser.parse_args(["batch", "queries.jsonl"])
    assert args.input_file == "queries.jsonl"
    assert args.input is None


def test_batch_missing_input_reports_cleanly(capsys):
    parser = _main.create_parser()
    args = parser.parse_args(["batch"])
    cli = _main.CLIInterface()
    cli.console = None
    code = _main._handle_batch_command(args, cli)
    captured = capsys.readouterr()
    assert code == 1
    assert "No input file" in captured.out + captured.err


def test_chat_accepts_tools_flag():
    parser = _main.create_parser()
    args = parser.parse_args(["chat", "--tools", "calculator", "wikipedia"])
    assert args.tools == ["calculator", "wikipedia"]


# --------------------------------------------------------------------------- #
# the duplicate library ERROR line is quieted at default verbosity
# --------------------------------------------------------------------------- #
def test_echoed_error_filter_drops_effgen_errors_keeps_others():
    filt = _main._CLIEchoedErrorFilter()

    def rec(name, level):
        return logging.LogRecord(name, level, __file__, 1, "msg", None, None)

    assert filt.filter(rec("effgen.core.agent_generation", logging.ERROR)) is False
    assert filt.filter(rec("effgen.models.openai_adapter", logging.ERROR)) is False
    # A non-effgen logger and a warning-level effgen record are kept.
    assert filt.filter(rec("uvicorn.error", logging.ERROR)) is True
    assert filt.filter(rec("effgen.core.agent", logging.WARNING)) is True


def test_self_rendering_commands_include_run_but_not_serve():
    assert "run" in _main._SELF_RENDERING_ERROR_COMMANDS
    assert "serve" not in _main._SELF_RENDERING_ERROR_COMMANDS


# --------------------------------------------------------------------------- #
# a failure reads as one red panel (no crash on either path)
# --------------------------------------------------------------------------- #
def test_error_panel_plain_fallback(capsys):
    cli = _main.CLIInterface()
    cli.console = None
    cli.print_error_panel("could not load model 'x'", title="Error")
    out = capsys.readouterr().out
    assert "could not load model 'x'" in out


# --------------------------------------------------------------------------- #
# compact, balanced tool-call args
# --------------------------------------------------------------------------- #
def test_format_tool_call_from_json_string():
    got = format_tool_call("calculator", '{"expression": "25 * 17", "operation": "calc"}')
    assert got == 'calculator(expression="25 * 17", operation="calc")'


def test_format_tool_call_from_dict():
    assert format_tool_call("calc", {"expression": "6*7"}) == 'calc(expression="6*7")'


def test_format_tool_call_always_balances_parens():
    long = {"expression": "1+1", "note": "x" * 200}
    got = format_tool_call("calc", long, limit=40)
    assert got.startswith("calc(")
    assert got.endswith(")")
    assert not got.rstrip(")").endswith(",")


def test_format_tool_call_plain_string():
    assert format_tool_call("search", "quantum computing") == "search(quantum computing)"


# --------------------------------------------------------------------------- #
# presets --json
# --------------------------------------------------------------------------- #
def test_presets_json_parses_and_lists(capsys):
    import sys

    argv = sys.argv
    sys.argv = ["effgen", "presets", "--json"]
    try:
        with pytest.raises(SystemExit) as exc:
            _main.main()
    finally:
        sys.argv = argv
    assert exc.value.code == 0
    rows = json.loads(capsys.readouterr().out)
    names = {r["name"] for r in rows}
    assert {"math", "research", "rag"} <= names
    assert all("tool_count" in r for r in rows)


# --------------------------------------------------------------------------- #
# tutorial is a documented alias, not a silent duplicate
# --------------------------------------------------------------------------- #
def test_tutorial_help_documents_alias(capsys):
    import sys

    argv = sys.argv
    sys.argv = ["effgen", "tutorial", "--help"]
    try:
        with pytest.raises(SystemExit):
            _main.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out.lower()
    assert "alias" in out and "quickstart" in out


# --------------------------------------------------------------------------- #
# per-command Examples epilog
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cmd", ["run", "chat", "batch", "compare"])
def test_command_help_has_examples(cmd, capsys):
    import sys

    argv = sys.argv
    sys.argv = ["effgen", cmd, "--help"]
    try:
        with pytest.raises(SystemExit):
            _main.main()
    finally:
        sys.argv = argv
    assert "Examples:" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# shared table renderer: rich on a TTY, clean plain when piped
# --------------------------------------------------------------------------- #
def test_render_table_plain_when_not_interactive(capsys):
    render_table(
        columns=["Session", "Messages"],
        rows=[["writerchat", 6], ["story", 4]],
        console=None,
        justify=["left", "right"],
    )
    out = capsys.readouterr().out
    # No box-drawing glyphs and no ANSI escapes in the piped path.
    assert "│" not in out and "┏" not in out
    assert "\x1b[" not in out
    assert "writerchat" in out and "Session" in out
    # Right-justified numeric column lines up.
    assert "6" in out and "4" in out


def test_render_table_rich_on_forced_terminal():
    from rich.console import Console

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=60)
    assert console_is_interactive(console) is True
    render_table(
        columns=["Model", "Accuracy"],
        rows=[["gpt-5-nano", "100.0%"]],
        console=console,
        title="Accuracy",
    )
    rendered = buf.getvalue()
    assert "│" in rendered or "┏" in rendered  # box-drawing present


def test_console_is_interactive_false_for_none_and_pipe():
    from rich.console import Console

    assert console_is_interactive(None) is False
    assert console_is_interactive(Console(file=io.StringIO())) is False


def test_render_table_file_keeps_stdout_clean(capsys):
    # A caller routing human output away from stdout under --json passes
    # file=stderr; nothing may land on stdout even with an interactive console.
    from rich.console import Console

    console = Console(force_terminal=True)
    render_table(
        columns=["Metric", "Value"],
        rows=[["Accuracy", "100.0%"]],
        console=console,
        file=sys.stderr,
    )
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout stays a clean JSON stream
    assert "Accuracy" in captured.err
    # No box-drawing/ANSI on the plain stderr path.
    assert "│" not in captured.err and "\x1b[" not in captured.err
