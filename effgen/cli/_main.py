"""
effGen CLI - Command-line interface for the effGen framework.

This module provides a comprehensive CLI for interacting with effGen:
- Run agents with tasks (with interactive wizard support)
- Interactive chat mode
- API server mode
- Configuration management
- Tool listing and testing
- Model management
- Example runner

Usage:
    # Direct task execution
    effgen run "What is 2+2?" --model Qwen/Qwen2.5-1.5B-Instruct

    # Interactive wizard (launches when no task provided)
    effgen run
    effgen  # Same as above

    # effgen-agent command (similar to smolagent)
    effgen-agent "Plan a trip to Tokyo" --model Qwen/Qwen2.5-1.5B-Instruct --tools web_search
    effgen-agent  # Interactive mode

    # Chat mode
    effgen chat --model Qwen/Qwen2.5-3B-Instruct

    # Other commands
    effgen serve --port 8000
    effgen config show
    effgen tools list
    effgen models list
    effgen examples run basic_agent

Interactive mode guides you through:
    - Agent type selection (CodeAgent vs ToolCallingAgent vs ReActAgent)
    - Tool selection from available toolbox
    - Model configuration (type, ID, API settings)
    - Advanced options (temperature, max iterations, etc.)
    - Task prompt input
"""

from __future__ import annotations

import argparse
import asyncio  # noqa: F401 - module attribute for command modules that import it from here
import importlib.util  # noqa: F401 - module attribute for command modules
import json
import logging
import os
import sys
from datetime import datetime  # noqa: F401 - module attribute for command modules
from pathlib import Path
from typing import Any

# Rich terminal output (fallback to basic if not available)
try:
    from rich import print as rprint  # noqa: F401
    from rich.console import Console
    from rich.layout import Layout  # noqa: F401
    from rich.live import Live  # noqa: F401
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: F401 - module attrs
    from rich.syntax import Syntax  # noqa: F401 - availability shim / module attribute
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


# Import effGen components
try:
    from effgen import (  # noqa: F401
        Agent,
        AgentConfig,
        ConfigLoader,
        __version__,
        get_tool_registry,
        load_model,
    )
    from effgen.core.agent import AgentMode  # noqa: F401 - read as a module attribute
    from effgen.tools.builtin import *
except ImportError:
    print("Error: effGen package not found. Please install it first.")
    sys.exit(1)

# .env discovery + loading is shared with the library-facing ``effgen.load_env()``
# so the CLI and a script/notebook resolve keys the same way. These aliases keep
# the historical CLI-internal names (used by tests and callers) working.
from effgen._env import dotenv_disabled as _dotenv_disabled  # noqa: F401 - re-export
from effgen._env import env_search_paths as _env_search_paths  # noqa: F401 - re-export
from effgen._env import load_env as load_env_files

# Tips, first-run welcome, "did you mean?" and teaching-error helpers.
from effgen.cli import onboarding as _onboarding

# Live status / progress presentation (TTY-aware; degrades to plain text).
from effgen.cli import progress as _progress

# Helpers shared by several commands. They live in ``effgen.cli.commands._shared``
# and are re-exported here so ``effgen.cli._main`` stays the import path callers
# and tests use.
from effgen.cli.commands._shared import (  # noqa: F401 - re-export
    _QUICKSTART_CLOUD_MODELS,
    _QUICKSTART_LOCAL_MODEL,
    _RUN_CONFIG_APPLIED_KEYS,
    KNOWN_PROVIDERS,
    PROVIDER_ALIASES,
    _checkpoint_run_kwargs,
    _invoked_command,
    _preflight_model_hint,
    _print_group_help,
    _quickstart_suggest_model,
    _warn_unapplied_config_keys,
    filter_incompatible_tools,
    resolve_provider_name,
)

# The ``batch`` command. Imported at module scope so the handler is an attribute
# of this module as soon as it finishes importing.
from effgen.cli.commands.batch import (  # noqa: F401 - re-export
    _batch_structured_kwargs,
    _handle_batch_command,
    _read_done_indices,
)

# The ``compare`` command handler and its per-metric terminal tables.
# Module-level free functions; imported at module scope + re-exported (tests
# reach ``_main._handle_compare_command`` / ``_render_comparison_tables``).
from effgen.cli.commands.compare import (  # noqa: F401 - re-export
    _handle_compare_command,
    _render_comparison_tables,
)

# The ``cost`` command handler (spend dashboard + budget). Imported at module
# scope + re-exported; ``effgen.cli.__init__`` re-exports it from here, so
# ``from effgen.cli import _handle_cost_command`` keeps resolving.
from effgen.cli.commands.cost import _handle_cost_command  # noqa: F401 - re-export

# The ``eval`` command handler plus ``--suite`` resolution. Module-level free
# functions taking ``(args, cli)``; imported at module scope + re-exported (tests
# reach ``_main._handle_eval_command`` / ``_resolve_eval_suite``).
from effgen.cli.commands.eval import (  # noqa: F401 - re-export
    _handle_eval_command,
    _resolve_eval_suite,
)

# The ``report`` command handler and the ``-o``/``--report`` artifact writers
# shared by ``eval``/``compare``/``cost``. Imported at module scope so the names
# are attributes of this module (tests reach ``_main._artifact_format`` /
# ``_write_result_artifact`` and the ``eval``/``compare``/``cost`` handlers call
# the bare names).
from effgen.cli.commands.report import (  # noqa: F401 - re-export
    _artifact_format,
    _handle_report_command,
    _write_html_report_arg,
    _write_result_artifact,
)

# The ``serve`` command launcher and its ``/run``, ``/tools``, ``/``, ``/slo``,
# ``/ws`` convenience routes. ``CLIInterface.serve_api`` /
# ``._register_convenience_routes`` delegate there. The request models,
# ``WebSocket`` type, and default-tool helper are re-exported so FastAPI's
# annotation resolution and any ``_main.TaskRequest``/``_main.WebSocket``
# reference keep working (the models must live in the handler's own module).
from effgen.cli.commands.serve import (  # noqa: F401 - re-export
    TaskRequest,
    TaskResponse,
    WebSocket,
    _general_purpose_tool_names,
    _PydanticBaseModel,
    _PydanticConfigDict,
)

# The ``sessions`` and ``runs`` command handlers plus their formatting helpers.
# Module-level free functions taking ``(args, cli)``; imported at module scope so
# every name is an attribute of this module as soon as it finishes importing
# (tests reach them as ``_main._handle_sessions_command`` / ``_main._fmt_cost``).
from effgen.cli.commands.sessions import (  # noqa: F401 - re-export
    _browse_sessions,
    _fmt_cost,
    _handle_runs_command,
    _handle_sessions_command,
    _one_line,
    _render_session,
    _session_matches,
    _short_ts,
)

# The ``workflow`` command handler. A module-level free function taking
# ``(args, cli)``; imported at module scope so it is an attribute of this module
# as soon as it finishes importing (tests reach it as ``_main._handle_workflow_command``).
from effgen.cli.commands.workflow import (  # noqa: F401 - re-export
    _handle_workflow_command,
)

# Shared Rich theme + console factory (one palette across the whole CLI).
# ``console_is_interactive``/``render_table`` are re-exported so callers and
# tests keep reaching them as ``effgen.cli._main`` attributes.
from effgen.ui.tables import (  # noqa: F401 - re-export
    console_is_interactive,
    render_table,
)
from effgen.ui.theme import CODE_THEME  # noqa: F401 - module attribute for command modules
from effgen.ui.theme import get_console as _get_console

# Commands that render their own failures (a classified message pointing at the
# fix, a red error panel). For these, a raw ``<timestamp> - <logger> - ERROR``
# console line from the library beneath the framed message is duplicate noise at
# default verbosity — it is kept only under ``--verbose`` and in a ``--log-file``.
_SELF_RENDERING_ERROR_COMMANDS = frozenset({
    "run", "batch", "chat", "eval", "compare", "debug", "resume",
    "quickstart", "tutorial",
})


class _CLIEchoedErrorFilter(logging.Filter):
    """Drop console ERROR records the CLI already surfaces as a framed message.

    Applied to the console handler only (a ``--log-file`` still captures the full
    stream) and only for the commands that render their own errors, so a library
    ``ERROR`` log does not print a second time beneath the CLI's own message.
    ``effgen serve`` is excluded — an operator wants those server-side ERRORs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR and (
            record.name == "effgen" or record.name.startswith("effgen.")
        ):
            return False
        return True


def _suppress_echoed_error_logs() -> None:
    """Attach :class:`_CLIEchoedErrorFilter` to the root console handler."""
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.addFilter(_CLIEchoedErrorFilter())


# Configure logging
def setup_logging(
    verbose: bool = False,
    log_file: str | None = None,
    quiet: bool = False,
) -> None:
    """
    Configure logging for CLI.

    The CLI is quiet by default so command output (tables, answers) stays clean
    and copy-pasteable: library diagnostics are kept at WARNING and above, and
    informational chatter from tool discovery / config loading is hidden. Use
    ``--verbose`` to surface DEBUG/INFO, or ``--quiet`` to show errors only.

    Args:
        verbose: Show DEBUG/INFO diagnostics.
        log_file: Optional log file path (always captures full DEBUG detail).
        quiet: Show errors only (suppress warnings).
    """
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.ERROR
    else:
        # Default: quiet, professional CLI output — warnings and errors only.
        level = logging.WARNING

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )
    # Route library logs to stderr so they never interleave with stdout tables,
    # and keep the console at the requested level even if a log file is attached.
    logging.getLogger().setLevel(min(level, logging.DEBUG) if log_file else level)
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(level)


class CLIInterface:
    """Main CLI interface for effGen."""

    def __init__(self) -> None:
        """Initialize CLI interface."""
        self.console = _get_console() if RICH_AVAILABLE else None
        self.config_loader = ConfigLoader()
        self.tool_registry = get_tool_registry()
        # When True, all human-facing chatter is routed to stderr so stdout
        # carries only machine-readable output (e.g. `effgen run --json`). A
        # stderr-bound rich console is created lazily on first use.
        self._human_to_stderr = False
        self._err_console = None

    def _human(self):
        """Return the console human-facing output should go to (stdout/stderr)."""
        if self._human_to_stderr:
            if self._err_console is None and RICH_AVAILABLE:
                self._err_console = _get_console(stderr=True)
            return self._err_console
        return self.console

    def _animate(self, args) -> bool:
        """Whether to show live animation for this invocation (TTY-aware, opt-out)."""
        return _progress.animation_enabled(
            quiet=getattr(args, 'quiet', False),
            no_animation=getattr(args, 'no_animation', False),
        )

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print with rich formatting if available."""
        console = self._human()
        if console:
            console.print(*args, **kwargs)
        else:
            print(*args, file=sys.stderr if self._human_to_stderr else None, **kwargs)

    def print_data(self, text: str) -> None:
        """Write *text* to stdout byte-for-byte.

        Used for output that is data rather than chatter — an exported session,
        a rendered prompt, a model answer. It bypasses the markup renderer, so
        square-bracket content such as ``[bold]`` or ``[1]`` survives instead of
        being read as console markup and dropped.
        """
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    def print_line(self, text: str, style: str | None = None) -> None:
        """Print one line with no markup parsing and no value highlighting.

        Stored values such as model ids, timestamps and durations stay intact
        instead of being read as console markup or split into separately
        colored tokens.
        """
        console = self._human()
        if console:
            console.print(text, highlight=False, markup=False, style=style)
        else:
            print(text, file=sys.stderr if self._human_to_stderr else None)

    def print_header(self, text: str) -> None:
        """Print a header."""
        console = self._human()
        if console:
            console.print(f"\n[effgen.heading]{text}[/effgen.heading]")
        else:
            print(f"\n=== {text} ===", file=sys.stderr if self._human_to_stderr else None)

    def print_success(self, text: str) -> None:
        """Print success message."""
        console = self._human()
        if console:
            console.print(f"[green]✓[/green] {text}")
        else:
            print(f"✓ {text}", file=sys.stderr if self._human_to_stderr else None)

    def print_error(self, text: str) -> None:
        """Print error message."""
        console = self._human()
        if console:
            console.print(f"[red]✗[/red] {text}")
        else:
            print(f"✗ {text}", file=sys.stderr if self._human_to_stderr else None)

    def print_warning(self, text: str) -> None:
        """Print warning message."""
        console = self._human()
        if console:
            console.print(f"[effgen.warning]⚠[/effgen.warning] {text}")
        else:
            print(f"⚠ {text}", file=sys.stderr if self._human_to_stderr else None)

    def print_error_panel(self, message: str, *, title: str = "Error") -> None:
        """Render a failure as a red-bordered panel, or a styled line without rich.

        A run that fails at generation and a run that fails to load its model
        then read the same way — one red panel — instead of a panel in one case
        and a bare line in the other. The message is shown as plain text so
        provider error strings can't inject console markup.
        """
        console = self._human()
        if console and RICH_AVAILABLE:
            from rich.panel import Panel
            from rich.text import Text
            console.print(
                Panel(Text(message or ""), title=f"[red]{title}[/red]", border_style="red")
            )
        else:
            self.print_error(message)

    def interactive_wizard(self, args: Any) -> int | None:
        """
        Interactive setup wizard for configuring and running agents.

        Similar to smolagents CLI, guides users through:
        - Agent type selection
        - Tool selection from available toolbox
        - Model configuration (type, ID, API settings)
        - Advanced options like additional imports
        - Task prompt input

        Args:
            args: Parsed command-line arguments (may have partial values)

        Returns:
            Exit code
        """
        from effgen.cli.commands.run import interactive_wizard

        return interactive_wizard(self, args)

    def run_agent(self, args: argparse.Namespace) -> int | None:
        """
        Run an agent with a task.

        Args:
            args: Parsed command-line arguments
        """
        from effgen.cli.commands.run import run_agent

        return run_agent(self, args)

    def _stream_tokens(self, token_iter, *, animate: bool) -> str:
        """Print streamed tokens with an optional soft cursor; return the text.

        On an interactive terminal a single-cell soft cursor (``▌``) trails the
        latest token and is erased before the next one, giving a live-typing
        feel. When not animating (piped/redirected/non-TTY) tokens are written
        plainly so the output is clean to capture.
        """
        from effgen.cli.commands.run import stream_tokens

        return stream_tokens(self, token_iter, animate=animate)

    def _handle_interrupt(self, agent) -> None:
        """Render a friendly Ctrl-C stop (partial trace + 'Stopped.'), no traceback."""
        from effgen.cli.commands.run import handle_interrupt

        return handle_interrupt(self, agent)

    def _create_stats_table(self, stats: dict[str, Any]) -> Any:
        """Create statistics table."""
        if not self.console:
            return stats

        table = Table(title="Execution Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in stats.items():
            table.add_row(key, str(value))

        return table

    def chat_mode(self, args: argparse.Namespace) -> int | None:
        """Interactive chat REPL.

        Delegates to :class:`effgen.cli.chat.ChatREPL`, which provides streaming
        answers with a thinking spinner, a model/tool-aware prompt, slash
        commands (``/model``, ``/tools``, ``/cost``, ``/trace``, …), persistent
        ↑/↓ history, multiline input, and a per-turn Ctrl-C that cancels the
        current turn without exiting the session.
        """
        from effgen.cli.commands.chat import chat_mode

        return chat_mode(self, args)

    def serve_api(self, args: argparse.Namespace) -> int | None:
        """Start the effGen API server.

        Serves the OpenAI-compatible ``/v1`` app plus the ``/run``, ``/tools``,
        ``/``, ``/slo``, ``/ws`` convenience routes.
        """
        from effgen.cli.commands.serve import serve_api
        return serve_api(self, args)

    def _register_convenience_routes(self, app: Any) -> None:
        """Attach the legacy ``/run``, ``/tools``, ``/``, ``/slo``, ``/ws``
        routes onto the secure app from ``create_app``."""
        from effgen.cli.commands.serve import register_convenience_routes
        return register_convenience_routes(self, app)

    def config_commands(self, args: argparse.Namespace) -> int | None:
        """Configuration management commands."""
        from effgen.cli.commands.config import config_commands
        return config_commands(self, args)

    def _config_set(self, args):
        """Handle 'effgen config set <key> <value>'."""
        from effgen.cli.commands.config import config_set
        return config_set(self, args)

    def _config_show(self, args):
        """Show current configuration."""
        from effgen.cli.commands.config import config_show
        return config_show(self, args)

    def _config_validate(self, args):
        """Validate configuration file."""
        from effgen.cli.commands.config import config_validate
        return config_validate(self, args)

    def _config_init(self, args):
        """Initialize a new configuration file."""
        from effgen.cli.commands.config import config_init
        return config_init(self, args)

    def tools_commands(self, args: argparse.Namespace) -> int | None:
        """Tool management commands."""
        from effgen.cli.commands.tools import tools_commands
        return tools_commands(self, args)

    def _suggest_tool(self, name: str) -> None:
        """Print a 'tool not found' error with close-match suggestions."""
        from effgen.cli.commands.tools import suggest_tool
        return suggest_tool(self, name)

    def _tools_list(self, args):
        """List available tools."""
        from effgen.cli.commands.tools import tools_list
        return tools_list(self, args)

    def _example_input(self, metadata, tool=None) -> dict:
        """Build a runnable example input for a tool from its metadata."""
        from effgen.cli.commands.tools import example_input
        return example_input(self, metadata, tool)

    def _print_tool_usage(self, metadata, tool=None) -> None:
        """Print a tool's input schema and a copy-paste runnable example."""
        from effgen.cli.commands.tools import print_tool_usage
        return print_tool_usage(self, metadata, tool)

    def _tools_info(self, args: argparse.Namespace) -> int | None:
        """Show detailed tool information."""
        from effgen.cli.commands.tools import tools_info
        return tools_info(self, args)

    def _tools_test(self, args: argparse.Namespace) -> int | None:
        """Test a tool with sample input."""
        from effgen.cli.commands.tools import tools_test
        return tools_test(self, args)

    def models_commands(self, args: argparse.Namespace) -> int | None:
        """Model management commands."""
        from effgen.cli.commands.models import models_commands
        return models_commands(self, args)

    @staticmethod
    def _price_cell(rec) -> str:
        """Format a model's input/output price per 1M tokens for a table cell."""
        from effgen.cli.commands.models import price_cell
        return price_cell(rec)

    # File extensions that count as actual model weights (an ".index.json" is a
    # shard manifest, not weights — a repo with only a manifest is still partial).
    _WEIGHT_SUFFIXES = (
        ".safetensors", ".bin", ".gguf", ".pt", ".pth",
        ".onnx", ".msgpack", ".h5", ".tflite", ".ot",
    )

    def _local_cached_models(self) -> list[dict]:
        """Models actually downloaded in the local HuggingFace cache (on disk)."""
        from effgen.cli.commands.models import local_cached_models
        return local_cached_models(self)

    def _local_model_context_window(self, path: str) -> int | None:
        """Read the model's max context length from its on-disk ``config.json``."""
        from effgen.cli.commands.models import local_model_context_window
        return local_model_context_window(self, path)

    def _models_list(self, args):
        """List models from the drift-aware registry (not a static yaml)."""
        from effgen.cli.commands.models import models_list
        return models_list(self, args)

    def _rich_tables(self) -> bool:
        """True when rich table rendering fits the destination (a real terminal)."""
        from effgen.cli.commands.models import rich_tables
        return rich_tables(self)

    @staticmethod
    def _browse_filter_sort(recs, args):
        """Apply the browse filters/sort to a list of catalog records."""
        from effgen.cli.commands.models import browse_filter_sort
        return browse_filter_sort(recs, args)

    def _models_browse(self, args):
        """Browse the full cross-provider catalog with search/filter/sort/paging."""
        from effgen.cli.commands.models import models_browse
        return models_browse(self, args)

    @classmethod
    def _price_in_out_cells(cls, rec) -> tuple[str, str]:
        """Return (input, output) price cells for the split-column browse table."""
        from effgen.cli.commands.models import price_in_out_cells
        return price_in_out_cells(rec)

    def _local_model_payload(self, entry: dict) -> dict:
        """Build the local-cache facts for one model: engines, size, ctx, status."""
        from effgen.cli.commands.models import local_model_payload
        return local_model_payload(self, entry)

    def _render_local_model_info(self, payload: dict) -> None:
        """Render the 'this model is in your local cache' block for `models info`."""
        from effgen.cli.commands.models import render_local_model_info
        return render_local_model_info(self, payload)

    def _models_info(self, args):
        """Show detailed information for one model from the registry."""
        from effgen.cli.commands.models import models_info
        return models_info(self, args)

    def _models_load(self, args):
        """Pre-load a model into the model pool."""
        from effgen.cli.commands.models import models_load
        return models_load(self, args)

    def _models_unload(self, args):
        """Unload a model from memory."""
        from effgen.cli.commands.models import models_unload
        return models_unload(self, args)

    def _models_status(self, args):
        """Show loaded models and GPU memory status."""
        from effgen.cli.commands.models import models_status
        return models_status(self, args)

    def _models_status_json(self) -> int:
        """Emit the GPU table + loaded models as JSON for ops/edge tooling."""
        from effgen.cli.commands.models import models_status_json
        return models_status_json(self)

    def _models_refresh(self, args):
        """Refresh the bundled model catalog from each provider's live API."""
        from effgen.cli.commands.models import models_refresh
        return models_refresh(self, args)

    def examples_commands(self, args: argparse.Namespace) -> int | None:
        """Run example scripts."""
        from effgen.cli.commands.examples import examples_commands
        return examples_commands(self, args)

    @staticmethod
    def _find_examples_dir() -> "Path | None":
        """Locate the bundled `examples/` directory."""
        from effgen.cli.commands.examples import find_examples_dir
        return find_examples_dir()

    def _examples_list(self, args):
        """List available examples."""
        from effgen.cli.commands.examples import examples_list
        return examples_list(self, args)

    def _examples_run(self, args):
        """Run an example script."""
        from effgen.cli.commands.examples import examples_run
        return examples_run(self, args)


class _EffgenArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that turns an "invalid choice" into a teaching error.

    A mistyped subcommand (``effgen rnu``) or a mistyped ``choices=`` option
    value (``--preset codng``) gets a fuzzy "did you mean 'run'?" /
    "did you mean 'coding'?" suggestion instead of just dumping the usage
    banner, so the CLI is self-correcting like the model/tool suggesters
    elsewhere. Because ``add_subparsers`` inherits this parser class, the
    handler fires for both the top-level command and every sub-command.
    """

    def error(self, message: str):  # noqa: D401 - argparse hook
        if "invalid choice:" in message:
            import re

            m_bad = re.search(r"invalid choice: '([^']*)'", message)
            m_arg = re.search(r"argument ([^:]+): invalid choice", message)
            # The offending argument's valid choices are spelled out in the
            # message ("(choose from 'a', 'b', …)"); fall back to the
            # subparsers action so an older argparse phrasing still works.
            choices: list[str] = []
            if "choose from" in message:
                tail = message.split("choose from", 1)[1]
                choices = re.findall(r"'([^']*)'", tail)
            sub_action = next(
                (a for a in self._actions
                 if isinstance(a, argparse._SubParsersAction)), None)
            if not choices and sub_action is not None:
                choices = list(sub_action.choices.keys())
            if m_bad and choices:
                bad = m_bad.group(1)
                arg_name = m_arg.group(1) if m_arg else None
                hint = _onboarding.did_you_mean(bad, choices, n=1, cutoff=0.5)
                self.print_usage(sys.stderr)
                # A positional sub-command (dest matches the subparsers action)
                # reads best as "unknown command"; a flag value as "invalid value".
                is_subcommand = sub_action is not None and (
                    arg_name is None or arg_name in (sub_action.dest, sub_action.metavar)
                )
                if is_subcommand:
                    line = f"{self.prog}: error: unknown command '{bad}'."
                    if hint:
                        line += f" {hint}"
                    line += f"\nAvailable commands: {', '.join(choices)}."
                else:
                    label = f" for {arg_name}" if arg_name else ""
                    line = f"{self.prog}: error: invalid value '{bad}'{label}."
                    if hint:
                        line += f" {hint}"
                    line += f"\nValid choices: {', '.join(choices)}."
                print(line, file=sys.stderr)
                self.exit(2)
        super().error(message)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    # Drive --preset choices from the preset registry so all 9 presets are
    # accepted (rag/media/multimodal/notify were previously rejected).
    try:
        from effgen.presets import list_presets as _list_presets
        _preset_choices = sorted(_list_presets().keys())
    except Exception:
        _preset_choices = ['math', 'research', 'coding', 'general', 'minimal']

    parser = _EffgenArgumentParser(
        description=f"effGen v{__version__} - CLI for agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  effgen run "What is 25 * 17?" --model Qwen/Qwen2.5-1.5B-Instruct
  effgen run "Summarize quantum computing" -m gpt-5-nano --provider openai
  effgen run "Tell me a joke" -m groq:llama-3.1-8b-instant
  effgen chat -m gpt-5-nano --provider openai
  effgen models list                 # provider registry overview
  effgen models list --provider groq # full per-model detail
  effgen doctor --live --cheap       # confirm providers are actually usable
  effgen tools list

Model id formats:
  - Local HuggingFace repo:   Qwen/Qwen2.5-1.5B-Instruct
  - Provider-prefixed:        openai:gpt-5-nano   groq:llama-3.1-8b-instant
  - Bare id + --provider:     -m gpt-5-nano --provider openai
  Providers: openai, anthropic, gemini, cerebras, groq, together,
             fireworks, replicate, hf
        """
    )

    parser.add_argument('--version', action='version', version=f'effGen {__version__}')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output (show DEBUG/INFO logs)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet output (errors only)')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--no-animation', action='store_true',
                        help='Disable live spinners/progress animation '
                             '(also via NO_COLOR or EFFGEN_NO_ANIM=1)')
    parser.add_argument('--theme', choices=['default', 'high-contrast', 'monochrome', 'light'],
                        help='Color theme for terminal output (also via EFFGEN_THEME). '
                             'high-contrast targets low-vision readers; monochrome keeps '
                             'structure without hue; NO_COLOR still turns color off entirely')
    parser.add_argument('--completion', choices=['bash', 'zsh', 'fish'],
                        help='Print shell completion script and exit')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run command
    run_parser = subparsers.add_parser(
        'run', help='Run an agent with a task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen run \"What is 25 * 17?\" -t calculator\n"
            "  effgen run \"Summarize this\" --file report.pdf -m gpt-5-nano\n"
            "  effgen run \"Draft a reply\" --persona \"terse, formal\" --json | jq .output\n"
            "\n"
            "Environment:\n"
            "  EFFGEN_WORKSPACE   directory where the file and shell tools read\n"
            "                     and write by default. Set it to keep files an\n"
            "                     agent generates out of the current directory\n"
            "                     (created if missing). Unset: the current dir.\n"
        ),
    )
    run_parser.add_argument('task', nargs='?', default=None, help='Task description (launches interactive wizard if not provided)')
    run_parser.add_argument('-m', '--model', help='Model to use')
    run_parser.add_argument(
        '--provider',
        help='Provider for a bare model id (e.g. openai, groq, cerebras, gemini, '
             'together, fireworks, replicate, anthropic, hf). '
             'Equivalent to the "provider:model" prefix.',
    )
    run_parser.add_argument('-v', '--verbose', action='store_true', default=argparse.SUPPRESS,
                            help='Verbose output (show DEBUG/INFO logs)')
    run_parser.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                            help='Quiet output (errors only)')
    run_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                            help='Disable live spinners/progress animation')
    run_parser.add_argument('-n', '--name', help='Agent name')
    run_parser.add_argument('-t', '--tools', nargs='+', help='Tools to enable')
    run_parser.add_argument('-c', '--config', help='Configuration file')
    run_parser.add_argument(
        '--system-prompt', '--persona', dest='system_prompt', metavar='TEXT',
        help='Custom persona / system prompt for this run, e.g. '
             '"You are a patient Socratic tutor who never gives the answer."',
    )
    run_parser.add_argument('--temperature', type=float, help='Temperature')
    run_parser.add_argument('--max-tokens', type=int,
                            help='Max output tokens (raise for token-heavy or '
                                 'reasoning models, e.g. gpt-5/o-series, which '
                                 'spend part of the budget on hidden reasoning '
                                 'before any visible text)')
    run_parser.add_argument('--max-iterations', type=int, help='Max iterations')
    run_parser.add_argument('--mode', choices=['auto', 'single', 'sub_agents'], help='Execution mode')
    run_parser.add_argument('--no-sub-agents', action='store_true', help='Disable sub-agents')
    run_parser.add_argument('--stream', action='store_true', help='Stream output')
    run_parser.add_argument('-o', '--output',
                            help='Write the full result as a JSON document to this '
                                 'file (output, success, tool_calls, tokens, cost, '
                                 'trace, citations, metadata)')
    run_parser.add_argument('--card', metavar='PATH.html',
                            help='Write a shareable HTML card for this run to PATH — '
                                 'the task, the answer, the tool trace with per-step '
                                 'durations, sources and citations, and tokens/cost/'
                                 'latency. The file is self-contained and opens with '
                                 'no network access. Terminal and --json output are '
                                 'unchanged.')
    run_parser.add_argument('--json', dest='output_json', action='store_true',
                            help='Emit that same JSON result object to stdout (for '
                                 'piping to jq). Human output goes to stderr; '
                                 'combine with -q for clean stdout.')
    run_parser.add_argument('--preset', choices=_preset_choices,
                            help='Use a preset agent configuration')
    run_parser.add_argument(
        '--guardrails', metavar='NAME',
        help='Apply a guardrail preset to redact/block PII and screen for '
             'prompt injection before the task reaches the model: '
             '"strict", "standard" (alias "default"/"balanced"), "phi" '
             '(alias "hipaa"/"deidentify"), "minimal", or "none". Also '
             'honored from a `-c/--config` file\'s "guardrails" key.',
    )
    run_parser.add_argument('--explain', action='store_true',
                            help='Show why the agent chose each tool')
    run_parser.add_argument('--trace', action='store_true',
                            help='Show a step-by-step timeline with per-step durations')
    run_parser.add_argument('--checkpoint-dir', help='Directory to write agent checkpoints')
    run_parser.add_argument('--checkpoint-interval', type=int, default=0,
                            help='Checkpoint every N iterations (requires --checkpoint-dir)')
    run_parser.add_argument(
        '--session-id', metavar='ID',
        help='Persistent conversation session id (shared with `effgen chat '
             '--session-id` and `effgen sessions`). Recalls prior turns and '
             'saves new ones. (Distinct from `effgen resume --checkpoint`, which '
             'restores a mid-run checkpoint snapshot.)',
    )
    run_parser.add_argument(
        '--file', '--input', dest='input_files', action='append', metavar='PATH',
        help='Attach a file to the task. An image (.png/.jpg/.gif/.webp/...) is '
             'passed as multimodal input; a document (.pdf/.docx/.xlsx/.txt/'
             '.md/.csv/...) or a source file (.py/.js/.ts/.go/.rs/.java/.sql/'
             '...) is read and prepended to the task as context. Any other file '
             'that decodes as UTF-8 text is read as plain text. Repeatable.',
    )

    # Resume command
    resume_parser = subparsers.add_parser(
        'resume',
        help='Resume an interrupted agent run from a saved checkpoint snapshot '
             '(distinct from a conversation session — see `--session-id`)')
    resume_parser.add_argument('--checkpoint', required=True,
                               help='Checkpoint id, JSON path, or directory (uses latest)')
    resume_parser.add_argument('-m', '--model', help='Model to use')
    resume_parser.add_argument('--preset', choices=_preset_choices)

    # Sessions commands
    sessions_parser = subparsers.add_parser(
        'sessions', help='Browse and manage saved conversations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen sessions list --search refund --since 2026-07-01\n"
            "  effgen sessions show support-4471 --last 4\n"
            "  effgen sessions browse\n"
            "\n"
            "Continue a saved conversation with `effgen chat --session-id <id>` or\n"
            "`effgen run \"...\" --session-id <id>`. (`effgen resume` replays a\n"
            "workflow checkpoint, which is a different store.)\n"
        ),
    )
    sessions_parser.set_defaults(_group_parser=sessions_parser)
    sessions_subparsers = sessions_parser.add_subparsers(dest='session_command', help='Sessions command')
    _sessions_list = sessions_subparsers.add_parser('list', help='List sessions')
    _sessions_list.add_argument('--json', dest='output_json', action='store_true',
                                help='Output the session list as JSON')
    _sessions_list.add_argument('--search', help='Only sessions whose id, agent or messages match this text')
    _sessions_list.add_argument('--since', help='Only sessions updated on/after this date (YYYY-MM-DD)')
    _sessions_list.add_argument('--until', help='Only sessions updated on/before this date (YYYY-MM-DD)')
    _sessions_list.add_argument('--model', help='Only sessions answered by a model matching this text')
    _sessions_list.add_argument('--limit', type=int, default=50, help='Maximum sessions to show (default: 50)')
    ss = sessions_subparsers.add_parser(
        'show', help='Read a conversation turn by turn',
    )
    ss.add_argument('session_id', help='Session id')
    ss.add_argument('--last', type=int, help='Show only the last N messages')
    ss.add_argument('--json', dest='output_json', action='store_true',
                    help='Output the conversation as JSON')
    sb = sessions_subparsers.add_parser(
        'browse', help='Pick a session from the list and read it',
    )
    sb.add_argument('--json', dest='output_json', action='store_true',
                    help='Output the session list as JSON instead of prompting')
    sb.add_argument('--limit', type=int, default=20, help='Sessions to offer (default: 20)')
    sd = sessions_subparsers.add_parser('delete', help='Delete a session')
    sd.add_argument('session_id', help='Session id')
    se = sessions_subparsers.add_parser('export', help='Export a session')
    se.add_argument('session_id', help='Session id')
    se.add_argument('--format', choices=['json', 'text'], default='json')
    sc = sessions_subparsers.add_parser('cleanup', help='Delete sessions older than N days')
    sc.add_argument('--days', type=int, default=30)

    # Runs commands (agent run history)
    runs_parser = subparsers.add_parser(
        'runs', help='Browse agent run history',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen runs list --status failed --since 2026-07-17\n"
            "  effgen runs list --model gpt-5-nano --json\n"
            "  effgen runs show 3f9a1c2b\n"
            "\n"
            "Runs are appended to $EFFGEN_HOME/runs (default ~/.effgen/runs) as\n"
            "one JSONL file per day. Set EFFGEN_RUN_HISTORY=0 to keep history in\n"
            "memory only; EFFGEN_RUN_HISTORY_MAX_DAYS sets retention (default 30).\n"
        ),
    )
    runs_parser.set_defaults(_group_parser=runs_parser)
    runs_subparsers = runs_parser.add_subparsers(dest='runs_command', help='Runs command')
    rl = runs_subparsers.add_parser('list', help='List recent runs')
    rl.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')
    rl.add_argument('--status', choices=['ok', 'error', 'failed'],
                    help="Only runs with this status ('failed' is an alias for 'error')")
    rl.add_argument('--model', help='Only runs on a model matching this text')
    rl.add_argument('--search', help='Only runs whose task, answer, id or error match this text')
    rl.add_argument('--session-id', dest='session_filter', help='Only runs from this session')
    rl.add_argument('--since', help='Only runs on/after this date (YYYY-MM-DD)')
    rl.add_argument('--until', help='Only runs on/before this date (YYYY-MM-DD)')
    rl.add_argument('--limit', type=int, default=20, help='Maximum runs to show (default: 20)')
    rs = runs_subparsers.add_parser('show', help='Show one run in full')
    rs.add_argument('run_id', help='Run id (as shown by `effgen runs list`)')
    rs.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')
    rs.add_argument('--card', metavar='PATH.html',
                    help='Write a summary HTML card for this stored run to PATH. '
                         'History keeps a truncated answer and no step trace, so '
                         'the card states that; use `effgen run --card` at run time '
                         'for the full answer, trace and sources.')
    rc = runs_subparsers.add_parser('cleanup', help='Delete run history older than N days')
    rc.add_argument('--days', type=int, default=30)

    # Chat command
    chat_parser = subparsers.add_parser(
        'chat', help='Interactive chat mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen chat -m gpt-5-nano --provider openai\n"
            "  effgen chat --preset research -t calculator wikipedia\n"
            "  effgen chat --session-id support-42   # resume a saved session\n"
            "\n"
            "In-session slash commands: /help /model /tools /cost /trace /reset "
            "/save /load /doctor /exit\n"
        ),
    )
    chat_parser.add_argument('-m', '--model', help='Model to use')
    chat_parser.add_argument(
        '--provider',
        help='Provider for a bare model id (e.g. openai, groq, cerebras, gemini). '
             'Equivalent to the "provider:model" prefix.',
    )
    chat_parser.add_argument(
        '--preset', choices=_preset_choices,
        help='Agent preset for the session (e.g. math, research) — attaches the '
             "preset's tools and system prompt, same as `effgen run --preset`",
    )
    chat_parser.add_argument(
        '-t', '--tools', nargs='+', metavar='TOOL',
        help='Tools to enable for the session, same as `effgen run --tools` '
             '(e.g. calculator wikipedia). Also addable mid-session with /tools.',
    )
    chat_parser.add_argument(
        '--system-prompt', '--persona', dest='system_prompt', metavar='TEXT',
        help='Custom persona / system prompt for the session, e.g. '
             '"You are a patient Socratic tutor who never gives the answer." '
             'Steers every reply (unlike --preset, which only labels the session).',
    )
    chat_parser.add_argument(
        '--guardrails', metavar='NAME',
        help='Apply a guardrail preset to redact/block PII and screen for '
             'prompt injection on every turn: "strict", "standard" (alias '
             '"default"/"balanced"), "phi" (alias "hipaa"/"deidentify"), '
             '"minimal", or "none". Carries across a /model or /tools rebuild.',
    )
    chat_parser.add_argument('--temperature', type=float, help='Temperature')
    chat_parser.add_argument('--max-tokens', type=int,
                             help='Max output tokens per reply (raise for token-heavy '
                                  'or reasoning models, e.g. gpt-5/o-series, which '
                                  'spend part of the budget on hidden reasoning '
                                  'before any visible text)')
    chat_parser.add_argument('--no-sub-agents', action='store_true', help='Disable sub-agents')
    chat_parser.add_argument('-v', '--verbose', action='store_true', default=argparse.SUPPRESS,
                             help='Verbose output (show DEBUG/INFO logs)')
    chat_parser.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                             help='Quiet output (errors only)')
    chat_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                             help='Disable live spinners/progress animation')
    chat_parser.add_argument(
        '--session-id', '--resume', dest='session_id', metavar='ID',
        help='Continue a persistent session by id (same store as '
             '`effgen run --session-id` and `effgen sessions list`). Prior turns '
             'are recalled and new turns are saved; a new id starts a fresh session.',
    )

    # Serve command
    serve_parser = subparsers.add_parser(
        'serve', help='Start API server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Operational settings (environment variables):\n"
            "  EFFGEN_API_KEY        static API key (Bearer or X-API-Key). If unset\n"
            "                        and not in dev mode, an ephemeral key is minted\n"
            "                        and printed once — never unauthenticated.\n"
            "  EFFGEN_DEV_MODE=1     disable auth (loud warning; local dev only).\n"
            "  EFFGEN_RATE_LIMIT     requests/minute per client IP (0 disables;\n"
            "                        health probes are always exempt). Or use\n"
            "                        --rate-limit.\n"
            "  EFFGEN_TRUST_PROXY=1  trust the first X-Forwarded-For hop as the\n"
            "                        rate-limit client IP (default: off — the raw\n"
            "                        socket peer is used, since a caller can set\n"
            "                        that header to anything). Enable only behind\n"
            "                        a reverse proxy that sets/overwrites it.\n"
            "  EFFGEN_CORS_ORIGINS   comma-separated allowed origins (default: none;\n"
            "                        cross-origin is fail-closed for a backend API).\n"
            "  EFFGEN_OIDC_ISSUER /  enable OIDC/JWT auth instead of a static key.\n"
            "  EFFGEN_OIDC_CLIENT_ID\n"
            "  EFFGEN_PUBLIC_METRICS=1   serve /metrics without auth (default: auth).\n"
            "  EFFGEN_PUBLIC_DASHBOARD=1 serve /dashboard/data.json + /dashboard/spans\n"
            "                        without auth, for local viewing (default: auth;\n"
            "                        the /dashboard page itself always loads, but its\n"
            "                        data calls 401 without this or an API key).\n"
            "  EFFGEN_MODEL_POOL_SIZE    loaded models kept warm (default 4).\n"
            "  EFFGEN_NO_DOTENV=1    skip the .env filesystem search entirely, so\n"
            "                        only environment variables the orchestrator set\n"
            "                        are visible (EFFGEN_DOTENV=none is equivalent).\n"
            "\n"
            "Watching a running server: `effgen top` reads this server's\n"
            "/dashboard/data.json. Point it with --url/--port, or export\n"
            "EFFGEN_SERVER_URL so it finds this server with no flags.\n"
            "\n"
            "Scaling: `effgen serve` runs a single worker. For multiple workers,\n"
            "run the app factory under uvicorn/gunicorn, e.g.:\n"
            "  uvicorn effgen.server.app:create_app --factory --workers 4 --port 8000\n"
        ),
    )
    serve_parser.add_argument(
        '--host', default='127.0.0.1',
        help='Host to bind to (default 127.0.0.1, loopback-only). '
             'Pass --host 0.0.0.0 to expose on all interfaces (set EFFGEN_API_KEY first).',
    )
    serve_parser.add_argument('-p', '--port', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument(
        '--rate-limit', type=int, default=None, metavar='N',
        help='Requests/minute per client IP (overrides EFFGEN_RATE_LIMIT; '
             '0 disables). Health probes are always exempt.',
    )
    serve_parser.add_argument(
        '--trust-proxy', action='store_true', default=None,
        help='Trust the first X-Forwarded-For hop as the rate-limit client IP '
             '(overrides EFFGEN_TRUST_PROXY). Enable only behind a reverse '
             'proxy that sets/overwrites this header — otherwise any caller '
             'can spoof it to bypass the rate limit.',
    )

    # Live mission-control view (`top`, with `monitor` as an alias)
    from effgen.cli.monitor import (
        DEFAULT_ACTIVITY_LIMIT as _MON_LIMIT,
    )
    from effgen.cli.monitor import (
        DEFAULT_INTERVAL as _MON_INTERVAL,
    )
    _top_epilog = (
        "Examples:\n"
        "  effgen top                          # live view, refreshing in place\n"
        "  effgen top --interval 1             # faster refresh\n"
        "  effgen top --url http://host:8000   # watch a remote server\n"
        "  effgen top --once                   # one static snapshot\n"
        "  effgen top --json | jq .spend       # machine-readable snapshot\n"
        "\n"
        "Panels and their sources:\n"
        "  Activity    completed runs from the local run history (all processes)\n"
        "  Traffic     the server's own counters, since that process started\n"
        "  Per-model   per-model calls, errors, p95 latency and cost from the server\n"
        "  Spend       the local cost ledger: 24h total, daily budget, $/h burn\n"
        "  GPU         physical device memory and utilization, across all processes\n"
        "\n"
        "Each panel names the window and process it measures; figures from\n"
        "different sources are never combined. Activity, Spend and GPU need no\n"
        "server, so the view is useful on a host running only local agents; the\n"
        "server-backed panels then read as unavailable, naming the URL tried.\n"
        "\n"
        "The server URL comes from --url, --port, EFFGEN_SERVER_URL, or\n"
        "http://127.0.0.1:8000. Reading a server's data needs its API key unless\n"
        "it was started with EFFGEN_PUBLIC_DASHBOARD=1; pass --api-key or set\n"
        "EFFGEN_API_KEY.\n"
        "\n"
        "Piped output, --json, --once, --no-animation, NO_COLOR and\n"
        "EFFGEN_NO_ANIM=1 all print a single snapshot and exit instead of taking\n"
        "over the screen. On a terminal, q or Ctrl-C quits.\n"
    )
    for _top_name, _top_help in (
        ('top', 'Live terminal view of runs, traffic, spend and GPU'),
        ('monitor', 'Alias for `effgen top`'),
    ):
        _top_parser = subparsers.add_parser(
            _top_name, help=_top_help,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=_top_epilog,
        )
        _top_parser.add_argument('--json', dest='output_json', action='store_true',
                                 help='Print one snapshot as JSON and exit')
        _top_parser.add_argument('--once', action='store_true',
                                 help='Print one static snapshot and exit (no refresh loop)')
        _top_parser.add_argument('--interval', type=float, default=_MON_INTERVAL, metavar='SECONDS',
                                 help=f'Seconds between refreshes (default: {_MON_INTERVAL:g})')
        _top_parser.add_argument('--count', type=int, default=None, metavar='N',
                                 help='Stop after N refreshes (default: run until you quit)')
        _top_parser.add_argument('--limit', type=int, default=_MON_LIMIT, metavar='N',
                                 help=f'Runs to show in the activity panel (default: {_MON_LIMIT})')
        _top_parser.add_argument('--url', metavar='URL',
                                 help='Server base URL (default: EFFGEN_SERVER_URL '
                                      'or http://127.0.0.1:8000)')
        _top_parser.add_argument('-p', '--port', type=int, metavar='PORT',
                                 help='Server port on 127.0.0.1 (shorthand for --url)')
        _top_parser.add_argument('--api-key', dest='api_key', metavar='KEY',
                                 help='API key for the server (default: EFFGEN_API_KEY)')
        _top_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                                 help='Print one static snapshot instead of the live view')

    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.set_defaults(_group_parser=config_parser)
    config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config command')

    config_show = config_subparsers.add_parser('show', help='Show configuration')
    config_show.add_argument('-f', '--file', help='Configuration file')

    config_validate = config_subparsers.add_parser('validate', help='Validate configuration')
    config_validate.add_argument('-f', '--file', required=True, help='Configuration file')

    config_init = config_subparsers.add_parser('init', help='Initialize new configuration')
    config_init.add_argument('-o', '--output', help='Output file')
    config_init.add_argument('--force', action='store_true', help='Overwrite existing file')

    config_set = config_subparsers.add_parser('set', help='Set a configuration value (e.g. budget.daily 1.0)')
    config_set.add_argument('key', help='Config key (e.g. budget.daily, budget.monthly)')
    config_set.add_argument('value', help='Config value')

    # Tools commands
    tools_parser = subparsers.add_parser('tools', help='Tool management')
    tools_parser.set_defaults(_group_parser=tools_parser)
    tools_subparsers = tools_parser.add_subparsers(dest='tool_command', help='Tools command')

    tools_list = tools_subparsers.add_parser('list', help='List tools')
    tools_list.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')
    tools_list.add_argument('--category', help='Filter by category')

    tools_info = tools_subparsers.add_parser('info', help='Show tool information')
    tools_info.add_argument('name', help='Tool name')

    tools_test = tools_subparsers.add_parser('test', help='Test a tool')
    tools_test.add_argument('name', help='Tool name')
    tools_test.add_argument('-i', '--input', help='Tool input (JSON or string)')

    # Models commands
    models_parser = subparsers.add_parser('models', help='Model management')
    models_parser.set_defaults(_group_parser=models_parser)
    models_subparsers = models_parser.add_subparsers(dest='model_command', help='Models command')

    models_list = models_subparsers.add_parser('list', help='List models')
    models_list.add_argument('--provider', help='Show only this provider\'s models (full detail)')
    models_list.add_argument('--free', action='store_true', help='Show only free-tier models')
    models_list.add_argument('--tools', action='store_true', help='Show only tool-capable models')
    models_list.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')

    models_browse = models_subparsers.add_parser(
        'browse',
        help='Browse every provider in one table — search, filter, sort, page')
    models_browse.add_argument(
        '--search', metavar='TEXT',
        help='Case-insensitive substring match on model id, family, or provider')
    models_browse.add_argument('--provider', help='Limit to one provider')
    models_browse.add_argument(
        '--free', action='store_true', help='Only free-tier models')
    models_browse.add_argument(
        '--tools', action='store_true', help='Only tool-calling models')
    models_browse.add_argument(
        '--vision', action='store_true', help='Only vision-capable models')
    models_browse.add_argument(
        '--audio', action='store_true', help='Only audio-capable models')
    models_browse.add_argument(
        '--min-context', type=int, metavar='N', dest='min_context',
        help='Only models with a context window of at least N tokens')
    models_browse.add_argument(
        '--max-price-in', type=float, metavar='USD', dest='max_price_in',
        help='Only models whose input price ($/1M) is at most USD')
    models_browse.add_argument(
        '--max-price-out', type=float, metavar='USD', dest='max_price_out',
        help='Only models whose output price ($/1M) is at most USD')
    models_browse.add_argument(
        '--sort', choices=['provider', 'id', 'context', 'max-out',
                           'price-in', 'price-out'],
        default='provider',
        help='Sort order (default: provider then id)')
    models_browse.add_argument(
        '--desc', action='store_true', help='Sort in descending order')
    models_browse.add_argument(
        '--limit', type=int, metavar='N', help='Show at most N rows')
    models_browse.add_argument(
        '--offset', type=int, default=0, metavar='N',
        help='Skip the first N rows (paging)')
    models_browse.add_argument(
        '--include-local', action='store_true', dest='include_local',
        help='Also list models downloaded in the local HuggingFace cache')
    models_browse.add_argument(
        '--json', dest='output_json', action='store_true', help='Output as JSON')

    models_info = models_subparsers.add_parser('info', help='Show model information')
    models_info.add_argument('name', help='Model name (e.g. gpt-5-nano or openai:gpt-5-nano)')
    models_info.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')

    models_load = models_subparsers.add_parser('load', help='Pre-load a model into memory')
    models_load.add_argument('name', help='Model name (e.g. Qwen/Qwen2.5-1.5B-Instruct)')
    models_load.add_argument('-e', '--engine', help='Engine (vllm, transformers)', default=None)

    models_unload = models_subparsers.add_parser('unload', help='Unload a model from memory')
    models_unload.add_argument('name', help='Model name')

    models_status = models_subparsers.add_parser('status', help='Show loaded models and GPU memory status')
    models_status.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')

    models_refresh = models_subparsers.add_parser(
        'refresh', help="Refresh the model catalog from each provider's live API")
    models_refresh.add_argument(
        '--provider', default=None,
        help='Only refresh this provider (default: all providers with a key)')
    models_refresh.add_argument(
        '--dry-run', action='store_true',
        help='Show what would change without writing the snapshot')

    # Examples commands
    examples_parser = subparsers.add_parser('examples', help='Run example scripts')
    examples_parser.set_defaults(_group_parser=examples_parser)
    examples_subparsers = examples_parser.add_subparsers(dest='example_command', help='Examples command')

    examples_subparsers.add_parser('list', help='List examples')

    examples_run = examples_subparsers.add_parser('run', help='Run an example')
    examples_run.add_argument('name', help='Example name')

    # Health check command
    health_parser = subparsers.add_parser(
        'health', help='Check effGen infrastructure health (contacts external services)')
    health_parser.add_argument(
        '--remote', '--online', dest='health_remote', action='store_true',
        help='Opt in to network checks (effgen.org / PyPI). Without this, health '
             'does not contact any external service. EFFGEN_HEALTH_REMOTE=1 also enables it.')

    # Doctor command — API key availability check
    doctor_parser = subparsers.add_parser('doctor', help='Check provider API key availability')
    doctor_parser.add_argument('--json', dest='output_json', action='store_true',
                               help='Output as JSON')
    doctor_parser.add_argument('--provider', dest='doctor_provider',
                               help='Check a specific provider only')
    doctor_parser.add_argument('--live', action='store_true',
                               help='Make a tiny live call per keyed provider to confirm the '
                                    'default model is actually usable (not just that a key exists)')
    doctor_parser.add_argument('--cheap', action='store_true',
                               help='With --live, use the cheapest/default model and minimal tokens')

    # Plugin commands
    plugin_parser = subparsers.add_parser('create-plugin', help='Generate a plugin project scaffold')
    plugin_parser.add_argument('plugin_name', help='Plugin name (e.g. my_tools)')
    plugin_parser.add_argument('-o', '--output-dir', default='.', help='Output directory')

    # Presets command
    presets_parser = subparsers.add_parser('presets', help='List available agent presets')
    presets_parser.add_argument('--json', dest='output_json', action='store_true',
                                help='Output the preset list as JSON')

    # Quickstart / tutorial — a short guided first run. `tutorial` is a
    # documented alias so both names a newcomer might try lead to the same
    # guided run rather than one dead-ending.
    _qs_help = {
        'quickstart': 'Guided first run: pick a model, run an agent, see the trace and cost',
        'tutorial': 'Alias of quickstart — the same guided first run',
    }
    for _qs_name in ('quickstart', 'tutorial'):
        qs_parser = subparsers.add_parser(
            _qs_name,
            help=_qs_help[_qs_name],
            description=_qs_help['quickstart']
            + ('.  (`effgen tutorial` is an alias of `effgen quickstart`.)'
               if _qs_name == 'tutorial' else '.'),
        )
        qs_parser.add_argument('-m', '--model', help='Model to use (skips the model prompt)')
        qs_parser.add_argument('--provider', help='Provider for a bare model id')
        qs_parser.add_argument('--task', help='Task to run (defaults to a sample task)')
        qs_parser.add_argument('-y', '--yes', action='store_true',
                               help='Run non-interactively with sensible defaults')

    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Run a DAG-based workflow')
    workflow_parser.set_defaults(_group_parser=workflow_parser)
    workflow_subparsers = workflow_parser.add_subparsers(dest='workflow_command', help='Workflow command')

    workflow_run = workflow_subparsers.add_parser('run', help='Run a workflow from YAML file')
    workflow_run.add_argument('file', help='Path to workflow YAML file')
    workflow_run.add_argument('-m', '--model', help='Default model for all agents')
    workflow_run.add_argument('--input', action='append', nargs=2, metavar=('NODE', 'TASK'),
                              help='Input for a specific node (can be repeated)')
    workflow_run.add_argument('--task', help='A single task string routed to the '
                              'workflow entry node(s) (alternative to --input)')
    workflow_run.add_argument('--json', dest='output_json', action='store_true',
                              help='Emit the workflow result as JSON to stdout (for CI gating)')
    workflow_run.add_argument('--diagram', action='store_true',
                              help='Draw the workflow as a dependency graph (nodes by '
                                   'level, edges, per-node status/duration/cost)')
    workflow_run.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                              help='Quiet output (errors only); --json still emits to stdout')

    workflow_validate = workflow_subparsers.add_parser('validate', help='Validate a workflow YAML file')
    workflow_validate.add_argument('file', help='Path to workflow YAML file')
    workflow_validate.add_argument('--json', dest='output_json', action='store_true',
                                   help='Emit the validation result as JSON to stdout')
    workflow_validate.add_argument('--diagram', action='store_true',
                                   help='Draw the workflow dependency graph (nodes by level, edges)')
    workflow_validate.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                                   help='Quiet output (errors only); --json still emits to stdout')

    # Batch command
    batch_parser = subparsers.add_parser(
        'batch', help='Run batch queries from a file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen batch queries.jsonl -o answers.jsonl -m gpt-5-nano\n"
            "  effgen batch -i rows.csv --query-field question -o out.csv --excel\n"
            "  effgen batch prompts.txt --json -q | jq '.rows[].output'\n"
        ),
    )
    batch_parser.add_argument('input_file', nargs='?', default=None, metavar='INPUT',
                              help='Input file (JSONL, CSV, JSON, or plain text). '
                                   'Same as -i/--input.')
    batch_parser.add_argument('-i', '--input', help='Input file (JSONL, CSV, JSON, or plain text)')
    batch_parser.add_argument('-o', '--output',
                              help='Output file (JSONL, CSV, or JSON). .jsonl rows are '
                                   'written as each query finishes, so their file order '
                                   'is completion order, not input order, at any '
                                   '--concurrency above 1; .csv/.json rows are written '
                                   'once at the end in input order. Every row carries an '
                                   '"index" field back to its input position — sort on '
                                   'it if your consumer assumes line N corresponds to '
                                   'input row N.')
    batch_parser.add_argument('--concurrency', type=int, default=5, help='Max concurrent queries (default: 5)')
    batch_parser.add_argument('--batch-size', type=int, default=0, help='Batch size (0 = all at once)')
    batch_parser.add_argument('--timeout', type=float, default=120.0, help='Timeout per query in seconds')
    batch_parser.add_argument('--retries', type=int, default=1, help='Retries for failed queries')
    batch_parser.add_argument('-m', '--model', help='Model to use')
    batch_parser.add_argument('--preset', choices=_preset_choices,
                              help='Use a preset agent configuration')
    batch_parser.add_argument(
        '--guardrails', metavar='NAME',
        help='Apply a guardrail preset to redact/block PII and screen for '
             'prompt injection on every row: "strict", "standard" (alias '
             '"default"/"balanced"), "phi" (alias "hipaa"/"deidentify"), '
             '"minimal", or "none".',
    )
    batch_parser.add_argument(
        '--system-prompt', '--persona', dest='system_prompt', metavar='TEXT',
        help='System prompt applied to every row, e.g. a target language, '
             'glossary, and tone instruction for a localization batch '
             '("Translate into formal European French (vous); keep {placeholders} '
             'and HTML tags verbatim."). Overrides the preset\'s default prompt.',
    )
    batch_parser.add_argument('--query-field', default='query', help='Field name for queries in JSONL/CSV (default: query)')
    batch_parser.add_argument('--max-tokens', type=int, default=None,
                              help='Max output tokens per query (raise for token-heavy or reasoning models)')
    batch_parser.add_argument('--temperature', type=float, default=None,
                              help='Sampling temperature per query (0 for deterministic reruns where the provider supports it)')
    batch_parser.add_argument('--schema', dest='schema_path', default=None,
                              help='JSON Schema file; each row is validated against it and its parsed object is written')
    batch_parser.add_argument('--output-model', dest='output_model', default=None,
                              help='Pydantic model as module:ClassName to validate each row against')
    batch_parser.add_argument('--strict', action='store_true',
                              help='Abort on the first malformed input line instead of skipping it')
    batch_parser.add_argument('--resume', action='store_true',
                              help='Skip input rows already present in the JSONL --output file and append the rest')
    batch_parser.add_argument(
        '--excel', '--bom', dest='excel_bom', action='store_true',
        help='Prepend a UTF-8 BOM to CSV output so Excel on Windows opens '
             'non-Latin scripts (Arabic, CJK, Devanagari, ...) correctly on '
             'double-click. Only affects --output ending in .csv.',
    )
    batch_parser.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                              help='Quiet output (suppress the progress bar)')
    batch_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                              help='Disable the live progress bar (plain output)')
    batch_parser.add_argument('--json', dest='output_json', action='store_true',
                              help='Emit the job summary and every row as a JSON '
                                   'document to stdout (for piping to jq), in addition '
                                   'to any -o file. Human output goes to stderr; '
                                   'combine with -q for clean stdout.')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate an agent against a test suite')
    eval_parser.add_argument('--suite', required=True,
                              help='Built-in suite name (math, tool_use, reasoning, safety, '
                                   'conversation) OR a path to your own .jsonl/.json test cases')
    eval_parser.add_argument('-m', '--model', help='Model to use')
    eval_parser.add_argument(
        '--provider',
        help='Provider for a bare model id (e.g. openai, groq, cerebras, gemini, '
             'together, fireworks, replicate, anthropic, hf). '
             'Equivalent to the "provider:model" prefix.',
    )
    eval_parser.add_argument('--preset', choices=_preset_choices,
                              help='Use a preset agent configuration')
    eval_parser.add_argument('--scoring', choices=['exact_match', 'contains', 'regex', 'semantic_similarity', 'llm_judge'],
                              default='contains', help='Scoring mode (default: contains)')
    eval_parser.add_argument('--threshold', type=float, default=0.5,
                              help='Per-case pass score for continuous scoring modes '
                                   '(semantic_similarity, llm_judge); has no effect on '
                                   'exact_match/contains/regex, whose scores are already binary '
                                   '(0 or 1) (default: 0.5). Use --fail-under to gate the exit '
                                   'code on suite accuracy.')
    eval_parser.add_argument('--fail-under', type=float, default=0.5, metavar='ACCURACY',
                              help='Minimum suite accuracy required for a zero exit code '
                                   '(default: 0.5). This is the CI gate; a --compare-baseline '
                                   'regression always fails regardless of this value.')
    eval_parser.add_argument('--temperature', type=float, default=None,
                              help='Sampling temperature for the evaluated agent (0 for '
                                   'deterministic, reproducible scoring where the provider '
                                   'supports it; default: the model/preset default)')
    eval_parser.add_argument('--save-baseline', action='store_true',
                              help='Save results as regression baseline')
    eval_parser.add_argument('--compare-baseline', action='store_true',
                              help='Compare results against stored baseline')
    eval_parser.add_argument('--baseline-dir', dest='baseline_dir', default=None, metavar='DIR',
                              help='Directory for --save-baseline/--compare-baseline files '
                                   '(default: ./.effgen/baselines under the current directory, '
                                   'created if missing). A baseline saved under the installed '
                                   'package tree by an older effGen version is still read.')
    eval_parser.add_argument('-o', '--output',
                              help='Output file for results. The extension chooses the '
                                   'format: .html renders the shareable report, .md writes '
                                   'Markdown, anything else writes JSON.')
    eval_parser.add_argument('--report', metavar='PATH.html',
                              help='Write a self-contained HTML report to PATH — pass rate, '
                                   'exit gate, by-difficulty breakdown, and every case. The '
                                   'file opens offline with no external references.')
    eval_parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                              help='Filter test cases by difficulty')
    eval_parser.add_argument('--max-cases', type=int, default=None,
                              help='Only run the first N cases (quick subsample)')
    eval_parser.add_argument('--json', dest='output_json', action='store_true',
                              help='Emit the results object as JSON to stdout (for CI gating)')
    eval_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                              help='Disable the live progress bar (plain output)')

    # Compare command
    compare_parser = subparsers.add_parser(
        'compare', help='Compare multiple models on a test suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen compare --models gpt-5-nano,groq:llama-3.1-8b-instant --suite math\n"
            "  effgen compare --models gpt-5-nano,gpt-5-mini --suite reasoning --optimize cost\n"
            "  effgen compare --models a,b --suite ./cases.jsonl --json | jq .recommendations\n"
            "\n"
            "compare reports a bake-off and always exits 0; use `eval --fail-under` "
            "to gate a build.\n"
        ),
    )
    compare_parser.add_argument('--models', required=True,
                                 help='Comma-separated model ids. Use a '
                                      'provider:model prefix to pin a provider '
                                      'for a bare id (e.g. '
                                      'groq:llama-3.1-8b-instant,gpt-5-nano).')
    compare_parser.add_argument('--suite', required=True,
                                 help='Built-in suite name (math, tool_use, '
                                      'reasoning, safety, conversation) OR a path '
                                      'to your own .jsonl/.json test cases')
    compare_parser.add_argument('--scoring', choices=['exact_match', 'contains', 'regex', 'semantic_similarity', 'llm_judge'],
                                 default='contains', help='Scoring mode (default: contains)')
    compare_parser.add_argument('--threshold', type=float, default=0.5,
                                 help='Per-case pass score for continuous scoring modes '
                                      '(semantic_similarity, llm_judge); has no effect on '
                                      'exact_match/contains/regex, whose scores are already '
                                      'binary (0 or 1) (default: 0.5). compare always exits 0 — '
                                      'it reports a bake-off rather than gating a build; use '
                                      '`eval --fail-under` for CI gating.')
    compare_parser.add_argument('--temperature', type=float, default=None,
                                 help='Sampling temperature for every compared model (0 for '
                                      'deterministic, reproducible scoring where the provider '
                                      'supports it; default: the model/preset default)')
    compare_parser.add_argument(
        '--provider',
        help='Provider applied to any bare model id in --models that has no '
             '"provider:" prefix of its own (e.g. openai, groq, cerebras, gemini, '
             'together, fireworks, replicate, anthropic, hf).',
    )
    compare_parser.add_argument('--max-cases', type=int, default=None,
                                 help='Only run the first N cases (quick bake-off '
                                      'on a big suite)')
    compare_parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                                 help='Filter test cases by difficulty')
    compare_parser.add_argument('-o', '--output',
                                 help='Output file for results. The extension chooses the '
                                      'format: .html renders the shareable report, .md writes '
                                      'Markdown, anything else writes JSON.')
    compare_parser.add_argument('--report', metavar='PATH.html',
                                 help='Write a self-contained HTML report to PATH — the '
                                      'recommended model and why, a per-model table, and '
                                      'accuracy/cost/latency charts. The file opens offline '
                                      'with no external references.')
    compare_parser.add_argument('--json', dest='output_json', action='store_true',
                                 help='Emit the comparison matrix as JSON to stdout (for CI gating)')
    compare_parser.add_argument('--preset', choices=_preset_choices,
                                 help='Use a preset agent configuration')
    compare_parser.add_argument('--optimize', choices=['accuracy', 'cost', 'latency'],
                                 default='accuracy',
                                 help="What the recommendation optimizes for (default: accuracy — "
                                      "highest accuracy, tie-broken on lower latency then fewer "
                                      "tokens). 'cost'/'latency' recommend the cheapest/fastest "
                                      "model among those meeting --threshold accuracy (falling "
                                      "back to the full field if none qualify), tie-broken on "
                                      "higher accuracy.")
    compare_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                                 help='Disable the live progress bar (plain output)')
    compare_parser.add_argument('--judge', metavar='MODEL',
                                 help='Model that grades answers under --scoring llm_judge. '
                                      'Without it each model grades its own answers; naming a '
                                      'judge has one model grade the whole field. The judge is '
                                      'named in the output.')

    # Battle command — several models racing one prompt
    battle_parser = subparsers.add_parser(
        'battle', help='Race several models on one prompt, side by side',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen battle \"Explain a B-tree in two sentences.\" \\\n"
            "      -m openai:gpt-5-nano,groq:llama-3.1-8b-instant\n"
            "  effgen battle \"Write a haiku about caching.\" -m a,b,c --judge openai:gpt-5-mini\n"
            "  effgen battle \"...\" -m a,b --json | jq '.contenders[].cost_usd'\n"
            "  effgen battle \"...\" -m a,b --report battle.html\n"
            "\n"
            "Every model answers the same prompt at once. On a terminal the answers\n"
            "stream side by side, each column showing its own time to first token,\n"
            "elapsed time, tokens and cost, and a verdict panel closes the race.\n"
            "\n"
            "The verdict reports what was measured — fastest, cheapest, longest —\n"
            "and needs no judge. --judge names a separate model to pick a winner on\n"
            "quality; that pick is reported apart from the measurements and names\n"
            "the judge. A model that fails is reported as failed and cannot win.\n"
            "\n"
            "Piped output, --json, --no-animation and NO_COLOR skip the live view\n"
            "and print one structured result carrying every model's full answer.\n"
        ),
    )
    battle_parser.add_argument('prompt', help='The prompt every model answers')
    battle_parser.add_argument('-m', '--models', required=True, metavar='A,B[,C]',
                                help='Comma-separated model ids to race (at least two), '
                                     'e.g. openai:gpt-5-nano,groq:llama-3.1-8b-instant')
    battle_parser.add_argument('--judge', metavar='MODEL',
                                help='Model asked to pick the best answer. Optional — the '
                                     'measured outcomes need no judge.')
    battle_parser.add_argument('--temperature', type=float, default=None,
                                help='Sampling temperature applied to every model')
    battle_parser.add_argument('--max-tokens', type=int, default=None, dest='max_tokens',
                                help='Output token cap applied to every model')
    battle_parser.add_argument('--system-prompt', dest='system_prompt', metavar='TEXT',
                                help='System prompt applied to every model')
    battle_parser.add_argument('-o', '--output', metavar='PATH',
                                help='Save the battle. The extension chooses the format: '
                                     '.md writes Markdown, anything else writes JSON.')
    battle_parser.add_argument('--report', metavar='PATH.html',
                                help='Write a self-contained HTML report of the battle')
    battle_parser.add_argument('--json', dest='output_json', action='store_true',
                                help='Print the battle as JSON to stdout (no live view)')
    battle_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                                help='Skip the live side-by-side view and print the result')

    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Run an agent in interactive debug mode')
    debug_parser.add_argument('task', help='Task to execute')
    debug_parser.add_argument('-m', '--model', help='Model to use')
    debug_parser.add_argument('--provider',
                              help='Provider for a bare model id (e.g. groq). '
                                   'Equivalent to the provider:model prefix.')
    debug_parser.add_argument('--preset', choices=_preset_choices,
                              help='Use a preset agent configuration')
    debug_parser.add_argument('--step', action='store_true', help='Step through each iteration')

    # Cost command — spend dashboard + budget management
    _cost_output_help = ('Output file for the spend summary. The extension chooses the '
                         'format: .html renders the shareable report, anything else '
                         'writes JSON.')
    _cost_report_help = ('Write a self-contained HTML spend report to PATH — total against '
                         'the daily budget, a per-provider/model table, and cost-share '
                         'charts. The file opens offline with no external references.')
    cost_parser = subparsers.add_parser('cost', help='View cost spend and manage budgets')
    cost_parser.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')
    cost_parser.add_argument('-o', '--output', help=_cost_output_help)
    cost_parser.add_argument('--report', metavar='PATH.html', help=_cost_report_help)
    cost_subparsers = cost_parser.add_subparsers(dest='cost_command', help='Cost command')
    for _cost_sub, _cost_help in (
        ('today', 'Show per-provider/model spend for the last 24 hours'),
        ('week', 'Show rolling 7-day spend summary'),
        ('by-provider', 'Show lifetime totals grouped by provider'),
    ):
        _cost_period = cost_subparsers.add_parser(_cost_sub, help=_cost_help)
        _cost_period.add_argument('--json', dest='output_json', action='store_true',
                                  default=argparse.SUPPRESS, help='Output as JSON')
        _cost_period.add_argument('-o', '--output', default=argparse.SUPPRESS,
                                  help=_cost_output_help)
        _cost_period.add_argument('--report', metavar='PATH.html',
                                  default=argparse.SUPPRESS, help=_cost_report_help)
    cost_set_budget = cost_subparsers.add_parser('set-budget', help='Set a daily spend budget')
    cost_set_budget.add_argument('amount', type=float, help='Daily budget in USD (e.g. 1.0)')
    cost_subparsers.add_parser('clear-budget', help='Remove configured budget limits')

    # Report command — render a saved result JSON as a shareable HTML file
    from effgen.ui.report_html import REPORT_KINDS as _REPORT_KINDS
    report_parser = subparsers.add_parser(
        'report',
        help='Render a saved run/compare/eval/cost/loadtest JSON result as an HTML report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen eval --suite math --json > eval.json && effgen report eval.json\n"
            "  effgen run 'summarize this' -o run.json && effgen report run.json\n"
            "  effgen report bakeoff.json -o bakeoff.html\n"
            "  effgen report spend.json --kind cost\n"
            "\n"
            "The report kind is inferred from the JSON shape; --kind overrides it.\n"
            "A document that carries none of the fields the kind renders is\n"
            "refused, and no file is written.\n"
            "The written file is self-contained and opens with no network access.\n"
        ),
    )
    report_parser.add_argument('result',
                               help='Path to a JSON result saved from run/compare/eval/cost/loadtest')
    report_parser.add_argument('-o', '--output', metavar='PATH.html',
                               help='Where to write the HTML report '
                                    '(default: the result path with an .html extension)')
    report_parser.add_argument('--kind', choices=list(_REPORT_KINDS),
                               help='Report shape to render, when the JSON cannot be identified')

    # Prompt library subcommand
    prompts_parser = subparsers.add_parser('prompts', help='Prompt library management')
    prompts_subparsers = prompts_parser.add_subparsers(dest='prompts_command', help='Prompts command')

    prompts_list = prompts_subparsers.add_parser('list', help='List prompt templates')
    prompts_list.add_argument('--domain', help='Filter by domain')
    prompts_list.add_argument('--variant', help='Filter by variant')
    prompts_list.add_argument('--format', choices=['table', 'json', 'markdown'], default='table',
                              dest='list_format', help='Output format')
    prompts_list.add_argument('--json', action='store_const', const='json', dest='list_format',
                              help='Shorthand for --format json (consistent with `models list`, `cost`).')

    prompts_show = prompts_subparsers.add_parser('show', help='Show prompt details')
    prompts_show.add_argument('name', help='Prompt name')

    prompts_eval = prompts_subparsers.add_parser('eval', help='Evaluate prompts')
    prompts_eval.add_argument('--domain', help='Evaluate only this domain')
    prompts_eval.add_argument('--live', action='store_true', help='Run live model evaluation')
    prompts_eval.add_argument('-m', '--model', help='Model to use for live evaluation')
    prompts_eval.add_argument('--delay', type=float, default=35.0,
                              help='Seconds to wait between live model calls (default: 35)')
    prompts_eval.add_argument('--output', help='Write eval table to this file')
    prompts_eval.add_argument('--fail-under', type=float, default=None, metavar='FRACTION',
                              help='Exit non-zero if the pass rate is below this fraction '
                                   '(0.0-1.0). Without it, any failing eval exits non-zero.')

    # Playground subcommands
    prompts_subparsers.add_parser('playground', help='Launch interactive prompt playground REPL')

    prompts_render = prompts_subparsers.add_parser('render', help='Non-interactive: render a prompt to stdout')
    prompts_render.add_argument('prompt_name', metavar='name', help='Prompt name (e.g. research.literature_review.v1)')
    prompts_render.add_argument('--input', dest='input_file', metavar='FILE',
                                help="JSON file with input variables, validated against the prompt's "
                                     "input_schema (see 'prompts show <name>'); omit to render the fixture")

    prompts_run = prompts_subparsers.add_parser('run', help='Non-interactive: render + run through a model')
    prompts_run.add_argument('prompt_name', metavar='name', help='Prompt name')
    prompts_run.add_argument('--input', dest='input_file', metavar='FILE',
                             help="JSON file with input variables, validated against the prompt's "
                                  "input_schema (see 'prompts show <name>'); omit to render the fixture")
    prompts_run.add_argument('-m', '--model', required=True, help='Model identifier to run against')
    prompts_run.add_argument('--max-tokens', type=int, default=None,
                             help='Completion token cap for this run (raise it when a reasoning '
                                  'or structured prompt returns empty/truncated output)')
    prompts_run.add_argument('--temperature', type=float, default=None,
                             help='Sampling temperature for this run')

    # Load test command
    from effgen.cli.loadtest import add_loadtest_subparser  # noqa: PLC0415
    add_loadtest_subparser(subparsers)

    return parser


def _render_plugin_template(filename: str, replacements: dict[str, str]) -> str:
    """Load a scaffold template from package data and substitute placeholders.

    Templates live in ``effgen/cli/_templates/plugin/`` (shipped as package data)
    rather than being embedded here, so the generated plugin stays in sync with
    BaseTool and is covered by a create→install→import→run test.
    """
    from importlib import resources

    # Anchor on the real ``effgen.cli`` package and traverse into the data dir,
    # which works for both editable checkouts and installed wheels.
    text = (
        resources.files("effgen.cli")
        .joinpath("_templates", "plugin", filename)
        .read_text(encoding="utf-8")
    )
    for token, value in replacements.items():
        text = text.replace(token, value)
    return text


def _create_plugin_scaffold(plugin_name: str, output_dir: str = ".") -> int:
    """Generate a plugin project scaffold."""
    # Normalize to a valid Python package name (entry points + imports need it).
    pkg_name = plugin_name.replace("-", "_")
    if not pkg_name.isidentifier():
        print(
            f"Error: '{plugin_name}' is not a valid plugin name. Use letters, "
            "digits and underscores (must start with a letter)."
        )
        return 1

    plugin_class = pkg_name.title().replace("_", "")
    replacements = {
        "__PLUGIN_NAME__": pkg_name,
        "__PLUGIN_CLASS__": plugin_class,
    }

    base = Path(output_dir) / f"effgen-plugin-{pkg_name}"
    pkg = base / pkg_name
    try:
        pkg.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Error: Directory {base} already exists.")
        return 1

    try:
        (pkg / "__init__.py").write_text(_render_plugin_template("init.py.tmpl", replacements))
        (pkg / "tools.py").write_text(_render_plugin_template("tools.py.tmpl", replacements))
        (pkg / "plugin.py").write_text(_render_plugin_template("plugin.py.tmpl", replacements))
        (base / "pyproject.toml").write_text(_render_plugin_template("pyproject.toml.tmpl", replacements))
        (base / "README.md").write_text(_render_plugin_template("README.md.tmpl", replacements))
    except Exception as e:
        print(f"Error: failed to write scaffold files: {e}")
        return 1

    print(f"Created plugin scaffold at {base}/")
    print(f"  {pkg / 'tools.py'}       — add your custom tools here")
    print(f"  {pkg / 'plugin.py'}     — register tools in the plugin class")
    print(f"  {base / 'pyproject.toml'} — package metadata & entry point")
    print("\nNext: cd into it and `pip install -e .` to register the plugin.")
    return 0


def _handle_doctor_command(args) -> int:
    """Handle the 'effgen doctor' subcommand — check API key availability."""
    import json as _json

    # Load .env from the documented search paths before checking keys.
    load_env_files()

    from effgen.models.auth import check_keys
    from effgen.models.registry import ProviderRegistry

    # Ensure all adapters are imported so they self-register
    try:
        import effgen.models.anthropic_adapter  # noqa: F401
        import effgen.models.cerebras_adapter  # noqa: F401
        import effgen.models.fireworks_adapter  # noqa: F401
        import effgen.models.gemini_adapter  # noqa: F401
        import effgen.models.groq_adapter  # noqa: F401
        import effgen.models.hf_inference_adapter  # noqa: F401
        import effgen.models.openai_adapter  # noqa: F401
        import effgen.models.replicate_adapter  # noqa: F401
        import effgen.models.together_adapter  # noqa: F401
    except Exception:
        pass

    provider_filter = getattr(args, 'doctor_provider', None)
    providers_to_check = [provider_filter] if provider_filter else None

    results = check_keys(providers_to_check)

    live = bool(getattr(args, 'live', False))

    # Optional live usability probe: a tiny call per keyed provider that tells
    # "key present" apart from "default model actually callable".
    live_results: dict[str, dict] = {}
    if live:
        live_results = _doctor_live_probe(
            [p for p in results if results[p].get("available")]
        )
        for prov, lr in live_results.items():
            results[prov]["live"] = lr

    # System / CUDA / vLLM / pip-check report.
    system_report = _doctor_system_report(include_pip_check=live)

    # Circuit-breaker/bulkhead state for any provider that has been routed
    # through effgen.reliability middleware this process — surfaces an open
    # circuit or a saturated bulkhead without the caller instrumenting their
    # own code.
    reliability_report = _doctor_reliability_report()

    # Exit nonzero if a live probe was requested and a keyed provider failed.
    # Computed once so every output format (JSON and human) agrees.
    exit_code = _doctor_exit_code(results, live)

    if getattr(args, 'output_json', False):
        print(_json.dumps({
            "providers": results,
            "system": system_report,
            "reliability": reliability_report,
        }, indent=2))
        return exit_code

    # Pretty-print
    if RICH_AVAILABLE:
        console = _get_console()
        table = Table(title="effgen doctor — Provider Status")
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Key", style="white")
        table.add_column("Env Var", style="dim")
        table.add_column("Models", style="dim", justify="right")
        if live:
            table.add_column("Live", style="white")
            table.add_column("Default Model", style="magenta", overflow="fold")

        for prov in sorted(results):
            info = results[prov]
            available = info.get("available", False)
            env_key = info.get("env_key") or "—"
            status = "[green]present[/green]" if available else "[red]missing[/red]"
            try:
                n_models = str(len(ProviderRegistry.list_models(prov)))
            except Exception:
                n_models = "?"
            row = [prov, status, env_key, n_models]
            if live:
                lr = info.get("live", {})
                if not available:
                    row += ["[dim]—[/dim]", "—"]
                elif lr.get("ok"):
                    row += ["[green]usable[/green]", lr.get("model", "—")]
                else:
                    row += [f"[red]{lr.get('status', 'fail')}[/red]", lr.get("model", "—")]
            table.add_row(*row)

        console.print(table)

        if live:
            console.print("\n[bold]Live probe[/bold] — a tiny call confirms the default "
                          "model is callable (not just that a key is set).")
            for prov in sorted(live_results):
                lr = live_results[prov]
                if not lr.get("ok") and lr.get("detail"):
                    console.print(f"  [yellow]{prov}[/yellow]: {lr['detail']}")

        # System section
        console.print("\n[bold cyan]System[/bold cyan]")
        sys_table = Table(show_header=False)
        sys_table.add_column("Check", style="cyan")
        sys_table.add_column("Value", style="white", overflow="fold")
        for k, v in system_report.items():
            sys_table.add_row(k, str(v))
        console.print(sys_table)

        # Reliability section — only shown once a provider has actually been
        # routed through circuit-breaker/bulkhead middleware this process.
        if reliability_report:
            console.print("\n[bold cyan]Reliability[/bold cyan]")
            rel_table = Table(show_header=True)
            rel_table.add_column("Provider", style="cyan", no_wrap=True)
            rel_table.add_column("Circuit", style="white")
            rel_table.add_column("Bulkhead", style="white")
            for prov, rec in sorted(reliability_report.items()):
                cb = rec.get("circuit_breaker")
                bh = rec.get("bulkhead")
                if cb is None:
                    circuit_cell = "[dim]—[/dim]"
                elif cb["state"] == "closed":
                    circuit_cell = "[green]closed[/green]"
                elif cb["state"] == "half_open":
                    circuit_cell = "[yellow]half_open[/yellow]"
                else:
                    circuit_cell = "[red]open[/red]"
                if bh is None:
                    bulkhead_cell = "[dim]—[/dim]"
                else:
                    bulkhead_cell = f"active={bh['active']}/{bh['max_concurrency']}, queued={bh['queued']}/{bh['queue_size']}"
                rel_table.add_row(prov, circuit_cell, bulkhead_cell)
            console.print(rel_table)

        # Print hints for missing keys
        missing = [p for p, i in results.items() if not i.get("available")]
        if missing:
            console.print("\n[yellow]Missing keys — set in ~/.effgen/.env or export:[/yellow]")
            for prov in missing:
                keys = results[prov].get("env_keys_checked", [])
                key_str = " or ".join(keys) if keys else f"{prov.upper()}_API_KEY"
                console.print(f"  export {key_str}=<your-key>")
    else:
        print("effgen doctor — Provider Status")
        print("-" * 50)
        for prov in sorted(results):
            info = results[prov]
            available = info.get("available", False)
            env_key = info.get("env_key") or "not set"
            status = "key present" if available else "key missing"
            line = f"  {prov:12s} {status:12s}  (env: {env_key})"
            if live and available:
                lr = info.get("live", {})
                line += f"  live={'usable' if lr.get('ok') else lr.get('status', 'fail')}"
            print(line)
        print("\nSystem:")
        for k, v in system_report.items():
            print(f"  {k}: {v}")
        if reliability_report:
            print("\nReliability:")
            for prov, rec in sorted(reliability_report.items()):
                cb = rec.get("circuit_breaker")
                bh = rec.get("bulkhead")
                circuit_str = cb["state"] if cb else "—"
                bulkhead_str = (
                    f"active={bh['active']}/{bh['max_concurrency']}, queued={bh['queued']}/{bh['queue_size']}"
                    if bh else "—"
                )
                print(f"  {prov:12s} circuit={circuit_str:10s} bulkhead={bulkhead_str}")
        missing = [p for p, i in results.items() if not i.get("available")]
        if missing:
            print("\nMissing keys — set in ~/.effgen/.env or export:")
            for prov in missing:
                keys = results[prov].get("env_keys_checked", [])
                key_str = " or ".join(keys) if keys else f"{prov.upper()}_API_KEY"
                print(f"  export {key_str}=<your-key>")

    return exit_code


def _doctor_exit_code(results: dict[str, dict], live: bool) -> int:
    """Exit code for `effgen doctor`: nonzero iff a live probe was requested and
    a keyed (key-present) provider's default model was not actually usable.

    Kept format-independent so `--json` and the human table return the same code
    for the same provider state.
    """
    if live and any(
        results[p].get("available") and not results[p].get("live", {}).get("ok")
        for p in results
    ):
        return 1
    return 0


def _doctor_reliability_report() -> dict[str, dict]:
    """Circuit-breaker/bulkhead state for providers routed through reliability
    middleware this process, keyed by provider name (empty if none have).

    A provider only appears once ``ProviderRegistry.get_circuit_breaker``/
    ``get_bulkhead`` has been used for it — no calls yet made means no state
    to report, which is the common case for a fresh CLI invocation.
    """
    try:
        from effgen.models.registry import ProviderRegistry

        stats = ProviderRegistry.reliability_stats()
    except Exception:
        return {}
    return {
        prov: rec
        for prov, rec in stats.items()
        if rec.get("circuit_breaker") is not None or rec.get("bulkhead") is not None
    }


def _doctor_system_report(*, include_pip_check: bool = False) -> dict[str, Any]:
    """Collect a CUDA / torch / vLLM / pip-check diagnostic snapshot."""
    report: dict[str, Any] = {}
    try:
        from effgen.gpu import cuda_compat
        status = cuda_compat.get_cuda_status()
        report["Physical GPUs (NVML)"] = status.physical_gpus
        report["Driver CUDA"] = status.driver_cuda or "n/a"
        report["torch CUDA build"] = status.torch_cuda or "cpu-only"
        report["torch.cuda.is_available()"] = status.usable
        if status.mismatch:
            report["CUDA mismatch"] = "YES — GPUs present but torch runs on CPU"
    except Exception as e:  # noqa: BLE001
        report["CUDA"] = f"unavailable ({e})"

    # torch version
    try:
        import torch
        report["torch"] = torch.__version__
    except Exception:
        report["torch"] = "not installed"

    # vLLM import status (a frequent ABI casualty)
    try:
        import importlib.util
        if importlib.util.find_spec("vllm") is None:
            report["vLLM"] = "not installed"
        else:
            try:
                import vllm  # noqa: F401
                report["vLLM"] = f"importable ({getattr(vllm, '__version__', '?')})"
            except Exception as e:  # noqa: BLE001
                report["vLLM"] = f"installed but import failed ({type(e).__name__})"
    except Exception:
        report["vLLM"] = "unknown"

    if include_pip_check:
        try:
            import subprocess
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0:
                report["pip check"] = "no broken requirements"
            else:
                lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
                report["pip check"] = f"{len(lines)} issue(s): " + "; ".join(lines[:3])
        except Exception as e:  # noqa: BLE001
            report["pip check"] = f"could not run ({e})"

    return report


def _doctor_live_probe(providers: list[str], *, timeout: float = 30.0) -> dict[str, dict]:
    """Make a tiny live call per provider to confirm its default model is usable.

    Returns ``{provider: {"ok": bool, "model": str, "status": str, "detail": str}}``.
    Runs providers concurrently with a bounded wall-clock budget so the command
    stays responsive even if one provider hangs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from effgen.models import _catalog

    def _probe_one(prov: str) -> dict:
        model_id = _catalog.default_model(prov)
        out = {"ok": False, "model": model_id or "—", "status": "no-default", "detail": ""}
        if not model_id:
            out["detail"] = "no default model in catalog"
            return out
        try:
            from effgen import load_model
            model = load_model(model_id, provider=prov)
            model.load()
            # Keep it minimal but DON'T force max_tokens: newer reasoning models
            # (e.g. OpenAI gpt-5.x) reject max_tokens and need a token budget for
            # reasoning, so a hard cap of 1 produces a false "error". A one-word
            # reply to "Reply with: ok" is already negligibly cheap.
            resp = model.generate("Reply with the single word: ok", temperature=0.0)
            text = getattr(resp, "content", None) or getattr(resp, "text", "") or str(resp)
            out["ok"] = True
            out["status"] = "usable"
            out["detail"] = (text or "").strip()[:40]
        except Exception as e:  # noqa: BLE001 - classify for a friendly status
            from effgen.models.errors import (
                ModelAuthError,
                ModelNotFoundError,
            )
            if isinstance(e, ModelAuthError):
                out["status"] = "auth-failed"
            elif isinstance(e, ModelNotFoundError):
                out["status"] = "model-404"
            else:
                out["status"] = "error"
            # Message is already redacted by the typed errors / adapters.
            out["detail"] = str(e)[:160]
        return out

    results: dict[str, dict] = {}
    if not providers:
        return results
    with ThreadPoolExecutor(max_workers=min(len(providers), 6)) as pool:
        futs = {pool.submit(_probe_one, p): p for p in providers}
        for fut in as_completed(futs, timeout=timeout + 5):
            prov = futs[fut]
            try:
                results[prov] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[prov] = {"ok": False, "model": "—", "status": "timeout", "detail": str(e)[:120]}
    return results


def _handle_resume_command(args, cli) -> int:
    """Handle 'effgen resume' command."""
    from effgen import Agent, AgentConfig
    from effgen.core.checkpoint import CheckpointManager
    from effgen.errors import CorruptStateError

    cp_arg = args.checkpoint
    # Determine directory + id
    import os as _os
    if _os.path.isdir(cp_arg):
        ckpt_dir = cp_arg
        cp_id = None
    elif cp_arg.endswith(".json") and _os.path.exists(cp_arg):
        ckpt_dir = _os.path.dirname(_os.path.abspath(cp_arg)) or "."
        cp_id = cp_arg
    else:
        ckpt_dir = "./checkpoints"
        cp_id = cp_arg

    mgr = CheckpointManager(ckpt_dir)
    try:
        cp = mgr.load(cp_id) if cp_id else mgr.load_latest()
    except FileNotFoundError as e:
        cli.print(f"Error: {e}")
        cli.print("List available checkpoints by pointing --checkpoint at their directory.")
        cli.print(
            "Looking for a saved conversation instead? Those are listed by "
            "`effgen sessions list` and continued with `effgen chat --session-id <id>`."
        )
        return 2
    except CorruptStateError as e:
        cli.print(f"Error: {e}")
        return 2
    cli.print(f"Resuming '{cp.task[:80]}' from iteration {cp.iteration}")

    # Choose the model: an explicit --model wins; otherwise reuse the model the
    # checkpoint was created with so the run continues on the same model. Warn
    # loudly if the two disagree (a different model may not complete the task
    # coherently). Fall back to a small local model only if nothing is known.
    saved_model = getattr(cp, "model", "") or ""
    if args.model:
        chosen_model = args.model
        if saved_model and saved_model != args.model:
            cli.print(
                f"Warning: checkpoint was created with '{saved_model}' but resuming "
                f"with '{args.model}'. Results may differ."
            )
    elif saved_model:
        chosen_model = saved_model
        cli.print(f"Using checkpoint's model: {saved_model}")
    else:
        chosen_model = "Qwen/Qwen2.5-1.5B-Instruct"
        cli.print(
            "This checkpoint did not record a model; resuming on a small local "
            f"model ({chosen_model}). Pass -m/--model to choose another."
        )

    try:
        if getattr(args, 'preset', None):
            from effgen.presets import create_agent as _create_preset_agent
            agent = _create_preset_agent(args.preset, chosen_model)
        else:
            cfg = AgentConfig(name=cp.agent_name, model=chosen_model, tools=[])
            agent = Agent(cfg)
    except Exception as e:  # noqa: BLE001 - surface a clean error, no stack trace
        cli.print(f"Error: could not load model '{chosen_model}' to resume: {e}")
        return 1

    try:
        response = agent.resume(checkpoint_id=cp_id, checkpoint_dir=ckpt_dir)
        cli.print_data(response.output if hasattr(response, 'output') else str(response))
        return 0 if getattr(response, 'success', True) else 1
    finally:
        # Release the agent so resume never emits the "garbage-collected
        # without calling close()" warning (matches the run path).
        try:
            agent.close()
        except Exception as e:  # noqa: BLE001
            logging.debug(f"Agent close after resume failed: {e}")


def _handle_quickstart_command(args, cli: "CLIInterface") -> int:
    """Handle 'effgen quickstart' / 'effgen tutorial' — a guided first run.

    Walks a brand-new user from nothing to a successful agent run in well under
    two minutes: pick a model → run a sample task → see the trace → see the
    cost. Fully scriptable (``--model``/``--task``/``--yes``) for CI and docs.
    """
    interactive = _onboarding.is_interactive() and not getattr(args, 'yes', False)

    cli.print_header("effGen quickstart")
    cli.print("Let's run your first agent. This takes about a minute.\n")

    # --- 1. Choose a model -------------------------------------------------
    provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
    if prov_err:
        cli.print_error(prov_err)
        return 1

    model_id = getattr(args, 'model', None)
    if not model_id:
        model_id, suggested_provider, reason = _quickstart_suggest_model()
        provider = provider or suggested_provider
        cli.print(f"Suggested model: [bold]{model_id}[/bold]"
                  f"{f' (provider: {provider})' if provider else ''} — {reason}"
                  if cli.console else
                  f"Suggested model: {model_id}"
                  f"{f' (provider: {provider})' if provider else ''} — {reason}")
        if interactive:
            cli.print("Press Enter to accept, or type a model id (e.g. gpt-5-nano):")
            try:
                typed = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                typed = ""
                cli.print("")
            if typed:
                model_id = typed
                # A bare cloud id without a provider gets routed by the registry;
                # leave provider as-is so the loader can infer it.
        cli.print("")

    # --- 2. Run a sample task ---------------------------------------------
    _default_task = "What is 25 * 17?"
    task = getattr(args, 'task', None) or _default_task
    is_default_task = task == _default_task
    cli.print(f"Task: [italic]{task}[/italic]\n" if cli.console else f"Task: {task}\n")

    agent = None
    try:
        # Only equip the calculator for the math sample task. A user-supplied
        # task gets a clean direct-answer run so an unrelated question never
        # makes a small local model loop on a tool it doesn't need.
        tools = []
        if is_default_task:
            cli.tool_registry.discover_builtin_tools()
            if "calculator" in cli.tool_registry.list_tools():
                try:
                    tools.append(cli.tool_registry.get_tool_sync("calculator"))
                except Exception as e:  # noqa: BLE001
                    logging.debug(f"quickstart: calculator load failed: {e}")

        agent_config = AgentConfig(
            name="quickstart-agent",
            model=model_id,
            provider=provider,
            tools=tools,
            system_prompt="You are a helpful AI assistant.",
            max_iterations=5,
        )
        agent = Agent(agent_config)

        animate = cli._animate(args)
        # Keep the first impression clean: the agent transparently retries and
        # recovers from transient provider hiccups, so suppress that internal
        # log churn during the guided run (restored immediately after). A real
        # failure still surfaces via response.success below.
        _effgen_logger = logging.getLogger("effgen")
        _prev_level = _effgen_logger.level
        _effgen_logger.setLevel(logging.CRITICAL)
        try:
            if animate:
                with _progress.LiveStatus(
                    cli.console,
                    model_label=_progress.short_model_label(model_id),
                    reasoning=_progress.is_reasoning_agent(agent),
                    tracker=agent.execution_tracker,
                ):
                    response = agent.run(task)
            else:
                cli.print("Thinking...")
                response = agent.run(task)
        except KeyboardInterrupt:
            cli._handle_interrupt(agent)
            return 130
        finally:
            _effgen_logger.setLevel(_prev_level)

        # --- 3. Show the answer ------------------------------------------
        cli.print_header("Answer")
        if cli.console:
            cli.console.print(Panel(
                Markdown(response.output or "(no output)"),
                border_style="green" if response.success else "red",
            ))
        else:
            print(response.output)

        if not response.success:
            err = (response.metadata or {}).get("error", {})
            cli.print_error(_onboarding.teach(
                f"The run did not succeed: {err.get('message', 'unknown error')}",
                fix="Run 'effgen doctor --live --cheap' to confirm the model is reachable, "
                    "or try a different model with -m.",
                doc="docs/getting-started.md",
            ))
            return 1

        # --- 4. Show the trace -------------------------------------------
        # The agent's execution_trace is a list of event dicts (type/message/
        # data) — not a ReAct action/observation shape — so render it with the
        # shared formatter that understands those events. Hand-reading
        # ``step["action"]`` here used to miss every tool call and wrongly print
        # "answered directly" even when a tool ran (and tool_calls reported it).
        cli.print_header("What the agent did")
        _trace_lines = _progress.execution_trace_lines(response.execution_trace)
        if _trace_lines:
            for _style, _text in _trace_lines:
                if cli.console:
                    cli.console.print(f"  [{_style}]{_text}[/{_style}]")
                else:
                    cli.print(f"  {_text}")
        else:
            cli.print("  (answered directly, no tools needed)")

        # --- 5. Show the cost --------------------------------------------
        _progress.print_summary(cli, response)

        # --- Next steps --------------------------------------------------
        cli.print_header("You're set! Next steps")
        cli.print("  • effgen run \"your question\" -m " + model_id + "   # run any task")
        cli.print("  • effgen chat -m " + model_id + "                  # interactive chat")
        cli.print("  • effgen tools list                              # see available tools")
        cli.print("  • effgen doctor --live --cheap                   # check all providers")
        return 0

    except Exception as e:  # noqa: BLE001
        cli.print_error(_onboarding.teach(
            f"Quickstart could not complete: {e}",
            fix="Run 'effgen doctor' to check your setup, then 'effgen quickstart -m <model>'.",
            doc="docs/getting-started.md",
        ))
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception as e:  # noqa: BLE001
                logging.debug(f"quickstart: agent close failed: {e}")


def _handle_prompts_command(args, cli: "CLIInterface") -> int:
    """Handle 'effgen prompts' subcommands."""
    import json as _json

    try:
        from effgen.prompts.library import PromptEval, registry
    except ImportError as exc:
        cli.print_error(f"Prompt library not available: {exc}")
        return 1

    cmd = getattr(args, 'prompts_command', None)

    # ---- list ----
    if cmd == 'list' or cmd is None:
        domain_filter = getattr(args, 'domain', None)
        variant_filter = getattr(args, 'variant', None)
        fmt = getattr(args, 'list_format', 'table')

        prompts = registry.search(domain=domain_filter, variant=variant_filter)

        if fmt == 'json':
            rows = [
                {
                    'name': p.name,
                    'domain': p.domain,
                    'variant': p.variant,
                    'description': p.description,
                    'tags': p.tags,
                }
                for p in prompts
            ]
            print(_json.dumps(rows, indent=2, ensure_ascii=False))
            return 0

        if fmt == 'markdown':
            print("| Name | Domain | Variant | Description |")
            print("|------|--------|---------|-------------|")
            for p in prompts:
                print(f"| {p.name} | {p.domain} | {p.variant} | {p.description} |")
            if not prompts:
                print("| — | — | — | No prompts registered yet |")
            return 0

        # table (default)
        if RICH_AVAILABLE and cli.console:
            from rich.table import Table
            t = Table(title="Prompt Library", show_lines=False)
            # Names must never be clipped — they're the id a user types back
            # into `prompts show`/`run`/`render`. "fold" wraps onto extra
            # lines instead of the default ellipsis truncation.
            t.add_column("Name", style="cyan", overflow="fold")
            t.add_column("Domain")
            t.add_column("Variant")
            t.add_column("Description")
            for p in prompts:
                t.add_row(p.name, p.domain, p.variant, p.description)
            if not prompts:
                t.add_row("—", "—", "—", "No prompts registered yet")
            cli.console.print(t)
        else:
            if not prompts:
                print("No prompts registered yet.")
            else:
                print(f"{'Name':<45} {'Domain':<12} {'Variant':<12} Description")
                print("-" * 100)
                for p in prompts:
                    print(f"{p.name:<45} {p.domain:<12} {p.variant:<12} {p.description}")
        cli.print(f"\nTotal: {len(prompts)} prompt(s)")
        return 0

    # ---- show ----
    if cmd == 'show':
        name = args.name
        from effgen.cli.playground import _key_error_message, _resolve_prompt
        try:
            p, _resolved_name = _resolve_prompt(name)
        except KeyError as exc:
            cli.print_error(_key_error_message(exc, name))
            return 1

        cli.print_header(f"Prompt: {p.name}")
        cli.print(f"  Domain:      {p.domain}")
        cli.print(f"  Variant:     {p.variant}")
        cli.print(f"  Description: {p.description}")
        cli.print(f"  Tags:        {', '.join(p.tags) or '—'}")
        cli.print("\n[bold]Input Schema:[/bold]" if RICH_AVAILABLE else "\nInput Schema:")
        cli.print_data("  " + _json.dumps(p.input_schema, indent=2).replace("\n", "\n  "))
        cli.print("\n[bold]Fixture:[/bold]" if RICH_AVAILABLE else "\nFixture:")
        cli.print_data("  " + _json.dumps(p.fixture, indent=2).replace("\n", "\n  "))
        try:
            rendered = p.render_fixture()
            cli.print(
                "\n[bold]Rendered (fixture):[/bold]"
                if RICH_AVAILABLE
                else "\nRendered (fixture):"
            )
            cli.print_data(rendered)
        except Exception as exc:
            cli.print_warning(f"Could not render: {exc}")
        return 0

    # ---- eval ----
    if cmd == 'eval':
        domain_filter = getattr(args, 'domain', None)
        live = getattr(args, 'live', False)
        model = getattr(args, 'model', None)
        delay = getattr(args, 'delay', 35.0)
        output_path = getattr(args, 'output', None)
        fail_under = getattr(args, 'fail_under', None)

        prompts = registry.search(domain=domain_filter)
        evaluator = PromptEval()

        cli.print_header("Running golden eval...")
        golden_report = evaluator.eval_all_golden(prompts)
        table = golden_report.as_table()
        print(table)

        full_table = "=== Golden Eval ===\n" + table

        total = len(golden_report.results)
        passed = len(golden_report.passed())

        if live:
            if not model:
                cli.print_error("--model is required for --live eval")
                return 1
            cli.print_header(f"Running live eval with model '{model}'...")
            live_report = evaluator.eval_all_live(prompts, model, delay=delay)
            live_table = live_report.as_table()
            print(live_table)
            full_table += "\n=== Live Eval ===\n" + live_table
            total += len(live_report.results)
            passed += len(live_report.passed())

        if output_path:
            Path(output_path).write_text(full_table)
            cli.print_success(f"Eval table written to {output_path}")

        # Reflect failures in the exit code so the harness can gate CI. With
        # --fail-under, compare the pass rate to the threshold; otherwise any
        # single failure exits non-zero.
        failed = total - passed
        if fail_under is not None:
            rate = (passed / total) if total else 1.0
            if rate < fail_under:
                cli.print_error(
                    f"Pass rate {rate:.1%} ({passed}/{total}) is below the "
                    f"--fail-under threshold of {fail_under:.1%}."
                )
                return 1
            return 0
        if failed:
            cli.print_error(f"{failed} of {total} eval(s) failed.")
            return 1
        return 0

    # ---- playground ----
    if cmd == 'playground':
        from effgen.cli.playground import PlaygroundREPL
        repl = PlaygroundREPL()
        return repl.run()

    # ---- render (non-interactive) ----
    if cmd == 'render':
        from effgen.cli.playground import cmd_render
        name = getattr(args, 'prompt_name', None)
        input_file = getattr(args, 'input_file', None)
        inputs: dict = {}
        if input_file:
            try:
                inputs = _json.loads(Path(input_file).read_text())
            except Exception as exc:
                cli.print_error(f"Could not read input file: {exc}")
                return 1
        return cmd_render(name, inputs)

    # ---- run (non-interactive) ----
    if cmd == 'run':
        from effgen.cli.playground import cmd_run
        name = getattr(args, 'prompt_name', None)
        input_file = getattr(args, 'input_file', None)
        model = getattr(args, 'model', None)
        max_tokens = getattr(args, 'max_tokens', None)
        temperature = getattr(args, 'temperature', None)
        inputs = {}
        if input_file:
            try:
                inputs = _json.loads(Path(input_file).read_text())
            except Exception as exc:
                cli.print_error(f"Could not read input file: {exc}")
                return 1
        return cmd_run(name, inputs, model, max_tokens=max_tokens, temperature=temperature)

    cli.print("Usage: effgen prompts [list|show|eval|playground|render|run]")
    return 1




def _extract_theme_arg(argv: list[str]) -> tuple[list[str], str | None]:
    """Pull a ``--theme <name>`` (any position) out of *argv*.

    ``--theme`` selects the terminal color theme for every command's output, so
    it is accepted before or after the subcommand. Returns the remaining
    arguments and the requested theme name (or ``None``).
    """
    remaining: list[str] = []
    theme: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--theme":
            if i + 1 < len(argv):
                theme = argv[i + 1]
                i += 2
                continue
            i += 1
            continue
        if arg.startswith("--theme="):
            theme = arg.split("=", 1)[1]
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return remaining, theme


def main() -> None:
    """Main entry point for CLI."""
    # Load .env early so all subcommands see API keys (see load_env_files).
    load_env_files()

    # A --theme selects the color theme for every console built below; accept it
    # in any position (EFFGEN_THEME serves the same purpose without a flag).
    argv, theme_arg = _extract_theme_arg(sys.argv[1:])
    if theme_arg:
        os.environ["EFFGEN_THEME"] = theme_arg

    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle completion script generation
    if getattr(args, 'completion', None):
        from effgen.completion import get_completion
        print(get_completion(args.completion))
        sys.exit(0)

    # Setup logging. --verbose / --quiet may appear either before the
    # subcommand (global) or after it (per-command); honor whichever is set.
    _verbose = getattr(args, 'verbose', False)
    setup_logging(
        verbose=_verbose,
        log_file=getattr(args, 'log_file', None),
        quiet=getattr(args, 'quiet', False),
    )
    # For a command that renders its own failures, keep the console free of the
    # duplicate raw library ERROR line at default verbosity (--verbose / a
    # --log-file still carry the full diagnostic stream).
    if not _verbose and getattr(args, 'command', None) in _SELF_RENDERING_ERROR_COMMANDS:
        _suppress_echoed_error_logs()

    # Create CLI interface
    cli = CLIInterface()

    # One-time friendly welcome on first interactive use (records a flag so it
    # only ever shows once). Silent under --quiet / non-interactive / CI.
    _onboarding.maybe_show_first_run_welcome(quiet=getattr(args, 'quiet', False))

    # Route to appropriate handler
    try:
        if args.command == 'run':
            exit_code = cli.run_agent(args)
        elif args.command == 'chat':
            exit_code = cli.chat_mode(args)
        elif args.command == 'serve':
            exit_code = cli.serve_api(args)
        elif args.command == 'config':
            exit_code = cli.config_commands(args)
        elif args.command == 'tools':
            exit_code = cli.tools_commands(args)
        elif args.command == 'models':
            exit_code = cli.models_commands(args)
        elif args.command == 'examples':
            exit_code = cli.examples_commands(args)
        elif args.command == 'health':
            # Fail-closed on privacy: contacting effgen.org / PyPI is opt-in.
            remote = (
                getattr(args, 'health_remote', False)
                or os.environ.get("EFFGEN_HEALTH_REMOTE", "").strip().lower() in ("1", "true", "yes")
            )
            if not remote:
                print("effgen health performs network checks against external services "
                      "(effgen.org, docs.effgen.org, PyPI).")
                print("These are not run without consent. Re-run with --remote (or set "
                      "EFFGEN_HEALTH_REMOTE=1) to enable them.")
                exit_code = 0
            else:
                from effgen.utils.health import HealthChecker
                checker = HealthChecker()
                all_passed = checker.print_results()
                exit_code = 0 if all_passed else 1
        elif args.command == 'doctor':
            exit_code = _handle_doctor_command(args)
        elif args.command == 'resume':
            exit_code = _handle_resume_command(args, cli)
        elif args.command == 'sessions':
            exit_code = _handle_sessions_command(args, cli)
        elif args.command == 'runs':
            exit_code = _handle_runs_command(args, cli)
        elif args.command == 'create-plugin':
            exit_code = _create_plugin_scaffold(args.plugin_name, args.output_dir)
        elif args.command == 'presets':
            from effgen.presets import list_presets as _list_presets
            from effgen.presets.registry import preset_tool_overhead
            if getattr(args, 'output_json', False):
                rows = []
                for name, desc in _list_presets().items():
                    try:
                        n_tools, approx = preset_tool_overhead(name)
                    except Exception:  # noqa: BLE001 - listing never fails on one preset
                        n_tools, approx = 0, 0
                    rows.append({
                        "name": name,
                        "description": desc,
                        "tool_count": n_tools,
                        "approx_tokens_per_call": approx,
                    })
                print(json.dumps(rows, indent=2, ensure_ascii=False))
                exit_code = 0
            else:
                cli.print_header("Available Agent Presets")
                for name, desc in _list_presets().items():
                    try:
                        n_tools, approx = preset_tool_overhead(name)
                    except Exception:  # noqa: BLE001 - listing never fails on one preset
                        n_tools, approx = 0, 0
                    if n_tools:
                        overhead = f"{n_tools} tool{'s' if n_tools != 1 else ''} · ~{approx} tok/call"
                    else:
                        overhead = "no tools"
                    cli.print(f"  {name:12s}  {overhead}")
                    cli.print(f"               {desc}")
                cli.print(
                    "\n'~N tok/call' is the approximate tool-schema size sent on every "
                    "request — a tool-heavy preset costs more per call and can exceed "
                    "a small-context or rate-limited model."
                )
                cli.print("\nUsage: effgen run --preset <name> \"your task\"")
                exit_code = 0
        elif args.command in ('quickstart', 'tutorial'):
            exit_code = _handle_quickstart_command(args, cli)
        elif args.command == 'workflow':
            exit_code = _handle_workflow_command(args, cli)
        elif args.command == 'batch':
            exit_code = _handle_batch_command(args, cli)
        elif args.command == 'eval':
            exit_code = _handle_eval_command(args, cli)
        elif args.command == 'compare':
            exit_code = _handle_compare_command(args, cli)
        elif args.command == 'battle':
            from effgen.cli.battle import run_battle_command
            exit_code = run_battle_command(args)
        elif args.command in ('top', 'monitor'):
            from effgen.cli.monitor import run_monitor_command
            exit_code = run_monitor_command(args)
        elif args.command == 'cost':
            exit_code = _handle_cost_command(args, cli)
        elif args.command == 'report':
            exit_code = _handle_report_command(args, cli)
        elif args.command == 'prompts':
            exit_code = _handle_prompts_command(args, cli)
        elif args.command == 'loadtest':
            func = getattr(args, 'func', None)
            if func:
                exit_code = func(args)
            else:
                from effgen.cli.loadtest import run_loadtest_command
                exit_code = run_loadtest_command(args)
        elif args.command == 'debug':
            from effgen.debug.inspector import run_debug_cli
            # Validate an explicit --provider up front (a typo like "grok"
            # should fail fast with a suggestion), mirroring `run`/`chat`.
            provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
            if prov_err:
                cli.print_error(prov_err)
                exit_code = 1
            else:
                exit_code = run_debug_cli(
                    task=args.task,
                    preset=getattr(args, 'preset', None),
                    model=getattr(args, 'model', None),
                    provider=provider,
                    step=getattr(args, 'step', False),
                )
        elif args.command is None:
            # No command - launch interactive wizard
            # Create a namespace with default values for run command
            class WizardArgs:
                task = None
                model = None
                name = None
                tools = None
                config = None
                system_prompt = None
                temperature = None
                max_iterations = None
                mode = None
                no_sub_agents = False
                stream = False
                output = None
                verbose = getattr(args, 'verbose', False)
            exit_code = cli.interactive_wizard(WizardArgs())
        else:
            parser.print_help()
            exit_code = 0

        # A gentle, rotating tip at a natural moment — only after the commands a
        # human watches finish cleanly, never under --quiet / non-interactive /
        # EFFGEN_TIPS=0, and only every few runs (see onboarding.maybe_print_tip).
        _TIP_COMMANDS = {'run', 'chat', 'quickstart', 'tutorial', 'presets', 'doctor'}
        if (
            exit_code == 0
            and args.command in _TIP_COMMANDS
            and not getattr(args, 'output_json', False)
        ):
            _onboarding.maybe_print_tip(quiet=getattr(args, 'quiet', False))

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def agent_main() -> None:
    """
    Entry point for effgen-agent CLI (similar to smolagent).

    A generalist command to run a multi-step agent that can be equipped with various tools.

    Usage:
        # Run with direct prompt and options
        effgen-agent "Plan a trip to Tokyo" --model Qwen/Qwen2.5-1.5B-Instruct --tools web_search calculator

        # Run in interactive mode (launches setup wizard when no prompt provided)
        effgen-agent

    Interactive mode guides you through:
        - Reasoning style (auto / react / single — one Agent, different strategies)
        - Tool selection from available toolbox
        - Model configuration (type, ID, API settings)
        - Advanced options like additional imports
        - Task prompt input
    """
    import sys

    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Direct task mode - pass to run command
        task = sys.argv[1]
        remaining_args = sys.argv[2:]

        # Build new argv for main()
        new_argv = [sys.argv[0], 'run', task] + remaining_args
        sys.argv = new_argv
    elif len(sys.argv) == 1:
        # No arguments - launch interactive wizard
        sys.argv = [sys.argv[0], 'run']  # run without task triggers wizard
    # else: arguments starting with '-' will be handled by argparse

    main()


def web_agent_main() -> None:
    """
    Entry point for web agent CLI (effgen-web).

    A specialized agent for web browsing tasks.

    Usage:
        effgen-web "go to example.com and get the page title"
        effgen-web  # Interactive mode
    """
    import sys

    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Direct task mode
        task = sys.argv[1]
        sys.argv = [sys.argv[0], 'run', task, '--tools', 'web_search'] + sys.argv[2:]
    else:
        # Interactive mode - show help
        print(f"effGen Web Agent v{__version__}")
        print()
        print("Usage:")
        print("  effgen-web \"<task>\"           Run a web task")
        print("  effgen-web --model <model>    Specify model")
        print()
        print("Example:")
        print("  effgen-web \"Search for the latest Python release\"")
        print()
        return

    main()


if __name__ == "__main__":
    main()
