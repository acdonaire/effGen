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
import asyncio
import importlib.util
import json
import logging
import os
import sys
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
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
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
    from effgen.core.agent import AgentMode
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

# Shared Rich theme + console factory (one palette across the whole CLI).
from effgen.ui.tables import console_is_interactive, render_table
from effgen.ui.theme import CODE_THEME
from effgen.ui.theme import get_console as _get_console

# ---------------------------------------------------------------------------
# Server request models + types — defined at MODULE level (not inside serve_api).
#
# FastAPI resolves a route's parameter types from the handler's *module* globals
# via get_type_hints(); a function-local class/type is invisible there (acute
# under `from __future__ import annotations`, where every annotation is a
# string). A function-local body model made FastAPI treat the body as a query
# param (422 on a JSON POST); a function-local ``WebSocket`` annotation made it
# treat ``ws`` as a required query param and reject the handshake (1008). Keeping
# both ``TaskRequest`` and ``WebSocket`` at module scope makes `/run` and `/ws`
# work and lets `/openapi.json` build.
# ---------------------------------------------------------------------------
try:
    from fastapi import WebSocket  # noqa: F401 - module-level for annotation resolution
except Exception:  # pragma: no cover - fastapi present only with [server]
    WebSocket = None  # type: ignore[assignment,misc]

try:
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import ConfigDict as _PydanticConfigDict

    class TaskRequest(_PydanticBaseModel):
        model_config = _PydanticConfigDict(extra="ignore")

        task: str
        model: str | None = "Qwen/Qwen2.5-3B-Instruct"
        tools: list[str] | None = None
        preset: str | None = None
        temperature: float | None = 0.7
        max_iterations: int | None = 10
        stream: bool = False

    class TaskResponse(_PydanticBaseModel):
        model_config = _PydanticConfigDict(extra="ignore")

        output: str
        success: bool
        metadata: dict[str, Any]
except Exception:  # pragma: no cover - pydantic always present with [server]
    TaskRequest = None  # type: ignore[assignment,misc]
    TaskResponse = None  # type: ignore[assignment,misc]


def _general_purpose_tool_names(registry: Any, limit: int = 5) -> list:
    """Pick up to *limit* model-agnostic tool names for the convenience routes.

    The ``/run`` and ``/ws`` endpoints attach a small default tool set so a bare
    task can still use tools. Provider-*native* tools (tagged ``*-native``, e.g.
    Anthropic computer-use, OpenAI/Gemini built-ins) are executed server-side by
    a specific provider and raise "incompatible with model" on any other model,
    so they must be excluded from a generic, any-model default set.
    """
    names = []
    for name in registry.list_tools():
        try:
            tags = getattr(registry.get_metadata(name), "tags", None) or []
            if any("native" in str(t).lower() for t in tags):
                continue
        except Exception:  # noqa: BLE001 - if metadata is unavailable, keep the tool
            pass
        names.append(name)
        if len(names) >= limit:
            break
    return names


def filter_incompatible_tools(
    tools: list,
    model_id: str,
    *,
    warn: Any = None,
) -> tuple[list, list[tuple[str, str]]]:
    """Drop provider-native tools the chosen model cannot execute.

    Provider-*native* tools (Anthropic computer-use ``bash``/``text_editor``/
    ``computer``, OpenAI built-ins like ``web_search_preview``) are run
    server-side by one specific provider and raise "incompatible with model" on
    any other model. ``effgen run`` and ``effgen chat`` both attach a default
    tool set, so both must filter these out for the common case (a non-Claude,
    non-OpenAI model) instead of crashing at agent construction.

    Returns ``(kept_tools, skipped)`` where *skipped* is a list of
    ``(tool_name, reason)``. If *warn* is callable it is invoked once per
    skipped tool with a friendly one-line note.
    """
    model_id = model_id or ""
    is_anthropic_model = model_id.startswith("claude") or "anthropic" in model_id.lower()
    is_openai_model = (
        model_id.startswith("gpt-")
        or model_id.startswith("o1")
        or model_id.startswith("o3")
        or model_id.startswith("o4")
        or "openai" in model_id.lower()
    )
    kept: list = []
    skipped: list[tuple[str, str]] = []
    for tool in tools:
        tname = str(getattr(tool, "name", "") or "")
        cls_name = type(tool).__name__
        is_anthropic_native = "AnthropicNative" in cls_name or "anthropic" in tname.lower()
        is_openai_native = "OpenAINative" in cls_name
        if is_anthropic_native and not is_anthropic_model:
            skipped.append((tname, "requires a Claude model"))
            continue
        if is_openai_native and not is_openai_model:
            skipped.append((tname, "requires a gpt/o-series model"))
            continue
        kept.append(tool)
    if warn is not None:
        for name, why in skipped:
            warn(f"Skipping native tool '{name}' ({why})")
    return kept, skipped


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
):
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


# Providers effGen can route a bare model id to. Keep in sync with the model
# loader / ProviderRegistry; aliases map common spellings to the canonical name.
KNOWN_PROVIDERS = (
    "openai", "anthropic", "gemini", "cerebras", "groq",
    "together", "fireworks", "replicate", "hf",
)
PROVIDER_ALIASES = {
    "google": "gemini",
    "googleai": "gemini",
    "huggingface": "hf",
    "hf_inference": "hf",
    "claude": "anthropic",
    "gpt": "openai",
    "openai-compat": "openai",
}


def resolve_provider_name(provider: str | None) -> tuple[str | None, str | None]:
    """Validate/normalize a user-supplied provider name.

    Returns ``(canonical_provider, error_message)``. On success the error is
    ``None``; on a typo (e.g. ``grok``) the canonical name is ``None`` and the
    error carries a fuzzy "did you mean" suggestion so the CLI never silently
    falls through to a local model download.
    """
    if provider is None:
        return None, None
    raw = provider.strip()
    lower = raw.lower()
    if lower in KNOWN_PROVIDERS:
        return lower, None
    if lower in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[lower], None
    import difflib
    pool = list(KNOWN_PROVIDERS) + list(PROVIDER_ALIASES)
    close = difflib.get_close_matches(lower, pool, n=1, cutoff=0.5)
    hint = ""
    if close:
        suggestion = PROVIDER_ALIASES.get(close[0], close[0])
        hint = f" Did you mean '{suggestion}'?"
    return None, (
        f"Unknown provider '{raw}'.{hint} "
        f"Known providers: {', '.join(KNOWN_PROVIDERS)}."
    )


# Config-file keys that `run` reads and applies to the AgentConfig it builds
# (directly or via the CLI arg of the same name winning first). Keep this in
# sync with every `config.get(...)` call in `run_agent`.
_RUN_CONFIG_APPLIED_KEYS = frozenset({
    "system_prompt", "temperature", "max_iterations", "max_tokens", "guardrails",
})


def _warn_unapplied_config_keys(config: dict, cli: "CLIInterface") -> None:
    """Warn about a config-file key that names a real AgentConfig field but
    isn't one `run` currently applies from a config file.

    A `-c/--config` value the loader doesn't wire through should never be a
    silent no-op — that's a fail-open surprise for a security-relevant field
    (such as ``guardrails``) and a source of confusion for everything else.
    Anything not in :data:`_RUN_CONFIG_APPLIED_KEYS` gets a one-line heads-up
    naming the field and pointing at the matching CLI flag.
    """
    from dataclasses import fields as _dataclass_fields

    from effgen.core.agent import AgentConfig

    valid_fields = {f.name for f in _dataclass_fields(AgentConfig)}
    unapplied = sorted(
        k for k in config if k in valid_fields and k not in _RUN_CONFIG_APPLIED_KEYS
    )
    if not unapplied:
        return
    cli.print_warning(
        f"Configuration file sets {', '.join(unapplied)}, which `effgen run` "
        "does not read from a config file — pass the matching CLI flag "
        "instead, or build the agent through the Python API."
    )


class CLIInterface:
    """Main CLI interface for effGen."""

    def __init__(self):
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

    def print(self, *args, **kwargs):
        """Print with rich formatting if available."""
        console = self._human()
        if console:
            console.print(*args, **kwargs)
        else:
            print(*args, file=sys.stderr if self._human_to_stderr else None, **kwargs)

    def print_header(self, text: str):
        """Print a header."""
        console = self._human()
        if console:
            console.print(f"\n[effgen.heading]{text}[/effgen.heading]")
        else:
            print(f"\n=== {text} ===", file=sys.stderr if self._human_to_stderr else None)

    def print_success(self, text: str):
        """Print success message."""
        console = self._human()
        if console:
            console.print(f"[green]✓[/green] {text}")
        else:
            print(f"✓ {text}", file=sys.stderr if self._human_to_stderr else None)

    def print_error(self, text: str):
        """Print error message."""
        console = self._human()
        if console:
            console.print(f"[red]✗[/red] {text}")
        else:
            print(f"✗ {text}", file=sys.stderr if self._human_to_stderr else None)

    def print_warning(self, text: str):
        """Print warning message."""
        console = self._human()
        if console:
            console.print(f"[effgen.warning]⚠[/effgen.warning] {text}")
        else:
            print(f"⚠ {text}", file=sys.stderr if self._human_to_stderr else None)

    def print_error_panel(self, message: str, *, title: str = "Error"):
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

    def interactive_wizard(self, args):
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
        self.print_header(f"effGen v{__version__} - Interactive Setup Wizard")
        self.print()

        if self.console:
            self.console.print(Panel(
                "[bold cyan]Welcome to effGen Interactive Mode![/bold cyan]\n\n"
                "This wizard will guide you through setting up and running an agent.\n"
                "Press Ctrl+C at any time to exit.",
                title="Interactive Setup",
                border_style="cyan"
            ))
        else:
            print("=" * 60)
            print("Welcome to effGen Interactive Mode!")
            print("This wizard will guide you through setting up an agent.")
            print("Press Ctrl+C at any time to exit.")
            print("=" * 60)

        try:
            # Step 1: Reasoning style. effGen has ONE Agent class whose behavior
            # adapts to the task; this picks the tool-using strategy (not a
            # separate agent class).
            self.print_header("Step 1: Select Reasoning Style")
            agent_types = [
                ("1", "auto", "Let effGen choose: native tool-calling when the model "
                              "supports it, else ReAct (recommended)"),
                ("2", "react", "Explicit Reason → Act → Observe loop with tools"),
                ("3", "single", "One model call, no tool loop (plain Q&A)"),
            ]

            if self.console:
                table = Table(title="Reasoning Styles (one Agent, different strategies)")
                table.add_column("#", style="cyan", width=3)
                table.add_column("Style", style="magenta")
                table.add_column("Description", style="white")
                for num, name, desc in agent_types:
                    table.add_row(num, name, desc)
                self.console.print(table)
            else:
                for num, name, desc in agent_types:
                    print(f"  [{num}] {name}: {desc}")

            agent_type_input = input("\nSelect reasoning style [1]: ").strip() or "1"
            agent_type_map = {"1": "auto", "2": "react", "3": "single"}
            agent_type = agent_type_map.get(agent_type_input, "auto")
            self.print_success(f"Selected: {agent_type}")

            # Step 2: Tool Selection
            self.print_header("Step 2: Select Tools")

            # Discover and list available tools
            self.tool_registry.discover_builtin_tools()
            available_tools = self.tool_registry.list_tools()

            if self.console:
                table = Table(title=f"Available Tools ({len(available_tools)})")
                table.add_column("#", style="cyan", width=3)
                table.add_column("Name", style="magenta")
                table.add_column("Description", style="white")

                for i, tool_name in enumerate(available_tools, 1):
                    try:
                        metadata = self.tool_registry.get_metadata(tool_name)
                        desc = metadata.description[:40] + "..." if len(metadata.description) > 40 else metadata.description
                        table.add_row(str(i), tool_name, desc)
                    except Exception:
                        table.add_row(str(i), tool_name, "No description")
                self.console.print(table)
            else:
                for i, tool_name in enumerate(available_tools, 1):
                    print(f"  [{i}] {tool_name}")

            self.print("\nEnter tool numbers separated by commas (e.g., 1,2,3)")
            self.print("Or press Enter to use all tools, 'none' for no tools")
            tool_input = input("Tools [all]: ").strip().lower()

            selected_tools = []
            if tool_input == "none":
                pass
            elif tool_input == "" or tool_input == "all":
                for name in available_tools:
                    try:
                        tool = asyncio.run(self.tool_registry.get_tool(name))
                        selected_tools.append(tool)
                    except Exception:
                        pass
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in tool_input.split(",")]
                    for idx in indices:
                        if 0 <= idx < len(available_tools):
                            tool_name = available_tools[idx]
                            try:
                                tool = asyncio.run(self.tool_registry.get_tool(tool_name))
                                selected_tools.append(tool)
                            except Exception as e:
                                self.print_warning(f"Failed to load {tool_name}: {e}")
                except ValueError:
                    self.print_warning("Invalid input, using all tools")
                    for name in available_tools:
                        try:
                            tool = asyncio.run(self.tool_registry.get_tool(name))
                            selected_tools.append(tool)
                        except Exception:
                            pass

            self.print_success(f"Selected {len(selected_tools)} tool(s)")

            # Step 3: Model Configuration
            self.print_header("Step 3: Configure Model")

            model_types = [
                ("1", "TransformersModel", "Local Hugging Face model (e.g., Qwen/Qwen2.5-1.5B-Instruct)"),
                ("2", "OpenAIModel", "OpenAI API (requires OPENAI_API_KEY)"),
                ("3", "AnthropicModel", "Anthropic API (requires ANTHROPIC_API_KEY)"),
                ("4", "vLLMModel", "vLLM server (requires running vLLM instance)"),
                ("5", "LiteLLMModel", "LiteLLM proxy (supports multiple backends)")
            ]

            if self.console:
                table = Table(title="Model Types")
                table.add_column("#", style="cyan", width=3)
                table.add_column("Type", style="magenta")
                table.add_column("Description", style="white")
                for num, name, desc in model_types:
                    table.add_row(num, name, desc)
                self.console.print(table)
            else:
                for num, name, desc in model_types:
                    print(f"  [{num}] {name}: {desc}")

            model_type_input = input("\nSelect model type [1]: ").strip() or "1"

            # Get model ID based on type
            default_models = {
                "1": "Qwen/Qwen2.5-1.5B-Instruct",
                "2": "gpt-4o-mini",
                "3": "claude-3-haiku-20240307",
                "4": "Qwen/Qwen2.5-7B-Instruct",
                "5": "openai/gpt-4o-mini"
            }

            default_model = default_models.get(model_type_input, "Qwen/Qwen2.5-1.5B-Instruct")
            model_id = input(f"Model ID [{default_model}]: ").strip() or default_model
            self.print_success(f"Model: {model_id}")

            # Step 4: Advanced Options
            self.print_header("Step 4: Advanced Options")

            temp_input = input("Temperature [0.7]: ").strip()
            temperature = float(temp_input) if temp_input else 0.7

            max_iter_input = input("Max iterations [10]: ").strip()
            max_iterations = int(max_iter_input) if max_iter_input else 10

            sub_agents_input = input("Enable sub-agents? [Y/n]: ").strip().lower()
            enable_sub_agents = sub_agents_input != "n"

            stream_input = input("Stream output? [y/N]: ").strip().lower()
            enable_streaming = stream_input == "y"

            # Step 5: Task Input
            self.print_header("Step 5: Enter Task")

            if self.console:
                self.console.print("[italic]Enter your task or question for the agent.[/italic]")
                self.console.print("[dim]For multi-line input, end with an empty line.[/dim]\n")
            else:
                print("Enter your task or question for the agent.")
                print("For multi-line input, end with an empty line.\n")

            lines = []
            while True:
                try:
                    line = input("> " if not lines else "  ")
                    if line == "" and lines:
                        break
                    lines.append(line)
                except EOFError:
                    break

            task = "\n".join(lines).strip()

            if not task:
                self.print_error("No task provided")
                return 1

            # Confirm and Run
            self.print_header("Configuration Summary")

            summary = {
                "Reasoning Style": agent_type,
                "Model": model_id,
                "Tools": len(selected_tools),
                "Temperature": temperature,
                "Max Iterations": max_iterations,
                "Sub-agents": "enabled" if enable_sub_agents else "disabled",
                "Streaming": "enabled" if enable_streaming else "disabled",
                "Task": task[:50] + "..." if len(task) > 50 else task
            }

            if self.console:
                table = Table(title="Configuration")
                table.add_column("Setting", style="cyan")
                table.add_column("Value", style="magenta")
                for key, value in summary.items():
                    table.add_row(key, str(value))
                self.console.print(table)
            else:
                for key, value in summary.items():
                    print(f"  {key}: {value}")

            confirm = input("\nProceed with this configuration? [Y/n]: ").strip().lower()
            if confirm == "n":
                self.print_warning("Cancelled by user")
                return 0

            # Create agent and run task
            self.print_header("Running Agent")

            # Filter provider-specific native tools that are incompatible with
            # the selected model so the agent doesn't reject them at startup.
            selected_tools, _skipped = filter_incompatible_tools(
                selected_tools, model_id, warn=self.print_warning
            )

            agent_config = AgentConfig(
                name="interactive-agent",
                model=model_id,
                tools=selected_tools,
                temperature=temperature,
                max_iterations=max_iterations,
                enable_sub_agents=enable_sub_agents,
                enable_streaming=enable_streaming
            )

            agent = Agent(agent_config)

            if enable_streaming:
                if self.console:
                    self.console.print("\n[bold green]Agent:[/bold green] ", end="")
                else:
                    print("\nAgent: ", end="", flush=True)

                for token in agent.stream(task):
                    print(token, end='', flush=True)
                print()
            else:
                if self.console:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=self.console
                    ) as progress:
                        progress.add_task("Thinking...", total=None)
                        response = agent.run(task)
                else:
                    print("Thinking...")
                    response = agent.run(task)

                # Display response
                self.print_header("Response")

                if self.console:
                    self.console.print(Panel(
                        Markdown(response.output),
                        title="Agent Response",
                        border_style="green" if response.success else "red"
                    ))
                else:
                    print(response.output)

                # Display statistics
                self.print_header("Execution Statistics")
                stats = {
                    "Success": "Yes" if response.success else "No",
                    "Iterations": response.iterations,
                    "Tool Calls": response.tool_calls,
                    "Tokens Used": response.tokens_used,
                    "Execution Time": f"{response.execution_time:.2f}s"
                }

                if self.console:
                    stats_table = Table()
                    stats_table.add_column("Metric", style="cyan")
                    stats_table.add_column("Value", style="magenta")
                    for key, value in stats.items():
                        stats_table.add_row(key, str(value))
                    self.console.print(stats_table)
                else:
                    for key, value in stats.items():
                        print(f"  {key}: {value}")

            # Ask if user wants to continue
            continue_input = input("\nRun another task? [y/N]: ").strip().lower()
            if continue_input == "y":
                return self.interactive_wizard(args)

            return 0

        except KeyboardInterrupt:
            self.print("\n\nWizard cancelled")
            return 130
        except EOFError:
            # No interactive terminal (piped stdin, CI, Docker, IDE run button).
            # This is an expected condition, not a crash — give a clean,
            # actionable message instead of dumping a traceback.
            self.print_error(
                "No interactive terminal detected, so the setup wizard can't "
                "prompt for input. Pass your task directly, e.g.:\n"
                "    effgen run \"What is the capital of France?\"\n"
                "  (add --provider/--model to pick a backend; see 'effgen run --help')."
            )
            return 2
        except Exception as e:
            self.print_error(f"Error in interactive wizard: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def run_agent(self, args):
        """
        Run an agent with a task.

        Args:
            args: Parsed command-line arguments
        """
        input_files = getattr(args, 'input_files', None)

        # Check if we need to launch interactive wizard
        if args.task is None and not input_files:
            return self.interactive_wizard(args)
        if args.task is None:
            args.task = ""

        # Attach --file/--input content: an image becomes multimodal `inputs=`;
        # a document is read with the same loaders RAG ingestion uses and
        # prepended to the task as text context. With ``--preset rag`` the
        # documents are instead indexed into the agent's retrieval tool, so a
        # CLI user can point the rag preset at a knowledge base without dropping
        # into Python — the agent retrieves and cites instead of seeing the raw
        # text as context.
        extra_inputs: list[Any] = []
        rag_docs: list[str] = []
        _preset_is_rag = getattr(args, 'preset', None) == 'rag'
        if input_files:
            from effgen.core.multimodal import _media_kind_from_name, image_from
            from effgen.rag.ingest import DocumentIngester

            doc_sections = []
            for file_path in input_files:
                p = Path(file_path)
                if not p.exists():
                    self.print_error(f"--file: file not found: {file_path}")
                    return 1
                if _media_kind_from_name(str(p)) == "image":
                    extra_inputs.append(image_from(p))
                    continue
                if _preset_is_rag:
                    rag_docs.append(str(p))
                    continue
                ingester = DocumentIngester(
                    show_progress=False, chunk_size=10_000_000,
                    chunk_overlap=0, dedupe=False,
                )
                doc_chunks = ingester.ingest(p)
                if doc_chunks:
                    doc_text = "\n\n".join(c.content for c in doc_chunks)
                    doc_sections.append(f"--- {p.name} ---\n{doc_text}")
                    continue
                reason = (
                    ingester.last_skipped[0][1] if ingester.last_skipped
                    else "no extractable text"
                )
                # A type without a dedicated loader may still be plain text the
                # user wants read as-is (an uncommon source extension, a config
                # file). Read it as UTF-8 text when it decodes; only a genuinely
                # binary or unreadable file is refused.
                if "unsupported extension" in reason:
                    try:
                        text = p.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, OSError):
                        text = None
                    if text and text.strip():
                        self.print(f"Reading {p.name} as plain text.")
                        doc_sections.append(f"--- {p.name} ---\n{text}")
                        continue
                self.print_error(f"--file: could not read {file_path}: {reason}")
                return 1
            if doc_sections:
                args.task = "\n\n".join(doc_sections) + "\n\n---\n\n" + args.task
            if extra_inputs and getattr(args, 'stream', False):
                self.print_error(
                    "--file with an image is not supported together with "
                    "--stream; drop --stream or attach a document instead."
                )
                return 1

        # Headless JSON contract: keep stdout pure (only the JSON result object)
        # by routing all human chatter to stderr and never streaming. `-q --json`
        # gives a clean document to pipe straight into `jq`.
        json_mode = getattr(args, 'output_json', False)
        if json_mode:
            self._human_to_stderr = True
            args.stream = False

        # Validate an explicit --provider before doing any work, so a typo
        # (e.g. "grok") fails fast with a suggestion instead of falling through
        # to a multi-gigabyte local model download.
        provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
        if prov_err:
            self.print_error(prov_err)
            return 1

        self.print_header(f"effGen v{__version__} - Running Task")

        # Resolve the model once so the preset and plain paths agree. With no
        # -m/--model, mirror `quickstart`: prefer a detected cheap cloud model,
        # else a small local model — and say why, so the choice is never a silent
        # surprise (a paid cloud call or a multi-GB local download).
        if args.model:
            run_model = args.model
            _preflight_model_hint(self, run_model, provider)
        else:
            run_model, _sugg_provider, _sugg_reason = _quickstart_suggest_model()
            if provider is None and _sugg_provider:
                provider = _sugg_provider
            self.print(f"Using model {run_model} ({_sugg_reason}); override with -m/--model.")

        agent = None
        try:
            # Load configuration if provided
            config = {}
            if args.config:
                config_path = Path(args.config)
                if config_path.exists():
                    loaded_config = self.config_loader.load_config(config_path)
                    config = loaded_config.to_dict()
                    self.print_success(f"Loaded configuration from {config_path}")
                    _warn_unapplied_config_keys(config, self)
                else:
                    self.print_error(f"Configuration file not found: {config_path}")
                    return 1

            guardrails = getattr(args, 'guardrails', None) or config.get("guardrails")

            # Use preset if specified
            if getattr(args, 'preset', None):
                from effgen.presets import create_agent as _create_preset_agent
                model_id = run_model
                self.print(f"Using preset: {args.preset}")
                _preset_overrides = {"provider": provider} if provider else {}
                if rag_docs:
                    _preset_overrides["knowledge_base"] = rag_docs
                agent = _create_preset_agent(
                    args.preset,
                    model_id,
                    agent_name=args.name,
                    system_prompt=args.system_prompt or config.get("system_prompt"),
                    max_iterations=args.max_iterations,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    enable_streaming=args.stream,
                    session_id=getattr(args, 'session_id', None),
                    guardrails=guardrails,
                    **_preset_overrides,
                )
                self.print_success(f"Created {args.preset} preset agent")
                self.print(f"Model: {model_id}")
                tools = agent.config.tools if hasattr(agent, 'config') else []
            else:
                # Initialize tools
                tools = []
                if args.tools:
                    self.print(f"Loading tools: {', '.join(args.tools)}")
                    for tool_name in args.tools:
                        try:
                            tool = self.tool_registry.get_tool_sync(tool_name)
                            tools.append(tool)
                            self.print_success(f"Loaded tool: {tool_name}")
                        except KeyError:
                            self._suggest_tool(tool_name)
                            return 1
                else:
                    # Conservative default tool set: a single deterministic
                    # utility tool that rarely fires on general questions.
                    # (web_search/weather used to be defaults and triggered bogus
                    # calls like weather("Paris") for "capital of France?".) Users
                    # who want more tools pass --tools explicitly.
                    _default_safe_tools = ["calculator"]
                    self.tool_registry.discover_builtin_tools()
                    all_tool_names = self.tool_registry.list_tools()
                    for name in _default_safe_tools:
                        if name in all_tool_names:
                            try:
                                tools.append(self.tool_registry.get_tool_sync(name))
                            except Exception as e:
                                logging.debug(f"Failed to load default tool {name}: {e}")

                # Create agent configuration
                agent_config = AgentConfig(
                    name=args.name or "cli-agent",
                    model=run_model,
                    provider=provider,
                    tools=tools,
                    system_prompt=args.system_prompt or config.get("system_prompt",
                        "You are a helpful AI assistant."),
                    temperature=args.temperature or config.get("temperature", 0.7),
                    max_iterations=args.max_iterations or config.get("max_iterations", 10),
                    max_tokens=args.max_tokens or config.get("max_tokens"),
                    enable_sub_agents=not args.no_sub_agents,
                    enable_streaming=args.stream,
                    guardrails=guardrails,
                )

                # Create agent
                self.print(f"\nInitializing agent: {agent_config.name}")
                self.print(f"Model: {agent_config.model}")
                self.print(f"Tools: {len(tools)} available")
                self.print(f"Sub-agents: {'enabled' if agent_config.enable_sub_agents else 'disabled'}")
                if guardrails:
                    self.print(f"Guardrails: {guardrails}")

                agent = Agent(agent_config, session_id=getattr(args, 'session_id', None))

            # Determine execution mode. Default to the agent's own
            # config.mode (SINGLE unless set otherwise) instead of forcing
            # AUTO, so a plain `effgen run "<task>"` doesn't switch to
            # sub-agent decomposition on its own — pass --mode auto to
            # opt in explicitly for a call.
            mode = None
            if args.mode == "single":
                mode = AgentMode.SINGLE
            elif args.mode == "sub_agents":
                mode = AgentMode.SUB_AGENTS
            elif args.mode == "auto":
                mode = AgentMode.AUTO

            # Run task
            self.print(f"\n[bold]Task:[/bold] {args.task}" if self.console else f"\nTask: {args.task}")
            self.print()

            exit_code = 0
            # --json emits a single JSON document to stdout: no live spinner,
            # which would otherwise render there on an interactive terminal.
            animate = self._animate(args) and not json_mode
            quiet = getattr(args, 'quiet', False)
            model_label = _progress.short_model_label(
                getattr(agent.config, "model", None) if hasattr(agent, "config") else None
            )
            if args.stream:
                # Streaming output with an optional soft cursor for a live feel.
                if not quiet:
                    self.print(
                        "[italic]Streaming response...[/italic]\n"
                        if self.console else "Streaming response...\n"
                    )
                try:
                    self._stream_tokens(agent.stream(args.task, mode=mode), animate=animate)
                except KeyboardInterrupt:
                    self._handle_interrupt(agent)
                    return 130
                print()  # New line after streaming
            else:
                # Regular output with a live, accurate status line.
                run_kwargs = _checkpoint_run_kwargs(args)
                if extra_inputs:
                    run_kwargs['inputs'] = extra_inputs
                try:
                    if animate:
                        reasoning = _progress.is_reasoning_agent(agent)
                        with _progress.LiveStatus(
                            self.console,
                            model_label=model_label,
                            reasoning=reasoning,
                            tracker=agent.execution_tracker,
                        ):
                            response = agent.run(args.task, mode=mode, **run_kwargs)
                    else:
                        if not quiet:
                            self.print("Thinking...")
                        response = agent.run(args.task, mode=mode, **run_kwargs)
                except KeyboardInterrupt:
                    self._handle_interrupt(agent)
                    return 130

                # Surface failure in the process exit code.
                if not response.success:
                    exit_code = 1

                # Human presentation is skipped in --json mode so stdout stays a
                # single clean JSON document; the JSON is emitted below.
                if not json_mode:
                    # Display response
                    self.print_header("Response")

                    # A partial (iteration-cap) run still shows its recovered
                    # text, framed distinctly from a success or an outright
                    # failure. An outright failure reads the same as a
                    # model-load failure below — a red "Error" panel.
                    _partial = bool((response.metadata or {}).get("partial"))
                    if not response.success and not _partial:
                        self.print_error_panel(response.output, title="Error")
                    elif self.console:
                        _border = "green" if response.success else "yellow"
                        # Rich markdown formatting
                        self.console.print(Panel(
                            Markdown(response.output),
                            title="Agent Response",
                            border_style=_border
                        ))
                    else:
                        print(response.output)

                    # Frozen one-glance summary: ✓ Done in 3.2s · 2 tools · 1,204 tokens · $…
                    if not quiet:
                        _progress.print_summary(self, response)

                    _explain = getattr(args, 'explain', False)
                    _trace = getattr(args, 'trace', False)

                    # A per-step timeline (bars + durations) shows where the
                    # wall-clock went across the run's steps.
                    if _trace and response.execution_trace:
                        self.print_header("Timeline")
                        _tl = _progress.execution_timeline_lines(response.execution_trace)
                        if not _tl:
                            self.print("(no timed steps recorded for this run)")
                        for _style, _text in _tl:
                            if self.console:
                                self.console.print(f"[{_style}]{_text}[/{_style}]")
                            else:
                                print(_text)

                    # Display the step trace (tool reasoning + per-step timing).
                    if (_explain or _trace) and response.execution_trace:
                        self.print_header("Execution Trace")
                        _lines = _progress.execution_trace_lines(response.execution_trace)
                        if not _lines:
                            self.print("(no detailed steps recorded for this run)")
                        for _style, _text in _lines:
                            if self.console:
                                self.console.print(f"[{_style}]{_text}[/{_style}]")
                            else:
                                print(_text)

                    # On a multi-step run without an explicit trace flag, point
                    # the user at the timeline rather than leaving it hidden.
                    elif not quiet and not _explain and int(getattr(response, "tool_calls", 0) or 0) >= 1:
                        _steps = int(getattr(response, "tool_calls", 0) or 0)
                        _hint = f"{_steps} tool step{'s' if _steps != 1 else ''} — run with --trace to see the timeline"
                        if self.console:
                            self.console.print(f"[effgen.muted]{_hint}[/effgen.muted]")
                        else:
                            print(_hint)

                    # Display execution statistics
                    if getattr(args, 'verbose', False) or _explain or _trace:
                        self.print_header("Execution Statistics")
                        stats_table = self._create_stats_table({
                            "Mode": response.mode.value,
                            "Success": "Yes" if response.success else "No",
                            "Iterations": response.iterations,
                            "Tool Calls": response.tool_calls,
                            "Tokens Used": response.tokens_used,
                            "Execution Time": f"{response.execution_time:.2f}s"
                        })

                        if self.console:
                            self.console.print(stats_table)
                        else:
                            for key, value in stats_table.items():
                                print(f"{key}: {value}")

                        # Full verbose trace
                        if getattr(args, 'verbose', False) and response.execution_trace:
                            self.print_header("Full ReAct Trace")
                            trace_json = json.dumps(response.execution_trace, indent=2, default=str)
                            if self.console:
                                self.console.print(Syntax(trace_json, "json", line_numbers=True))
                            else:
                                print(trace_json)

                # Save response if output file specified
                if args.output:
                    output_path = Path(args.output)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(response.to_dict(), f, indent=2, ensure_ascii=False)
                    self.print_success(f"Response saved to {output_path}")

                # Emit the result object to stdout for piping (same document the
                # -o file carries). Goes to real stdout regardless of the
                # stderr-routed human output above.
                if json_mode:
                    print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))

            return exit_code

        except Exception as e:
            # Same presentation as a generation failure above — a red "Error"
            # panel — so a run that fails to load its model reads the same as
            # one that fails mid-generation.
            self.print_error_panel(str(e), title="Error")
            if getattr(args, 'verbose', False):
                import traceback
                traceback.print_exc()
            # A failure here happens before any AgentResponse exists (e.g. an
            # unknown model id) — --json still gets one clean JSON object on
            # stdout, matching the envelope a successful/runtime-failure run
            # would have emitted, instead of empty stdout with a nonzero exit.
            if json_mode:
                print(json.dumps({
                    "success": False,
                    "error": {"type": type(e).__name__, "message": str(e)},
                }, indent=2, ensure_ascii=False))
            return 1
        finally:
            # Release the agent explicitly so the CLI never emits the
            # "Agent was garbage-collected without calling close()" warning.
            if agent is not None:
                try:
                    agent.close()
                except Exception as e:
                    logging.debug(f"Agent close failed: {e}")

    def _stream_tokens(self, token_iter, *, animate: bool) -> str:
        """Print streamed tokens with an optional soft cursor; return the text.

        On an interactive terminal a single-cell soft cursor (``▌``) trails the
        latest token and is erased before the next one, giving a live-typing
        feel. When not animating (piped/redirected/non-TTY) tokens are written
        plainly so the output is clean to capture.
        """
        cursor = "▌"
        wrote_cursor = False
        collected = []
        for token in token_iter:
            if not token:
                continue
            collected.append(token)
            if animate:
                if wrote_cursor:
                    sys.stdout.write("\b \b")  # erase the previous single-cell cursor
                sys.stdout.write(token + cursor)
                wrote_cursor = True
            else:
                sys.stdout.write(token)
            sys.stdout.flush()
        if animate and wrote_cursor:
            sys.stdout.write("\b \b")  # erase the trailing cursor
            sys.stdout.flush()
        return "".join(collected)

    def _handle_interrupt(self, agent) -> None:
        """Render a friendly Ctrl-C stop (partial trace + 'Stopped.'), no traceback."""
        # Move to a clean line after any in-flight spinner/stream output.
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        # Surface whatever partial progress the tracker captured.
        try:
            tracker = getattr(agent, "execution_tracker", None)
            tools = sorted(getattr(tracker, "active_tools", set()) or set())
            done = [
                n.name for n in getattr(tracker, "nodes", {}).values()
                if getattr(n, "node_type", "") == "tool" and getattr(n, "status", "") == "completed"
            ]
            if done:
                self.print(f"Partial progress: {len(done)} tool call(s) completed "
                           f"({', '.join(done[:5])}).")
            if tools:
                self.print(f"Cancelled in-flight: {', '.join(tools)}.")
        except Exception:  # noqa: BLE001 - never let cleanup add noise
            pass
        if self.console:
            self.console.print("[yellow]Stopped.[/yellow]")
        else:
            print("Stopped.")

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

    def chat_mode(self, args):
        """Interactive chat REPL.

        Delegates to :class:`effgen.cli.chat.ChatREPL`, which provides streaming
        answers with a thinking spinner, a model/tool-aware prompt, slash
        commands (``/model``, ``/tools``, ``/cost``, ``/trace``, …), persistent
        ↑/↓ history, multiline input, and a per-turn Ctrl-C that cancels the
        current turn without exiting the session.
        """
        # Validate an explicit --provider up front (a typo like "grok" should
        # fail fast with a suggestion, exactly as ``run`` does).
        provider, prov_err = resolve_provider_name(getattr(args, "provider", None))
        if prov_err:
            self.print_error(prov_err)
            return 1
        try:
            args._provider = provider
        except Exception:  # noqa: BLE001 - argparse Namespace always allows this
            pass

        from effgen.cli.chat import ChatREPL

        try:
            return ChatREPL(self, args).run()
        except Exception as e:  # noqa: BLE001
            self.print_error(f"Error in chat mode: {e}")
            if getattr(args, "verbose", False):
                import traceback

                traceback.print_exc()
            return 1

    def serve_api(self, args):
        """
        Start the effGen API server.

        ``effgen serve`` and the ``effgen.server.app:create_app`` factory now
        converge on **one** secure application: the OpenAI-compatible ``/v1/*``
        endpoints (with SSE streaming), auth, RBAC/budget, audit, metrics, and
        the dashboard all come from :func:`effgen.server.app.create_app`. This
        method layers a few convenience routes (``/run``, ``/tools``, ``/``,
        ``/slo``, ``/ws``) on top.

        Auth posture (fail-closed by default):
          * ``EFFGEN_API_KEY`` set  → static API-key auth (Bearer or X-API-Key).
          * ``EFFGEN_DEV_MODE=1``   → auth disabled (loud warning; dev only).
          * OIDC env vars set       → JWT auth.
          * none of the above       → an **ephemeral** API key is generated and
            printed once, so the server is never unauthenticated by default.
        """
        self.print_header(f"effGen v{__version__} - API Server")

        try:
            import uvicorn
        except ImportError:
            self.print_error("FastAPI and uvicorn are required for server mode.")
            self.print("Install with: pip install 'effgen[server]'")
            return 1

        try:
            import secrets

            host = args.host
            port = args.port
            loopback = host in ("127.0.0.1", "localhost", "::1", "")
            verbose = getattr(args, "verbose", False)

            dev_mode = os.environ.get("EFFGEN_DEV_MODE", "0").strip() == "1"
            api_key = (os.environ.get("EFFGEN_API_KEY", "") or "").strip()
            oidc = bool(
                os.environ.get("EFFGEN_OIDC_ISSUER")
                or os.environ.get("EFFGEN_OIDC_JWKS_URI")
            )

            ephemeral_key = False
            if not api_key and not dev_mode and not oidc:
                # Secure by default: rather than serving unauthenticated, mint a
                # one-off key and print it. The operator copies it into their
                # client. Set EFFGEN_API_KEY (or EFFGEN_DEV_MODE=1) to override.
                api_key = secrets.token_urlsafe(24)
                os.environ["EFFGEN_API_KEY"] = api_key
                ephemeral_key = True

            if not loopback and not api_key and not oidc and dev_mode:
                self.print_error(
                    f"Binding to {host} exposes this server on all interfaces "
                    "with auth DISABLED (EFFGEN_DEV_MODE=1). Unset dev mode and "
                    "set EFFGEN_API_KEY, or bind to 127.0.0.1 (the default)."
                )

            cors_env = os.environ.get("EFFGEN_CORS_ORIGINS", "").strip()
            cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()] or None

            from effgen.server.app import create_app

            app = create_app(
                api_key=api_key or None,
                cors_origins=cors_origins,
                dev_mode=dev_mode,
                rate_limit_per_minute=getattr(args, "rate_limit", None),
                trust_proxy=getattr(args, "trust_proxy", None),
            )

            # Discover tools for the /run + /tools convenience routes and stash
            # the CLI instance on the app for those handlers to reach.
            self.tool_registry.discover_builtin_tools()
            app.state.cli = self

            self._register_convenience_routes(app)

            # --- Auth posture banner ---
            if dev_mode:
                self.print("Auth: DISABLED (EFFGEN_DEV_MODE=1) — do not use in production")
            elif ephemeral_key:
                self.print_success("Auth: ephemeral API key (set EFFGEN_API_KEY to pin one)")
                self.print(f"  API key: {api_key}")
                self.print(f'  Example: curl -H "Authorization: Bearer {api_key}" '
                           f"http://{host}:{port}/v1/models")
            elif api_key:
                self.print_success("Auth: static API key (EFFGEN_API_KEY)")
            elif oidc:
                self.print_success("Auth: OIDC / JWT")

            public_dashboard = dev_mode or os.environ.get(
                "EFFGEN_PUBLIC_DASHBOARD", "0"
            ).strip() == "1"
            public_playground = dev_mode or os.environ.get(
                "EFFGEN_PUBLIC_PLAYGROUND", "0"
            ).strip() == "1"

            self.print(f"Starting server on {host}:{port}")
            self.print(f"  OpenAI-compatible API : http://{host}:{port}/v1")
            self.print(f"  Interactive docs      : http://{host}:{port}/docs")
            dashboard_line = f"  Dashboard             : http://{host}:{port}/dashboard"
            if not public_dashboard:
                dashboard_line += "  (data requires an API key; set EFFGEN_PUBLIC_DASHBOARD=1 for local viewing)"
            self.print(dashboard_line)
            playground_line = f"  Playground            : http://{host}:{port}/playground"
            if not public_playground:
                playground_line += "  (paste an API key, or set EFFGEN_PUBLIC_PLAYGROUND=1 for local viewing)"
            self.print(playground_line)
            self.print()

            # Keep uvicorn's proxy-header handling consistent with the rate
            # limiter's trust decision. uvicorn rewrites scope["client"] from
            # X-Forwarded-For for any peer in ``forwarded_allow_ips`` (default
            # 127.0.0.1), which would let a loopback/same-host caller set the
            # rate-limit client IP even though trust_proxy defaults to off. When
            # the proxy is not trusted, disable that rewriting so the limiter
            # keys on the real socket peer; when it is trusted, effGen reads the
            # header itself and uvicorn's same-host default is left in place.
            from effgen.api.middleware import _resolve_trust_proxy
            uvicorn_kwargs: dict[str, Any] = {}
            if not _resolve_trust_proxy(getattr(args, "trust_proxy", None)):
                uvicorn_kwargs["forwarded_allow_ips"] = []

            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info" if verbose else "warning",
                **uvicorn_kwargs,
            )
            return 0

        except Exception as e:
            self.print_error(f"Error starting server: {e}")
            if getattr(args, "verbose", False):
                import traceback
                traceback.print_exc()
            return 1

    def _register_convenience_routes(self, app: Any) -> None:
        """Attach the legacy ``/run``, ``/tools``, ``/``, ``/slo``, ``/ws``
        routes onto the secure app from ``create_app``.

        Auth for these is provided by the app's ``AuthMiddleware`` (so they
        honor the same static-key / OIDC / dev posture as ``/v1``); they do not
        re-implement their own auth dependency.
        """
        # WebSocket is imported at module level (above) so FastAPI can resolve
        # the ``ws: WebSocket`` annotation under ``from __future__ import
        # annotations``; importing it only here would leave the annotation
        # unresolvable and break the /ws handshake.
        from fastapi import HTTPException, WebSocketDisconnect
        from fastapi.responses import JSONResponse

        from effgen.server.app import _normalize_model_id

        @app.post("/run")
        async def run_task(request: TaskRequest) -> Any:
            """Run a task with an agent (convenience endpoint)."""
            try:
                # Normalize OpenAI-style ``provider/model`` ids to effGen's
                # ``provider:model`` routing, matching the /v1 endpoints — so a
                # ``groq/…`` id loads the provider adapter instead of falling
                # through to the local Transformers path and 500-ing.
                model_id = _normalize_model_id(request.model) if request.model else request.model
                if request.preset:
                    from effgen.presets import create_agent as _create_preset_agent

                    agent_instance = _create_preset_agent(
                        request.preset,
                        model_id,
                        temperature=request.temperature,
                        max_iterations=request.max_iterations,
                    )
                else:
                    tools = []
                    for name in _general_purpose_tool_names(app.state.cli.tool_registry):
                        try:
                            tools.append(await app.state.cli.tool_registry.get_tool(name))
                        except Exception as tool_err:  # noqa: BLE001
                            logging.debug("Failed to load tool %s: %s", name, tool_err)
                    agent_instance = Agent(AgentConfig(
                        name="api-agent",
                        model=model_id,
                        tools=tools,
                        temperature=request.temperature,
                        max_iterations=request.max_iterations,
                    ))

                try:
                    response = agent_instance.run(request.task)
                finally:
                    agent_instance.close()  # release per-request agent resources
                return JSONResponse(content={
                    "output": response.output,
                    "success": response.success,
                    "metadata": {
                        "mode": response.mode.value if hasattr(response.mode, "value") else str(response.mode),
                        "iterations": response.iterations,
                        "tool_calls": response.tool_calls,
                        "execution_time": response.execution_time,
                    },
                })
            except Exception as e:  # noqa: BLE001
                logging.exception("Error running task")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/slo")
        async def slo_endpoint() -> Any:
            """SLO burn-rate status for all registered SLOs."""
            try:
                from effgen.observability.slo import get_tracker as _get_tracker

                return {"slos": _get_tracker().all_statuses()}
            except Exception as exc:  # noqa: BLE001
                return {"slos": [], "error": str(exc)}

        @app.get("/tools")
        async def list_tools_endpoint() -> Any:
            """List available tools."""
            tools = app.state.cli.tool_registry.list_tools()
            tool_info = []
            for tool_name in tools:
                try:
                    metadata = app.state.cli.tool_registry.get_metadata(tool_name)
                    tool_info.append({
                        "name": tool_name,
                        "description": metadata.description,
                        "category": metadata.category.value
                        if hasattr(metadata.category, "value") else str(metadata.category),
                    })
                except Exception:  # noqa: BLE001
                    tool_info.append({"name": tool_name, "description": "N/A", "category": "unknown"})
            return {"tools": tool_info, "count": len(tools)}

        @app.get("/")
        async def root() -> Any:
            """Root endpoint with API information."""
            return {
                "name": "effGen API",
                "version": __version__,
                "endpoints": {
                    "POST /v1/chat/completions": "OpenAI-compatible chat (SSE streaming)",
                    "POST /v1/completions": "OpenAI-compatible text completion",
                    "GET /v1/models": "List available model aliases",
                    "POST /run": "Run a task with an agent",
                    "WS /ws": "WebSocket streaming",
                    "GET /health": "Health check",
                    "GET /metrics": "Prometheus metrics (auth)",
                    "GET /tools": "List available tools",
                    "GET /docs": "OpenAPI documentation",
                    "GET /dashboard": "Local metrics dashboard",
                },
            }

        @app.websocket("/ws")
        async def websocket_stream(ws: WebSocket) -> None:
            """WebSocket endpoint for streaming agent responses."""
            await ws.accept()
            try:
                while True:
                    data = await ws.receive_json()
                    task = data.get("task", "")
                    model_id = _normalize_model_id(data.get("model", "Qwen/Qwen2.5-3B-Instruct"))
                    preset_name = data.get("preset")
                    if preset_name:
                        from effgen.presets import create_agent as _create_preset_agent

                        agent_instance = _create_preset_agent(preset_name, model_id)
                    else:
                        tools = []
                        for name in _general_purpose_tool_names(app.state.cli.tool_registry):
                            try:
                                tools.append(await app.state.cli.tool_registry.get_tool(name))
                            except Exception:  # noqa: BLE001
                                pass
                        agent_instance = Agent(AgentConfig(
                            name="ws-agent", model=model_id, tools=tools,
                            enable_streaming=True,
                        ))
                    await ws.send_json({"type": "start", "task": task})
                    try:
                        for token in agent_instance.stream(task):
                            await ws.send_json({"type": "token", "content": token})
                        await ws.send_json({"type": "done"})
                    except Exception as e:  # noqa: BLE001
                        await ws.send_json({"type": "error", "detail": str(e)})
                    finally:
                        agent_instance.close()  # release per-turn agent resources
            except WebSocketDisconnect:
                logging.info("WebSocket client disconnected")

    def config_commands(self, args):
        """
        Configuration management commands.

        Args:
            args: Parsed command-line arguments
        """
        if args.config_command == 'show':
            self._config_show(args)
        elif args.config_command == 'validate':
            self._config_validate(args)
        elif args.config_command == 'init':
            self._config_init(args)
        elif args.config_command == 'set':
            self._config_set(args)
        elif args.config_command is None:
            return _print_group_help(args)
        else:
            self.print_error(f"Unknown config command: {args.config_command}")
            return 1

        return 0

    def _config_set(self, args):
        """Handle 'effgen config set <key> <value>'."""
        key: str = args.key
        value_str: str = args.value

        _supported_keys = ("budget.daily", "budget.monthly")
        # Route budget.* keys to the cost tracker's budget config.
        if key.startswith("budget."):
            budget_key = key[len("budget."):]
            if budget_key not in {"daily", "monthly"}:
                hint = _onboarding.did_you_mean(key, _supported_keys, n=1, cutoff=0.5)
                self.print_error(
                    f"Unknown budget key: {key!r}. {hint + ' ' if hint else ''}"
                    f"Supported: {', '.join(_supported_keys)}."
                )
                return
            try:
                value = float(value_str)
            except ValueError:
                self.print_error(f"Budget value must be a number, got: {value_str!r}")
                return
            from effgen.models._cost import _budget_config_path
            budget_path = _budget_config_path()
            budget_path.parent.mkdir(parents=True, exist_ok=True)
            existing: dict = {}
            if budget_path.exists():
                try:
                    existing = json.loads(budget_path.read_text())
                except Exception:
                    pass
            existing[budget_key] = value
            budget_path.write_text(json.dumps(existing, indent=2))
            self.print_success(f"Set {key} = {value}")
        else:
            hint = _onboarding.did_you_mean(key, _supported_keys, n=1, cutoff=0.5)
            self.print_error(
                f"Unknown config key: {key!r}. {hint + ' ' if hint else ''}"
                f"Supported: {', '.join(_supported_keys)}."
            )

    def _config_show(self, args):
        """Show current configuration."""
        self.print_header("Configuration")

        if args.file:
            try:
                config = self.config_loader.load_config(args.file)

                if self.console:
                    syntax = Syntax(
                        json.dumps(config.to_dict(), indent=2),
                        "json",
                        theme=CODE_THEME,
                        line_numbers=True
                    )
                    self.console.print(syntax)
                else:
                    print(json.dumps(config.to_dict(), indent=2))

            except Exception as e:
                self.print_error(f"Error loading config: {e}")
        else:
            self.print_warning("No configuration file specified")
            self.print("Use: effgen config show --file <path>")

    def _config_validate(self, args):
        """Validate configuration file."""
        if not args.file:
            self.print_error("Configuration file required")
            return

        try:
            self.config_loader.load_config(args.file, validate=True)
            self.print_success(f"Configuration is valid: {args.file}")
        except Exception as e:
            self.print_error(f"Configuration validation failed: {e}")

    def _config_init(self, args):
        """Initialize a new configuration file."""
        output_path = Path(args.output or "config.yaml")

        if output_path.exists() and not args.force:
            self.print_error(f"File already exists: {output_path}")
            self.print("Use --force to overwrite")
            return

        # Create default configuration
        default_config = {
            "models": {
                "default": "Qwen/Qwen2.5-3B-Instruct",
                "phi3_mini": {
                    "model_path": "microsoft/Phi-3-mini-4k-instruct",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            },
            "tools": {
                "enabled": ["calculator", "web_search", "file_ops"]
            },
            "system_prompt": "You are a helpful AI assistant.",
            "max_iterations": 10
        }

        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        self.print_success(f"Configuration initialized: {output_path}")

    def tools_commands(self, args):
        """
        Tool management commands.

        Args:
            args: Parsed command-line arguments
        """
        if args.tool_command == 'list':
            return self._tools_list(args) or 0
        elif args.tool_command == 'info':
            return self._tools_info(args)
        elif args.tool_command == 'test':
            return self._tools_test(args)
        elif args.tool_command is None:
            return _print_group_help(args)
        else:
            self.print_error(f"Unknown tools command: {args.tool_command}")
            return 1

    def _suggest_tool(self, name: str) -> None:
        """Print a 'tool not found' error with close-match suggestions."""
        import difflib

        self.print_error(f"Tool not found: {name}")
        try:
            available = self.tool_registry.list_tools()
        except Exception:
            available = []
        close = difflib.get_close_matches(name, available, n=3, cutoff=0.5)
        if close:
            self.print(f"Did you mean: {', '.join(close)}?")
        self.print("Run 'effgen tools list' to see all available tools.")

    def _tools_list(self, args):
        """List available tools."""
        # Get tools (the registry auto-discovers built-ins on first access)
        tools = self.tool_registry.list_tools()
        category_filter = getattr(args, "category", None)

        def _meta(name):
            try:
                return self.tool_registry.get_metadata(name)
            except Exception as e:
                logging.debug(f"Error getting metadata for {name}: {e}")
                return None

        if category_filter:
            kept = []
            for name in tools:
                m = _meta(name)
                if m and m.category.value == category_filter:
                    kept.append(name)
            tools = kept

        # JSON output — machine-readable, no decorative header/table.
        if getattr(args, "output_json", False):
            out = []
            for name in tools:
                m = _meta(name)
                if m is None:
                    continue
                out.append({
                    "name": m.name,
                    "category": m.category.value,
                    "description": m.description,
                    "version": getattr(m, "version", None),
                })
            print(json.dumps(out, indent=2))
            return 0

        self.print_header("Available Tools")

        if not tools:
            self.print_warning("No tools registered")
            return 0

        if self.console:
            table = Table(title=f"Registered Tools ({len(tools)})")
            table.add_column("Name", style="cyan")
            table.add_column("Category", style="magenta")
            table.add_column("Description", style="white")

            for tool_name in tools:
                try:
                    metadata = self.tool_registry.get_metadata(tool_name)
                    table.add_row(
                        metadata.name,
                        metadata.category.value,
                        metadata.description[:50] + "..." if len(metadata.description) > 50 else metadata.description
                    )
                except Exception as e:
                    logging.debug(f"Error getting metadata for {tool_name}: {e}")

            self.console.print(table)
        else:
            for tool_name in tools:
                print(f"- {tool_name}")
        return 0

    def _example_input(self, metadata, tool=None) -> dict:
        """Build a runnable example input for a tool from its metadata."""
        # Prefer a curated example (drop the illustrative 'output' field).
        for ex in metadata.examples or []:
            if isinstance(ex, dict):
                example = {k: v for k, v in ex.items() if k != "output"}
                if example:
                    return example
        # Otherwise synthesize from required parameters.
        from effgen.tools.base_tool import ParameterType

        sample = {
            ParameterType.STRING: "example",
            ParameterType.INTEGER: 1,
            ParameterType.FLOAT: 1.0,
            ParameterType.BOOLEAN: True,
            ParameterType.ARRAY: [],
            ParameterType.OBJECT: {},
            ParameterType.ANY: "example",
        }
        example: dict = {}
        # Mirror the printed schema (to_json_schema), which excludes developer-only
        # toggles, so the copy-paste example never references a hidden parameter.
        for p in metadata.model_facing_parameters:
            if p.required or p.default is not None:
                if p.enum:
                    example[p.name] = p.enum[0]
                elif p.default is not None:
                    example[p.name] = p.default
                else:
                    example[p.name] = sample.get(p.type, "example")
        return example

    def _print_tool_usage(self, metadata, tool=None) -> None:
        """Print a tool's input schema and a copy-paste runnable example."""
        self.print("\n[bold]Input schema:[/bold]" if self.console else "\nInput schema:")
        schema = metadata.to_json_schema()
        if self.console:
            self.console.print(Syntax(json.dumps(schema, indent=2), "json", theme=CODE_THEME))
        else:
            print(json.dumps(schema, indent=2))

        example = self._example_input(metadata, tool)
        if example:
            cmd = f"effgen tools test {metadata.name} -i '{json.dumps(example)}'"
            self.print("\n[bold]Example:[/bold]" if self.console else "\nExample:")
            self.print(f"  {cmd}")

    def _tools_info(self, args):
        """Show detailed tool information."""
        if not args.name:
            self.print_error("Tool name required")
            return 1

        try:
            # get_metadata auto-discovers built-ins, so info works standalone.
            metadata = self.tool_registry.get_metadata(args.name)
        except KeyError:
            self._suggest_tool(args.name)
            return 1
        except Exception as e:
            self.print_error(f"Error getting tool info: {e}")
            return 1

        self.print_header(f"Tool: {metadata.name}")
        self.print(f"\n[bold]Description:[/bold] {metadata.description}" if self.console else f"\nDescription: {metadata.description}")
        self.print(f"[bold]Category:[/bold] {metadata.category.value}" if self.console else f"Category: {metadata.category.value}")
        self.print(f"[bold]Version:[/bold] {metadata.version}" if self.console else f"Version: {metadata.version}")

        if metadata.tags:
            self.print(f"[bold]Tags:[/bold] {', '.join(metadata.tags)}" if self.console else f"Tags: {', '.join(metadata.tags)}")

        # Selector aliases, if this tool accepts natural operation names.
        tool = None
        try:
            tool = self.tool_registry.get_tool_sync(args.name, initialize=False)
        except Exception:
            tool = None
        aliases = getattr(tool, "operation_aliases", {}) if tool else {}
        if aliases:
            alias_str = ", ".join(f"{a} -> {c}" for a, c in sorted(aliases.items()))
            self.print(f"[bold]Operation aliases:[/bold] {alias_str}" if self.console else f"Operation aliases: {alias_str}")

        # Show parameters
        if metadata.parameters:
            self.print("\n[bold]Parameters:[/bold]" if self.console else "\nParameters:")
            schema = metadata.to_json_schema()
            if self.console:
                self.console.print(Syntax(json.dumps(schema, indent=2), "json", theme=CODE_THEME))
            else:
                print(json.dumps(schema, indent=2))

        # Show a runnable example.
        example = self._example_input(metadata, tool)
        if example:
            cmd = f"effgen tools test {metadata.name} -i '{json.dumps(example)}'"
            self.print("\n[bold]Example:[/bold]" if self.console else "\nExample:")
            self.print(f"  {cmd}")

        return 0

    def _tools_test(self, args):
        """Test a tool with sample input."""
        if not args.name:
            self.print_error("Tool name required")
            return 1

        try:
            # Synchronous accessor: no asyncio.run boilerplate, and it auto-
            # discovers built-ins so 'test' works without a prior 'list'.
            tool = self.tool_registry.get_tool_sync(args.name)
        except KeyError:
            self._suggest_tool(args.name)
            return 1
        except Exception as e:
            self.print_error(f"Error loading tool: {e}")
            return 1

        metadata = tool.metadata

        # No input? Show the schema and a runnable example instead of guessing.
        if not args.input:
            self.print_header(f"Tool: {metadata.name}")
            self.print_warning("No input provided. Supply one with -i/--input as JSON.")
            self._print_tool_usage(metadata, tool)
            return 1

        # Parse input — must be a JSON object of parameters.
        try:
            input_data = json.loads(args.input)
        except json.JSONDecodeError as e:
            self.print_error(f"Input must be valid JSON: {e}")
            self._print_tool_usage(metadata, tool)
            return 1
        if not isinstance(input_data, dict):
            self.print_error("Input must be a JSON object of parameters, e.g. '{\"expression\": \"2+2\"}'.")
            self._print_tool_usage(metadata, tool)
            return 1

        self.print_header(f"Testing Tool: {metadata.name}")
        self.print(f"Input: {input_data}\n")
        try:
            result = asyncio.run(tool.execute(**input_data))
        except Exception as e:
            self.print_error(f"Error testing tool: {e}")
            if getattr(args, "verbose", False):
                import traceback
                traceback.print_exc()
            return 1

        self.print("[bold]Result:[/bold]" if self.console else "Result:")
        border = "green" if result.success else "red"
        if self.console:
            self.console.print(Panel(str(result), border_style=border))
        else:
            print(result)
        return 0 if result.success else 1

    def models_commands(self, args):
        """
        Model management commands.

        Args:
            args: Parsed command-line arguments
        """
        if args.model_command == 'list':
            return self._models_list(args) or 0
        elif args.model_command == 'browse':
            return self._models_browse(args) or 0
        elif args.model_command == 'info':
            return self._models_info(args) or 0
        elif args.model_command == 'load':
            self._models_load(args)
        elif args.model_command == 'unload':
            self._models_unload(args)
        elif args.model_command == 'status':
            self._models_status(args)
        elif args.model_command == 'refresh':
            return self._models_refresh(args) or 0
        elif args.model_command is None:
            return _print_group_help(args)
        else:
            self.print_error(f"Unknown models command: {args.model_command}")
            return 1

        return 0

    @staticmethod
    def _price_cell(rec) -> str:
        """Format a model's input/output price per 1M tokens for a table cell."""
        pin, pout = rec.price_in_per_1m, rec.price_out_per_1m
        # A genuinely nonzero published rate is shown as-is (mirrors
        # ``_catalog_pricing`` in ``effgen.models._cost``, the single source of
        # truth for how a $0 row is labeled).
        if (pin or 0) > 0 or (pout or 0) > 0:
            fmt = lambda v: ("?" if v is None else f"${v:g}")  # noqa: E731
            return f"{fmt(pin)}/{fmt(pout)}"
        # No nonzero rate: a genuine free tier reads "free"; a non-token billing
        # note reads "metered"; anything else (including an explicit 0/0 with no
        # free-tier flag) has no published price and reads "unpriced" rather than
        # a fabricated "$0".
        if rec.free_tier:
            return "free"
        if rec.price_note:
            return "metered"
        return "unpriced"

    # File extensions that count as actual model weights (an ".index.json" is a
    # shard manifest, not weights — a repo with only a manifest is still partial).
    _WEIGHT_SUFFIXES = (
        ".safetensors", ".bin", ".gguf", ".pt", ".pth",
        ".onnx", ".msgpack", ".h5", ".tflite", ".ot",
    )

    def _local_cached_models(self) -> list[dict]:
        """Models actually downloaded in the local HuggingFace cache (on disk).

        Each entry carries a ``complete`` flag: a snapshot with no real weight
        files (only an interrupted download, e.g. ``.incomplete`` blobs plus a
        shard manifest) is reported as incomplete so it isn't mistaken for ready.
        """
        out: list[dict] = []
        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir()
            for repo in sorted(info.repos, key=lambda r: r.repo_id):
                if repo.repo_type != "model":
                    continue
                weight_files = {
                    f.file_name
                    for rev in repo.revisions
                    for f in rev.files
                    if f.file_name.endswith(self._WEIGHT_SUFFIXES)
                    and not f.file_name.endswith(".index.json")
                }
                out.append({
                    "id": repo.repo_id,
                    "size_gb": repo.size_on_disk / (1024 ** 3),
                    "path": str(repo.repo_path),
                    "complete": bool(weight_files),
                })
        except Exception as e:  # noqa: BLE001 - cache scan is best-effort
            logging.debug(f"HF cache scan failed: {e}")
        return out

    def _local_model_context_window(self, path: str) -> int | None:
        """Read the model's max context length from its on-disk ``config.json``."""
        import glob
        for cfg in glob.glob(os.path.join(path, "snapshots", "*", "config.json")):
            try:
                with open(cfg, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
                val = data.get(key)
                if isinstance(val, int) and val > 0:
                    return val
        return None

    def _models_list(self, args):
        """List models from the drift-aware registry (not a static yaml).

        Shows three views — the provider registry (the bundled, refreshable
        catalog), the local HuggingFace cache (what's actually downloaded), and
        a per-provider summary with auth readiness and the snapshot's
        "verified on" date so users can tell when the data was last confirmed.
        """
        from effgen.models import _catalog, _refresh

        provider_filter, prov_err = resolve_provider_name(getattr(args, "provider", None))
        if prov_err:
            self.print_error(prov_err)
            return 1
        free_only = bool(getattr(args, "free", False))
        tools_only = bool(getattr(args, "tools", False))

        providers = [provider_filter] if provider_filter else list(_catalog.known_providers())

        def _records(prov: str):
            recs = _catalog.list_models(prov)
            if free_only:
                recs = [r for r in recs if r.free_tier]
            if tools_only:
                recs = [r for r in recs if r.supports_tools]
            return recs

        # ---- JSON output ----------------------------------------------------
        if getattr(args, "output_json", False):
            payload: dict[str, Any] = {"providers": {}, "local_cache": self._local_cached_models()}
            for prov in providers:
                meta = _catalog.snapshot_meta(prov)
                payload["providers"][prov] = {
                    "verified_on": meta.get("verified_on"),
                    "count": len(_catalog.list_models(prov)),
                    "default_model": _catalog.default_model(prov),
                    "auth_ready": _refresh.has_credentials(prov),
                    "models": [
                        {
                            "id": r.id, "family": r.family,
                            "context_window": r.context_window,
                            "max_output": r.max_output,
                            "price_in_per_1m": r.price_in_per_1m,
                            "price_out_per_1m": r.price_out_per_1m,
                            "supports_tools": r.supports_tools,
                            "supports_vision": r.supports_vision,
                            "supports_audio": r.supports_audio,
                            "free_tier": r.free_tier, "deprecated": r.deprecated,
                            "is_priced": r.is_priced,
                            "price_source": r.price_source,
                        }
                        for r in _records(prov)
                    ],
                }
            print(json.dumps(payload, indent=2))
            return 0

        # ---- Provider registry view ----------------------------------------
        self.print_header("Available Models")

        if provider_filter:
            # Full per-model detail for one provider.
            recs = _records(provider_filter)
            meta = _catalog.snapshot_meta(provider_filter)
            auth = "ready" if _refresh.has_credentials(provider_filter) else "no key"
            verified = meta.get("verified_on") or "unknown"
            default_id = _catalog.default_model(provider_filter)
            if self._rich_tables():
                title = (f"{provider_filter} — {len(recs)} models "
                         f"(auth: {auth}, verified: {verified})")
                table = Table(title=title)
                table.add_column("Model ID", style="cyan", overflow="fold")
                table.add_column("Context", style="white", justify="right", no_wrap=True)
                table.add_column("Max Out", style="white", justify="right", no_wrap=True)
                table.add_column("$/1M in/out", style="green", no_wrap=True, overflow="fold")
                table.add_column("Tools", justify="center", no_wrap=True)
                table.add_column("Vision", justify="center", no_wrap=True)
                table.add_column("Free", justify="center", no_wrap=True)
                table.add_column("Status", style="yellow")
                for r in recs:
                    status = "deprecated" if r.deprecated else ("default" if r.id == default_id else "")
                    table.add_row(
                        r.id,
                        f"{r.context_window:,}" if r.context_window else "—",
                        f"{r.max_output:,}" if r.max_output else "—",
                        self._price_cell(r),
                        "✓" if r.supports_tools else "",
                        "✓" if r.supports_vision else "",
                        "✓" if r.free_tier else "",
                        status,
                    )
                self.console.print(table)
                self.console.print(
                    f"\n[dim]Pricing source: catalog snapshot. "
                    f"Run [cyan]effgen models refresh --provider {provider_filter}[/cyan] "
                    f"to update from the live API.[/dim]"
                )
            else:
                id_w = min(max((len(r.id) for r in recs), default=8), 60)
                print(f"{provider_filter} — {len(recs)} models "
                      f"(auth: {auth}, verified: {verified})")
                for r in recs:
                    pin, pout = self._price_in_out_cells(r)
                    mark = " *" if r.id == default_id else ""
                    print(f"  {r.id:<{id_w}}  ctx={r.context_window or '-':>9}  "
                          f"in={pin:>9}  out={pout:>9}  "
                          f"{'tools' if r.supports_tools else '':<5} "
                          f"{'vision' if r.supports_vision else ''}{mark}")
            return 0

        # Overview: per-provider summary + filtered flat table when filtering.
        if free_only or tools_only:
            label = "free-tier" if free_only else "tool-capable"
            if tools_only and free_only:
                label = "free + tool-capable"
            if self._rich_tables():
                table = Table(title=f"{label.capitalize()} models (all providers)")
                table.add_column("Model ID", style="cyan", overflow="fold")
                table.add_column("Provider", style="magenta", no_wrap=True)
                table.add_column("Context", justify="right", no_wrap=True)
                table.add_column("$/1M in/out", style="green", no_wrap=True, overflow="fold")
                table.add_column("Tools", justify="center", no_wrap=True)
                table.add_column("Free", justify="center", no_wrap=True)
                for prov in providers:
                    for r in _records(prov):
                        table.add_row(
                            r.id, prov,
                            f"{r.context_window:,}" if r.context_window else "—",
                            self._price_cell(r),
                            "✓" if r.supports_tools else "",
                            "✓" if r.free_tier else "",
                        )
                self.console.print(table)
            else:
                flat = [r for prov in providers for r in _records(prov)]
                id_w = min(max((len(r.id) for r in flat), default=8), 60)
                for r in flat:
                    print(f"{r.id:<{id_w}}  {r.provider:<10}  "
                          f"ctx={r.context_window or '-':>9}  {self._price_cell(r)}")
            return 0

        # Default overview: one row per provider.
        stale = set(_catalog.stale_providers())
        if self.console:
            table = Table(title="Provider Registry (bundled catalog)")
            table.add_column("Provider", style="cyan")
            table.add_column("Models", justify="right")
            table.add_column("Default", style="magenta", overflow="fold")
            table.add_column("Auth", justify="center")
            table.add_column("Verified", style="dim")
            for prov in providers:
                meta = _catalog.snapshot_meta(prov)
                n = len(_catalog.list_models(prov))
                auth = "[green]key[/green]" if _refresh.has_credentials(prov) else "[dim]—[/dim]"
                verified = meta.get("verified_on") or "?"
                if prov in stale:
                    verified += " (stale)"
                table.add_row(prov, str(n), _catalog.default_model(prov) or "—", auth, verified)
            self.console.print(table)
            self.console.print(
                "\n[dim]Detail: [cyan]effgen models list --provider <name>[/cyan]  ·  "
                "Filter: [cyan]--free[/cyan] / [cyan]--tools[/cyan]  ·  "
                "Update: [cyan]effgen models refresh[/cyan][/dim]"
            )
        else:
            for prov in providers:
                n = len(_catalog.list_models(prov))
                auth = "key" if _refresh.has_credentials(prov) else "-"
                print(f"{prov:12s} {n:>4} models  default={_catalog.default_model(prov)}  auth={auth}")

        # ---- Local HuggingFace cache view ----------------------------------
        local = self._local_cached_models()
        if local:
            n_ready = sum(1 for m in local if m.get("complete", True))
            if self.console:
                ltable = Table(title=f"Local HuggingFace cache ({n_ready} ready)")
                ltable.add_column("Model", style="cyan", overflow="fold")
                ltable.add_column("Size", justify="right", style="white")
                ltable.add_column("Status", justify="center")
                for m in local:
                    ready = m.get("complete", True)
                    ltable.add_row(
                        m["id"], f"{m['size_gb']:.1f} GB",
                        "ready" if ready else "[yellow]incomplete[/yellow]",
                    )
                self.console.print(ltable)
            else:
                print("\nLocal HuggingFace cache:")
                for m in local:
                    tag = "" if m.get("complete", True) else "  (incomplete)"
                    print(f"  {m['id']}  ({m['size_gb']:.1f} GB){tag}")
        return 0

    def _rich_tables(self) -> bool:
        """True when rich table rendering fits the destination (a real terminal).

        Piped or redirected output narrows to a default width that truncates or
        drops columns; there the catalog views emit complete, aligned plain text
        instead so no model id or price is lost.
        """
        return bool(self.console) and bool(getattr(self.console, "is_terminal", False))

    @staticmethod
    def _browse_filter_sort(recs, args):
        """Apply the browse filters/sort to a list of catalog records.

        Filters compose (a record must satisfy every one supplied); records with
        no published input/output price are excluded by a ``--max-price-*``
        ceiling rather than treated as free. Returns the filtered, sorted list.
        """
        search = (getattr(args, "search", None) or "").lower().strip()
        min_ctx = getattr(args, "min_context", None)
        max_pin = getattr(args, "max_price_in", None)
        max_pout = getattr(args, "max_price_out", None)

        def keep(r) -> bool:
            if getattr(args, "free", False) and not r.free_tier:
                return False
            if getattr(args, "tools", False) and not r.supports_tools:
                return False
            if getattr(args, "vision", False) and not r.supports_vision:
                return False
            if getattr(args, "audio", False) and not r.supports_audio:
                return False
            if min_ctx is not None and (r.context_window or 0) < min_ctx:
                return False
            if max_pin is not None and (r.price_in_per_1m is None or r.price_in_per_1m > max_pin):
                return False
            if max_pout is not None and (r.price_out_per_1m is None or r.price_out_per_1m > max_pout):
                return False
            if search and search not in (
                f"{r.id} {r.family} {r.provider}".lower()
            ):
                return False
            return True

        out = [r for r in recs if keep(r)]

        sort = getattr(args, "sort", "provider") or "provider"
        # A missing numeric value sorts last on an ascending sort (unknown price
        # or context is worst-case), so it never masquerades as the cheapest.
        big = float("inf")

        def price_in(r):
            return r.price_in_per_1m if r.price_in_per_1m is not None else big

        def price_out(r):
            return r.price_out_per_1m if r.price_out_per_1m is not None else big

        keyers = {
            "provider": lambda r: (r.provider, r.id.lower()),
            "id": lambda r: r.id.lower(),
            "context": lambda r: (r.context_window or 0, r.id.lower()),
            "max-out": lambda r: (r.max_output or 0, r.id.lower()),
            "price-in": lambda r: (price_in(r), r.id.lower()),
            "price-out": lambda r: (price_out(r), r.id.lower()),
        }
        out.sort(key=keyers.get(sort, keyers["provider"]))
        if getattr(args, "desc", False):
            out.reverse()
        return out

    def _models_browse(self, args):
        """Browse the full cross-provider catalog with search/filter/sort/paging.

        Reads the bundled, refreshable catalog (the same source ``models list``
        and ``models info`` use). Every provider's models appear in one table so
        a single view answers "cheapest vision model over 128k context" without
        leaving the terminal. Price labeling is exact: an unpriced row reads
        ``unpriced``, a free tier reads ``free``.
        """
        from effgen.models import _catalog

        provider_filter, prov_err = resolve_provider_name(getattr(args, "provider", None))
        if prov_err:
            self.print_error(prov_err)
            return 1

        recs = _catalog.list_models(provider_filter)
        matched = self._browse_filter_sort(recs, args)
        total = len(matched)

        offset = max(0, getattr(args, "offset", 0) or 0)
        limit = getattr(args, "limit", None)
        page = matched[offset:offset + limit] if limit else matched[offset:]

        include_local = bool(getattr(args, "include_local", False))
        local = self._local_cached_models() if include_local else []

        # The snapshot "verified on" date is per provider, not stamped on the
        # bundled record; resolve it once per provider on the page so the JSON
        # provenance field carries the same date the table footer and the
        # dashboard show, rather than a null.
        verified_by_provider: dict[str, str | None] = {}

        def _verified_on(prov: str) -> str | None:
            if prov not in verified_by_provider:
                verified_by_provider[prov] = _catalog.snapshot_meta(prov).get("verified_on")
            return verified_by_provider[prov]

        # ---- JSON output ----------------------------------------------------
        if getattr(args, "output_json", False):
            payload: dict[str, Any] = {
                "count": total,
                "offset": offset,
                "limit": limit,
                "models": [
                    {
                        "id": r.id, "provider": r.provider, "family": r.family,
                        "context_window": r.context_window, "max_output": r.max_output,
                        "price_in_per_1m": r.price_in_per_1m,
                        "price_out_per_1m": r.price_out_per_1m,
                        "supports_tools": r.supports_tools,
                        "supports_vision": r.supports_vision,
                        "supports_audio": r.supports_audio,
                        "free_tier": r.free_tier, "deprecated": r.deprecated,
                        "is_priced": r.is_priced,
                        "price_source": r.price_source,
                        "verified_on": r.verified_on or _verified_on(r.provider),
                    }
                    for r in page
                ],
            }
            if include_local:
                payload["local_cache"] = local
            print(json.dumps(payload, indent=2))
            return 0

        # ---- Human table ----------------------------------------------------
        self.print_header("Model Catalog")
        if not matched:
            self.print("No models match those filters. Loosen a filter or run "
                       "[cyan]effgen models browse[/cyan] with no filters." if self.console
                       else "No models match those filters.")
            return 0

        # The cross-provider table carries nine columns; on a narrow terminal
        # rich would starve the (foldable) Model ID column — the one field this
        # view exists for — to keep the fixed numeric columns, folding or even
        # hiding the id. Below this width the complete aligned plain-text table
        # reads better and never drops an id or a price.
        wide_enough = getattr(self.console, "width", 0) >= 100
        if self._rich_tables() and wide_enough:
            shown = f"showing {len(page)} of {total}"
            if offset:
                shown += f" (from #{offset + 1})"
            table = Table(title=f"Models across providers — {shown}")
            table.add_column("Provider", style="magenta", no_wrap=True)
            table.add_column("Model ID", style="cyan", overflow="fold")
            table.add_column("Context", justify="right", no_wrap=True)
            table.add_column("Max Out", justify="right", no_wrap=True)
            table.add_column("$/1M in", style="green", justify="right",
                             no_wrap=True, overflow="fold")
            table.add_column("$/1M out", style="green", justify="right",
                             no_wrap=True, overflow="fold")
            table.add_column("Tools", justify="center", no_wrap=True)
            table.add_column("Vision", justify="center", no_wrap=True)
            table.add_column("Free", justify="center", no_wrap=True)
            for r in page:
                pin, pout = self._price_in_out_cells(r)
                table.add_row(
                    r.provider, r.id,
                    f"{r.context_window:,}" if r.context_window else "—",
                    f"{r.max_output:,}" if r.max_output else "—",
                    pin, pout,
                    "✓" if r.supports_tools else "",
                    "✓" if r.supports_vision else "",
                    "✓" if r.free_tier else "",
                )
            self.console.print(table)
            if limit and offset + limit < total:
                self.console.print(
                    f"\n[dim]More: [cyan]--offset {offset + limit}[/cyan] "
                    f"for the next page.[/dim]"
                )
            self.console.print(
                "\n[dim]Pricing source: catalog snapshot. "
                "Update: [cyan]effgen models refresh[/cyan]  ·  "
                "Detail: [cyan]effgen models info <id>[/cyan][/dim]"
            )
        else:
            # Complete, aligned plain text for piped/redirected output — every
            # model id and price in full, no width-driven truncation.
            id_w = max((len(r.id) for r in page), default=8)
            id_w = min(max(id_w, 8), 60)
            prov_w = max((len(r.provider) for r in page), default=8)
            header = (f"{'PROVIDER':<{prov_w}}  {'MODEL ID':<{id_w}}  "
                      f"{'CONTEXT':>9}  {'MAXOUT':>7}  {'$/1M IN':>9}  "
                      f"{'$/1M OUT':>9}  TOOLS  VIS  FREE")
            print(header)
            for r in page:
                pin, pout = self._price_in_out_cells(r)
                print(f"{r.provider:<{prov_w}}  {r.id:<{id_w}}  "
                      f"{(f'{r.context_window:,}' if r.context_window else '-'):>9}  "
                      f"{(f'{r.max_output:,}' if r.max_output else '-'):>7}  "
                      f"{pin:>9}  {pout:>9}  "
                      f"{'yes' if r.supports_tools else '-':>5}  "
                      f"{'yes' if r.supports_vision else '-':>3}  "
                      f"{'yes' if r.free_tier else '-':>4}")
            print(f"\nshowing {len(page)} of {total}"
                  + (f" (from #{offset + 1})" if offset else "")
                  + "  ·  pricing from catalog snapshot")

        if include_local and local:
            if self.console:
                self.console.print(f"\n[dim]Local cache: {len(local)} model(s) "
                                   f"— [cyan]effgen models list[/cyan] for detail.[/dim]")
            else:
                print("\nLocal cache:")
                for m in local:
                    print(f"  {m['id']}  ({m['size_gb']:.1f} GB)")
        return 0

    @classmethod
    def _price_in_out_cells(cls, rec) -> tuple[str, str]:
        """Return (input, output) price cells for the split-column browse table.

        A published nonzero rate shows as ``$<n>``; a genuine free tier reads
        ``free``, non-token billing reads ``metered``, and an unknown rate reads
        ``unpriced`` — never a fabricated ``$0`` (mirrors :meth:`_price_cell`).
        """
        pin, pout = rec.price_in_per_1m, rec.price_out_per_1m
        if (pin or 0) > 0 or (pout or 0) > 0:
            fmt = lambda v: ("?" if v is None else f"${v:g}")  # noqa: E731
            return fmt(pin), fmt(pout)
        label = "free" if rec.free_tier else ("metered" if rec.price_note else "unpriced")
        return label, label

    def _local_model_payload(self, entry: dict) -> dict:
        """Build the local-cache facts for one model: engines, size, ctx, status."""
        import importlib.util
        engines = ["transformers"]
        if importlib.util.find_spec("vllm") is not None:
            engines.append("vllm")
        return {
            "id": entry["id"],
            "cached": True,
            "complete": entry.get("complete", True),
            "size_gb": entry["size_gb"],
            "path": entry.get("path"),
            "context_window": self._local_model_context_window(entry.get("path", "")),
            "engines": engines,
        }

    def _render_local_model_info(self, payload: dict) -> None:
        """Render the 'this model is in your local cache' block for `models info`."""
        ctx = payload.get("context_window")
        status = "ready" if payload.get("complete", True) else "incomplete download"
        rows = {
            "Local copy": "yes (HuggingFace cache)",
            "Status": status,
            "On-disk size": f"{payload['size_gb']:.1f} GB",
            "Local engines": ", ".join(payload["engines"]),
            "Context window": f"{ctx:,}" if ctx else "—",
        }
        run_hint = (f"effgen run -m {payload['id']} --engine transformers \"...\"")
        if self.console:
            from rich.table import Table
            table = Table(show_header=False, title="Local cache")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white", overflow="fold")
            for k, v in rows.items():
                table.add_row(k, str(v))
            self.console.print(table)
            self.console.print(f"\n[dim]Run locally: [cyan]{run_hint}[/cyan]"
                               f"  ·  or [cyan]load_model(\"{payload['id']}\", "
                               f"engine=\"transformers\")[/cyan][/dim]")
        else:
            print("\nLocal cache:")
            for k, v in rows.items():
                print(f"  {k}: {v}")
            print(f"  Run locally: {run_hint}")

    def _models_info(self, args):
        """Show detailed information for one model from the registry."""
        if not args.name:
            self.print_error("Model name required")
            return 1

        from effgen.models import _catalog, _refresh
        from effgen.models.model_loader import ModelLoader

        # An engine-prefixed id (e.g. "transformers:Qwen/Qwen2.5-1.5B-Instruct")
        # names a local engine + a bare repo id. Strip the prefix so the id
        # matches the local cache and the catalog, and remember the engine so we
        # can lead with the local view.
        lookup_name = args.name
        requested_engine = None
        if ":" in lookup_name:
            _prefix, _rest = lookup_name.split(":", 1)
            if _prefix in ModelLoader._LOCAL_ENGINE_PREFIXES and _rest:
                requested_engine = _prefix
                lookup_name = _rest

        # Is this id sitting in the local HF cache? If so we can describe it as
        # locally-runnable even when the cloud catalog has no (or a different) row.
        local_entry = next(
            (m for m in self._local_cached_models() if m["id"] == lookup_name), None
        )

        # An explicit local-engine request is answered from the local cache: show
        # the cached copy, or report a cache miss naming what is cached (rather
        # than cloud catalog suggestions the engine can't run).
        if requested_engine is not None:
            if local_entry is not None:
                local_payload = self._local_model_payload(local_entry)
                if getattr(args, "output_json", False):
                    print(json.dumps({"id": lookup_name, "provider": None,
                                       "engine": requested_engine,
                                       "local": local_payload}, indent=2))
                    return 0
                self.print_header(f"Model: {lookup_name} ({requested_engine})")
                self._render_local_model_info(local_payload)
                return 0
            cached = [m["id"] for m in self._local_cached_models()]
            if getattr(args, "output_json", False):
                print(json.dumps({"id": lookup_name, "provider": None,
                                   "engine": requested_engine, "local": None,
                                   "cached_models": cached}, indent=2))
                return 1
            self.print_error(
                f"Model '{lookup_name}' is not in the local cache, so the "
                f"'{requested_engine}' engine can't run it yet."
            )
            if cached:
                self.print("Locally cached models:")
                for cid in cached:
                    self.print(f"  {cid}")
                self.print(f"\nDownload it first: effgen run -m {lookup_name} "
                           f"--engine {requested_engine} \"...\" (with network access).")
            return 1

        rec = _catalog.lookup(lookup_name)
        if rec is None:
            if local_entry is not None:
                # Downloaded locally but not in the cloud catalog: describe the local
                # copy instead of a misleading "not found in catalog".
                local_payload = self._local_model_payload(local_entry)
                if getattr(args, "output_json", False):
                    print(json.dumps({"id": args.name, "provider": None,
                                       "local": local_payload}, indent=2))
                    return 0
                self.print_header(f"Model: {args.name} (local)")
                self._render_local_model_info(local_payload)
                return 0
            # Helpful not-found: suggest the nearest catalog ids + provider:id form.
            self.print_error(f"Model not found in catalog: {args.name}")
            alts = _catalog.nearest_alternatives(args.name, n=5)
            if alts:
                self.print("Did you mean:")
                for a in alts:
                    self.print(f"  {a.provider}:{a.id}")
            self.print("\nRun 'effgen models list' to browse the catalog, or "
                       "'effgen models refresh' to update it.")
            return 1

        # The same model id can be served by more than one provider at different
        # prices; surface every alternative so the choice of *where* to run it is
        # visible (the resolved provider stays first).
        others = [v for v in _catalog.variants(rec.id) if v.provider != rec.provider]

        if getattr(args, "output_json", False):
            print(json.dumps({
                "id": rec.id, "provider": rec.provider, "display_name": rec.display_name,
                "family": rec.family, "context_window": rec.context_window,
                "max_output": rec.max_output, "price_in_per_1m": rec.price_in_per_1m,
                "price_out_per_1m": rec.price_out_per_1m, "supports_tools": rec.supports_tools,
                "supports_vision": rec.supports_vision, "supports_audio": rec.supports_audio,
                "free_tier": rec.free_tier, "deprecated": rec.deprecated,
                "rpm": rec.rpm, "tpm": rec.tpm, "rpd": rec.rpd,
                "price_source": rec.price_source, "verified_on": rec.verified_on,
                "notes": rec.notes,
                "also_available": [
                    {
                        "provider": v.provider,
                        "price_in_per_1m": v.price_in_per_1m,
                        "price_out_per_1m": v.price_out_per_1m,
                        "context_window": v.context_window,
                        "supports_tools": v.supports_tools,
                        "supports_vision": v.supports_vision,
                        "free_tier": v.free_tier,
                    }
                    for v in others
                ],
                "local": self._local_model_payload(local_entry) if local_entry else None,
            }, indent=2))
            return 0

        self.print_header(f"Model: {rec.provider}:{rec.id}")
        meta = _catalog.snapshot_meta(rec.provider)
        rows = {
            "Provider": rec.provider,
            "Display name": rec.display_name or rec.id,
            "Family": rec.family or "—",
            "Context window": f"{rec.context_window:,}" if rec.context_window else "—",
            "Max output": f"{rec.max_output:,}" if rec.max_output else "—",
            "Price ($/1M in / out)": self._price_cell(rec),
            "Tool calling": "yes" if rec.supports_tools else "no",
            "Vision": "yes" if rec.supports_vision else "no",
            "Audio": "yes" if rec.supports_audio else "no",
            "Free tier": "yes" if rec.free_tier else "no",
            "Rate limits (rpm/tpm/rpd)": f"{rec.rpm or '—'} / {rec.tpm or '—'} / {rec.rpd or '—'}",
            "Deprecated": "yes" if rec.deprecated else "no",
            "Price source": rec.price_source,
            "Verified on": rec.verified_on or meta.get("verified_on") or "unknown",
            "Auth ready": "yes" if _refresh.has_credentials(rec.provider) else "no (set key)",
        }
        if self.console:
            table = Table(show_header=False)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white", overflow="fold")
            for k, v in rows.items():
                table.add_row(k, str(v))
            self.console.print(table)
        else:
            for k, v in rows.items():
                print(f"  {k}: {v}")

        # When several providers serve this id, compare them so the analyst can
        # pick where to run it (pin the choice with a ``provider:id`` form).
        if others:
            if self.console:
                vtable = Table(title=f"Also served by ({len(others)} other provider(s))")
                vtable.add_column("Provider", style="magenta", no_wrap=True)
                vtable.add_column("$/1M in/out", style="green", no_wrap=True, overflow="fold")
                vtable.add_column("Context", justify="right", no_wrap=True)
                vtable.add_column("Tools", justify="center", no_wrap=True)
                vtable.add_column("Vision", justify="center", no_wrap=True)
                vtable.add_column("Free", justify="center", no_wrap=True)
                for v in others:
                    vtable.add_row(
                        v.provider, self._price_cell(v),
                        f"{v.context_window:,}" if v.context_window else "—",
                        "✓" if v.supports_tools else "",
                        "✓" if v.supports_vision else "",
                        "✓" if v.free_tier else "",
                    )
                self.console.print(vtable)
                self.console.print(
                    f"\n[dim]Pin a provider: "
                    f"[cyan]effgen run --provider <name> -m {rec.id}[/cyan][/dim]"
                )
            else:
                names = ", ".join(v.provider for v in others)
                print(f"\n  Also served by: {names} "
                      f"(pin with 'provider:{rec.id}')")

        cloud_hint = f"effgen run --provider {rec.provider} -m {rec.id} \"...\""
        # If the same id is also downloaded locally, lead with the local engine
        # path (the local block prints its own "Run locally" hint) and show the
        # cloud invocation as an alternative — don't present it as cloud-only.
        if local_entry is not None:
            self._render_local_model_info(self._local_model_payload(local_entry))
            if self.console:
                self.console.print(f"\n[dim]Cloud alternative: [cyan]{cloud_hint}[/cyan][/dim]")
            else:
                print(f"\n  Cloud alternative: {cloud_hint}")
        elif self.console:
            self.console.print(f"\n[dim]Use: [cyan]{cloud_hint}[/cyan][/dim]")
        return 0

    def _models_load(self, args):
        """Pre-load a model into the model pool."""
        from effgen.models.pool import ModelPool

        model_name = args.name
        engine = getattr(args, 'engine', None)
        self.print(f"Loading model: {model_name}...")

        try:
            pool = ModelPool()
            pool.get_or_load(model_name, engine=engine)
            self.print_success(f"Model '{model_name}' loaded successfully")

            # Show status
            for entry in pool.status():
                if entry["model_name"] == model_name:
                    self.print(f"  GPU memory: ~{entry['gpu_memory_gb']:.1f} GB")
        except Exception as e:
            self.print_error(f"Failed to load model: {e}")
            return 1

    def _models_unload(self, args):
        """Unload a model from memory."""
        from effgen.models.model_loader import ModelLoader

        model_name = args.name
        self.print(f"Unloading model: {model_name}...")

        try:
            loader = ModelLoader()
            if model_name in loader.loaded_models:
                loader.unload_model(model_name)
                self.print_success(f"Model '{model_name}' unloaded")
            else:
                self.print_warning(f"Model '{model_name}' is not currently loaded")
        except Exception as e:
            self.print_error(f"Failed to unload model: {e}")
            return 1

    def _models_status(self, args):
        """Show loaded models and GPU memory status."""
        if getattr(args, "output_json", False):
            return self._models_status_json()

        self.print_header("Model & GPU Status")

        # GPU memory info — physical (driver) view across all processes, so this
        # reflects which GPUs are actually free, not just this process's usage.
        try:
            from effgen.gpu.cuda_compat import per_gpu_status
            gpus = per_gpu_status()
            gib = 1024 ** 3
            if gpus:
                if self.console:
                    from rich.table import Table
                    gpu_table = Table(title="GPU Status (physical, all processes)")
                    gpu_table.add_column("GPU", style="cyan")
                    gpu_table.add_column("Name", style="white")
                    gpu_table.add_column("Total", style="white")
                    gpu_table.add_column("Used", style="yellow")
                    gpu_table.add_column("Free", style="green")
                    gpu_table.add_column("Util", justify="right")

                    for g in gpus:
                        util = f"{g.utilization_pct:.0f}%" if g.utilization_pct is not None else "—"
                        gpu_table.add_row(
                            str(g.index), g.name,
                            f"{g.total_bytes / gib:.1f} GB",
                            f"{g.used_bytes / gib:.1f} GB",
                            f"{g.free_bytes / gib:.1f} GB",
                            util,
                        )
                    self.console.print(gpu_table)
                else:
                    for g in gpus:
                        util = f", {g.utilization_pct:.0f}% util" if g.utilization_pct is not None else ""
                        print(f"GPU {g.index}: {g.name} — "
                              f"{g.total_bytes / gib:.1f} GB total, "
                              f"{g.used_bytes / gib:.1f} GB used, "
                              f"{g.free_bytes / gib:.1f} GB free{util}")
            else:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        self.print_warning("CUDA not available")
                    else:
                        self.print_warning("Could not query GPU memory status")
                except ImportError:
                    self.print_warning("PyTorch not installed — cannot query GPU status")
        except ImportError:
            self.print_warning("PyTorch not installed — cannot query GPU status")

        # Loaded models
        from effgen.models.model_loader import ModelLoader
        loader = ModelLoader()
        loaded = loader.get_loaded_models()

        if loaded:
            self.print("")
            self.print_header("Loaded Models")
            for name, model in loaded.items():
                status = "loaded" if model.is_loaded() else "unloaded"
                self.print(f"  {name}: {status}")
        else:
            self.print("\nNo models currently loaded in this process.")

        # Capability registry
        from effgen.models.capabilities import list_registered_models
        registered = list_registered_models()
        self.print(f"\nCapability profiles registered: {len(registered)}")

    def _models_status_json(self) -> int:
        """Emit the GPU table + loaded models as JSON for ops/edge tooling."""
        gib = 1024 ** 3
        gpu_list: list[dict] = []
        cuda_available = True
        try:
            from effgen.gpu.cuda_compat import per_gpu_status
            for g in per_gpu_status():
                gpu_list.append({
                    "index": g.index,
                    "name": g.name,
                    "total_gb": round(g.total_bytes / gib, 3),
                    "used_gb": round(g.used_bytes / gib, 3),
                    "free_gb": round(g.free_bytes / gib, 3),
                    "utilization_pct": g.utilization_pct,
                })
            if not gpu_list:
                try:
                    import torch
                    cuda_available = torch.cuda.is_available()
                except ImportError:
                    cuda_available = False
        except ImportError:
            cuda_available = False

        from effgen.models.capabilities import list_registered_models
        from effgen.models.model_loader import ModelLoader
        loaded = ModelLoader().get_loaded_models()
        loaded_list = [
            {"name": name, "loaded": bool(model.is_loaded())}
            for name, model in loaded.items()
        ]
        payload = {
            "cuda_available": cuda_available,
            "gpus": gpu_list,
            "loaded_models": loaded_list,
            "capability_profiles": len(list_registered_models()),
        }
        print(json.dumps(payload, indent=2))
        return 0

    def _models_refresh(self, args):
        """Refresh the bundled model catalog from each provider's live API.

        Fetches the live model list for the requested provider(s), reports what
        was added / removed / changed versus the bundled snapshot, and (unless
        ``--dry-run``) updates the snapshot so later runs see the fresh list
        offline. Providers without a configured key are skipped with a note.
        """
        from effgen.models import _refresh

        requested = getattr(args, "provider", None)
        dry_run = bool(getattr(args, "dry_run", False))

        if requested:
            if requested not in _refresh.refreshable_providers():
                self.print_error(
                    f"Unknown provider '{requested}'. "
                    f"Refreshable: {', '.join(_refresh.refreshable_providers())}"
                )
                return 1
            providers = [requested]
        else:
            providers = _refresh.refreshable_providers()

        self.print_header("Refresh model catalog" + (" (dry run)" if dry_run else ""))
        any_done = False
        had_error = False
        for provider in providers:
            if not _refresh.has_credentials(provider) and provider != "hf":
                if requested:  # explicit request for a keyless provider is an error
                    self.print_error(f"No API key for '{provider}'.")
                    had_error = True
                else:
                    self.print(f"  {provider}: skipped (no key)")
                continue
            try:
                rep = _refresh.refresh_models(provider, persist=not dry_run)
            except Exception as e:  # noqa: BLE001 - report per-provider, keep going
                self.print_error(f"{provider}: refresh failed: {e}")
                had_error = True
                continue
            any_done = True
            diff = rep["diff"]
            n_add, n_rem, n_chg = (
                len(diff["added"]), len(diff["removed"]), len(diff["changed"])
            )
            verb = "would update" if dry_run else "updated"
            self.print_success(
                f"{provider}: {rep['live_count']} live models "
                f"(+{n_add} / -{n_rem} / ~{n_chg} changed) — {verb} snapshot"
            )
            for mid in diff["added"][:10]:
                self.print(f"    + {mid}")
            for mid in diff["removed"][:10]:
                self.print(f"    - {mid}")

        if had_error:
            return 1
        if not any_done:
            self.print_warning(
                "No providers refreshed. Set a provider API key, e.g. "
                "OPENAI_API_KEY / CEREBRAS_API_KEY / GROQ_API_KEY."
            )
        return 0

    def examples_commands(self, args):
        """
        Run example scripts.

        Args:
            args: Parsed command-line arguments
        """
        if args.example_command == 'list':
            return self._examples_list(args) or 0
        elif args.example_command == 'run':
            return self._examples_run(args) or 0
        elif args.example_command is None:
            return _print_group_help(args)
        else:
            self.print_error(f"Unknown examples command: {args.example_command}")
            return 1

    @staticmethod
    def _find_examples_dir() -> "Path | None":
        """Locate the bundled `examples/` directory.

        Examples ship with the source tree (repo root), not inside the installed
        `effgen` package, so probe several real locations rather than the old
        package-relative path that was broken for every pip-installed user.
        """
        candidates = []
        env_dir = os.environ.get("EFFGEN_EXAMPLES_DIR")
        if env_dir:
            candidates.append(Path(env_dir))
        # repo root: effgen/cli/_main.py -> <repo>/examples
        candidates.append(Path(__file__).resolve().parent.parent.parent / "examples")
        # current working directory (running from a checkout)
        candidates.append(Path.cwd() / "examples")
        for c in candidates:
            if c.is_dir() and any(c.rglob("*.py")):
                return c
        return None

    def _examples_list(self, args):
        """List available examples."""
        self.print_header("Available Examples")

        examples_dir = self._find_examples_dir()

        if examples_dir is None:
            self.print_warning("No examples directory found.")
            self.print(
                "Examples ship with the source repository, not the installed wheel.\n"
                "Run from a cloned checkout (repo root), or set "
                "EFFGEN_EXAMPLES_DIR=/path/to/examples."
            )
            return 0

        # Examples are grouped into subdirectories (basic/, advanced/, …), so
        # walk the tree and present each as its path relative to examples/.
        examples = []
        for file in examples_dir.rglob("*.py"):
            if file.name.startswith("_"):
                continue
            rel = file.relative_to(examples_dir).with_suffix("")
            examples.append(rel.as_posix())

        if self.console:
            table = Table(title=f"Example Scripts ({len(examples)})")
            table.add_column("Name", style="cyan")
            table.add_column("Command", style="magenta")

            for example in sorted(examples):
                table.add_row(example, f"effgen examples run {example}")

            self.console.print(table)
        else:
            for example in sorted(examples):
                print(f"- {example}")

    def _examples_run(self, args):
        """Run an example script."""
        if not args.name:
            self.print_error("Example name required")
            return 1

        examples_dir = self._find_examples_dir()
        if examples_dir is None:
            self.print_error(
                "No examples directory found. Run from a source checkout or set "
                "EFFGEN_EXAMPLES_DIR=/path/to/examples."
            )
            return 1
        # Accept either a bare name or a subdir path (e.g. "basic/quickstart").
        name = args.name[:-3] if args.name.endswith(".py") else args.name
        example_path = (examples_dir / f"{name}.py").resolve()
        # Path-traversal guard: stay within the examples directory.
        try:
            example_path.relative_to(examples_dir.resolve())
        except ValueError:
            self.print_error(f"Invalid example path: {args.name}")
            return 1

        if not example_path.exists():
            # Try to match by basename across subdirectories for convenience.
            matches = list(examples_dir.rglob(f"{Path(name).name}.py"))
            if len(matches) == 1:
                example_path = matches[0]
            else:
                self.print_error(f"Example not found: {args.name}")
                if matches:
                    self.print("Did you mean one of:")
                    for m in matches:
                        self.print(f"  {m.relative_to(examples_dir).with_suffix('').as_posix()}")
                return 1

        self.print_header(f"Running Example: {args.name}")
        self.print()

        # Load and run example
        try:
            spec = importlib.util.spec_from_file_location("example", example_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Run main function if exists
            if hasattr(module, 'main'):
                module.main()
            else:
                self.print_warning("Example does not have a main() function")

        except Exception as e:
            self.print_error(f"Error running example: {e}")
            if getattr(args, 'verbose', False):
                import traceback
                traceback.print_exc()
            return 1
        return 0


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


def create_parser():
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
    sessions_parser = subparsers.add_parser('sessions', help='Manage persistent sessions')
    sessions_parser.set_defaults(_group_parser=sessions_parser)
    sessions_subparsers = sessions_parser.add_subparsers(dest='session_command', help='Sessions command')
    _sessions_list = sessions_subparsers.add_parser('list', help='List sessions')
    _sessions_list.add_argument('--json', dest='output_json', action='store_true',
                                help='Output the session list as JSON')
    sd = sessions_subparsers.add_parser('delete', help='Delete a session')
    sd.add_argument('session_id', help='Session id')
    se = sessions_subparsers.add_parser('export', help='Export a session')
    se.add_argument('session_id', help='Session id')
    se.add_argument('--format', choices=['json', 'text'], default='json')
    sc = sessions_subparsers.add_parser('cleanup', help='Delete sessions older than N days')
    sc.add_argument('--days', type=int, default=30)

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
    eval_parser.add_argument('-o', '--output', help='Output file for results (JSON)')
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
    compare_parser.add_argument('-o', '--output', help='Output file for results (JSON or Markdown)')
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
    cost_parser = subparsers.add_parser('cost', help='View cost spend and manage budgets')
    cost_parser.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')
    cost_subparsers = cost_parser.add_subparsers(dest='cost_command', help='Cost command')
    _cost_today = cost_subparsers.add_parser('today', help='Show per-provider/model spend for the last 24 hours')
    _cost_today.add_argument('--json', dest='output_json', action='store_true', default=argparse.SUPPRESS, help='Output as JSON')
    _cost_week = cost_subparsers.add_parser('week', help='Show rolling 7-day spend summary')
    _cost_week.add_argument('--json', dest='output_json', action='store_true', default=argparse.SUPPRESS, help='Output as JSON')
    _cost_byprov = cost_subparsers.add_parser('by-provider', help='Show lifetime totals grouped by provider')
    _cost_byprov.add_argument('--json', dest='output_json', action='store_true', default=argparse.SUPPRESS, help='Output as JSON')
    cost_set_budget = cost_subparsers.add_parser('set-budget', help='Set a daily spend budget')
    cost_set_budget.add_argument('amount', type=float, help='Daily budget in USD (e.g. 1.0)')
    cost_subparsers.add_parser('clear-budget', help='Remove configured budget limits')

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


def _handle_cost_command(args, cli: "CLIInterface") -> int:
    """Handle the 'effgen cost' subcommand: spend dashboard and budget management."""
    import json as _json

    try:
        from effgen.models._cost_store import SQLiteCostStore
    except ImportError:
        cli.print_error("Cost store not available. Please reinstall effGen.")
        return 1

    cost_cmd = getattr(args, 'cost_command', None)

    # Budget management subcommands
    from effgen.models._cost import _budget_config_path, format_usd
    budget_path = _budget_config_path()

    if cost_cmd == 'set-budget':
        amount = float(args.amount)
        budget_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if budget_path.exists():
            try:
                existing = _json.loads(budget_path.read_text())
            except Exception:
                pass
        existing['daily'] = amount
        budget_path.write_text(_json.dumps(existing, indent=2))
        cli.print_success(f"Daily budget set to {format_usd(amount)} USD")
        return 0

    if cost_cmd == 'clear-budget':
        if budget_path.exists():
            try:
                cfg = _json.loads(budget_path.read_text())
                cfg.pop('daily', None)
                cfg.pop('monthly', None)
                budget_path.write_text(_json.dumps(cfg, indent=2))
                cli.print_success("Budget limits cleared.")
            except Exception as e:
                cli.print_error(f"Failed to clear budget: {e}")
                return 1
        else:
            cli.print("No budget configured.")
        return 0

    # Spend-report subcommands
    store = SQLiteCostStore()

    if cost_cmd == 'today' or cost_cmd is None:
        events = store.query_today()
        period_label = "Last 24 hours"
    elif cost_cmd == 'week':
        events = store.query_week()
        period_label = "Last 7 days"
    elif cost_cmd == 'by-provider':
        events = store.query_all()
        period_label = "Lifetime"
    else:
        cli.print_error(f"Unknown cost command: {cost_cmd}")
        cli.print("Usage: effgen cost [today|week|by-provider|set-budget|clear-budget]")
        return 1

    # Aggregate events by (provider, model), except by-provider which intentionally
    # collapses all models for each provider into one lifetime row.
    group_by_provider = cost_cmd == 'by-provider'
    agg: dict[tuple[str, str], dict] = {}
    for ev in events:
        model_label = "all models" if group_by_provider else ev.model
        key = (ev.provider, model_label)
        if key not in agg:
            agg[key] = {
                'provider': ev.provider,
                'model': model_label,
                'requests': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'cost_usd': 0.0,
            }
        agg[key]['requests'] += 1
        agg[key]['prompt_tokens'] += ev.prompt_tokens
        agg[key]['completion_tokens'] += ev.completion_tokens
        agg[key]['cost_usd'] += ev.cost_usd

    rows = sorted(agg.values(), key=lambda r: r['cost_usd'], reverse=True)
    total_cost = sum(r['cost_usd'] for r in rows)
    total_requests = sum(r['requests'] for r in rows)

    # Cost label: a genuine free tier reads "free" and a model with no
    # published price reads "unpriced", instead of a misleading "$0.000000".
    from effgen.models._cost import pricing_status as _pricing_status

    def _cost_label(row: dict) -> str:
        cost = row['cost_usd']
        if cost > 0 or row['model'] == 'all models':
            return f"${cost:.6f}"
        status = _pricing_status(row['provider'], row['model'])
        if status == 'free':
            return 'free'
        if status == 'unpriced':
            return 'unpriced'
        return f"${cost:.6f}"

    # Load budget for display
    budget_cfg = {}
    if budget_path.exists():
        try:
            budget_cfg = _json.loads(budget_path.read_text())
        except Exception:
            pass
    daily_budget = budget_cfg.get('daily')

    # JSON output — machine-readable spend report.
    if getattr(args, 'output_json', False):
        print(_json.dumps({
            "period": period_label,
            "total_requests": total_requests,
            "total_cost_usd": round(total_cost, 8),
            "daily_budget_usd": daily_budget,
            "rows": [
                {
                    "provider": r["provider"],
                    "model": r["model"],
                    "requests": r["requests"],
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
                    "cost_usd": round(r["cost_usd"], 8),
                    "cost_label": _cost_label(r),
                }
                for r in rows
            ],
        }, indent=2))
        return 0

    # Friendly empty state: warm message + a next step instead of a blank table.
    if not rows:
        cli.print_header(f"effGen Cost Summary — {period_label}")
        cli.print("No spend recorded yet. 🎉")
        cli.print("Run an agent to start tracking — e.g. effgen run \"What is 2+2?\" -m gpt-5-nano "
                  "--provider openai")
        cli.print("Then set a cap with: effgen cost set-budget 1.00")
        return 0

    if console_is_interactive(cli.console):
        table = Table(title=f"effGen Cost Summary — {period_label}", show_footer=True)
        table.add_column("Provider", style="cyan", no_wrap=True)
        # Wrap (fold) long model ids instead of truncating with an ellipsis.
        table.add_column("Model", style="white", overflow="fold")
        table.add_column("Requests", style="yellow", justify="right")
        table.add_column("Prompt Tokens", style="blue", justify="right")
        table.add_column("Completion Tokens", style="blue", justify="right")
        table.add_column("Cost (USD)", style="green", justify="right",
                         footer=f"${total_cost:.6f}")

        for r in rows:
            table.add_row(
                r['provider'],
                r['model'],
                str(r['requests']),
                f"{r['prompt_tokens']:,}",
                f"{r['completion_tokens']:,}",
                _cost_label(r),
            )

        cli.console.print(table)
        cli.console.print(f"\n[bold]Total:[/bold] {total_requests} requests  "
                          f"[green]${total_cost:.6f} USD[/green]")
        if daily_budget is not None and cost_cmd in (None, 'today'):
            ratio = total_cost / daily_budget if daily_budget > 0 else 0
            filled = min(20, max(0, int(ratio * 20)))
            bar = "█" * filled + "░" * (20 - filled)
            color = "red" if ratio >= 1.0 else "yellow" if ratio >= 0.8 else "green"
            cli.console.print(
                f"[bold]Daily budget:[/bold] [{color}]{bar}[/{color}] "
                f"{format_usd(total_cost)} / {format_usd(daily_budget)} ({ratio*100:.0f}%)"
            )
    else:
        print(f"\neffGen Cost Summary — {period_label}")
        print("-" * 80)
        print(f"{'Provider':<12} {'Model':<48} {'Reqs':>5} {'Cost (USD)':>12}")
        print("-" * 80)
        for r in rows:
            # Show the full model id (wrap rather than truncate).
            model = r['model']
            cost_label = _cost_label(r)
            if len(model) > 48:
                print(f"{r['provider']:<12} {model}")
                print(f"{'':<12} {'':<48} {r['requests']:>5} {cost_label:>12}")
            else:
                print(f"{r['provider']:<12} {model:<48} {r['requests']:>5} {cost_label:>12}")
        print("-" * 80)
        print(f"{'TOTAL':<12} {'':<48} {total_requests:>5} ${total_cost:>11.6f}")
        if daily_budget is not None and cost_cmd in (None, 'today'):
            ratio = total_cost / daily_budget if daily_budget > 0 else 0
            print(f"\nDaily budget: {format_usd(total_cost)} / {format_usd(daily_budget)} ({ratio*100:.0f}%)")

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


def _handle_workflow_command(args, cli) -> int:
    """Handle the 'workflow' CLI subcommand."""
    from effgen.core.workflow import WorkflowDAG

    wf_cmd = getattr(args, 'workflow_command', None)
    json_mode = getattr(args, 'output_json', False)
    if json_mode:
        cli._human_to_stderr = True
    show_diagram = getattr(args, 'diagram', False)

    def _print_diagram(dag, node_results=None):
        from effgen.ui.workflow_viz import workflow_diagram_lines
        order = dag.topological_order()
        levels = dag._compute_levels(order)
        lines = workflow_diagram_lines(
            dag.name,
            [n.id for n in dag.nodes],
            [e.to_dict() for e in dag.edges],
            levels,
            node_results=node_results,
        )
        for style, text in lines:
            if cli.console and style:
                cli.console.print(f"[{style}]{text}[/{style}]")
            else:
                cli.print(text)

    if wf_cmd == 'validate':
        try:
            dag = WorkflowDAG.from_yaml(args.file)
            order = dag.topological_order()
            if json_mode:
                print(json.dumps({
                    "valid": True,
                    "name": dag.name,
                    "nodes": len(dag.nodes),
                    "edges": len(dag.edges),
                    "execution_order": order,
                }, indent=2, ensure_ascii=False))
                return 0
            cli.print(f"Workflow '{dag.name}' is valid.")
            cli.print(f"  Nodes: {len(dag.nodes)}")
            cli.print(f"  Edges: {len(dag.edges)}")
            cli.print(f"  Execution order: {' -> '.join(order)}")
            if show_diagram:
                cli.print("")
                _print_diagram(dag)
            return 0
        except Exception as e:
            if json_mode:
                print(json.dumps({"valid": False, "error": str(e)}, indent=2, ensure_ascii=False))
                return 1
            cli.print(f"Validation failed: {e}")
            return 1

    elif wf_cmd == 'run':
        try:
            model_name = getattr(args, 'model', None)

            def _agent_factory(nd):
                from effgen.core.agent import Agent, AgentConfig
                from effgen.models import load_model
                agent_field = nd.get('agent')
                explicit = model_name or nd.get('model')
                if explicit:
                    model = load_model(explicit)
                elif agent_field:
                    # No top-level -m/--model and no per-node 'model:' key: the
                    # node's 'agent:' value is the natural place a user sets a
                    # model id (e.g. `agent: gpt-5-nano`). Try it as one before
                    # falling back to the local default; a value that does not
                    # resolve to a real model fails loudly instead of silently
                    # running a different (free, local) model with no warning.
                    try:
                        model = load_model(agent_field)
                    except Exception as exc:
                        raise ValueError(
                            f"Workflow node '{nd['id']}' has agent: {agent_field!r}, "
                            f"which does not resolve to a model ({exc}). Set a "
                            "'model:' key on the node, or pass -m/--model, to "
                            "choose its model explicitly."
                        ) from exc
                else:
                    model = load_model('Qwen/Qwen2.5-1.5B-Instruct')
                # A node may name a preset (research/coding/general/...) to get a
                # ready-made tool-equipped agent; otherwise build a plain agent.
                preset = nd.get('preset')
                if preset:
                    from effgen.presets import create_agent
                    return create_agent(preset, model=model)
                config = AgentConfig(
                    name=agent_field or nd['id'],
                    model=model,
                    max_iterations=nd.get('max_iterations', 5),
                )
                return Agent(config)

            quiet = getattr(args, 'quiet', False)
            dag = WorkflowDAG.from_yaml(args.file, agent_factory=_agent_factory)
            if not quiet:
                cli.print(f"Running workflow '{dag.name}' ({len(dag.nodes)} nodes)...")

            # Per-node ``task:`` strings declared in the YAML become each node's
            # default input (so `effgen workflow run workflow.yaml` works with no
            # flags). --input / --task then override or supplement them.
            yaml_inputs: dict = {}
            for node in dag.nodes:
                node_task = node.metadata.get('task')
                if node_task:
                    yaml_inputs[node.id] = node_task

            bare_task = getattr(args, 'task', None)
            initial_inputs: dict | str = dict(yaml_inputs)
            if getattr(args, 'input', None):
                for node_id, task_str in args.input:
                    initial_inputs[node_id] = task_str
            if bare_task:
                if dag.entry_nodes():
                    for nid in dag.entry_nodes():
                        initial_inputs[nid] = bare_task
                elif not initial_inputs:
                    initial_inputs = bare_task

            try:
                result = dag.run(initial_inputs=initial_inputs)
            finally:
                # Release each node's agent so we don't leak handles / emit
                # garbage-collected-without-close warnings.
                for node in dag.nodes:
                    agent = getattr(node, "agent", None)
                    if agent is not None and hasattr(agent, "close"):
                        try:
                            agent.close()
                        except Exception:
                            pass

            if json_mode:
                print(json.dumps(result.to_dict(), indent=2, default=str, ensure_ascii=False))
                return 0 if result.success else 1

            if not quiet:
                cli.print(f"\nWorkflow {'succeeded' if result.success else 'FAILED'} "
                          f"in {result.execution_time:.2f}s")

                if show_diagram:
                    cli.print("")
                    _print_diagram(dag, node_results=result.node_results)
                else:
                    for nr in result.node_results:
                        status = nr['status']
                        cli.print(f"  [{status:>9s}] {nr['id']} ({nr['execution_time']:.2f}s)")

                if result.success:
                    # Show final outputs
                    cli.print("\nOutputs:")
                    for key, val in result.outputs.items():
                        cli.print(f"  {key}: {str(val)[:200]}")

            return 0 if result.success else 1

        except Exception as e:
            if json_mode:
                print(json.dumps({"success": False, "error": str(e)}, indent=2, ensure_ascii=False))
                return 1
            cli.print(f"Workflow execution failed: {e}")
            return 1

    else:
        return _print_group_help(args)


def _batch_structured_kwargs(args) -> dict:
    """Build ``output_schema`` / ``output_model`` run-kwargs from the CLI flags.

    ``--schema PATH`` loads a JSON Schema file; ``--output-model module:Class``
    imports a Pydantic model. Each validates every row and writes the parsed
    object; a row that cannot be coerced to the schema is reported as a failed
    row with a reason rather than a silently off-schema string. Raises
    ``ValueError`` with an actionable message on a bad path or spec.
    """
    schema_path = getattr(args, 'schema_path', None)
    model_spec = getattr(args, 'output_model', None)
    if schema_path and model_spec:
        raise ValueError("Use only one of --schema / --output-model, not both.")
    if schema_path:
        p = Path(schema_path)
        if not p.exists():
            raise ValueError(f"Schema file not found: {schema_path}")
        try:
            schema = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{schema_path}: not valid JSON: {e}") from e
        if not isinstance(schema, dict):
            raise ValueError(f"{schema_path}: a JSON Schema must be a JSON object.")
        return {"output_schema": schema}
    if model_spec:
        if ":" not in model_spec:
            raise ValueError(
                "--output-model must be 'module:ClassName' "
                "(e.g. myproject.schemas:Ticket)."
            )
        mod_name, _, cls_name = model_spec.partition(":")
        import importlib
        # Make a project-local module importable from a headless run.
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            raise ValueError(f"Could not import module '{mod_name}': {e}") from e
        cls = getattr(mod, cls_name, None)
        if cls is None:
            raise ValueError(f"Module '{mod_name}' has no attribute '{cls_name}'.")
        from effgen.core.structured_output import is_pydantic_model_class
        if not is_pydantic_model_class(cls):
            raise ValueError(
                f"{model_spec} is not a Pydantic model class "
                "(it must subclass pydantic.BaseModel)."
            )
        return {"output_model": cls}
    return {}


def _read_done_indices(output_path: Path) -> dict:
    """Read an existing JSONL output file into ``{index: row}`` for --resume."""
    done: dict[int, dict] = {}
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = row.get("index") if isinstance(row, dict) else None
            if isinstance(idx, int):
                done[idx] = row
    return done


def _handle_batch_command(args, cli) -> int:
    """Handle the 'batch' CLI subcommand."""
    from effgen.core.batch import _QUERY_ALIASES, BatchConfig, BatchRunner
    from effgen.core.batch import SUPPORTED_OUTPUT_FORMATS as _BATCH_OUTPUT_FORMATS

    # Accept the input file as a positional argument or via -i/--input; the
    # explicit flag wins if both are given.
    input_path = getattr(args, 'input', None) or getattr(args, 'input_file', None)
    output_path = getattr(args, 'output', None)
    model_name = getattr(args, 'model', None) or 'Qwen/Qwen2.5-1.5B-Instruct'
    preset_name = getattr(args, 'preset', None)
    guardrails = getattr(args, 'guardrails', None)
    query_field = getattr(args, 'query_field', 'query')
    max_tokens = getattr(args, 'max_tokens', None)
    temperature = getattr(args, 'temperature', None)
    system_prompt = getattr(args, 'system_prompt', None)
    strict = getattr(args, 'strict', False)
    resume = getattr(args, 'resume', False)

    # Headless JSON contract, same as `run --json`: keep stdout pure (only the
    # JSON result document) by routing human chatter to stderr, and emit a
    # typed error object to stdout on EVERY failure path — early argument/read
    # failures included — so a `| jq` consumer never gets empty input.
    json_mode = getattr(args, 'output_json', False)
    if json_mode:
        cli._human_to_stderr = True

    def _json_error(exc: Exception) -> int:
        if json_mode:
            print(json.dumps({
                "success": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }, indent=2, ensure_ascii=False))
        return 1

    if not input_path:
        msg = ("No input file given. Pass one as `effgen batch FILE` "
               "or with -i/--input (JSONL, CSV, JSON, or plain text).")
        cli.print_error(msg)
        return _json_error(ValueError(msg))

    # Extra kwargs forwarded to each agent.run() call.
    run_kwargs: dict = {}
    if max_tokens is not None:
        run_kwargs['max_tokens'] = max_tokens
    if temperature is not None:
        run_kwargs['temperature'] = temperature
    try:
        run_kwargs.update(_batch_structured_kwargs(args))
    except ValueError as e:
        cli.print_error(str(e))
        return _json_error(e)

    out_suffix = Path(output_path).suffix.lower() if output_path else None
    # Reject an unsupported --output extension before loading a model or making
    # a single billed call, naming the formats that do work — rather than
    # running the whole batch and failing at write time.
    if output_path and out_suffix not in _BATCH_OUTPUT_FORMATS:
        shown = out_suffix or "(none)"
        msg = (
            f"Unsupported --output format: {shown}. "
            f"Use one of: {', '.join(sorted(_BATCH_OUTPUT_FORMATS))}."
        )
        cli.print_error(msg)
        return _json_error(ValueError(msg))
    # A .jsonl output streams each finished row as it completes, so a crash
    # mid-job keeps the rows already done; --resume then skips those on rerun.
    stream_jsonl = bool(output_path) and out_suffix == ".jsonl"
    if resume and not stream_jsonl:
        msg = "--resume requires a .jsonl --output file."
        cli.print_error(msg)
        return _json_error(ValueError(msg))

    if not output_path and not json_mode:
        cli.print(
            "Warning: no -o/--output and no --json — each row's answer, cost, "
            "tokens, and any error detail will be discarded; only the summary "
            "line at the end of this run is kept."
        )

    agent = None
    out_fh = None
    try:
        # Create agent
        if preset_name:
            from effgen.models import load_model
            from effgen.presets import create_agent
            model = load_model(model_name)
            agent = create_agent(
                preset_name, model, system_prompt=system_prompt, guardrails=guardrails,
            )
        else:
            from effgen.core.agent import Agent, AgentConfig
            from effgen.models import load_model
            model = load_model(model_name)
            config_kwargs: dict = {}
            if system_prompt is not None:
                config_kwargs['system_prompt'] = system_prompt
            config = AgentConfig(
                name="batch-agent", model=model, max_iterations=5,
                guardrails=guardrails, **config_kwargs,
            )
            agent = Agent(config)

        runner = BatchRunner(agent)
        cli.print(f"Loading queries from {input_path}...")

        # Read queries once. A malformed input line is skipped with a message
        # naming the file and line number (not the parser's byte offset);
        # --strict turns the first bad line into a hard failure instead.
        skipped: list[int] = []
        empty_rows: list[tuple[int, list[str]]] = []

        def _on_skip(lineno: int, msg: str) -> None:
            skipped.append(lineno)
            cli.print(f"Skipping malformed input at {input_path}:{lineno}: {msg}")

        def _on_empty(lineno: int, keys: list[str]) -> None:
            empty_rows.append((lineno, keys))

        try:
            queries = runner._read_queries(
                Path(input_path), query_field, strict=strict,
                on_skip=_on_skip, on_empty=_on_empty,
            )
        except Exception as e:  # noqa: BLE001 - one clear message, no traceback
            cli.print_error(f"Could not read {input_path}: {e}")
            return _json_error(e)
        if skipped:
            cli.print(
                f"Skipped {len(skipped)} malformed line(s); "
                f"{len(queries)} queries loaded."
            )

        # A row with no recognized query text (neither --query-field nor the
        # aliases query/input/prompt/question/text) can't run. Name the fields it
        # did carry and how to point at the right one, rather than letting each
        # empty row fail with a generic empty-task message.
        if empty_rows:
            lineno, keys = empty_rows[0]
            fields = ", ".join(keys) if keys else "none"
            more = f" (and {len(empty_rows) - 1} more)" if len(empty_rows) > 1 else ""
            msg = (
                f"Row {lineno}{more} has no query text. Fields present: {fields}. "
                f"Set the query column with --query-field NAME, or key rows on one "
                f"of: {', '.join(_QUERY_ALIASES)}."
            )
            cli.print_error(msg)
            return _json_error(ValueError(msg))

        # --resume: skip input rows already present in the JSONL output.
        done_rows: dict[int, dict] = {}
        if resume:
            done_rows = {
                i: row for i, row in _read_done_indices(Path(output_path)).items()
                if 0 <= i < len(queries)
            }
        run_positions = [i for i in range(len(queries)) if i not in done_rows]
        run_queries = [queries[i] for i in run_positions]
        if done_rows:
            cli.print(
                f"Resuming: {len(done_rows)} row(s) already present, "
                f"running the remaining {len(run_queries)}."
            )

        # Open the streaming output before the run so completed rows persist
        # immediately. Resume appends to the existing file; a fresh run truncates.
        if stream_jsonl:
            mode = "a" if (resume and Path(output_path).exists()) else "w"
            out_fh = open(output_path, mode, encoding="utf-8")

        def _on_result(pos: int, query: str, resp) -> None:
            if out_fh is None:
                return
            orig_idx = run_positions[pos]
            row = BatchRunner._result_row(orig_idx, resp, query)
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_fh.flush()

        # --json emits a single JSON document to stdout: no live progress bar,
        # which would otherwise render there on an interactive terminal.
        animate = not json_mode and _progress.animation_enabled(
            quiet=getattr(args, 'quiet', False),
            no_animation=getattr(args, 'no_animation', False),
        )
        with _progress.StepProgress(
            cli.console, total=len(run_queries), description="Batch", animate=animate,
        ) as _bar:
            batch_config = BatchConfig(
                max_concurrency=args.concurrency,
                batch_size=args.batch_size,
                retry_failed=args.retries,
                timeout_per_item=args.timeout,
                progress_callback=lambda done, total: _bar.update(done, total),
                on_result=_on_result if out_fh is not None else None,
            )
            result = runner.run(run_queries, config=batch_config, **run_kwargs)

        if out_fh is not None:
            out_fh.close()
            out_fh = None

        # Combine this run with any rows carried over by --resume so the
        # headline counts and totals reflect the whole job, not just the rerun.
        done_success = sum(1 for r in done_rows.values() if r.get("success"))
        total = len(queries)
        succeeded = done_success + result.succeeded
        failed = total - succeeded

        done_cost = sum(
            r["cost_usd"] for r in done_rows.values()
            if isinstance(r.get("cost_usd"), int | float)
        )
        done_tokens = sum(
            r.get("total_tokens", 0) for r in done_rows.values()
            if isinstance(r.get("total_tokens"), int)
        )
        done_prompt_tokens = sum(
            r.get("prompt_tokens", 0) for r in done_rows.values()
            if isinstance(r.get("prompt_tokens"), int)
        )
        done_completion_tokens = sum(
            r.get("completion_tokens", 0) for r in done_rows.values()
            if isinstance(r.get("completion_tokens"), int)
        )
        cost_present = result.total_cost_usd is not None or done_cost > 0
        total_cost = (result.total_cost_usd or 0.0) + done_cost if cost_present else None
        total_tokens = result.total_tokens + done_tokens
        total_prompt_tokens = result.total_prompt_tokens + done_prompt_tokens
        total_completion_tokens = result.total_completion_tokens + done_completion_tokens

        summary = (
            f"\nBatch complete: {succeeded}/{total} succeeded "
            f"in {result.total_time:.2f}s"
        )
        if total_tokens:
            summary += f" · {total_tokens:,} tokens"
        from effgen.ui.render import format_cost
        cost_str = format_cost(total_cost)
        if cost_str is not None:
            summary += f" · {cost_str}"
        cli.print(summary)

        # Non-streaming formats (.csv/.json) get one batched write at the end.
        if output_path and not stream_jsonl:
            runner.write_results(
                result, output_path, query_list=run_queries,
                excel_bom=getattr(args, 'excel_bom', False),
            )
            cli.print(f"Results written to {output_path}")
        elif output_path:
            cli.print(f"Results written to {output_path}")

        if json_mode:
            rows = list(done_rows.values())
            for pos, resp in enumerate(result.results):
                rows.append(BatchRunner._result_row(run_positions[pos], resp, run_queries[pos]))
            rows.sort(key=lambda r: r.get("index", 0))
            print(json.dumps({
                "total": total,
                "succeeded": succeeded,
                "failed": failed,
                "success_rate": round(succeeded / total, 4) if total else 0.0,
                "total_time": round(result.total_time, 2),
                "total_cost_usd": round(total_cost, 8) if total_cost is not None else None,
                "total_tokens": total_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "rows": rows,
            }, indent=2, ensure_ascii=False))

        return 0 if failed == 0 else 1

    except Exception as e:
        cli.print(f"Batch execution failed: {e}")
        return _json_error(e)
    finally:
        if out_fh is not None:
            try:
                out_fh.close()
            except Exception:  # noqa: BLE001
                pass
        # Release the agent's resources so the CLI never trips its own
        # "garbage-collected without close()" warning.
        if agent is not None:
            try:
                agent.close()
            except Exception:
                logging.getLogger(__name__).debug("Batch agent close() failed", exc_info=True)


def _resolve_eval_suite(suite_arg: str, difficulty=None, max_cases=None):
    """Resolve a ``--suite`` argument to a ``TestSuite``.

    Accepts a built-in suite name **or** a path to a ``.jsonl`` / ``.json`` file
    of your own test cases (each an object with ``query``/``expected_output`` and
    optional ``difficulty``/``tags``), so a bake-off can run on your own data —
    not just the bundled suites. Optionally filters by ``difficulty`` and trims to
    the first ``max_cases``. Raises ``KeyError`` (with the list of valid names)
    for an unknown name, or ``FileNotFoundError`` / ``ValueError`` for a bad file.
    """
    from effgen.eval import get_suite
    from effgen.eval.suites import TestSuite

    p = Path(suite_arg)
    if p.suffix.lower() in (".jsonl", ".json") or p.exists():
        if not p.exists():
            raise FileNotFoundError(f"Test-case file not found: {suite_arg}")
        from effgen.eval.evaluator import TestCase
        raw = p.read_text(encoding="utf-8")
        records = []
        if p.suffix.lower() == ".json":
            data = json.loads(raw)
            records = data if isinstance(data, list) else [data]
        else:
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        cases = [TestCase.from_dict(r) for r in records]
        if not cases:
            raise ValueError(f"No test cases found in {suite_arg}")
        suite = TestSuite(test_cases=cases)
        suite.name = p.stem
    else:
        suite = get_suite(suite_arg)

    if difficulty:
        from effgen.eval.evaluator import Difficulty
        suite.test_cases = suite.filter(difficulty=Difficulty(difficulty))
    if max_cases is not None and max_cases > 0:
        suite.test_cases = suite.test_cases[:max_cases]
    return suite


def _handle_eval_command(args, cli) -> int:
    """Handle 'effgen eval' subcommand."""
    from effgen.eval import AgentEvaluator, RegressionTracker, list_suites
    from effgen.eval.evaluator import ScoringMode

    # --json: route human chatter to stderr so stdout carries only the JSON
    # results document (CI gates parse it).
    json_mode = getattr(args, 'output_json', False)
    if json_mode:
        cli._human_to_stderr = True

    provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
    if prov_err:
        cli.print_error(prov_err)
        return 2

    suite_name = args.suite
    model_name = getattr(args, 'model', None) or 'Qwen/Qwen2.5-1.5B-Instruct'
    preset_name = getattr(args, 'preset', None)
    scoring = ScoringMode(args.scoring)
    threshold = args.threshold
    fail_under = getattr(args, 'fail_under', 0.5)
    baseline_dir = getattr(args, 'baseline_dir', None)
    temperature = getattr(args, 'temperature', None)
    difficulty = getattr(args, 'difficulty', None)
    max_cases = getattr(args, 'max_cases', None)

    agent = None
    try:
        # List suites if requested
        if suite_name == 'list':
            suites = list_suites()
            if json_mode:
                print(json.dumps(
                    [{"name": n, "description": d} for n, d in suites.items()],
                    indent=2, ensure_ascii=False,
                ))
                return 0
            cli.print_header("Available Evaluation Suites")
            for name, desc in suites.items():
                cli.print(f"  {name:16s} — {desc}")
            return 0

        # A bad data file (unknown field, empty, missing) is a user error, not a
        # crash — report it and exit 2 (matching `compare`) instead of a
        # traceback.
        try:
            suite = _resolve_eval_suite(suite_name, difficulty=difficulty, max_cases=max_cases)
        except (ValueError, FileNotFoundError) as exc:
            cli.print(
                f"Could not load suite '{suite_name}' ({exc}). "
                f"Use a built-in suite ({', '.join(list_suites())}) "
                "or a path to a .jsonl/.json file of test cases."
            )
            return 2

        # Report any narrowing applied
        if difficulty:
            cli.print(f"Filtered to {len(suite.test_cases)} {difficulty} test cases")
        if max_cases:
            cli.print(f"Limited to first {len(suite.test_cases)} cases")

        cli.print(f"Loading model {model_name}...")

        # Create agent
        if preset_name:
            from effgen.models import load_model
            from effgen.presets import create_agent
            model = load_model(model_name, provider=provider)
            agent = create_agent(preset_name, model, temperature=temperature)
        else:
            from effgen.core.agent import Agent, AgentConfig
            from effgen.models import load_model
            model = load_model(model_name, provider=provider)
            config_kwargs: dict = {"name": "eval-agent", "model": model, "max_iterations": 10}
            if temperature is not None:
                config_kwargs["temperature"] = temperature
            config = AgentConfig(**config_kwargs)
            agent = Agent(config)

        cli.print(f"Running {suite_name} suite ({len(suite)} cases, scoring={args.scoring})...")
        evaluator = AgentEvaluator(agent, scoring=scoring, pass_threshold=threshold)
        # --json emits a single JSON document to stdout: no live progress bar,
        # which would otherwise render there on an interactive terminal.
        animate = not json_mode and _progress.animation_enabled(
            quiet=getattr(args, 'quiet', False),
            no_animation=getattr(args, 'no_animation', False),
        )
        with _progress.StepProgress(
            cli.console, total=len(suite), description="Eval", animate=animate,
        ) as _bar:
            results = evaluator.run_suite(
                suite, progress_callback=lambda done, total: _bar.update(done, total),
            )

        # Display results
        summary = results.summary()
        cli.print_header(f"Evaluation Results: {suite_name}")
        # Under --json the summary is human chatter: route it to stderr so
        # stdout carries only the JSON document below.
        render_table(
            columns=["Metric", "Value"],
            rows=[
                ["Accuracy", f"{summary['accuracy']:.1%} ({summary['passed']}/{summary['total']})"],
                ["Avg Latency", f"{summary['avg_latency']:.4f}s"],
                ["Total Tokens", f"{summary['total_tokens']}"],
                ["Tool Accuracy", f"{summary['avg_tool_accuracy']:.1%}"],
            ],
            console=None if json_mode else cli.console,
            styles=["cyan", None],
            file=sys.stderr if json_mode else None,
        )

        if summary.get('by_difficulty'):
            cli.print("\n  By Difficulty:")
            for d, info in sorted(summary['by_difficulty'].items()):
                cli.print(f"    {d:8s}: {info['accuracy']:.1%} ({info['passed']}/{info['total']})")

        # Show per-case details for failures
        failures = [r for r in results.results if not r.passed]
        if failures:
            cli.print(f"\n  Failed cases ({len(failures)}):")
            for r in failures[:10]:
                cli.print(f"    - {r.test_case.query[:60]}...")
                cli.print(f"      Expected: {r.test_case.expected_output[:40]}")
                cli.print(f"      Got:      {r.agent_output[:40]}")

        # Save baseline — keyed on the resolved suite name (a custom dataset's
        # name is the file stem, never the full path) so a baseline file for a
        # nested or absolute suite path does not fail to write.
        if args.save_baseline:
            from effgen import __version__
            tracker = RegressionTracker(baselines_dir=baseline_dir)
            path = tracker.save_baseline(suite.name, results, version=__version__)
            cli.print(f"\n  Baseline saved to {path}")

        # Compare baseline
        report = None
        if args.compare_baseline:
            from effgen import __version__
            tracker = RegressionTracker(baselines_dir=baseline_dir)
            report = tracker.compare(suite.name, results, version=__version__)
            cli.print(f"\n{report.to_markdown()}")

        # Write output
        if args.output:
            Path(args.output).write_text(results.to_json(), encoding="utf-8")
            cli.print(f"\n  Results written to {args.output}")

        # Emit the same results document to stdout for piping/CI gating.
        if json_mode:
            print(results.to_json())

        # Exit-code gate. A detected blocking regression against a saved
        # baseline always fails the build; otherwise the gate is the suite
        # accuracy against --fail-under (--threshold is a separate per-case
        # setting and does not drive the exit code).
        if report is not None and report.has_regressions:
            cli.print(
                "\n  Exit gate: FAIL — blocking regression against baseline (--compare-baseline)."
            )
            return 1
        gate_passed = results.accuracy >= fail_under
        cli.print(
            f"\n  Exit gate: {'PASS' if gate_passed else 'FAIL'} — accuracy "
            f"{results.accuracy:.1%} {'>=' if gate_passed else '<'} "
            f"--fail-under {fail_under:.0%}"
        )
        return 0 if gate_passed else 1

    except KeyError as e:
        cli.print(f"Error: {e}")
        cli.print("Available suites:")
        for name, desc in list_suites().items():
            cli.print(f"  {name:16s} — {desc}")
        return 1
    except Exception as e:
        cli.print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Release agent resources so eval stops emitting the GC-without-close
        # warning on every run (matches `compare`'s cleanup).
        if agent is not None:
            close = getattr(agent, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass


def _render_comparison_tables(cli, matrix) -> None:
    """Render a comparison matrix as one Rich table per metric (terminal view).

    Carries the same accuracy / latency / cost cells as ``matrix.to_markdown``
    — including ``ERROR`` for a failed model, ``unpriced`` for a model with no
    published price, and ``—`` for a missing cell — so the terminal and the
    piped Markdown say the same thing.
    """
    if not matrix.scores:
        cli.print("No scores recorded.")
        return
    suites = sorted({s.suite_name for s in matrix.scores})
    models = sorted({s.model_name for s in matrix.scores})
    lookup = {(s.model_name, s.suite_name): s for s in matrix.scores}

    def _cell(sc, kind):
        if sc is None:
            return "—"
        if sc.error:
            return "ERROR"
        if kind == "accuracy":
            return f"{sc.accuracy:.1%}"
        if kind == "latency":
            return f"{sc.avg_latency:.3f}"
        if sc.avg_cost_usd is not None:
            return f"${sc.avg_cost_usd:.6f}"
        return "unpriced"

    for kind, title in (
        ("accuracy", "Accuracy"),
        ("latency", "Avg Latency (s)"),
        ("cost", "Avg Cost (USD/run)"),
    ):
        rows = [
            [m] + [_cell(lookup.get((m, su)), kind) for su in suites]
            for m in models
        ]
        render_table(
            columns=["Model", *suites],
            rows=rows,
            console=cli.console,
            title=title,
            justify=["left", *(["right"] * len(suites))],
            styles=["cyan", *([None] * len(suites))],
        )
    if matrix.recommendations:
        cli.print(f"\nRecommendations (optimized for {matrix.optimize}):")
        for su, model in sorted(matrix.recommendations.items()):
            cli.print(f"  {su}: {model}")


def _handle_compare_command(args, cli) -> int:
    """Handle 'effgen compare' subcommand."""
    from effgen.eval import ModelComparison
    from effgen.eval.evaluator import ScoringMode

    provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
    if prov_err:
        cli.print_error(prov_err)
        return 2

    model_names = [m.strip() for m in args.models.split(',')]
    suite_name = args.suite
    scoring = ScoringMode(args.scoring)
    threshold = args.threshold
    temperature = getattr(args, 'temperature', None)
    preset_name = getattr(args, 'preset', None)
    difficulty = getattr(args, 'difficulty', None)
    max_cases = getattr(args, 'max_cases', None)
    optimize = getattr(args, 'optimize', 'accuracy')
    json_mode = getattr(args, 'output_json', False)
    if json_mode:
        cli._human_to_stderr = True

    # Unknown suite is a user error, not a crash — report cleanly (no traceback)
    # and exit 2 with the list of valid suites. A bad data file is reported the
    # same way.
    try:
        suite = _resolve_eval_suite(suite_name, difficulty=difficulty, max_cases=max_cases)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        from effgen.eval import list_suites
        cli.print(
            f"Could not load suite '{suite_name}' ({exc}). "
            f"Use a built-in suite ({', '.join(list_suites())}) "
            "or a path to a .jsonl/.json file of test cases."
        )
        return 2

    agents: dict = {}
    try:
        # Load all models and create agents
        from effgen.models import load_model

        for model_name in model_names:
            cli.print(f"Loading model {model_name}...")
            # --provider is a fallback for a bare id; a model_name that
            # already carries its own "provider:"/"engine:" prefix keeps it.
            model_provider = provider if (provider and ":" not in model_name) else None
            try:
                model = load_model(model_name, provider=model_provider)
                if preset_name:
                    from effgen.presets import create_agent
                    agent = create_agent(preset_name, model, temperature=temperature)
                else:
                    from effgen.core.agent import Agent, AgentConfig
                    config_kwargs: dict = {
                        "name": f"compare-{model_name}", "model": model, "max_iterations": 10,
                    }
                    if temperature is not None:
                        config_kwargs["temperature"] = temperature
                    config = AgentConfig(**config_kwargs)
                    agent = Agent(config)
                agents[model_name] = agent
            except Exception as e:
                cli.print(f"  Warning: Failed to load {model_name}: {e}")

        if not agents:
            cli.print(
                "Error: No models loaded successfully. Check the model ids "
                "(`effgen models list`) and provider keys (`effgen doctor`)."
            )
            return 1

        cli.print(f"\nComparing {len(agents)} models on {suite_name} ({len(suite)} cases)...")
        comparison = ModelComparison(scoring=scoring, pass_threshold=threshold)
        matrix = comparison.run(agents, [suite], optimize=optimize)

        # Display: rich per-metric tables on a terminal, copy-pasteable Markdown
        # (the same content) when piped or redirected. Under --json the Markdown
        # goes to stderr (via cli.print) so stdout carries only the JSON below.
        if not json_mode and console_is_interactive(cli.console):
            _render_comparison_tables(cli, matrix)
        else:
            cli.print(matrix.to_markdown())

        # Write output
        if args.output:
            output_path = args.output
            if output_path.endswith('.md'):
                Path(output_path).write_text(matrix.to_markdown(), encoding="utf-8")
            else:
                Path(output_path).write_text(matrix.to_json(), encoding="utf-8")
            cli.print(f"\nResults written to {output_path}")

        # Emit the comparison matrix as JSON to stdout for piping/CI gating.
        if json_mode:
            print(matrix.to_json())

        return 0

    except Exception as e:
        cli.print(f"Comparison failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Release agent resources so the run leaves no GC-close warnings.
        for agent in agents.values():
            close = getattr(agent, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass


def _checkpoint_run_kwargs(args) -> dict:
    """Extract checkpoint run() kwargs from CLI args."""
    out: dict = {}
    if getattr(args, 'checkpoint_dir', None):
        out['checkpoint_dir'] = args.checkpoint_dir
    if getattr(args, 'checkpoint_interval', 0):
        out['checkpoint_interval'] = args.checkpoint_interval
    return out


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
        cli.print(response.output if hasattr(response, 'output') else str(response))
        return 0 if getattr(response, 'success', True) else 1
    finally:
        # Release the agent so resume never emits the "garbage-collected
        # without calling close()" warning (matches the run path).
        try:
            agent.close()
        except Exception as e:  # noqa: BLE001
            logging.debug(f"Agent close after resume failed: {e}")


# Cheapest well-known model per cloud provider, used to suggest a first model in
# the quickstart. Order = auto-pick preference (fast/free first).
_QUICKSTART_CLOUD_MODELS: tuple[tuple[str, str], ...] = (
    ("groq", "llama-3.1-8b-instant"),
    ("openai", "gpt-5-nano"),
    ("gemini", "gemini-3.1-flash-lite"),
    ("cerebras", "gpt-oss-120b"),
)
_QUICKSTART_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _quickstart_suggest_model() -> tuple[str, str | None, str]:
    """Pick a sensible first model: a keyed cloud model if any, else local.

    Returns ``(model_id, provider, reason)``.
    """
    try:
        from effgen.models.auth import check_keys
        keys = check_keys()
    except Exception:  # noqa: BLE001
        keys = {}
    for provider, model_id in _QUICKSTART_CLOUD_MODELS:
        info = keys.get(provider)
        if info and info.get("available"):
            return model_id, provider, f"{provider} key detected"
    return _QUICKSTART_LOCAL_MODEL, None, "no cloud key found — using a small local model"


def _preflight_model_hint(cli: "CLIInterface", model_id: str, provider: str | None) -> None:
    """Surface a clean "did you mean" for an unknown model id, once, up front.

    When the user passes an explicit ``-m`` id that isn't in the local catalog,
    a high-confidence typo (e.g. ``gpt-5-nanoo``) otherwise only reveals itself
    mid-run as a provider 404 wall. This checks the local catalog first and, if
    the id is unknown but has near matches, prints a single tidy suggestion line.
    It never blocks — the catalog can be stale, so an unknown-but-real new id is
    allowed through; we only inform.
    """
    try:
        from effgen.models import _catalog

        _bare = model_id.split(":", 1)[-1]
        # Local / HF-hub ids ("org/model") are resolved by download, not via the
        # cloud catalog, so a catalog miss there is meaningless — never warn on a
        # legitimate local model (e.g. meta-llama/Llama-3.2-3B-Instruct). The hint
        # targets slash-free cloud chat ids, where a typo is the likely cause.
        if "/" in _bare:
            return
        if _catalog.lookup(model_id, provider) is not None:
            return
        alts = _catalog.nearest_alternatives(model_id, provider, n=3)
        if not alts:
            return
        names = ", ".join(r.id for r in alts)
        cli.print(
            f"Note: '{_bare}' isn't in the local catalog. Did you mean: {names}? "
            "Proceeding anyway — run 'effgen models refresh' if it's a new id."
        )
    except Exception:  # noqa: BLE001 - a hint must never break the run
        pass


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


def _handle_sessions_command(args, cli) -> int:
    """Handle 'effgen sessions' subcommands."""
    from effgen.core.session import SessionManager
    from effgen.errors import CorruptStateError
    mgr = SessionManager()
    cmd = getattr(args, 'session_command', None)
    if cmd == 'list':
        sessions = mgr.list_sessions()
        if getattr(args, 'output_json', False):
            import json as _json
            print(_json.dumps({
                "sessions": sessions,
                "sessions_dir": str(mgr.sessions_dir),
            }, indent=2, default=str, ensure_ascii=False))
            return 0
        if not sessions:
            cli.print(
                "No sessions yet. Start one with: effgen chat  (or effgen run \"...\" "
                "creates a session you can resume)."
            )
            return 0
        render_table(
            columns=["Session", "Messages", "Updated"],
            rows=[
                [s['session_id'], s['messages'], s.get('updated_at') or "—"]
                for s in sessions
            ],
            console=cli.console,
            justify=["left", "right", "left"],
            styles=["cyan", "yellow", None],
            caption=f"Stored in: {mgr.sessions_dir}",
        )
        return 0
    if cmd == 'delete':
        ok = mgr.delete(args.session_id)
        cli.print("Deleted." if ok else f"Session not found: {args.session_id}")
        return 0 if ok else 1
    if cmd == 'export':
        try:
            cli.print(mgr.export(args.session_id, format=args.format))
        except FileNotFoundError:
            cli.print(f"Session not found: {args.session_id}")
            return 1
        except CorruptStateError as e:
            cli.print(f"Error: {e}")
            return 2
        return 0
    if cmd == 'cleanup':
        n = mgr.cleanup(older_than_days=args.days)
        cli.print(f"Removed {n} old session(s).")
        return 0
    return _print_group_help(args)


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
        cli.print("  " + _json.dumps(p.input_schema, indent=2).replace("\n", "\n  "))
        cli.print("\n[bold]Fixture:[/bold]" if RICH_AVAILABLE else "\nFixture:")
        cli.print("  " + _json.dumps(p.fixture, indent=2).replace("\n", "\n  "))
        try:
            rendered = p.render_fixture()
            cli.print(
                "\n[bold]Rendered (fixture):[/bold]"
                if RICH_AVAILABLE
                else "\nRendered (fixture):"
            )
            cli.print(rendered)
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




def _print_group_help(args) -> int:
    """Print a command group's help when it is invoked with no subcommand.

    A bare group command (``effgen tools``, ``effgen models``, ...) has nothing
    to do on its own, so it shows the group's usage and subcommand list instead
    of an error, matching what ``--help`` prints.
    """
    parser = getattr(args, "_group_parser", None)
    if parser is not None:
        parser.print_help()
    return 0


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


def main():
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
        elif args.command == 'cost':
            exit_code = _handle_cost_command(args, cli)
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


def agent_main():
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


def web_agent_main():
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
