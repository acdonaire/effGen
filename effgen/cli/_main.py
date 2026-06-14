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
from datetime import datetime
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

# Tips, first-run welcome, "did you mean?" and teaching-error helpers.
from effgen.cli import onboarding as _onboarding

# Live status / progress presentation (TTY-aware; degrades to plain text).
from effgen.cli import progress as _progress

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


class CLIInterface:
    """Main CLI interface for effGen."""

    def __init__(self):
        """Initialize CLI interface."""
        self.console = Console() if RICH_AVAILABLE else None
        self.config_loader = ConfigLoader()
        self.tool_registry = get_tool_registry()

    def _animate(self, args) -> bool:
        """Whether to show live animation for this invocation (TTY-aware, opt-out)."""
        return _progress.animation_enabled(
            quiet=getattr(args, 'quiet', False),
            no_animation=getattr(args, 'no_animation', False),
        )

    def print(self, *args, **kwargs):
        """Print with rich formatting if available."""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def print_header(self, text: str):
        """Print a header."""
        if self.console:
            self.console.print(f"\n[bold cyan]{text}[/bold cyan]")
        else:
            print(f"\n=== {text} ===")

    def print_success(self, text: str):
        """Print success message."""
        if self.console:
            self.console.print(f"[green]✓[/green] {text}")
        else:
            print(f"✓ {text}")

    def print_error(self, text: str):
        """Print error message."""
        if self.console:
            self.console.print(f"[red]✗[/red] {text}")
        else:
            print(f"✗ {text}")

    def print_warning(self, text: str):
        """Print warning message."""
        if self.console:
            self.console.print(f"[yellow]⚠[/yellow] {text}")
        else:
            print(f"⚠ {text}")

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
            _is_anthropic_model = (
                model_id.startswith("claude") or "anthropic" in model_id.lower()
            )
            _is_openai_model = (
                model_id.startswith("gpt-") or model_id.startswith("o1") or
                model_id.startswith("o3") or model_id.startswith("o4") or
                "openai" in model_id.lower()
            )
            _filtered_tools: list = []
            for _t in selected_tools:
                _tname = getattr(getattr(_t, "name", None), "__str__", lambda: "")() or str(getattr(_t, "name", ""))
                _cls_name = type(_t).__name__
                _is_anthropic_native = "AnthropicNative" in _cls_name or "anthropic" in _tname.lower()
                _is_openai_native = "OpenAINative" in _cls_name
                # Skip Anthropic native tools unless model is Anthropic
                if _is_anthropic_native and not _is_anthropic_model:
                    self.print_warning(f"Skipping Anthropic native tool '{_tname}' (requires claude model)")
                    continue
                # Skip OpenAI native tools (web_search_preview etc.) unless model is OpenAI
                if _is_openai_native and not _is_openai_model:
                    self.print_warning(f"Skipping OpenAI native tool '{_tname}' (requires gpt/o1/o3 model)")
                    continue
                _filtered_tools.append(_t)
            if len(_filtered_tools) < len(selected_tools):
                skipped = len(selected_tools) - len(_filtered_tools)
                self.print_warning(
                    f"Filtered out {skipped} provider-specific tool(s) incompatible with '{model_id}'"
                )
            selected_tools = _filtered_tools

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
        # Check if we need to launch interactive wizard
        if args.task is None:
            return self.interactive_wizard(args)

        # Validate an explicit --provider before doing any work, so a typo
        # (e.g. "grok") fails fast with a suggestion instead of falling through
        # to a multi-gigabyte local model download.
        provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
        if prov_err:
            self.print_error(prov_err)
            return 1

        self.print_header(f"effGen v{__version__} - Running Task")

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
                else:
                    self.print_error(f"Configuration file not found: {config_path}")
                    return 1

            # Use preset if specified
            if getattr(args, 'preset', None):
                from effgen.presets import create_agent as _create_preset_agent
                model_id = args.model or "Qwen/Qwen2.5-3B-Instruct"
                self.print(f"Using preset: {args.preset}")
                _preset_overrides = {"provider": provider} if provider else {}
                agent = _create_preset_agent(
                    args.preset,
                    model_id,
                    agent_name=args.name,
                    system_prompt=args.system_prompt or config.get("system_prompt"),
                    max_iterations=args.max_iterations,
                    temperature=args.temperature,
                    enable_streaming=args.stream,
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
                    model=args.model or "Qwen/Qwen2.5-3B-Instruct",
                    provider=provider,
                    tools=tools,
                    system_prompt=args.system_prompt or config.get("system_prompt",
                        "You are a helpful AI assistant."),
                    temperature=args.temperature or config.get("temperature", 0.7),
                    max_iterations=args.max_iterations or config.get("max_iterations", 10),
                    enable_sub_agents=not args.no_sub_agents,
                    enable_streaming=args.stream
                )

                # Create agent
                self.print(f"\nInitializing agent: {agent_config.name}")
                self.print(f"Model: {agent_config.model}")
                self.print(f"Tools: {len(tools)} available")
                self.print(f"Sub-agents: {'enabled' if agent_config.enable_sub_agents else 'disabled'}")

                agent = Agent(agent_config, session_id=getattr(args, 'session_id', None))

            # Determine execution mode
            mode = AgentMode.AUTO
            if args.mode:
                if args.mode == "single":
                    mode = AgentMode.SINGLE
                elif args.mode == "sub_agents":
                    mode = AgentMode.SUB_AGENTS

            # Run task
            self.print(f"\n[bold]Task:[/bold] {args.task}" if self.console else f"\nTask: {args.task}")
            self.print()

            exit_code = 0
            animate = self._animate(args)
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
                try:
                    if animate:
                        reasoning = _progress.is_reasoning_agent(agent)
                        with _progress.LiveStatus(
                            self.console,
                            model_label=model_label,
                            reasoning=reasoning,
                            tracker=agent.execution_tracker,
                        ):
                            response = agent.run(args.task, mode=mode, **_checkpoint_run_kwargs(args))
                    else:
                        if not quiet:
                            self.print("Thinking...")
                        response = agent.run(args.task, mode=mode, **_checkpoint_run_kwargs(args))
                except KeyboardInterrupt:
                    self._handle_interrupt(agent)
                    return 130

                # Surface failure in the process exit code.
                if not response.success:
                    exit_code = 1

                # Display response
                self.print_header("Response")

                if self.console:
                    # Rich markdown formatting
                    self.console.print(Panel(
                        Markdown(response.output),
                        title="Agent Response",
                        border_style="green" if response.success else "red"
                    ))
                else:
                    print(response.output)

                # Frozen one-glance summary: ✓ Done in 3.2s · 2 tools · 1,204 tokens · $…
                if not quiet:
                    _progress.print_summary(self, response)

                # Display explain trace (tool reasoning)
                if getattr(args, 'explain', False) and response.execution_trace:
                    self.print_header("Execution Trace (Explain Mode)")
                    for i, step in enumerate(response.execution_trace, 1):
                        thought = step.get("thought", step.get("input", ""))
                        action = step.get("action", step.get("tool", ""))
                        observation = step.get("observation", step.get("output", ""))
                        if self.console:
                            self.console.print(f"[bold cyan]Step {i}[/bold cyan]")
                            if thought:
                                self.console.print(f"  [yellow]Thought:[/yellow] {str(thought)[:300]}")
                            if action:
                                self.console.print(f"  [green]Action:[/green] {action}")
                            if observation:
                                self.console.print(f"  [blue]Result:[/blue] {str(observation)[:200]}")
                        else:
                            print(f"Step {i}")
                            if thought:
                                print(f"  Thought: {str(thought)[:300]}")
                            if action:
                                print(f"  Action: {action}")
                            if observation:
                                print(f"  Result: {str(observation)[:200]}")

                # Display execution statistics
                if getattr(args, 'verbose', False) or getattr(args, 'explain', False):
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
                    with open(output_path, 'w') as f:
                        json.dump(response.to_dict(), f, indent=2)
                    self.print_success(f"Response saved to {output_path}")

            return exit_code

        except Exception as e:
            self.print_error(f"Error running agent: {e}")
            if getattr(args, 'verbose', False):
                import traceback
                traceback.print_exc()
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
        """
        Interactive chat mode.

        Args:
            args: Parsed command-line arguments
        """
        self.print_header(f"effGen v{__version__} - Chat Mode")
        self.print("Type 'exit' or 'quit' to end the conversation")
        self.print("Type 'clear' to clear conversation history")
        self.print("Type 'help' for available commands\n")

        try:
            # Initialize agent (similar to run_agent)
            tools = []
            self.tool_registry.discover_builtin_tools()
            tool_names = self.tool_registry.list_tools()[:5]
            for name in tool_names:
                try:
                    tool = asyncio.run(self.tool_registry.get_tool(name))
                    tools.append(tool)
                except Exception:
                    pass

            agent_config = AgentConfig(
                name="chat-agent",
                model=args.model or "Qwen/Qwen2.5-3B-Instruct",
                tools=tools,
                temperature=args.temperature or 0.7,
                enable_sub_agents=not args.no_sub_agents,
                enable_streaming=True
            )

            agent = Agent(agent_config)
            conversation_history = []

            while True:
                try:
                    # Get user input
                    if self.console:
                        user_input = self.console.input("\n[bold cyan]You:[/bold cyan] ")
                    else:
                        user_input = input("\nYou: ")

                    if not user_input.strip():
                        continue

                    # Handle commands
                    if user_input.lower() in ['exit', 'quit']:
                        self.print("\nGoodbye!")
                        break
                    elif user_input.lower() == 'clear':
                        agent.reset_memory()
                        conversation_history = []
                        self.print_success("Conversation history cleared")
                        continue
                    elif user_input.lower() == 'help':
                        self._print_chat_help()
                        continue
                    elif user_input.lower() == 'save':
                        self._save_conversation(conversation_history)
                        continue
                    elif user_input.lower() == 'history':
                        self._list_conversations()
                        continue
                    elif user_input.lower() == 'load':
                        loaded = self._load_conversation()
                        if loaded:
                            conversation_history = loaded
                        continue

                    # Add to history
                    conversation_history.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Get agent response with thinking spinner
                    response_text = ""
                    import time as _time
                    _turn_start = _time.monotonic()

                    if self.console:
                        # Show thinking spinner until first token arrives
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=self.console,
                            transient=True  # Remove spinner when done
                        ) as progress:
                            progress.add_task("Thinking...", total=None)

                            # Get iterator and wait for first token
                            token_iter = iter(agent.stream(user_input))
                            try:
                                first = next(token_iter)
                                response_text += first
                            except StopIteration:
                                first = None

                        # Now print the response
                        self.console.print("\n[bold green]Agent:[/bold green] ", end="")
                        if first:
                            print(first, end='', flush=True)

                        # Continue with remaining tokens
                        for token in token_iter:
                            print(token, end='', flush=True)
                            response_text += token
                    else:
                        print("\nThinking...", end="", flush=True)
                        token_iter = iter(agent.stream(user_input))
                        try:
                            first = next(token_iter)
                            response_text += first
                        except StopIteration:
                            first = None

                        # Clear "Thinking..." and print response
                        print("\r" + " " * 20 + "\r", end="")  # Clear line
                        print("Agent: ", end="", flush=True)
                        if first:
                            print(first, end='', flush=True)

                        for token in token_iter:
                            print(token, end='', flush=True)
                            response_text += token

                    print()  # New line

                    # Subtle per-turn footer (elapsed time) for a live feel.
                    _turn_elapsed = _time.monotonic() - _turn_start
                    if not getattr(args, 'quiet', False):
                        _footer = f"· {_turn_elapsed:.1f}s"
                        if self.console:
                            self.console.print(f"[dim]{_footer}[/dim]")
                        else:
                            print(_footer)

                    # Add to history
                    conversation_history.append({
                        "role": "agent",
                        "content": response_text,
                        "timestamp": datetime.now().isoformat()
                    })

                except KeyboardInterrupt:
                    self.print("\n\nInterrupted. Type 'exit' to quit.")
                    continue
                except Exception as e:
                    self.print_error(f"Error: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()

            return 0

        except Exception as e:
            self.print_error(f"Error in chat mode: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _print_chat_help(self):
        """Print chat mode help."""
        help_text = """
        [bold]Available Commands:[/bold]
        - exit, quit: Exit chat mode
        - clear: Clear conversation history
        - save: Save conversation to file
        - load: Load a previous conversation
        - history: List saved conversations
        - help: Show this help message
        """ if self.console else """
        Available Commands:
        - exit, quit: Exit chat mode
        - clear: Clear conversation history
        - save: Save conversation to file
        - load: Load a previous conversation
        - history: List saved conversations
        - help: Show this help message
        """
        self.print(help_text)

    @staticmethod
    def _history_dir() -> Path:
        """Return the chat history directory, creating it if needed."""
        d = Path.home() / ".effgen" / "history"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_conversation(self, history: list[dict]):
        """Save conversation history to ~/.effgen/history/."""
        hist_dir = self._history_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = hist_dir / f"conversation_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)

        self.print_success(f"Conversation saved to {filename}")

    def _list_conversations(self):
        """List saved conversation files."""
        hist_dir = self._history_dir()
        files = sorted(hist_dir.glob("conversation_*.json"), reverse=True)
        if not files:
            self.print("No saved conversations found.")
            return
        self.print("Saved conversations:")
        for i, f in enumerate(files[:20], 1):
            size = f.stat().st_size
            self.print(f"  {i}. {f.name}  ({size} bytes)")

    def _load_conversation(self) -> list[dict] | None:
        """Load a previous conversation by index."""
        hist_dir = self._history_dir()
        files = sorted(hist_dir.glob("conversation_*.json"), reverse=True)
        if not files:
            self.print("No saved conversations found.")
            return None
        self._list_conversations()
        try:
            choice = input("Enter number to load (or 'cancel'): ").strip()
            if choice.lower() == "cancel":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                with open(files[idx]) as f:
                    history = json.load(f)
                self.print_success(f"Loaded {files[idx].name} ({len(history)} messages)")
                for msg in history:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")[:100]
                    self.print(f"  [{role}] {content}...")
                return history
        except (ValueError, IndexError):
            self.print_error("Invalid selection.")
        return None

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

            self.print(f"Starting server on {host}:{port}")
            self.print(f"  OpenAI-compatible API : http://{host}:{port}/v1")
            self.print(f"  Interactive docs      : http://{host}:{port}/docs")
            self.print(f"  Dashboard             : http://{host}:{port}/dashboard")
            self.print()

            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info" if verbose else "warning",
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
            from effgen.models._cost import _BUDGET_CONFIG_PATH
            budget_path = _BUDGET_CONFIG_PATH
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
                        theme="monokai",
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
        for p in metadata.parameters:
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
            self.console.print(Syntax(json.dumps(schema, indent=2), "json", theme="monokai"))
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
                self.console.print(Syntax(json.dumps(schema, indent=2), "json", theme="monokai"))
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
        else:
            self.print_error(f"Unknown models command: {args.model_command}")
            return 1

        return 0

    @staticmethod
    def _price_cell(rec) -> str:
        """Format a model's input/output price per 1M tokens for a table cell."""
        if rec.free_tier and rec.price_in_per_1m in (None, 0) and rec.price_out_per_1m in (None, 0):
            return "free"
        pin, pout = rec.price_in_per_1m, rec.price_out_per_1m
        if pin is None and pout is None:
            return "—"
        fmt = lambda v: ("?" if v is None else (f"${v:g}" if v else "$0"))  # noqa: E731
        return f"{fmt(pin)}/{fmt(pout)}"

    def _local_cached_models(self) -> list[dict]:
        """Models actually downloaded in the local HuggingFace cache (on disk)."""
        out: list[dict] = []
        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir()
            for repo in sorted(info.repos, key=lambda r: r.repo_id):
                if repo.repo_type == "model":
                    out.append({
                        "id": repo.repo_id,
                        "size_gb": repo.size_on_disk / (1024 ** 3),
                        "path": str(repo.repo_path),
                    })
        except Exception as e:  # noqa: BLE001 - cache scan is best-effort
            logging.debug(f"HF cache scan failed: {e}")
        return out

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
                            "id": r.id, "context_window": r.context_window,
                            "max_output": r.max_output,
                            "price_in_per_1m": r.price_in_per_1m,
                            "price_out_per_1m": r.price_out_per_1m,
                            "supports_tools": r.supports_tools,
                            "supports_vision": r.supports_vision,
                            "free_tier": r.free_tier, "deprecated": r.deprecated,
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
            if self.console:
                title = (f"{provider_filter} — {len(recs)} models "
                         f"(auth: {auth}, verified: {verified})")
                table = Table(title=title)
                table.add_column("Model ID", style="cyan", overflow="fold")
                table.add_column("Context", style="white", justify="right")
                table.add_column("Max Out", style="white", justify="right")
                table.add_column("$/1M in/out", style="green")
                table.add_column("Tools", justify="center")
                table.add_column("Vision", justify="center")
                table.add_column("Free", justify="center")
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
                for r in recs:
                    print(f"- {r.id}  ctx={r.context_window}  {self._price_cell(r)}")
            return 0

        # Overview: per-provider summary + filtered flat table when filtering.
        if free_only or tools_only:
            label = "free-tier" if free_only else "tool-capable"
            if tools_only and free_only:
                label = "free + tool-capable"
            if self.console:
                table = Table(title=f"{label.capitalize()} models (all providers)")
                table.add_column("Model ID", style="cyan", overflow="fold")
                table.add_column("Provider", style="magenta")
                table.add_column("Context", justify="right")
                table.add_column("$/1M in/out", style="green")
                table.add_column("Tools", justify="center")
                table.add_column("Free", justify="center")
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
                for prov in providers:
                    for r in _records(prov):
                        print(f"- {prov}:{r.id}")
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
            if self.console:
                ltable = Table(title=f"Local HuggingFace cache ({len(local)} downloaded)")
                ltable.add_column("Model", style="cyan", overflow="fold")
                ltable.add_column("Size", justify="right", style="white")
                for m in local:
                    ltable.add_row(m["id"], f"{m['size_gb']:.1f} GB")
                self.console.print(ltable)
            else:
                print("\nLocal HuggingFace cache:")
                for m in local:
                    print(f"  {m['id']}  ({m['size_gb']:.1f} GB)")
        return 0

    def _models_info(self, args):
        """Show detailed information for one model from the registry."""
        if not args.name:
            self.print_error("Model name required")
            return 1

        from effgen.models import _catalog, _refresh

        rec = _catalog.lookup(args.name)
        if rec is None:
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
            self.console.print(f"\n[dim]Use: [cyan]effgen run --provider {rec.provider} "
                               f"-m {rec.id} \"...\"[/cyan][/dim]")
        else:
            for k, v in rows.items():
                print(f"  {k}: {v}")
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
        self.print_header("Model & GPU Status")

        # GPU memory info
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if self.console:
                    from rich.table import Table
                    gpu_table = Table(title="GPU Status")
                    gpu_table.add_column("GPU", style="cyan")
                    gpu_table.add_column("Name", style="white")
                    gpu_table.add_column("Total", style="white")
                    gpu_table.add_column("Used", style="yellow")
                    gpu_table.add_column("Free", style="green")

                    for i in range(num_gpus):
                        props = torch.cuda.get_device_properties(i)
                        total_gb = props.total_memory / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        free_gb = total_gb - reserved
                        gpu_table.add_row(
                            str(i), props.name,
                            f"{total_gb:.1f} GB",
                            f"{reserved:.1f} GB",
                            f"{free_gb:.1f} GB",
                        )
                    self.console.print(gpu_table)
                else:
                    for i in range(num_gpus):
                        props = torch.cuda.get_device_properties(i)
                        total_gb = props.total_memory / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        print(f"GPU {i}: {props.name} — "
                              f"{total_gb:.1f} GB total, "
                              f"{reserved:.1f} GB used, "
                              f"{total_gb - reserved:.1f} GB free")
            else:
                self.print_warning("CUDA not available")
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
    parser.add_argument('--completion', choices=['bash', 'zsh', 'fish'],
                        help='Print shell completion script and exit')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run an agent with a task')
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
    run_parser.add_argument('--system-prompt', help='System prompt')
    run_parser.add_argument('--temperature', type=float, help='Temperature')
    run_parser.add_argument('--max-iterations', type=int, help='Max iterations')
    run_parser.add_argument('--mode', choices=['auto', 'single', 'sub_agents'], help='Execution mode')
    run_parser.add_argument('--no-sub-agents', action='store_true', help='Disable sub-agents')
    run_parser.add_argument('--stream', action='store_true', help='Stream output')
    run_parser.add_argument('-o', '--output', help='Output file for response')
    run_parser.add_argument('--preset', choices=_preset_choices,
                            help='Use a preset agent configuration')
    run_parser.add_argument('--explain', action='store_true',
                            help='Show why the agent chose each tool')
    run_parser.add_argument('--checkpoint-dir', help='Directory to write agent checkpoints')
    run_parser.add_argument('--checkpoint-interval', type=int, default=0,
                            help='Checkpoint every N iterations (requires --checkpoint-dir)')
    run_parser.add_argument('--session-id', help='Persistent session id')

    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume an agent run from a checkpoint')
    resume_parser.add_argument('--checkpoint', required=True,
                               help='Checkpoint id, JSON path, or directory (uses latest)')
    resume_parser.add_argument('-m', '--model', help='Model to use')
    resume_parser.add_argument('--preset', choices=_preset_choices)

    # Sessions commands
    sessions_parser = subparsers.add_parser('sessions', help='Manage persistent sessions')
    sessions_subparsers = sessions_parser.add_subparsers(dest='session_command', help='Sessions command')
    sessions_subparsers.add_parser('list', help='List sessions')
    sd = sessions_subparsers.add_parser('delete', help='Delete a session')
    sd.add_argument('session_id', help='Session id')
    se = sessions_subparsers.add_parser('export', help='Export a session')
    se.add_argument('session_id', help='Session id')
    se.add_argument('--format', choices=['json', 'text'], default='json')
    sc = sessions_subparsers.add_parser('cleanup', help='Delete sessions older than N days')
    sc.add_argument('--days', type=int, default=30)

    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument('-m', '--model', help='Model to use')
    chat_parser.add_argument(
        '--provider',
        help='Provider for a bare model id (e.g. openai, groq, cerebras, gemini). '
             'Equivalent to the "provider:model" prefix.',
    )
    chat_parser.add_argument('--temperature', type=float, help='Temperature')
    chat_parser.add_argument('--no-sub-agents', action='store_true', help='Disable sub-agents')
    chat_parser.add_argument('-v', '--verbose', action='store_true', default=argparse.SUPPRESS,
                             help='Verbose output (show DEBUG/INFO logs)')
    chat_parser.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                             help='Quiet output (errors only)')
    chat_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                             help='Disable live spinners/progress animation')

    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument(
        '--host', default='127.0.0.1',
        help='Host to bind to (default 127.0.0.1, loopback-only). '
             'Pass --host 0.0.0.0 to expose on all interfaces (set EFFGEN_API_KEY first).',
    )
    serve_parser.add_argument('-p', '--port', type=int, default=8000, help='Port to bind to')

    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
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
    models_subparsers = models_parser.add_subparsers(dest='model_command', help='Models command')

    models_list = models_subparsers.add_parser('list', help='List models')
    models_list.add_argument('--provider', help='Show only this provider\'s models (full detail)')
    models_list.add_argument('--free', action='store_true', help='Show only free-tier models')
    models_list.add_argument('--tools', action='store_true', help='Show only tool-capable models')
    models_list.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')

    models_info = models_subparsers.add_parser('info', help='Show model information')
    models_info.add_argument('name', help='Model name (e.g. gpt-5-nano or openai:gpt-5-nano)')
    models_info.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')

    models_load = models_subparsers.add_parser('load', help='Pre-load a model into memory')
    models_load.add_argument('name', help='Model name (e.g. Qwen/Qwen2.5-1.5B-Instruct)')
    models_load.add_argument('-e', '--engine', help='Engine (vllm, transformers)', default=None)

    models_unload = models_subparsers.add_parser('unload', help='Unload a model from memory')
    models_unload.add_argument('name', help='Model name')

    models_subparsers.add_parser('status', help='Show loaded models and GPU memory status')

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
    subparsers.add_parser('presets', help='List available agent presets')

    # Quickstart / tutorial — a short guided first run.
    for _qs_name in ('quickstart', 'tutorial'):
        qs_parser = subparsers.add_parser(
            _qs_name,
            help='Guided first run: pick a model, run an agent, see the trace and cost',
        )
        qs_parser.add_argument('-m', '--model', help='Model to use (skips the model prompt)')
        qs_parser.add_argument('--provider', help='Provider for a bare model id')
        qs_parser.add_argument('--task', help='Task to run (defaults to a sample task)')
        qs_parser.add_argument('-y', '--yes', action='store_true',
                               help='Run non-interactively with sensible defaults')

    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Run a DAG-based workflow')
    workflow_subparsers = workflow_parser.add_subparsers(dest='workflow_command', help='Workflow command')

    workflow_run = workflow_subparsers.add_parser('run', help='Run a workflow from YAML file')
    workflow_run.add_argument('file', help='Path to workflow YAML file')
    workflow_run.add_argument('-m', '--model', help='Default model for all agents')
    workflow_run.add_argument('--input', action='append', nargs=2, metavar=('NODE', 'TASK'),
                              help='Input for a specific node (can be repeated)')

    workflow_validate = workflow_subparsers.add_parser('validate', help='Validate a workflow YAML file')
    workflow_validate.add_argument('file', help='Path to workflow YAML file')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Run batch queries from a file')
    batch_parser.add_argument('--input', required=True, help='Input file (JSONL, CSV, JSON, or plain text)')
    batch_parser.add_argument('--output', help='Output file (JSONL, CSV, or JSON)')
    batch_parser.add_argument('--concurrency', type=int, default=5, help='Max concurrent queries (default: 5)')
    batch_parser.add_argument('--batch-size', type=int, default=0, help='Batch size (0 = all at once)')
    batch_parser.add_argument('--timeout', type=float, default=120.0, help='Timeout per query in seconds')
    batch_parser.add_argument('--retries', type=int, default=1, help='Retries for failed queries')
    batch_parser.add_argument('-m', '--model', help='Model to use')
    batch_parser.add_argument('--preset', choices=_preset_choices,
                              help='Use a preset agent configuration')
    batch_parser.add_argument('--query-field', default='query', help='Field name for queries in JSONL/CSV (default: query)')
    batch_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                              help='Disable the live progress bar (plain output)')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate an agent against a test suite')
    eval_parser.add_argument('--suite', required=True,
                              help='Test suite name (math, tool_use, reasoning, safety, conversation)')
    eval_parser.add_argument('-m', '--model', help='Model to use')
    eval_parser.add_argument('--preset', choices=_preset_choices,
                              help='Use a preset agent configuration')
    eval_parser.add_argument('--scoring', choices=['exact_match', 'contains', 'regex', 'semantic_similarity', 'llm_judge'],
                              default='contains', help='Scoring mode (default: contains)')
    eval_parser.add_argument('--threshold', type=float, default=0.5,
                              help='Pass threshold (default: 0.5)')
    eval_parser.add_argument('--save-baseline', action='store_true',
                              help='Save results as regression baseline')
    eval_parser.add_argument('--compare-baseline', action='store_true',
                              help='Compare results against stored baseline')
    eval_parser.add_argument('-o', '--output', help='Output file for results (JSON)')
    eval_parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                              help='Filter test cases by difficulty')
    eval_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                              help='Disable the live progress bar (plain output)')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models on a test suite')
    compare_parser.add_argument('--models', required=True,
                                 help='Comma-separated model names')
    compare_parser.add_argument('--suite', required=True,
                                 help='Test suite name')
    compare_parser.add_argument('--scoring', choices=['exact_match', 'contains', 'regex', 'semantic_similarity', 'llm_judge'],
                                 default='contains', help='Scoring mode (default: contains)')
    compare_parser.add_argument('--threshold', type=float, default=0.5,
                                 help='Pass threshold (default: 0.5)')
    compare_parser.add_argument('-o', '--output', help='Output file for results (JSON or Markdown)')
    compare_parser.add_argument('--preset', choices=_preset_choices,
                                 help='Use a preset agent configuration')

    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Run an agent in interactive debug mode')
    debug_parser.add_argument('task', help='Task to execute')
    debug_parser.add_argument('-m', '--model', help='Model to use')
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

    prompts_show = prompts_subparsers.add_parser('show', help='Show prompt details')
    prompts_show.add_argument('name', help='Prompt name')

    prompts_eval = prompts_subparsers.add_parser('eval', help='Evaluate prompts')
    prompts_eval.add_argument('--domain', help='Evaluate only this domain')
    prompts_eval.add_argument('--live', action='store_true', help='Run live model evaluation')
    prompts_eval.add_argument('--model', help='Model to use for live evaluation')
    prompts_eval.add_argument('--delay', type=float, default=35.0,
                              help='Seconds to wait between live model calls (default: 35)')
    prompts_eval.add_argument('--output', help='Write eval table to this file')

    # Playground subcommands
    prompts_subparsers.add_parser('playground', help='Launch interactive prompt playground REPL')

    prompts_render = prompts_subparsers.add_parser('render', help='Non-interactive: render a prompt to stdout')
    prompts_render.add_argument('prompt_name', metavar='name', help='Prompt name (e.g. research.literature_review.v1)')
    prompts_render.add_argument('--input', dest='input_file', metavar='FILE',
                                help='JSON file with input variables (merged over fixture defaults)')

    prompts_run = prompts_subparsers.add_parser('run', help='Non-interactive: render + run through a model')
    prompts_run.add_argument('prompt_name', metavar='name', help='Prompt name')
    prompts_run.add_argument('--input', dest='input_file', metavar='FILE',
                             help='JSON file with input variables')
    prompts_run.add_argument('--model', required=True, help='Model identifier to run against')

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
    from effgen.models._cost import _BUDGET_CONFIG_PATH
    budget_path = _BUDGET_CONFIG_PATH

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
        cli.print_success(f"Daily budget set to ${amount:.4f} USD")
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

    # Honest cost label: a genuine free tier reads "free" and a model with no
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

    if RICH_AVAILABLE and cli.console:
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
        if daily_budget is not None:
            ratio = total_cost / daily_budget if daily_budget > 0 else 0
            filled = min(20, max(0, int(ratio * 20)))
            bar = "█" * filled + "░" * (20 - filled)
            color = "red" if ratio >= 1.0 else "yellow" if ratio >= 0.8 else "green"
            cli.console.print(
                f"[bold]Daily budget:[/bold] [{color}]{bar}[/{color}] "
                f"${total_cost:.4f} / ${daily_budget:.4f} ({ratio*100:.0f}%)"
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
        if daily_budget is not None:
            ratio = total_cost / daily_budget if daily_budget > 0 else 0
            print(f"\nDaily budget: ${total_cost:.4f} / ${daily_budget:.4f} ({ratio*100:.0f}%)")

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

    # Exit nonzero if a live probe was requested and a keyed provider failed.
    # Computed once so every output format (JSON and human) agrees.
    exit_code = _doctor_exit_code(results, live)

    if getattr(args, 'output_json', False):
        print(_json.dumps({"providers": results, "system": system_report}, indent=2))
        return exit_code

    # Pretty-print
    if RICH_AVAILABLE:
        console = Console()
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

    if wf_cmd == 'validate':
        try:
            dag = WorkflowDAG.from_yaml(args.file)
            order = dag.topological_order()
            cli.print(f"Workflow '{dag.name}' is valid.")
            cli.print(f"  Nodes: {len(dag.nodes)}")
            cli.print(f"  Edges: {len(dag.edges)}")
            cli.print(f"  Execution order: {' -> '.join(order)}")
            return 0
        except Exception as e:
            cli.print(f"Validation failed: {e}")
            return 1

    elif wf_cmd == 'run':
        try:
            model_name = getattr(args, 'model', None)

            def _agent_factory(nd):
                from effgen.core.agent import Agent, AgentConfig
                from effgen.models import load_model
                m = model_name or nd.get('model', 'Qwen/Qwen2.5-1.5B-Instruct')
                model = load_model(m)
                config = AgentConfig(
                    name=nd.get('agent', nd['id']),
                    model=model,
                    max_iterations=nd.get('max_iterations', 5),
                )
                return Agent(config)

            dag = WorkflowDAG.from_yaml(args.file, agent_factory=_agent_factory)
            cli.print(f"Running workflow '{dag.name}' ({len(dag.nodes)} nodes)...")

            # Build initial inputs from --input flags
            initial_inputs = {}
            if getattr(args, 'input', None):
                for node_id, task_str in args.input:
                    initial_inputs[node_id] = task_str

            result = dag.run(initial_inputs=initial_inputs)

            cli.print(f"\nWorkflow {'succeeded' if result.success else 'FAILED'} "
                       f"in {result.execution_time:.2f}s")
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
            cli.print(f"Workflow execution failed: {e}")
            return 1

    else:
        cli.print("Usage: effgen workflow [run|validate] <file.yaml>")
        return 0


def _handle_batch_command(args, cli) -> int:
    """Handle the 'batch' CLI subcommand."""
    from effgen.core.batch import BatchConfig, BatchRunner

    input_path = args.input
    output_path = getattr(args, 'output', None)
    model_name = getattr(args, 'model', None) or 'Qwen/Qwen2.5-1.5B-Instruct'
    preset_name = getattr(args, 'preset', None)
    query_field = getattr(args, 'query_field', 'query')

    try:
        # Create agent
        if preset_name:
            from effgen.models import load_model
            from effgen.presets import create_agent
            model = load_model(model_name)
            agent = create_agent(preset_name, model)
        else:
            from effgen.core.agent import Agent, AgentConfig
            from effgen.models import load_model
            model = load_model(model_name)
            config = AgentConfig(name="batch-agent", model=model, max_iterations=5)
            agent = Agent(config)

        runner = BatchRunner(agent)
        cli.print(f"Loading queries from {input_path}...")

        # Count queries up front so the progress bar shows a real total/ETA.
        try:
            _queries = runner._read_queries(Path(input_path), query_field)
            _total = len(_queries)
        except Exception:  # noqa: BLE001 - fall back to an indeterminate bar
            _total = None

        animate = _progress.animation_enabled(
            quiet=getattr(args, 'quiet', False),
            no_animation=getattr(args, 'no_animation', False),
        )
        with _progress.StepProgress(
            cli.console, total=_total, description="Batch", animate=animate,
        ) as _bar:
            batch_config = BatchConfig(
                max_concurrency=args.concurrency,
                batch_size=args.batch_size,
                retry_failed=args.retries,
                timeout_per_item=args.timeout,
                progress_callback=lambda done, total: _bar.update(done, total),
            )
            result = runner.run_from_file(
                input_path, config=batch_config, query_field=query_field,
            )

        cli.print(
            f"\nBatch complete: {result.succeeded}/{result.total} succeeded "
            f"in {result.total_time:.2f}s"
        )

        if output_path:
            queries = runner._read_queries(
                __import__('pathlib').Path(input_path), query_field,
            )
            runner.write_results(result, output_path, query_list=queries)
            cli.print(f"Results written to {output_path}")

        return 0 if result.failed == 0 else 1

    except Exception as e:
        cli.print(f"Batch execution failed: {e}")
        return 1


def _handle_eval_command(args, cli) -> int:
    """Handle 'effgen eval' subcommand."""
    from effgen.eval import AgentEvaluator, RegressionTracker, get_suite, list_suites
    from effgen.eval.evaluator import ScoringMode

    suite_name = args.suite
    model_name = getattr(args, 'model', None) or 'Qwen/Qwen2.5-1.5B-Instruct'
    preset_name = getattr(args, 'preset', None)
    scoring = ScoringMode(args.scoring)
    threshold = args.threshold
    difficulty = getattr(args, 'difficulty', None)

    try:
        # List suites if requested
        if suite_name == 'list':
            cli.print_header("Available Evaluation Suites")
            for name, desc in list_suites().items():
                cli.print(f"  {name:16s} — {desc}")
            return 0

        suite = get_suite(suite_name)

        # Filter by difficulty if specified
        if difficulty:
            from effgen.eval.evaluator import Difficulty
            suite.test_cases = suite.filter(difficulty=Difficulty(difficulty))
            cli.print(f"Filtered to {len(suite.test_cases)} {difficulty} test cases")

        cli.print(f"Loading model {model_name}...")

        # Create agent
        if preset_name:
            from effgen.models import load_model
            from effgen.presets import create_agent
            model = load_model(model_name)
            agent = create_agent(preset_name, model)
        else:
            from effgen.core.agent import Agent, AgentConfig
            from effgen.models import load_model
            model = load_model(model_name)
            config = AgentConfig(name="eval-agent", model=model, max_iterations=10)
            agent = Agent(config)

        cli.print(f"Running {suite_name} suite ({len(suite)} cases, scoring={args.scoring})...")
        evaluator = AgentEvaluator(agent, scoring=scoring, pass_threshold=threshold)
        animate = _progress.animation_enabled(
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
        cli.print(f"  Accuracy:       {summary['accuracy']:.1%} ({summary['passed']}/{summary['total']})")
        cli.print(f"  Avg Latency:    {summary['avg_latency']:.4f}s")
        cli.print(f"  Total Tokens:   {summary['total_tokens']}")
        cli.print(f"  Tool Accuracy:  {summary['avg_tool_accuracy']:.1%}")

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

        # Save baseline
        if args.save_baseline:
            from effgen import __version__
            tracker = RegressionTracker()
            path = tracker.save_baseline(suite_name, results, version=__version__)
            cli.print(f"\n  Baseline saved to {path}")

        # Compare baseline
        if args.compare_baseline:
            from effgen import __version__
            tracker = RegressionTracker()
            report = tracker.compare(suite_name, results, version=__version__)
            cli.print(f"\n{report.to_markdown()}")

        # Write output
        if args.output:
            Path(args.output).write_text(results.to_json(), encoding="utf-8")
            cli.print(f"\n  Results written to {args.output}")

        return 0 if results.accuracy >= 0.5 else 1

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


def _handle_compare_command(args, cli) -> int:
    """Handle 'effgen compare' subcommand."""
    from effgen.eval import ModelComparison, get_suite
    from effgen.eval.evaluator import ScoringMode

    model_names = [m.strip() for m in args.models.split(',')]
    suite_name = args.suite
    scoring = ScoringMode(args.scoring)
    threshold = args.threshold
    preset_name = getattr(args, 'preset', None)

    try:
        suite = get_suite(suite_name)

        # Load all models and create agents
        from effgen.models import load_model
        agents: dict = {}

        for model_name in model_names:
            cli.print(f"Loading model {model_name}...")
            try:
                model = load_model(model_name)
                if preset_name:
                    from effgen.presets import create_agent
                    agent = create_agent(preset_name, model)
                else:
                    from effgen.core.agent import Agent, AgentConfig
                    config = AgentConfig(name=f"compare-{model_name}", model=model, max_iterations=10)
                    agent = Agent(config)
                agents[model_name] = agent
            except Exception as e:
                cli.print(f"  Warning: Failed to load {model_name}: {e}")

        if not agents:
            cli.print("Error: No models loaded successfully.")
            return 1

        cli.print(f"\nComparing {len(agents)} models on {suite_name} ({len(suite)} cases)...")
        comparison = ModelComparison(scoring=scoring, pass_threshold=threshold)
        matrix = comparison.run(agents, [suite])

        # Display
        cli.print(matrix.to_markdown())

        # Write output
        if args.output:
            output_path = args.output
            if output_path.endswith('.md'):
                Path(output_path).write_text(matrix.to_markdown(), encoding="utf-8")
            else:
                Path(output_path).write_text(matrix.to_json(), encoding="utf-8")
            cli.print(f"\nResults written to {output_path}")

        return 0

    except Exception as e:
        cli.print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


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
    cp = mgr.load(cp_id) if cp_id else mgr.load_latest()
    cli.print(f"Resuming '{cp.task[:80]}' from iteration {cp.iteration}")

    if getattr(args, 'preset', None):
        from effgen.presets import create_agent as _create_preset_agent
        agent = _create_preset_agent(args.preset, args.model or "Qwen/Qwen2.5-3B-Instruct")
    else:
        cfg = AgentConfig(name=cp.agent_name, model=args.model or "Qwen/Qwen2.5-3B-Instruct", tools=[])
        agent = Agent(cfg)

    response = agent.resume(checkpoint_id=cp_id, checkpoint_dir=ckpt_dir)
    cli.print(response.output if hasattr(response, 'output') else str(response))
    return 0 if getattr(response, 'success', True) else 1


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
        if response.execution_trace:
            cli.print_header("What the agent did")
            for i, step in enumerate(response.execution_trace, 1):
                action = step.get("action", step.get("tool", ""))
                observation = step.get("observation", step.get("output", ""))
                if action:
                    cli.print(f"  {i}. used [cyan]{action}[/cyan] → {str(observation)[:80]}"
                              if cli.console else
                              f"  {i}. used {action} -> {str(observation)[:80]}")
            if not any(s.get("action") or s.get("tool") for s in response.execution_trace):
                cli.print("  (answered directly, no tools needed)")
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
    mgr = SessionManager()
    cmd = getattr(args, 'session_command', None)
    if cmd == 'list':
        sessions = mgr.list_sessions()
        if not sessions:
            cli.print(
                "No sessions yet. Start one with: effgen chat  (or effgen run \"...\" "
                "creates a session you can resume)."
            )
            return 0
        for s in sessions:
            cli.print(f"  {s['session_id']:36s}  msgs={s['messages']:<4d}  updated={s.get('updated_at')}")
        return 0
    if cmd == 'delete':
        ok = mgr.delete(args.session_id)
        cli.print("Deleted." if ok else "Not found.")
        return 0 if ok else 1
    if cmd == 'export':
        cli.print(mgr.export(args.session_id, format=args.format))
        return 0
    if cmd == 'cleanup':
        n = mgr.cleanup(older_than_days=args.days)
        cli.print(f"Removed {n} old session(s).")
        return 0
    cli.print("Usage: effgen sessions [list|delete|export|cleanup]")
    return 1


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
            print(_json.dumps(rows, indent=2))
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
            t.add_column("Name", style="cyan")
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
        try:
            p = registry.get(name)
        except KeyError:
            cli.print_error(f"Prompt '{name}' not found.")
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

        prompts = registry.search(domain=domain_filter)
        evaluator = PromptEval()

        cli.print_header("Running golden eval...")
        golden_report = evaluator.eval_all_golden(prompts)
        table = golden_report.as_table()
        print(table)

        full_table = "=== Golden Eval ===\n" + table

        if live:
            if not model:
                cli.print_error("--model is required for --live eval")
                return 1
            cli.print_header(f"Running live eval with model '{model}'...")
            live_report = evaluator.eval_all_live(prompts, model, delay=delay)
            live_table = live_report.as_table()
            print(live_table)
            full_table += "\n=== Live Eval ===\n" + live_table

        if output_path:
            Path(output_path).write_text(full_table)
            cli.print_success(f"Eval table written to {output_path}")

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
        inputs = {}
        if input_file:
            try:
                inputs = _json.loads(Path(input_file).read_text())
            except Exception as exc:
                cli.print_error(f"Could not read input file: {exc}")
                return 1
        return cmd_run(name, inputs, model)

    cli.print("Usage: effgen prompts [list|show|eval|playground|render|run]")
    return 1


def _env_search_paths() -> list[Path]:
    """The ordered list of ``.env`` locations the CLI loads (earliest wins).

    Search order (documented so pip-installed users aren't surprised):
      1. ``$EFFGEN_DOTENV`` — explicit override, if set.
      2. ``~/.effgen/.env`` — per-user effGen config.
      3. ``./.env`` and each parent directory up to the filesystem root — the
         nearest project ``.env`` to the current working directory.
    Values are loaded non-overriding, so a real environment variable always
    wins over a file, and earlier files win over later ones.
    """
    paths: list[Path] = []
    override = os.environ.get("EFFGEN_DOTENV")
    if override:
        paths.append(Path(override))
    paths.append(Path.home() / ".effgen" / ".env")
    # Walk up from the cwd to find the nearest project .env (a checkout's repo
    # root, for example) instead of a confusing package-relative path.
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        paths.append(d / ".env")
    return paths


def load_env_files() -> list[str]:
    """Load ``.env`` files from the documented search paths (non-overriding).

    Returns the list of paths actually loaded (for diagnostics).
    """
    loaded: list[str] = []
    try:
        from dotenv import load_dotenv as _load_dotenv
    except ImportError:
        return loaded
    seen: set[Path] = set()
    for ep in _env_search_paths():
        try:
            rp = ep.resolve()
        except Exception:
            rp = ep
        if rp in seen:
            continue
        seen.add(rp)
        if ep.exists():
            _load_dotenv(ep, override=False)
            loaded.append(str(ep))
    return loaded


def main():
    """Main entry point for CLI."""
    # Load .env early so all subcommands see API keys (see load_env_files).
    load_env_files()

    parser = create_parser()
    args = parser.parse_args()

    # Handle completion script generation
    if getattr(args, 'completion', None):
        from effgen.completion import get_completion
        print(get_completion(args.completion))
        sys.exit(0)

    # Setup logging. --verbose / --quiet may appear either before the
    # subcommand (global) or after it (per-command); honor whichever is set.
    setup_logging(
        verbose=getattr(args, 'verbose', False),
        log_file=getattr(args, 'log_file', None),
        quiet=getattr(args, 'quiet', False),
    )

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
            cli.print_header("Available Agent Presets")
            for name, desc in _list_presets().items():
                cli.print(f"  {name:12s} — {desc}")
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
            run_debug_cli(
                task=args.task,
                preset=getattr(args, 'preset', None),
                model=getattr(args, 'model', None),
                step=getattr(args, 'step', False),
            )
            exit_code = 0
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
