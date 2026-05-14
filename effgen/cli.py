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


# Configure logging
def setup_logging(verbose: bool = False, log_file: str | None = None):
    """
    Configure logging for CLI.

    Args:
        verbose: Enable verbose logging
        log_file: Optional log file path
    """
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class CLIInterface:
    """Main CLI interface for effGen."""

    def __init__(self):
        """Initialize CLI interface."""
        self.console = Console() if RICH_AVAILABLE else None
        self.config_loader = ConfigLoader()
        self.tool_registry = get_tool_registry()

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
            # Step 1: Agent Type Selection
            self.print_header("Step 1: Select Agent Type")
            agent_types = [
                ("1", "CodeAgent", "Agent that generates and executes code (recommended)"),
                ("2", "ToolCallingAgent", "Agent that calls tools via structured outputs"),
                ("3", "ReActAgent", "Agent using Reason+Act pattern (default)")
            ]

            if self.console:
                table = Table(title="Available Agent Types")
                table.add_column("#", style="cyan", width=3)
                table.add_column("Type", style="magenta")
                table.add_column("Description", style="white")
                for num, name, desc in agent_types:
                    table.add_row(num, name, desc)
                self.console.print(table)
            else:
                for num, name, desc in agent_types:
                    print(f"  [{num}] {name}: {desc}")

            agent_type_input = input("\nSelect agent type [3]: ").strip() or "3"
            agent_type_map = {"1": "code", "2": "tool_calling", "3": "react"}
            agent_type = agent_type_map.get(agent_type_input, "react")
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
                "Agent Type": agent_type,
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

        self.print_header(f"effGen v{__version__} - Running Task")

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
                agent = _create_preset_agent(
                    args.preset,
                    model_id,
                    agent_name=args.name,
                    system_prompt=args.system_prompt or config.get("system_prompt"),
                    max_iterations=args.max_iterations,
                    temperature=args.temperature,
                    enable_streaming=args.stream,
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
                            tool = asyncio.run(self.tool_registry.get_tool(tool_name))
                            tools.append(tool)
                            self.print_success(f"Loaded tool: {tool_name}")
                        except KeyError:
                            self.print_error(f"Tool not found: {tool_name}")
                            return 1
                else:
                    # Load all builtin tools by default
                    self.tool_registry.discover_builtin_tools()
                    tool_names = self.tool_registry.list_tools()
                    for name in tool_names[:5]:  # Limit to first 5 tools
                        try:
                            tool = asyncio.run(self.tool_registry.get_tool(name))
                            tools.append(tool)
                        except Exception as e:
                            logging.debug(f"Failed to load tool {name}: {e}")

                # Create agent configuration
                agent_config = AgentConfig(
                    name=args.name or "cli-agent",
                    model=args.model or "Qwen/Qwen2.5-3B-Instruct",
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

            if args.stream:
                # Streaming output
                self.print("[italic]Streaming response...[/italic]\n" if self.console else "Streaming response...\n")
                for token in agent.stream(args.task, mode=mode):
                    print(token, end='', flush=True)
                print()  # New line after streaming
            else:
                # Regular output with spinner
                if self.console:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=self.console
                    ) as progress:
                        progress.add_task("Thinking...", total=None)
                        response = agent.run(args.task, mode=mode, **_phase7_run_kwargs(args))
                else:
                    self.print("Thinking...")
                    response = agent.run(args.task, mode=mode)

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

            return 0

        except Exception as e:
            self.print_error(f"Error running agent: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

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
        Start API server.

        Args:
            args: Parsed command-line arguments
        """
        self.print_header(f"effGen v{__version__} - API Server")

        try:
            import time as _time
            from contextlib import asynccontextmanager

            import uvicorn
            from fastapi import (
                Depends,
                FastAPI,
                HTTPException,
                Request,
                WebSocket,
                WebSocketDisconnect,
            )
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse
            from fastapi.security import APIKeyHeader
            from pydantic import BaseModel as PydanticBaseModel
            from pydantic import ConfigDict
        except ImportError:
            self.print_error("FastAPI and uvicorn are required for server mode.")
            self.print("Install with: pip install fastapi uvicorn")
            return 1

        try:
            # Define request/response models with Pydantic v2 style
            class TaskRequest(PydanticBaseModel):
                model_config = ConfigDict(extra="ignore")

                task: str
                model: str | None = "Qwen/Qwen2.5-3B-Instruct"
                tools: list[str] | None = None
                preset: str | None = None
                temperature: float | None = 0.7
                max_iterations: int | None = 10
                stream: bool = False

            class TaskResponse(PydanticBaseModel):
                model_config = ConfigDict(extra="ignore")

                output: str
                success: bool
                metadata: dict[str, Any]

            # Store reference to self for use in lifespan
            cli_instance = self

            # --- Rate limiter (simple in-memory token bucket) ---
            _rate_buckets: dict[str, list] = {}
            _rate_limit = int(os.environ.get("EFFGEN_RATE_LIMIT", "60"))  # requests/min

            def _check_rate(client_ip: str) -> bool:
                now = _time.time()
                bucket = _rate_buckets.setdefault(client_ip, [])
                # Remove entries older than 60s
                _rate_buckets[client_ip] = bucket = [t for t in bucket if now - t < 60]
                if len(bucket) >= _rate_limit:
                    return False
                bucket.append(now)
                return True

            # --- API key auth (optional) ---
            api_key_name = "X-API-Key"
            api_key_header = APIKeyHeader(name=api_key_name, auto_error=False)
            expected_key = os.environ.get("EFFGEN_API_KEY")  # None = auth disabled

            async def verify_api_key(key: str | None = Depends(api_key_header)):
                if expected_key and key != expected_key:
                    raise HTTPException(status_code=401, detail="Invalid or missing API key")

            # --- Metrics state ---
            _metrics = {"requests": 0, "errors": 0, "total_time": 0.0}

            @asynccontextmanager
            async def lifespan(app: FastAPI):
                """Lifespan context manager for startup/shutdown."""
                cli_instance.print_success("Server starting up...")
                cli_instance.tool_registry.discover_builtin_tools()
                cli_instance.print_success(f"Discovered {len(cli_instance.tool_registry.list_tools())} tools")
                if expected_key:
                    cli_instance.print("API key auth enabled (set via EFFGEN_API_KEY)")
                cli_instance.print(f"Rate limit: {_rate_limit} req/min (set via EFFGEN_RATE_LIMIT)")
                yield
                cli_instance.print("Server shutting down...")

            # Create FastAPI app with lifespan
            app = FastAPI(
                title="effGen API",
                description="API server for effGen framework. "
                            "Set EFFGEN_API_KEY to enable authentication. "
                            "Set EFFGEN_RATE_LIMIT to configure rate limiting (default 60 req/min).",
                version=__version__,
                lifespan=lifespan
            )

            # Add CORS middleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # --- Request logging & rate limiting middleware ---
            @app.middleware("http")
            async def request_middleware(request: Request, call_next):
                start = _time.time()
                client_ip = request.client.host if request.client else "unknown"

                if not _check_rate(client_ip):
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded. Try again later."}
                    )

                _metrics["requests"] += 1
                logging.info("API %s %s from %s", request.method, request.url.path, client_ip)

                try:
                    response = await call_next(request)
                except Exception:
                    _metrics["errors"] += 1
                    raise
                finally:
                    elapsed = _time.time() - start
                    _metrics["total_time"] += elapsed
                    logging.info("API %s %s completed in %.3fs", request.method, request.url.path, elapsed)

                return response

            # Store state in app
            app.state.cli = cli_instance

            @app.post("/run", dependencies=[Depends(verify_api_key)])
            async def run_task(request: TaskRequest):
                """Run a task with an agent."""
                try:
                    # Use preset if specified
                    if request.preset:
                        from effgen.presets import create_agent as _create_preset_agent
                        agent_instance = _create_preset_agent(
                            request.preset,
                            request.model,
                            temperature=request.temperature,
                            max_iterations=request.max_iterations,
                        )
                    else:
                        # Create agent for each request to handle different models
                        tools = []
                        tool_names = app.state.cli.tool_registry.list_tools()[:5]
                        for name in tool_names:
                            try:
                                tool = await app.state.cli.tool_registry.get_tool(name)
                                tools.append(tool)
                            except Exception as tool_err:
                                logging.debug(f"Failed to load tool {name}: {tool_err}")

                        agent_config = AgentConfig(
                            name="api-agent",
                            model=request.model,
                            tools=tools,
                            temperature=request.temperature,
                            max_iterations=request.max_iterations
                        )
                        agent_instance = Agent(agent_config)

                    # Run task
                    response = agent_instance.run(request.task)

                    return JSONResponse(content={
                        "output": response.output,
                        "success": response.success,
                        "metadata": {
                            "mode": response.mode.value if hasattr(response.mode, 'value') else str(response.mode),
                            "iterations": response.iterations,
                            "tool_calls": response.tool_calls,
                            "execution_time": response.execution_time
                        }
                    })

                except Exception as e:
                    _metrics["errors"] += 1
                    logging.exception("Error running task")
                    raise HTTPException(status_code=500, detail=str(e))

            @app.websocket("/ws")
            async def websocket_stream(ws: WebSocket):
                """WebSocket endpoint for streaming agent responses."""
                await ws.accept()
                try:
                    while True:
                        data = await ws.receive_json()
                        task = data.get("task", "")
                        model_id = data.get("model", "Qwen/Qwen2.5-3B-Instruct")
                        preset_name = data.get("preset")

                        if preset_name:
                            from effgen.presets import create_agent as _create_preset_agent
                            agent_instance = _create_preset_agent(preset_name, model_id)
                        else:
                            tools = []
                            for name in app.state.cli.tool_registry.list_tools()[:5]:
                                try:
                                    tool = await app.state.cli.tool_registry.get_tool(name)
                                    tools.append(tool)
                                except Exception:
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
                        except Exception as e:
                            await ws.send_json({"type": "error", "detail": str(e)})

                except WebSocketDisconnect:
                    logging.info("WebSocket client disconnected")

            @app.get("/health")
            async def health():
                """Health check endpoint."""
                return {"status": "healthy", "version": __version__}

            @app.get("/metrics", dependencies=[Depends(verify_api_key)])
            async def metrics():
                """Prometheus-style metrics endpoint."""
                avg_time = (_metrics["total_time"] / _metrics["requests"]
                            if _metrics["requests"] > 0 else 0)
                return {
                    "requests_total": _metrics["requests"],
                    "errors_total": _metrics["errors"],
                    "avg_response_time_seconds": round(avg_time, 4),
                }

            @app.get("/tools")
            async def list_tools_endpoint():
                """List available tools."""
                tools = app.state.cli.tool_registry.list_tools()
                tool_info = []
                for tool_name in tools:
                    try:
                        metadata = app.state.cli.tool_registry.get_metadata(tool_name)
                        tool_info.append({
                            "name": tool_name,
                            "description": metadata.description,
                            "category": metadata.category.value if hasattr(metadata.category, 'value') else str(metadata.category)
                        })
                    except Exception:
                        tool_info.append({"name": tool_name, "description": "N/A", "category": "unknown"})
                return {"tools": tool_info, "count": len(tools)}

            @app.get("/")
            async def root():
                """Root endpoint with API information."""
                return {
                    "name": "effGen API",
                    "version": __version__,
                    "endpoints": {
                        "POST /run": "Run a task with an agent",
                        "WS /ws": "WebSocket streaming",
                        "GET /health": "Health check",
                        "GET /metrics": "Server metrics",
                        "GET /tools": "List available tools",
                        "GET /docs": "OpenAPI documentation"
                    }
                }

            # Start server
            self.print(f"Starting server on {args.host}:{args.port}")
            self.print(f"API docs available at http://{args.host}:{args.port}/docs")
            self.print()

            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                log_level="info" if getattr(args, 'verbose', False) else "warning"
            )

            return 0

        except Exception as e:
            self.print_error(f"Error starting server: {e}")
            if getattr(args, 'verbose', False):
                import traceback
                traceback.print_exc()
            return 1

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

        # Route budget.* keys to the cost tracker's budget config.
        if key.startswith("budget."):
            budget_key = key[len("budget."):]
            if budget_key not in {"daily", "monthly"}:
                self.print_error(
                    f"Unknown budget key: {key!r}. Supported: budget.daily, budget.monthly"
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
            self.print_error(f"Unknown config key: {key!r}. Supported: budget.daily, budget.monthly")

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
            self._tools_list(args)
        elif args.tool_command == 'info':
            self._tools_info(args)
        elif args.tool_command == 'test':
            self._tools_test(args)
        else:
            self.print_error(f"Unknown tools command: {args.tool_command}")
            return 1

        return 0

    def _tools_list(self, args):
        """List available tools."""
        self.print_header("Available Tools")

        # Discover builtin tools
        self.tool_registry.discover_builtin_tools()

        # Get tools
        tools = self.tool_registry.list_tools()

        if not tools:
            self.print_warning("No tools registered")
            return

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

    def _tools_info(self, args):
        """Show detailed tool information."""
        if not args.name:
            self.print_error("Tool name required")
            return

        try:
            metadata = self.tool_registry.get_metadata(args.name)

            self.print_header(f"Tool: {metadata.name}")
            self.print(f"\n[bold]Description:[/bold] {metadata.description}" if self.console else f"\nDescription: {metadata.description}")
            self.print(f"[bold]Category:[/bold] {metadata.category.value}" if self.console else f"Category: {metadata.category.value}")
            self.print(f"[bold]Version:[/bold] {metadata.version}" if self.console else f"Version: {metadata.version}")

            if metadata.tags:
                self.print(f"[bold]Tags:[/bold] {', '.join(metadata.tags)}" if self.console else f"Tags: {', '.join(metadata.tags)}")

            # Show parameters
            if metadata.parameters:
                self.print("\n[bold]Parameters:[/bold]" if self.console else "\nParameters:")
                schema = metadata.to_json_schema()
                if self.console:
                    syntax = Syntax(
                        json.dumps(schema, indent=2),
                        "json",
                        theme="monokai"
                    )
                    self.console.print(syntax)
                else:
                    print(json.dumps(schema, indent=2))

        except KeyError:
            self.print_error(f"Tool not found: {args.name}")
        except Exception as e:
            self.print_error(f"Error getting tool info: {e}")

    def _tools_test(self, args):
        """Test a tool with sample input."""
        if not args.name:
            self.print_error("Tool name required")
            return

        try:
            tool = asyncio.run(self.tool_registry.get_tool(args.name))

            self.print_header(f"Testing Tool: {args.name}")

            # Parse input
            if args.input:
                try:
                    input_data = json.loads(args.input)
                except json.JSONDecodeError:
                    input_data = {"input": args.input}
            else:
                input_data = {}

            # Execute tool
            self.print(f"Input: {input_data}\n")
            result = asyncio.run(tool.execute(**input_data))

            self.print("[bold]Result:[/bold]" if self.console else "Result:")
            if self.console:
                self.console.print(Panel(str(result), border_style="green"))
            else:
                print(result)

        except KeyError:
            self.print_error(f"Tool not found: {args.name}")
        except Exception as e:
            self.print_error(f"Error testing tool: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    def models_commands(self, args):
        """
        Model management commands.

        Args:
            args: Parsed command-line arguments
        """
        if args.model_command == 'list':
            self._models_list(args)
        elif args.model_command == 'info':
            self._models_info(args)
        elif args.model_command == 'load':
            self._models_load(args)
        elif args.model_command == 'unload':
            self._models_unload(args)
        elif args.model_command == 'status':
            self._models_status(args)
        else:
            self.print_error(f"Unknown models command: {args.model_command}")
            return 1

        return 0

    def _models_list(self, args):
        """List available models."""
        self.print_header("Available Models")

        # Load models from config if available
        config_dir = Path("configs")
        models_config = config_dir / "models.yaml"

        if models_config.exists():
            config = self.config_loader.load_config(models_config)
            models = config.get("models", {})

            if self.console:
                table = Table(title="Configured Models")
                table.add_column("Name", style="cyan")
                table.add_column("Path/API", style="magenta")
                table.add_column("Type", style="white")

                for name, model_config in models.items():
                    if isinstance(model_config, dict):
                        table.add_row(
                            name,
                            model_config.get("model_path", model_config.get("api", "N/A")),
                            model_config.get("type", "unknown")
                        )

                self.console.print(table)
            else:
                for name in models.keys():
                    print(f"- {name}")
        else:
            self.print_warning("No models configuration found")
            self.print("Common models:")
            common_models = [
                "Qwen/Qwen2.5-3B-Instruct",
                "mistral-7b",
                "llama-2-7b",
                "gemma-7b"
            ]
            for model in common_models:
                print(f"- {model}")

    def _models_info(self, args):
        """Show model information."""
        if not args.name:
            self.print_error("Model name required")
            return

        self.print_header(f"Model: {args.name}")
        self.print("Model information coming soon...")

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

    def examples_commands(self, args):
        """
        Run example scripts.

        Args:
            args: Parsed command-line arguments
        """
        if args.example_command == 'list':
            self._examples_list(args)
        elif args.example_command == 'run':
            self._examples_run(args)
        else:
            self.print_error(f"Unknown examples command: {args.example_command}")
            return 1

        return 0

    def _examples_list(self, args):
        """List available examples."""
        self.print_header("Available Examples")

        examples_dir = Path(__file__).parent.parent / "examples"

        if not examples_dir.exists():
            self.print_warning("Examples directory not found")
            return

        examples = []
        for file in examples_dir.glob("*.py"):
            if not file.name.startswith("_"):
                examples.append(file.stem)

        # Also check agents subdirectory
        agents_dir = examples_dir / "agents"
        if agents_dir.exists():
            for file in agents_dir.glob("*.py"):
                if not file.name.startswith("_"):
                    examples.append(f"agents/{file.stem}")

        if self.console:
            table = Table(title="Example Scripts")
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
            return

        examples_dir = Path(__file__).parent.parent / "examples"
        example_path = examples_dir / f"{args.name}.py"

        if not example_path.exists():
            self.print_error(f"Example not found: {args.name}")
            return

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
            if args.verbose:
                import traceback
                traceback.print_exc()


def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description=f"effGen v{__version__} - CLI for agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  effgen run "What is the weather in Paris?" --model Qwen/Qwen2.5-3B-Instruct
  effgen chat --model Qwen/Qwen2.5-3B-Instruct --temperature 0.8
  effgen serve --port 8000
  effgen config show --file configs/models.yaml
  effgen tools list
  effgen examples run basic_agent
        """
    )

    parser.add_argument('--version', action='version', version=f'effGen {__version__}')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--completion', choices=['bash', 'zsh', 'fish'],
                        help='Print shell completion script and exit')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run an agent with a task')
    run_parser.add_argument('task', nargs='?', default=None, help='Task description (launches interactive wizard if not provided)')
    run_parser.add_argument('-m', '--model', help='Model to use')
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
    run_parser.add_argument('--preset', choices=['math', 'research', 'coding', 'general', 'minimal'],
                            help='Use a preset agent configuration')
    run_parser.add_argument('--explain', action='store_true',
                            help='Show why the agent chose each tool')
    run_parser.add_argument('--checkpoint-dir', help='Directory to write agent checkpoints (Phase 7)')
    run_parser.add_argument('--checkpoint-interval', type=int, default=0,
                            help='Checkpoint every N iterations (requires --checkpoint-dir)')
    run_parser.add_argument('--session-id', help='Persistent session id (Phase 7)')

    # Resume command (Phase 7.1)
    resume_parser = subparsers.add_parser('resume', help='Resume an agent run from a checkpoint')
    resume_parser.add_argument('--checkpoint', required=True,
                               help='Checkpoint id, JSON path, or directory (uses latest)')
    resume_parser.add_argument('-m', '--model', help='Model to use')
    resume_parser.add_argument('--preset', choices=['math', 'research', 'coding', 'general', 'minimal'])

    # Sessions commands (Phase 7.2)
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
    chat_parser.add_argument('--temperature', type=float, help='Temperature')
    chat_parser.add_argument('--no-sub-agents', action='store_true', help='Disable sub-agents')

    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
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

    tools_subparsers.add_parser('list', help='List tools')

    tools_info = tools_subparsers.add_parser('info', help='Show tool information')
    tools_info.add_argument('name', help='Tool name')

    tools_test = tools_subparsers.add_parser('test', help='Test a tool')
    tools_test.add_argument('name', help='Tool name')
    tools_test.add_argument('-i', '--input', help='Tool input (JSON or string)')

    # Models commands
    models_parser = subparsers.add_parser('models', help='Model management')
    models_subparsers = models_parser.add_subparsers(dest='model_command', help='Models command')

    models_subparsers.add_parser('list', help='List models')

    models_info = models_subparsers.add_parser('info', help='Show model information')
    models_info.add_argument('name', help='Model name')

    models_load = models_subparsers.add_parser('load', help='Pre-load a model into memory')
    models_load.add_argument('name', help='Model name (e.g. Qwen/Qwen2.5-1.5B-Instruct)')
    models_load.add_argument('-e', '--engine', help='Engine (vllm, transformers)', default=None)

    models_unload = models_subparsers.add_parser('unload', help='Unload a model from memory')
    models_unload.add_argument('name', help='Model name')

    models_subparsers.add_parser('status', help='Show loaded models and GPU memory status')

    # Examples commands
    examples_parser = subparsers.add_parser('examples', help='Run example scripts')
    examples_subparsers = examples_parser.add_subparsers(dest='example_command', help='Examples command')

    examples_subparsers.add_parser('list', help='List examples')

    examples_run = examples_subparsers.add_parser('run', help='Run an example')
    examples_run.add_argument('name', help='Example name')

    # Health check command
    subparsers.add_parser('health', help='Check effGen infrastructure health')

    # Doctor command — API key availability check
    doctor_parser = subparsers.add_parser('doctor', help='Check provider API key availability')
    doctor_parser.add_argument('--json', dest='output_json', action='store_true',
                               help='Output as JSON')
    doctor_parser.add_argument('--provider', dest='doctor_provider',
                               help='Check a specific provider only')

    # Plugin commands
    plugin_parser = subparsers.add_parser('create-plugin', help='Generate a plugin project scaffold')
    plugin_parser.add_argument('plugin_name', help='Plugin name (e.g. my_tools)')
    plugin_parser.add_argument('-o', '--output-dir', default='.', help='Output directory')

    # Presets command
    subparsers.add_parser('presets', help='List available agent presets')

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
    batch_parser.add_argument('--preset', choices=['math', 'research', 'coding', 'general', 'minimal'],
                              help='Use a preset agent configuration')
    batch_parser.add_argument('--query-field', default='query', help='Field name for queries in JSONL/CSV (default: query)')

    # Eval command (Phase 11)
    eval_parser = subparsers.add_parser('eval', help='Evaluate an agent against a test suite')
    eval_parser.add_argument('--suite', required=True,
                              help='Test suite name (math, tool_use, reasoning, safety, conversation)')
    eval_parser.add_argument('-m', '--model', help='Model to use')
    eval_parser.add_argument('--preset', choices=['math', 'research', 'coding', 'general', 'minimal'],
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

    # Compare command (Phase 11)
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
    compare_parser.add_argument('--preset', choices=['math', 'research', 'coding', 'general', 'minimal'],
                                 help='Use a preset agent configuration')

    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Run an agent in interactive debug mode')
    debug_parser.add_argument('task', help='Task to execute')
    debug_parser.add_argument('-m', '--model', help='Model to use')
    debug_parser.add_argument('--preset', choices=['math', 'research', 'coding', 'general', 'minimal'],
                              help='Use a preset agent configuration')
    debug_parser.add_argument('--step', action='store_true', help='Step through each iteration')

    # Cost command — spend dashboard + budget management
    cost_parser = subparsers.add_parser('cost', help='View cost spend and manage budgets')
    cost_subparsers = cost_parser.add_subparsers(dest='cost_command', help='Cost command')
    cost_subparsers.add_parser('today', help='Show per-provider/model spend for the last 24 hours')
    cost_subparsers.add_parser('week', help='Show rolling 7-day spend summary')
    cost_subparsers.add_parser('by-provider', help='Show lifetime totals grouped by provider')
    cost_set_budget = cost_subparsers.add_parser('set-budget', help='Set a daily spend budget')
    cost_set_budget.add_argument('amount', type=float, help='Daily budget in USD (e.g. 1.0)')
    cost_subparsers.add_parser('clear-budget', help='Remove configured budget limits')

    return parser


def _create_plugin_scaffold(plugin_name: str, output_dir: str = ".") -> int:
    """Generate a plugin project scaffold."""
    base = Path(output_dir) / f"effgen-plugin-{plugin_name}"
    pkg = base / plugin_name
    try:
        pkg.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Error: Directory {base} already exists.")
        return 1

    (pkg / "__init__.py").write_text(
        f'"""effGen plugin: {plugin_name}"""\n'
    )

    (pkg / "tools.py").write_text(
        f'''"""Custom tools for the {plugin_name} plugin."""

from effgen.tools.base_tool import (
    BaseTool, ToolMetadata, ToolCategory, ParameterSpec, ParameterType,
)


class ExampleTool(BaseTool):
    """An example custom tool — replace with your implementation."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="example_tool",
            description="An example tool that echoes input.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[
                ParameterSpec(
                    name="text",
                    type=ParameterType.STRING,
                    description="Text to echo",
                    required=True,
                ),
            ],
            returns={{"type": "object", "properties": {{"echo": {{"type": "string"}}}}}},
        )

    async def _execute(self, **kwargs):
        text = kwargs.get("text", "")
        return {{"echo": text}}
'''
    )

    (pkg / "plugin.py").write_text(
        f'''"""Plugin registration for {plugin_name}."""

from effgen.tools.plugin import ToolPlugin
from {plugin_name}.tools import ExampleTool


class {plugin_name.title().replace("_", "")}Plugin(ToolPlugin):
    name = "{plugin_name}"
    version = "{__version__}"
    description = "A custom effGen tool plugin."
    tools = [ExampleTool]
'''
    )

    (base / "pyproject.toml").write_text(
        f'''[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "effgen-plugin-{plugin_name}"
version = "{__version__}"
description = "An effGen tool plugin"
requires-python = ">=3.9"
dependencies = ["effgen"]

[project.entry-points."effgen.plugins"]
{plugin_name} = "{plugin_name}.plugin:{plugin_name.title().replace("_", "")}Plugin"
'''
    )

    (base / "README.md").write_text(
        f"# effgen-plugin-{plugin_name}\\n\\n"
        f"An effGen tool plugin.\\n\\n"
        "## Install\\n\\n"
        "```bash\\n"
        f"pip install -e .\\n"
        "```\\n\\n"
        "The plugin will be auto-discovered by effGen via entry points.\\n"
    )

    print(f"Created plugin scaffold at {base}/")
    print(f"  {pkg / 'tools.py':}       — add your custom tools here")
    print(f"  {pkg / 'plugin.py'}     — register tools in the plugin class")
    print(f"  {base / 'pyproject.toml'} — package metadata & entry point")
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

    # Load budget for display
    budget_cfg = {}
    if budget_path.exists():
        try:
            budget_cfg = _json.loads(budget_path.read_text())
        except Exception:
            pass
    daily_budget = budget_cfg.get('daily')

    if RICH_AVAILABLE and cli.console:
        table = Table(title=f"effGen Cost Summary — {period_label}", show_footer=True)
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Model", style="white")
        table.add_column("Requests", style="yellow", justify="right")
        table.add_column("Prompt Tokens", style="blue", justify="right")
        table.add_column("Completion Tokens", style="blue", justify="right")
        table.add_column("Cost (USD)", style="green", justify="right",
                         footer=f"${total_cost:.6f}")

        if not rows:
            table.add_row("—", "No data", "0", "0", "0", "$0.000000")
        else:
            for r in rows:
                table.add_row(
                    r['provider'],
                    r['model'],
                    str(r['requests']),
                    f"{r['prompt_tokens']:,}",
                    f"{r['completion_tokens']:,}",
                    f"${r['cost_usd']:.6f}",
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
        print(f"{'Provider':<15} {'Model':<35} {'Reqs':>5} {'Cost (USD)':>12}")
        print("-" * 80)
        if not rows:
            print("  (no data)")
        else:
            for r in rows:
                model_short = r['model'][:34] if len(r['model']) > 34 else r['model']
                print(f"{r['provider']:<15} {model_short:<35} {r['requests']:>5} ${r['cost_usd']:>11.6f}")
        print("-" * 80)
        print(f"{'TOTAL':<15} {'':<35} {total_requests:>5} ${total_cost:>11.6f}")
        if daily_budget is not None:
            ratio = total_cost / daily_budget if daily_budget > 0 else 0
            print(f"\nDaily budget: ${total_cost:.4f} / ${daily_budget:.4f} ({ratio*100:.0f}%)")

    return 0


def _handle_doctor_command(args) -> int:
    """Handle the 'effgen doctor' subcommand — check API key availability."""
    import json as _json

    # Load .env from standard locations before checking keys (all, non-overriding)
    try:
        from dotenv import load_dotenv
        for _env_path in [
            Path.home() / ".effgen" / ".env",
            Path(".env"),
            Path(__file__).parent.parent / ".env",
        ]:
            if _env_path.exists():
                load_dotenv(_env_path, override=False)
    except ImportError:
        pass

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

    if getattr(args, 'output_json', False):
        print(_json.dumps(results, indent=2))
        return 0

    # Pretty-print
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="effgen doctor — Provider API Key Status")
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Status", style="white")
        table.add_column("Env Key Found", style="dim")
        table.add_column("Models", style="dim", justify="right")

        for prov in sorted(results):
            info = results[prov]
            available = info.get("available", False)
            env_key = info.get("env_key") or "—"
            status = "[green]READY[/green]" if available else "[red]MISSING KEY[/red]"
            try:
                n_models = str(len(ProviderRegistry.list_models(prov)))
            except Exception:
                n_models = "?"
            table.add_row(prov, status, env_key, n_models)

        console.print(table)

        # Print hints for missing keys
        missing = [p for p, i in results.items() if not i.get("available")]
        if missing:
            console.print("\n[yellow]Missing keys — set in ~/.effgen/.env or export:[/yellow]")
            for prov in missing:
                keys = results[prov].get("env_keys_checked", [])
                key_str = " or ".join(keys) if keys else f"{prov.upper()}_API_KEY"
                console.print(f"  export {key_str}=<your-key>")
        else:
            console.print("\n[green]All providers ready![/green]")
    else:
        print("effgen doctor — Provider API Key Status")
        print("-" * 50)
        for prov in sorted(results):
            info = results[prov]
            available = info.get("available", False)
            env_key = info.get("env_key") or "not set"
            status = "READY" if available else "MISSING KEY"
            print(f"  {prov:15s} {status:12s}  (key: {env_key})")
        missing = [p for p, i in results.items() if not i.get("available")]
        if missing:
            print("\nMissing keys — set in ~/.effgen/.env or export:")
            for prov in missing:
                keys = results[prov].get("env_keys_checked", [])
                key_str = " or ".join(keys) if keys else f"{prov.upper()}_API_KEY"
                print(f"  export {key_str}=<your-key>")

    return 0


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

        batch_config = BatchConfig(
            max_concurrency=args.concurrency,
            batch_size=args.batch_size,
            retry_failed=args.retries,
            timeout_per_item=args.timeout,
        )

        runner = BatchRunner(agent)
        cli.print(f"Loading queries from {input_path}...")
        result = runner.run_from_file(input_path, config=batch_config, query_field=query_field)

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
        results = evaluator.run_suite(suite)

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


def _phase7_run_kwargs(args) -> dict:
    """Extract Phase 7 run() kwargs from CLI args."""
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


def _handle_sessions_command(args, cli) -> int:
    """Handle 'effgen sessions' subcommands."""
    from effgen.core.session import SessionManager
    mgr = SessionManager()
    cmd = getattr(args, 'session_command', None)
    if cmd == 'list':
        sessions = mgr.list_sessions()
        if not sessions:
            cli.print("No sessions found.")
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


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle completion script generation
    if getattr(args, 'completion', None):
        from effgen.completion import get_completion
        print(get_completion(args.completion))
        sys.exit(0)

    # Setup logging
    setup_logging(getattr(args, 'verbose', False), getattr(args, 'log_file', None))

    # Create CLI interface
    cli = CLIInterface()

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
        - Agent type selection (CodeAgent vs ToolCallingAgent)
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
