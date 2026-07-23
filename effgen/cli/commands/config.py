"""The ``effgen config`` command group: show, validate, init, and set config.

``_main`` parses arguments and dispatches; the ``CLIInterface._config_*``
methods delegate here. Holds the config-file round-trip and budget-key logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from effgen.cli import onboarding as _onboarding

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface


def config_commands(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """Configuration management commands.

    Args:
        args: Parsed command-line arguments
    """
    from effgen.cli.commands._shared import _print_group_help

    if args.config_command == 'show':
        cli._config_show(args)
    elif args.config_command == 'validate':
        cli._config_validate(args)
    elif args.config_command == 'init':
        cli._config_init(args)
    elif args.config_command == 'set':
        cli._config_set(args)
    elif args.config_command is None:
        return _print_group_help(args)
    else:
        cli.print_error(f"Unknown config command: {args.config_command}")
        return 1

    return 0


def config_set(cli: "CLIInterface", args: argparse.Namespace) -> None:
    """Handle 'effgen config set <key> <value>'."""
    key: str = args.key
    value_str: str = args.value

    _supported_keys = ("budget.daily", "budget.monthly")
    # Route budget.* keys to the cost tracker's budget config.
    if key.startswith("budget."):
        budget_key = key[len("budget."):]
        if budget_key not in {"daily", "monthly"}:
            hint = _onboarding.did_you_mean(key, _supported_keys, n=1, cutoff=0.5)
            cli.print_error(
                f"Unknown budget key: {key!r}. {hint + ' ' if hint else ''}"
                f"Supported: {', '.join(_supported_keys)}."
            )
            return
        try:
            value = float(value_str)
        except ValueError:
            cli.print_error(f"Budget value must be a number, got: {value_str!r}")
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
        cli.print_success(f"Set {key} = {value}")
    else:
        hint = _onboarding.did_you_mean(key, _supported_keys, n=1, cutoff=0.5)
        cli.print_error(
            f"Unknown config key: {key!r}. {hint + ' ' if hint else ''}"
            f"Supported: {', '.join(_supported_keys)}."
        )


def config_show(cli: "CLIInterface", args: argparse.Namespace) -> None:
    """Show current configuration."""
    from rich.syntax import Syntax

    from effgen.ui.theme import CODE_THEME

    cli.print_header("Configuration")

    if args.file:
        try:
            config = cli.config_loader.load_config(args.file)

            if cli.console:
                syntax = Syntax(
                    json.dumps(config.to_dict(), indent=2),
                    "json",
                    theme=CODE_THEME,
                    line_numbers=True
                )
                cli.console.print(syntax)
            else:
                print(json.dumps(config.to_dict(), indent=2))

        except Exception as e:
            cli.print_error(f"Error loading config: {e}")
    else:
        cli.print_warning("No configuration file specified")
        cli.print("Use: effgen config show --file <path>")


def config_validate(cli: "CLIInterface", args: argparse.Namespace) -> None:
    """Validate configuration file."""
    if not args.file:
        cli.print_error("Configuration file required")
        return

    try:
        cli.config_loader.load_config(args.file, validate=True)
        cli.print_success(f"Configuration is valid: {args.file}")
    except Exception as e:
        cli.print_error(f"Configuration validation failed: {e}")


def config_init(cli: "CLIInterface", args: argparse.Namespace) -> None:
    """Initialize a new configuration file."""
    output_path = Path(args.output or "config.yaml")

    if output_path.exists() and not args.force:
        cli.print_error(f"File already exists: {output_path}")
        cli.print("Use --force to overwrite")
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

    cli.print_success(f"Configuration initialized: {output_path}")
