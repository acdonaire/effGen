"""
effGen Prompt Playground

Interactive REPL and non-interactive CLI for exploring prompt templates.

REPL commands:
  select <name>           — pick a prompt from the registry
  set <key> <value>       — bind a variable (JSON-decoded if possible)
  unset <key>             — remove a variable binding
  render                  — print the rendered prompt using current vars
  run [--model <id>]      — run rendered prompt through a model
  save [<path>]           — save session to JSON
  load <path>             — restore a saved session
  reload                  — re-import the template module (hot-reload)
  list [--domain D]       — list registered prompts
  show <name>             — show prompt details
  help                    — show this help
  exit / quit / Ctrl-D    — exit the REPL

Non-interactive (CLI):
  effgen prompts render <name> --input <file.json>
  effgen prompts run <name> --input <file.json> --model <id>
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from effgen.prompts.library.registry import registry
from effgen.prompts.library.session import PlaygroundSession

try:
    from rich.console import Console
    from rich.panel import Panel
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print(msg: str) -> None:
    if _RICH and _console:
        _console.print(msg)
    else:
        print(msg)


def _print_prompt(text: str) -> None:
    if _RICH and _console:
        _console.print(Panel(text, title="Rendered Prompt", border_style="green"))
    else:
        print("\n--- Rendered Prompt ---")
        print(text)
        print("--- End ---")


def _print_output(text: str, model: str) -> None:
    if _RICH and _console:
        _console.print(Panel(text, title=f"Model Output ({model})", border_style="blue"))
    else:
        print(f"\n--- Model Output ({model}) ---")
        print(text)
        print("--- End ---")


def _print_err(msg: str) -> None:
    if _RICH and _console:
        _console.print(f"[red]Error:[/red] {msg}")
    else:
        print(f"Error: {msg}", file=sys.stderr)


def _print_warn(msg: str) -> None:
    if _RICH and _console:
        _console.print(f"[yellow]Warning:[/yellow] {msg}")
    else:
        print(f"Warning: {msg}")


def _print_ok(msg: str) -> None:
    if _RICH and _console:
        _console.print(f"[green]{msg}[/green]")
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Variable parsing
# ---------------------------------------------------------------------------

def _parse_value(raw: str) -> Any:
    """Try JSON decoding, fall back to plain string."""
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _split_set_args(line: str) -> tuple[str, str] | None:
    """Parse 'key value' from a `set` command line.

    The value part may be a JSON literal or a bare string (possibly
    space-separated after the key).  We peel off the first token as the
    key and treat everything after it as the raw value.
    """
    parts = line.split(None, 1)
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# Hot-reload helper
# ---------------------------------------------------------------------------

def _reload_prompt(name: str) -> bool:
    """Re-import the module that registered *name*.

    Walk sys.modules for any module under effgen.prompts.library.domains
    whose file contains the prompt's domain, and reload it.  After reload
    the registry should contain the updated template.
    """
    try:
        p = registry.get(name)
    except KeyError:
        _print_err(f"Prompt '{name}' not in registry")
        return False

    domain = p.domain
    domain_pkg = f"effgen.prompts.library.domains.{domain}"
    reloaded: list[str] = []
    for modname, mod in list(sys.modules.items()):
        if not (modname == domain_pkg or modname.startswith(domain_pkg + ".")):
            continue
        try:
            importlib.reload(mod)
            reloaded.append(modname)
        except Exception as exc:
            _print_warn(f"Reload of {modname} raised: {exc}")

    if reloaded:
        _print_ok(f"Reloaded: {', '.join(reloaded)}")
        # Force registry re-discovery next access
        registry._discovered = False
        return True

    _print_warn("No matching modules found to reload")
    return False


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------

def _run_prompt(rendered: str, model: str) -> str:
    """Call a model via effgen's model loader and return text."""
    from effgen.prompts.library.eval import PromptEval
    evaluator = PromptEval()
    return evaluator._run_model(rendered, model)


# ---------------------------------------------------------------------------
# Non-interactive entry-points
# ---------------------------------------------------------------------------

def cmd_render(name: str, inputs: dict[str, Any]) -> int:
    """Non-interactive render: print rendered prompt to stdout."""
    try:
        p = registry.get(name)
    except KeyError:
        _print_err(f"Prompt '{name}' not found in registry")
        return 1
    try:
        merged = {**p.fixture, **inputs}
        rendered = p.template(**merged)
    except Exception as exc:
        _print_err(f"Render failed: {exc}")
        return 1
    _print_prompt(rendered)
    return 0


def cmd_run(name: str, inputs: dict[str, Any], model: str) -> int:
    """Non-interactive run: render + call model + print output."""
    try:
        p = registry.get(name)
    except KeyError:
        _print_err(f"Prompt '{name}' not found in registry")
        return 1
    try:
        merged = {**p.fixture, **inputs}
        rendered = p.template(**merged)
    except Exception as exc:
        _print_err(f"Render failed: {exc}")
        return 1
    _print_prompt(rendered)
    try:
        output = _run_prompt(rendered, model)
    except Exception as exc:
        _print_err(f"Model call failed: {exc}")
        return 1
    _print_output(output, model)
    return 0


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

_HELP = """\
REPL commands:
  select <name>           pick a prompt from the registry
  set <key> <value>       bind a variable (JSON or plain string)
  unset <key>             remove a variable
  render                  print the rendered prompt
  run [--model <id>]      run rendered prompt through a model
  save [<path>]           save session to JSON
  load <path>             restore a saved session
  reload                  hot-reload the template module
  list [--domain <d>]     list registered prompts
  show <name>             show prompt details
  help                    show this help
  exit / quit / Ctrl-D    exit
"""


class PlaygroundREPL:
    """Interactive REPL for the prompt playground."""

    def __init__(self, default_model: str | None = None) -> None:
        self.session = PlaygroundSession()
        self.default_model = default_model or "llama3.1-8b"
        self._running = False

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Start the interactive REPL loop.  Returns exit code."""
        self._running = True
        _print("[bold cyan]effGen Prompt Playground[/bold cyan]" if _RICH else "effGen Prompt Playground")
        _print("Type [bold]help[/bold] for commands, [bold]exit[/bold] to quit." if _RICH else "Type 'help' for commands, 'exit' to quit.")
        _print("")
        try:
            while self._running:
                try:
                    line = self._readline()
                except EOFError:
                    _print("\nBye!")
                    break
                line = line.strip()
                if not line:
                    continue
                self._dispatch(line)
        except KeyboardInterrupt:
            _print("\nInterrupted. Bye!")
        return 0

    def _readline(self) -> str:
        prompt_label = f"[{self.session.prompt_name or '(none)'}]> "
        try:
            return input(prompt_label)
        except EOFError:
            raise

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, line: str) -> None:
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        dispatch = {
            "select": self._cmd_select,
            "set": self._cmd_set,
            "unset": self._cmd_unset,
            "render": lambda _: self._cmd_render(),
            "run": self._cmd_run,
            "save": self._cmd_save,
            "load": self._cmd_load,
            "reload": lambda _: self._cmd_reload(),
            "list": self._cmd_list,
            "show": self._cmd_show,
            "help": lambda _: _print(_HELP),
            "exit": lambda _: self._exit(),
            "quit": lambda _: self._exit(),
        }

        handler = dispatch.get(cmd)
        if handler is None:
            _print_err(f"Unknown command '{cmd}'. Type 'help' for a list.")
            return
        handler(rest)

    def _exit(self) -> None:
        self._running = False
        _print("Bye!")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_select(self, rest: str) -> None:
        name = rest.strip()
        if not name:
            _print_err("Usage: select <prompt-name>")
            return
        try:
            p = registry.get(name)
        except KeyError:
            _print_err(f"Prompt '{name}' not found. Try 'list' to see available prompts.")
            return
        self.session.prompt_name = name
        self.session.variables = dict(p.fixture)  # seed with fixture defaults
        _print_ok(f"Selected: {name}")
        _print(f"  Domain: {p.domain}  Variant: {p.variant}")
        _print(f"  Pre-loaded fixture inputs: {list(p.fixture.keys())}")

    def _cmd_set(self, rest: str) -> None:
        parsed = _split_set_args(rest)
        if parsed is None:
            _print_err("Usage: set <key> <value>")
            return
        key, raw_value = parsed
        value = _parse_value(raw_value)
        self.session.set_var(key, value)
        _print_ok(f"Set {key} = {json.dumps(value)}")

    def _cmd_unset(self, rest: str) -> None:
        key = rest.strip()
        if not key:
            _print_err("Usage: unset <key>")
            return
        if self.session.unset_var(key):
            _print_ok(f"Unset {key}")
        else:
            _print_warn(f"Variable '{key}' was not set")

    def _cmd_render(self) -> None:
        if not self.session.prompt_name:
            _print_err("No prompt selected. Use 'select <name>' first.")
            return
        try:
            p = registry.get(self.session.prompt_name)
        except KeyError:
            _print_err(f"Prompt '{self.session.prompt_name}' not in registry")
            return
        try:
            rendered = p.template(**self.session.variables)
        except Exception as exc:
            _print_err(f"Render failed: {exc}")
            return
        self.session.add_render(rendered)
        _print_prompt(rendered)

    def _cmd_run(self, rest: str) -> None:
        if not self.session.prompt_name:
            _print_err("No prompt selected. Use 'select <name>' first.")
            return

        model = self.default_model
        args = rest.split()
        i = 0
        while i < len(args):
            if args[i] in ("--model", "-m") and i + 1 < len(args):
                model = args[i + 1]
                i += 2
            else:
                i += 1

        try:
            p = registry.get(self.session.prompt_name)
        except KeyError:
            _print_err(f"Prompt '{self.session.prompt_name}' not in registry")
            return
        try:
            rendered = p.template(**self.session.variables)
        except Exception as exc:
            _print_err(f"Render failed: {exc}")
            return

        _print(f"Running with model: {model} ...")
        try:
            output = _run_prompt(rendered, model)
        except Exception as exc:
            _print_err(f"Model call failed: {exc}")
            return
        self.session.add_run(model=model, rendered=rendered, output=output)
        _print_output(output, model)

    def _cmd_save(self, rest: str) -> None:
        path_str = rest.strip()
        path = Path(path_str) if path_str else None
        try:
            saved_path = self.session.save(path)
            _print_ok(f"Session saved to: {saved_path}")
        except Exception as exc:
            _print_err(f"Save failed: {exc}")

    def _cmd_load(self, rest: str) -> None:
        path_str = rest.strip()
        if not path_str:
            _print_err("Usage: load <path>")
            return
        path = Path(path_str)
        if not path.exists():
            _print_err(f"File not found: {path}")
            return
        try:
            self.session = PlaygroundSession.load(path)
            _print_ok(f"Session loaded: {self.session.summary()}")
        except Exception as exc:
            _print_err(f"Load failed: {exc}")

    def _cmd_reload(self) -> None:
        if not self.session.prompt_name:
            _print_err("No prompt selected. Use 'select <name>' first.")
            return
        _reload_prompt(self.session.prompt_name)

    def _cmd_list(self, rest: str) -> None:
        domain = None
        args = rest.split()
        i = 0
        while i < len(args):
            if args[i] == "--domain" and i + 1 < len(args):
                domain = args[i + 1]
                i += 2
            else:
                i += 1
        prompts = registry.search(domain=domain)
        if not prompts:
            _print("No prompts registered.")
            return
        _print(f"{'Name':<45} {'Domain':<12} {'Variant':<12} Description")
        _print("-" * 100)
        for p in prompts:
            _print(f"{p.name:<45} {p.domain:<12} {p.variant:<12} {p.description}")
        _print(f"\nTotal: {len(prompts)} prompt(s)")

    def _cmd_show(self, rest: str) -> None:
        name = rest.strip()
        if not name:
            _print_err("Usage: show <prompt-name>")
            return
        try:
            p = registry.get(name)
        except KeyError:
            _print_err(f"Prompt '{name}' not found")
            return
        _print(f"\nName:        {p.name}")
        _print(f"Domain:      {p.domain}")
        _print(f"Variant:     {p.variant}")
        _print(f"Description: {p.description}")
        _print(f"Tags:        {', '.join(p.tags) or '—'}")
        _print("\nInput Schema:")
        _print("  " + json.dumps(p.input_schema, indent=2).replace("\n", "\n  "))
        _print("\nFixture:")
        _print("  " + json.dumps(p.fixture, indent=2).replace("\n", "\n  "))
