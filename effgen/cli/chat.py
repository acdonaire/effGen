"""Interactive chat REPL for ``effgen chat``.

This is the most "alive" surface in the CLI, so it aims to feel like a real
chat: streaming tokens with a thinking spinner, a model/tool-aware prompt,
slash commands (``/model``, ``/tools``, ``/cost``, ``/trace``, …), persistent
↑/↓ history, multiline input, and a graceful per-turn Ctrl-C that cancels the
current answer without dropping you out of the session.

Everything here is presentation/coordination over the existing :class:`Agent`
machinery — it adds no model, provider, or tool behavior. Plain chat (no tools)
streams the model's answer directly for the cleanest live feel; once tools are
attached (via ``/tools``) a turn runs under the shared live-status spinner so
per-tool ticks and an accurate token/cost footer can be shown.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from effgen.cli import progress as _progress

# Slash commands surfaced in /help and tab-completion. Kept in one place so the
# help text, the completer, and the dispatcher never drift apart.
_SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help",
    "/model": "Hot-swap the active model:  /model gpt-5-nano",
    "/tools": "List tools, or toggle one:  /tools calculator",
    "/cost": "Session token + cost total",
    "/reset": "Clear the conversation memory",
    "/save": "Save this conversation:  /save [name]",
    "/load": "Load a saved conversation:  /load [name|number]",
    "/trace": "Show the last turn's reasoning/tool steps",
    "/doctor": "Run a quick environment check",
    "/clear": "Clear the screen",
    "/exit": "Leave chat (also: /quit, exit, quit)",
}


def _history_dir() -> Path:
    """Return ``~/.effgen/history`` (configurable via ``EFFGEN_HOME``)."""
    home = os.environ.get("EFFGEN_HOME")
    base = Path(home) if home else Path.home() / ".effgen"
    d = base / "history"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ChatREPL:
    """A polished interactive chat session over a single :class:`Agent`."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, cli: Any, args: Any):
        self.cli = cli
        self.args = args
        self.console = getattr(cli, "console", None)

        self.quiet = getattr(args, "quiet", False)
        self.verbose = getattr(args, "verbose", False)
        self.animate = cli._animate(args)

        self.model_id = getattr(args, "model", None) or self.DEFAULT_MODEL
        self.preset = getattr(args, "preset", None)
        # Optional custom persona: steers every reply (e.g. a Socratic tutor),
        # unlike --preset which only labels the session. Carried across /model
        # and /tools rebuilds so the persona persists for the whole session.
        self.system_prompt = getattr(args, "system_prompt", None)
        # Optional persistent session: when set, the chat agent loads prior turns
        # for this id and saves new ones (same store as `effgen run --session-id`
        # and `effgen sessions`), so a customer's conversation can be continued.
        self.session_id = getattr(args, "session_id", None)
        self.temperature = getattr(args, "temperature", None)
        self.max_tokens = getattr(args, "max_tokens", None)
        # Provider was validated by the caller; keep the canonical name.
        self.provider = getattr(args, "_provider", None) or getattr(args, "provider", None)

        # Plain chat starts tool-free for the cleanest streaming; users opt into
        # tools with ``/tools <name>``. (This also means ``effgen chat -m <any
        # non-Claude model>`` never trips the native-tool incompatibility that
        # used to crash the command on startup.)
        self.tool_names: list[str] = []

        self.agent: Any = None
        self.session_tokens = 0
        self.session_cost = 0.0
        self.turns = 0
        self.last_trace: list[dict[str, Any]] | None = None
        self.last_answer: str = ""
        self._history_file = _history_dir() / "chat_input_history"

    # ------------------------------------------------------------------
    # Agent construction & tools
    # ------------------------------------------------------------------
    def _load_tools(self, names: list[str]) -> list:
        """Load the named builtin tools and drop any incompatible with the model."""
        from effgen.cli._main import filter_incompatible_tools

        registry = self.cli.tool_registry
        registry.discover_builtin_tools()
        available = set(registry.list_tools())
        tools = []
        for name in names:
            if name not in available:
                continue
            try:
                tools.append(registry.get_tool_sync(name))
            except Exception:  # noqa: BLE001 - a single bad tool shouldn't sink chat
                pass
        tools, _skipped = filter_incompatible_tools(
            tools, self.model_id, warn=self.cli.print_warning
        )
        return tools

    def _build_agent(self, carry_from: Any = None) -> Any:
        """Build the chat Agent, optionally carrying memory from a prior agent."""
        from effgen import Agent, AgentConfig

        tools = self._load_tools(self.tool_names)
        config_kwargs: dict[str, Any] = {
            "name": "chat-agent",
            "model": self.model_id,
            "provider": self.provider,
            "tools": tools,
            "temperature": self.temperature if self.temperature is not None else 0.7,
            "max_tokens": self.max_tokens,
            "enable_sub_agents": not getattr(self.args, "no_sub_agents", False),
            "enable_streaming": True,
        }
        if self.system_prompt:
            config_kwargs["system_prompt"] = self.system_prompt
        config = AgentConfig(**config_kwargs)
        # Attach the persistent session only on the FIRST build, so its saved
        # turns are loaded into memory exactly once. On a /model or /tools rebuild
        # we carry memory from the old agent instead (below) and reuse the same
        # Session object — re-loading from disk would double the history.
        if self.session_id and carry_from is None:
            agent = Agent(config, session_id=self.session_id)
        else:
            agent = Agent(config)

        # Carry the running conversation across a /model or /tools rebuild so the
        # new model still "remembers" what was said, and keep persisting to the
        # same session file.
        if carry_from is not None:
            try:
                from effgen.memory.short_term import MessageRole

                for msg in carry_from.short_term_memory.get_messages():
                    if msg.role == MessageRole.USER:
                        agent.short_term_memory.add_user_message(msg.content)
                    elif msg.role == MessageRole.ASSISTANT:
                        agent.short_term_memory.add_assistant_message(msg.content)
            except Exception:  # noqa: BLE001 - memory carry is best-effort
                pass
            # Reuse the live Session object so new turns keep saving to the same id.
            if getattr(carry_from, "session", None) is not None:
                agent.session = carry_from.session
                agent._session_id = getattr(carry_from, "_session_id", self.session_id)
        return agent

    def _rebuild(self) -> None:
        """Rebuild the agent in place, preserving conversation memory."""
        old = self.agent
        self.agent = self._build_agent(carry_from=old)
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def tool_count(self) -> int:
        agent = self.agent
        if agent is None:
            return 0
        return len(getattr(agent, "tools", {}) or {})

    # ------------------------------------------------------------------
    # Prompt / banner
    # ------------------------------------------------------------------
    def _prompt_str(self) -> str:
        """A model/tool-aware input prompt, e.g. ``math · gpt-5-nano · 1 tool › ``.

        Kept as plain text (no ANSI) so ``readline`` width math and ↑/↓ history
        stay correct across terminals.
        """
        label = _progress.short_model_label(self.model_id)
        bits = []
        if self.preset:
            bits.append(self.preset)
        bits.append(label)
        n = self.tool_count
        if n:
            bits.append(f"{n} tool" + ("s" if n != 1 else ""))
        return " · ".join(bits) + " › "

    def _banner(self) -> None:
        from effgen import __version__

        self.cli.print_header(f"effGen v{__version__} · chat")
        label = _progress.short_model_label(self.model_id)
        meta = f"Model: {label}"
        if self.provider:
            meta += f"  (provider: {self.provider})"
        if self.preset:
            meta = f"Preset: {self.preset}  ·  " + meta
        self.cli.print(meta)
        # Show that we're continuing a named session (and how many turns it has).
        if self.session_id:
            try:
                from effgen.core.session import Session

                prior = len(Session.load_or_create(self.session_id).messages)
            except Exception:  # noqa: BLE001
                prior = 0
            if prior:
                self.cli.print(
                    f"Session: {self.session_id}  ·  resuming {prior} prior message(s)"
                )
            else:
                self.cli.print(f"Session: {self.session_id}  ·  new (will be saved)")
        self.cli.print(
            "Type your message and press Enter.  "
            "End a line with \\ for multi-line input.\n"
            "Slash commands: /help  /model  /tools  /cost  /trace  /reset  "
            "/save  /load  /doctor  /exit"
        )
        # Friendly resume: if there are saved conversations, point at the most
        # recent one so a returning user can pick up where they left off. A
        # single non-spammy line — we never auto-load (that would silently mix
        # an old context into a fresh session).
        try:
            saved = sorted(_history_dir().glob("chat_*.json"), reverse=True)
        except OSError:
            saved = []
        if saved:
            last = saved[0].name[len("chat_"):-len(".json")] or saved[0].name
            self.cli.print(
                f"Resume: {len(saved)} saved — `/load` to list, "
                f"or `/load {last}` for the most recent."
            )

    # ------------------------------------------------------------------
    # readline history + tab completion
    # ------------------------------------------------------------------
    def _setup_readline(self) -> None:
        try:
            import readline
        except Exception:  # noqa: BLE001 - readline absent (e.g. Windows); skip
            return
        try:
            readline.read_history_file(str(self._history_file))
        except (FileNotFoundError, OSError):
            pass
        try:
            readline.set_history_length(1000)
        except Exception:  # noqa: BLE001
            pass

        commands = sorted(_SLASH_COMMANDS)

        def _completer(text: str, state: int) -> str | None:
            if not text.startswith("/"):
                return None
            matches = [c + " " for c in commands if c.startswith(text)]
            return matches[state] if state < len(matches) else None

        try:
            readline.set_completer(_completer)
            readline.parse_and_bind("tab: complete")
        except Exception:  # noqa: BLE001
            pass

    def _save_readline(self) -> None:
        try:
            import readline

            readline.write_history_file(str(self._history_file))
        except Exception:  # noqa: BLE001
            pass

    def _read_input(self) -> str | None:
        """Read one (possibly multi-line) user entry. ``None`` on EOF."""
        prompt = self._prompt_str()
        lines: list[str] = []
        while True:
            try:
                line = input(prompt if not lines else "… ")
            except EOFError:
                return None
            if line.endswith("\\"):
                lines.append(line[:-1])
                continue
            lines.append(line)
            break
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        self._banner()
        self._setup_readline()
        try:
            self.agent = self._build_agent()
        except Exception as e:  # noqa: BLE001
            self.cli.print_error(f"Could not start chat: {e}")
            self._teach_model_error(e)
            return 1

        try:
            while True:
                try:
                    user_input = self._read_input()
                except KeyboardInterrupt:
                    # Ctrl-C at an empty prompt: clear the line, keep the session.
                    self.cli.print("")
                    continue
                if user_input is None:  # EOF / Ctrl-D
                    self.cli.print("\nGoodbye!")
                    break
                if not user_input.strip():
                    continue

                action = self._dispatch(user_input.strip())
                if action == "exit":
                    self.cli.print("Goodbye!")
                    break
                if action == "handled":
                    continue

                self._do_turn(user_input)
        finally:
            self._save_readline()
            if self.agent is not None:
                try:
                    self.agent.close()
                except Exception:  # noqa: BLE001
                    pass
        return 0

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, text: str) -> str | None:
        """Route a line. Returns 'exit', 'handled', or None (treat as a message)."""
        low = text.lower()
        # Back-compat bare words from the old chat.
        if low in ("exit", "quit"):
            return "exit"
        if low == "help":
            self._cmd_help()
            return "handled"
        if low == "clear":
            self._cmd_reset()
            return "handled"
        if low in ("history", "sessions"):
            self._list_saved()
            return "handled"

        if not text.startswith("/"):
            return None

        parts = text[1:].split(maxsplit=1)
        cmd = ("/" + parts[0]).lower() if parts else "/"
        arg = parts[1].strip() if len(parts) > 1 else ""

        handlers = {
            "/help": lambda: self._cmd_help(),
            "/exit": lambda: "exit",
            "/quit": lambda: "exit",
            "/model": lambda: self._cmd_model(arg),
            "/tools": lambda: self._cmd_tools(arg),
            "/cost": lambda: self._cmd_cost(),
            "/reset": lambda: self._cmd_reset(),
            "/save": lambda: self._cmd_save(arg),
            "/load": lambda: self._cmd_load(arg),
            "/trace": lambda: self._cmd_trace(),
            "/doctor": lambda: self._cmd_doctor(),
            "/clear": lambda: self._cmd_clear_screen(),
        }
        handler = handlers.get(cmd)
        if handler is None:
            self.cli.print_warning(
                f"Unknown command '{text.split()[0]}'. Type /help for the list."
            )
            return "handled"
        result = handler()
        return result if result == "exit" else "handled"

    # ------------------------------------------------------------------
    # A turn
    # ------------------------------------------------------------------
    def _snapshot(self) -> tuple[int, float]:
        m = getattr(self.agent, "model", None)
        return (
            int(getattr(m, "total_tokens", 0) or 0),
            float(getattr(m, "total_cost", 0.0) or 0.0),
        )

    def _do_turn(self, user_input: str) -> None:
        tok0, cost0 = self._snapshot()
        t0 = time.monotonic()
        answer = ""
        interrupted = False
        self.last_trace = None

        try:
            if self.tool_count > 0:
                # agent.run() persists to the session itself.
                answer = self._run_with_tools(user_input)
            else:
                answer = self._stream_plain(user_input)
                # The streaming path bypasses agent.run(), so persist the turn to
                # the session ourselves when one is attached.
                self._persist_session_turn(user_input, answer)
        except KeyboardInterrupt:
            interrupted = True
            self.cli.print("")
            if self.console:
                self.console.print("[yellow]Stopped.[/yellow]")
            else:
                print("Stopped.")
        except Exception as e:  # noqa: BLE001
            self.cli.print_error(f"{e}")
            self._teach_model_error(e)
            return

        self.last_answer = answer
        self.turns += 1

        # Per-turn footer: · 1.2s · 318 tok · $0.0003 (accurate about what we know).
        elapsed = time.monotonic() - t0
        tok1, cost1 = self._snapshot()
        dtok = tok1 - tok0
        dcost = cost1 - cost0
        if dtok <= 0 and answer:
            dtok = self._count_tokens(answer)  # local models: count the output
        self.session_tokens += max(dtok, 0)
        self.session_cost += max(dcost, 0.0)
        if not self.quiet:
            self._print_footer(elapsed, dtok, dcost, interrupted)

    def _persist_session_turn(self, user_input: str, answer: str) -> None:
        """Append a plain (streamed) turn to the persistent session, if any."""
        session = getattr(self.agent, "session", None)
        if session is None or not answer:
            return
        try:
            session.add_user_message(user_input)
            session.add_assistant_message(answer)
            session.save()
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass

    def _stream_plain(self, user_input: str) -> str:
        """Stream the model's answer token-by-token with a thinking spinner."""
        gen = iter(self.agent.stream(user_input))
        first = None

        if self.animate and self.console:
            from rich.progress import Progress, SpinnerColumn, TextColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True,
            ) as progress:
                progress.add_task("Thinking…", total=None)
                for tok in gen:
                    if tok:
                        first = tok
                        break
        else:
            if not self.quiet:
                print("Thinking…", end="", flush=True)
            for tok in gen:
                if tok:
                    first = tok
                    break
            if not self.quiet:
                print("\r" + " " * 10 + "\r", end="", flush=True)

        if self.console:
            self.console.print("[bold green]assistant[/bold green] ", end="")
        else:
            print("assistant ", end="", flush=True)

        collected: list[str] = []
        if first:
            print(first, end="", flush=True)
            collected.append(first)
        for tok in gen:
            print(tok, end="", flush=True)
            collected.append(tok)
        print()
        return "".join(collected).strip()

    def _run_with_tools(self, user_input: str) -> str:
        """Run a tool-enabled turn under the live-status spinner, then show the answer."""
        from effgen.core.agent import AgentMode

        if self.animate and self.console:
            reasoning = _progress.is_reasoning_agent(self.agent)
            with _progress.LiveStatus(
                self.console,
                model_label=_progress.short_model_label(self.model_id),
                reasoning=reasoning,
                tracker=self.agent.execution_tracker,
            ):
                response = self.agent.run(user_input, mode=AgentMode.AUTO)
        else:
            if not self.quiet:
                print("Thinking…")
            response = self.agent.run(user_input, mode=AgentMode.AUTO)

        try:
            self.last_trace = response.execution_trace or None
        except Exception:  # noqa: BLE001
            self.last_trace = None

        answer = response.output or ""
        if self.console:
            from rich.markdown import Markdown

            self.console.print("[bold green]assistant[/bold green] ", end="")
            self.console.print(Markdown(answer))
        else:
            print(f"assistant {answer}")

        # Prefer the response's own accounting for the footer when available.
        meta = response.metadata or {}
        self._turn_response_tokens = int(getattr(response, "tokens_used", 0) or 0)
        self._turn_response_cost = _progress._extract_cost(meta) or 0.0
        if not response.success:
            reason = meta.get("reason", "failed")
            self.cli.print_warning(f"(turn did not fully succeed: {reason})")
        return answer

    def _count_tokens(self, text: str) -> int:
        try:
            return int(self.agent.model.count_tokens(text).count)
        except Exception:  # noqa: BLE001
            return 0

    def _print_footer(
        self, elapsed: float, dtok: int, dcost: float, interrupted: bool
    ) -> None:
        # The tool path has exact per-turn accounting from the response; prefer it.
        rtok = getattr(self, "_turn_response_tokens", 0)
        rcost = getattr(self, "_turn_response_cost", 0.0)
        self._turn_response_tokens = 0
        self._turn_response_cost = 0.0
        tok = rtok or max(dtok, 0)
        cost = rcost or max(dcost, 0.0)

        parts = [f"{elapsed:.1f}s"]
        if tok:
            parts.append(f"{tok:,} tok")
        if cost > 0:
            from effgen.ui.render import format_cost
            parts.append(format_cost(cost))
        footer = "· " + " · ".join(parts)
        if interrupted:
            footer += " · stopped"
        if self.console:
            self.console.print(f"[dim]{footer}[/dim]")
        else:
            print(footer)

    # ------------------------------------------------------------------
    # Slash command implementations
    # ------------------------------------------------------------------
    def _cmd_help(self) -> None:
        if self.console:
            from rich.table import Table

            table = Table(title="Chat commands", show_header=True, header_style="cyan")
            table.add_column("Command", style="green", no_wrap=True)
            table.add_column("Does")
            for cmd, desc in _SLASH_COMMANDS.items():
                table.add_row(cmd, desc)
            self.console.print(table)
        else:
            print("Chat commands:")
            for cmd, desc in _SLASH_COMMANDS.items():
                print(f"  {cmd:9s} {desc}")

    def _resolve_swap_target(self, new_id: str) -> tuple[str | None, str]:
        """Resolve a ``/model`` argument the same way a fresh ``chat -m <id>``
        (no ``--provider``) would.

        A ``provider:model`` prefix pins that provider explicitly, mirroring
        startup resolution. A bare id clears any provider left over from a
        prior explicit ``--provider`` flag, so the loader's own auto-detection
        (catalog lookup / known-prefix matching) picks the right adapter
        instead of forcing the new id through the *previous* session's
        provider — which silently keeps the old adapter and fails every turn.
        """
        if ":" in new_id:
            from effgen.models.registry import ProviderRegistry

            prefix, rest = new_id.split(":", 1)
            try:
                known = ProviderRegistry.list_providers()
            except Exception:  # noqa: BLE001
                known = []
            if prefix in known and rest:
                return prefix, rest
        return None, new_id

    def _cmd_model(self, arg: str) -> None:
        if not arg:
            self.cli.print(f"Active model: {self.model_id}")
            self.cli.print("Swap with:  /model <model-id>   (e.g. /model gpt-5-nano)")
            return
        new_id = arg.split()[0]
        old_id = self.model_id
        old_provider = self.provider
        new_provider, resolved_id = self._resolve_swap_target(new_id)
        self.model_id = resolved_id
        self.provider = new_provider
        # A failed swap is fully handled below (a styled ✗ message + the
        # restore rebuild) — suppress the library's own ERROR-level log line
        # for it so the transcript shows one failure presentation, not two.
        # --verbose still surfaces it.
        _effgen_logger = logging.getLogger("effgen")
        _prev_level = _effgen_logger.level
        if not self.verbose:
            _effgen_logger.setLevel(logging.CRITICAL)
        try:
            self._rebuild()
        except Exception as e:  # noqa: BLE001
            self.model_id = old_id
            self.provider = old_provider
            self.cli.print_error(f"Could not switch to '{new_id}': {e}")
            self._teach_model_error(e)
            # Restore the working agent.
            try:
                self._rebuild()
            except Exception:  # noqa: BLE001
                pass
            return
        finally:
            _effgen_logger.setLevel(_prev_level)
        self.cli.print_success(
            f"Switched model: {_progress.short_model_label(new_id)} "
            f"(conversation kept)"
        )

    def _cmd_tools(self, arg: str) -> None:
        registry = self.cli.tool_registry
        registry.discover_builtin_tools()
        if not arg:
            active = sorted(getattr(self.agent, "tools", {}) or {})
            if active:
                self.cli.print(f"Active tools ({len(active)}): {', '.join(active)}")
            else:
                self.cli.print("No tools active. Add one:  /tools calculator")
            sample = ", ".join(sorted(registry.list_tools())[:12])
            self.cli.print(f"Available (sample): {sample} …  (effgen tools list)")
            return
        name = arg.split()[0]
        if name not in set(registry.list_tools()):
            self.cli.print_warning(f"No tool named '{name}'. See: effgen tools list")
            return
        if name in self.tool_names:
            self.tool_names.remove(name)
            self._rebuild()
            self.cli.print_success(f"Removed tool '{name}' ({self.tool_count} active)")
        else:
            self.tool_names.append(name)
            before = self.tool_count
            self._rebuild()
            if self.tool_count <= before and name not in (getattr(self.agent, "tools", {}) or {}):
                # filter_incompatible_tools dropped it for this model.
                self.tool_names.remove(name)
                self.cli.print_warning(
                    f"Tool '{name}' is not compatible with {self.model_id}."
                )
            else:
                self.cli.print_success(f"Added tool '{name}' ({self.tool_count} active)")

    def _cmd_cost(self) -> None:
        self.cli.print_header("Session cost")
        from effgen.ui.render import format_cost

        self.cli.print(f"Turns: {self.turns}")
        self.cli.print(f"Tokens: {self.session_tokens:,}")
        # A $0.00 here is accurate for free/local sessions; for cheap cloud turns
        # the adaptive formatter keeps sub-cent spend visible instead of $0.0000.
        self.cli.print(f"Cost: {format_cost(self.session_cost) if self.session_cost else '$0.00'}")
        # Cross-check against the process-wide cost tracker when keyed providers
        # reported real usage (local models stay $0).
        try:
            from effgen.models._cost import CostTracker

            tracker = CostTracker.get()
            total = tracker.total_cost()
            if total > 0:
                self.cli.print(f"(process total across all models: {format_cost(total)})")
        except Exception:  # noqa: BLE001
            pass

    def _cmd_reset(self) -> None:
        try:
            self.agent.reset_memory()
        except Exception:  # noqa: BLE001
            pass
        self.last_trace = None
        self.last_answer = ""
        self.cli.print_success("Conversation memory cleared.")

    def _cmd_clear_screen(self) -> None:
        try:
            os.system("cls" if os.name == "nt" else "clear")  # noqa: S605,S607
        except Exception:  # noqa: BLE001
            pass

    def _cmd_save(self, arg: str) -> None:
        history = self._dump_history()
        if not history:
            self.cli.print("Nothing to save yet.")
            return
        name = arg.split()[0] if arg else datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name.endswith(".json"):
            name += ".json"
        path = _history_dir() / f"chat_{name}"
        with open(path, "w") as f:
            json.dump(
                {"model": self.model_id, "tools": self.tool_names, "messages": history},
                f,
                indent=2,
            )
        self.cli.print_success(f"Saved to {path}")

    def _cmd_load(self, arg: str) -> None:
        files = sorted(_history_dir().glob("chat_*.json"), reverse=True)
        if not files:
            self.cli.print("No saved conversations found.")
            return
        target: Path | None = None
        if not arg:
            self._list_saved()
            self.cli.print("Load with:  /load <name|number>")
            return
        token = arg.split()[0]
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(files):
                target = files[idx]
        else:
            cand = token if token.endswith(".json") else token + ".json"
            for f in files:
                if f.name == cand or f.name == f"chat_{cand}":
                    target = f
                    break
        if target is None:
            self.cli.print_warning(f"No saved conversation matching '{token}'.")
            return
        try:
            with open(target) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.cli.print_error(f"Could not read {target.name}: {e}")
            return
        messages = data.get("messages", data if isinstance(data, list) else [])
        saved_model = data.get("model")
        if saved_model and saved_model != self.model_id:
            self.model_id = saved_model
        self.tool_names = list(data.get("tools", []) or [])
        self._rebuild()
        # Replay the saved turns into memory.
        try:
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    self.agent.short_term_memory.add_user_message(content)
                elif role in ("assistant", "agent"):
                    self.agent.short_term_memory.add_assistant_message(content)
        except Exception:  # noqa: BLE001
            pass
        self.cli.print_success(
            f"Loaded {target.name} ({len(messages)} messages, model {self.model_id})."
        )

    def _cmd_trace(self) -> None:
        if not self.last_trace:
            if self.last_answer:
                self.cli.print(
                    "Last turn was a direct streamed answer (no tool steps). "
                    "Add a tool with /tools to see reasoning steps."
                )
            else:
                self.cli.print("No turn to trace yet.")
            return
        self.cli.print_header("Last turn — reasoning trace")
        lines = _progress.execution_trace_lines(self.last_trace)
        if not lines:
            self.cli.print("(no detailed steps recorded for this turn)")
            return
        for style, text in lines:
            if self.console:
                self.console.print(f"[{style}]{text}[/{style}]")
            else:
                print(text)

    def _cmd_doctor(self) -> None:
        from effgen.cli._main import _handle_doctor_command

        class _DoctorArgs:
            doctor_provider = self.provider
            live = False
            output_json = False

        try:
            _handle_doctor_command(_DoctorArgs())
        except Exception as e:  # noqa: BLE001
            self.cli.print_error(f"doctor failed: {e}")

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def _dump_history(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        try:
            from effgen.memory.short_term import MessageRole

            for msg in self.agent.short_term_memory.get_messages():
                role = "user" if msg.role == MessageRole.USER else (
                    "assistant" if msg.role == MessageRole.ASSISTANT else "system"
                )
                out.append({"role": role, "content": msg.content})
        except Exception:  # noqa: BLE001
            pass
        return out

    def _list_saved(self) -> None:
        files = sorted(_history_dir().glob("chat_*.json"), reverse=True)
        if not files:
            self.cli.print("No saved conversations found.")
            return
        self.cli.print("Saved conversations:")
        for i, f in enumerate(files[:20], 1):
            self.cli.print(f"  {i}. {f.name}  ({f.stat().st_size} bytes)")

    # ------------------------------------------------------------------
    # Friendly model errors
    # ------------------------------------------------------------------
    def _teach_model_error(self, exc: Exception) -> None:
        msg = str(exc).lower()
        if "not found" in msg or "404" in msg or "model_not_found" in msg:
            self.cli.print(
                "Tip: check the id with `effgen models list`, or hot-swap here "
                "with `/model <id>`."
            )
        elif "api key" in msg or "auth" in msg or "401" in msg:
            self.cli.print(
                "Tip: run `effgen doctor` (or `/doctor`) to see which provider "
                "keys are set."
            )
