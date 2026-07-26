"""Interactive chat REPL for ``effgen chat``.

This is the most "alive" surface in the CLI, so it aims to feel like a real
chat: streaming tokens with a thinking spinner, a model/tool-aware prompt,
slash commands (``/model``, ``/tools``, ``/status``, ``/cost``, ``/trace``, …),
persistent ↑/↓ history, multiline input, and a per-turn Ctrl-C that cancels the
current answer without dropping you out of the session.

Everything here is presentation/coordination over the existing :class:`Agent`
machinery — it adds no model, provider, or tool behavior. Plain chat (no tools)
streams the model's answer directly for the cleanest live feel; once tools are
attached (via ``/tools``) a turn runs under the shared live-status spinner so
per-tool ticks and an accurate token/cost footer can be shown.

A non-interactive invocation (``echo "hi" | effgen chat``) keeps a plain
fallback: the interactive chrome (banner, prompt echo, ``assistant`` label,
footer, "Goodbye!") is kept off stdout so the answer is the only thing there,
with ``-q`` yielding one answer per input line that a caller can parse directly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from effgen.cli import progress as _progress
from effgen.ui.theme import color_enabled

# Slash commands surfaced in /help and tab-completion. Kept in one place so the
# help text, the completer, and the dispatcher never drift apart.
_SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help (or type just / for the menu)",
    "/model": "Hot-swap the active model:  /model gpt-5-nano",
    "/tools": "List tools, or toggle one:  /tools calculator",
    "/status": "Show the session state: model, persona, tools, running totals",
    "/cost": "Session token + cost total",
    "/reset": "Clear the conversation memory",
    "/save": "Save this chat to a file:  /save [name]  (see /session to resume)",
    "/session": "Show or name the resumable session id (used by `effgen sessions`)",
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
    """An interactive chat session over a single :class:`Agent`."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, cli: Any, args: Any) -> None:
        self.cli = cli
        self.args = args
        self.console = getattr(cli, "console", None)

        self.quiet = getattr(args, "quiet", False)
        self.verbose = getattr(args, "verbose", False)
        self.animate = cli._animate(args)
        # Whether styled output (bold/dim/color SGR) is allowed. ``NO_COLOR``
        # turns this off, and the REPL then emits its own surfaces (banner,
        # answer label, footer, /help) as plain text with no escape codes.
        self._color = color_enabled()
        # A piped session (``echo hi | effgen chat``) has no terminal, so the
        # interactive chrome is kept off stdout and the answer is emitted plainly.
        try:
            self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        except (ValueError, OSError):  # pragma: no cover - closed std streams
            self.interactive = False
        # Route all human-facing chatter (banner, warnings, footer, "Goodbye!")
        # to stderr when there is no terminal, so stdout carries only answers and
        # `echo … | effgen chat -q` is directly parseable.
        if not self.interactive:
            try:
                self.cli._human_to_stderr = True
            except Exception:  # noqa: BLE001 - CLI without the stderr routing hook
                pass

        # Whether the model was chosen by the user or fell back to the local
        # default (drives a one-line first-run download note in the banner).
        self._model_defaulted = getattr(args, "model", None) is None
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
        # Optional guardrail preset name (e.g. "phi", "strict") applied to
        # every turn; None matches the previous no-guardrails default.
        self.guardrails = getattr(args, "guardrails", None)
        # Provider was validated by the caller; keep the canonical name.
        self.provider = getattr(args, "_provider", None) or getattr(args, "provider", None)

        # A tool-bearing preset attaches its tools by default, mirroring
        # `effgen run --preset` (a preset chosen for its tools must actually
        # route to them, not just label the session). Plain chat with no
        # --preset still starts tool-free for the cleanest streaming; users opt
        # into tools with ``/tools <name>``. Incompatible-tool filtering below
        # (`_load_tools` -> `filter_incompatible_tools`) still applies, so this
        # never trips the native-tool incompatibility for a non-Claude model.
        self.tool_names: list[str] = []
        self._preset_system_prompt: str | None = None
        self._preset_temperature: float | None = None
        if self.preset:
            try:
                from effgen.presets import get_preset

                _preset_cfg = get_preset(self.preset)
            except Exception:  # noqa: BLE001 - fall back to label-only on lookup failure
                pass
            else:
                self.tool_names = list(_preset_cfg.tool_names)
                self._preset_system_prompt = _preset_cfg.system_prompt
                self._preset_temperature = _preset_cfg.temperature

        # Tools named with --tools/-t start attached (in addition to any the
        # preset brought), matching `effgen run --tools`. An unknown name is
        # reported with a close-match hint rather than silently ignored.
        requested_tools = getattr(args, "tools", None) or []
        if requested_tools:
            try:
                registry = self.cli.tool_registry
                registry.discover_builtin_tools()
                available = set(registry.list_tools())
            except Exception:  # noqa: BLE001 - never let tool discovery sink startup
                available = set()
            for name in requested_tools:
                if available and name not in available:
                    import difflib

                    hint = difflib.get_close_matches(name, sorted(available), n=1)
                    suffix = f" Did you mean: {hint[0]}?" if hint else ""
                    self.cli.print_warning(
                        f"No tool named '{name}'.{suffix} See: effgen tools list"
                    )
                    continue
                if name not in self.tool_names:
                    self.tool_names.append(name)

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
            "temperature": (
                self.temperature
                if self.temperature is not None
                else (
                    self._preset_temperature if self._preset_temperature is not None else 0.7
                )
            ),
            "max_tokens": self.max_tokens,
            "enable_sub_agents": not getattr(self.args, "no_sub_agents", False),
            "enable_streaming": True,
            "guardrails": self.guardrails,
        }
        # An explicit --system-prompt/--persona always wins; otherwise a
        # tool-bearing preset's own system prompt applies (it tells the model
        # when to use the preset's tools, the same as `effgen run --preset`).
        if self.system_prompt:
            config_kwargs["system_prompt"] = self.system_prompt
        elif self._preset_system_prompt:
            config_kwargs["system_prompt"] = self._preset_system_prompt
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
        """Number of tools attached to the running agent (0 before startup)."""
        agent = self.agent
        if agent is None:
            return 0
        return len(getattr(agent, "tools", {}) or {})

    # ------------------------------------------------------------------
    # Output helpers (color-aware, non-TTY-aware)
    # ------------------------------------------------------------------
    def _chrome_console(self) -> Any:
        """The console human-facing chrome should print to (stderr when piped)."""
        human = getattr(self.cli, "_human", None)
        if callable(human):
            try:
                return human()
            except Exception:  # noqa: BLE001
                pass
        return getattr(self.cli, "console", None)

    def _human_stream(self) -> Any:
        """The text stream human output goes to, for the typography fallback.

        A CLI stand-in need not carry the hook; fall back to ``sys.stdout``, which
        is where the prompt and banner are written when it is absent.
        """
        getter = getattr(self.cli, "_human_stream", None)
        return getter() if callable(getter) else sys.stdout

    def _banner_line(self, text: str, style: str | None = None) -> None:
        """Print one banner line without number/repr auto-highlighting.

        Styling is applied only when color is enabled, so versions/counts read
        as one token and a ``NO_COLOR`` banner carries no escape codes. The text
        is ASCII-folded when the console cannot encode the separators, so a
        non-UTF-8 terminal prints the banner instead of failing on it.
        """
        from effgen.ui.render import ascii_fold

        text = ascii_fold(text, self._human_stream())
        console = getattr(self.cli, "console", None)
        if console:
            markup = f"[{style}]{text}[/{style}]" if (style and self._color) else text
            console.print(markup, highlight=False)
        else:
            self.cli.print(text)

    # ------------------------------------------------------------------
    # Prompt / banner
    # ------------------------------------------------------------------
    def _prompt_str(self) -> str:
        """A model/tool-aware input prompt, e.g. ``math · gpt-5-nano · 1 tool › ``.

        Kept as plain text (no ANSI) so ``readline`` width math and ↑/↓ history
        stay correct across terminals, and ASCII-folded when the terminal cannot
        encode the separators.
        """
        from effgen.ui.render import ascii_fold

        label = _progress.short_model_label(self.model_id)
        bits = []
        if self.preset:
            bits.append(self.preset)
        bits.append(label)
        n = self.tool_count
        if n:
            bits.append(f"{n} tool" + ("s" if n != 1 else ""))
        return ascii_fold(" · ".join(bits) + " › ", self._human_stream())

    def _banner(self) -> None:
        from effgen import __version__

        self._banner_line(f"\neffGen v{__version__} · chat", style="bold cyan")
        label = _progress.short_model_label(self.model_id)
        meta = f"Model: {label}"
        if self.provider:
            meta += f"  (provider: {self.provider})"
        if self.preset:
            meta = f"Preset: {self.preset}  ·  " + meta
        self._banner_line(meta)
        if self.system_prompt:
            self._banner_line("Persona: custom (steers every reply)")
        if self.tool_names:
            self._banner_line(f"Tools: {', '.join(self.tool_names)}")
        if self.guardrails:
            self._banner_line(f"Guardrails: {self.guardrails}")
        # Show that we're continuing a named session (and how many turns it has).
        if self.session_id:
            try:
                from effgen.core.session import Session

                prior = len(Session.load_or_create(self.session_id).messages)
            except Exception:  # noqa: BLE001
                prior = 0
            if prior:
                self._banner_line(
                    f"Session: {self.session_id}  ·  resuming {prior} prior message(s)"
                )
            else:
                self._banner_line(f"Session: {self.session_id}  ·  new (will be saved)")
        self._banner_line(
            "Type your message and press Enter.  "
            "End a line with \\ for multi-line input.\n"
            "Slash commands (type / for the menu): /help  /model  /tools  /status  "
            "/cost  /trace  /reset  /save  /session  /load  /doctor  /exit"
        )
        # A first-timer running bare `effgen chat` gets a local model by default;
        # name the download and offer a fast keyed alternative so the wait is not
        # a surprise.
        if self._model_defaulted and "/" in self.model_id and ":" not in self.model_id:
            self._banner_line(
                f"Loading a local model ({self.model_id}); the first run downloads it. "
                "For a fast keyed model:  effgen chat -m groq:llama-3.1-8b-instant",
                style="dim",
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
            self._banner_line(
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
        from effgen.ui.render import ascii_fold

        # A piped session shows no prompt so it can't leak onto answer-only stdout.
        prompt = self._prompt_str() if self.interactive else ""
        cont = ascii_fold("… ", self._human_stream()) if self.interactive else ""
        lines: list[str] = []
        while True:
            try:
                line = input(prompt if not lines else cont)
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
        """Run the chat loop until exit; returns the process exit code."""
        # The banner is interactive chrome; a piped run stays answer-only.
        if self.interactive:
            self._banner()
        self._setup_readline()
        try:
            self.agent = self._build_agent()
        except Exception as e:  # noqa: BLE001
            hint = self._unknown_model_message(self.model_id, e)
            if hint:
                self.cli.print_error(f"Could not start chat: {hint}")
            else:
                self.cli.print_error(f"Could not start chat: {e}")
                self._teach_model_error(e)
            return 1

        try:
            while True:
                try:
                    user_input = self._read_input()
                except KeyboardInterrupt:
                    # Ctrl-C at an empty prompt: clear the line, keep the session.
                    if self.interactive:
                        self.cli.print("")
                    continue
                if user_input is None:  # EOF / Ctrl-D
                    if self.interactive:
                        self.cli.print("\nGoodbye!")
                    break
                if not user_input.strip():
                    continue

                action = self._dispatch(user_input.strip())
                if action == "exit":
                    if self.interactive:
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

        # A bare "/" opens the command menu, so slash commands stay discoverable
        # after the start banner scrolls away.
        if text.strip() == "/":
            self._cmd_help()
            return "handled"

        parts = text[1:].split(maxsplit=1)
        cmd = ("/" + parts[0]).lower() if parts else "/"
        arg = parts[1].strip() if len(parts) > 1 else ""

        handlers = {
            "/help": lambda: self._cmd_help(),
            "/exit": lambda: "exit",
            "/quit": lambda: "exit",
            "/model": lambda: self._cmd_model(arg),
            "/tools": lambda: self._cmd_tools(arg),
            "/status": lambda: self._cmd_status(),
            "/session": lambda: self._cmd_session(arg),
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
            import difflib

            typed = text.split()[0]
            close = difflib.get_close_matches(cmd, list(_SLASH_COMMANDS), n=3)
            hint = (
                f" Did you mean: {', '.join(close)}?"
                if close
                else " Type /help for the list."
            )
            self.cli.print_warning(f"Unknown command '{typed}'.{hint}")
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
        plain_turn = False
        self.last_trace = None

        try:
            if self.tool_count > 0:
                # agent.run() persists to the session itself.
                answer = self._run_with_tools(user_input)
            else:
                answer = self._stream_plain(user_input)
                # The streaming path bypasses agent.run(), so the turn is
                # persisted below once its tokens/cost/latency are known.
                plain_turn = True
        except KeyboardInterrupt:
            interrupted = True
            self.cli.print("")
            if self._color and self.console:
                self.console.print("[yellow]Stopped.[/yellow]")
            else:
                self.cli.print("Stopped.")
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
        if plain_turn and not interrupted:
            self._persist_session_turn(
                user_input, answer,
                tokens=max(dtok, 0), cost=max(dcost, 0.0), elapsed=elapsed,
            )
        if not self.quiet:
            self._print_footer(elapsed, dtok, dcost, interrupted)

    def _persist_session_turn(
        self,
        user_input: str,
        answer: str,
        *,
        tokens: int = 0,
        cost: float = 0.0,
        elapsed: float | None = None,
    ) -> None:
        """Append a plain (streamed) turn to the persistent session, if any.

        The turn carries the model, token count, cost and latency it was
        answered with, matching what `agent.run()` stores for tool turns.
        """
        session = getattr(self.agent, "session", None)
        if session is None or not answer:
            return
        meta: dict[str, object] = {"model": self.model_id}
        if tokens:
            meta["tokens_used"] = tokens
        if cost:
            meta["cost_usd"] = round(cost, 8)
        if elapsed is not None:
            meta["latency_ms"] = round(elapsed * 1000, 1)
        try:
            session.add_message("user", user_input, **meta)
            session.add_message("assistant", answer, **meta)
            session.metadata["model"] = self.model_id
            session.save()
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass

    # ------------------------------------------------------------------
    # Answer rendering
    # ------------------------------------------------------------------
    def _show_answer(self, answer: str) -> None:
        """Print a finished answer, rendering markdown when a rich console is up."""
        from effgen.ui.render import answer_surface

        answer_surface(answer, framed=False, label="assistant", console=self.console)

    def _stream_plain(self, user_input: str) -> str:
        """Stream the model's answer, rendering markdown the way tool turns do.

        A piped/non-TTY run emits the raw answer only; an interactive terminal
        renders lists/headings/code the same whether or not a tool ran — a live
        markdown region on a colour TTY, a single rendered block otherwise.
        """
        gen = self.agent.stream(user_input)
        if not self.interactive:
            return _progress.stream_answer(
                self.console, gen,
                animate=False, interactive=False, quiet=self.quiet,
                trailing_newline=True,
            ).strip()

        return _progress.stream_answer(
            self.console, gen,
            animate=bool(self.console and self.animate and self._color),
            interactive=True, quiet=self.quiet,
            label="assistant", render_plain=True,
        ).strip()

    def _run_with_tools(self, user_input: str) -> str:
        """Run a tool-enabled turn under the live-status spinner, then show the answer.

        The turn uses the agent's own configured mode (single-agent unless the
        caller set another), the same default `effgen run` uses. Forcing
        automatic routing here would let the router decompose an ordinary
        question into sub-agents, which costs several extra model calls and
        answers from a synthesis step rather than from the tool result.
        """
        if self.animate and self.console:
            reasoning = _progress.is_reasoning_agent(self.agent)
            with _progress.LiveStatus(
                self.console,
                model_label=_progress.short_model_label(self.model_id),
                reasoning=reasoning,
                tracker=self.agent.execution_tracker,
                hint="Ctrl-C to cancel",
            ):
                response = self.agent.run(user_input)
        else:
            if not self.quiet and self.interactive:
                print("Thinking…")
            response = self.agent.run(user_input)

        try:
            self.last_trace = response.execution_trace or None
        except Exception:  # noqa: BLE001
            self.last_trace = None

        answer = response.output or ""
        if not self.interactive:
            sys.stdout.write((answer or "") + "\n")
            sys.stdout.flush()
        else:
            self._show_answer(answer)

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

    def _footer_text(
        self, elapsed: float, tok: int, cost: float, interrupted: bool
    ) -> str:
        """Build the per-turn footer, appending a running session total over time.

        Shares the one summary builder with ``effgen run``; the ``chat`` preset
        keeps the compact per-turn shape and the running session total.
        """
        from effgen.ui.render import summary_line

        footer, _ = summary_line(
            mode="chat",
            elapsed=elapsed,
            tokens=tok,
            cost=cost,
            interrupted=interrupted,
            session=(self.turns, self.session_tokens, self.session_cost),
        )
        return footer

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

        from effgen.ui.render import ascii_fold

        footer = ascii_fold(self._footer_text(elapsed, tok, cost, interrupted),
                            self.cli._human_stream())
        console = self._chrome_console()
        if console:
            console.print(
                f"[dim]{footer}[/dim]" if self._color else footer, highlight=False
            )
        else:
            print(footer, file=None if self.interactive else sys.stderr)

    # ------------------------------------------------------------------
    # Slash command implementations
    # ------------------------------------------------------------------
    def _cmd_help(self) -> None:
        # A rich table only on an interactive color terminal; a piped or
        # NO_COLOR run gets plain lines with no box-drawing or escape codes.
        if self.console and self.interactive and self._color:
            from rich.table import Table

            table = Table(title="Chat commands", show_header=True, header_style="cyan")
            table.add_column("Command", style="green", no_wrap=True)
            table.add_column("Does")
            for cmd, desc in _SLASH_COMMANDS.items():
                table.add_row(cmd, desc)
            self.console.print(table)
        else:
            self.cli.print("Chat commands:")
            for cmd, desc in _SLASH_COMMANDS.items():
                self.cli.print(f"  {cmd:9s} {desc}")

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

    def _unknown_model_message(self, attempted: str, exc: Exception) -> str | None:
        """Return a teach message when *attempted* resolves to no known model.

        A bare id (no ``provider:`` prefix, not a ``org/repo`` path) that fails
        to load is almost always a mistyped/nonexistent model rather than an
        auth or download problem, so it earns a "check the id / name a provider"
        hint instead of the loader's Hugging Face token message.
        """
        if ":" in attempted or "/" in attempted:
            return None
        m = str(exc).lower()
        markers = (
            "not a valid model identifier",
            "is not a local folder",
            "not found",
            "404",
            "model_not_found",
            "no model",
        )
        if any(s in m for s in markers):
            return (
                f"No model '{attempted}'. Check the id with `effgen models list`, "
                "or name a provider with `/model provider:id` "
                "(e.g. /model openai:gpt-5-nano)."
            )
        return None

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
            # An unresolvable id gets a "check the id / name a provider" hint
            # rather than the loader's raw download/auth message.
            hint = self._unknown_model_message(new_id, e)
            if hint:
                self.cli.print_error(hint)
            else:
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

    def _cmd_status(self) -> None:
        """Show the session state a long chat needs to stay legible."""
        from effgen.ui.render import format_cost

        self.cli.print_header("Session")
        self.cli.print(f"Model: {self.model_id}")
        if self.provider:
            self.cli.print(f"Provider: {self.provider}")
        if self.preset:
            self.cli.print(f"Preset: {self.preset}")
        if self.system_prompt:
            self.cli.print("Persona: custom (set with --persona/--system-prompt)")
        if self.guardrails:
            self.cli.print(f"Guardrails: {self.guardrails}")
        active = sorted(getattr(self.agent, "tools", {}) or {})
        self.cli.print(f"Tools: {', '.join(active) if active else '(none)'}")
        if self.session_id:
            self.cli.print(
                f"Session: {self.session_id}  (resumable via `effgen sessions`)"
            )
        else:
            self.cli.print("Session: not persisted  (start one with /session <id>)")
        cost = format_cost(self.session_cost) if self.session_cost else "$0.00"
        self.cli.print(
            f"Totals: {self.turns} turns · {self.session_tokens:,} tok · {cost}"
        )

    def _cmd_session(self, arg: str) -> None:
        """Show, or begin persisting under, the resumable session id.

        This id maps to the core Session store that ``effgen sessions`` lists and
        ``effgen chat --session-id`` / ``effgen run --session-id`` resume — a
        different store from the file-based ``/save`` + ``/load`` snapshots.
        """
        if not arg:
            if self.session_id:
                self.cli.print(f"Session id: {self.session_id}")
                self.cli.print(
                    "Resume later with:  effgen chat --session-id "
                    f"{self.session_id}   (also shown by `effgen sessions`)."
                )
            else:
                self.cli.print("This chat is not being saved to the session store.")
                self.cli.print(
                    "Start persisting:  /session <id>   "
                    "(then `effgen sessions` and --session-id can resume it)."
                )
            return
        name = arg.split()[0]
        if self.session_id == name:
            self.cli.print(f"Already on session '{name}'.")
            return
        if self.session_id:
            self.cli.print_warning(
                f"Already saving to '{self.session_id}'. Restart with "
                f"`effgen chat --session-id {name}` to switch id."
            )
            return
        try:
            from effgen.core.session import Session

            session = Session.load_or_create(name)
            for msg in self._dump_history():
                if msg["role"] == "user":
                    session.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    session.add_assistant_message(msg["content"])
            session.save()
            self.session_id = name
            if self.agent is not None:
                self.agent.session = session
                self.agent._session_id = name
            self.cli.print_success(
                f"Saving this conversation as session '{name}'. "
                f"Resume with `effgen chat --session-id {name}`."
            )
        except Exception as e:  # noqa: BLE001
            self.cli.print_error(f"Could not start session '{name}': {e}")

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
        from effgen.ui.render import ascii_fold

        self.cli.print_header(
            ascii_fold("Last turn — reasoning trace", self._human_stream())
        )
        lines = _progress.execution_trace_lines(
            self.last_trace, stream=self.cli._human_stream()
        )
        if not lines:
            self.cli.print("(no detailed steps recorded for this turn)")
            return
        for style, text in lines:
            text = ascii_fold(text, self.cli._human_stream())
            if self.console and self._color:
                self.console.print(f"[{style}]{text}[/{style}]")
            else:
                self.cli.print(text)

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
