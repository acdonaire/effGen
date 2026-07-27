"""Interactive REPL for ``effgen code``.

``effgen code`` with a terminal and no task opens this loop: the coding agent
answers turn by turn, writing files, running code in the configured sandbox, and
reading the real output — with every change previewed as a unified diff and gated
by the session's permission mode. It parallels :class:`~effgen.cli.chat.ChatREPL`
and reuses its plumbing (TTY detection, ``NO_COLOR`` handling, stderr routing for
a piped session, readline history + tab-completion, the ``/model`` hot-swap, and
the session store), adding the parts a coding session needs:

- a **files-in-context** set the user manages with ``/add`` / ``/drop`` /
  ``/context``, injected into each turn so the agent sees the files being worked
  on, with a token/size estimate so context stays bounded;
- a **plan → review → apply** flow: ``/plan <task>`` proposes edits without
  writing them, ``/diff`` shows the staged edits, and ``/apply`` / ``/reject``
  carry them out or discard them (reusing the diff/apply/undo layer);
- ``/run`` / ``/test`` to execute a command through the confined sandbox and feed
  the output back into the conversation;
- ``/undo`` over the on-disk edit journal, ``/compact`` to summarize a long
  conversation, and ``/save`` / ``/session`` / ``/load`` to round-trip a session;
- **project awareness**: the repository's branch, short status and file layout
  are read at startup and travel with each turn, ``/git`` shows the read-only git
  surface and pulls the staged diff into context, and ``/git commit`` records the
  session's edits after an explicit confirmation.

The interactive chrome is a terminal affordance only. When there is no terminal
the command never reaches this class — :func:`effgen.cli.commands.code.run_code_command`
runs the single-shot path instead — so the piped/``--json`` output stays clean.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from effgen.cli.chat import _history_dir
from effgen.cli.code.engine import CodeEngine, resolve_workspace, undo_workspace, workspace_env
from effgen.cli.code.permissions import (
    MODE_DESCRIPTIONS,
    ActionRecord,
    PermissionMode,
    default_mode,
)
from effgen.cli.code.project import build_project_context, staged_diff
from effgen.cli.code.render import print_action, print_diff, print_plain, print_status
from effgen.ui.theme import color_enabled

logger = logging.getLogger(__name__)

# Slash commands surfaced in /help and tab-completion. One source of truth so the
# help text, the completer and the dispatcher never drift apart.
_SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help (or type just / for the menu)",
    "/plan": "Propose a change without writing it:  /plan add retries to fetch.py",
    "/run": "Run a shell command in the workspace:  /run ls -la",
    "/test": "Run tests in the workspace:  /test [pytest args]",
    "/diff": "Show the edits staged by the last /plan (or the last turn's edits)",
    "/apply": "Apply the staged edits:  /apply [n]  (n = one edit by number)",
    "/reject": "Discard the staged edits:  /reject [n]",
    "/undo": "Reverse the last applied edit(s):  /undo [n]",
    "/context": "Show the files in context and their token/size estimate "
                "(/context refresh re-reads the project layout)",
    "/add": "Add a workspace file to context:  /add src/app.py",
    "/drop": "Remove a file from context:  /drop src/app.py",
    "/mode": "Show or set the permission mode:  /mode ask|auto-edit|yes|plan",
    "/model": "Hot-swap the active model:  /model gpt-5-nano",
    "/tools": "List the coding tools the agent can call",
    "/cost": "Session token + cost total",
    "/trace": "Show the last turn's reasoning/tool steps",
    "/git": "Git for the workspace:  /git [status|diff|log|branch|show|remote] · "
            "/git staged (pull the staged diff into context) · /git commit [message]",
    "/reset": "Clear the conversation memory (keep files-in-context)",
    "/clear": "Reset live context (memory + files) but keep the session on disk",
    "/compact": "Summarize the conversation so far to extend a long session",
    "/save": "Save this coding session to a file:  /save [name]",
    "/session": "Show or name the resumable session id (used by `effgen sessions`)",
    "/load": "Load a saved coding session:  /load [name|number]",
    "/doctor": "Run a quick environment check",
    "/exit": "Leave the coding agent (also: /quit, exit, quit)",
}

# The four tools the coding preset gives the agent, by tool name, for /tools.
_CODING_TOOLS: tuple[tuple[str, str], ...] = (
    ("file_operations", "read, write, list and search files in the workspace"),
    ("code_executor", "run code in the configured sandbox"),
    ("python_repl", "evaluate short Python expressions in a restricted namespace"),
    ("bash", "run a shell command in the workspace"),
)


def _estimate_tokens(text: str) -> int:
    """A rough token estimate (~4 characters per token) for context budgeting."""
    return max(1, len(text) // 4) if text else 0




class CodeREPL:
    """An interactive coding session over a single :class:`CodeEngine`."""

    def __init__(self, cli: Any, args: Any) -> None:
        self.cli = cli
        self.args = args
        self.console = getattr(cli, "console", None)

        self.quiet = bool(getattr(args, "quiet", False))
        self.verbose = bool(getattr(args, "verbose", False))
        self._color = color_enabled()
        try:
            self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        except (ValueError, OSError):  # pragma: no cover - closed std streams
            self.interactive = False
        if not self.interactive:
            try:
                self.cli._human_to_stderr = True
            except Exception:  # noqa: BLE001
                pass

        self.workspace: Path = resolve_workspace(getattr(args, "workspace", None))
        self.provider = getattr(args, "provider", None)
        self._model_defaulted = getattr(args, "model", None) is None
        self.model_id = getattr(args, "model", None)
        self.temperature = getattr(args, "temperature", None)
        self.max_tokens = getattr(args, "max_tokens", None)
        self.max_iterations = getattr(args, "max_iterations", None)

        # The permission mode for the session. A named flag pins it; otherwise a
        # terminal session confirms each action (ask).
        self.mode, self.mode_explicit = self._initial_mode(args)

        # Files the user has pulled into context; their live content is injected
        # into each turn so the agent works from what is on disk right now.
        self.context_files: list[str] = []
        # Edits proposed by the last /plan turn, keyed by path (latest wins),
        # awaiting /apply or /reject.
        self.pending_edits: dict[str, Any] = {}
        # True only while a /plan turn runs, so on_diff stages instead of applying.
        self._staging = False

        # The repository state, file layout and project brief, read once at
        # startup and refreshed on request with /context refresh.
        self.project = build_project_context(self.workspace)
        # Workspace-relative paths this session has written, in order, so
        # /git commit records exactly the session's own edits.
        self.session_files: list[str] = []

        self.engine: CodeEngine | None = None
        self.session_id: str | None = None
        self.session_tokens = 0
        self.session_cost = 0.0
        self.turns = 0
        self.last_trace: list[dict[str, Any]] | None = None
        self.last_result: Any = None
        self._last_task = ""
        self._history_file = _history_dir() / "code_input_history"

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------
    def _initial_mode(self, args: Any) -> tuple[PermissionMode, bool]:
        """Return ``(mode, explicit)`` from the permission flags, defaulting to ask."""
        if getattr(args, "plan_only", False):
            return PermissionMode.PLAN, True
        if getattr(args, "auto_edit", False):
            return PermissionMode.AUTO_EDIT, True
        if getattr(args, "assume_yes", False):
            return PermissionMode.YES, True
        return default_mode(self.interactive), False

    # ------------------------------------------------------------------
    # Engine / agent construction
    # ------------------------------------------------------------------
    def _resolve_model(self) -> None:
        """Pick a model when the user named none, mirroring the single-shot path."""
        if self.model_id:
            return
        from effgen.cli.commands._shared import _quickstart_suggest_model

        model, suggested_provider, reason = _quickstart_suggest_model()
        self.model_id = model
        if self.provider is None and suggested_provider:
            self.provider = suggested_provider
        if not self.quiet:
            self._say(f"Using model {model} ({reason}); swap with /model.")

    def _build_engine(self, carry_from: CodeEngine | None = None) -> CodeEngine:
        """Build the engine and its agent, carrying conversation memory across a swap."""
        engine = CodeEngine(
            model=self.model_id or "",
            provider=self.provider,
            workspace=self.workspace,
            mode=self.mode,
            mode_explicit=self.mode_explicit,
            interactive=self.interactive,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            on_event=self._on_event,
            on_diff=self._on_diff,
            project=self.project,
        )
        agent = engine.build_agent()
        if carry_from is not None and carry_from._agent is not None:
            self._carry_memory(carry_from._agent, agent)
        return engine

    def _carry_memory(self, old_agent: Any, new_agent: Any) -> None:
        """Copy the running conversation from *old_agent* into *new_agent*."""
        try:
            from effgen.memory.short_term import MessageRole

            for msg in old_agent.short_term_memory.get_messages():
                if msg.role == MessageRole.USER:
                    new_agent.short_term_memory.add_user_message(msg.content)
                elif msg.role == MessageRole.ASSISTANT:
                    new_agent.short_term_memory.add_assistant_message(msg.content)
        except Exception:  # noqa: BLE001 - memory carry is best-effort
            pass
        if getattr(old_agent, "session", None) is not None:
            new_agent.session = old_agent.session
            new_agent._session_id = getattr(old_agent, "_session_id", self.session_id)

    @property
    def agent(self) -> Any:
        """The live agent, or ``None`` before startup."""
        return self.engine._agent if self.engine is not None else None

    def _tool(self, name: str) -> Any:
        """Return the named gated tool from the live agent, or ``None``."""
        agent = self.agent
        if agent is None:
            return None
        return (getattr(agent, "tools", {}) or {}).get(name)

    # ------------------------------------------------------------------
    # Gate callbacks (live per-action ticks and diffs)
    # ------------------------------------------------------------------
    def _on_event(self, record: ActionRecord) -> None:
        """Print a decided action as a live tick (interactive, non-quiet only)."""
        if self.interactive and not self.quiet:
            print_action(self.cli, record)

    def _on_diff(self, edit: Any) -> None:
        """Show a pending edit's diff, and stage it while a /plan turn runs."""
        if not self.quiet:
            print_diff(self.cli, edit)
        if self._staging and not edit.unchanged:
            self.pending_edits[edit.rel_path] = edit

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _say(self, text: str = "", *, style: str | None = None) -> None:
        """Print one line of session output verbatim (see :func:`print_plain`)."""
        print_plain(self.cli, text, style=style)

    def _status(self, kind: str, text: str) -> None:
        """Print a success/warning/error line with its glyph, without markup."""
        print_status(self.cli, kind, text)

    # ------------------------------------------------------------------
    # Banner / prompt / readline
    # ------------------------------------------------------------------
    def _banner_line(self, text: str, style: str | None = None) -> None:
        self._say(text, style=style)

    def _banner(self) -> None:
        from effgen import __version__

        self._banner_line(f"\neffGen v{__version__} · code", style="bold cyan")
        from effgen.cli import progress as _progress

        self._banner_line(f"Model: {_progress.short_model_label(self.model_id)}")
        self._banner_line(f"Workspace: {self.workspace}")
        self._banner_line(self.project.summary_line())
        if self.project.brief_path:
            self._banner_line(f"Project instructions: {self.project.brief_path}")
        self._banner_line(
            f"Permissions: {self.mode.value} — {MODE_DESCRIPTIONS[self.mode]}"
        )
        self._banner_line(
            "Describe a change and press Enter.  End a line with \\ for multi-line.\n"
            "Slash commands (type / for the menu): /help  /plan  /run  /test  /diff  "
            "/apply  /undo  /context  /add  /model  /git  /exit"
        )
        if self._model_defaulted and "/" in (self.model_id or "") and ":" not in (self.model_id or ""):
            self._banner_line(
                f"Loading a local model ({self.model_id}); the first run downloads it. "
                "For a fast keyed model:  /model groq:llama-3.1-8b-instant",
                style="dim",
            )

    def _prompt_str(self) -> str:
        from effgen.cli import progress as _progress

        label = _progress.short_model_label(self.model_id)
        bits = ["code", label, self.mode.value]
        if self.context_files:
            bits.append(f"{len(self.context_files)} file" + ("s" if len(self.context_files) != 1 else ""))
        return " · ".join(bits) + " › "

    def _setup_readline(self) -> None:
        try:
            import readline
        except Exception:  # noqa: BLE001 - readline absent (e.g. Windows)
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
        """Read one (possibly multi-line) entry. ``None`` on EOF."""
        prompt = self._prompt_str() if self.interactive else ""
        cont = "… " if self.interactive else ""
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
        """Run the coding REPL until exit; returns the process exit code."""
        self._resolve_model()
        if self.interactive:
            self._banner()
        self._setup_readline()
        try:
            self.engine = self._build_engine()
        except Exception as e:  # noqa: BLE001
            self._status("error", f"Could not start the coding agent: {e}")
            return 1

        try:
            while True:
                try:
                    user_input = self._read_input()
                except KeyboardInterrupt:
                    if self.interactive:
                        self._say("")
                    continue
                if user_input is None:  # EOF / Ctrl-D
                    if self.interactive:
                        self._say("\nGoodbye!")
                    break
                if not user_input.strip():
                    continue

                action = self._dispatch(user_input.strip())
                if action == "exit":
                    if self.interactive:
                        self._say("Goodbye!")
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
        """Route a line. Returns 'exit', 'handled', or None (treat as a task)."""
        low = text.lower()
        if low in ("exit", "quit"):
            return "exit"
        if low == "help":
            self._cmd_help()
            return "handled"

        if not text.startswith("/"):
            return None
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
            "/plan": lambda: self._cmd_plan(arg),
            "/run": lambda: self._cmd_run(arg),
            "/test": lambda: self._cmd_test(arg),
            "/diff": lambda: self._cmd_diff(),
            "/apply": lambda: self._cmd_apply(arg),
            "/reject": lambda: self._cmd_reject(arg),
            "/undo": lambda: self._cmd_undo(arg),
            "/context": lambda: self._cmd_context(arg),
            "/add": lambda: self._cmd_add(arg),
            "/drop": lambda: self._cmd_drop(arg),
            "/mode": lambda: self._cmd_mode(arg),
            "/model": lambda: self._cmd_model(arg),
            "/tools": lambda: self._cmd_tools(),
            "/cost": lambda: self._cmd_cost(),
            "/trace": lambda: self._cmd_trace(),
            "/git": lambda: self._cmd_git(arg),
            "/reset": lambda: self._cmd_reset(),
            "/clear": lambda: self._cmd_clear(),
            "/compact": lambda: self._cmd_compact(arg),
            "/save": lambda: self._cmd_save(arg),
            "/session": lambda: self._cmd_session(arg),
            "/load": lambda: self._cmd_load(arg),
            "/doctor": lambda: self._cmd_doctor(),
        }
        handler = handlers.get(cmd)
        if handler is None:
            import difflib

            typed = text.split()[0]
            close = difflib.get_close_matches(cmd, list(_SLASH_COMMANDS), n=3)
            hint = (
                f" Did you mean: {', '.join(close)}?" if close else " Type /help for the list."
            )
            self._status("warning", f"Unknown command '{typed}'.{hint}")
            return "handled"
        result = handler()
        return result if result == "exit" else "handled"

    # ------------------------------------------------------------------
    # A turn
    # ------------------------------------------------------------------
    def _compose_task(self, message: str) -> str:
        """Prepend the live content of any files-in-context to *message*."""
        if not self.context_files:
            return message
        blocks: list[str] = []
        for rel in self.context_files:
            content = self._read_context_file(rel)
            if content is None:
                continue
            blocks.append(f"=== {rel} ===\n{content}")
        if not blocks:
            return message
        joined = "\n\n".join(blocks)
        return (
            "Files currently in context (their content on disk right now):\n\n"
            f"{joined}\n\n---\n\n{message}"
        )

    def _read_context_file(self, rel: str) -> str | None:
        try:
            return (self.workspace / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def _snapshot(self) -> tuple[int, float]:
        m = getattr(self.agent, "model", None)
        return (
            int(getattr(m, "total_tokens", 0) or 0),
            float(getattr(m, "total_cost", 0.0) or 0.0),
        )

    def _do_turn(self, message: str) -> None:
        """Run one coding turn and render the result."""
        assert self.engine is not None
        task = self._compose_task(message)
        self._last_task = message
        self.engine.gate.begin_turn()
        tok0, cost0 = self._snapshot()
        t0 = time.monotonic()
        self.last_trace = None

        try:
            with workspace_env(self.workspace):
                response = self._run_agent(task)
        except KeyboardInterrupt:
            self._status("warning", "Interrupted; no further actions were taken.")
            return
        except Exception as e:  # noqa: BLE001
            self._status("error", f"{e}")
            return

        result = self.engine.result_from_response(task, response)
        self.last_result = result
        self._note_written(result.files_written)
        try:
            self.last_trace = response.execution_trace or None
        except Exception:  # noqa: BLE001
            self.last_trace = None

        self._render_answer(result)
        self.turns += 1
        elapsed = time.monotonic() - t0
        tok1, cost1 = self._snapshot()
        rtok = int(getattr(response, "tokens_used", 0) or 0) or max(tok1 - tok0, 0)
        rcost = result.cost_usd if result.cost_usd is not None else max(cost1 - cost0, 0.0)
        self.session_tokens += max(rtok, 0)
        self.session_cost += max(rcost or 0.0, 0.0)
        if not self.quiet:
            self._print_footer(elapsed, rtok, rcost or 0.0)
        self._post_turn_notes(result)

    def _note_written(self, paths: list[str]) -> None:
        """Record the paths just written and re-read the project from disk.

        The paths are remembered for ``/git commit``, and the layout and status
        the agent is working from are rebuilt so the next turn sees the files the
        last one created rather than a listing that predates them.
        """
        if not paths:
            return
        for rel in paths:
            if rel not in self.session_files:
                self.session_files.append(rel)
        if self.engine is not None:
            self.project = self.engine.load_project(refresh=True)

    def _run_agent(self, task: str) -> Any:
        """Run one turn under the live status line ``chat`` uses.

        The turn goes through ``agent.run``, which drives the tool loop with the
        model's native tool calling where the provider supports it. The status
        line follows the same execution events (``Running file_operations…``),
        and each write's diff and each decided action print through it as they
        happen, so the session shows its work while the loop runs.
        """
        from effgen.cli import progress as _progress
        from effgen.core.agent import AgentMode

        console = self.cli._human()
        animate = console is not None and _progress.animation_enabled(
            quiet=self.quiet,
            no_animation=bool(getattr(self.args, "no_animation", False)),
            stream=self.cli._human_stream(),
        )
        if not animate:
            if not self.quiet and self.interactive:
                self._banner_line("· working…", style="dim")
            return self.agent.run(task, mode=AgentMode.AUTO)

        with _progress.LiveStatus(
            console,
            model_label=_progress.short_model_label(self.model_id),
            reasoning=_progress.is_reasoning_agent(self.agent),
            tracker=getattr(self.agent, "execution_tracker", None),
            hint="Ctrl-C to cancel",
        ):
            return self.agent.run(task, mode=AgentMode.AUTO)

    def _render_answer(self, result: Any) -> None:
        from effgen.ui.render import answer_surface

        console = self.cli._human()
        if not result.success and not result.partial:
            self.cli.print_error_panel(
                result.answer or "The run produced no answer.", title="Error"
            )
        elif console:
            answer_surface(
                result.answer,
                success=result.success,
                partial=result.partial,
                framed=False,
                label="agent",
                console=console,
            )
        else:
            self._say(result.answer)

    def _post_turn_notes(self, result: Any) -> None:
        """Surface files written, refusals, withheld actions and the iteration cap."""
        if self.quiet:
            return
        if result.files_written:
            self._say(f"Files written: {', '.join(result.files_written)}")
        for diff in result.diffs:
            failed = int(diff.get("hunks_failed", 0) or 0)
            if failed:
                self._status(
                    "warning",
                    f"{diff.get('path')}: {failed} hunk(s) did not apply (the file "
                    "changed since it was read); the rest were applied."
                )
        for record in result.refused:
            self._status("warning", record.reason)
        if result.withheld:
            kinds = ", ".join(sorted({a.kind for a in result.withheld}))
            # Name the mode the turn actually ran under: ``/plan`` runs one turn
            # in plan mode without changing the session's mode.
            turn_mode = getattr(result, "permission_mode", "") or self.mode.value
            if turn_mode == PermissionMode.PLAN.value:
                self._say(
                    f"Plan mode: {len(result.withheld)} action(s) ({kinds}) were "
                    "proposed, not carried out. /apply to write staged edits, or "
                    "/mode auto-edit."
                )
            else:
                self._say(
                    f"{len(result.withheld)} action(s) ({kinds}) were not permitted "
                    f"in {turn_mode} mode. /mode auto-edit (or /mode yes) to allow."
                )
        if result.hit_iteration_cap:
            self._status(
                "warning",
                f"Stopped at the iteration cap ({result.iterations} iterations). "
                "Raise it with --max-iterations or narrow the task."
            )

    def _print_footer(self, elapsed: float, tokens: int, cost: float) -> None:
        from effgen.ui.render import ascii_fold, summary_line

        footer, _ = summary_line(
            mode="chat",
            elapsed=elapsed,
            tokens=tokens,
            cost=cost,
            session=(self.turns, self.session_tokens, self.session_cost),
        )
        self._say(ascii_fold(footer, self.cli._human_stream()), style="dim")

    # ------------------------------------------------------------------
    # /plan, /diff, /apply, /reject
    # ------------------------------------------------------------------
    def _cmd_plan(self, arg: str) -> None:
        """Propose a change for *arg* without writing anything; stage the edits."""
        if not arg:
            self._say(
                "Usage: /plan <task>. Proposes the edits for a task without writing "
                "them, then /diff to review and /apply to carry them out."
            )
            return
        assert self.engine is not None
        prev_mode = self.engine.gate.mode
        self.pending_edits.clear()
        self.engine.gate.mode = PermissionMode.PLAN
        self._staging = True
        try:
            self._do_turn(arg)
        finally:
            self._staging = False
            self.engine.gate.mode = prev_mode
        if self.pending_edits:
            self._say(
                f"{len(self.pending_edits)} edit(s) staged: "
                f"{', '.join(self.pending_edits)}. /diff to review, /apply to write, "
                "/reject to discard."
            )
        else:
            self._say("No edits were proposed.")

    def _cmd_diff(self) -> None:
        """Show the staged edits, or the edits from the last turn."""
        if self.pending_edits:
            self._say(f"Staged edits ({len(self.pending_edits)}):")
            for edit in self.pending_edits.values():
                print_diff(self.cli, edit)
            return
        result = self.last_result
        if result is None or not result.diffs:
            self._say("No staged edits, and the last turn changed no files.")
            return
        from effgen.cli.code.diffs import render_diff

        stream = self.cli._human_stream()
        console = self.cli._human()

        def show(diffs: list[Any], heading: str) -> None:
            if not diffs:
                return
            self._say(f"{heading} ({len(diffs)}):")
            for diff in diffs:
                plain, markup = render_diff(str(diff.get("diff", "")), stream)
                if console:
                    console.print(markup, highlight=False)
                else:
                    self._say(plain)

        show([d for d in result.diffs if d.get("applied", True)], "Last turn's edits")
        show(
            [d for d in result.diffs if not d.get("applied", True)],
            "Proposed last turn, not written",
        )

    def _staged_selection(self, arg: str) -> list[Any] | None:
        """Resolve ``/apply``/``/reject`` arguments to a list of staged edits.

        A bare command selects every staged edit; a 1-based number selects one.
        Returns ``None`` (after reporting) when the selection is invalid.
        """
        edits = list(self.pending_edits.values())
        if not edits:
            self._say("Nothing is staged. Use /plan <task> first.")
            return None
        if not arg:
            return edits
        token = arg.split()[0]
        if not token.isdigit():
            self._status("warning", f"Expected an edit number, got '{token}'.")
            return None
        idx = int(token) - 1
        if not 0 <= idx < len(edits):
            self._status("warning", f"No staged edit #{token} (have {len(edits)}).")
            return None
        return [edits[idx]]

    def _cmd_apply(self, arg: str) -> None:
        """Write the staged edits to disk through the gated, journaled write path."""
        selected = self._staged_selection(arg)
        if selected is None:
            return
        assert self.engine is not None
        fops = self._tool("file_operations")
        if fops is None:
            self._status("error", "The file tool is unavailable.")
            return
        prev_mode = self.engine.gate.mode
        self.engine.gate.mode = PermissionMode.AUTO_EDIT
        self.engine.gate.begin_turn()
        written: list[str] = []
        try:
            for edit in selected:
                result = asyncio.run(
                    fops.execute(operation="write", path=edit.rel_path, content=edit.new_content)
                )
                ok = getattr(result, "success", None)
                if ok is None and isinstance(result, dict):
                    ok = result.get("success")
                if ok is not False:
                    written.append(edit.rel_path)
                    self.pending_edits.pop(edit.rel_path, None)
        finally:
            self.engine.gate.mode = prev_mode
        if written:
            self._note_written(written)
            self._status("success", f"Applied: {', '.join(written)} (/undo to reverse).")
        else:
            self._status("warning", "No edits were applied.")

    def _cmd_reject(self, arg: str) -> None:
        """Discard the staged edits without writing them."""
        selected = self._staged_selection(arg)
        if selected is None:
            return
        for edit in selected:
            self.pending_edits.pop(edit.rel_path, None)
        names = ", ".join(e.rel_path for e in selected)
        self._say(f"Discarded staged edit(s): {names}.")

    def _cmd_undo(self, arg: str) -> None:
        """Reverse the last applied edit(s) via the on-disk journal."""
        count = 1
        if arg:
            token = arg.split()[0]
            if token.isdigit():
                count = max(1, int(token))
        outcomes, remaining = undo_workspace(self.workspace, count)
        if not outcomes:
            self._say(f"Nothing to undo in {self.workspace}.")
            return
        for outcome in outcomes:
            if outcome.action == "restored":
                self._say(f"Restored {outcome.rel_path} to its previous content.")
            elif outcome.action == "removed":
                self._say(f"Removed {outcome.rel_path} (created by a coding run).")
            elif outcome.action == "skipped":
                self._status("warning", f"Skipped {outcome.rel_path}: {outcome.detail}")
            else:
                self._status("error", f"Could not undo {outcome.rel_path}: {outcome.detail}")
        self._say(f"{remaining} earlier change(s) can still be undone.")

    # ------------------------------------------------------------------
    # /run, /test
    # ------------------------------------------------------------------
    def _cmd_run(self, arg: str) -> None:
        if not arg:
            self._say("Usage: /run <command>   (e.g. /run pytest -q)")
            return
        self._execute_shell(arg)

    def _cmd_test(self, arg: str) -> None:
        command = f"pytest {arg}".strip() if arg else "pytest"
        self._say(f"Running: {command}")
        self._execute_shell(command)

    def _execute_shell(self, command: str) -> None:
        """Run *command* in the workspace and feed the output back to the agent."""
        assert self.engine is not None
        bash = self._tool("bash")
        if bash is None:
            self._status("error", "The shell tool is unavailable.")
            return
        prev_mode = self.engine.gate.mode
        # The user typed the command explicitly, so it is allowed without a second
        # prompt; the shell's own blocked-command and credential-file guards still
        # apply underneath, and it stays confined to the workspace.
        self.engine.gate.mode = PermissionMode.YES
        self.engine.gate.begin_turn()
        try:
            with workspace_env(self.workspace):
                result = asyncio.run(bash.execute(command=command))
        except Exception as e:  # noqa: BLE001
            self._status("error", f"Command failed to start: {e}")
            return
        finally:
            self.engine.gate.mode = prev_mode

        text = self._shell_output_text(result)
        self._say(text or "(no output)")
        # Feed the run back so the next turn can react to it.
        self._feed_back(f"I ran `{command}` in the workspace. Output:\n{text or '(no output)'}")

    @staticmethod
    def _shell_output_text(result: Any) -> str:
        """Return the combined stdout/stderr text from a shell tool result.

        ``bash``/``code_executor`` return their run detail as a dict
        (``stdout``/``stderr``/``return_code``); a gated refusal returns
        ``{"success": False, "error": ...}``. Both are reduced to the lines a
        user wants to read, so ``/run`` shows the command's output, not a dict.
        A command the shell tool blocked (or one that failed before producing
        any output) carries its reason on the result rather than in the output;
        that reason is returned, so a refusal is never reported as no output.
        """
        output = getattr(result, "output", None)
        if output is None and isinstance(result, dict):
            output = result
        if isinstance(output, dict):
            if output.get("success") is False and output.get("error"):
                return str(output["error"]).rstrip()
            parts = [str(output.get("stdout") or ""), str(output.get("stderr") or "")]
            text = "\n".join(p for p in parts if p.strip()).rstrip()
            if text:
                return text
        elif output:
            return str(output).rstrip()
        return str(getattr(result, "error", "") or "").rstrip()

    def _feed_back(self, note: str) -> None:
        """Add an observation to the conversation so the next turn sees it."""
        agent = self.agent
        if agent is None:
            return
        try:
            agent.short_term_memory.add_user_message(note)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Files in context
    # ------------------------------------------------------------------
    def _cmd_context(self, arg: str = "") -> None:
        """Show what the agent is working from; ``refresh`` re-reads the project."""
        if arg.split()[:1] == ["refresh"]:
            self.project = (
                self.engine.load_project(refresh=True)
                if self.engine is not None
                else build_project_context(self.workspace)
            )
            self._status("success", f"Project context refreshed — {self.project.summary_line()}")
            return

        self.cli.print_header("Files in context")
        self._say(f"Workspace: {self.workspace}")
        self._say(self.project.summary_line())
        if self.project.brief_path:
            self._say(f"Project instructions: {self.project.brief_path}")
        self._say(
            f"Project layout in the prompt: {self.project.file_count_text()}, "
            f"{len(self.project.layout_lines())} line(s) — /context refresh to re-read"
        )
        if not self.context_files:
            self._say("Files pinned with /add: (none) — add one with /add <file>")
            return
        self._say("Files pinned with /add (their content is sent every turn):")
        total = 0
        for rel in self.context_files:
            content = self._read_context_file(rel)
            if content is None:
                self._status("warning", f"{rel} — unreadable (removed or not text)")
                continue
            tokens = _estimate_tokens(content)
            total += tokens
            self._say(f"  {rel}  ({len(content):,} chars, ~{tokens:,} tok)")
        self._say(f"Total: {len(self.context_files)} file(s), ~{total:,} tokens (estimate)")

    def _cmd_add(self, arg: str) -> None:
        if not arg:
            self._say("Usage: /add <file>   (a path inside the workspace)")
            return
        rel = arg.split()[0]
        try:
            resolved = (self.workspace / rel).resolve()
            resolved.relative_to(self.workspace.resolve())
        except (ValueError, OSError):
            self._status(
                "warning",
                f"'{rel}' is outside the workspace {self.workspace}; not added."
            )
            return
        if not resolved.is_file():
            self._status("warning", f"No file at '{rel}' in the workspace.")
            return
        norm = str(resolved.relative_to(self.workspace.resolve()))
        if norm in self.context_files:
            self._say(f"'{norm}' is already in context.")
            return
        if self._read_context_file(norm) is None:
            self._status("warning", f"'{norm}' is not readable as text; not added.")
            return
        self.context_files.append(norm)
        self._status("success", f"Added '{norm}' to context ({len(self.context_files)} file(s)).")

    def _cmd_drop(self, arg: str) -> None:
        if not arg:
            self._say("Usage: /drop <file>")
            return
        rel = arg.split()[0]
        # Accept either the stored relative path or the raw argument.
        match = None
        for stored in self.context_files:
            if stored == rel or Path(stored).name == rel:
                match = stored
                break
        if match is None:
            self._status("warning", f"'{rel}' is not in context.")
            return
        self.context_files.remove(match)
        self._status("success", f"Dropped '{match}' ({len(self.context_files)} file(s) left).")

    # ------------------------------------------------------------------
    # Mode / model / tools
    # ------------------------------------------------------------------
    def _cmd_mode(self, arg: str) -> None:
        if not arg:
            self._say(f"Permission mode: {self.mode.value} — {MODE_DESCRIPTIONS[self.mode]}")
            self._say("Change with:  /mode ask | auto-edit | yes | plan")
            return
        name = arg.split()[0].lower()
        try:
            new_mode = PermissionMode(name)
        except ValueError:
            self._status(
                "warning",
                f"Unknown mode '{name}'. Choose: ask, auto-edit, yes, plan."
            )
            return
        self.mode = new_mode
        self.mode_explicit = True
        if self.engine is not None:
            self.engine.gate.mode = new_mode
        self._status("success", f"Permission mode: {new_mode.value} — {MODE_DESCRIPTIONS[new_mode]}")

    def _resolve_swap_target(self, new_id: str) -> tuple[str | None, str]:
        """Resolve a ``/model`` argument like a fresh ``code -m <id>`` would."""
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
            self._say(f"Active model: {self.model_id}")
            self._say("Swap with:  /model <model-id>   (e.g. /model gpt-5-nano)")
            return
        new_id = arg.split()[0]
        old_engine = self.engine
        old_id, old_provider = self.model_id, self.provider
        new_provider, resolved_id = self._resolve_swap_target(new_id)
        self.model_id, self.provider = resolved_id, new_provider

        _effgen_logger = logging.getLogger("effgen")
        _prev_level = _effgen_logger.level
        if not self.verbose:
            _effgen_logger.setLevel(logging.CRITICAL)
        try:
            self.engine = self._build_engine(carry_from=old_engine)
        except Exception as e:  # noqa: BLE001
            self.model_id, self.provider = old_id, old_provider
            self.engine = old_engine
            self._status("error", f"Could not switch to '{new_id}': {e}")
            return
        finally:
            _effgen_logger.setLevel(_prev_level)
        if old_engine is not None and old_engine._agent is not None:
            try:
                old_engine._agent.close()
            except Exception:  # noqa: BLE001
                pass
        from effgen.cli import progress as _progress

        self._status(
            "success",
            f"Switched model: {_progress.short_model_label(new_id)} (conversation kept)"
        )

    def _cmd_tools(self) -> None:
        self.cli.print_header("Coding tools")
        self._say(
            "The agent calls these directly; every write and command passes the "
            f"permission gate ({self.mode.value} mode)."
        )
        for name, desc in _CODING_TOOLS:
            self._say(f"  {name:16s} {desc}")

    # ------------------------------------------------------------------
    # Cost / trace / git
    # ------------------------------------------------------------------
    def _cmd_cost(self) -> None:
        from effgen.ui.render import format_cost

        self.cli.print_header("Session cost")
        self._say(f"Turns: {self.turns}")
        self._say(f"Tokens: {self.session_tokens:,}")
        self._say(
            f"Cost: {format_cost(self.session_cost) if self.session_cost else '$0.00'}"
        )

    def _cmd_trace(self) -> None:
        if not self.last_trace:
            self._say("No turn to trace yet.")
            return
        from effgen.cli import progress as _progress
        from effgen.ui.render import ascii_fold

        self.cli.print_header("Last turn — reasoning trace")
        lines = _progress.execution_trace_lines(self.last_trace, stream=self.cli._human_stream())
        if not lines:
            self._say("(no detailed steps recorded for this turn)")
            return
        for style, text in lines:
            self._say(ascii_fold(text, self.cli._human_stream()), style=style)

    def _cmd_git(self, arg: str) -> None:
        """Git for the workspace: the read-only surface, plus a confirmed commit.

        ``status``/``diff``/``log``/``branch``/``show``/``remote`` go through the
        read-only ``git`` tool. ``staged`` shows the staged patch and adds it to
        the conversation so the next turn can work from it. ``commit`` is the one
        action that changes the repository, and it asks first.
        """
        from effgen.tools.builtin.devops import GitTool

        parts = arg.split(maxsplit=1) if arg else []
        op = parts[0].lower() if parts else "status"
        rest = parts[1].strip() if len(parts) > 1 else ""

        if op == "staged":
            self._git_staged()
            return
        if op == "commit":
            self._git_commit(rest)
            return

        if op not in GitTool.ALLOWED:
            self._status(
                "warning",
                f"'{op}' is not available. Choose a read-only operation "
                f"({', '.join(sorted(GitTool.ALLOWED))}), 'staged' to pull the "
                "staged diff into context, or 'commit' to record this session's "
                "edits after confirming."
            )
            return
        try:
            result = asyncio.run(GitTool().execute(operation=op, cwd=str(self.workspace)))
        except Exception as e:  # noqa: BLE001
            self._status("error", f"git {op} failed: {e}")
            return
        output = getattr(result, "output", None)
        if isinstance(output, dict):
            body = str(output.get("stdout") or "")
            if int(output.get("returncode", 0) or 0) != 0 and not body.strip():
                self._status(
                    "warning",
                    (output.get("stderr") or "").strip()
                    or "Not a git repository (or git is unavailable)."
                )
                return
        else:
            body = str(output or "")
        self._say(body.rstrip() or "(no output)")

    def _git_staged(self) -> None:
        """Show the staged patch and add it to the conversation as context."""
        repo = self.project.repo
        if repo is None:
            self._status("warning", f"{self.workspace} is not inside a git repository.")
            return
        patch = staged_diff(repo.root)
        if not patch.strip():
            self._say("Nothing is staged in this repository.")
            return
        self._say(patch.rstrip())
        self._feed_back(
            "The staged changes in this repository (git diff --cached):\n" + patch
        )
        self._status("success", "The staged diff is now part of this conversation.")

    def _git_commit(self, message: str) -> None:
        """Commit the files this session wrote, after an explicit confirmation.

        Only the session's own paths are committed; anything else the user has
        staged stays staged. The permission gate asks before the commit runs, so
        a commit never happens on an implied yes.
        """
        assert self.engine is not None
        if not self.session_files:
            self._say(
                "This session has written no files yet, so there is nothing to commit."
            )
            return
        plan = self.engine.plan_commit(
            self.session_files, task=self._last_task, message=message or None
        )
        if plan.error:
            self._status("warning", plan.error)
            return
        self._say(plan.describe())
        self._say(f'Message: "{plan.subject}"')
        if plan.other_staged:
            self._status(
                "warning",
                f"{len(plan.other_staged)} other path(s) are staged; they stay "
                "staged and are not part of this commit: "
                + ", ".join(plan.other_staged[:5])
                + ("…" if len(plan.other_staged) > 5 else ""),
            )
        outcome = self.engine.perform_commit(plan)
        if outcome.success:
            self._status(
                "success",
                f"Committed {len(outcome.paths)} file(s) as "
                f"{outcome.commit or 'HEAD'}: {plan.subject}",
            )
            self.session_files.clear()
            self.project = self.engine.load_project(refresh=True)
        else:
            self._status("warning", f"Nothing was committed: {outcome.detail}")

    # ------------------------------------------------------------------
    # Memory: reset / clear / compact
    # ------------------------------------------------------------------
    def _cmd_reset(self) -> None:
        try:
            self.agent.reset_memory()
        except Exception:  # noqa: BLE001
            pass
        self.last_trace = None
        self.last_result = None
        self._status("success", "Conversation memory cleared (files-in-context kept).")

    def _cmd_clear(self) -> None:
        try:
            self.agent.reset_memory()
        except Exception:  # noqa: BLE001
            pass
        self.context_files.clear()
        self.pending_edits.clear()
        self.last_trace = None
        self.last_result = None
        note = "Live context reset (memory, files, staged edits cleared)."
        if self.session_id:
            note += f" Session '{self.session_id}' stays on disk."
        self._status("success", note)

    def _cmd_compact(self, arg: str) -> None:
        """Summarize the conversation so far and replace memory with the summary."""
        history = self._dump_history()
        if len(history) < 2:
            self._say("Not enough conversation to compact yet.")
            return
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        extra = f" Focus on: {arg}." if arg else ""
        prompt = (
            "Summarize the coding conversation below into a compact brief that lets "
            "the work continue: the goal, decisions made, files touched, and what is "
            f"left to do.{extra}\n\n{transcript}"
        )
        try:
            with workspace_env(self.workspace):
                from effgen.core.agent import AgentMode

                response = self.agent.run(prompt, mode=AgentMode.AUTO)
            summary = response.output or ""
        except Exception as e:  # noqa: BLE001
            self._status("error", f"Could not compact: {e}")
            return
        if not summary.strip():
            self._status("warning", "The summary came back empty; memory is unchanged.")
            return
        try:
            self.agent.reset_memory()
            self.agent.short_term_memory.add_assistant_message(
                f"[compacted context]\n{summary}"
            )
        except Exception:  # noqa: BLE001
            pass
        self._status("success", "Conversation compacted into a summary.")
        console = self.cli._human()
        if console:
            from effgen.ui.render import answer_surface

            answer_surface(summary, framed=True, title="Compacted context", console=console)
        else:
            self._say(summary)

    # ------------------------------------------------------------------
    # Session persistence: save / session / load
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

    def _cmd_save(self, arg: str) -> None:
        history = self._dump_history()
        if not history:
            self._say("Nothing to save yet.")
            return
        name = arg.split()[0] if arg else datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name.endswith(".json"):
            name += ".json"
        path = _history_dir() / f"code_{name}"
        payload = {
            "model": self.model_id,
            "workspace": str(self.workspace),
            "mode": self.mode.value,
            "files_in_context": list(self.context_files),
            "files_written": list(self.session_files),
            "messages": history,
        }
        try:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            self._status("error", f"Could not save: {e}")
            return
        self._status("success", f"Saved to {path}")

    def _cmd_session(self, arg: str) -> None:
        """Show, or begin persisting under, the resumable session id."""
        if not arg:
            if self.session_id:
                self._say(f"Session id: {self.session_id}")
                self._say(
                    "It is listed by `effgen sessions`; the conversation is saved as it grows."
                )
            else:
                self._say("This session is not being saved to the session store.")
                self._say("Start persisting:  /session <id>")
            return
        name = arg.split()[0]
        if self.session_id == name:
            self._say(f"Already on session '{name}'.")
            return
        if self.session_id:
            self._status("warning", f"Already saving to '{self.session_id}'.")
            return
        try:
            from effgen.core.session import Session

            session = Session.load_or_create(name)
            for msg in self._dump_history():
                if msg["role"] == "user":
                    session.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    session.add_assistant_message(msg["content"])
            session.metadata["model"] = self.model_id
            session.metadata["files_in_context"] = list(self.context_files)
            session.save()
            self.session_id = name
            if self.agent is not None:
                self.agent.session = session
                self.agent._session_id = name
            self._status("success", f"Saving this session as '{name}'.")
        except Exception as e:  # noqa: BLE001
            self._status("error", f"Could not start session '{name}': {e}")

    def _cmd_load(self, arg: str) -> None:
        files = sorted(_history_dir().glob("code_*.json"), reverse=True)
        if not files:
            self._say("No saved coding sessions found.")
            return
        if not arg:
            self._say("Saved coding sessions:")
            for i, f in enumerate(files[:20], 1):
                self._say(f"  {i}. {f.name}  ({f.stat().st_size} bytes)")
            self._say("Load with:  /load <name|number>")
            return
        token = arg.split()[0]
        target: Path | None = None
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(files):
                target = files[idx]
        else:
            cand = token if token.endswith(".json") else token + ".json"
            for f in files:
                if f.name in (cand, f"code_{cand}"):
                    target = f
                    break
        if target is None:
            self._status("warning", f"No saved session matching '{token}'.")
            return
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self._status("error", f"Could not read {target.name}: {e}")
            return
        messages = data.get("messages", [])
        saved_model = data.get("model")
        if saved_model and saved_model != self.model_id:
            self.model_id = saved_model
            try:
                self.engine = self._build_engine()
            except Exception as e:  # noqa: BLE001
                self._status("error", f"Could not load model '{saved_model}': {e}")
                return
        self.context_files = list(data.get("files_in_context", []) or [])
        self.session_files = list(data.get("files_written", []) or [])
        saved_mode = data.get("mode")
        if saved_mode:
            try:
                self.mode = PermissionMode(saved_mode)
                if self.engine is not None:
                    self.engine.gate.mode = self.mode
            except ValueError:
                pass
        try:
            self.agent.reset_memory()
            for msg in messages:
                if msg.get("role") == "user":
                    self.agent.short_term_memory.add_user_message(msg.get("content", ""))
                elif msg.get("role") in ("assistant", "agent"):
                    self.agent.short_term_memory.add_assistant_message(msg.get("content", ""))
        except Exception:  # noqa: BLE001
            pass
        self._status(
            "success",
            f"Loaded {target.name} ({len(messages)} messages, "
            f"{len(self.context_files)} file(s) in context, model {self.model_id})."
        )

    # ------------------------------------------------------------------
    # Help / doctor
    # ------------------------------------------------------------------
    def _cmd_help(self) -> None:
        if self.console and self.interactive and self._color:
            from rich.table import Table
            from rich.text import Text

            table = Table(title="Coding commands", show_header=True, header_style="cyan")
            table.add_column("Command", style="green", no_wrap=True)
            table.add_column("Does")
            for cmd, desc in _SLASH_COMMANDS.items():
                # Plain text, so a usage example such as "/apply [n]" keeps its
                # brackets instead of being read as console markup.
                table.add_row(cmd, Text(desc))
            self.console.print(table)
        else:
            self._say("Coding commands:")
            for cmd, desc in _SLASH_COMMANDS.items():
                self._say(f"  {cmd:9s} {desc}")

    def _cmd_doctor(self) -> None:
        """Report the provider keys, the system, and this session's readiness.

        The workspace is handed to ``doctor`` so its coding section describes
        the directory this session is confined to rather than the one the
        process was started in.
        """
        from effgen.cli._main import _handle_doctor_command

        class _DoctorArgs:
            doctor_provider = self.provider
            live = False
            output_json = False
            workspace = str(self.workspace)

        try:
            _handle_doctor_command(_DoctorArgs())
        except Exception as e:  # noqa: BLE001
            self._status("error", f"doctor failed: {e}")
