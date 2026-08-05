"""The slash commands that retune, inspect or persist a coding session.

:mod:`effgen.cli.code.repl` owns the session itself and mixes these methods into
:class:`~effgen.cli.code.repl.CodeREPL`, so ``effgen.cli.code.repl`` stays the
import path callers use. Holds the commands that act on the session rather than
on the working set: the permission mode and the model hot-swap, the cost and
reasoning-trace views, the git surface and the confirmed commit, the memory
reset/clear/compact, the save/session/load round trip, and the environment
check.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from effgen.cli.chat import _history_dir
from effgen.cli.code.engine import workspace_env
from effgen.cli.code.permissions import MODE_DESCRIPTIONS, PermissionMode
from effgen.cli.code.project import staged_diff


class CodeSessionMixin:
    """The session-control commands of :class:`~effgen.cli.code.repl.CodeREPL`.

    Every method reads state ``CodeREPL.__init__`` sets — ``self.engine``,
    ``self.mode``, ``self.model_id``, ``self.session_files`` and the session
    counters — and prints through the view methods.
    """

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
