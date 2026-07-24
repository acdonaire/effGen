"""The ``coding`` preset's tools with a permission gate in front of them.

Each class here subclasses the shipped tool and overrides only ``_execute``: it
asks the :class:`~effgen.cli.code.permissions.PermissionGate` whether the write
or command may proceed, then delegates to the original implementation. Nothing
about the tool's name, description or parameter schema changes, so the model
sees exactly the same ``coding`` toolbox and the existing sandbox, path
confinement and blocked-command lists all still apply underneath.

The shell carries one extra limit: a command that would publish, rewrite or
discard git history (``git push``, ``git reset --hard``, ``git checkout``, ...)
is refused in every permission mode, so the narrow confirmed-commit action in
:mod:`effgen.cli.code.git_actions` stays the only way a session changes a
repository. Reading a repository from the shell is unaffected.

A withheld action returns a normal tool failure (``{"success": False, "error":
...}``) naming what was blocked and which flag would allow it. The agent reads
that as an observation and can adjust — a refusal is visible to the model, never
a silent no-op.
"""

from __future__ import annotations

from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from effgen.tools.base_tool import BaseTool
from effgen.tools.builtin._fs import PathNotAllowedError
from effgen.tools.builtin.bash_tool import BashTool
from effgen.tools.builtin.code_executor import CodeExecutor
from effgen.tools.builtin.file_ops import FileOperations
from effgen.tools.builtin.python_repl import PythonREPL

from .edits import ProposedEdit
from .git_actions import unsafe_shell_git
from .permissions import Decision, PermissionGate

# ``file_operations`` operations that put bytes on disk. Reads, listings,
# searches and metadata lookups are not gated: they are already confined to the
# workspace and a coding agent needs them to see what it is working on.
_WRITING_OPERATIONS = frozenset({"write", "convert"})

# Languages ``code_executor`` runs inside the sandbox. Anything else it accepts
# (``bash``) is treated as a shell command and gated as one.
_SANDBOXED_LANGUAGES = frozenset({"python", "javascript", "js", "node"})


def _blocked(reason: str) -> dict[str, Any]:
    """Return the tool-failure envelope used for every gated refusal."""
    return {"success": False, "error": reason}


def _refuse_unsafe_git(gate: PermissionGate, summary: str, command: str) -> dict[str, Any] | None:
    """Refuse a shell command that publishes, rewrites or discards git history.

    A session reads a repository from the shell as much as it likes; the one
    repository-changing action it offers is the confirmed commit, so a ``git
    push``/``reset --hard``/``checkout``-shaped command is refused in every
    permission mode rather than confirmed. ``None`` means the command may go on
    to the gate.
    """
    reason = unsafe_shell_git(command)
    if reason is None:
        return None
    decision = gate.refuse(
        "shell",
        summary,
        f"{reason} Commit the run's edits with --commit (or /git commit); run "
        "anything else yourself.",
        target=_shorten(command, 200),
    )
    return _blocked(decision.reason)


def _shorten(text: str, limit: int = 120) -> str:
    """Collapse *text* to a single line of at most *limit* characters."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _read_text_or_none(path: Path) -> str | None:
    """Return the file's text, or ``None`` when it is absent or not UTF-8 text.

    ``None`` means "do not diff or snapshot this write" — a binary or unreadable
    file falls back to the plain gated write with no preview, rather than a
    corrupt diff or a lossy undo snapshot.
    """
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


async def _recorded(gate: PermissionGate, decision: Decision, coro: Awaitable[Any]) -> Any:
    """Await *coro*, recording an error outcome on *decision* if it raises.

    A tool that raises — a command the shell's blocked list rejects, a sandbox
    that could not start — is turned into a failed result by
    :meth:`BaseTool.execute` further up. Recording the failure here keeps the
    action log and the live ticks accurate: the action is reported as attempted
    and failed, with the reason, rather than left with no outcome at all.
    """
    try:
        return await coro
    except Exception as exc:
        gate.note_outcome(decision.record, "error", _shorten(str(exc)))
        raise


class GatedFileOperations(FileOperations):
    """``file_operations`` whose writes are previewed, gated, and reversible.

    A write is shown as a unified diff before it is decided, applied against the
    file's current content hunk-by-hunk (so a file that changed underneath is not
    overwritten whole), and recorded on the run's undo journal.
    """

    def __init__(self, gate: PermissionGate, **kwargs: Any) -> None:
        super().__init__(allowed_directories=[str(gate.workspace)], **kwargs)
        self.gate = gate

    async def _execute(  # type: ignore[override]
        self, operation: str, path: str, content: str | None = None, **kwargs: Any
    ) -> Any:
        if operation not in _WRITING_OPERATIONS:
            return await super()._execute(operation, path, content=content, **kwargs)
        if operation != "write":
            # ``convert`` writes derived output; gate it without a text diff.
            return await self._gated_plain_write(operation, path, content, **kwargs)

        summary = f"write {path}"
        try:
            resolved = self.gate.resolve_path(path)
        except PathNotAllowedError as exc:
            decision = self.gate.refuse("write", summary, str(exc), target=path)
            return _blocked(decision.reason)

        try:
            rel = str(resolved.relative_to(self.gate.workspace.resolve()))
        except ValueError:  # pragma: no cover - resolve_path guarantees containment
            rel = str(resolved)

        before = _read_text_or_none(resolved)
        if before is None:
            # Not diffable text: fall back to a plain gated write, no snapshot.
            return await self._gated_plain_write("write", path, content, **kwargs)

        edit = ProposedEdit(
            rel_path=rel,
            old_content=before,
            new_content=content or "",
            is_new=not resolved.exists(),
        )
        if not edit.unchanged:
            self.gate.announce_edit(edit)

        decision = self.gate.request("write", f"Write {rel} ({edit.stat()})", target=rel)
        if not decision.allowed:
            return _blocked(decision.reason)

        # Re-read at apply time and merge, so a file that changed since the diff
        # was shown keeps the hunks that still apply rather than being clobbered.
        current = _read_text_or_none(resolved)
        if current is None:  # became unreadable between preview and write
            current = before
        final, _applied, failed = edit.resolve_against_current(current)
        snapshot = None if not resolved.exists() else current

        result = await _recorded(
            self.gate, decision, super()._execute("write", path, content=final, **kwargs)
        )
        failed_write = isinstance(result, dict) and result.get("success") is False
        if failed_write:
            self.gate.note_outcome(
                decision.record, "error", _shorten(result.get("error", ""))
            )
            return result

        detail = ""
        if failed:
            detail = (
                f"{len(failed)} hunk(s) did not apply (the file changed since it "
                "was read); the rest were applied."
            )
        self.gate.note_outcome(decision.record, "ok", detail)
        self.gate.record_applied_edit(
            edit, before=snapshot, after=final, failed_hunks=len(failed)
        )
        return result

    async def _gated_plain_write(
        self, operation: str, path: str, content: str | None, **kwargs: Any
    ) -> Any:
        """Gate a write that carries no text diff (``convert``, binary target)."""
        summary = f"write {path}"
        try:
            resolved = self.gate.resolve_path(path)
        except PathNotAllowedError as exc:
            decision = self.gate.refuse("write", summary, str(exc), target=path)
            return _blocked(decision.reason)
        try:
            rel = str(resolved.relative_to(self.gate.workspace.resolve()))
        except ValueError:  # pragma: no cover - resolve_path guarantees containment
            rel = str(resolved)

        decision = self.gate.request("write", f"Write {rel}", target=rel)
        if not decision.allowed:
            return _blocked(decision.reason)
        result = await _recorded(
            self.gate, decision, super()._execute(operation, path, content=content, **kwargs)
        )
        failed = isinstance(result, dict) and result.get("success") is False
        self.gate.note_outcome(
            decision.record,
            "error" if failed else "ok",
            _shorten(result.get("error", "")) if failed else "",
        )
        return result


class GatedCodeExecutor(CodeExecutor):
    """``code_executor`` whose runs pass the permission gate first."""

    def __init__(self, gate: PermissionGate) -> None:
        super().__init__(workdir=str(gate.workspace))
        self.gate = gate

    async def _execute(  # type: ignore[override]
        self, code: str, language: str, **kwargs: Any
    ) -> Any:
        lang = (language or "").strip().lower()
        kind = "run" if lang in _SANDBOXED_LANGUAGES else "shell"
        summary = f"Run {lang or 'code'}: {_shorten(code, 80)}"
        if kind == "shell":
            refusal = _refuse_unsafe_git(self.gate, summary, code)
            if refusal is not None:
                return refusal
        decision = self.gate.request(kind, summary, target=lang or "code")
        if not decision.allowed:
            return _blocked(decision.reason)

        result = await _recorded(
            self.gate, decision, super()._execute(code, language, **kwargs)
        )
        failed = isinstance(result, dict) and result.get("success") is False
        self.gate.note_outcome(
            decision.record,
            "error" if failed else "ok",
            _shorten(result.get("error", "")) if failed else "",
        )
        return result


class GatedPythonREPL(PythonREPL):
    """``python_repl`` whose executions pass the permission gate first."""

    def __init__(self, gate: PermissionGate) -> None:
        super().__init__()
        self.gate = gate

    async def _execute(self, code: str, **kwargs: Any) -> Any:  # type: ignore[override]
        summary = f"Run python: {_shorten(code, 80)}"
        decision = self.gate.request("run", summary, target="python_repl")
        if not decision.allowed:
            return _blocked(decision.reason)

        result = await _recorded(self.gate, decision, super()._execute(code, **kwargs))
        failed = isinstance(result, dict) and result.get("success") is False
        self.gate.note_outcome(
            decision.record,
            "error" if failed else "ok",
            _shorten(result.get("error", "")) if failed else "",
        )
        return result


class GatedBashTool(BashTool):
    """``bash`` whose commands pass the permission gate first."""

    def __init__(self, gate: PermissionGate) -> None:
        super().__init__(working_directory=str(gate.workspace))
        self.gate = gate

    async def _execute(self, command: str, **kwargs: Any) -> Any:  # type: ignore[override]
        summary = f"Run shell: {_shorten(command, 100)}"
        refusal = _refuse_unsafe_git(self.gate, summary, command)
        if refusal is not None:
            return refusal
        decision = self.gate.request("shell", summary, target=_shorten(command, 200))
        if not decision.allowed:
            return _blocked(decision.reason)

        result = await _recorded(self.gate, decision, super()._execute(command, **kwargs))
        failed = isinstance(result, dict) and result.get("success") is False
        self.gate.note_outcome(
            decision.record,
            "error" if failed else "ok",
            _shorten(result.get("error", "")) if failed else "",
        )
        return result


def build_code_tools(gate: PermissionGate) -> list[BaseTool]:
    """Return the ``coding`` preset's tools, each wired to *gate*.

    The order matches the preset so the model's tool listing is unchanged.
    """
    return [
        GatedCodeExecutor(gate),
        GatedPythonREPL(gate),
        GatedFileOperations(gate),
        GatedBashTool(gate),
    ]
