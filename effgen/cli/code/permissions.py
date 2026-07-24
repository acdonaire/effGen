"""Permission modes and the gate that applies them to a coding run.

A coding agent's two side effects are *writing files* and *executing commands*.
Both go through :class:`PermissionGate` before they reach the underlying tool, so
a run's disk and process effects are decided by an explicit policy rather than by
whatever the model happened to call.

Four action kinds are distinguished, because they carry different risk:

- ``write`` — a file write into the workspace (``file_operations``).
- ``run`` — code executed in the configured sandbox (``code_executor`` with a
  sandboxed language, ``python_repl``).
- ``shell`` — a shell command (``bash``, or ``code_executor`` with
  ``language="bash"``). The shell runs with the caller's own permissions, so it
  is gated more tightly than a sandboxed run.
- ``git`` — a commit of the files a run wrote (:mod:`effgen.cli.code.git_actions`).
  No tool can request one: it is only ever asked for by the person driving the
  session, and it still passes the gate so the decision is recorded like any
  other.

The four modes:

============  ==========  ==========  ==========  ==========
mode          write       run         shell       git
============  ==========  ==========  ==========  ==========
``plan``      deny        deny        deny        deny
``ask``       confirm     confirm     confirm     confirm
``auto-edit`` allow       allow       confirm     confirm
``yes``       allow       allow       allow       allow
============  ==========  ==========  ==========  ==========

``confirm`` asks the human and needs a terminal. Without one there is nobody to
ask, so the action is withheld and recorded — a non-interactive run never writes
or executes on an implied "yes". Every decision, including the refusals, lands in
:attr:`PermissionGate.actions` so the caller can report exactly what happened and
what did not.

Confinement is enforced here too: :meth:`PermissionGate.resolve_path` resolves a
path against the workspace root and refuses anything that escapes it or that
carries a credentials-file name, before the tool is reached.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from effgen.tools.builtin._fs import PathNotAllowedError, is_credential_filename

from .edits import EditJournal, ProposedEdit

#: Action kinds the gate distinguishes, in the order they are reported.
ACTION_KINDS: tuple[str, ...] = ("write", "run", "shell", "git")

# Policy verbs.
_ALLOW = "allow"
_CONFIRM = "confirm"
_DENY = "deny"


class PermissionMode(str, Enum):
    """How a coding run treats file writes and command execution."""

    PLAN = "plan"
    ASK = "ask"
    AUTO_EDIT = "auto-edit"
    YES = "yes"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


_POLICY: dict[PermissionMode, dict[str, str]] = {
    PermissionMode.PLAN: {"write": _DENY, "run": _DENY, "shell": _DENY, "git": _DENY},
    PermissionMode.ASK: {
        "write": _CONFIRM, "run": _CONFIRM, "shell": _CONFIRM, "git": _CONFIRM,
    },
    PermissionMode.AUTO_EDIT: {
        "write": _ALLOW, "run": _ALLOW, "shell": _CONFIRM, "git": _CONFIRM,
    },
    PermissionMode.YES: {"write": _ALLOW, "run": _ALLOW, "shell": _ALLOW, "git": _ALLOW},
}

#: One-line description per mode, used by ``--help`` and the run summary.
MODE_DESCRIPTIONS: dict[PermissionMode, str] = {
    PermissionMode.PLAN: "propose only; no file is written and no command runs",
    PermissionMode.ASK: "confirm every write and every command",
    PermissionMode.AUTO_EDIT: "apply writes and sandboxed runs; confirm shell commands",
    PermissionMode.YES: "apply writes, runs and shell commands without asking",
}


def default_mode(interactive: bool) -> PermissionMode:
    """Return the mode to use when the caller named none.

    A terminal session can ask, so it defaults to :attr:`PermissionMode.ASK`.
    Without a terminal there is nobody to ask, so it defaults to
    :attr:`PermissionMode.PLAN`: the run reports what it would do and the caller
    opts in with ``--auto-edit`` or ``--yes``.
    """
    return PermissionMode.ASK if interactive else PermissionMode.PLAN


@dataclass
class ActionRecord:
    """One write/run/shell request and what the gate decided about it.

    ``decision`` is one of:

    - ``"allowed"`` — the action ran (see ``outcome`` for whether it succeeded).
    - ``"withheld"`` — the mode did not permit it and no human was available to
      ask; nothing happened.
    - ``"declined"`` — a human was asked and answered no.
    - ``"refused"`` — the target was outside the workspace or otherwise blocked
      by confinement; nothing happened.
    """

    kind: str
    summary: str
    decision: str
    reason: str = ""
    target: str | None = None
    outcome: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-serializable dict."""
        return {
            "kind": self.kind,
            "summary": self.summary,
            "decision": self.decision,
            "reason": self.reason,
            "target": self.target,
            "outcome": self.outcome,
            "detail": self.detail,
        }


@dataclass
class Decision:
    """The gate's answer to one request, plus the record it appended."""

    allowed: bool
    reason: str
    record: ActionRecord

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.allowed


def _prompt_confirm(question: str) -> str:
    """Ask *question* on the terminal and return ``y``, ``n`` or ``a``.

    Written to stderr so a piped stdout stays clean. Anything other than
    ``y``/``yes`` or ``a``/``always`` reads as no, and an interrupted or closed
    input also reads as no.
    """
    try:
        print(f"{question} [y/N/a] ", end="", file=sys.stderr, flush=True)
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        print(file=sys.stderr)
        return "n"
    if answer in ("y", "yes"):
        return "y"
    if answer in ("a", "always"):
        return "a"
    return "n"


class PermissionGate:
    """Applies a :class:`PermissionMode` to every write and command of a run.

    Args:
        mode: The active permission mode.
        workspace: The one directory the run may read and write. Every path the
            gate resolves must land inside it.
        mode_explicit: True when the caller named the mode on the command line.
            A mode that was merely defaulted (because there was no terminal) is
            reported differently, so a piped run that withheld work can exit
            non-zero while ``--plan`` stays a successful proposal.
        interactive: Whether a human can be asked. False turns every ``confirm``
            into a withheld action.
        confirm: Callable taking a question and returning ``y``/``n``/``a``.
            Defaults to a stderr prompt. Injected by tests and by callers that
            drive their own UI.
        on_event: Optional callback invoked with each :class:`ActionRecord` as it
            is decided, for live progress output.
        on_diff: Optional callback invoked with each :class:`ProposedEdit` before
            a write is decided, so its diff can be shown before it touches disk.
        journal: Optional :class:`EditJournal`; when set, every applied write is
            recorded so ``effgen code --undo`` can reverse it.
    """

    def __init__(
        self,
        mode: PermissionMode,
        workspace: Path,
        *,
        mode_explicit: bool = False,
        interactive: bool = False,
        confirm: Callable[[str], str] | None = None,
        on_event: Callable[[ActionRecord], None] | None = None,
        on_diff: Callable[[ProposedEdit], None] | None = None,
        journal: EditJournal | None = None,
    ) -> None:
        self.mode = mode
        self.workspace = Path(workspace)
        self.mode_explicit = mode_explicit
        self.interactive = interactive
        self._confirm = confirm or _prompt_confirm
        self._on_event = on_event
        self._on_diff = on_diff
        self.journal = journal
        self.actions: list[ActionRecord] = []
        # A summary of each edit that reached disk, in write order, for the run
        # report and ``--json`` output.
        self.edits: list[dict[str, Any]] = []
        # Kinds the human answered "always" to for the rest of this run.
        self._always: set[str] = set()

    # -- confinement --------------------------------------------------------

    def resolve_path(self, path: str) -> Path:
        """Resolve *path* inside the workspace, or raise.

        A relative path is resolved against the workspace root, so the model's
        ``"fib.py"`` lands in the workspace rather than the process's current
        directory. Symlinks are followed before the containment check.

        Raises:
            PathNotAllowedError: The resolved path is outside the workspace, or
                its filename matches a common credentials shape.
        """
        try:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = self.workspace / candidate
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:
            raise PathNotAllowedError(f"Invalid path {path!r}: {exc}") from exc

        if is_credential_filename(resolved.name):
            raise PathNotAllowedError(
                f"Refusing to write {path!r}: the filename matches a common "
                "credentials shape (.env, id_rsa, credentials, ...)."
            )

        root = self.workspace.resolve()
        if resolved != root and root not in resolved.parents:
            raise PathNotAllowedError(
                f"Refusing to write {path!r}: it resolves to {resolved}, which is "
                f"outside the workspace {root}. Files stay inside the workspace; "
                "point -w/--workspace at another directory to work there."
            )
        return resolved

    # -- decisions ----------------------------------------------------------

    def request(
        self, kind: str, summary: str, *, target: str | None = None
    ) -> Decision:
        """Decide whether an action of *kind* may proceed.

        Args:
            kind: One of :data:`ACTION_KINDS`.
            summary: One line describing the action, shown when confirming.
            target: The file path or command the action applies to.
        """
        policy = _POLICY[self.mode].get(kind, _DENY)
        if kind in self._always:
            policy = _ALLOW

        if policy == _ALLOW:
            return self._record(kind, summary, "allowed", "", target)

        if policy == _DENY:
            return self._record(
                kind, summary, "withheld",
                f"{self.mode.value} mode does not {_VERB[kind]}; "
                f"re-run with {_OPT_IN[kind]} to allow it.",
                target,
            )

        # policy == confirm
        if not self.interactive:
            return self._record(
                kind, summary, "withheld",
                f"no terminal to confirm on; re-run with {_OPT_IN[kind]} to "
                f"allow it without asking.",
                target,
            )
        answer = self._confirm(summary)
        if answer == "a":
            self._always.add(kind)
            return self._record(kind, summary, "allowed", "", target)
        if answer == "y":
            return self._record(kind, summary, "allowed", "", target)
        return self._record(
            kind, summary, "declined", "declined at the prompt", target
        )

    def refuse(self, kind: str, summary: str, reason: str, *, target: str | None = None) -> Decision:
        """Record an action blocked by confinement, before any policy applies."""
        return self._record(kind, summary, "refused", reason, target)

    def _record(
        self, kind: str, summary: str, decision: str, reason: str, target: str | None
    ) -> Decision:
        record = ActionRecord(
            kind=kind, summary=summary, decision=decision, reason=reason, target=target
        )
        self.actions.append(record)
        if self._on_event is not None:
            self._on_event(record)
        return Decision(allowed=decision == "allowed", reason=reason, record=record)

    def note_outcome(self, record: ActionRecord, outcome: str, detail: str = "") -> None:
        """Attach the result of an allowed action to its record."""
        record.outcome = outcome
        record.detail = detail
        if self._on_event is not None:
            self._on_event(record)

    # -- edits --------------------------------------------------------------

    def announce_edit(self, edit: ProposedEdit) -> None:
        """Surface a pending edit's diff before the write is decided."""
        if self._on_diff is not None:
            self._on_diff(edit)

    def record_applied_edit(
        self,
        edit: ProposedEdit,
        *,
        before: str | None,
        after: str,
        failed_hunks: int = 0,
    ) -> None:
        """Record a write that reached disk: onto the undo stack and the report.

        *before* is the file's content before the write (``None`` when the run
        created it), captured so ``--undo`` can restore it. *failed_hunks* is how
        many of the edit's hunks did not apply because the file changed
        underneath — zero for the common case.
        """
        if self.journal is not None:
            self.journal.record_write(edit.rel_path, before, after)
        self.edits.append(
            {
                "path": edit.rel_path,
                "added": edit.added,
                "removed": edit.removed,
                "is_new": edit.is_new,
                "hunks_failed": failed_hunks,
                "diff": edit.diff_text,
            }
        )

    # -- reporting ----------------------------------------------------------

    def begin_turn(self) -> None:
        """Clear the per-turn action and edit log before a new REPL turn.

        Each interactive turn reports only its own writes, runs and refusals, so
        the log is reset here between turns. The "always" answers the human gave
        and the on-disk undo journal both persist for the whole session and are
        left untouched.
        """
        self.actions = []
        self.edits = []

    @property
    def withheld(self) -> list[ActionRecord]:
        """Actions the mode did not permit (excludes explicit human declines)."""
        return [a for a in self.actions if a.decision == "withheld"]

    @property
    def refused(self) -> list[ActionRecord]:
        """Actions blocked by workspace confinement."""
        return [a for a in self.actions if a.decision == "refused"]

    @property
    def files_written(self) -> list[str]:
        """Workspace-relative paths written by this run, in write order."""
        seen: list[str] = []
        for action in self.actions:
            if action.kind != "write" or action.decision != "allowed":
                continue
            if action.outcome not in (None, "ok"):
                continue
            if action.target and action.target not in seen:
                seen.append(action.target)
        return seen

    def summary_counts(self) -> dict[str, int]:
        """Return ``{decision: count}`` across every recorded action."""
        counts: dict[str, int] = {}
        for action in self.actions:
            counts[action.decision] = counts.get(action.decision, 0) + 1
        return counts


# The wording used in a withheld-action message, per kind.
_VERB: dict[str, str] = {
    "write": "write files",
    "run": "run code",
    "shell": "run shell commands",
    "git": "commit to git",
}

# The flag that turns a withheld action of each kind into an allowed one.
_OPT_IN: dict[str, str] = {
    "write": "--auto-edit (or --yes)",
    "run": "--auto-edit (or --yes)",
    "shell": "--yes",
    "git": "--yes",
}
