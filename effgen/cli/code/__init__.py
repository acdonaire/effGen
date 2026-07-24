"""The coding-agent surface behind ``effgen code``.

``effgen code`` runs a plan → act → observe loop over the ``coding`` preset: the
model proposes an approach, writes files, executes code in the configured
sandbox, reads the real output, and iterates until the task is done or the
iteration cap is reached. Everything it can touch is confined to one workspace
root, and every write or command it wants to run passes a permission gate first.

The pieces:

- :mod:`~effgen.cli.code.permissions` — the permission modes (``plan``, ``ask``,
  ``auto-edit``, ``yes``), the gate that applies them, and the action log.
- :mod:`~effgen.cli.code.diffs` — unified-diff generation, colorized rendering,
  and hunk-level application for the edit preview.
- :mod:`~effgen.cli.code.edits` — a pending edit (:class:`ProposedEdit`) and the
  per-workspace undo journal (:class:`EditJournal`) behind ``--undo``.
- :mod:`~effgen.cli.code.tools` — the ``coding`` preset's tools with the gate
  wired in front of their write/execute paths. The tool names and schemas the
  model sees are unchanged.
- :mod:`~effgen.cli.code.engine` — workspace resolution, agent construction, and
  the run-result record the CLI prints or serializes.

The command handler lives in :mod:`effgen.cli.commands.code`.
"""

from __future__ import annotations

from .diffs import (
    ApplyResult,
    DiffStat,
    Hunk,
    apply_hunks,
    diff_stat,
    render_diff,
    split_hunks,
    unified_diff_text,
)
from .edits import AppliedEdit, EditJournal, ProposedEdit, UndoOutcome
from .engine import (
    CodeEngine,
    CodeRunResult,
    resolve_workspace,
    undo_workspace,
    workspace_env,
    workspace_execution_note,
)
from .permissions import (
    ACTION_KINDS,
    ActionRecord,
    Decision,
    PermissionGate,
    PermissionMode,
    default_mode,
)
from .tools import build_code_tools

__all__ = [
    "ACTION_KINDS",
    "ActionRecord",
    "AppliedEdit",
    "ApplyResult",
    "CodeEngine",
    "CodeRunResult",
    "Decision",
    "DiffStat",
    "EditJournal",
    "Hunk",
    "PermissionGate",
    "PermissionMode",
    "ProposedEdit",
    "UndoOutcome",
    "apply_hunks",
    "build_code_tools",
    "default_mode",
    "diff_stat",
    "render_diff",
    "resolve_workspace",
    "split_hunks",
    "undo_workspace",
    "unified_diff_text",
    "workspace_env",
    "workspace_execution_note",
]
