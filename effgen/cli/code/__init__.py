"""The coding-agent surface behind ``effgen code``.

``effgen code`` runs a plan → act → observe loop over the ``coding`` preset: the
model proposes an approach, writes files, executes code in the configured
sandbox, reads the real output, and iterates until the task is done or the
iteration cap is reached. Everything it can touch is confined to one workspace
root, and every write or command it wants to run passes a permission gate first.

The pieces:

- :mod:`~effgen.cli.code.permissions` — the permission modes (``plan``, ``ask``,
  ``auto-edit``, ``yes``), the gate that applies them, and the action log.
- :mod:`~effgen.cli.code.tools` — the ``coding`` preset's tools with the gate
  wired in front of their write/execute paths. The tool names and schemas the
  model sees are unchanged.
- :mod:`~effgen.cli.code.engine` — workspace resolution, agent construction, and
  the run-result record the CLI prints or serializes.

The command handler lives in :mod:`effgen.cli.commands.code`.
"""

from __future__ import annotations

from .engine import (
    CodeEngine,
    CodeRunResult,
    resolve_workspace,
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
    "CodeEngine",
    "CodeRunResult",
    "Decision",
    "PermissionGate",
    "PermissionMode",
    "build_code_tools",
    "default_mode",
    "resolve_workspace",
    "workspace_env",
    "workspace_execution_note",
]
