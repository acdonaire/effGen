"""The ``coding`` preset's tools with a permission gate in front of them.

Each class here subclasses the shipped tool and overrides only ``_execute``: it
asks the :class:`~effgen.cli.code.permissions.PermissionGate` whether the write
or command may proceed, then delegates to the original implementation. Nothing
about the tool's name, description or parameter schema changes, so the model
sees exactly the same ``coding`` toolbox and the existing sandbox, path
confinement and blocked-command lists all still apply underneath.

A withheld action returns a normal tool failure (``{"success": False, "error":
...}``) naming what was blocked and which flag would allow it. The agent reads
that as an observation and can adjust — a refusal is visible to the model, never
a silent no-op.
"""

from __future__ import annotations

from typing import Any

from effgen.tools.base_tool import BaseTool
from effgen.tools.builtin._fs import PathNotAllowedError
from effgen.tools.builtin.bash_tool import BashTool
from effgen.tools.builtin.code_executor import CodeExecutor
from effgen.tools.builtin.file_ops import FileOperations
from effgen.tools.builtin.python_repl import PythonREPL

from .permissions import PermissionGate

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


def _shorten(text: str, limit: int = 120) -> str:
    """Collapse *text* to a single line of at most *limit* characters."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class GatedFileOperations(FileOperations):
    """``file_operations`` whose write paths pass the permission gate first."""

    def __init__(self, gate: PermissionGate, **kwargs: Any) -> None:
        super().__init__(allowed_directories=[str(gate.workspace)], **kwargs)
        self.gate = gate

    async def _execute(  # type: ignore[override]
        self, operation: str, path: str, content: str | None = None, **kwargs: Any
    ) -> Any:
        if operation not in _WRITING_OPERATIONS:
            return await super()._execute(operation, path, content=content, **kwargs)

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

        result = await super()._execute(operation, path, content=content, **kwargs)
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
        decision = self.gate.request(kind, summary, target=lang or "code")
        if not decision.allowed:
            return _blocked(decision.reason)

        result = await super()._execute(code, language, **kwargs)
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

        result = await super()._execute(code, **kwargs)
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
        decision = self.gate.request("shell", summary, target=_shorten(command, 200))
        if not decision.allowed:
            return _blocked(decision.reason)

        result = await super()._execute(command, **kwargs)
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
