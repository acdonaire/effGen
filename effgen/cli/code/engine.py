"""Workspace resolution, agent construction and the run record for ``effgen code``.

:class:`CodeEngine` assembles a coding agent from parts that already ship: the
``coding`` preset's system prompt and iteration budget, its four tools with a
permission gate in front of them (:mod:`effgen.cli.code.tools`), and the ReAct
loop that feeds every tool result back into the next step. One run returns a
:class:`CodeRunResult` carrying the answer, what the agent did to the workspace,
what was withheld, and the usual token/cost/timing numbers.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from effgen.tools.builtin._fs import WORKSPACE_ENV_VAR, default_workspace

from .edits import EditJournal, ProposedEdit, UndoOutcome
from .permissions import ActionRecord, PermissionGate, PermissionMode
from .tools import build_code_tools

#: The preset whose prompt, tools and iteration budget ``effgen code`` builds on.
CODE_PRESET = "coding"

#: Added to the preset prompt so the model works inside the one workspace root
#: and treats a gate refusal as a real constraint rather than something to retry.
_WORKSPACE_PROMPT = (
    "You are working inside the directory {workspace}. Read and write files "
    "with relative paths inside it; never write outside it. To create or change "
    "a file, use the file tool's write operation — not shell redirection such as "
    "'cat > file', 'tee', or 'sed -i' — so each change is shown as a diff before "
    "it is applied and can be undone. Run code with the code execution tool, "
    "which starts in that directory, so a relative path such as 'main.py' reads "
    "the file you just wrote. The Python REPL tool evaluates short expressions in "
    "a restricted namespace with a small import allow-list; it cannot import "
    "project files, so use the code execution tool for anything that touches the "
    "workspace. Base every claim on a tool's real output. If a tool reports that "
    "an action was not permitted, tell the user instead of retrying the same "
    "action or answering as if it had succeeded."
)


def resolve_workspace(explicit: str | None = None) -> Path:
    """Return the directory a coding run is confined to.

    An explicit ``-w/--workspace`` wins; otherwise ``EFFGEN_WORKSPACE`` when it
    is set; otherwise the current directory. The directory is created if it does
    not exist, so a fresh workspace can be named on the command line.

    Raises:
        OSError: The directory could not be created.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    configured = default_workspace()
    if configured is not None:
        return configured
    return Path.cwd().resolve()


def workspace_execution_note(workspace: Path) -> str | None:
    """Return a note when executed code cannot reach *workspace*, else ``None``.

    The subprocess sandbox mounts a private ``tmpfs`` over ``/tmp``, so a
    workspace inside the system temp directory is written by the file tool but
    invisible to the code the sandbox runs — an import of a file the agent just
    created would fail with a confusing "no such module". This runs one short
    probe through the configured sandbox and reports the mismatch up front
    instead of leaving it to surface mid-loop. Any error in the probe returns
    ``None``: it informs, it never blocks a run.
    """
    import asyncio

    from effgen.security.sandbox import SandboxConfig, get_sandbox

    probe = "import os\nprint('visible' if os.path.isdir(os.getcwd()) else 'hidden')\n"

    async def _probe() -> str:
        config = SandboxConfig.from_env()
        config.workdir = str(workspace)
        sandbox = await get_sandbox(config)
        result = await sandbox.run(code=probe, language="python", config=config)
        return (result.stdout or "").strip()

    try:
        verdict = asyncio.run(_probe())
    except Exception:  # noqa: BLE001 - a diagnostic never breaks the run
        return None
    if verdict != "hidden":
        return None
    return (
        f"Code executed in the sandbox cannot see {workspace}: the sandbox gives "
        "executed code a private temp directory, so a workspace under the system "
        "temp directory is not readable from it. Files written there are real, "
        "but running them will fail. Pass -w/--workspace with a directory outside "
        "the system temp directory."
    )


@contextmanager
def workspace_env(workspace: Path) -> Iterator[Path]:
    """Set ``EFFGEN_WORKSPACE`` to *workspace* for the duration of the block.

    Every file and shell tool resolves its root from that variable, so setting it
    once makes the whole run agree on a single directory even when the caller
    named it with ``-w``. The previous value is restored on exit.
    """
    previous = os.environ.get(WORKSPACE_ENV_VAR)
    os.environ[WORKSPACE_ENV_VAR] = str(workspace)
    try:
        yield workspace
    finally:
        if previous is None:
            os.environ.pop(WORKSPACE_ENV_VAR, None)
        else:
            os.environ[WORKSPACE_ENV_VAR] = previous


def undo_workspace(workspace: Path, count: int = 1) -> tuple[list[UndoOutcome], int]:
    """Reverse the last *count* applied edits in *workspace*.

    Returns the outcomes performed (newest first) and how many edits remain on
    the stack afterwards. An empty stack yields no outcomes.
    """
    journal = EditJournal(Path(workspace))
    outcomes: list[UndoOutcome] = []
    for _ in range(max(1, count)):
        outcome = journal.undo()
        if outcome is None:
            break
        outcomes.append(outcome)
    return outcomes, len(journal)


@dataclass
class CodeRunResult:
    """The outcome of one ``effgen code`` task."""

    task: str
    answer: str
    success: bool
    reason: str
    model: str
    provider: str | None
    workspace: str
    permission_mode: str
    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float | None = None
    duration_s: float = 0.0
    partial: bool = False
    actions: list[ActionRecord] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    diffs: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None

    # ``effgen.ui.render.summary_line`` reads a run's metrics off these names,
    # so the coding footer is the same one ``effgen run`` prints.
    @property
    def execution_time(self) -> float:
        """Wall-clock seconds the run took."""
        return self.duration_s

    @property
    def tokens_used(self) -> int:
        """Total prompt + completion tokens for the run."""
        return self.tokens

    @property
    def metadata(self) -> dict[str, Any]:
        """The subset of run metadata the shared renderers read."""
        return {"reason": self.reason, "cost_usd": self.cost_usd, "partial": self.partial}

    @property
    def withheld(self) -> list[ActionRecord]:
        """Actions the permission mode did not allow."""
        return [a for a in self.actions if a.decision == "withheld"]

    @property
    def refused(self) -> list[ActionRecord]:
        """Actions blocked because their target was outside the workspace."""
        return [a for a in self.actions if a.decision == "refused"]

    @property
    def hit_iteration_cap(self) -> bool:
        """True when the loop stopped at its iteration cap without an answer."""
        return self.reason in ("max_iterations_partial", "max_iterations_exhausted")

    def to_dict(self) -> dict[str, Any]:
        """Return the result as the JSON document ``--json`` prints."""
        return {
            "task": self.task,
            "answer": self.answer,
            "success": self.success,
            "reason": self.reason,
            "model": self.model,
            "provider": self.provider,
            "workspace": self.workspace,
            "permission_mode": self.permission_mode,
            "files_written": list(self.files_written),
            "diffs": [dict(d) for d in self.diffs],
            "actions": [a.to_dict() for a in self.actions],
            "withheld": [a.to_dict() for a in self.withheld],
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
        }


class CodeEngine:
    """Builds and runs the coding agent behind ``effgen code``.

    Args:
        model: Model id, optionally ``provider:model`` prefixed.
        provider: Explicit provider for a bare model id.
        workspace: The directory the run is confined to.
        mode: The active :class:`~effgen.cli.code.permissions.PermissionMode`.
        mode_explicit: True when the caller named the mode on the command line.
        interactive: Whether a human can be asked to confirm an action.
        max_iterations: Override the preset's iteration cap.
        temperature: Override the preset's temperature.
        max_tokens: Per-call output-token cap, when the model needs one raised.
        confirm: Confirmation callback, passed through to the gate.
        on_event: Called with each :class:`ActionRecord` as it is decided.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        workspace: Path,
        mode: PermissionMode,
        mode_explicit: bool = False,
        interactive: bool = False,
        max_iterations: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        confirm: Callable[[str], str] | None = None,
        on_event: Callable[[ActionRecord], None] | None = None,
        on_diff: Callable[[ProposedEdit], None] | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.workspace = Path(workspace)
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.gate = PermissionGate(
            mode,
            self.workspace,
            mode_explicit=mode_explicit,
            interactive=interactive,
            confirm=confirm,
            on_event=on_event,
            on_diff=on_diff,
            journal=EditJournal(self.workspace),
        )
        self._agent: Any = None

    @property
    def mode(self) -> PermissionMode:
        """The permission mode this run is using."""
        return self.gate.mode

    def system_prompt(self) -> str:
        """Return the preset prompt plus the workspace instruction."""
        from effgen.presets import get_preset

        base = get_preset(CODE_PRESET).system_prompt
        return base + "\n\n" + _WORKSPACE_PROMPT.format(workspace=self.workspace)

    def build_agent(self) -> Any:
        """Construct the agent, reusing the ``coding`` preset's configuration."""
        from effgen.core.agent import Agent, AgentConfig
        from effgen.presets import get_preset

        preset = get_preset(CODE_PRESET)
        config = AgentConfig(
            name="code-agent",
            model=self.model,
            provider=self.provider,
            tools=build_code_tools(self.gate),
            system_prompt=self.system_prompt(),
            max_iterations=(
                self.max_iterations if self.max_iterations is not None
                else preset.max_iterations
            ),
            temperature=(
                self.temperature if self.temperature is not None else preset.temperature
            ),
            max_tokens=self.max_tokens,
            enable_memory=preset.enable_memory,
            enable_sub_agents=False,
        )
        self._agent = Agent(config=config)
        return self._agent

    def run(self, task: str) -> CodeRunResult:
        """Run *task* to completion and return its :class:`CodeRunResult`.

        ``EFFGEN_WORKSPACE`` is set for the duration of the run so every tool
        resolves the same root, and restored afterwards.
        """
        agent = self._agent or self.build_agent()
        with workspace_env(self.workspace):
            response = agent.run(task)
        return self.result_from_response(task, response)

    def result_from_response(self, task: str, response: Any) -> CodeRunResult:
        """Assemble a :class:`CodeRunResult` from *response* and the gate's log.

        The agent run is separated from record assembly so the REPL, which drives
        the agent itself (under a live status render, and sometimes with the mode
        overridden for a single turn), builds the same result the single-shot
        path does.
        """
        metadata = response.metadata or {}
        return CodeRunResult(
            task=task,
            answer=response.output or "",
            success=bool(response.success),
            reason=str(metadata.get("reason", "")),
            model=self.model,
            provider=self.provider or getattr(response, "provider", None),
            workspace=str(self.workspace),
            permission_mode=self.mode.value,
            iterations=int(getattr(response, "iterations", 0) or 0),
            tool_calls=int(getattr(response, "tool_calls", 0) or 0),
            tokens=int(getattr(response, "tokens_used", 0) or 0),
            cost_usd=metadata.get("cost_usd"),
            duration_s=float(getattr(response, "execution_time", 0.0) or 0.0),
            partial=bool(metadata.get("partial")),
            actions=list(self.gate.actions),
            files_written=self.gate.files_written,
            diffs=list(self.gate.edits),
            error=metadata.get("error"),
        )
