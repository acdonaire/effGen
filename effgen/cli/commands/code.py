"""The ``effgen code`` command — a coding agent that writes, runs and fixes code.

Resolves the workspace, the permission mode and the task (from the argument, from
``-p``, or from piped stdin), runs the loop in
:class:`~effgen.cli.code.engine.CodeEngine`, and prints either a human-readable
report or a single JSON document.

Output contract:

- With a terminal, the answer is framed and followed by a one-line summary and,
  when anything was written or withheld, a short action list.
- Piped or with ``--json``, every human-facing line goes to stderr and stdout
  carries only the answer text (or the JSON document), so the command composes
  with ``jq`` and with shell pipelines.
- The exit code is ``0`` for a completed run, ``1`` for a failed one, and ``2``
  when the run completed but changes were withheld because there was no terminal
  to confirm on and no ``--auto-edit``/``--yes`` was given.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import TYPE_CHECKING, Any

from effgen.cli.code.edits import ProposedEdit
from effgen.cli.code.engine import (
    CodeEngine,
    CodeRunResult,
    resolve_workspace,
    undo_workspace,
    workspace_execution_note,
)
from effgen.cli.code.permissions import (
    MODE_DESCRIPTIONS,
    ActionRecord,
    PermissionMode,
    default_mode,
)
from effgen.cli.commands._shared import (
    _preflight_model_hint,
    _quickstart_suggest_model,
    resolve_provider_name,
)
from effgen.ui.render import ascii_fold as _ascii_fold

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface

logger = logging.getLogger(__name__)

#: Exit code for a run that completed but withheld every change.
EXIT_WITHHELD = 2

# Glyph per decision, so an action list reads without color.
_DECISION_GLYPH: dict[str, str] = {
    "allowed": "success",
    "withheld": "warning",
    "declined": "warning",
    "refused": "error",
}


def _resolve_mode(args: argparse.Namespace, interactive: bool) -> tuple[PermissionMode, bool, str | None]:
    """Return ``(mode, explicit, error)`` from the permission flags.

    At most one of ``--plan``/``--auto-edit``/``--yes`` may be given; naming two
    is an error rather than a silent precedence rule.
    """
    chosen = [
        (PermissionMode.PLAN, "--plan", bool(getattr(args, "plan_only", False))),
        (PermissionMode.AUTO_EDIT, "--auto-edit", bool(getattr(args, "auto_edit", False))),
        (PermissionMode.YES, "--yes", bool(getattr(args, "assume_yes", False))),
    ]
    named = [(mode, flag) for mode, flag, given in chosen if given]
    if len(named) > 1:
        flags = ", ".join(flag for _, flag in named)
        return default_mode(interactive), False, (
            f"{flags} cannot be combined — they select different permission "
            "modes. Pass one."
        )
    if named:
        return named[0][0], True, None
    return default_mode(interactive), False, None


def _read_piped_stdin() -> str:
    """Return piped stdin, or an empty string when stdin is a terminal."""
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return ""
        return sys.stdin.read()
    except (ValueError, OSError):  # pragma: no cover - closed stdin
        return ""


def _resolve_task(args: argparse.Namespace, interactive: bool) -> tuple[str | None, str | None]:
    """Return ``(task, error)`` from the positional argument, ``-p`` and stdin.

    Piped stdin with a task becomes context in front of it (``cat err.log |
    effgen code -p "explain this"``); piped stdin with no task *is* the task.
    """
    explicit = getattr(args, "print_task", None)
    task = explicit if explicit else getattr(args, "task", None)
    piped = _read_piped_stdin()

    if piped.strip():
        if task and task.strip():
            task = f"--- stdin ---\n{piped.strip()}\n\n---\n\n{task}"
        else:
            task = piped.strip()

    if task and task.strip():
        return task, None

    if interactive:
        try:
            typed = input("Task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return None, None
        if typed:
            return typed, None
    return None, (
        "effgen code needs a task. Pass one as an argument "
        '(effgen code "add a retry to fetch.py"), with -p, or on stdin '
        '(echo "..." | effgen code).'
    )


def _print_action(cli: "CLIInterface", record: ActionRecord) -> None:
    """Print one decided action as a progress tick."""
    from effgen.ui.palette import glyph

    stream = cli._human_stream()
    mark = glyph(_DECISION_GLYPH.get(record.decision, "muted"), stream)
    if record.decision == "allowed" and record.outcome is None:
        # The tick is printed once the outcome is known, not twice.
        return
    text = record.summary
    if record.decision == "allowed":
        if record.outcome == "error":
            text = f"{text} — failed: {record.detail}"
    else:
        text = f"{text} — {record.decision}: {record.reason}"
    line = _ascii_fold(f"  {mark} {text}", stream)
    console = cli._human()
    if console:
        style = "effgen.success" if record.decision == "allowed" else "effgen.warning"
        if record.decision == "refused" or record.outcome == "error":
            style = "effgen.error"
        console.print(f"[{style}]{line}[/{style}]", highlight=False)
    else:
        print(line, file=stream)


def _print_diff(cli: "CLIInterface", edit: ProposedEdit) -> None:
    """Show a pending edit as a colorized unified diff before it is decided."""
    from effgen.cli.code.diffs import render_diff
    from effgen.ui.render import ascii_fold

    stream = cli._human_stream()
    kind = "new file" if edit.is_new else "edit"
    header = f"{kind} {edit.rel_path} ({edit.stat()})"
    plain, markup = render_diff(edit.diff_text, stream)
    console = cli._human()
    if console is not None:
        console.print(f"[effgen.heading]{header}[/effgen.heading]", highlight=False)
        console.print(markup, highlight=False)
    else:
        print(ascii_fold(header, stream), file=stream)
        print(ascii_fold(plain, stream), file=stream)


def _print_summary(cli: "CLIInterface", result: CodeRunResult) -> None:
    """Print the one-glance run summary via the CLI's human stream.

    Uses :func:`effgen.ui.render.summary_line` (the same footer ``effgen run``
    prints) but routes it through ``cli._human()`` so a piped run keeps it off
    stdout.
    """
    from effgen.ui.render import ascii_fold, summary_line

    stream = cli._human_stream()
    plain, markup = summary_line(result, stream=stream)
    console = cli._human()
    if console is not None:
        console.print(ascii_fold(markup, stream))
    else:
        print(ascii_fold(plain, stream), file=stream)


def _report(cli: "CLIInterface", result: CodeRunResult, *, quiet: bool) -> None:
    """Print the framed answer, the summary line and the file/refusal list.

    Used for an interactive run; the answer is rendered to the human console.
    """
    from effgen.ui.render import answer_surface

    console = cli._human()
    if not result.success and not result.partial:
        cli.print_error_panel(result.answer or "The run produced no answer.", title="Error")
    elif console:
        answer_surface(
            result.answer,
            success=result.success,
            partial=result.partial,
            framed=True,
            title="Coding Agent",
            console=console,
        )
    else:
        print(result.answer, file=cli._human_stream())

    if quiet:
        return

    _print_summary(cli, result)

    if result.files_written:
        names = ", ".join(result.files_written)
        cli.print(f"Files written in {result.workspace}: {names}")
    _report_partial_hunks(cli, result)
    for record in result.refused:
        cli.print_warning(record.reason)


def _withheld_note(result: CodeRunResult, explicit_mode: bool) -> str:
    """Return the one-line explanation for a run that changed nothing."""
    kinds = sorted({a.kind for a in result.withheld})
    what = ", ".join(kinds)
    if explicit_mode and result.permission_mode == PermissionMode.PLAN.value:
        return (
            f"Plan mode: {len(result.withheld)} action(s) ({what}) were proposed "
            "and not carried out. Re-run with --auto-edit to apply them."
        )
    return (
        f"No changes were made: {len(result.withheld)} action(s) ({what}) needed "
        "confirmation and this session has no terminal to confirm on. Re-run with "
        "--auto-edit to apply file writes and sandboxed runs, or --yes to also "
        "allow shell commands."
    )


def _run_undo(cli: "CLIInterface", args: argparse.Namespace) -> int:
    """Reverse the last applied edit(s) in the workspace and report what changed."""
    json_mode = bool(getattr(args, "output_json", False))
    try:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    except (ValueError, OSError):  # pragma: no cover - closed std streams
        interactive = False
    if json_mode or not interactive:
        cli._human_to_stderr = True

    try:
        workspace = resolve_workspace(getattr(args, "workspace", None))
    except OSError as exc:
        cli.print_error(f"--workspace: {exc}")
        return 1

    count = int(getattr(args, "undo_count", None) or 1)
    outcomes, remaining = undo_workspace(workspace, count)

    if json_mode:
        print(json.dumps(
            {
                "undone": [
                    {"path": o.rel_path, "action": o.action, "detail": o.detail}
                    for o in outcomes
                ],
                "remaining": remaining,
            },
            indent=2, ensure_ascii=False,
        ))
        return 0

    if not outcomes:
        cli.print(f"Nothing to undo in {workspace}.")
        return 0
    for outcome in outcomes:
        if outcome.action == "restored":
            cli.print(f"Restored {outcome.rel_path} to its previous content.")
        elif outcome.action == "removed":
            cli.print(f"Removed {outcome.rel_path} (created by a coding run).")
        elif outcome.action == "skipped":
            cli.print_warning(f"Skipped {outcome.rel_path}: {outcome.detail}")
        else:
            cli.print_error(f"Could not undo {outcome.rel_path}: {outcome.detail}")
    cli.print(f"{remaining} earlier change(s) can still be undone.")
    return 0


def _report_partial_hunks(cli: "CLIInterface", result: CodeRunResult) -> None:
    """Warn about any edit whose hunks did not all apply."""
    for diff in result.diffs:
        failed = int(diff.get("hunks_failed", 0) or 0)
        if failed:
            cli.print_warning(
                f"{diff.get('path')}: {failed} hunk(s) did not apply because the "
                "file changed since it was read; the rest were applied."
            )


def run_code_command(cli: "CLIInterface", args: argparse.Namespace) -> int:
    """Run one ``effgen code`` task and return the process exit code."""
    if bool(getattr(args, "undo", False)):
        return _run_undo(cli, args)

    json_mode = bool(getattr(args, "output_json", False))
    quiet = bool(getattr(args, "quiet", False))

    try:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    except (ValueError, OSError):  # pragma: no cover - closed std streams
        interactive = False

    # Piped or JSON output keeps stdout for the result alone; everything a
    # human reads goes to stderr.
    if json_mode or not interactive:
        cli._human_to_stderr = True

    mode, mode_explicit, mode_error = _resolve_mode(args, interactive)
    if mode_error:
        cli.print_error(mode_error)
        return 1

    task, task_error = _resolve_task(args, interactive)
    if task is None:
        if task_error:
            cli.print_error(task_error)
        return 1

    provider, provider_error = resolve_provider_name(getattr(args, "provider", None))
    if provider_error:
        cli.print_error(provider_error)
        return 1

    try:
        workspace = resolve_workspace(getattr(args, "workspace", None))
    except OSError as exc:
        cli.print_error(f"--workspace: {exc}")
        return 1

    model = getattr(args, "model", None)
    if model:
        _preflight_model_hint(cli, model, provider)
    else:
        model, suggested_provider, reason = _quickstart_suggest_model()
        if provider is None and suggested_provider:
            provider = suggested_provider
        if not quiet:
            cli.print(f"Using model {model} ({reason}); override with -m/--model.")

    if not quiet:
        cli.print(f"Workspace: {workspace}")
        cli.print(f"Permissions: {mode.value} — {MODE_DESCRIPTIONS[mode]}")
        if mode is not PermissionMode.PLAN:
            note = workspace_execution_note(workspace)
            if note:
                cli.print_warning(note)

    # Live per-action ticks are only shown on an interactive run; a piped or
    # JSON run reports the actions once, at the end, so stdout stays clean.
    show_ticks = interactive and not quiet and not json_mode
    on_event = (lambda record: _print_action(cli, record)) if show_ticks else None

    # Each pending edit's diff is shown (on the human stream) before the write is
    # decided, so a proposal is visible even in plan mode and even when piped.
    # Under --json the diffs are in the document, so the live render is skipped.
    show_diffs = not quiet and not json_mode
    on_diff = (lambda edit: _print_diff(cli, edit)) if show_diffs else None

    engine = CodeEngine(
        model=model,
        provider=provider,
        workspace=workspace,
        mode=mode,
        mode_explicit=mode_explicit,
        interactive=interactive,
        max_iterations=getattr(args, "max_iterations", None),
        temperature=getattr(args, "temperature", None),
        max_tokens=getattr(args, "max_tokens", None),
        on_event=on_event,
        on_diff=on_diff,
    )

    agent: Any = None
    try:
        agent = engine.build_agent()
        result = engine.run(task)
    except KeyboardInterrupt:
        cli.print_warning("Interrupted; no further actions were taken.")
        return 130
    except Exception as exc:  # noqa: BLE001 - reported to the user, not swallowed
        cli.print_error_panel(str(exc), title="Error")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        if json_mode:
            print(json.dumps(
                {"success": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                indent=2, ensure_ascii=False,
            ))
        return 1
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception as exc:  # noqa: BLE001 - close is best effort
                logger.debug("Agent close failed: %s", exc)

    if json_mode:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    elif interactive:
        _report(cli, result, quiet=quiet)
    else:
        # Piped: stdout is the answer text and nothing else.
        print(result.answer)
        if not quiet:
            _report_to_stderr(cli, result)

    if result.hit_iteration_cap and not quiet:
        cli.print_warning(
            f"Stopped at the iteration cap ({result.iterations} iterations) "
            "without a final answer. Raise it with --max-iterations or narrow "
            "the task."
        )

    if result.withheld and not quiet:
        cli.print(_withheld_note(result, mode_explicit))

    if not result.success:
        return 1
    if result.withheld and not mode_explicit:
        return EXIT_WITHHELD
    return 0


def _report_to_stderr(cli: "CLIInterface", result: CodeRunResult) -> None:
    """Print the summary and file list for a piped run (stdout stays the answer)."""
    _print_summary(cli, result)
    if result.files_written:
        cli.print(f"Files written in {result.workspace}: {', '.join(result.files_written)}")
    _report_partial_hunks(cli, result)
    for record in result.refused:
        cli.print_warning(record.reason)
