"""Shared terminal rendering for the coding agent's diffs, ticks and summary.

Both the single-shot command (:mod:`effgen.cli.commands.code`) and the
interactive REPL (:mod:`effgen.cli.code.repl`) show the same things — a pending
edit as a colorized unified diff, a decided action as a one-line tick, and the
one-glance run footer — so those renderers live here and are called from both.

Every function routes through the CLI's human stream (``cli._human()`` /
``cli._human_stream()``), so a piped or ``--json`` invocation keeps this chatter
on stderr and leaves stdout for the answer or the JSON document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from effgen.ui.render import ascii_fold
from effgen.ui.theme import color_enabled

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface

    from .edits import ProposedEdit
    from .engine import CodeRunResult
    from .permissions import ActionRecord

# Glyph per decision, so an action list reads without color.
_DECISION_GLYPH: dict[str, str] = {
    "allowed": "success",
    "withheld": "warning",
    "declined": "warning",
    "refused": "error",
}

#: Theme role per status line kind.
_STATUS_ROLES: dict[str, str] = {
    "success": "effgen.success",
    "warning": "effgen.warning",
    "error": "effgen.error",
}


def print_plain(cli: "CLIInterface", text: str = "", *, style: str | None = None) -> None:
    """Print *text* exactly as given, with no console markup parsing.

    Most of what the coding agent prints is content it did not author — a
    command's output, a file path, a git line, model text, a usage example such
    as ``/apply [n]``. A console reads square brackets as markup, which drops
    ``[n]`` from a help line and raises on an unbalanced tag such as
    ``[/usr/bin]`` in a command's output, so these lines go out verbatim.
    """
    cli.print_line(text, style=style if color_enabled() else None)


def print_status(cli: "CLIInterface", kind: str, text: str) -> None:
    """Print a glyph-prefixed success/warning/error line, without markup."""
    from effgen.ui.palette import glyph

    print_plain(cli, f"{glyph(kind, cli._human_stream())} {text}", style=_STATUS_ROLES.get(kind))


def print_action(cli: "CLIInterface", record: "ActionRecord") -> None:
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
    line = ascii_fold(f"  {mark} {text}", stream)
    console = cli._human()
    if console and color_enabled():
        style = "effgen.success" if record.decision == "allowed" else "effgen.warning"
        if record.decision == "refused" or record.outcome == "error":
            style = "effgen.error"
        console.print(f"[{style}]{line}[/{style}]", highlight=False)
    else:
        # NO_COLOR (or no rich console): plain text, no escape codes.
        print(line, file=stream)


def print_diff(cli: "CLIInterface", edit: "ProposedEdit") -> None:
    """Show a pending edit as a colorized unified diff before it is decided."""
    from .diffs import render_diff

    stream = cli._human_stream()
    kind = "new file" if edit.is_new else "edit"
    header = f"{kind} {edit.rel_path} ({edit.stat()})"
    plain, markup = render_diff(edit.diff_text, stream)
    console = cli._human()
    if console is not None and color_enabled():
        console.print(f"[effgen.heading]{header}[/effgen.heading]", highlight=False)
        console.print(markup, highlight=False)
    else:
        # NO_COLOR (or no rich console): plain diff text, no escape codes.
        print(ascii_fold(header, stream), file=stream)
        print(ascii_fold(plain, stream), file=stream)


def print_summary(cli: "CLIInterface", result: "CodeRunResult") -> None:
    """Print the one-glance run summary via the CLI's human stream.

    Uses :func:`effgen.ui.render.summary_line` (the same footer ``effgen run``
    prints) but routes it through ``cli._human()`` so a piped run keeps it off
    stdout.
    """
    from effgen.ui.render import summary_line

    stream = cli._human_stream()
    plain, markup = summary_line(result, stream=stream)
    console = cli._human()
    if console is not None and color_enabled():
        console.print(ascii_fold(markup, stream))
    else:
        print(ascii_fold(plain, stream), file=stream)
