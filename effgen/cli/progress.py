"""Live progress and status presentation for the command-line interface.

This is a *presentation layer only*: it never changes what an agent does, only
how the agent's existing execution events are surfaced to a human watching an
interactive terminal. It turns the silent "dead pause" while a model thinks into
an honest, ticking status line — ``Thinking…`` → ``Calling <model>…`` →
``Running <tool>…`` — followed by a single glanceable summary line.

Everything here is **opt-out and TTY-aware**. Animation is shown only when:

* output is an interactive terminal (``stdout.isatty()``),
* the ``rich`` library is installed,
* and none of the following ask for plain output:
  ``--quiet``, ``--no-animation``, ``EFFGEN_NO_ANIM=1``, ``NO_COLOR``, or a CI
  environment.

When animation is off the caller falls back to plain, single-line text that is
safe to pipe, redirect, or capture in logs. The live status is *transient* (it
erases itself when the work finishes) so it never corrupts the final output.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

try:  # rich is an install dependency, but degrade gracefully if it is absent.
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - rich is normally present
    _RICH_AVAILABLE = False


# Braille shimmer frames for the manual spinner (kept in sync with the icons
# used elsewhere in the CLI so the whole product feels like one tool).
_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Environment variables that indicate a non-interactive CI runner. Animation is
# pointless (and noisy) in CI logs, so we fall back to plain output there.
_CI_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
)


def _env_flag(name: str) -> bool:
    """Return True if an environment flag is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def color_enabled() -> bool:
    """Honor the ``NO_COLOR`` convention (https://no-color.org/)."""
    return not os.environ.get("NO_COLOR")


def is_ci() -> bool:
    """True when running under a known continuous-integration environment."""
    return any(os.environ.get(v) for v in _CI_ENV_VARS)


def animation_enabled(
    *,
    quiet: bool = False,
    no_animation: bool = False,
    stream: Any = None,
) -> bool:
    """Decide whether to show live animated status for the current invocation.

    Args:
        quiet: The user asked for quiet output (errors/answer only).
        no_animation: The user passed ``--no-animation``.
        stream: The output stream to test for interactivity (defaults to
            ``sys.stdout``).

    Returns:
        True only for an interactive terminal that has not opted out.
    """
    if not _RICH_AVAILABLE:
        return False
    if quiet or no_animation:
        return False
    if _env_flag("EFFGEN_NO_ANIM"):
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if is_ci():
        return False
    out = stream if stream is not None else sys.stdout
    try:
        if not out.isatty():
            return False
    except Exception:  # noqa: BLE001 - a stream without isatty() is non-interactive
        return False
    return True


def is_reasoning_agent(agent: Any) -> bool:
    """Best-effort detection of a reasoning model (pauses before any token).

    Reasoning models (OpenAI o-series / gpt-5 reasoning, thinking-budget Gemini,
    DeepSeek-R, …) can sit silent for many seconds before emitting output. We
    surface a ``🧠 reasoning…`` indicator for them so the wait feels alive rather
    than hung. Detection prefers the adapter's own capability flag and falls back
    to a conservative model-id heuristic.
    """
    model = getattr(agent, "model", None)
    if model is None:
        return False
    if getattr(model, "_is_reasoning_model", False):
        return True
    name = (getattr(model, "model_name", "") or "").lower()
    return any(
        marker in name
        for marker in ("o1-", "o3-", "o4-mini", "reasoner", "deepseek-r", "-thinking")
    )


def short_model_label(model_name: str | None, limit: int = 28) -> str:
    """A compact, human-friendly model label for status lines.

    Strips a ``provider/`` or ``org/`` prefix so ``Qwen/Qwen2.5-1.5B-Instruct``
    reads as ``Qwen2.5-1.5B-Instruct`` and ``openai:gpt-5-nano`` as ``gpt-5-nano``.
    """
    if not model_name:
        return "model"
    label = str(model_name)
    for sep in (":", "/"):
        if sep in label:
            label = label.split(sep)[-1]
    if len(label) > limit:
        label = label[: limit - 1] + "…"
    return label


# ---------------------------------------------------------------------------
# Live status line
# ---------------------------------------------------------------------------


class _StatusState:
    """Mutable status shared between the agent thread and the Live refresher.

    The agent's :class:`ExecutionTracker` invokes :meth:`LiveStatus.on_event`
    (synchronously, on the agent thread) which mutates this object; rich's
    background refresh thread reads it ~12×/s to redraw the line. Plain
    attribute reads/writes of immutable values are safe across the two threads
    for this purely cosmetic use.
    """

    # A tool/sub-step label stays visible at least this long so instant tools
    # (e.g. the calculator, which finishes in well under a frame) still flash a
    # readable "Running <tool>…" rather than being skipped between refreshes.
    _STICKY_SECONDS = 0.6

    def __init__(self, model_label: str, reasoning: bool):
        self.start = time.monotonic()
        self.model_label = model_label
        self.reasoning = reasoning
        self.base_label = self._idle_label()
        self.sticky_label: str | None = None
        self.sticky_until = 0.0

    def _idle_label(self) -> str:
        if self.reasoning:
            return f"🧠 {self.model_label} reasoning…"
        return f"Calling {self.model_label}…"

    def set_step(self, label: str) -> None:
        """Show *label* now and hold it for at least the sticky window."""
        self.sticky_label = label
        self.sticky_until = time.monotonic() + self._STICKY_SECONDS

    def set_base(self, label: str) -> None:
        """Set the steady-state label shown once any sticky step has expired."""
        self.base_label = label

    def effective_label(self) -> str:
        if self.sticky_label is not None and time.monotonic() < self.sticky_until:
            return self.sticky_label
        return self.base_label


class _StatusRenderable:
    """A rich renderable rebuilt on every refresh so time/spinner keep ticking."""

    def __init__(self, state: _StatusState):
        self.state = state

    def __rich__(self) -> Any:
        s = self.state
        frame = _BRAILLE[int(time.monotonic() * 10) % len(_BRAILLE)]
        elapsed = time.monotonic() - s.start
        text = Text()
        text.append(frame + " ", style="cyan")
        text.append(s.effective_label())
        text.append(f"   ⏱ {elapsed:4.1f}s", style="dim")
        return text


class LiveStatus:
    """Context manager that animates a single live status line during a run.

    Usage::

        with LiveStatus(console, model_label="gpt-5-nano", tracker=agent.execution_tracker):
            response = agent.run(task)

    It is transient: when the ``with`` block exits, the status line is erased so
    the caller can print the answer and a frozen summary on a clean slate. When
    ``rich`` is unavailable or ``console`` is ``None`` it becomes a no-op (the
    caller should already have decided not to animate via :func:`animation_enabled`).
    """

    def __init__(
        self,
        console: Any,
        *,
        model_label: str = "model",
        reasoning: bool = False,
        tracker: Any = None,
    ):
        self.console = console
        self.tracker = tracker
        self.state = _StatusState(model_label, reasoning)
        self._live: Any = None
        self._active = bool(_RICH_AVAILABLE and console is not None)

    def __enter__(self) -> "LiveStatus":
        if not self._active:
            return self
        if self.tracker is not None:
            try:
                self.tracker.add_listener(self.on_event)
            except Exception:  # noqa: BLE001 - never let cosmetics break a run
                pass
        self._live = Live(
            _StatusRenderable(self.state),
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self.tracker is not None:
            try:
                self.tracker.remove_listener(self.on_event)
            except Exception:  # noqa: BLE001
                pass
        if self._live is not None:
            try:
                self._live.__exit__(*exc)
            except Exception:  # noqa: BLE001
                pass
            self._live = None
        return False  # never suppress exceptions (Ctrl-C must propagate)

    def on_event(self, event: Any) -> None:
        """Update the status label from an execution event (tracker listener)."""
        try:
            etype = getattr(event.type, "value", str(event.type))
        except Exception:  # noqa: BLE001
            return
        if etype == "tool_call_start":
            name = (event.data or {}).get("tool_name", "tool")
            self.state.set_step(f"Running {name}…")
        elif etype in ("tool_call_complete", "tool_call_failed"):
            # The model is called again next; revert the steady-state label but
            # let any still-visible "Running <tool>…" finish its sticky window.
            self.state.set_base(self.state._idle_label())
        elif etype == "sub_agent_start":
            name = (event.data or {}).get("agent_name") or event.agent_id or "sub-agent"
            self.state.set_step(f"Delegating to {name}…")
        elif etype == "task_decomposition":
            self.state.set_base("Planning…")


# ---------------------------------------------------------------------------
# Final summary line
# ---------------------------------------------------------------------------


def _extract_cost(metadata: dict[str, Any] | None) -> float | None:
    if not metadata:
        return None
    for key in ("cost_usd", "cost", "total_cost"):
        val = metadata.get(key)
        if isinstance(val, int | float) and val > 0:
            return float(val)
    return None


def summary_line(response: Any) -> tuple[str, str]:
    """Build the one-glance final summary for a finished run.

    Returns ``(plain, markup)`` where *plain* is safe for non-rich output and
    *markup* carries rich color tags.

    Example (success)::

        ✓ Done in 3.2s · 2 tools · 1,204 tokens · $0.0006
    """
    secs = float(getattr(response, "execution_time", 0.0) or 0.0)
    tools = int(getattr(response, "tool_calls", 0) or 0)
    tokens = int(getattr(response, "tokens_used", 0) or 0)
    cost = _extract_cost(getattr(response, "metadata", None))
    success = bool(getattr(response, "success", False))

    parts = [f"{secs:.1f}s"]
    if tools:
        parts.append(f"{tools} tool" + ("s" if tools != 1 else ""))
    if tokens:
        parts.append(f"{tokens:,} tokens")
    if cost is not None:
        parts.append(f"${cost:.4f}")
    body = " · ".join(parts)

    if success:
        plain = f"✓ Done in {body}"
        markup = f"[green]✓[/green] Done in {body}"
    else:
        meta = getattr(response, "metadata", None) or {}
        reason = meta.get("reason", "failed")
        tail = f" · {body}" if body else ""
        plain = f"✗ Failed ({reason}){tail}"
        markup = f"[red]✗[/red] Failed ([yellow]{reason}[/yellow]){tail}"
    return plain, markup


def print_summary(cli: Any, response: Any) -> None:
    """Print the frozen summary line via the CLI's rich-or-plain print."""
    plain, markup = summary_line(response)
    console = getattr(cli, "console", None)
    if console is not None:
        console.print(markup)
    else:
        print(plain)


# ---------------------------------------------------------------------------
# Progress bars for the long-running, countable commands
# ---------------------------------------------------------------------------


class StepProgress:
    """A count/rate/ETA progress bar for batch, eval, ingest, and downloads.

    Animated rich bar on an interactive terminal; a quiet no-op elsewhere (the
    caller still prints its own plain start/finish lines). Use as a context
    manager and call :meth:`update` / :meth:`advance` as work completes::

        with StepProgress(console, total=len(items), description="Batch", animate=ok) as bar:
            for i, item in enumerate(items, 1):
                ...
                bar.update(i)
    """

    def __init__(
        self,
        console: Any,
        *,
        total: int | None,
        description: str = "Working",
        animate: bool = True,
    ):
        self.total = total
        self.description = description
        self.console = console
        self.animate = bool(animate and _RICH_AVAILABLE and console is not None)
        self._progress: Any = None
        self._task: Any = None
        self._completed = 0

    def __enter__(self) -> "StepProgress":
        if self.animate:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=True,
            )
            self._progress.__enter__()
            self._task = self._progress.add_task(self.description, total=self.total)
        return self

    def update(self, completed: int, total: int | None = None) -> None:
        """Set the absolute number of completed steps (and optionally the total)."""
        self._completed = completed
        if total is not None:
            self.total = total
        if self._progress is not None:
            self._progress.update(self._task, completed=completed, total=self.total)

    def advance(self, n: int = 1) -> None:
        """Advance the bar by *n* steps."""
        self.update(self._completed + n)

    def __exit__(self, *exc: Any) -> bool:
        if self._progress is not None:
            try:
                self._progress.__exit__(*exc)
            except Exception:  # noqa: BLE001
                pass
            self._progress = None
        return False
