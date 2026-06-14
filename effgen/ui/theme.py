"""The single Rich theme and console factory for effGen.

One palette drives every table, spinner, panel, and status line so the whole
product reads as one polished tool. ``NO_COLOR`` (https://no-color.org/) is
honored: when it is set the console renders structure (tables, panels, layout)
with no ANSI color.
"""

from __future__ import annotations

import os
from typing import Any

try:  # rich is a base dependency, but degrade gracefully if it is absent.
    from rich.console import Console
    from rich.theme import Theme

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - rich is normally present
    Console = None  # type: ignore[assignment,misc]
    Theme = None  # type: ignore[assignment,misc]
    _RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Palette — semantic names, not raw colors, so the whole CLI stays consistent.
# ---------------------------------------------------------------------------
#
# Use the semantic keys (``effgen.success`` etc.) in markup throughout the CLI
# rather than hardcoding ``[green]``/``[red]`` so a future palette change is one
# edit here. The plain ``green``/``red``/… names are also aliased so existing
# markup keeps working unchanged.
_PALETTE: dict[str, str] = {
    "effgen.success": "bold green",
    "effgen.error": "bold red",
    "effgen.warning": "yellow",
    "effgen.info": "cyan",
    "effgen.accent": "magenta",
    "effgen.muted": "dim",
    "effgen.heading": "bold cyan",
    "effgen.title": "bold magenta",
    "effgen.label": "bold",
    "effgen.value": "default",
    "effgen.cost": "green",
    "effgen.tool": "magenta",
    "effgen.metric": "cyan",
    "effgen.model": "cyan",
    "effgen.prompt": "bold cyan",
    # Tables/panels pick these up automatically.
    "table.header": "bold cyan",
    "repr.number": "cyan",
}

# Syntax-highlighting theme used for code blocks everywhere (CLI + notebook).
# Kept in one place so code rendering is uniform.
CODE_THEME = "monokai"


def _build_theme() -> Any:
    if not _RICH_AVAILABLE:
        return None
    return Theme(_PALETTE)


EFFGEN_THEME = _build_theme()


def rich_available() -> bool:
    """True when the optional ``rich`` library is importable."""
    return _RICH_AVAILABLE


def color_enabled() -> bool:
    """Honor the ``NO_COLOR`` convention (https://no-color.org/).

    Returns ``False`` when ``NO_COLOR`` is set to any value, so callers can both
    build a colorless console and skip ANSI markup entirely.
    """
    return not os.environ.get("NO_COLOR")


def get_console(**kwargs: Any) -> Any:
    """Build a :class:`~rich.console.Console` carrying the effGen theme.

    Honors ``NO_COLOR`` (renders without color) and accepts any extra
    ``Console`` keyword arguments. Returns ``None`` when ``rich`` is unavailable
    so callers can fall back to plain ``print``.
    """
    if not _RICH_AVAILABLE:
        return None
    kwargs.setdefault("theme", EFFGEN_THEME)
    if not color_enabled():
        kwargs["no_color"] = True
    return Console(**kwargs)
