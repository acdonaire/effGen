"""Shared palette and named themes for both effGen surfaces.

This module is the single source of truth for the product's colors. It defines,
in one place:

- **Semantic roles** — the meaning-carrying names (``accent``, ``success``,
  ``warning``, ``error``, ``muted``, …) that the rest of the code refers to
  instead of raw colors.
- **CLI themes** — a Rich style table per named theme (``default``,
  ``high-contrast``, ``monochrome``, ``light``). ``default`` is the unchanged
  look; the others are opt-in via ``EFFGEN_THEME`` / ``--theme``.
- **Dashboard palette** — the dark and light CSS-variable values the dashboard
  stylesheet uses. The stylesheet is a static asset, so these values are kept in
  lockstep with ``dashboard/static/style.css`` by a test rather than generated.
- **Status → role map** — the one mapping from a run/node status to a semantic
  role, so a completed step, a failure, or an in-progress step reads as the same
  color on the terminal and in the dashboard.

Nothing here imports anything heavier than the standard library.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Status → semantic role. One mapping shared by every surface.
# ---------------------------------------------------------------------------
#
# Both the terminal workflow diagram and the dashboard resolve a run/node status
# through this table, so "green" means *completed/success*, "red" means
# *failed/error*, and the brand accent means *in progress* on both surfaces.
STATUS_ROLE: dict[str, str] = {
    "completed": "success",
    "succeeded": "success",
    "success": "success",
    "running": "accent",
    "in_progress": "accent",
    "pending": "muted",
    "queued": "muted",
    "skipped": "warning",
    "failed": "error",
    "error": "error",
}

# Glyph carried alongside the status color so the state is legible without color
# (piped output, NO_COLOR, monochrome theme).
STATUS_GLYPH: dict[str, str] = {
    "completed": "●",
    "succeeded": "●",
    "success": "●",
    "running": "◐",
    "in_progress": "◐",
    "pending": "○",
    "queued": "○",
    "skipped": "⊘",
    "failed": "✗",
    "error": "✗",
}


def status_style(status: str) -> str:
    """Return the theme style name for a run/node *status* (e.g. ``effgen.success``)."""
    role = STATUS_ROLE.get(str(status), "muted")
    return f"effgen.{role}"


def status_glyph(status: str) -> str:
    """Return the status glyph for *status* (``○`` for anything unknown)."""
    return STATUS_GLYPH.get(str(status), "○")


# ---------------------------------------------------------------------------
# CLI themes — a Rich style table per named theme.
# ---------------------------------------------------------------------------
#
# Keys are the semantic style names used in CLI markup (``effgen.success`` etc.);
# the plain color names (``green``, ``cyan``, …) are also mapped in the non-default
# themes so hardcoded ``[green]``/``[cyan]`` markup follows the selected theme too.
# ``default`` is intentionally identical to the historical palette — selecting no
# theme changes nothing.
#
# Brand accent: the dashboard's accent is indigo (``#818cf8``); on the terminal the
# nearest ANSI named color is ``magenta``, so ``effgen.accent``/``effgen.title`` use
# it as the terminal stand-in for that same brand hue. ``cyan``/``accent2`` is the
# shared secondary on both surfaces.

_DEFAULT: dict[str, str] = {
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
    "table.header": "bold cyan",
    "repr.number": "cyan",
}

# High-contrast: bright, bold hues on the terminal's maximum-contrast colors, and
# no ``dim`` (low-vision users lose dimmed text). Base color names are lifted to
# their bright variants so all markup gains contrast.
_HIGH_CONTRAST: dict[str, str] = {
    "effgen.success": "bold bright_green",
    "effgen.error": "bold bright_red",
    "effgen.warning": "bold bright_yellow",
    "effgen.info": "bold bright_cyan",
    "effgen.accent": "bold bright_magenta",
    "effgen.muted": "bold white",
    "effgen.heading": "bold bright_cyan",
    "effgen.title": "bold bright_magenta",
    "effgen.label": "bold bright_white",
    "effgen.value": "bright_white",
    "effgen.cost": "bold bright_green",
    "effgen.tool": "bold bright_magenta",
    "effgen.metric": "bold bright_cyan",
    "effgen.model": "bold bright_cyan",
    "effgen.prompt": "bold bright_cyan",
    "table.header": "bold bright_white",
    "repr.number": "bold bright_cyan",
    # Lift hardcoded base colors to bright variants.
    "green": "bright_green",
    "red": "bright_red",
    "yellow": "bright_yellow",
    "cyan": "bright_cyan",
    "magenta": "bright_magenta",
    "blue": "bright_blue",
    "white": "bright_white",
    "dim": "white",
}

# Monochrome: structure only — bold/underline/italic carry meaning, hue is dropped.
# A middle option between the default palette and NO_COLOR (which drops styling too).
_MONOCHROME: dict[str, str] = {
    "effgen.success": "bold",
    "effgen.error": "bold reverse",
    "effgen.warning": "bold",
    "effgen.info": "default",
    "effgen.accent": "bold",
    "effgen.muted": "dim",
    "effgen.heading": "bold underline",
    "effgen.title": "bold",
    "effgen.label": "bold",
    "effgen.value": "default",
    "effgen.cost": "default",
    "effgen.tool": "italic",
    "effgen.metric": "default",
    "effgen.model": "italic",
    "effgen.prompt": "bold",
    "table.header": "bold",
    "repr.number": "default",
    # Neutralize hardcoded base colors so nothing shows hue.
    "green": "default",
    "red": "bold",
    "yellow": "default",
    "cyan": "default",
    "magenta": "default",
    "blue": "default",
    "bright_green": "default",
    "bright_red": "bold",
    "bright_yellow": "default",
    "bright_cyan": "default",
    "bright_magenta": "default",
    "bright_blue": "default",
}

# Light: tuned for a light terminal background — standard (darker) hues rather than
# bright ones that wash out on white, blue in place of cyan for headings, and an
# orange warning since yellow is illegible on white.
_LIGHT: dict[str, str] = {
    "effgen.success": "bold green",
    "effgen.error": "bold red",
    "effgen.warning": "dark_orange3",
    "effgen.info": "blue",
    "effgen.accent": "magenta",
    "effgen.muted": "grey42",
    "effgen.heading": "bold blue",
    "effgen.title": "bold magenta",
    "effgen.label": "bold",
    "effgen.value": "default",
    "effgen.cost": "green",
    "effgen.tool": "magenta",
    "effgen.metric": "blue",
    "effgen.model": "blue",
    "effgen.prompt": "bold blue",
    "table.header": "bold blue",
    "repr.number": "blue",
    # Steer washed-out brights and low-contrast yellow toward readable hues on white.
    "cyan": "blue",
    "yellow": "dark_orange3",
    "bright_cyan": "blue",
    "bright_yellow": "dark_orange3",
    "bright_green": "green",
    "bright_red": "red",
    "bright_magenta": "magenta",
}

CLI_THEMES: dict[str, dict[str, str]] = {
    "default": _DEFAULT,
    "high-contrast": _HIGH_CONTRAST,
    "monochrome": _MONOCHROME,
    "light": _LIGHT,
}

# Accepted aliases for theme names (dashes/underscores/spellings).
_THEME_ALIASES: dict[str, str] = {
    "hc": "high-contrast",
    "high_contrast": "high-contrast",
    "highcontrast": "high-contrast",
    "mono": "monochrome",
    "monochromatic": "monochrome",
    "dark": "default",
}

DEFAULT_THEME_NAME = "default"


def normalize_theme_name(name: str | None) -> str | None:
    """Resolve *name* to a known CLI theme, or ``None`` if it is unknown.

    Case- and separator-insensitive; accepts a small set of aliases
    (``hc``/``mono``/``dark``). Returns ``None`` for an unrecognized name so the
    caller can warn and fall back to the default.
    """
    if not name:
        return None
    key = str(name).strip().lower().replace("_", "-")
    if key in CLI_THEMES:
        return key
    alias = _THEME_ALIASES.get(str(name).strip().lower().replace("-", "_"))
    return alias


def cli_palette(name: str | None = None) -> dict[str, str]:
    """Return the Rich style table for a theme name (default when unknown/None)."""
    resolved = normalize_theme_name(name) or DEFAULT_THEME_NAME
    return dict(CLI_THEMES[resolved])


# ---------------------------------------------------------------------------
# Dashboard palette — CSS-variable values, kept in lockstep with style.css.
# ---------------------------------------------------------------------------
DASHBOARD_DARK: dict[str, str] = {
    "bg": "#0f1117",
    "bg-panel": "#1a1d27",
    "bg-card": "#21253a",
    "bg-inset": "#0a0c14",
    "border": "#2c3150",
    "text": "#e2e8f0",
    "text-muted": "#aab4c5",
    "text-faint": "#93a0b4",
    "accent": "#818cf8",
    "accent2": "#22d3ee",
    "ok": "#34d399",
    "warn": "#fbbf24",
    "err": "#f87171",
    "focus": "#22d3ee",
    # Label ink for waterfall bars: near-black reads against the light/saturated
    # dark-theme bar fills (white on them fails WCAG).
    "wf-label": "#0a0c14",
}

DASHBOARD_LIGHT: dict[str, str] = {
    "bg": "#f6f7fb",
    "bg-panel": "#ffffff",
    "bg-card": "#f0f2f8",
    "bg-inset": "#eef0f6",
    "border": "#d3d8e6",
    "text": "#1a1d27",
    "text-muted": "#4a5468",
    "text-faint": "#5a6478",
    "accent": "#4f46e5",
    "accent2": "#0e7490",
    "ok": "#047857",
    "warn": "#b45309",
    "err": "#b91c1c",
    "focus": "#4f46e5",
    # Light-theme bar fills are dark, so white label ink reads on them.
    "wf-label": "#ffffff",
}
