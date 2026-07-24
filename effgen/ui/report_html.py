"""Self-contained HTML reports for ``run`` / ``compare`` / ``eval`` / ``cost`` / ``loadtest``.

Renders a result object that a command already produced into a single ``.html``
file that can be emailed, attached, or opened from disk. The output has no
external references of any kind: every style, script, and chart is inline, so
the file renders identically with the network disabled.

Colors come from :mod:`effgen.ui.palette`, the same source the dashboard uses,
and are emitted as CSS custom properties for a light and a dark palette. The
page follows the reader's ``prefers-color-scheme`` and carries a toggle that
overrides it per view.

Five report shapes share one visual system:

``run``
    One agent run as a shareable card: the task, the model it ran on, the
    answer (or the typed error), the step-by-step tool trace with per-step
    durations, sources and citations, and the run's tokens, cost and latency.
``comparison``
    Per-model accuracy / latency / cost for a bake-off, with the recommended
    model and the reason it was picked.
``eval``
    Suite pass rate, the exit gate, a by-difficulty breakdown, and every case.
``cost``
    Spend for a period against the configured daily budget, by provider/model.
``loadtest``
    Latency percentiles, throughput, error rate, and the error breakdown.

Typical use::

    from effgen.ui.report_html import write_html_report

    write_html_report("bakeoff.html", matrix.to_dict(), kind="comparison")

:func:`detect_report_kind` identifies a saved ``--json`` document, so a result
captured earlier can be rendered later without re-running any model.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .palette import DASHBOARD_DARK, DASHBOARD_LIGHT

__all__ = [
    "REPORT_KINDS",
    "ReportError",
    "build_html_report",
    "detect_report_kind",
    "load_result_document",
    "write_html_report",
]

REPORT_KINDS: tuple[str, ...] = ("run", "comparison", "eval", "cost", "loadtest", "battle")

#: The keys each kind renders from. A document carrying none of a kind's keys
#: cannot produce a report with data in it, so rendering is refused rather than
#: writing a file of em dashes.
_KIND_KEYS: dict[str, tuple[str, ...]] = {
    "run": ("output", "task", "execution_trace", "execution_tree"),
    "comparison": ("scores",),
    "eval": ("results", "suite", "accuracy"),
    "cost": ("rows", "total_cost_usd", "period"),
    "loadtest": ("latency", "throughput_rps", "total_requests"),
    "battle": ("contenders", "prompt", "verdict"),
}

# Series colors for multi-category charts, drawn from the shared palette roles
# and ordered so adjacent series stay distinguishable.
_SERIES_ROLES: tuple[str, ...] = ("accent", "accent2", "ok", "warn", "err")

_DASH = "—"


class ReportError(ValueError):
    """Raised when a result document cannot be rendered as a report."""


# ---------------------------------------------------------------------------
# Formatting helpers. A metric that is absent renders as an em dash; it is
# never substituted with a zero.
# ---------------------------------------------------------------------------

def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pct(value: Any, digits: int = 1) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"{value * 100:.{digits}f}%"


def _secs(value: Any, digits: int = 3) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"{value:.{digits}f}s"


def _ms(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"{value * 1000:.1f} ms"


def _usd(value: Any, digits: int = 6) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"${value:.{digits}f}"


def _rps(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"{value:,.2f} req/s"


def _int(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"{int(value):,}"


def _truncate(text: str, limit: int = 240) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fraction(value: Any, maximum: float) -> float:
    """Return *value* as a 0–1 fraction of *maximum* for bar widths."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return 0.0
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / maximum))


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _card(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="card-note">{_esc(note)}</div>' if note else ""
    return (
        '<div class="card">'
        f'<div class="card-label">{_esc(label)}</div>'
        f'<div class="card-value">{_esc(value)}</div>'
        f"{note_html}</div>"
    )


def _cards(items: list[tuple[str, str, str]]) -> str:
    return '<div class="cards">' + "".join(_card(*i) for i in items) + "</div>"


def _table(columns: list[str], rows: list[list[str]], *, caption: str = "") -> str:
    """Render a table. Cells are pre-escaped markup so a cell may carry a badge."""
    if not rows:
        return '<p class="empty">No rows to show.</p>'
    head = "".join(f"<th scope=\"col\">{_esc(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return (
        '<div class="table-wrap"><table>'
        f"{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody>"
        "</table></div>"
    )


def _badge(text: str, role: str) -> str:
    """A status pill. The text carries the state, so color is never the only cue."""
    return f'<span class="badge badge-{_esc(role)}">{_esc(text)}</span>'


def _link(url: Any, text: Any = None) -> str:
    """Render *url* as a link when its scheme is ``http`` or ``https``.

    Sources and citations come from model and tool output, so any other scheme
    (``javascript:``, ``data:``, ``file:``, a bare fragment) is rendered as
    inert escaped text instead of a clickable target.
    """
    raw = "" if url is None else str(url)
    label = _esc(raw if text is None else text)
    try:
        scheme = urlparse(raw.strip()).scheme.lower()
    except ValueError:
        scheme = ""
    if scheme not in ("http", "https"):
        return f'<span class="wrap">{label}</span>'
    return (
        f'<a class="wrap" href="{_esc(raw.strip())}" rel="noreferrer noopener" '
        f'target="_blank">{label}</a>'
    )


def _bar_chart(
    title: str,
    rows: list[tuple[str, float | None, str]],
    *,
    role: str = "accent",
    maximum: float | None = None,
) -> str:
    """Horizontal bars: ``(label, value, formatted_value)`` per row.

    A row whose value is ``None`` renders as a labeled gap rather than a
    zero-length bar, so an unpriced or unmeasured entry is visibly distinct
    from a genuine zero.
    """
    numeric = [v for _, v, _ in rows if isinstance(v, int | float) and not isinstance(v, bool)]
    top = maximum if maximum is not None else (max(numeric) if numeric else 0.0)
    items = []
    for label, value, formatted in rows:
        if value is None:
            fill = '<div class="bar-track"><div class="bar-none"></div></div>'
        else:
            width = _fraction(value, top) * 100
            fill = (
                '<div class="bar-track">'
                f'<div class="bar-fill bar-{_esc(role)}" style="width:{width:.2f}%"></div>'
                "</div>"
            )
        items.append(
            '<div class="bar-row">'
            f'<div class="bar-label" title="{_esc(label)}">{_esc(label)}</div>'
            f"{fill}"
            f'<div class="bar-value">{_esc(formatted)}</div>'
            "</div>"
        )
    return (
        f'<section class="chart"><h3>{_esc(title)}</h3>'
        + "".join(items)
        + "</section>"
    )


def _donut(title: str, slices: list[tuple[str, float]]) -> str:
    """Inline-SVG donut of cost share. Each slice is also listed in the legend."""
    total = sum(v for _, v in slices if v > 0)
    if total <= 0:
        return ""
    radius, stroke = 60.0, 26.0
    circumference = 2 * math.pi * radius
    offset = 0.0
    arcs, legend = [], []
    for idx, (label, value) in enumerate(slices):
        if value <= 0:
            continue
        role = _SERIES_ROLES[idx % len(_SERIES_ROLES)]
        share = value / total
        length = share * circumference
        arcs.append(
            f'<circle class="slice slice-{role}" cx="80" cy="80" r="{radius}" '
            f'fill="none" stroke-width="{stroke}" '
            f'stroke-dasharray="{length:.3f} {circumference - length:.3f}" '
            f'stroke-dashoffset="{-offset:.3f}" transform="rotate(-90 80 80)">'
            f"<title>{_esc(label)}: {share * 100:.1f}%</title></circle>"
        )
        legend.append(
            '<li><span class="swatch swatch-' + role + '" aria-hidden="true"></span>'
            f"{_esc(label)} <span class=\"legend-value\">{share * 100:.1f}%</span></li>"
        )
        offset += length
    return (
        f'<section class="chart"><h3>{_esc(title)}</h3>'
        '<div class="donut-wrap">'
        '<svg viewBox="0 0 160 160" role="img" width="160" height="160" '
        f'aria-label="{_esc(title)}">' + "".join(arcs) + "</svg>"
        '<ul class="legend">' + "".join(legend) + "</ul>"
        "</div></section>"
    )


def _meter(label: str, used: float, limit: float | None, *, note: str = "") -> str:
    """Budget meter: spend against a configured limit, with the state spelled out."""
    if not isinstance(limit, int | float) or isinstance(limit, bool) or limit <= 0:
        return (
            '<section class="chart"><h3>' + _esc(label) + "</h3>"
            '<p class="empty">No daily budget configured. '
            "Set one with <code>effgen cost set-budget 1.00</code>.</p></section>"
        )
    ratio = used / limit
    role = "err" if ratio >= 1.0 else "warn" if ratio >= 0.8 else "ok"
    state = "over budget" if ratio >= 1.0 else "near limit" if ratio >= 0.8 else "within budget"
    return (
        f'<section class="chart"><h3>{_esc(label)}</h3>'
        '<div class="bar-row">'
        '<div class="bar-track" role="img" '
        f'aria-label="{ratio * 100:.0f} percent of the daily budget used">'
        f'<div class="bar-fill bar-{role}" style="width:{min(ratio, 1.0) * 100:.2f}%"></div>'
        "</div>"
        f'<div class="bar-value">{_usd(used, 4)} / {_usd(limit, 2)}</div>'
        "</div>"
        f'<p class="meter-note">{ratio * 100:.0f}% used {_badge(state, role)}'
        + (f" {_esc(note)}" if note else "")
        + "</p></section>"
    )


# ---------------------------------------------------------------------------
# Page shell
# ---------------------------------------------------------------------------

def _css() -> str:
    def block(selector: str, palette: dict[str, str]) -> str:
        body = "".join(f"--{k}:{v};" for k, v in palette.items())
        return f"{selector}{{{body}}}"

    return (
        block(":root", DASHBOARD_LIGHT)
        + "@media (prefers-color-scheme: dark){" + block(":root", DASHBOARD_DARK) + "}"
        + block(':root[data-theme="light"]', DASHBOARD_LIGHT)
        + block(':root[data-theme="dark"]', DASHBOARD_DARK)
        + """
*{box-sizing:border-box}
body{margin:0;padding:0;background:var(--bg);color:var(--text);
 font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 font-size:15px;line-height:1.5}
main{max-width:1040px;margin:0 auto;padding:2rem 1.25rem 4rem}
header.report-head{border-bottom:1px solid var(--border);padding-bottom:1rem;margin-bottom:1.75rem;
 display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-start;justify-content:space-between}
h1{font-size:1.6rem;margin:0 0 .35rem}
h2{font-size:1.15rem;margin:2rem 0 .75rem;padding-bottom:.35rem;border-bottom:1px solid var(--border)}
h3{font-size:.95rem;margin:0 0 .6rem;color:var(--text-muted);font-weight:600;
 letter-spacing:.02em;text-transform:uppercase}
.subtitle{color:var(--text-muted);margin:0}
.provenance{margin:.6rem 0 0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:.4rem}
.provenance li{background:var(--bg-card);border:1px solid var(--border);border-radius:999px;
 padding:.15rem .6rem;font-size:.78rem;color:var(--text-muted)}
.provenance code{font-size:.78rem;color:var(--text-muted);word-break:break-all}
button.theme-toggle{background:var(--bg-card);color:var(--text);border:1px solid var(--border);
 border-radius:8px;padding:.4rem .75rem;font:inherit;font-size:.82rem;cursor:pointer}
button.theme-toggle:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.verdict{background:var(--bg-panel);border:1px solid var(--border);border-left:4px solid var(--accent);
 border-radius:10px;padding:1rem 1.15rem;margin:0 0 1.5rem}
.verdict-label{font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--text-muted)}
.verdict-value{font-size:1.3rem;font-weight:650;margin:.15rem 0 .3rem;word-break:break-word}
.verdict-note{color:var(--text-muted);margin:0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.75rem;margin-bottom:1.5rem}
.card{background:var(--bg-panel);border:1px solid var(--border);border-radius:10px;padding:.85rem 1rem}
.card-label{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;color:var(--text-muted)}
.card-value{font-size:1.35rem;font-weight:650;margin-top:.2rem;word-break:break-word}
.card-note{font-size:.78rem;color:var(--text-faint);margin-top:.15rem}
.chart{background:var(--bg-panel);border:1px solid var(--border);border-radius:10px;
 padding:1rem 1.15rem;margin-bottom:1rem}
.bar-row{display:grid;grid-template-columns:minmax(90px,1.3fr) 3fr minmax(84px,auto);
 gap:.6rem;align-items:center;margin-bottom:.45rem}
.bar-label{font-size:.85rem;color:var(--text-muted);overflow-wrap:anywhere}
.bar-track{background:var(--bg-inset);border-radius:5px;height:14px;overflow:hidden}
.bar-fill{height:100%;border-radius:5px}
.bar-none{height:100%;background:repeating-linear-gradient(45deg,var(--border),var(--border) 4px,
 transparent 4px,transparent 8px)}
.bar-accent{background:var(--accent)}.bar-accent2{background:var(--accent2)}
.bar-ok{background:var(--ok)}.bar-warn{background:var(--warn)}.bar-err{background:var(--err)}
.bar-value{font-variant-numeric:tabular-nums;font-size:.85rem;text-align:right}
.meter-note{margin:.5rem 0 0;color:var(--text-muted);font-size:.85rem}
.donut-wrap{display:flex;flex-wrap:wrap;gap:1.25rem;align-items:center}
.slice-accent{stroke:var(--accent)}.slice-accent2{stroke:var(--accent2)}
.slice-ok{stroke:var(--ok)}.slice-warn{stroke:var(--warn)}.slice-err{stroke:var(--err)}
.legend{list-style:none;margin:0;padding:0;font-size:.85rem}
.legend li{display:flex;align-items:center;gap:.45rem;margin-bottom:.3rem}
.legend-value{color:var(--text-muted);font-variant-numeric:tabular-nums}
.swatch{width:11px;height:11px;border-radius:3px;display:inline-block;flex:none}
.swatch-accent{background:var(--accent)}.swatch-accent2{background:var(--accent2)}
.swatch-ok{background:var(--ok)}.swatch-warn{background:var(--warn)}.swatch-err{background:var(--err)}
.table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px;background:var(--bg-panel)}
table{border-collapse:collapse;width:100%;font-size:.88rem}
caption{caption-side:top;text-align:left;padding:.7rem 1rem 0;color:var(--text-muted);font-size:.82rem}
th,td{text-align:left;padding:.55rem .8rem;border-bottom:1px solid var(--border);vertical-align:top}
th{background:var(--bg-card);font-weight:600;font-size:.78rem;text-transform:uppercase;
 letter-spacing:.03em;color:var(--text-muted);position:sticky;top:0}
tbody tr:last-child td{border-bottom:none}
.num{display:block;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.wrap{display:block;max-width:340px;overflow-wrap:anywhere}
.badge{display:inline-block;border-radius:999px;padding:.1rem .55rem;font-size:.76rem;
 font-weight:600;border:1px solid}
.badge-ok{color:var(--ok);border-color:var(--ok)}
.badge-err{color:var(--err);border-color:var(--err)}
.badge-warn{color:var(--warn);border-color:var(--warn)}
.badge-muted{color:var(--text-muted);border-color:var(--border)}
.empty{color:var(--text-muted);margin:.25rem 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
a{color:var(--accent)}
.prose{background:var(--bg-panel);border:1px solid var(--border);border-radius:10px;
 padding:1rem 1.15rem;margin:0 0 1rem;white-space:pre-wrap;overflow-wrap:anywhere;
 max-width:72ch;font-size:.95rem}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.82rem;
 white-space:pre-wrap;overflow-wrap:anywhere;display:block;max-width:420px}
.task-text{white-space:pre-wrap;overflow-wrap:anywhere}
.actions{display:flex;flex-wrap:wrap;gap:.5rem;margin:0 0 1.5rem}
button.copy-btn{background:var(--bg-card);color:var(--text);border:1px solid var(--border);
 border-radius:8px;padding:.4rem .75rem;font:inherit;font-size:.82rem;cursor:pointer}
button.copy-btn:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.sources{margin:0;padding-left:1.4rem;font-size:.9rem}
.sources li{margin-bottom:.3rem;overflow-wrap:anywhere}
.quote{border-left:3px solid var(--border);margin:.3rem 0 0;padding-left:.7rem;
 color:var(--text-muted);font-size:.86rem}
footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--border);
 color:var(--text-faint);font-size:.8rem}
@media print{body{background:#fff}.theme-toggle,.actions{display:none}
 .card,.chart,.table-wrap,.verdict,.prose{break-inside:avoid}}
"""
    )


_THEME_SCRIPT = """
(function(){
  var root=document.documentElement,btn=document.querySelector('.theme-toggle');
  if(!btn){return;}
  function label(){
    var explicit=root.getAttribute('data-theme');
    var dark=explicit?explicit==='dark'
      :window.matchMedia('(prefers-color-scheme: dark)').matches;
    btn.textContent=dark?'Light theme':'Dark theme';
    btn.setAttribute('aria-label','Switch to '+(dark?'light':'dark')+' theme');
  }
  btn.addEventListener('click',function(){
    var explicit=root.getAttribute('data-theme');
    var dark=explicit?explicit==='dark'
      :window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.setAttribute('data-theme',dark?'light':'dark');
    label();
  });
  label();
})();
(function(){
  var buttons=document.querySelectorAll('button.copy-btn');
  if(!buttons.length){return;}
  Array.prototype.forEach.call(buttons,function(btn){
    var node=document.getElementById(btn.getAttribute('data-copy-from'));
    if(!node){btn.disabled=true;return;}
    var original=btn.textContent;
    btn.addEventListener('click',function(){
      var text=node.textContent;
      function done(ok){
        btn.textContent=ok?'Copied':'Press Ctrl+C to copy';
        window.setTimeout(function(){btn.textContent=original;},2000);
      }
      if(navigator.clipboard&&navigator.clipboard.writeText){
        navigator.clipboard.writeText(text).then(function(){done(true);},
          function(){fallback(text,done);});
      }else{fallback(text,done);}
    });
  });
  function fallback(text,done){
    var area=document.createElement('textarea');
    area.value=text;area.setAttribute('readonly','');
    area.style.position='fixed';area.style.opacity='0';
    document.body.appendChild(area);area.select();
    var ok=false;
    try{ok=document.execCommand('copy');}catch(e){ok=false;}
    document.body.removeChild(area);done(ok);
  }
})();
"""


def _provenance_items(command: str | None, generated_at: str) -> str:
    from effgen import __version__

    items = [
        f"<li>Generated {_esc(generated_at)}</li>",
        f"<li>effGen {_esc(__version__)}</li>",
    ]
    if command:
        items.append(f"<li><code>{_esc(command)}</code></li>")
    return '<ul class="provenance">' + "".join(items) + "</ul>"


def _page(
    *,
    title: str,
    subtitle: str,
    command: str | None,
    generated_at: str,
    body: str,
) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{_css()}</style>\n</head>\n<body>\n<main>\n"
        '<header class="report-head"><div>'
        f"<h1>{_esc(title)}</h1>"
        f'<p class="subtitle">{_esc(subtitle)}</p>'
        f"{_provenance_items(command, generated_at)}"
        "</div>"
        '<button type="button" class="theme-toggle">Dark theme</button>'
        "</header>\n"
        f"{body}\n"
        "<footer>Rendered from the result of the command above. "
        "All styles, scripts, and charts are contained in this file.</footer>\n"
        "</main>\n"
        f"<script>{_THEME_SCRIPT}</script>\n</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# Report kinds
# ---------------------------------------------------------------------------

def detect_report_kind(data: dict[str, Any]) -> str | None:
    """Return the report kind a result document describes, or ``None``.

    Recognizes the ``--json`` documents emitted by ``run`` (``output`` +
    ``success``), ``compare`` (``scores``), ``eval`` (``suite`` + ``results``),
    ``cost`` (``rows`` + ``period``), ``loadtest`` (``latency`` +
    ``throughput_rps``), and ``battle`` (``contenders`` + ``prompt``), plus a
    stored run-history record (``run_id`` + ``status``).
    """
    if not isinstance(data, dict):
        return None
    if "contenders" in data and "prompt" in data:
        return "battle"
    if "scores" in data and "recommendations" in data:
        return "comparison"
    if "results" in data and "suite" in data:
        return "eval"
    if "rows" in data and "period" in data:
        return "cost"
    if "latency" in data and "throughput_rps" in data:
        return "loadtest"
    if "output" in data and ("success" in data or "execution_tree" in data):
        return "run"
    if "run_id" in data and "status" in data and "ts" in data:
        return "run"
    return None


def _children(node: Any) -> list[Any]:
    """The child nodes of a tree node, ignoring a malformed ``children`` value."""
    children = node.get("children") if isinstance(node, dict) else None
    return children if isinstance(children, list) else []


def _sort_key(value: Any) -> float:
    """A start time as a number, placing an unusable one first."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _tool_steps(tree: Any) -> list[dict[str, Any]]:
    """Flatten a run's execution tree into its tool nodes, in start order."""
    steps: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if str(node.get("node_type") or "") == "tool":
            steps.append(node)
        for child in _children(node):
            walk(child)

    for child in _children(tree):
        walk(child)
    steps.sort(key=lambda n: _sort_key(n.get("started_at")))
    return steps


def _mapping(value: Any) -> dict[str, Any]:
    """*value* when it is a mapping, an empty mapping otherwise."""
    return value if isinstance(value, dict) else {}


#: Engine prefixes that mark an ``engine:model_id`` string as running on the
#: caller's own hardware, mirroring the model loader's local engines.
_LOCAL_ENGINES = ("transformers:", "vllm:", "gguf:", "mlx:")


def _unpriced_label(model: str, provider: str) -> str:
    """What to show in place of a cost the run does not carry.

    A run on local hardware has no price to report. A hosted model can also
    reach here when the catalog has no rate for it, and that is a gap in the
    pricing data rather than a free call, so the two are not labelled alike.
    """
    if model.startswith(_LOCAL_ENGINES) or not provider:
        return "unpriced (local)"
    return "unpriced (no published rate)"


def _sequence(value: Any) -> list[Any]:
    """*value* as a list of items, treating a non-sequence as empty.

    A bare string is not a one-item sequence here: iterating it would render
    one entry per character.
    """
    return list(value) if isinstance(value, list | tuple) else []


def _run_body(data: dict[str, Any]) -> tuple[str, str, str]:
    metadata = _mapping(data.get("metadata"))
    # A stored history record names its answer `output` and its timing
    # `duration_s`, so both the full run document and the history record
    # render through the same path.
    summary_only = "success" not in data and "status" in data
    success = (
        str(data.get("status")) == "ok" if summary_only else bool(data.get("success"))
    )
    task = str(data.get("task") or metadata.get("task") or "")
    model = str(data.get("model") or metadata.get("model") or "")
    provider = str(data.get("provider") or metadata.get("provider") or "")
    run_id = str(data.get("run_id") or metadata.get("run_id") or "")
    duration = data.get("execution_time")
    if duration is None:
        duration = data.get("duration_s", metadata.get("duration_s"))
    cost = metadata.get("cost_usd", data.get("cost_usd"))
    prompt_tokens = metadata.get("prompt_tokens", data.get("input_tokens"))
    completion_tokens = metadata.get("completion_tokens", data.get("output_tokens"))
    total_tokens = metadata.get("total_tokens", data.get("tokens_used"))
    if total_tokens is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total_tokens = prompt_tokens + completion_tokens

    target = model or "the configured model"
    if provider:
        target = f"{target} ({provider})"

    parts: list[str] = [
        '<div class="verdict">'
        '<div class="verdict-label">Task</div>'
        f'<div class="verdict-value task-text" id="run-task">{_esc(task) or _DASH}</div>'
        f'<p class="verdict-note">{_badge("succeeded" if success else "failed", "ok" if success else "err")} '
        f"on {_esc(target)}"
        + (f" · run {_esc(run_id)}" if run_id else "")
        + "</p></div>"
    ]

    if task:
        parts.append(
            '<div class="actions">'
            '<button type="button" class="copy-btn" data-copy-from="run-task">'
            "Copy task</button>"
            '<button type="button" class="copy-btn" data-copy-from="run-command">'
            "Copy command</button>"
            f'<span hidden id="run-command">{_esc(_run_command(task, model))}</span>'
            "</div>"
        )

    steps = _tool_steps(data.get("execution_tree"))
    tool_calls = data.get("tool_calls")
    if tool_calls is None:
        tool_calls = len(steps) if steps else None
    parts.append(_cards([
        ("Duration", _secs(duration, 2), f"{_int(data.get('iterations'))} iterations"
         if data.get("iterations") is not None else ""),
        ("Tool calls", _int(tool_calls), f"{len(steps)} recorded steps" if steps else ""),
        (
            "Tokens",
            _int(total_tokens),
            f"{_int(prompt_tokens)} in / {_int(completion_tokens)} out"
            if prompt_tokens is not None or completion_tokens is not None else "",
        ),
        # A model with no published price records no cost at all. Reporting
        # $0.00 for it would read as a free cloud call rather than a local run.
        ("Cost", _usd(cost, 6) if cost is not None else _unpriced_label(model, provider), ""),
    ]))

    error = metadata.get("error")
    if not success and isinstance(error, dict):
        parts.append("<h2>Error</h2>")
        parts.append(_table(
            ["Field", "Value"],
            [
                ["Type", _esc(error.get("type") or _DASH)],
                ["Category", _esc(error.get("category") or _DASH)],
                ["Provider", _esc(error.get("provider") or _DASH)],
                ["Model", _esc(error.get("model") or _DASH)],
                ["Retryable", _esc("yes" if error.get("retryable") else "no")],
                ["Message", f'<span class="wrap">{_esc(error.get("message") or _DASH)}</span>'],
            ],
        ))
    elif not success and data.get("error"):
        parts.append("<h2>Error</h2>")
        parts.append(f'<div class="prose">{_esc(data.get("error"))}</div>')

    answer = str(data.get("output") or "")
    parts.append("<h2>Answer</h2>")
    if answer.strip():
        parts.append(f'<div class="prose">{_esc(answer)}</div>')
        if summary_only:
            parts.append(
                '<p class="empty">Rendered from stored run history, which keeps a '
                "truncated answer and no step trace. Export at run time with "
                "<code>effgen run --card</code> for the full answer, the tool "
                "trace and sources.</p>"
            )
    else:
        parts.append('<p class="empty">The run produced no answer text.</p>')

    if steps:
        parts.append("<h2>Steps</h2>")
        parts.append(_bar_chart(
            "Step duration",
            [
                (
                    f"{idx}. {node.get('name') or 'step'}",
                    node.get("duration"),
                    _secs(node.get("duration"), 3),
                )
                for idx, node in enumerate(steps, start=1)
            ],
            role="accent2",
        ))
        rows = []
        for idx, node in enumerate(steps, start=1):
            meta = _mapping(node.get("metadata"))
            failed = str(node.get("status") or "") == "failed"
            outcome = meta.get("error") if failed else meta.get("result")
            rows.append([
                f'<span class="num">{idx}</span>',
                _esc(node.get("name") or _DASH),
                f'<span class="mono">{_esc(_truncate(meta.get("tool_input") or _DASH, 300))}</span>',
                _badge("failed" if failed else "ok", "err" if failed else "ok"),
                f'<span class="mono">{_esc(_truncate(outcome if outcome is not None else _DASH, 300))}</span>',
                f'<span class="num">{_secs(node.get("duration"), 3)}</span>',
            ])
        parts.append(_table(
            ["#", "Tool", "Input", "Status", "Result", "Duration"], rows,
        ))

    sources = [s for s in _sequence(data.get("sources")) if s]
    if sources:
        parts.append("<h2>Sources</h2>")
        parts.append(
            '<ol class="sources">'
            + "".join(f"<li>{_link(s)}</li>" for s in sources)
            + "</ol>"
        )

    citations = _sequence(data.get("citations"))
    if citations:
        parts.append("<h2>Citations</h2>")
        rows = []
        for c in citations:
            if not isinstance(c, dict):
                rows.append([_DASH, f'<span class="wrap">{_esc(c)}</span>'])
                continue
            quote = str(c.get("quote") or "")
            rows.append([
                f'<span class="num">{_esc(c.get("index", _DASH))}</span>',
                _link(c.get("source"))
                + (f'<p class="quote">{_esc(_truncate(quote, 400))}</p>' if quote else ""),
            ])
        parts.append(_table(["#", "Source"], rows))

    subtitle = _truncate(task, 160) if task else "Agent run"
    return "Run Card", subtitle, "".join(parts)


def _run_command(task: str, model: str) -> str:
    """The ``effgen run`` invocation that reproduces this run."""
    import shlex

    argv = ["effgen", "run", shlex.quote(task)]
    if model:
        argv.extend(["-m", shlex.quote(model)])
    return " ".join(argv)


def _comparison_body(data: dict[str, Any]) -> tuple[str, str, str]:
    scores = data.get("scores") or []
    if not scores:
        raise ReportError("The comparison result carries no scores to render.")
    suites = sorted({str(s.get("suite", "")) for s in scores})
    models = sorted({str(s.get("model", "")) for s in scores})
    optimize = str(data.get("optimize", "accuracy"))
    recommendations = data.get("recommendations") or {}
    rationales = data.get("recommendation_rationale") or {}

    parts: list[str] = []

    # Hero verdict — the recommended model per suite plus why it won.
    for suite in suites:
        model = recommendations.get(suite)
        if not model:
            continue
        note = rationales.get(suite) or f"Optimized for {optimize}."
        parts.append(
            '<div class="verdict">'
            f'<div class="verdict-label">Recommended for {_esc(suite)} '
            f"(optimized for {_esc(optimize)})</div>"
            f'<div class="verdict-value">{_esc(model)}</div>'
            f'<p class="verdict-note">{_esc(note)}</p></div>'
        )

    num_cases = data.get("num_cases")
    parts.append(_cards([
        ("Models", str(len(models)), ""),
        ("Suites", ", ".join(suites) if suites else _DASH, ""),
        ("Cases per model", _int(num_cases) if num_cases is not None else _DASH, ""),
        ("Scoring", str(data.get("scoring") or _DASH), ""),
    ]))

    # Under llm_judge scoring, say what graded the answers so a reader knows
    # whether the scores came from a model with a stake in them.
    self_judged = data.get("self_judged")
    if self_judged is not None:
        note = (
            "Each model graded its own answers."
            if self_judged
            else f"Graded by {data.get('judge_model') or 'a named judge'}, "
                 "which is not one of the compared models."
        )
        parts.append(f'<p class="verdict-note">{_esc(note)}</p>')

    parts.append("<h2>Results</h2>")
    rows = []
    for s in scores:
        error = s.get("error")
        if error:
            status = _badge("ERROR", "err")
            cells = [_DASH, _DASH, _DASH, _DASH]
        else:
            status = _badge("ok", "ok")
            cost = s.get("avg_cost_usd")
            cells = [
                _pct(s.get("accuracy")),
                _secs(s.get("avg_latency")),
                _usd(cost) if cost is not None else "unpriced",
                _int(s.get("total_tokens")),
            ]
        rows.append([
            f'<span class="wrap">{_esc(s.get("model"))}</span>',
            _esc(s.get("suite")),
            status,
            *[f'<span class="num">{c}</span>' for c in cells],
        ])
    parts.append(_table(
        ["Model", "Suite", "Status", "Accuracy", "Avg latency", "Avg cost/run", "Tokens"],
        rows,
    ))

    ok = [s for s in scores if not s.get("error")]
    if ok:
        parts.append("<h2>Charts</h2>")
        parts.append(_bar_chart(
            "Accuracy",
            [(str(s.get("model")), s.get("accuracy"), _pct(s.get("accuracy"))) for s in ok],
            role="ok",
            maximum=1.0,
        ))
        parts.append(_bar_chart(
            "Average cost per run (unpriced models shown as a hatched bar)",
            [
                (
                    str(s.get("model")),
                    s.get("avg_cost_usd"),
                    _usd(s.get("avg_cost_usd")) if s.get("avg_cost_usd") is not None else "unpriced",
                )
                for s in ok
            ],
            role="accent",
        ))
        parts.append(_bar_chart(
            "Average latency",
            [(str(s.get("model")), s.get("avg_latency"), _secs(s.get("avg_latency"))) for s in ok],
            role="accent2",
        ))

    # What each model actually answered, so a shared bake-off shows its work
    # rather than only the percentages above.
    answered = [s for s in scores if _sequence(s.get("responses"))]
    if answered:
        parts.append("<h2>Answers</h2>")
        for s in answered:
            parts.append(f'<h3 class="wrap">{_esc(s.get("model"))}</h3>')
            rows = []
            for raw in _sequence(s.get("responses")):
                r = _mapping(raw)
                if r.get("error"):
                    outcome = _badge("ERROR", "err")
                elif r.get("passed"):
                    outcome = _badge("pass", "ok")
                else:
                    outcome = _badge("fail", "warn")
                rows.append([
                    f'<span class="wrap">{_esc(_truncate(str(r.get("query", "")), 160))}</span>',
                    f'<span class="wrap">{_esc(_truncate(str(r.get("output", "")), 400))}</span>',
                    outcome,
                ])
            parts.append(_table(["Case", "Answer", "Result"], rows))

    subtitle = (
        f"{len(models)} models on {', '.join(suites)}"
        + (f" · {num_cases} cases" if num_cases is not None else "")
        + f" · scoring {data.get('scoring') or 'contains'}"
    )
    return "Model Comparison", subtitle, "".join(parts)


def _eval_body(data: dict[str, Any]) -> tuple[str, str, str]:
    suite = str(data.get("suite") or "suite")
    total = data.get("total")
    passed = data.get("passed")
    accuracy = data.get("accuracy")
    metadata = data.get("metadata") or {}
    fail_under = data.get("fail_under", metadata.get("fail_under"))

    parts: list[str] = []
    if isinstance(fail_under, int | float) and isinstance(accuracy, int | float):
        gate_passed = accuracy >= fail_under
        parts.append(
            '<div class="verdict">'
            '<div class="verdict-label">Exit gate</div>'
            '<div class="verdict-value">'
            f'{_badge("PASS" if gate_passed else "FAIL", "ok" if gate_passed else "err")}</div>'
            f'<p class="verdict-note">Accuracy {_pct(accuracy)} '
            f'{">=" if gate_passed else "<"} the required {_pct(fail_under, 0)}.</p></div>'
        )

    parts.append(_cards([
        ("Pass rate", _pct(accuracy), f"{_int(passed)} of {_int(total)} cases"),
        ("Avg latency", _secs(data.get("avg_latency"), 4), ""),
        ("Total tokens", _int(data.get("total_tokens")), ""),
        (
            "Total cost",
            _usd(data.get("total_cost_usd")) if data.get("total_cost_usd") is not None else "unpriced",
            str(metadata.get("model") or ""),
        ),
    ]))

    by_difficulty = data.get("by_difficulty") or {}
    if by_difficulty:
        parts.append("<h2>By difficulty</h2>")
        parts.append(_bar_chart(
            "Accuracy by difficulty",
            [
                (
                    f"{name} ({info.get('passed')}/{info.get('total')})",
                    info.get("accuracy"),
                    _pct(info.get("accuracy")),
                )
                for name, info in sorted(by_difficulty.items())
            ],
            role="ok",
            maximum=1.0,
        ))

    results = data.get("results") or []
    if results:
        parts.append("<h2>Cases</h2>")
        rows = []
        for r in results:
            passed_case = bool(r.get("passed"))
            rows.append([
                f'<span class="wrap">{_esc(_truncate(r.get("query", "")))}</span>',
                f'<span class="wrap">{_esc(_truncate(r.get("expected_output", ""), 120))}</span>',
                f'<span class="wrap">{_esc(_truncate(r.get("agent_output", ""), 200))}</span>',
                _badge("PASS" if passed_case else "FAIL", "ok" if passed_case else "err"),
                f'<span class="num">{_secs(r.get("latency"))}</span>',
                (f'<span class="num">'
                f'{_usd(r.get("cost_usd")) if r.get("cost_usd") is not None else "unpriced"}'
                "</span>"),
                _esc(r.get("difficulty") or _DASH),
            ])
        parts.append(_table(
            ["Query", "Expected", "Got", "Result", "Latency", "Cost", "Difficulty"],
            rows,
        ))

    subtitle = (
        f"{suite} · {_int(passed)}/{_int(total)} passed"
        + (f" · scoring {metadata['scoring']}" if metadata.get("scoring") else "")
    )
    return f"Evaluation Report — {suite}", subtitle, "".join(parts)


def _cost_body(data: dict[str, Any]) -> tuple[str, str, str]:
    period = str(data.get("period") or "Spend")
    rows_in = data.get("rows") or []
    total = data.get("total_cost_usd")
    budget = data.get("daily_budget_usd")

    # The budget is configured per day, so it is only a like-for-like comparison
    # once scaled to the window this spend covers. A lifetime total spans no
    # fixed window, so it is reported without a budget verdict.
    period_days = data.get("period_days")
    scaled_budget: float | None = None
    if isinstance(budget, int | float) and not isinstance(budget, bool):
        if isinstance(period_days, int) and not isinstance(period_days, bool) and period_days > 0:
            scaled_budget = float(budget) * period_days

    parts: list[str] = [_cards([
        ("Total spend", _usd(total, 4) if total is not None else _DASH, period),
        ("Requests", _int(data.get("total_requests")), ""),
        ("Providers", str(len({r.get("provider") for r in rows_in})) if rows_in else "0", ""),
        (
            "Daily budget",
            _usd(budget, 2) if budget is not None else "not set",
            "" if budget is not None else "effgen cost set-budget",
        ),
    ])]

    if scaled_budget is not None:
        window = "day" if period_days == 1 else f"{period_days} days"
        note = "" if period_days == 1 else f"({_usd(budget, 2)}/day over {window})"
        parts.append(_meter(
            f"Budget for this period — {window}",
            float(total or 0.0),
            scaled_budget,
            note=note,
        ))
    elif budget is not None:
        parts.append(
            '<section class="chart"><h3>Budget</h3>'
            f'<p class="empty">The configured budget is {_usd(budget, 2)} per day. '
            f"This total covers {_esc(period.lower())}, which spans no fixed budget "
            "window, so it is not measured against that limit.</p></section>"
        )
    else:
        parts.append(_meter("Daily budget", float(total or 0.0), budget))

    if not rows_in:
        parts.append('<p class="empty">No spend recorded for this period.</p>')
        return f"Spend Report — {period}", period, "".join(parts)

    parts.append("<h2>Spend by provider and model</h2>")
    table_rows = []
    for r in rows_in:
        table_rows.append([
            _esc(r.get("provider")),
            f'<span class="wrap">{_esc(r.get("model"))}</span>',
            f'<span class="num">{_int(r.get("requests"))}</span>',
            f'<span class="num">{_int(r.get("prompt_tokens"))}</span>',
            f'<span class="num">{_int(r.get("completion_tokens"))}</span>',
            f'<span class="num">{_esc(r.get("cost_label") or _usd(r.get("cost_usd")))}</span>',
        ])
    parts.append(_table(
        ["Provider", "Model", "Requests", "Prompt tokens", "Completion tokens", "Cost"],
        table_rows,
    ))

    by_provider: dict[str, float] = {}
    for r in rows_in:
        by_provider[str(r.get("provider"))] = by_provider.get(str(r.get("provider")), 0.0) + float(
            r.get("cost_usd") or 0.0
        )
    ranked = sorted(by_provider.items(), key=lambda kv: kv[1], reverse=True)
    parts.append("<h2>Charts</h2>")
    parts.append(_bar_chart(
        "Cost by provider",
        [(name, value, _usd(value)) for name, value in ranked],
        role="accent",
    ))
    donut = _donut("Share of spend", ranked)
    if donut:
        parts.append(donut)

    return (
        f"Spend Report — {period}",
        (f"{_usd(total, 4) if total is not None else _DASH} across "
        f"{_int(data.get('total_requests'))} requests"),
        "".join(parts),
    )


def _loadtest_body(data: dict[str, Any]) -> tuple[str, str, str]:
    lat = data.get("latency") or {}
    error_rate = data.get("error_rate")
    scenario = str(data.get("scenario") or "load test")

    role = "ok"
    state = "no failed requests"
    if isinstance(error_rate, int | float):
        if error_rate >= 0.05:
            role, state = "err", "elevated error rate"
        elif error_rate > 0:
            role, state = "warn", "some requests failed"

    parts: list[str] = [
        ('<div class="verdict">'
        '<div class="verdict-label">Outcome</div>'
        f'<div class="verdict-value">{_pct(error_rate, 2)} errors '
        f"{_badge(state, role)}</div>"
        f'<p class="verdict-note">{_int(data.get("successful_requests"))} of '
        f'{_int(data.get("total_requests"))} requests succeeded at '
        f'{data.get("concurrency")} concurrent users.</p></div>'),
        _cards([
            ("Throughput", _rps(data.get("throughput_rps")), ""),
            ("p95 latency", _ms(lat.get("p95")), f"p50 {_ms(lat.get('p50'))}"),
            ("p99 latency", _ms(lat.get("p99")), f"max {_ms(lat.get('max'))}"),
            (
                "Duration",
                _secs(data.get("duration_s"), 1),
                f"requested {data.get('requested_duration_s')}s",
            ),
        ]),
        "<h2>Latency distribution</h2>",
        _bar_chart(
            "Percentiles",
            [
                (name.upper(), lat.get(key), _ms(lat.get(key)))
                for name, key in (
                    ("min", "min"), ("p50", "p50"), ("mean", "mean"),
                    ("p95", "p95"), ("p99", "p99"), ("max", "max"),
                )
            ],
            role="accent2",
        ),
    ]

    parts.append("<h2>Run</h2>")
    target = data.get("model") or data.get("provider")
    parts.append(_table(
        ["Setting", "Value"],
        [
            ["Scenario", _esc(scenario)],
            ["Concurrency", f'<span class="num">{_int(data.get("concurrency"))}</span>'],
            ["Target", _esc(target) if target else "local mock"],
            ["Provider", _esc(data.get("provider")) if data.get("provider") else _DASH],
            [
                "Drain after window",
                f'<span class="num">{_secs(data.get("drain_s"), 1)}</span>',
            ],
        ],
    ))

    breakdown = data.get("error_breakdown") or {}
    parts.append("<h2>Errors</h2>")
    if breakdown:
        parts.append(_table(
            ["Error category", "Count"],
            [
                [_esc(k), f'<span class="num">{_int(v)}</span>']
                for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1])
            ],
        ))
    else:
        parts.append('<p class="empty">No failed requests to break down.</p>')

    return (
        f"Load Test Report — {scenario}",
        (f"{_int(data.get('total_requests'))} requests · "
        f"{data.get('concurrency')} concurrent · {_pct(error_rate, 2)} errors"),
        "".join(parts),
    )


def _battle_body(data: dict[str, Any]) -> tuple[str, str, str]:
    """Render a head-to-head: the prompt, the tally, the verdict, the answers."""
    contenders = _sequence(data.get("contenders"))
    if not contenders:
        raise ReportError("The battle result carries no contenders to render.")
    prompt = str(data.get("prompt") or "")
    verdict = _mapping(data.get("verdict"))
    finishers = [c for c in contenders if _mapping(c).get("answer") and not _mapping(c).get("error")]

    parts: list[str] = []

    for key, label in (("fastest", "Fastest"), ("cheapest", "Cheapest")):
        entry = _mapping(verdict.get(key))
        if entry.get("model"):
            parts.append(
                '<div class="verdict">'
                f'<div class="verdict-label">{label}</div>'
                f'<div class="verdict-value">{_esc(entry.get("model"))}</div>'
                f'<p class="verdict-note">{_esc(entry.get("detail", ""))}</p></div>'
            )

    judge = _mapping(verdict.get("judge"))
    if judge.get("winner"):
        parts.append(
            '<div class="verdict">'
            '<div class="verdict-label">Judged pick</div>'
            f'<div class="verdict-value">{_esc(judge.get("winner"))}</div>'
            f'<p class="verdict-note">{_esc(judge.get("reasoning", ""))} '
            f'Judged by {_esc(judge.get("judge_model"))}, separately from the '
            "measurements above.</p></div>"
        )

    parts.append('<h2>Prompt</h2><div class="answer">' + _esc(prompt) + "</div>")

    total_cost = data.get("total_cost_usd")
    if not finishers:
        # Nothing ran, so there is no price to report — not a set of models
        # that publish no price.
        spent = _DASH
    else:
        spent = _usd(total_cost) if total_cost is not None else "unpriced"
    parts.append(_cards([
        ("Contenders", str(len(contenders)), f"{len(finishers)} answered"),
        ("Wall clock", _secs(data.get("wall_s"), 2), "all models in parallel"),
        ("Total cost", spent, ""),
    ]))

    parts.append("<h2>Tally</h2>")
    rows = []
    estimated_any = False
    for raw in contenders:
        c = _mapping(raw)
        if c.get("error"):
            status = _badge("failed", "err")
            cells = [_DASH, _DASH, _DASH, _DASH]
        else:
            status = _badge("answered", "ok")
            cost = c.get("cost_usd")
            tokens = c.get("total_tokens")
            if c.get("estimated_tokens"):
                estimated_any = True
            cells = [
                _secs(c.get("ttft_s"), 2),
                _secs(c.get("latency_s"), 2),
                (_int(tokens) + ("*" if c.get("estimated_tokens") else "")
                 if tokens is not None else _DASH),
                _usd(cost) if cost is not None else "unpriced",
            ]
        rows.append([
            f'<span class="wrap">{_esc(c.get("model"))}</span>',
            status,
            *[f'<span class="num">{cell}</span>' for cell in cells],
        ])
    parts.append(_table(
        ["Model", "Result", "First token", "Latency", "Tokens", "Cost"],
        rows,
        caption=("* counted locally, not reported by the provider" if estimated_any else ""),
    ))

    timed = [
        (str(_mapping(c).get("model")), _mapping(c).get("latency_s"),
         _secs(_mapping(c).get("latency_s"), 2))
        for c in finishers
        if _mapping(c).get("latency_s") is not None
    ]
    if timed:
        parts.append("<h2>Charts</h2>")
        parts.append(_bar_chart("Latency", timed, role="accent2"))
        priced = [
            (str(_mapping(c).get("model")), _mapping(c).get("cost_usd"),
             _usd(_mapping(c).get("cost_usd"))
             if _mapping(c).get("cost_usd") is not None else "unpriced")
            for c in finishers
        ]
        parts.append(_bar_chart(
            "Cost for this run (unpriced models shown as a hatched bar)",
            priced,
            role="accent",
        ))

    parts.append("<h2>Answers</h2>")
    for raw in contenders:
        c = _mapping(raw)
        parts.append(f'<h3 class="wrap">{_esc(c.get("model"))}</h3>')
        if c.get("error"):
            parts.append(f'<div class="answer err">{_esc(c.get("error"))}</div>')
        else:
            parts.append(f'<div class="answer">{_esc(c.get("answer") or "")}</div>')

    subtitle = (
        f"{len(contenders)} models on one prompt · "
        f"{len(finishers)} answered in {_secs(data.get('wall_s'), 2)}"
    )
    return "Model Battle", subtitle, "".join(parts)


_BODY_RENDERERS = {
    "run": _run_body,
    "comparison": _comparison_body,
    "eval": _eval_body,
    "cost": _cost_body,
    "loadtest": _loadtest_body,
    "battle": _battle_body,
}


def _require_kind_data(data: dict[str, Any], kind: str) -> None:
    """Refuse a document that carries none of the keys *kind* renders.

    Rendering it would produce a page of em dashes, so the mismatch is reported
    instead — naming the keys the kind needs, the keys the document has, and
    the kind the document does look like when that can be told.

    Raises:
        ReportError: If the document has none of the kind's keys.
    """
    required = _KIND_KEYS.get(kind, ())
    if not required or any(key in data for key in required):
        return
    present = ", ".join(sorted(data)[:8]) or "no keys"
    actual = detect_report_kind(data)
    hint = (
        f" This document looks like a {actual} result — render it with --kind {actual}."
        if actual and actual != kind
        else " Pass the JSON that command emitted, or --kind for the shape it is."
    )
    raise ReportError(
        f"This document has none of the fields the '{kind}' report renders "
        f"({', '.join(required)}). It contains: {present}.{hint}"
    )


def build_html_report(
    data: dict[str, Any],
    *,
    kind: str | None = None,
    command: str | None = None,
    generated_at: str | None = None,
) -> str:
    """Return a self-contained HTML document for a result *data* mapping.

    Args:
        data: The result document — the same mapping a command's ``--json``
            emits (``AgentResponse.to_dict()``, ``ComparisonMatrix.to_dict()``,
            ``SuiteResults.summary()``, the ``cost`` spend document, or
            ``LoadReport.to_dict()``).
        kind: One of :data:`REPORT_KINDS`. Inferred from the document shape
            when omitted.
        command: The invocation that produced the result, stamped into the
            report header so a shared file can be traced back and re-run.
        generated_at: Timestamp for the header. Defaults to the current UTC
            time in ISO-8601 form.

    Returns:
        A complete HTML document with every style, script, and chart inline.

    Raises:
        ReportError: If *kind* is unknown, cannot be inferred, or the document
            is missing the data the report needs.
    """
    if not isinstance(data, dict):
        raise ReportError("A report is rendered from a JSON object, not a list or scalar.")
    resolved = kind or detect_report_kind(data)
    if resolved is None:
        raise ReportError(
            "Could not tell which report this result is. Expected the JSON emitted by "
            "`effgen run --json`, `effgen compare --json`, `effgen eval --json`, "
            "`effgen cost --json`, or `effgen loadtest`. Pass --kind to say explicitly "
            f"({', '.join(REPORT_KINDS)})."
        )
    if resolved not in _BODY_RENDERERS:
        raise ReportError(
            f"Unknown report kind '{resolved}'. Choose one of: {', '.join(REPORT_KINDS)}."
        )
    _require_kind_data(data, resolved)
    stamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title, subtitle, body = _BODY_RENDERERS[resolved](data)
    return _page(
        title=title, subtitle=subtitle, command=command, generated_at=stamp, body=body,
    )


def write_html_report(
    path: str | Path,
    data: dict[str, Any],
    *,
    kind: str | None = None,
    command: str | None = None,
) -> Path:
    """Render *data* and write it to *path*, creating parent directories.

    Returns the path written.
    """
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_html_report(data, kind=kind, command=command), encoding="utf-8"
    )
    return target


def load_result_document(path: str | Path) -> dict[str, Any]:
    """Read a saved ``--json`` result document from *path*.

    Raises:
        ReportError: If the file is missing or is not a JSON object.
    """
    src = Path(path)
    if not src.exists():
        raise ReportError(f"No such result file: {src}")
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportError(
            f"{src} is not valid JSON ({exc}). Save one first with, for example, "
            "`effgen eval --suite math --json > results.json`."
        ) from exc
    if not isinstance(data, dict):
        raise ReportError(
            f"{src} holds a {type(data).__name__}, not a result object. Expected the "
            "JSON emitted by run/compare/eval/cost/loadtest."
        )
    return data
