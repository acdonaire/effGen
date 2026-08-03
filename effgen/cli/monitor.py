"""Terminal mission-control view over effGen's existing telemetry.

``effgen top`` (alias ``effgen monitor``) assembles one screen from the data
effGen already records:

- **Activity** — completed runs from the durable run history
  (:func:`effgen.observability.run_log.read_runs`), which is written by every
  process on the host.
- **Traffic** and **Per-model** — the running server's own counters, scraped
  from its ``/dashboard/data.json`` endpoint. These cover a single server
  process since it started.
- **Spend** — the local SQLite cost ledger: a 24-hour total against the
  configured daily budget, plus a trailing-hour burn rate.
- **GPU** — physical device memory and utilisation across all processes, the
  same view ``effgen models status`` reports.

Those sources measure different things over different windows, so every panel
states its own scope and the numbers are never blended into a single figure.

Only the traffic and per-model panels need a server. When none is reachable
they are shown as unavailable, naming the URL that was tried, so an empty
screen is never mistaken for zero traffic.

On an interactive terminal the view refreshes in place until ``q`` or Ctrl-C.
When stdout is not a terminal, or ``--json``/``--once`` is passed, or animation
is disabled (``--no-animation``, ``NO_COLOR``, ``EFFGEN_NO_ANIM=1``), it prints
one snapshot and exits.
"""

from __future__ import annotations

import http.client
import json
import os
import select
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from effgen.ui.render import json_ensure_ascii
from effgen.ui.tables import console_is_interactive, render_table
from effgen.ui.theme import color_enabled, get_console

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_INTERVAL = 2.0
MIN_INTERVAL = 0.5
MAX_INTERVAL = 300.0
DEFAULT_ACTIVITY_LIMIT = 12
BURN_WINDOW_S = 3600.0
_HTTP_TIMEOUT_S = 3.0
# The GPU sample is the slowest source, so it runs on a slower cadence than the
# rest of the snapshot and the previous reading is reused in between.
GPU_SAMPLE_INTERVAL_S = 5.0

# Scope labels. Each panel carries one so a reader can tell which window and
# which process a number covers.
SCOPE_ACTIVITY = "completed runs, local run history (all processes)"
SCOPE_TRAFFIC = "this server process since start"
# The request/error counters cover generation calls; the status histogram comes
# from the HTTP request counter, which also records health checks, metrics
# scrapes and dashboard reads. The two populations differ, so they are labelled
# separately rather than left to look like they should reconcile.
SCOPE_HTTP_STATUS = "all HTTP routes, this server process since start"
SCOPE_SPEND = "local cost ledger, last 24 hours"
SCOPE_GPU = "physical devices, all processes"


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def resolve_server_url(url: str | None = None, port: int | None = None) -> str:
    """Resolve the server base URL from *url*, *port*, the environment, or the default.

    Precedence: an explicit ``--url``, then ``--port`` on localhost, then
    ``EFFGEN_SERVER_URL``, then ``http://127.0.0.1:8000`` (the port
    ``effgen serve`` binds by default). A bare ``host:port`` gains an
    ``http://`` scheme; a trailing slash is dropped.
    """
    candidate = url or None
    if candidate is None and port is not None:
        candidate = f"http://127.0.0.1:{int(port)}"
    if candidate is None:
        candidate = os.environ.get("EFFGEN_SERVER_URL") or ""
    candidate = candidate.strip() or DEFAULT_SERVER_URL
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def resolve_api_key(api_key: str | None = None) -> str | None:
    """Resolve the server API key from *api_key* or ``EFFGEN_API_KEY``."""
    key = (api_key or os.environ.get("EFFGEN_API_KEY") or "").strip()
    return key or None


def _fetch_json(url: str, *, api_key: str | None, timeout: float = _HTTP_TIMEOUT_S) -> tuple[
    dict[str, Any] | None, str | None
]:
    """GET *url* and decode JSON. Returns ``(document, error_message)``.

    A URL problem is reported the same way an unreachable server is, so a
    mistyped ``--url`` leaves the panels that need no server intact instead of
    failing the whole command.
    """
    if not url.startswith(("http://", "https://")):
        return None, f"unsupported URL scheme: {url}"
    try:
        request = urllib.request.Request(url, method="GET")  # noqa: S310 - http(s) only, checked above
    except ValueError as exc:
        return None, f"invalid server URL {url} ({exc})"
    request.add_header("Accept", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = response.read().decode("utf-8", errors="replace")
        return json.loads(payload), None
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return None, (
                f"HTTP {exc.code} — the server requires an API key. Pass --api-key, "
                "set EFFGEN_API_KEY, or start the server with EFFGEN_PUBLIC_DASHBOARD=1."
            )
        return None, f"HTTP {exc.code} from {url}"
    except urllib.error.URLError as exc:
        return None, f"no server reachable at {url} ({exc.reason})"
    except TimeoutError:
        return None, f"no response from {url} within {timeout:.0f}s"
    except json.JSONDecodeError:
        return None, f"{url} did not return JSON"
    except ValueError as exc:
        return None, f"invalid server URL {url} ({exc})"
    except http.client.HTTPException as exc:
        # A host with spaces or control characters (InvalidURL) only fails once
        # a connection is attempted, and a port serving something that is not
        # HTTP raises here too. Neither is a ValueError or an OSError. The
        # detail can quote raw bytes, so it is flattened to one line rather than
        # allowed to break the panel it is rendered in.
        detail = " ".join(f"{type(exc).__name__}: {exc}".split())
        return None, f"{url} did not answer as an HTTP server ({_preview(detail, 120)})"
    except OSError as exc:
        return None, f"could not read {url} ({exc})"


# ---------------------------------------------------------------------------
# Panel collection
# ---------------------------------------------------------------------------


def _collect_activity(limit: int) -> dict[str, Any]:
    """Recent completed runs from the durable, cross-process run history."""
    try:
        from effgen.observability.run_log import read_runs

        runs = read_runs(limit=max(1, limit))
    except Exception as exc:  # noqa: BLE001 - a missing history must not blank the screen
        return {
            "scope": SCOPE_ACTIVITY,
            "available": False,
            "unavailable_reason": f"run history unreadable ({exc})",
            "runs": [],
        }
    rows = [
        {
            "ts": run.get("ts"),
            "run_id": run.get("run_id"),
            "status": run.get("status"),
            "model": run.get("model"),
            "agent": run.get("agent"),
            "duration_s": run.get("duration_s"),
            # ``None`` means the model has no published price; it is rendered as
            # an em dash and never summed as zero.
            "cost_usd": run.get("cost_usd"),
            "task": _preview(run.get("task"), 60),
            "error": run.get("error"),
        }
        for run in runs
    ]
    return {
        "scope": SCOPE_ACTIVITY,
        "available": True,
        "unavailable_reason": None,
        "runs": rows,
    }


def _collect_server(url: str, api_key: str | None) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Traffic and per-model panels from the server's dashboard aggregation."""
    document, error = _fetch_json(f"{url}/dashboard/data.json", api_key=api_key)
    if document is None:
        unavailable = {
            "scope": SCOPE_TRAFFIC,
            "available": False,
            "unavailable_reason": error,
            "server_url": url,
        }
        return (
            {**unavailable},
            {**unavailable, "rows": []},
            False,
        )

    metrics = document.get("metrics") or {}
    slo = document.get("slo") or {}
    requests = int(metrics.get("total_requests") or 0)
    errors = int(metrics.get("total_errors") or 0)
    traffic = {
        "scope": SCOPE_TRAFFIC,
        "available": True,
        "unavailable_reason": None,
        "server_url": url,
        "requests": requests,
        "errors": errors,
        "error_rate": (errors / requests) if requests else 0.0,
        "tokens": int(metrics.get("total_tokens") or 0),
        "cost_usd": metrics.get("cost_usd"),
        "avg_latency_s": metrics.get("avg_latency_s"),
        "p50_latency_s": slo.get("p50_latency_s"),
        "p95_latency_s": slo.get("p95_latency_s"),
        "p99_latency_s": slo.get("p99_latency_s"),
        "availability": slo.get("availability"),
        "by_status": document.get("by_status") or {},
        "by_status_scope": SCOPE_HTTP_STATUS,
    }
    by_model = {
        "scope": SCOPE_TRAFFIC,
        "available": True,
        "unavailable_reason": None,
        "server_url": url,
        "rows": list(document.get("by_model") or []),
    }
    return traffic, by_model, True


def _collect_spend() -> dict[str, Any]:
    """24-hour spend, daily budget, and a trailing-hour burn rate from the ledger."""
    panel: dict[str, Any] = {
        "scope": SCOPE_SPEND,
        "available": False,
        "unavailable_reason": None,
        "total_cost_usd": 0.0,
        "requests": 0,
        "daily_budget_usd": None,
        "budget_used_pct": None,
        "burn_rate_usd_per_hour": None,
        "burn_window_s": BURN_WINDOW_S,
        "unpriced_models": [],
        "rows": [],
    }
    try:
        from effgen.models._cost import _budget_config_path, pricing_status
        from effgen.models._cost_store import SQLiteCostStore

        store = SQLiteCostStore()
        events = store.query_today()
        recent = store.query_since(time.time() - BURN_WINDOW_S)
    except Exception as exc:  # noqa: BLE001 - an unreadable ledger must not blank the screen
        panel["unavailable_reason"] = f"cost ledger unreadable ({exc})"
        return panel

    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        key = (event.provider, event.model)
        row = aggregate.setdefault(
            key,
            {
                "provider": event.provider,
                "model": event.model,
                "requests": 0,
                "tokens": 0,
                "cost_usd": 0.0,
            },
        )
        row["requests"] += 1
        row["tokens"] += int(event.prompt_tokens or 0) + int(event.completion_tokens or 0)
        row["cost_usd"] += float(event.cost_usd or 0.0)

    rows = sorted(aggregate.values(), key=lambda r: r["cost_usd"], reverse=True)
    unpriced: list[str] = []
    for row in rows:
        if row["cost_usd"] > 0:
            row["cost_label"] = f"${row['cost_usd']:.6f}"
            continue
        try:
            status = pricing_status(row["provider"], row["model"])
        except Exception:  # noqa: BLE001 - an unknown model is simply unlabelled
            status = None
        if status == "unpriced":
            row["cost_label"] = "—"
            unpriced.append(f"{row['provider']}:{row['model']}")
        elif status == "free":
            row["cost_label"] = "free"
        else:
            row["cost_label"] = f"${row['cost_usd']:.6f}"

    total_cost = sum(float(r["cost_usd"]) for r in rows)
    burn = sum(float(e.cost_usd or 0.0) for e in recent) / (BURN_WINDOW_S / 3600.0)

    daily_budget = None
    try:
        budget_path = _budget_config_path()
        if budget_path.exists():
            daily_budget = json.loads(budget_path.read_text()).get("daily")
    except Exception:  # noqa: BLE001 - an unreadable budget file just means no budget shown
        daily_budget = None

    panel.update(
        {
            "available": True,
            "total_cost_usd": round(total_cost, 8),
            "requests": sum(int(r["requests"]) for r in rows),
            "daily_budget_usd": daily_budget,
            "budget_used_pct": (
                round(100.0 * total_cost / float(daily_budget), 2)
                if daily_budget
                else None
            ),
            "burn_rate_usd_per_hour": round(burn, 8),
            "unpriced_models": unpriced,
            "rows": [
                {
                    "provider": r["provider"],
                    "model": r["model"],
                    "requests": r["requests"],
                    "tokens": r["tokens"],
                    "cost_usd": round(float(r["cost_usd"]), 8),
                    "cost_label": r["cost_label"],
                }
                for r in rows
            ],
        }
    )
    return panel


def _collect_gpu() -> dict[str, Any]:
    """Per-device memory and utilisation for the physical GPUs on this host."""
    try:
        from effgen.gpu.cuda_compat import per_gpu_status

        devices = per_gpu_status()
    except Exception as exc:  # noqa: BLE001 - no GPU layer is a normal state, not an error
        return {
            "scope": SCOPE_GPU,
            "available": False,
            "unavailable_reason": f"GPU status unavailable ({exc})",
            "devices": [],
        }
    if not devices:
        return {
            "scope": SCOPE_GPU,
            "available": False,
            "unavailable_reason": "no CUDA devices detected on this host",
            "devices": [],
        }
    gib = 1024**3
    return {
        "scope": SCOPE_GPU,
        "available": True,
        "unavailable_reason": None,
        "devices": [
            {
                "index": device.index,
                "name": device.name,
                "total_gb": round(device.total_bytes / gib, 2),
                "used_gb": round(device.used_bytes / gib, 2),
                "free_gb": round(device.free_bytes / gib, 2),
                "memory_used_pct": (
                    round(100.0 * device.used_bytes / device.total_bytes, 1)
                    if device.total_bytes
                    else None
                ),
                "utilization_pct": device.utilization_pct,
            }
            for device in devices
        ],
    }


def collect_snapshot(
    *,
    url: str | None = None,
    port: int | None = None,
    api_key: str | None = None,
    limit: int = DEFAULT_ACTIVITY_LIMIT,
    interval: float | None = None,
    gpu: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect one full monitor snapshot.

    Returns a document with a ``header`` plus the ``activity``, ``traffic``,
    ``by_model``, ``spend`` and ``gpu`` panels. Each panel carries a ``scope``
    describing the window and process it measures, and an ``available`` flag
    with an ``unavailable_reason`` when its source could not be read. Pass a
    previously collected *gpu* panel to reuse it instead of re-sampling.
    """
    server_url = resolve_server_url(url, port)
    key = resolve_api_key(api_key)
    traffic, by_model, reachable = _collect_server(server_url, key)
    return {
        "ts": datetime.now(timezone.utc).strftime(  # noqa: UP017 - `timezone.utc` needed for py3.10 support
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "header": {
            "version": _effgen_version(),
            "hostname": socket.gethostname(),
            "server_url": server_url,
            "server_reachable": reachable,
            "server_detail": traffic.get("unavailable_reason"),
            "interval_s": interval,
        },
        "activity": _collect_activity(limit),
        "traffic": traffic,
        "by_model": by_model,
        "spend": _collect_spend(),
        "gpu": gpu if gpu is not None else _collect_gpu(),
    }


def _effgen_version() -> str:
    try:
        from effgen import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - the version is cosmetic here
        return "unknown"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _preview(text: Any, width: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= width else value[: width - 1] + "…"


def _fmt_usd(value: Any, *, places: int = 6) -> str:
    """Format a dollar amount; ``None`` renders as an em dash, never ``$0``."""
    if value is None:
        return "—"
    try:
        return f"${float(value):.{places}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_seconds(value: Any) -> str:
    if value is None:
        return "—"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _fmt_pct(value: Any, *, places: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{places}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_clock(ts: Any) -> str:
    """Render a run timestamp as a local wall-clock time."""
    if not ts:
        return "—"
    try:
        return datetime.fromisoformat(str(ts)).strftime("%H:%M:%S")
    except ValueError:
        return str(ts)[:8]


_STATUS_STYLES = {
    "ok": "effgen.success",
    "error": "effgen.error",
    "failed": "effgen.error",
}


def _status_style(status: Any) -> str:
    return _STATUS_STYLES.get(str(status or "").lower(), "effgen.muted")


def _health_style(*, error_rate: float | None, reachable: bool) -> str:
    if not reachable:
        return "effgen.muted"
    if error_rate is None:
        return "effgen.success"
    if error_rate >= 0.1:
        return "effgen.error"
    if error_rate > 0:
        return "effgen.warning"
    return "effgen.success"


# ---------------------------------------------------------------------------
# Static snapshot rendering (non-TTY, --once, --no-animation)
# ---------------------------------------------------------------------------


def render_snapshot(snapshot: dict[str, Any], *, console: Any = None) -> None:
    """Print one snapshot as a sequence of labelled tables.

    Uses the shared table renderer, so an interactive terminal gets Rich tables
    and a pipe gets aligned plain text with no color or box drawing.
    """
    header = snapshot.get("header") or {}
    reach = (
        f"server {header.get('server_url')} reachable"
        if header.get("server_reachable")
        else f"server {header.get('server_url')} unavailable"
    )
    lines = [
        f"effGen {header.get('version')} — {header.get('hostname')} — {snapshot.get('ts')}",
        reach,
    ]
    for line in lines:
        _emit(console, line)
    _emit(console, "")

    _render_activity_table(snapshot.get("activity") or {}, console)
    _render_traffic_table(snapshot.get("traffic") or {}, console)
    _render_by_model_table(snapshot.get("by_model") or {}, console)
    _render_spend_table(snapshot.get("spend") or {}, console)
    _render_gpu_table(snapshot.get("gpu") or {}, console)


def _emit(console: Any, text: str) -> None:
    if console_is_interactive(console):
        console.print(text, highlight=False)
    else:
        print(text)


def _unavailable(console: Any, title: str, panel: dict[str, Any]) -> None:
    reason = panel.get("unavailable_reason") or "no data source"
    _emit(console, f"{title} — unavailable: {reason}")
    _emit(console, "")


def _render_activity(panel: dict[str, Any]) -> tuple[list[str], list[list[str]], str]:
    columns = ["Time", "Status", "Model", "Duration", "Cost", "Task"]
    rows = [
        [
            _fmt_clock(run.get("ts")),
            str(run.get("status") or "—"),
            str(run.get("model") or "—"),
            _fmt_seconds(run.get("duration_s")),
            _fmt_usd(run.get("cost_usd")),
            run.get("task") or "—",
        ]
        for run in panel.get("runs") or []
    ]
    return columns, rows, f"Activity — {panel.get('scope', SCOPE_ACTIVITY)}"


def _render_activity_table(panel: dict[str, Any], console: Any) -> None:
    if not panel.get("available"):
        _unavailable(console, "Activity", panel)
        return
    columns, rows, title = _render_activity(panel)
    if not rows:
        _emit(console, f"{title}\n  no runs recorded yet")
        _emit(console, "")
        return
    render_table(columns=columns, rows=rows, console=console, title=title)
    _emit(console, "")


def _render_traffic_table(panel: dict[str, Any], console: Any) -> None:
    title = f"Traffic — {panel.get('scope', SCOPE_TRAFFIC)}"
    if not panel.get("available"):
        _unavailable(console, "Traffic", panel)
        return
    status_counts = panel.get("by_status") or {}
    status_text = ", ".join(f"{code}×{count}" for code, count in sorted(status_counts.items()))
    rows = [
        ["Generation requests", str(panel.get("requests", 0))],
        ["Errors", f"{panel.get('errors', 0)} ({_fmt_pct(100 * (panel.get('error_rate') or 0))})"],
        ["Latency p50 / p95 / p99", " / ".join(
            _fmt_seconds(panel.get(key))
            for key in ("p50_latency_s", "p95_latency_s", "p99_latency_s")
        )],
        ["Tokens", str(panel.get("tokens", 0))],
        # Counted over every route, so this total is expected to exceed the
        # generation-request count above rather than match it.
        ["HTTP responses (all routes)", status_text or "—"],
    ]
    render_table(columns=["Metric", "Value"], rows=rows, console=console, title=title)
    _emit(console, "")


def _render_by_model_table(panel: dict[str, Any], console: Any) -> None:
    if not panel.get("available"):
        return
    rows = [
        [
            str(row.get("model") or "—"),
            str(row.get("provider") or "—"),
            str(row.get("calls") or 0),
            str(row.get("errors") or 0),
            _fmt_pct(100 * float(row.get("error_rate") or 0.0)),
            _fmt_seconds(row.get("p95_latency_s")),
            _fmt_usd(row.get("cost_usd")),
        ]
        for row in panel.get("rows") or []
    ]
    title = f"Per-model — {panel.get('scope', SCOPE_TRAFFIC)}"
    if not rows:
        _emit(console, f"{title}\n  no model traffic yet")
        _emit(console, "")
        return
    render_table(
        columns=["Model", "Provider", "Calls", "Errors", "Error rate", "p95", "Cost"],
        rows=rows,
        console=console,
        title=title,
    )
    _emit(console, "")


def _render_spend_table(panel: dict[str, Any], console: Any) -> None:
    title = f"Spend — {panel.get('scope', SCOPE_SPEND)}"
    if not panel.get("available"):
        _unavailable(console, "Spend", panel)
        return
    budget = panel.get("daily_budget_usd")
    rows = [
        ["Total", _fmt_usd(panel.get("total_cost_usd"))],
        ["Requests", str(panel.get("requests", 0))],
        [
            "Daily budget",
            f"{_fmt_usd(budget, places=2)} ({_fmt_pct(panel.get('budget_used_pct'))} used)"
            if budget
            else "not configured (effgen cost set-budget AMOUNT)",
        ],
        ["Burn rate", f"{_fmt_usd(panel.get('burn_rate_usd_per_hour'))}/h (last hour)"],
    ]
    unpriced = panel.get("unpriced_models") or []
    if unpriced:
        rows.append(["Unpriced", f"{len(unpriced)} model(s) excluded from the total"])
    render_table(columns=["Metric", "Value"], rows=rows, console=console, title=title)
    _emit(console, "")


def _render_gpu_table(panel: dict[str, Any], console: Any) -> None:
    title = f"GPU — {panel.get('scope', SCOPE_GPU)}"
    if not panel.get("available"):
        _unavailable(console, "GPU", panel)
        return
    rows = [
        [
            str(device.get("index")),
            str(device.get("name") or "—"),
            f"{device.get('used_gb')} / {device.get('total_gb')} GB",
            _fmt_pct(device.get("memory_used_pct"), places=0),
            _fmt_pct(device.get("utilization_pct"), places=0),
        ]
        for device in panel.get("devices") or []
    ]
    render_table(
        columns=["GPU", "Name", "Memory", "Mem %", "Util"],
        rows=rows,
        console=console,
        title=title,
    )
    _emit(console, "")


# ---------------------------------------------------------------------------
# Live rendering
# ---------------------------------------------------------------------------


def _build_live_view(snapshot: dict[str, Any]) -> Any:
    """Assemble the full-screen layout for one snapshot."""
    from rich.console import Group
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    def _table(
        columns: list[str],
        rows: list[list[Any]],
        *,
        justify: list[str] | None = None,
        widths: list[int | None] | None = None,
        ratios: list[int | None] | None = None,
        expand: bool = True,
    ) -> Any:
        """A borderless table sized for a panel.

        ``widths`` pins the narrow, fixed-shape columns (a clock, a status, a
        cost) so they survive a narrow terminal; the remaining columns share the
        leftover width by ``ratios`` and ellipsize rather than wrap.
        """
        table = Table(box=None, expand=expand, pad_edge=False, show_edge=False, padding=(0, 1, 0, 0))
        for index, column in enumerate(columns):
            table.add_column(
                column,
                justify=(justify[index] if justify else "left"),
                style="effgen.value",
                header_style="effgen.muted",
                no_wrap=True,
                overflow="ellipsis",
                width=(widths[index] if widths else None),
                ratio=(ratios[index] if ratios else None),
            )
        for row in rows:
            table.add_row(*row)
        return table

    header = snapshot.get("header") or {}
    traffic = snapshot.get("traffic") or {}
    reachable = bool(header.get("server_reachable"))
    health = _health_style(
        error_rate=traffic.get("error_rate") if reachable else None, reachable=reachable
    )

    identity = Text()
    identity.append(f"effGen {header.get('version')} ", style="effgen.title")
    identity.append(f"· {header.get('hostname')} · ", style="effgen.muted")
    identity.append(
        f"server {header.get('server_url')} " + ("up" if reachable else "unreachable"),
        style=health,
    )
    interval = header.get("interval_s")
    clock = Text(
        f"{snapshot.get('ts')}"
        + (f" · every {interval:g}s" if interval else "")
        + " · q quits",
        style="effgen.muted",
    )
    title = Table.grid(expand=True)
    title.add_column(justify="left", overflow="ellipsis", no_wrap=True)
    title.add_column(justify="right", overflow="ellipsis", no_wrap=True)
    title.add_row(identity, clock)

    # Activity
    activity = snapshot.get("activity") or {}
    if activity.get("available"):
        runs = activity.get("runs") or []
        activity_body: Any = (
            _table(
                ["Time", "Status", "Model", "Dur", "Cost", "Task"],
                [
                    [
                        _fmt_clock(run.get("ts")),
                        Text(str(run.get("status") or "—"), style=_status_style(run.get("status"))),
                        str(run.get("model") or "—"),
                        _fmt_seconds(run.get("duration_s")),
                        _fmt_usd(run.get("cost_usd")),
                        run.get("task") or "—",
                    ]
                    for run in runs
                ],
                justify=["left", "left", "left", "right", "right", "left"],
                widths=[8, 6, None, 7, 10, None],
                ratios=[None, None, 2, None, None, 3],
            )
            if runs
            else Text("no runs recorded yet", style="effgen.muted")
        )
    else:
        activity_body = Text(str(activity.get("unavailable_reason") or ""), style="effgen.warning")

    # Traffic
    if traffic.get("available"):
        status_counts = traffic.get("by_status") or {}
        traffic_body: Any = Group(
            _table(
                ["", ""],
                [
                    ["Generations", str(traffic.get("requests", 0))],
                    [
                        "Errors",
                        Text(
                            f"{traffic.get('errors', 0)} "
                            f"({_fmt_pct(100 * (traffic.get('error_rate') or 0))})",
                            style=health,
                        ),
                    ],
                    [
                        "p50 / p95 / p99",
                        " / ".join(
                            _fmt_seconds(traffic.get(key))
                            for key in ("p50_latency_s", "p95_latency_s", "p99_latency_s")
                        ),
                    ],
                    ["Tokens", str(traffic.get("tokens", 0))],
                    # Every route, not just generations — so this exceeds the
                    # generation count above rather than matching it.
                    [
                        "HTTP all routes",
                        ", ".join(f"{c}×{n}" for c, n in sorted(status_counts.items())) or "—",
                    ],
                ],
                widths=[16, None],
            )
        )
    else:
        traffic_body = Text(str(traffic.get("unavailable_reason") or ""), style="effgen.warning")

    # Per-model
    by_model = snapshot.get("by_model") or {}
    model_rows = by_model.get("rows") or []
    if not by_model.get("available"):
        model_body: Any = Text("server unavailable — see Traffic", style="effgen.muted")
    elif model_rows:
        model_body = _table(
            ["Model", "Calls", "Err", "p95", "Cost"],
            [
                [
                    str(row.get("model") or "—"),
                    str(row.get("calls") or 0),
                    str(row.get("errors") or 0),
                    _fmt_seconds(row.get("p95_latency_s")),
                    _fmt_usd(row.get("cost_usd")),
                ]
                for row in model_rows
            ],
            justify=["left", "right", "right", "right", "right"],
            widths=[None, 5, 4, 7, 10],
        )
    else:
        model_body = Text("no model traffic yet", style="effgen.muted")

    # Spend
    spend = snapshot.get("spend") or {}
    if spend.get("available"):
        budget = spend.get("daily_budget_usd")
        used_pct = spend.get("budget_used_pct")
        spend_rows = [
            ["Total (24h)", _fmt_usd(spend.get("total_cost_usd"))],
            ["Burn", f"{_fmt_usd(spend.get('burn_rate_usd_per_hour'))}/h"],
        ]
        if budget:
            bar_width = 16
            filled = max(0, min(bar_width, int(bar_width * (used_pct or 0) / 100.0)))
            bar_style = (
                "effgen.error"
                if (used_pct or 0) >= 100
                else "effgen.warning"
                if (used_pct or 0) >= 80
                else "effgen.success"
            )
            bar = Text("█" * filled, style=bar_style)
            bar.append("░" * (bar_width - filled), style="effgen.muted")
            bar.append(f" {_fmt_pct(used_pct)} of {_fmt_usd(budget, places=2)}")
            spend_rows.append(["Budget", bar])
        else:
            spend_rows.append(["Budget", Text("not configured", style="effgen.muted")])
        unpriced = spend.get("unpriced_models") or []
        if unpriced:
            spend_rows.append(
                ["Unpriced", Text(f"{len(unpriced)} model(s) not in total", style="effgen.muted")]
            )
        spend_body: Any = _table(["", ""], spend_rows, widths=[11, None])
    else:
        spend_body = Text(str(spend.get("unavailable_reason") or ""), style="effgen.warning")

    # GPU
    gpu = snapshot.get("gpu") or {}
    if gpu.get("available"):
        gpu_body: Any = _table(
            ["GPU", "Memory", "Mem%", "Util"],
            [
                [
                    str(device.get("index")),
                    f"{device.get('used_gb'):.1f}/{device.get('total_gb'):.0f}G",
                    _fmt_pct(device.get("memory_used_pct"), places=0),
                    _fmt_pct(device.get("utilization_pct"), places=0),
                ]
                for device in gpu.get("devices") or []
            ],
            justify=["right", "right", "right", "right"],
            widths=[4, 11, 6, 6],
            expand=False,
        )
    else:
        gpu_body = Text(str(gpu.get("unavailable_reason") or ""), style="effgen.muted")

    def _panel(body: Any, heading: str, scope: str) -> Any:
        return Panel(
            body,
            title=heading,
            subtitle=scope,
            subtitle_align="right",
            border_style="effgen.muted",
            padding=(0, 1),
        )

    layout = Layout()
    layout.split_column(
        Layout(Panel(title, border_style="effgen.muted", padding=(0, 1)), name="head", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
    layout["left"].split_column(
        Layout(
            _panel(activity_body, "Activity", activity.get("scope", SCOPE_ACTIVITY)),
            name="activity",
            ratio=3,
        ),
        Layout(
            _panel(model_body, "Per-model", by_model.get("scope", SCOPE_TRAFFIC)),
            name="models",
            ratio=2,
        ),
    )
    # The GPU panel is sized for the device count so a multi-GPU host is not
    # clipped to its first few cards.
    gpu_rows = len(gpu.get("devices") or [])
    layout["right"].split_column(
        Layout(
            _panel(traffic_body, "Traffic", traffic.get("scope", SCOPE_TRAFFIC)),
            name="traffic",
            size=9,
        ),
        Layout(
            _panel(gpu_body, "GPU", gpu.get("scope", SCOPE_GPU)),
            name="gpu",
            size=min(gpu_rows + 4, 14) if gpu_rows else 4,
        ),
        Layout(_panel(spend_body, "Spend", spend.get("scope", SCOPE_SPEND)), name="spend"),
    )
    return layout


class _QuitKeyReader:
    """Read single keypresses without blocking, restoring the terminal on exit.

    Falls back to a no-op (Ctrl-C remains the way to quit) on a platform or a
    stream where raw terminal input is unavailable.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._saved: Any = None

    def __enter__(self) -> _QuitKeyReader:
        try:
            import termios
            import tty

            if not sys.stdin.isatty():
                return self
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:  # noqa: BLE001 - no raw input available; Ctrl-C still quits
            self._fd = None
            self._saved = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fd is not None and self._saved is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:  # noqa: BLE001 - the terminal is being torn down anyway
                pass

    def quit_requested(self, timeout: float) -> bool:
        """Wait up to *timeout* seconds for a quit key (``q``, ``Q``, or Escape)."""
        if self._fd is None:
            time.sleep(max(0.0, timeout))
            return False
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                ready, _, _ = select.select([sys.stdin], [], [], remaining)
            except (OSError, ValueError):
                time.sleep(max(0.0, remaining))
                return False
            if not ready:
                return False
            char = sys.stdin.read(1)
            if char in ("q", "Q", "\x1b", "\x03"):
                return True


def run_live(
    *,
    console: Any,
    url: str | None,
    port: int | None,
    api_key: str | None,
    limit: int,
    interval: float,
    count: int | None = None,
) -> int:
    """Refresh the full-screen view in place until a quit key, Ctrl-C, or *count* refreshes."""
    from rich.live import Live

    gpu_panel: dict[str, Any] | None = None
    gpu_sampled_at = 0.0
    refreshes = 0
    try:
        with _QuitKeyReader() as keys, Live(
            console=console, screen=True, auto_refresh=False, transient=False
        ) as live:
            while True:
                now = time.monotonic()
                if gpu_panel is None or (now - gpu_sampled_at) >= GPU_SAMPLE_INTERVAL_S:
                    gpu_panel = _collect_gpu()
                    gpu_sampled_at = now
                snapshot = collect_snapshot(
                    url=url,
                    port=port,
                    api_key=api_key,
                    limit=limit,
                    interval=interval,
                    gpu=gpu_panel,
                )
                live.update(_build_live_view(snapshot), refresh=True)
                refreshes += 1
                if count is not None and refreshes >= count:
                    return 0
                if keys.quit_requested(interval):
                    return 0
    except KeyboardInterrupt:
        return 0


# ---------------------------------------------------------------------------
# Command entry point
# ---------------------------------------------------------------------------


def run_monitor_command(args: Any) -> int:
    """Entry point for ``effgen top`` / ``effgen monitor``."""
    from effgen.cli import progress as _progress

    interval = float(getattr(args, "interval", DEFAULT_INTERVAL) or DEFAULT_INTERVAL)
    if not MIN_INTERVAL <= interval <= MAX_INTERVAL:
        print(
            f"--interval must be between {MIN_INTERVAL:g} and {MAX_INTERVAL:g} seconds "
            f"(got {interval:g}).",
            file=sys.stderr,
        )
        return 1
    limit = max(1, int(getattr(args, "limit", DEFAULT_ACTIVITY_LIMIT) or DEFAULT_ACTIVITY_LIMIT))
    url = getattr(args, "url", None)
    port = getattr(args, "port", None)
    api_key = getattr(args, "api_key", None)
    json_mode = bool(getattr(args, "output_json", False))
    once = bool(getattr(args, "once", False))

    if json_mode:
        snapshot = collect_snapshot(
            url=url, port=port, api_key=api_key, limit=limit, interval=None
        )
        print(json.dumps(snapshot, indent=2, ensure_ascii=json_ensure_ascii()))
        return 0

    console = get_console(theme_name=getattr(args, "theme", None))
    live_capable = _progress.animation_enabled(
        quiet=getattr(args, "quiet", False),
        no_animation=getattr(args, "no_animation", False),
    )
    if once or not live_capable or not color_enabled() or not console_is_interactive(console):
        snapshot = collect_snapshot(
            url=url, port=port, api_key=api_key, limit=limit, interval=None
        )
        render_snapshot(snapshot, console=console)
        return 0

    return run_live(
        console=console,
        url=url,
        port=port,
        api_key=api_key,
        limit=limit,
        interval=interval,
        count=getattr(args, "count", None),
    )
