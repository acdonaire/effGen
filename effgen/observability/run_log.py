"""Agent run history — an in-memory ring plus an append-only file store.

Every completed agent run is appended to a process-local ring buffer (the hot
read path for the dashboard) and, unless history is disabled, to a daily
JSONL file under the effGen state directory::

    ~/.effgen/runs/2026-07-18.jsonl

The files are the durable source of truth: runs recorded by the CLI, by a
script, and by the server all land in the same place and survive a restart.
:func:`get_recent_runs` reads the ring, so live views stay scoped to the
current process; :func:`read_runs` and :func:`get_run` read the durable
history, merging the ring with the files and de-duplicating on ``run_id``.

Locations honor, in order, ``EFFGEN_RUN_HISTORY_DIR``, ``EFFGEN_HOME`` (as
``$EFFGEN_HOME/runs``), then ``~/.effgen/runs``. Set ``EFFGEN_RUN_HISTORY=0``
to keep history in memory only. Files older than
``EFFGEN_RUN_HISTORY_MAX_DAYS`` (default 30) are removed when the first record
of a process is written.

Writes are best-effort: a history write that fails is logged at debug level and
never propagates into the caller's run.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_RUNS = 200

#: Longest task/output preview persisted with a run record.
PREVIEW_CHARS = 500

DEFAULT_MAX_DAYS = 30

_lock: threading.Lock = threading.Lock()
_runs: deque[dict[str, Any]] = deque(maxlen=MAX_RUNS)
_pruned = False


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
def history_dir() -> Path:
    """Return the directory holding daily run-history files."""
    explicit = os.environ.get("EFFGEN_RUN_HISTORY_DIR")
    if explicit:
        return Path(os.path.expanduser(explicit)).absolute()
    home = os.environ.get("EFFGEN_HOME")
    if home:
        return Path(os.path.expanduser(home)).absolute() / "runs"
    return Path(os.path.expanduser("~/.effgen/runs"))


def history_enabled() -> bool:
    """Whether run records are written to disk."""
    return os.environ.get("EFFGEN_RUN_HISTORY", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _now_iso() -> str:
    """Local time as ISO-8601 with an explicit UTC offset, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _preview(text: Any, limit: int = PREVIEW_CHARS) -> str | None:
    if text is None:
        return None
    s = str(text)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def record_run(
    *,
    model: str = "unknown",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    duration_s: float | None = None,
    cost_usd: float | None = None,
    error: str | None = None,
    run_id: str | None = None,
    task: str | None = None,
    output: str | None = None,
    provider: str | None = None,
    session_id: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    """Append a run record to the ring buffer and the daily history file.

    Parameters
    ----------
    model:
        Model identifier used for the run.
    input_tokens:
        Number of input (prompt) tokens consumed, if known.
    output_tokens:
        Number of output (completion) tokens generated, if known.
    duration_s:
        Wall-clock duration of the run in seconds.
    cost_usd:
        Estimated cost in USD, if known.
    error:
        Short error message if the run failed; ``None`` on success.
    run_id:
        Identifier shared with the run's trace spans, so a record can be
        opened and joined to its trace.
    task:
        The task text; stored truncated to ``PREVIEW_CHARS``.
    output:
        The answer text; stored truncated to ``PREVIEW_CHARS``.
    provider:
        Provider that served the model, if known.
    session_id:
        Session the run belongs to, when it was run with one.
    agent:
        Name of the agent that performed the run.

    Returns the record that was stored.
    """
    record: dict[str, Any] = {
        "ts": _now_iso(),
        "run_id": run_id,
        "status": "error" if error else "ok",
        "model": model,
        "provider": provider,
        "agent": agent,
        "session_id": session_id,
        "task": _preview(task),
        "output": _preview(output),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_s": duration_s,
        "cost_usd": cost_usd,
        "error": error,
    }
    with _lock:
        _runs.append(record)
    _append_to_file(record)
    return record


def _append_to_file(record: dict[str, Any]) -> None:
    """Append one record to today's history file; failures are non-fatal."""
    if not history_enabled():
        return
    try:
        directory = history_dir()
        directory.mkdir(parents=True, exist_ok=True)
        _prune_once(directory)
        day = record["ts"][:10]
        line = json.dumps(record, default=str, ensure_ascii=False)
        with open(directory / f"{day}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001 - history must never fail a user's run
        logger.debug("Run history write failed", exc_info=True)


def _prune_once(directory: Path) -> None:
    """Drop history files older than the retention window (once per process)."""
    global _pruned
    if _pruned:
        return
    _pruned = True
    try:
        max_days = int(os.environ.get("EFFGEN_RUN_HISTORY_MAX_DAYS", DEFAULT_MAX_DAYS))
    except ValueError:
        max_days = DEFAULT_MAX_DAYS
    if max_days <= 0:
        return
    cutoff = (datetime.now().date() - timedelta(days=max_days)).isoformat()
    try:
        for path in directory.glob("*.jsonl"):
            if path.stem < cutoff:
                path.unlink()
    except OSError:
        logger.debug("Run history prune failed", exc_info=True)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def _read_files(*, limit: int) -> list[dict[str, Any]]:
    """Read up to *limit* records from the newest history files (newest first)."""
    directory = history_dir()
    if not directory.is_dir():
        return []
    out: list[dict[str, Any]] = []
    try:
        files = sorted(directory.glob("*.jsonl"), key=lambda p: p.stem, reverse=True)
    except OSError:
        return []
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            logger.debug("Unreadable run history file %s", path)
            continue
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                # One damaged line never hides the rest of the history.
                logger.debug("Skipping unparseable run history line in %s", path)
        if len(out) >= limit:
            break
    return out


def _sort_key(record: dict[str, Any]) -> str:
    return str(record.get("ts") or "")


def _identity(record: dict[str, Any]) -> str:
    """A key that matches the same run whether it came from the ring or a file.

    ``run_id`` identifies a run on its own. Without one, the stored fields are
    compared instead, so a run present in both the ring and today's file is
    still reported once rather than twice.
    """
    run_id = record.get("run_id")
    if run_id:
        return f"id:{run_id}"
    return "fields:" + "|".join(
        str(record.get(field) or "")
        for field in ("ts", "model", "status", "session_id", "task", "error")
    )


def read_runs(
    *,
    limit: int = 50,
    status: str | None = None,
    model: str | None = None,
    search: str | None = None,
    since: str | None = None,
    until: str | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return stored run records, newest first, after applying filters.

    Merges the in-memory ring with the history files and de-duplicates on
    ``run_id``. ``since``/``until`` accept a date (``2026-07-18``) or any
    ISO-8601 prefix and are compared against the record timestamp. ``search``
    matches the task, output, model, run id, session id, or error text.
    """
    with _lock:
        merged = list(_runs)
    merged.extend(_read_files(limit=max(limit * 4, 200)))

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in sorted(merged, key=_sort_key, reverse=True):
        key = _identity(record)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    needle = (search or "").lower()
    out: list[dict[str, Any]] = []
    for record in unique:
        if status and str(record.get("status") or "") != status:
            continue
        if model and model.lower() not in str(record.get("model") or "").lower():
            continue
        if session_id and str(record.get("session_id") or "") != session_id:
            continue
        ts = str(record.get("ts") or "")
        if since and ts < since:
            continue
        if until and ts[: len(until)] > until:
            continue
        if needle:
            haystack = " ".join(
                str(record.get(f) or "")
                for f in ("task", "output", "model", "run_id", "session_id", "error", "agent")
            ).lower()
            if needle not in haystack:
                continue
        out.append(record)
        if len(out) >= limit:
            break
    return out


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return the record with *run_id*, or ``None`` when it is not stored."""
    for record in read_runs(limit=10000, search=run_id):
        if str(record.get("run_id")) == run_id:
            return record
    return None


def get_recent_runs(*, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most-recent *limit* runs of this process (newest first).

    Reads the in-memory ring only, so live views stay scoped to current
    activity. Use :func:`read_runs` for the durable history, which spans
    processes and restarts.

    Returns an empty list when this process has recorded no runs.
    """
    with _lock:
        runs = list(_runs)
    return list(reversed(runs))[:limit]


def cleanup_runs(older_than_days: int = 30) -> int:
    """Delete history files older than *older_than_days*; returns the count."""
    directory = history_dir()
    if not directory.is_dir():
        return 0
    cutoff = (datetime.now().date() - timedelta(days=older_than_days)).isoformat()
    removed = 0
    for path in sorted(directory.glob("*.jsonl")):
        if path.stem < cutoff:
            try:
                path.unlink()
                removed += 1
            except OSError:
                logger.debug("Could not remove run history file %s", path)
    return removed


def clear() -> None:
    """Clear the in-memory ring (the history files are left untouched)."""
    global _pruned
    with _lock:
        _runs.clear()
    _pruned = False
