"""In-memory ring buffer for recent agent run records.

The dashboard reads from this buffer to populate the "Recent Agent Runs"
panel.  Records are appended by :func:`record_run` whenever an agent
completes a run; the buffer is capped at ``MAX_RUNS`` to bound memory use.

The buffer is process-local and not persisted across restarts.  For
persistent history use the audit log at ``~/.effgen/audit/``.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

MAX_RUNS = 200

_lock: threading.Lock = threading.Lock()
_runs: deque[dict[str, Any]] = deque(maxlen=MAX_RUNS)


def record_run(
    *,
    model: str = "unknown",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    duration_s: float | None = None,
    cost_usd: float | None = None,
    error: str | None = None,
) -> None:
    """Append a run record to the in-memory buffer.

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
    """
    record: dict[str, Any] = {
        "ts": time.strftime("%H:%M:%S", time.gmtime()),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_s": duration_s,
        "cost_usd": cost_usd,
        "error": error,
    }
    with _lock:
        _runs.append(record)


def get_recent_runs(*, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most-recent *limit* run records (newest first).

    Returns an empty list when no runs have been recorded.
    """
    with _lock:
        runs = list(_runs)
    return list(reversed(runs))[:limit]


def clear() -> None:
    """Clear all buffered run records (primarily for testing)."""
    with _lock:
        _runs.clear()
