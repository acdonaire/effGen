"""Per-principal daily cost tracking for RBAC cost caps.

Each authenticated principal accrues spend against the ``max_cost_per_day``
of its effective policy. When the accrued spend for the current UTC day meets
or exceeds the cap, further charged requests are rejected with
:class:`BudgetExceeded` (surfaced as HTTP 429 by the API server).

The tracker is process-local and in-memory by default. A daily snapshot is
optionally persisted to ``~/.effgen/budget/<YYYY-MM-DD>.json`` so that spend
survives a restart within the same day; set ``EFFGEN_BUDGET_PERSIST=0`` to
disable persistence.

This is intentionally simple — a single-process counter. For multi-replica
deployments back it with a shared store (Redis); the public API
(:func:`charge`, :func:`get_spend`, :func:`reset`) is the integration point.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    """Raised when a principal exceeds its daily cost cap.

    Surfaced by the API server as HTTP 429 (Too Many Requests).
    """

    def __init__(self, reason: str, status_code: int = 429):
        super().__init__(reason)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# In-memory daily ledger
# ---------------------------------------------------------------------------

_lock = threading.Lock()
# {day -> {principal -> spend_usd}}
_ledger: dict[str, dict[str, float]] = {}
_BUDGET_DIR: Path | None = None


def _today() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _persist_enabled() -> bool:
    return os.getenv("EFFGEN_BUDGET_PERSIST", "1").strip() != "0"


def _budget_dir() -> Path:
    global _BUDGET_DIR
    if _BUDGET_DIR is not None:
        return _BUDGET_DIR
    base = Path(os.getenv("EFFGEN_BUDGET_DIR", Path.home() / ".effgen" / "budget"))
    base.mkdir(parents=True, exist_ok=True)
    _BUDGET_DIR = base
    return _BUDGET_DIR


def _day_file(day: str) -> Path:
    return _budget_dir() / f"{day}.json"


def _load_day(day: str) -> dict[str, float]:
    """Return the ledger for *day*, loading from disk on first access."""
    if day in _ledger:
        return _ledger[day]
    data: dict[str, float] = {}
    if _persist_enabled():
        path = _day_file(day)
        if path.exists():
            try:
                data = {k: float(v) for k, v in json.loads(path.read_text()).items()}
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load budget ledger %s: %s", path, exc)
    _ledger[day] = data
    return data


def _save_day(day: str) -> None:
    if not _persist_enabled():
        return
    try:
        _day_file(day).write_text(json.dumps(_ledger.get(day, {})))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist budget ledger for %s: %s", day, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_spend(principal: str, *, day: str | None = None) -> float:
    """Return *principal*'s accrued spend (USD) for *day* (default today)."""
    day = day or _today()
    with _lock:
        return _load_day(day).get(principal, 0.0)


def check_budget(principal: str, cap: float, *, day: str | None = None) -> None:
    """Raise :class:`BudgetExceeded` if *principal* has already met *cap*.

    A *cap* of ``0.0`` means unlimited and never raises.
    """
    if cap <= 0.0:
        return
    spent = get_spend(principal, day=day)
    if spent >= cap:
        raise BudgetExceeded(
            f"BudgetExceeded: principal {principal!r} has spent "
            f"${spent:.4f} of its ${cap:.2f}/day cap"
        )


def charge(
    principal: str,
    amount: float,
    *,
    cap: float = 0.0,
    day: str | None = None,
) -> float:
    """Add *amount* USD to *principal*'s spend for *day* and return new total.

    If *cap* is positive and the principal has **already** met or exceeded the
    cap before this charge, :class:`BudgetExceeded` is raised and nothing is
    charged. The charge itself may push spend past the cap (the next call then
    rejects), which matches the spec: a cap of $0.01 admits the first call and
    rejects subsequent ones.
    """
    day = day or _today()
    with _lock:
        ledger = _load_day(day)
        current = ledger.get(principal, 0.0)
        if cap > 0.0 and current >= cap:
            raise BudgetExceeded(
                f"BudgetExceeded: principal {principal!r} has spent "
                f"${current:.4f} of its ${cap:.2f}/day cap"
            )
        ledger[principal] = current + max(0.0, amount)
        _save_day(day)
        return ledger[principal]


def reset(principal: str | None = None, *, day: str | None = None) -> None:
    """Reset spend. With no args, clears the entire in-memory ledger.

    Primarily for tests and admin tooling.
    """
    global _ledger, _BUDGET_DIR
    with _lock:
        if principal is None and day is None:
            _ledger = {}
            return
        d = day or _today()
        ledger = _load_day(d)
        if principal is None:
            ledger.clear()
        else:
            ledger.pop(principal, None)
        _save_day(d)
