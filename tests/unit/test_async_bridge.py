"""Unit tests for ``effgen.utils.run_coroutine_sync``.

The bridge must drive a coroutine to completion whether or not an event loop is
already running in the calling thread — and must never silently skip the work.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from effgen.utils import run_coroutine_sync
from effgen.utils.async_bridge import run_coroutine_sync as direct_import


def test_exported_from_utils_package():
    """The helper is importable from the package facade and the submodule."""
    assert run_coroutine_sync is direct_import


def test_runs_with_no_running_loop():
    async def work():
        await asyncio.sleep(0)
        return 41 + 1

    assert run_coroutine_sync(work()) == 42


def test_runs_from_inside_a_running_loop():
    """When a loop is already running in the thread, the coroutine still runs
    (on a worker thread) and the result is returned to the caller."""

    async def outer():
        async def inner():
            await asyncio.sleep(0.01)
            return "done"

        # Calling run_coroutine_sync from inside a running loop must not raise
        # "asyncio.run() cannot be called from a running event loop" and must
        # actually execute inner().
        return run_coroutine_sync(inner())

    assert asyncio.run(outer()) == "done"


def test_actually_executes_under_running_loop_not_skipped():
    """Regression: the work is performed, not no-op'd, under a loop."""
    ran = threading.Event()

    async def side_effect():
        await asyncio.sleep(0)
        ran.set()
        return True

    async def outer():
        return run_coroutine_sync(side_effect())

    result = asyncio.run(outer())
    assert result is True
    assert ran.is_set(), "coroutine body must have executed"


def test_propagates_exceptions_no_loop():
    async def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_coroutine_sync(boom())


def test_propagates_runtimeerror_from_coroutine_no_loop():
    """A RuntimeError raised *inside* the coroutine must surface as-is and not
    be mistaken for "loop already running" (which would try to re-run an
    already-consumed coroutine)."""

    async def boom():
        raise RuntimeError("inner runtime error")

    with pytest.raises(RuntimeError, match="inner runtime error"):
        run_coroutine_sync(boom())


def test_propagates_exceptions_under_running_loop():
    async def boom():
        raise ValueError("nested kaboom")

    async def outer():
        return run_coroutine_sync(boom())

    with pytest.raises(ValueError, match="nested kaboom"):
        asyncio.run(outer())


def test_blocks_until_complete_under_running_loop():
    """The caller blocks for the duration of the bridged coroutine."""

    async def slow():
        await asyncio.sleep(0.2)
        return "ok"

    async def outer():
        t0 = time.monotonic()
        res = run_coroutine_sync(slow())
        return res, time.monotonic() - t0

    res, elapsed = asyncio.run(outer())
    assert res == "ok"
    assert elapsed >= 0.18, f"should have blocked ~0.2s, got {elapsed:.3f}s"
