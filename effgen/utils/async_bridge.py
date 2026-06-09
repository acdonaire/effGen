"""Synchronous/asynchronous bridging helpers.

A single, correct way to drive a coroutine to completion from synchronous code,
regardless of whether an event loop is already running in the calling thread.

Why this exists
---------------
Synchronous entry points (e.g. ``Model.generate``) sometimes need to await an
async primitive such as a rate-limit coordinator.  The naive pattern

    loop = asyncio.get_event_loop()
    if loop.is_running():
        ...  # skip
    else:
        loop.run_until_complete(coro)

is wrong in two ways: ``get_event_loop()`` is deprecated when no loop is running,
and *skipping* the work under a running loop (Jupyter, FastAPI, any async caller)
silently disables it — exactly where it matters most.  :func:`run_coroutine_sync`
instead always runs the coroutine to completion: directly when the calling thread
has no running loop, or on a dedicated worker thread (blocking the caller) when a
loop is already running.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Awaitable
from typing import TypeVar

__all__ = ["run_coroutine_sync"]

T = TypeVar("T")


def run_coroutine_sync(coro: Awaitable[T], timeout: float = 120.0) -> T:
    """Run an awaitable from synchronous code and return its result.

    Behaviour:

    * **No running loop in the calling thread** — drive the coroutine directly
      with :func:`asyncio.run`.
    * **A loop is already running in the calling thread** (Jupyter, FastAPI, or
      any async caller that reached synchronous code) — run the coroutine on a
      dedicated worker thread with its own event loop and *block* the calling
      thread until it finishes.  The work is never silently skipped.

    Args:
        coro: The coroutine / awaitable to run.
        timeout: Maximum seconds to wait when bridging via a worker thread.

    Returns:
        The value produced by the coroutine.

    Raises:
        Any exception raised by the coroutine, plus
        :class:`concurrent.futures.TimeoutError` if the worker-thread path
        exceeds ``timeout``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running in this thread — safe to drive it directly. We do not
        # catch RuntimeError from ``asyncio.run`` here so that a RuntimeError
        # raised *inside* the coroutine is not mistaken for "loop already
        # running" and re-run on an already-consumed coroutine.
        return asyncio.run(coro)  # type: ignore[arg-type]

    # A loop is already running in this thread; we cannot use asyncio.run or
    # block the live loop. Hand the coroutine to a fresh loop on a worker thread
    # and wait synchronously for the result.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)  # type: ignore[arg-type]
        return future.result(timeout=timeout)
