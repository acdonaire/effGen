"""
Retry with jittered exponential backoff.

Usage::

    from effgen.reliability.retry import Retry, retryable, is_transient_error

    policy = Retry(
        max_attempts=4,
        base_delay=0.5,
        max_delay=30.0,
        jitter=True,
        retryable=is_transient_error,
    )

    @retryable(policy)
    def call_api():
        ...

    # Async version:
    @retryable(policy)
    async def call_api_async():
        ...

Each retry attempt emits an ``effgen.retry.attempt`` span *event* on the
currently active OTel span (no-op if tracing is not configured).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "Retry",
    "RetryExhausted",
    "retryable",
    "is_transient_error",
    "DEFAULT_RETRY",
]


# ---------------------------------------------------------------------------
# Retryable error classification
# ---------------------------------------------------------------------------


def is_transient_error(exc: BaseException) -> bool:
    """Return True for errors that are safe to retry.

    effGen's own typed provider errors (``effgen.models.errors``) carry an
    authoritative classification; we defer to it so this retry predicate can
    never drift from the core error taxonomy — an ``auth``/``not_found``/
    ``invalid``/``fatal`` error is *never* retried, even if it happens to carry
    a status code that the heuristics below might otherwise treat as transient.

    For everything else (raw SDK exceptions, stdlib errors) we fall back to:
    - Network-level errors (ConnectionError, OSError, TimeoutError)
    - HTTP 429 / 5xx responses (detected by status code attribute)
    - asyncio.TimeoutError
    - effGen's own TimeoutError
    """
    from effgen.reliability.timeouts import TimeoutError as EffGenTimeout

    # effGen typed provider errors → the Phase-0 taxonomy is the source of truth.
    if isinstance(exc, Exception) and type(exc).__module__ == "effgen.models.errors":
        from effgen.models.errors import classify_provider_error

        return classify_provider_error(exc).should_retry

    # Check for HTTP status codes on the exception
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status == 429 or (500 <= status < 600):
            return True

    # Check response attribute (httpx / requests style)
    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int):
            if resp_status == 429 or (500 <= resp_status < 600):
                return True

    # Network / transient errors
    transient_types = (
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        TimeoutError,
        asyncio.TimeoutError,
        EffGenTimeout,
        OSError,
    )
    if isinstance(exc, transient_types):
        return True

    # httpx-specific errors if available
    try:
        import httpx
        if isinstance(exc, httpx.NetworkError | httpx.TimeoutException | httpx.RemoteProtocolError):
            return True
    except ImportError:
        pass

    return False


def _get_retry_after(exc: BaseException) -> float | None:
    """Extract Retry-After seconds from a 429 exception, if available."""
    # Check for explicit retry_after attribute (our adapters set this)
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(retry_after, int | float) and retry_after > 0:
        return float(retry_after)

    # Try to parse from response headers
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass

    return None


# ---------------------------------------------------------------------------
# Retry dataclass
# ---------------------------------------------------------------------------


@dataclass
class Retry:
    """Retry policy with jittered exponential backoff.

    Attributes:
        max_attempts:  Total number of attempts (including the first).
                       ``1`` means no retries.
        base_delay:    Initial backoff delay in seconds.
        max_delay:     Cap on the computed backoff delay.
        jitter:        When True, add ``uniform(0, base_delay)`` to each delay.
        retryable:     Callable ``(exc) -> bool`` that decides whether an
                       exception is worth retrying.  Defaults to
                       :func:`is_transient_error`.
        honour_retry_after:  When True, sleep the Retry-After value from a 429
                             response instead of the computed backoff.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: bool = True
    retryable: Callable[[BaseException], bool] = field(
        default=is_transient_error
    )
    honour_retry_after: bool = True

    def compute_delay(self, attempt: int, exc: BaseException | None = None) -> float:
        """Compute the wait duration before the next attempt.

        Args:
            attempt:  0-based attempt index (0 = first retry).
            exc:      The exception that triggered this retry.  Used to
                      extract Retry-After header when present.

        Returns:
            Seconds to sleep before the next attempt.
        """
        # Honour Retry-After from 429 responses
        if self.honour_retry_after and exc is not None:
            ra = _get_retry_after(exc)
            if ra is not None:
                return min(ra, self.max_delay)

        # Exponential backoff: base * 2^attempt
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay += random.uniform(0.0, self.base_delay)
            delay = min(delay, self.max_delay)

        return delay


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class RetryExhausted(Exception):
    """Raised when all retry attempts have been consumed.

    Attributes:
        attempts:   Number of attempts made.
        last_error: The last exception that triggered a retry.
    """

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(
            f"All {attempts} retry attempts exhausted. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error


# ---------------------------------------------------------------------------
# OTel span event helper (non-blocking)
# ---------------------------------------------------------------------------


def _emit_retry_event(attempt: int, reason: str, delay_s: float) -> None:
    """Attach a retry event to the current OTel span (no-op if no span)."""
    try:
        from opentelemetry import trace

        from effgen.observability.spans import RetryAttrs, SpanName

        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(
                SpanName.RETRY_ATTEMPT,
                attributes={
                    RetryAttrs.ATTEMPT: attempt,
                    RetryAttrs.REASON: reason,
                    RetryAttrs.DELAY_S: delay_s,
                },
            )
    except Exception:
        pass  # telemetry is non-blocking


# ---------------------------------------------------------------------------
# retryable decorator
# ---------------------------------------------------------------------------


DEFAULT_RETRY = Retry()


def retryable(policy: Retry = DEFAULT_RETRY):
    """Decorator factory that wraps a sync or async function with *policy*.

    Usage::

        @retryable(Retry(max_attempts=4, base_delay=1.0))
        def call_api() -> str:
            ...

        @retryable(Retry(max_attempts=3))
        async def async_call() -> str:
            ...

    Raises:
        :class:`RetryExhausted`: When *max_attempts* is reached.
        The original exception is re-raised if it is not retryable.
    """

    def decorator(fn):
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: BaseException | None = None
                for attempt in range(policy.max_attempts):
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        if not policy.retryable(exc):
                            raise
                        if attempt + 1 >= policy.max_attempts:
                            break
                        reason = type(exc).__name__
                        delay = policy.compute_delay(attempt, exc)
                        log.warning(
                            "retry attempt %d/%d for %s after %.2fs "
                            "(reason: %s: %s)",
                            attempt + 1,
                            policy.max_attempts,
                            fn.__qualname__,
                            delay,
                            reason,
                            exc,
                        )
                        _emit_retry_event(attempt + 1, reason, delay)
                        await asyncio.sleep(delay)
                raise RetryExhausted(policy.max_attempts, last_exc)

            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: BaseException | None = None
                for attempt in range(policy.max_attempts):
                    try:
                        return fn(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        if not policy.retryable(exc):
                            raise
                        if attempt + 1 >= policy.max_attempts:
                            break
                        reason = type(exc).__name__
                        delay = policy.compute_delay(attempt, exc)
                        log.warning(
                            "retry attempt %d/%d for %s after %.2fs "
                            "(reason: %s: %s)",
                            attempt + 1,
                            policy.max_attempts,
                            fn.__qualname__,
                            delay,
                            reason,
                            exc,
                        )
                        _emit_retry_event(attempt + 1, reason, delay)
                        time.sleep(delay)
                raise RetryExhausted(policy.max_attempts, last_exc)

            return sync_wrapper

    return decorator
