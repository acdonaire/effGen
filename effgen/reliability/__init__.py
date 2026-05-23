"""
effGen Reliability Primitives.

Provides production-grade reliability patterns:
- :class:`~effgen.reliability.timeouts.TimeoutConfig` — explicit timeouts for every path
- :class:`~effgen.reliability.retry.Retry` / :func:`~effgen.reliability.retry.retryable` — jittered exponential backoff
- :class:`~effgen.reliability.circuit.CircuitBreaker` — closed → open → half-open state machine
- :class:`~effgen.reliability.bulkhead.Bulkhead` — concurrency cap + bounded queue

Quick start
-----------
    from effgen.reliability import (
        ReliabilityConfig,
        TimeoutConfig,
        Retry,
        retryable,
        CircuitBreaker,
        CircuitState,
        Bulkhead,
        with_timeout,
        is_transient_error,
    )

    # Retry decorator
    @retryable(Retry(max_attempts=3, base_delay=0.5))
    def call_model():
        ...

    # Circuit breaker (per-provider)
    cb = CircuitBreaker("openai", failure_threshold=5, recovery_timeout=30)
    if cb.is_call_permitted():
        try:
            result = call_model()
            cb.on_success()
        except Exception as exc:
            cb.on_failure(exc)
            raise

    # Bulkhead
    bh = Bulkhead("cerebras", max_concurrency=10, queue_size=50)
    with bh.acquire():
        result = call_model()
"""

from __future__ import annotations

from .bulkhead import Bulkhead, BulkheadFull
from .circuit import CircuitBreaker, CircuitBreakerOpen, CircuitState
from .config import ReliabilityConfig, TimeoutConfig
from .retry import Retry, RetryExhausted, is_transient_error, retryable
from .timeouts import TimeoutError as EffGenTimeoutError
from .timeouts import apply_timeout, with_timeout

__all__ = [
    # config
    "ReliabilityConfig",
    "TimeoutConfig",
    # timeouts
    "with_timeout",
    "apply_timeout",
    "EffGenTimeoutError",
    # retry
    "Retry",
    "RetryExhausted",
    "retryable",
    "is_transient_error",
    # circuit breaker
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitState",
    # bulkhead
    "Bulkhead",
    "BulkheadFull",
]
