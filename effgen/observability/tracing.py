"""
OpenTelemetry tracing for effGen.

Samplers
--------
- ``AlwaysOnSampler``   — sample every span
- ``AlwaysOffSampler``  — drop every span
- ``TraceIdRatioSampler(p)``  — probabilistic, 0 ≤ p ≤ 1
- ``RateLimitedSampler(per_second)``  — token-bucket rate limiter
- ``ParentBasedSampler(root)``  — honour parent sampling decision; delegate
  root spans to *root* sampler (usually ``TraceIdRatioSampler`` or
  ``RateLimitedSampler``).

All are valid ``Sampler`` objects understood by the OTel SDK.

Configuration
-------------
Samplers are declared in ``ObservabilityConfig.tracing.sampler``.  Pass a
pre-built sampler instance::

    from effgen.observability.tracing import (
        ParentBasedSampler, TraceIdRatioSampler, setup_tracing
    )
    setup_tracing(sampler=ParentBasedSampler(TraceIdRatioSampler(0.1)))

Context managers
----------------
Every hot path has a dedicated ``start_*`` helper that returns an OTel span
and sets the correct span name + seed attributes from ``spans.py`` constants::

    from effgen.observability.tracing import (
        start_agent_run, start_agent_iteration,
        start_model_call, start_tool_call, start_router_decision
    )

    with start_agent_run(preset="default", task="...", run_id="...") as span:
        span.set_attribute(AgentAttrs.ITERATION, 1)
        with start_model_call(provider="openai", model="gpt-4o-mini") as mspan:
            ...

Retry events
------------
Call ``record_retry_attempt(span, attempt, reason, delay_s)`` to attach a
retry event to an existing span (usually the model.call span).

Telemetry is non-blocking.  Every helper silently catches and swallows its
own exceptions so a tracing failure never propagates to inference.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OTel imports — graceful no-op when SDK is not installed
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.context import Context
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.sampling import (
        Decision,
        ParentBased,
        Sampler,
        SamplingResult,
    )
    from opentelemetry.trace import SpanKind, StatusCode
    from opentelemetry.trace.propagation import get_current_span as _get_current_span

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    Sampler = object  # type: ignore[misc,assignment]

from .spans import (  # noqa: E402  (must come after optional OTel imports)
    AgentAttrs,
    ModelAttrs,
    RetryAttrs,
    RouterAttrs,
    SpanName,
    ToolAttrs,
)

__all__ = [
    # Samplers
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "RateLimitedSampler",
    "ParentBasedSampler",
    # Setup / teardown
    "setup_tracing",
    "get_tracer",
    "shutdown_tracing",
    # Span context managers
    "start_agent_run",
    "start_agent_iteration",
    "start_model_call",
    "start_tool_call",
    "start_router_decision",
    # Span helpers
    "record_retry_attempt",
    "set_span_ok",
    "set_span_error",
    "set_span_attribute",
    # Legacy compat shims (for existing agent.py / utils/tracing.py callers)
    "trace_agent_run",
    "trace_agent_iterate",
    "trace_model_generate",
    "trace_tool_execute",
]


# ---------------------------------------------------------------------------
# Sampler implementations
# ---------------------------------------------------------------------------


if _OTEL_AVAILABLE:

    class AlwaysOnSampler(Sampler):
        """Sample every span.  Wraps the SDK ``ALWAYS_ON`` singleton."""

        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes=None,
            links=None,
            trace_state=None,
        ) -> SamplingResult:
            return SamplingResult(
                decision=Decision.RECORD_AND_SAMPLE,
                attributes=attributes,
                trace_state=trace_state,
            )

        def get_description(self) -> str:  # noqa: D401
            return "AlwaysOnSampler"

    class AlwaysOffSampler(Sampler):
        """Drop every span.  Useful for disabling tracing at runtime."""

        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes=None,
            links=None,
            trace_state=None,
        ) -> SamplingResult:
            return SamplingResult(
                decision=Decision.DROP,
                attributes=attributes,
                trace_state=trace_state,
            )

        def get_description(self) -> str:  # noqa: D401
            return "AlwaysOffSampler"

    class TraceIdRatioSampler(Sampler):
        """
        Probabilistic sampler based on trace-ID hash.

        Deterministic: the same trace ID always produces the same sampling
        decision, so all spans in a trace share the same fate.

        Args:
            ratio: Sampling ratio in [0.0, 1.0].  ``0.0`` → never; ``1.0`` →
                   always.  Clamped to [0, 1].
        """

        def __init__(self, ratio: float) -> None:
            self._ratio = max(0.0, min(1.0, float(ratio)))
            # Convert to integer threshold in [0, 2^128)
            self._threshold = int(self._ratio * (2**128))

        @property
        def ratio(self) -> float:
            return self._ratio

        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes=None,
            links=None,
            trace_state=None,
        ) -> SamplingResult:
            decision = (
                Decision.RECORD_AND_SAMPLE
                if trace_id < self._threshold
                else Decision.DROP
            )
            return SamplingResult(
                decision=decision,
                attributes=attributes,
                trace_state=trace_state,
            )

        def get_description(self) -> str:  # noqa: D401
            return f"TraceIdRatioSampler({self._ratio:.4f})"

    class RateLimitedSampler(Sampler):
        """
        Token-bucket rate-limited sampler.

        Guarantees at most *per_second* sampled traces per second regardless of
        traffic volume.  Implemented as a thread-safe token bucket.

        Args:
            per_second: Maximum number of traces to sample per second.  Must be
                        > 0.  Fractional values are supported (e.g. ``0.5``
                        for one trace every 2 seconds).
        """

        def __init__(self, per_second: float) -> None:
            if per_second <= 0:
                raise ValueError(f"per_second must be > 0, got {per_second!r}")
            self._per_second = float(per_second)
            self._tokens: float = per_second  # start full
            self._last_refill: float = time.monotonic()
            self._lock = threading.Lock()

        @property
        def per_second(self) -> float:
            return self._per_second

        def _refill(self, now: float) -> None:
            """Add tokens proportional to elapsed time (call under lock)."""
            elapsed = now - self._last_refill
            self._tokens = min(
                self._per_second,
                self._tokens + elapsed * self._per_second,
            )
            self._last_refill = now

        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes=None,
            links=None,
            trace_state=None,
        ) -> SamplingResult:
            now = time.monotonic()
            with self._lock:
                self._refill(now)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    decision = Decision.RECORD_AND_SAMPLE
                else:
                    decision = Decision.DROP
            return SamplingResult(
                decision=decision,
                attributes=attributes,
                trace_state=trace_state,
            )

        def get_description(self) -> str:  # noqa: D401
            return f"RateLimitedSampler({self._per_second}/s)"

    class ParentBasedSampler(Sampler):
        """
        Honour the parent span's sampling decision.

        - If there is **no parent** (root span), delegate to *root*.
        - If the parent was **sampled**, sample this span too.
        - If the parent was **not sampled**, drop.

        This mirrors the behaviour of the SDK's built-in ``ParentBased``
        but wraps our custom root samplers.

        Args:
            root: Sampler to use for root spans (no parent trace).
        """

        def __init__(self, root: Sampler) -> None:
            self._root = root
            # Use SDK's ParentBased to correctly read parent context
            self._sdk_parent_based = ParentBased(root=root)

        @property
        def root(self) -> Sampler:
            return self._root

        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes=None,
            links=None,
            trace_state=None,
        ) -> SamplingResult:
            return self._sdk_parent_based.should_sample(
                parent_context,
                trace_id,
                name,
                kind=kind,
                attributes=attributes,
                links=links,
                trace_state=trace_state,
            )

        def get_description(self) -> str:  # noqa: D401
            return f"ParentBasedSampler(root={self._root.get_description()})"

else:
    # Stub classes when OTel is not installed

    class AlwaysOnSampler:  # type: ignore[no-redef]
        def get_description(self) -> str:
            return "AlwaysOnSampler(no-op)"

    class AlwaysOffSampler:  # type: ignore[no-redef]
        def get_description(self) -> str:
            return "AlwaysOffSampler(no-op)"

    class TraceIdRatioSampler:  # type: ignore[no-redef]
        def __init__(self, ratio: float) -> None:
            self._ratio = ratio

        @property
        def ratio(self) -> float:
            return self._ratio

        def get_description(self) -> str:
            return f"TraceIdRatioSampler({self._ratio:.4f})"

    class RateLimitedSampler:  # type: ignore[no-redef]
        def __init__(self, per_second: float) -> None:
            self._per_second = per_second

        @property
        def per_second(self) -> float:
            return self._per_second

        def get_description(self) -> str:
            return f"RateLimitedSampler({self._per_second}/s)"

    class ParentBasedSampler:  # type: ignore[no-redef]
        def __init__(self, root: Any) -> None:
            self._root = root

        @property
        def root(self) -> Any:
            return self._root

        def get_description(self) -> str:
            return f"ParentBasedSampler(root={self._root.get_description()})"


# ---------------------------------------------------------------------------
# Global provider state
# ---------------------------------------------------------------------------

_provider: Any = None
_initialized: bool = False
_provider_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------


def setup_tracing(
    service_name: str | None = None,
    sampler: Any | None = None,
    exporter: Any | None = None,
    export_to_console: bool = False,
) -> None:
    """
    Initialise the OTel ``TracerProvider`` for effGen.

    Safe to call more than once — only the first call takes effect.

    Args:
        service_name:   OTel ``service.name`` resource attribute
                        (default: env ``OTEL_SERVICE_NAME`` or ``"effgen"``).
        sampler:        A sampler instance (default:
                        ``ParentBasedSampler(TraceIdRatioSampler(1.0))``).
        exporter:       A :class:`SpanExporter` to send spans to
                        (default: no-op, i.e. spans are recorded in-memory
                        only when an ``InMemorySpanExporter`` is passed via
                        tests).
        export_to_console: Also add a ``ConsoleSpanExporter`` processor.

    Notes:
        Telemetry is non-blocking — a failure here is logged but never raised.
    """
    global _provider, _initialized

    if not _OTEL_AVAILABLE:
        logger.debug(
            "opentelemetry-sdk not installed — tracing disabled. "
            "Install with: pip install opentelemetry-sdk"
        )
        return

    with _provider_lock:
        if _initialized:
            return

        try:
            svc = service_name or os.environ.get("OTEL_SERVICE_NAME", "effgen")
            resource = Resource.create({"service.name": svc})

            _sampler = sampler or ParentBasedSampler(TraceIdRatioSampler(1.0))

            provider = TracerProvider(resource=resource, sampler=_sampler)

            if exporter is not None:
                provider.add_span_processor(SimpleSpanProcessor(exporter))

            if export_to_console:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

            _otel_trace.set_tracer_provider(provider)
            _provider = provider
            _initialized = True
            logger.debug(
                "effGen tracing initialised: service=%s sampler=%s",
                svc,
                _sampler.get_description(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise tracing (non-fatal): %s", exc)


def get_tracer(name: str = "effgen") -> Any:
    """
    Return the OTel tracer for *name*.

    If the SDK is not installed or ``setup_tracing`` has not been called, a
    no-op context manager is returned so callers do not need guard clauses.
    """
    if _OTEL_AVAILABLE:
        return _otel_trace.get_tracer(name)
    return _NoOpTracer()


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the provider."""
    global _provider, _initialized
    with _provider_lock:
        if _provider is not None and hasattr(_provider, "shutdown"):
            try:
                _provider.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error shutting down tracing (non-fatal): %s", exc)
        _provider = None
        _initialized = False


def reset_tracing() -> None:
    """
    Reset global tracing state (for use in tests between test cases).

    Shuts down any existing provider and clears the OTel global provider lock
    so that the next ``setup_tracing`` call can register a fresh provider.

    .. warning::
        This function is intended for test isolation only.  It uses an
        internal OTel API to reset the "set-once" provider lock.  Do not
        call it in production code.
    """
    global _provider, _initialized
    with _provider_lock:
        if _provider is not None and hasattr(_provider, "shutdown"):
            try:
                _provider.shutdown()
            except Exception:  # noqa: BLE001
                pass
        _provider = None
        _initialized = False
    if _OTEL_AVAILABLE:
        try:
            # Force-reset the OTel SDK's "set-once" lock so the next call to
            # setup_tracing() can register a fresh TracerProvider.
            # This is the idiomatic approach for OTel test isolation.
            _otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
            _otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# No-op fallback objects (when OTel not installed)
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """No-op span context manager."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        pass

    def set_status(self, status: Any, description: str | None = None) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    @property
    def is_recording(self) -> bool:
        return False

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """No-op tracer."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    @contextmanager
    def start_span(self, name: str, **kwargs: Any) -> Generator[_NoOpSpan]:
        yield _NoOpSpan()


# ---------------------------------------------------------------------------
# Span helper context managers
# ---------------------------------------------------------------------------


@contextmanager
def start_agent_run(
    preset: str,
    task: str = "",
    run_id: str | None = None,
) -> Generator[Any]:
    """
    Context manager that opens an ``effgen.agent.run`` span.

    Args:
        preset:  Agent name / preset identifier.
        task:    Task text (truncated to 500 chars).
        run_id:  Optional run UUID.

    Yields:
        The active span (real or no-op).
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        AgentAttrs.PRESET: preset,
        AgentAttrs.TASK: task[:500],
    }
    if run_id:
        attrs[AgentAttrs.RUN_ID] = run_id
    start = time.monotonic()
    try:
        with tracer.start_as_current_span(SpanName.AGENT_RUN, attributes=attrs) as span:
            try:
                yield span
                _buffer_span(f"{SpanName.AGENT_RUN} {preset}", (time.monotonic() - start) * 1000)
            except Exception as exc:  # noqa: BLE001
                _mark_error(span, exc)
                _buffer_span(
                    f"{SpanName.AGENT_RUN} {preset}",
                    (time.monotonic() - start) * 1000,
                    error=str(exc),
                )
                raise
    except Exception:
        raise


@contextmanager
def start_agent_iteration(
    preset: str,
    iteration: int,
) -> Generator[Any]:
    """
    Context manager for a single ReAct iteration span.

    Args:
        preset:    Agent name / preset.
        iteration: 1-based iteration counter.

    Yields:
        The active span.
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        AgentAttrs.PRESET: preset,
        AgentAttrs.ITERATION: iteration,
    }
    try:
        with tracer.start_as_current_span(SpanName.AGENT_ITERATION, attributes=attrs) as span:
            try:
                yield span
            except Exception as exc:  # noqa: BLE001
                _mark_error(span, exc)
                raise
    except Exception:
        raise


@contextmanager
def start_model_call(
    provider: str,
    model: str,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
    parts_count: int | None = None,
) -> Generator[Any]:
    """
    Context manager for a model inference span.

    Seed attributes (tokens, cost, outcome, latency) should be set on the
    span after the call completes::

        with start_model_call("openai", "gpt-4o-mini") as span:
            result = client.generate(...)
            span.set_attribute(ModelAttrs.INPUT_TOKENS, result.prompt_tokens)
            span.set_attribute(ModelAttrs.OUTPUT_TOKENS, result.completion_tokens)
            span.set_attribute(ModelAttrs.OUTCOME, "ok")

    Args:
        provider:         Provider name.
        model:            Model identifier.
        reasoning_effort: Optional effort level for reasoning models.
        thinking_budget:  Optional token budget for reasoning.
        parts_count:      Number of multimodal parts (if applicable).

    Yields:
        The active span.
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        ModelAttrs.PROVIDER: provider,
        ModelAttrs.NAME: model,
    }
    if reasoning_effort is not None:
        attrs[ModelAttrs.REASONING_EFFORT] = reasoning_effort
    if thinking_budget is not None:
        attrs[ModelAttrs.THINKING_BUDGET] = thinking_budget
    if parts_count is not None:
        attrs[ModelAttrs.PARTS_COUNT] = parts_count

    # Model ids often already carry a ``provider:`` prefix (e.g. the caller
    # passes provider="cerebras", model="cerebras:llama3.1-8b"). Avoid a
    # doubled "cerebras:cerebras:..." label in the dashboard span stream.
    _model_label = model if model.startswith(f"{provider}:") else f"{provider}:{model}"

    start = time.monotonic()
    try:
        with tracer.start_as_current_span(SpanName.MODEL_CALL, attributes=attrs) as span:
            try:
                yield span
                # Set latency on the way out (only if span is still recording)
                _set_safe(span, ModelAttrs.LATENCY_MS, round((time.monotonic() - start) * 1000, 1))
                if not _has_outcome(span):
                    _set_safe(span, ModelAttrs.OUTCOME, "ok")
                _mark_ok(span)
                _buffer_span(f"{SpanName.MODEL_CALL} {_model_label}", (time.monotonic() - start) * 1000)
            except Exception as exc:  # noqa: BLE001
                _set_safe(span, ModelAttrs.OUTCOME, "error")
                _set_safe(span, ModelAttrs.LATENCY_MS, round((time.monotonic() - start) * 1000, 1))
                _mark_error(span, exc)
                _buffer_span(
                    f"{SpanName.MODEL_CALL} {_model_label}",
                    (time.monotonic() - start) * 1000,
                    error=str(exc),
                )
                raise
    except Exception:
        raise


@contextmanager
def start_tool_call(
    tool_name: str,
    tool_input: str = "",
) -> Generator[Any]:
    """
    Context manager for a tool execution span.

    Args:
        tool_name:  Tool identifier (e.g. ``"calculator"``).
        tool_input: Serialised input arguments (truncated to 500 chars).

    Yields:
        The active span.
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        ToolAttrs.NAME: tool_name,
        ToolAttrs.INPUT: str(tool_input)[:500],
    }
    start = time.monotonic()
    try:
        with tracer.start_as_current_span(SpanName.TOOL_CALL, attributes=attrs) as span:
            try:
                yield span
                _set_safe(span, ToolAttrs.LATENCY_MS, round((time.monotonic() - start) * 1000, 1))
                if not _has_tool_status(span):
                    _set_safe(span, ToolAttrs.STATUS, "ok")
                _mark_ok(span)
                _buffer_span(f"{SpanName.TOOL_CALL} {tool_name}", (time.monotonic() - start) * 1000)
            except Exception as exc:  # noqa: BLE001
                _set_safe(span, ToolAttrs.STATUS, "error")
                _set_safe(span, ToolAttrs.LATENCY_MS, round((time.monotonic() - start) * 1000, 1))
                _mark_error(span, exc)
                _buffer_span(
                    f"{SpanName.TOOL_CALL} {tool_name}",
                    (time.monotonic() - start) * 1000,
                    error=str(exc),
                )
                raise
    except Exception:
        raise


@contextmanager
def start_router_decision(
    policy: str,
    candidates: list[Any] | None = None,
) -> Generator[Any]:
    """
    Context manager for a routing decision span.

    Args:
        policy:     Routing policy name.
        candidates: List of candidate (provider, model) pairs being considered.

    Yields:
        The active span.
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {
        RouterAttrs.POLICY: policy,
    }
    if candidates:
        attrs[RouterAttrs.CONSIDERED] = ", ".join(
            f"{c[0]}/{c[1]}" if isinstance(c, list | tuple) else str(c)
            for c in candidates
        )
    try:
        with tracer.start_as_current_span(SpanName.ROUTER_DECISION, attributes=attrs) as span:
            try:
                yield span
                _mark_ok(span)
            except Exception as exc:  # noqa: BLE001
                _mark_error(span, exc)
                raise
    except Exception:
        raise


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def record_retry_attempt(
    span: Any,
    attempt: int,
    reason: str,
    delay_s: float = 0.0,
) -> None:
    """
    Add a ``effgen.retry.attempt`` event to *span*.

    Non-blocking: exceptions are swallowed.

    Args:
        span:     The span to attach the event to (may be a no-op span).
        attempt:  1-based attempt number.
        reason:   Short description of why retry was triggered.
        delay_s:  Delay before this attempt in seconds.
    """
    try:
        if span is not None and hasattr(span, "add_event"):
            span.add_event(
                SpanName.RETRY_ATTEMPT,
                attributes={
                    RetryAttrs.ATTEMPT: attempt,
                    RetryAttrs.REASON: reason,
                    RetryAttrs.DELAY_S: float(delay_s),
                },
            )
    except Exception:  # noqa: BLE001
        pass


def set_span_ok(span: Any) -> None:
    """Mark *span* as OK.  Non-blocking."""
    _mark_ok(span)


def set_span_error(span_or_exc: Any, exc: Exception | None = None) -> None:
    """
    Mark the given span (or the current active span) as errored.

    Can be called as:
    - ``set_span_error(span, exc)``
    - ``set_span_error(exc)``  ← marks current active span
    """
    if not _OTEL_AVAILABLE:
        return
    try:
        if exc is None:
            # Called as set_span_error(exc)
            actual_exc: Exception = span_or_exc  # type: ignore[assignment]
            span = _get_current_span()
        else:
            span = span_or_exc
            actual_exc = exc
        if span is not None and hasattr(span, "is_recording") and span.is_recording():
            span.set_status(StatusCode.ERROR, str(actual_exc))
            span.record_exception(actual_exc)
    except Exception:  # noqa: BLE001
        pass


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the currently active span.  Non-blocking."""
    if not _OTEL_AVAILABLE:
        return
    try:
        span = _get_current_span()
        if span is not None and hasattr(span, "is_recording") and span.is_recording():
            span.set_attribute(key, value)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _set_safe(span: Any, key: str, value: Any) -> None:
    """Set span attribute without raising."""
    try:
        if span is not None and hasattr(span, "is_recording") and span.is_recording():
            span.set_attribute(key, value)
    except Exception:  # noqa: BLE001
        pass


def _mark_ok(span: Any) -> None:
    """Set span status to OK without raising."""
    try:
        if _OTEL_AVAILABLE and span is not None and hasattr(span, "set_status"):
            span.set_status(StatusCode.OK)
    except Exception:  # noqa: BLE001
        pass


def _mark_error(span: Any, exc: Exception) -> None:
    """Set span status to ERROR and record exception without raising."""
    try:
        if _OTEL_AVAILABLE and span is not None and hasattr(span, "set_status"):
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
    except Exception:  # noqa: BLE001
        pass


def _has_outcome(span: Any) -> bool:
    """Return True if ModelAttrs.OUTCOME is already set on *span*."""
    try:
        if hasattr(span, "attributes") and span.attributes:
            return ModelAttrs.OUTCOME in span.attributes
    except Exception:  # noqa: BLE001
        pass
    return False


def _has_tool_status(span: Any) -> bool:
    """Return True if ToolAttrs.STATUS is already set on *span*."""
    try:
        if hasattr(span, "attributes") and span.attributes:
            return ToolAttrs.STATUS in span.attributes
    except Exception:  # noqa: BLE001
        pass
    return False


# ---------------------------------------------------------------------------
# Legacy compatibility shims
# (effgen.utils.tracing callers keep working without changes)
# ---------------------------------------------------------------------------


def trace_agent_run(agent_name: str, task: str, run_id: str | None = None) -> Any:
    """Legacy shim → ``start_agent_run``.  Returns a context manager."""
    return start_agent_run(preset=agent_name, task=task, run_id=run_id)


def trace_agent_iterate(agent_name: str, iteration: int) -> Any:
    """Legacy shim → ``start_agent_iteration``."""
    return start_agent_iteration(preset=agent_name, iteration=iteration)


def trace_model_generate(model_name: str, prompt_tokens: int = 0) -> Any:
    """Legacy shim → ``start_model_call`` with provider inferred from model name."""
    provider = _infer_provider(model_name)
    return start_model_call(provider=provider, model=model_name)


def trace_tool_execute(tool_name: str, tool_input: str) -> Any:
    """Legacy shim → ``start_tool_call``."""
    return start_tool_call(tool_name=tool_name, tool_input=tool_input)


def _infer_provider(model_name: str) -> str:
    """Best-effort provider inference from a model name string."""
    m = model_name.lower()
    if m.startswith(("gpt-", "o1", "o3", "o4", "text-")):
        return "openai"
    if m.startswith(("gemini", "models/gemini")):
        return "google"
    if m.startswith(("claude", "anthropic")):
        return "anthropic"
    if m.startswith(("llama", "qwen", "cerebras")):
        return "cerebras"
    if m.startswith(("mixtral", "mistral")):
        return "groq"
    return "unknown"


# ---------------------------------------------------------------------------
# In-memory span ring buffer (used by the dashboard)
# ---------------------------------------------------------------------------

_SPAN_BUFFER_LOCK: threading.Lock = threading.Lock()
_SPAN_BUFFER: deque[dict] = deque(maxlen=500)  # type: ignore[type-arg]


def _buffer_span(name: str, duration_ms: float, error: str | None = None) -> None:
    """Append a span dict to the in-memory ring buffer (best-effort, never raises)."""
    try:
        record = {
            "ts": time.strftime("%H:%M:%S", time.gmtime()),
            "name": name,
            "duration_ms": round(duration_ms, 1),
            "error": error,
        }
        with _SPAN_BUFFER_LOCK:
            _SPAN_BUFFER.append(record)
    except Exception:  # noqa: BLE001 - telemetry must never break inference
        pass


def get_recent_spans(*, limit: int = 100) -> list[dict]:  # type: ignore[type-arg]
    """Return the most-recent *limit* span records (newest first).

    These are populated when effGen agent / model-call context managers exit.
    Returns an empty list when no spans have been buffered.
    """
    with _SPAN_BUFFER_LOCK:
        spans = list(_SPAN_BUFFER)
    return list(reversed(spans))[:limit]
