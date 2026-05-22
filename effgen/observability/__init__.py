"""
effGen Observability -- structured logging, secret redaction, metrics, and SLOs.

Quick start
-----------
    from effgen.observability import get_logger, configure_logging

    # Configure once at application startup (call before first log line):
    configure_logging(level="INFO")

    # Then in any module:
    log = get_logger(__name__)
    log.event("model.call.started", model="gemini-3", cached_tokens=128)
    log.event("model.call.done",    model="gemini-3", latency_ms=340)
    log.info("something happened",  extra_key="value")

    # Metrics (Prometheus histograms + counters):
    from effgen.observability import record_model_call, record_tokens, export_metrics
    record_model_call(provider="cerebras", model="llama3.1-8b",
                      outcome="ok", latency=0.42)
    record_tokens(provider="cerebras", model="llama3.1-8b",
                  input_tokens=120, output_tokens=50)
    print(export_metrics())  # Prometheus text format

    # SLO tracking:
    from effgen.observability import SLO, get_slo_tracker
    tracker = get_slo_tracker()
    tracker.register(SLO("model_call_success", 99.0, 3600))
    tracker.record("model_call_success", ok=True)
    print(tracker.burn_rate("model_call_success"))

Sub-modules
-----------
- effgen.observability.logs    -- StructuredFormatter and EffGenLogger
- effgen.observability.redact  -- Redactor and get_redactor
- effgen.observability.metrics -- Prometheus histograms + counters
- effgen.observability.slo     -- SLO and SLOTracker
"""

from .logs import (
    EffGenLogger,
    StructuredFormatter,
    clear_run_context,
    configure_logging,
    get_effgen_logger,
    set_run_context,
)
from .metrics import (
    LabeledCounter,
    LabeledHistogram,
    agent_iteration_latency,
    export_metrics,
    model_call_latency,
    record_agent_iteration,
    record_model_call,
    record_tokens,
    record_tool_call,
    tokens_total,
    tool_call_latency,
)
from .metrics import (
    reset_all as reset_metrics,
)
from .redact import Redactor, get_redactor
from .slo import SLO, SLOTracker
from .slo import get_tracker as get_slo_tracker


def get_logger(name: str) -> "EffGenLogger":
    """
    Get a structured EffGenLogger for *name*.

    This is the canonical entry point for all effGen components.

    Args:
        name: Logger name -- pass ``__name__`` from the calling module.

    Returns:
        EffGenLogger instance (cached).

    Example::

        from effgen.observability import get_logger
        log = get_logger(__name__)
        log.event("model.call.started", model="gemini-3", cached_tokens=128)
    """
    return get_effgen_logger(name)


__all__ = [
    # Primary entry point
    "get_logger",
    # Configuration
    "configure_logging",
    # Core logging classes
    "EffGenLogger",
    "StructuredFormatter",
    "Redactor",
    # Factories
    "get_redactor",
    "get_effgen_logger",
    # Run context helpers (used by agent loop)
    "set_run_context",
    "clear_run_context",
    # Metrics (Phase 2)
    "model_call_latency",
    "tool_call_latency",
    "agent_iteration_latency",
    "tokens_total",
    "record_model_call",
    "record_tool_call",
    "record_agent_iteration",
    "record_tokens",
    "export_metrics",
    "reset_metrics",
    "LabeledHistogram",
    "LabeledCounter",
    # SLO (Phase 2)
    "SLO",
    "SLOTracker",
    "get_slo_tracker",
]
