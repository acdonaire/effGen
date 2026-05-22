"""
effGen Observability — structured logging, secret redaction, and telemetry.

Quick start
-----------
    from effgen.observability import get_logger, configure_logging

    # Configure once at application startup (call before first log line):
    configure_logging(level="INFO")

    # Then in any module:
    log = get_logger(__name__)
    log.event("model.call.started", model="llama3.1-8b", cached_tokens=0)
    log.event("model.call.done",    model="llama3.1-8b", latency_ms=340)
    log.info("something happened",  extra_key="value")

Sub-modules
-----------
- :mod:`effgen.observability.logs`    — :class:`~effgen.observability.logs.StructuredFormatter`
  and :class:`~effgen.observability.logs.EffGenLogger`
- :mod:`effgen.observability.redact`  — :class:`~effgen.observability.redact.Redactor`
  and :func:`~effgen.observability.redact.get_redactor`
"""

from .logs import (
    EffGenLogger,
    StructuredFormatter,
    clear_run_context,
    configure_logging,
    get_effgen_logger,
    set_run_context,
)
from .redact import Redactor, get_redactor


def get_logger(name: str) -> EffGenLogger:
    """
    Get a structured :class:`~effgen.observability.logs.EffGenLogger` for *name*.

    This is the canonical entry point for all effGen components.

    Args:
        name: Logger name — pass ``__name__`` from the calling module.

    Returns:
        :class:`~effgen.observability.logs.EffGenLogger` instance (cached).

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
    # Core classes
    "EffGenLogger",
    "StructuredFormatter",
    "Redactor",
    # Factories
    "get_redactor",
    "get_effgen_logger",
    # Run context helpers (used by agent loop)
    "set_run_context",
    "clear_run_context",
]
