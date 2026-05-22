"""
effGen Observability — Prometheus-style metrics with histograms and counters.

Provides:
- ``effgen_model_call_latency_seconds{provider,model,outcome}`` — Histogram
- ``effgen_tool_call_latency_seconds{tool,outcome}`` — Histogram
- ``effgen_agent_iteration_latency_seconds{preset}`` — Histogram
- ``effgen_tokens_total{provider,model,kind}`` — Counter (kind ∈ input/output/cached)

Plus the legacy counters from ``effgen.utils.prometheus_metrics`` (still exported
for backwards-compatibility).

Quick start
-----------
    from effgen.observability.metrics import (
        record_model_call,
        record_tool_call,
        record_agent_iteration,
        record_tokens,
        export_metrics,
    )

    # Time a model call:
    import time
    t0 = time.perf_counter()
    try:
        response = model.call(...)
        record_model_call(provider="cerebras", model="llama3.1-8b",
                          outcome="ok", latency=time.perf_counter() - t0)
    except Exception:
        record_model_call(provider="cerebras", model="llama3.1-8b",
                          outcome="error", latency=time.perf_counter() - t0)
        raise
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

# Default latency buckets: 50 ms … 60 s (roughly logarithmic)
_LATENCY_BUCKETS: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0
)


def _default_buckets() -> tuple[float, ...]:
    return _LATENCY_BUCKETS


# ---------------------------------------------------------------------------
# Core metric primitives
# ---------------------------------------------------------------------------


@dataclass
class LabeledHistogram:
    """
    A Prometheus-style histogram with configurable bucket boundaries
    and arbitrary label sets.

    Thread-safe via a per-instance lock.
    """

    name: str
    help: str
    label_names: tuple[str, ...]
    buckets: tuple[float, ...] = field(default_factory=_default_buckets)

    # Internal state — protected by _lock
    # Keyed by frozenset of (label_name, label_value) pairs → (sum, count, bucket_counts)
    _data: dict[tuple[tuple[str, str], ...], dict] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Make sure buckets end with +Inf
        inf = math.inf
        buckets = tuple(sorted(self.buckets))
        if not buckets or buckets[-1] != inf:
            buckets = buckets + (inf,)
        self.buckets = buckets

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record one observation with optional *labels*."""
        key = self._make_key(labels or {})
        with self._lock:
            entry = self._get_or_create(key)
            entry["sum"] += value
            entry["count"] += 1
            # Store NON-cumulative counts per bucket — the first bucket whose
            # upper bound is >= value gets the increment.  Cumulative sums are
            # computed at export time.
            for i, upper in enumerate(self.buckets):
                if value <= upper:
                    entry["buckets"][i] += 1
                    break  # only the first matching bucket

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> str:
        """Return Prometheus text-format lines for this histogram."""
        lines: list[str] = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            for key, entry in self._data.items():
                label_str = self._format_label_str(dict(key))
                cum = 0
                for i, upper in enumerate(self.buckets):
                    cum += entry["buckets"][i]
                    le = "+Inf" if math.isinf(upper) else str(upper)
                    le_label = self._add_label(label_str, "le", le)
                    lines.append(f"{self.name}_bucket{{{le_label}}} {cum}")
                # _sum and _count: use {} notation only when labels are present
                if label_str:
                    lines.append(f"{self.name}_sum{{{label_str}}} {entry['sum']}")
                    lines.append(f"{self.name}_count{{{label_str}}} {entry['count']}")
                else:
                    lines.append(f"{self.name}_sum {entry['sum']}")
                    lines.append(f"{self.name}_count {entry['count']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all observations (useful in tests)."""
        with self._lock:
            self._data.clear()

    def _make_key(self, labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(labels.items()))

    def _get_or_create(
        self, key: tuple[tuple[str, str], ...]
    ) -> dict:
        if key not in self._data:
            self._data[key] = {
                "sum": 0.0,
                "count": 0,
                "buckets": [0] * len(self.buckets),
            }
        return self._data[key]

    @staticmethod
    def _format_label_str(labels: dict[str, str]) -> str:
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    @staticmethod
    def _add_label(existing: str, name: str, value: str) -> str:
        new_part = f'{name}="{value}"'
        if not existing:
            return new_part
        return f"{existing},{new_part}"


@dataclass
class LabeledCounter:
    """
    A Prometheus-style monotonic counter with arbitrary label sets.

    Thread-safe via a per-instance lock.
    """

    name: str
    help: str

    _data: dict[tuple[tuple[str, str], ...], float] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment the counter by *amount*."""
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._data[key] = self._data.get(key, 0.0) + amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Return current value for *labels*."""
        key = tuple(sorted((labels or {}).items()))
        return self._data.get(key, 0.0)

    def reset(self) -> None:
        with self._lock:
            self._data.clear()

    def export(self) -> str:
        """Return Prometheus text-format lines for this counter."""
        lines: list[str] = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            for key, value in self._data.items():
                label_str = ",".join(f'{k}="{v}"' for k, v in key)
                if label_str:
                    lines.append(f"{self.name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase-2 metric instances (module-level singletons)
# ---------------------------------------------------------------------------

#: Latency histogram for model (LLM) calls.
#: Labels: provider, model, outcome  (outcome ∈ "ok" | "error" | "timeout")
model_call_latency = LabeledHistogram(
    name="effgen_model_call_latency_seconds",
    help="Latency of model (LLM) calls in seconds",
    label_names=("provider", "model", "outcome"),
    buckets=_LATENCY_BUCKETS,
)

#: Latency histogram for tool executions.
#: Labels: tool, outcome  (outcome ∈ "ok" | "error" | "timeout")
tool_call_latency = LabeledHistogram(
    name="effgen_tool_call_latency_seconds",
    help="Latency of tool calls in seconds",
    label_names=("tool", "outcome"),
    buckets=_LATENCY_BUCKETS,
)

#: Latency histogram for one agent iteration (prompt → model → tools → response).
#: Labels: preset
agent_iteration_latency = LabeledHistogram(
    name="effgen_agent_iteration_latency_seconds",
    help="Latency of a single agent iteration in seconds",
    label_names=("preset",),
    buckets=_LATENCY_BUCKETS,
)

#: Token counter.
#: Labels: provider, model, kind  (kind ∈ "input" | "output" | "cached")
tokens_total = LabeledCounter(
    name="effgen_tokens_total",
    help="Total tokens consumed, by provider/model/kind",
)

# ---------------------------------------------------------------------------
# Convenience recording functions
# ---------------------------------------------------------------------------


def record_model_call(
    *,
    provider: str,
    model: str,
    outcome: str,
    latency: float,
) -> None:
    """
    Record one model-call observation.

    Args:
        provider: Provider identifier (e.g. ``"cerebras"``).
        model: Model name (e.g. ``"llama3.1-8b"``).
        outcome: ``"ok"``, ``"error"``, or ``"timeout"``.
        latency: Elapsed time in **seconds**.
    """
    model_call_latency.observe(
        latency,
        labels={"provider": provider, "model": model, "outcome": outcome},
    )


def record_tool_call(
    *,
    tool: str,
    outcome: str,
    latency: float,
) -> None:
    """
    Record one tool-call observation.

    Args:
        tool: Tool name.
        outcome: ``"ok"`` or ``"error"``.
        latency: Elapsed time in **seconds**.
    """
    tool_call_latency.observe(
        latency,
        labels={"tool": tool, "outcome": outcome},
    )


def record_agent_iteration(
    *,
    preset: str,
    latency: float,
) -> None:
    """
    Record one agent-iteration observation.

    Args:
        preset: Preset/agent name.
        latency: Elapsed time in **seconds**.
    """
    agent_iteration_latency.observe(latency, labels={"preset": preset})


def record_tokens(
    *,
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> None:
    """
    Record token consumption for one model call.

    Args:
        provider: Provider identifier.
        model: Model name.
        input_tokens: Prompt tokens sent.
        output_tokens: Completion tokens received.
        cached_tokens: Prompt tokens that were cache hits.
    """
    base = {"provider": provider, "model": model}
    if input_tokens:
        tokens_total.inc(input_tokens, labels={**base, "kind": "input"})
    if output_tokens:
        tokens_total.inc(output_tokens, labels={**base, "kind": "output"})
    if cached_tokens:
        tokens_total.inc(cached_tokens, labels={**base, "kind": "cached"})


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_metrics() -> str:
    """
    Return all Phase-2 metrics in Prometheus text format.

    The output also includes the legacy counters from
    ``effgen.utils.prometheus_metrics`` so one scrape endpoint covers
    everything.
    """
    sections = [
        model_call_latency.export(),
        tool_call_latency.export(),
        agent_iteration_latency.export(),
        tokens_total.export(),
    ]
    # Append legacy metrics (non-blocking)
    try:
        from effgen.utils.prometheus_metrics import metrics as _legacy

        sections.append(_legacy.export())
    except Exception:  # pragma: no cover
        pass

    return "\n\n".join(s for s in sections if s) + "\n"


def reset_all() -> None:
    """Reset all Phase-2 metrics (used in tests)."""
    model_call_latency.reset()
    tool_call_latency.reset()
    agent_iteration_latency.reset()
    tokens_total.reset()


__all__ = [
    # Metric instances
    "model_call_latency",
    "tool_call_latency",
    "agent_iteration_latency",
    "tokens_total",
    # Recording helpers
    "record_model_call",
    "record_tool_call",
    "record_agent_iteration",
    "record_tokens",
    # Export
    "export_metrics",
    "reset_all",
    # Primitives (for external use)
    "LabeledHistogram",
    "LabeledCounter",
]
