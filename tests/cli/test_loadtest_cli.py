"""Tests for the ``effgen loadtest`` report rendering.

Covers the drain note (requested vs. wall-clock duration when in-flight
requests are drained past the requested window) and the per-category error
breakdown line, both read straight off ``LoadReport.to_dict()``.
"""

from __future__ import annotations

from effgen.cli.loadtest import _print_report
from effgen.tools.loadgen import LoadReport


def _report(**overrides) -> LoadReport:
    defaults = {
        "scenario": "fixed",
        "concurrency": 3,
        "duration": 10.0,
        "total_requests": 35,
        "successful_requests": 32,
        "failed_requests": 3,
        "error_rate": 0.0857,
        "throughput": 0.98,
        "p50_latency": 0.137,
        "p95_latency": 30.003,
        "p99_latency": 30.019,
        "min_latency": 0.093,
        "max_latency": 30.027,
        "mean_latency": 2.88,
        "stdev_latency": 8.46,
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "requested_duration": 10.0,
        "error_breakdown": {},
    }
    defaults.update(overrides)
    return LoadReport(**defaults)


def test_no_drain_note_when_duration_matches_requested(capsys):
    _print_report(_report(duration=10.0, requested_duration=10.0))
    out = capsys.readouterr().out
    assert "Duration      : 10.0s" in out
    assert "draining" not in out


def test_drain_note_on_overshoot(capsys):
    _print_report(_report(duration=35.636, requested_duration=10.0))
    out = capsys.readouterr().out
    assert "requested 10.0s" in out
    assert "wall 35.6s" in out
    assert "draining in-flight requests" in out


def test_no_drain_note_for_sub_precision_jitter(capsys):
    # A few ms of scheduler jitter must not print a "incl. 0.0s draining" note
    # that would display as zero and contradict itself.
    _print_report(_report(duration=2.003, requested_duration=2.0))
    out = capsys.readouterr().out
    assert "Duration      : 2.0s" in out
    assert "draining" not in out


def test_error_breakdown_line_present_when_failures(capsys):
    _print_report(_report(error_breakdown={"timeout": 2, "rate_limited": 1}))
    out = capsys.readouterr().out
    assert "Error types" in out
    assert "timeout=2" in out
    assert "rate_limited=1" in out


def test_error_breakdown_line_absent_when_no_failures(capsys):
    _print_report(_report(failed_requests=0, error_rate=0.0, error_breakdown={}))
    out = capsys.readouterr().out
    assert "Error types" not in out
