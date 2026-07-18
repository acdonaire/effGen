"""Tests for the self-contained HTML reports and the ``effgen report`` command.

The reports render from result documents that ``compare``/``eval``/``cost``/
``loadtest`` already produce, so the fixtures here are those documents — no
model call is involved in rendering.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from html.parser import HTMLParser

import pytest

from effgen.ui.report_html import (
    REPORT_KINDS,
    ReportError,
    build_html_report,
    detect_report_kind,
    load_result_document,
    write_html_report,
)

# ---------------------------------------------------------------------------
# Fixtures: one result document per report kind
# ---------------------------------------------------------------------------

COMPARISON_DOC = {
    "scores": [
        {
            "model": "openai:gpt-5-nano", "suite": "math", "accuracy": 1.0,
            "avg_latency": 1.35, "total_tokens": 366, "avg_cost_usd": 4.1e-05,
            "avg_tool_accuracy": 0.0, "error": None,
        },
        {
            "model": "groq:llama-3.1-8b-instant", "suite": "math", "accuracy": 1.0,
            "avg_latency": 0.27, "total_tokens": 214, "avg_cost_usd": 5e-06,
            "avg_tool_accuracy": 0.0, "error": None,
        },
        {
            "model": "transformers:Qwen/Qwen2.5-1.5B-Instruct", "suite": "math",
            "accuracy": 0.67, "avg_latency": 0.38, "total_tokens": 190,
            "avg_cost_usd": None, "avg_tool_accuracy": 0.0, "error": None,
        },
        {
            "model": "openai:does-not-exist", "suite": "math", "accuracy": 0.0,
            "avg_latency": 0.0, "total_tokens": 0, "avg_cost_usd": None,
            "avg_tool_accuracy": 0.0, "error": "model not found",
        },
    ],
    "recommendations": {"math": "groq:llama-3.1-8b-instant"},
    "recommendation_rationale": {
        "math": "cheapest at 100% accuracy — $0.000005/run vs $0.000041/run for openai:gpt-5-nano",
    },
    "optimize": "cost",
    "scoring": "contains",
    "num_cases": 3,
    "suite_sizes": {"math": 3},
    "generated_at": "2026-07-18T14:00:00+00:00",
}

EVAL_DOC = {
    "suite": "math", "total": 2, "passed": 1, "accuracy": 0.5,
    "avg_latency": 0.53, "total_tokens": 179, "total_cost_usd": 5.2e-05,
    "avg_tool_accuracy": 0.0,
    "by_difficulty": {"easy": {"total": 2, "passed": 1, "accuracy": 0.5}},
    "metadata": {"scoring": "contains", "pass_threshold": 0.5, "num_cases": 2,
                 "model": "gemini-3.1-flash-lite", "fail_under": 0.9},
    "results": [
        {"query": "What is 2+2?", "expected_output": "4", "agent_output": "4",
         "score": 1.0, "passed": True, "latency": 0.5, "tokens_used": 90,
         "cost_usd": 2.6e-05, "tool_accuracy": 0.0, "tools_called": [],
         "difficulty": "easy", "details": {}},
        {"query": "What is <b>7*6</b>?", "expected_output": "42",
         "agent_output": "I think it is 40 & maybe more", "score": 0.0,
         "passed": False, "latency": 0.56, "tokens_used": 89, "cost_usd": None,
         "tool_accuracy": 0.0, "tools_called": [], "difficulty": "easy",
         "details": {}},
    ],
}

COST_DOC = {
    "period": "Lifetime", "period_days": None,
    "total_requests": 120, "total_cost_usd": 1.5985,
    "daily_budget_usd": 1.0,
    "rows": [
        {"provider": "openai", "model": "gpt-5-nano", "requests": 80,
         "prompt_tokens": 9000, "completion_tokens": 4000, "cost_usd": 1.2,
         "cost_label": "$1.200000"},
        {"provider": "groq", "model": "llama-3.1-8b-instant", "requests": 30,
         "prompt_tokens": 2000, "completion_tokens": 800, "cost_usd": 0.3985,
         "cost_label": "$0.398500"},
        {"provider": "transformers", "model": "Qwen/Qwen2.5-1.5B-Instruct",
         "requests": 10, "prompt_tokens": 500, "completion_tokens": 200,
         "cost_usd": 0.0, "cost_label": "unpriced"},
    ],
}

LOADTEST_DOC = {
    "scenario": "synthetic", "concurrency": 8, "duration_s": 5.1,
    "requested_duration_s": 5.0, "drain_s": 0.1, "total_requests": 1000,
    "successful_requests": 990, "failed_requests": 10, "error_rate": 0.01,
    "error_breakdown": {"timeout": 7, "rate_limited": 3},
    "throughput_rps": 196.0784,
    "latency": {"p50": 0.03, "p95": 0.08, "p99": 0.12, "min": 0.01,
                "max": 0.4, "mean": 0.035, "stdev": 0.012},
    "provider": None, "model": None,
}

DOCS = {
    "comparison": COMPARISON_DOC,
    "eval": EVAL_DOC,
    "cost": COST_DOC,
    "loadtest": LOADTEST_DOC,
}


class _WellFormed(HTMLParser):
    """Checks that every element opened is closed in order."""

    VOID = {"meta", "br", "img", "input", "hr", "link", "source"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.mismatches: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.mismatches.append(tag)

    def handle_data(self, data):
        stripped = data.strip()
        if stripped:
            self.text.append(stripped)


def _parse(html: str) -> _WellFormed:
    stripped = re.sub(r"<style>.*?</style>|<script>.*?</script>", "", html, flags=re.S)
    parser = _WellFormed()
    parser.feed(stripped)
    return parser


# ---------------------------------------------------------------------------
# Kind detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_detect_report_kind_identifies_each_document(kind):
    assert detect_report_kind(DOCS[kind]) == kind


def test_detect_report_kind_returns_none_for_unrelated_json():
    assert detect_report_kind({"hello": "world"}) is None


def test_build_report_rejects_unidentifiable_document():
    with pytest.raises(ReportError) as exc:
        build_html_report({"hello": "world"})
    # The message names the commands whose JSON is renderable.
    assert "compare" in str(exc.value) and "loadtest" in str(exc.value)


def test_build_report_rejects_unknown_kind():
    with pytest.raises(ReportError):
        build_html_report(EVAL_DOC, kind="not-a-kind")


def test_build_report_rejects_non_mapping():
    with pytest.raises(ReportError):
        build_html_report([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Self-containment: the report must open with no network access
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_report_references_no_external_asset(kind):
    html = build_html_report(DOCS[kind], kind=kind)
    assert "http://" not in html
    assert "https://" not in html
    assert "//cdn" not in html
    # No element may pull in an external file.
    assert not re.search(r"\bsrc\s*=", html)
    assert not re.search(r"\bhref\s*=", html)
    assert "@import" not in html
    assert "url(" not in html


@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_report_is_a_complete_well_formed_document(kind):
    html = build_html_report(DOCS[kind], kind=kind)
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>" in html and "</html>" in html.rstrip()
    parsed = _parse(html)
    assert parsed.stack == []
    assert parsed.mismatches == []


@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_report_carries_provenance(kind):
    html = build_html_report(DOCS[kind], kind=kind, command="effgen demo --flag")
    from effgen import __version__
    assert "Generated" in html and "UTC" in html
    assert __version__ in html
    assert "effgen demo --flag" in html


@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_report_is_theme_aware(kind):
    html = build_html_report(DOCS[kind], kind=kind)
    from effgen.ui.palette import DASHBOARD_DARK, DASHBOARD_LIGHT
    assert "prefers-color-scheme: dark" in html
    assert '[data-theme="light"]' in html and '[data-theme="dark"]' in html
    assert DASHBOARD_DARK["accent"] in html
    assert DASHBOARD_LIGHT["accent"] in html


# ---------------------------------------------------------------------------
# Content: the numbers that exist are shown, the ones that don't are not invented
# ---------------------------------------------------------------------------

def test_comparison_report_shows_verdict_rationale_and_unpriced():
    html = build_html_report(COMPARISON_DOC, kind="comparison")
    assert "groq:llama-3.1-8b-instant" in html
    assert "cheapest at 100% accuracy" in html
    # A local model with no published price is labeled, never rendered as $0.
    assert "unpriced" in html
    assert "$0.000000" not in html
    # A failed model is marked rather than shown as a zero score.
    assert "ERROR" in html
    # Self-describing header fields.
    assert "contains" in html and "3" in html


def test_eval_report_shows_gate_and_every_case():
    html = build_html_report(EVAL_DOC, kind="eval")
    parsed = _parse(html)
    text = " ".join(parsed.text)
    assert "FAIL" in text  # accuracy 50% is under the 90% gate
    assert "PASS" in text  # the passing case
    assert "What is 2+2?" in text
    # Case text is escaped, not interpreted as markup.
    assert "<b>7*6</b>" not in html
    assert "&lt;b&gt;7*6&lt;/b&gt;" in html
    assert "&amp;" in html


def test_eval_report_pass_fail_is_not_color_only():
    """Each case states PASS/FAIL in text, so the result survives a mono print."""
    html = build_html_report(EVAL_DOC, kind="eval")
    assert html.count("PASS") >= 1 and html.count("FAIL") >= 1


def test_cost_report_shows_totals_and_share():
    html = build_html_report(COST_DOC, kind="cost")
    text = " ".join(_parse(html).text)
    assert "$1.5985" in text
    assert "$1.00" in text  # the daily budget
    assert "unpriced" in text
    assert "<svg" in html  # the share donut is inline SVG


def test_lifetime_total_is_not_judged_against_the_daily_budget():
    """A lifetime total spans no budget window, so it gets no over/under verdict."""
    text = " ".join(_parse(build_html_report(COST_DOC, kind="cost")).text)
    assert "spans no fixed budget window" in text
    assert "over budget" not in text
    assert "% used" not in text


def test_daily_spend_is_measured_against_the_daily_budget():
    doc = dict(COST_DOC, period="Last 24 hours", period_days=1,
               total_cost_usd=1.5985, daily_budget_usd=1.0)
    text = " ".join(_parse(build_html_report(doc, kind="cost")).text)
    assert "over budget" in text  # 1.5985 of 1.00 in one day
    assert "160% used" in text


def test_weekly_spend_scales_the_budget_to_the_window():
    """Seven days of spend is compared with seven days of budget, not one."""
    doc = dict(COST_DOC, period="Last 7 days", period_days=7,
               total_cost_usd=3.0, daily_budget_usd=1.0)
    text = " ".join(_parse(build_html_report(doc, kind="cost")).text)
    assert "$7.00" in text  # the daily budget scaled to the window
    assert "within budget" in text
    assert "over budget" not in text


def test_cost_report_without_budget_points_at_the_command():
    doc = dict(COST_DOC, daily_budget_usd=None)
    html = build_html_report(doc, kind="cost")
    assert "No daily budget configured" in html
    assert "effgen cost set-budget" in html


def test_cost_report_handles_an_empty_ledger():
    doc = {"period": "Last 24 hours", "total_requests": 0,
           "total_cost_usd": 0.0, "daily_budget_usd": None, "rows": []}
    html = build_html_report(doc, kind="cost")
    assert "No spend recorded" in html


def test_loadtest_report_shows_percentiles_and_error_breakdown():
    html = build_html_report(LOADTEST_DOC, kind="loadtest")
    text = " ".join(_parse(html).text)
    assert "30.0 ms" in text   # p50
    assert "80.0 ms" in text   # p95
    assert "120.0 ms" in text  # p99
    assert "timeout" in text and "rate_limited" in text
    assert "196.08 req/s" in text
    assert "1.00% errors" in text


def test_loadtest_report_without_errors_says_so():
    doc = dict(LOADTEST_DOC, error_rate=0.0, failed_requests=0,
               successful_requests=1000, error_breakdown={})
    html = build_html_report(doc, kind="loadtest")
    assert "No failed requests to break down." in html


def test_missing_metric_renders_a_dash_not_a_zero():
    doc = dict(LOADTEST_DOC, latency={"p50": None, "p95": None, "p99": None,
                                      "min": None, "max": None, "mean": None,
                                      "stdev": None},
               throughput_rps=None)
    html = build_html_report(doc, kind="loadtest")
    assert "—" in html
    # An absent measurement is never substituted with a zero.
    assert " ms" not in html
    assert "req/s" not in html


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def test_write_html_report_creates_parent_directories(tmp_path):
    out = tmp_path / "nested" / "dir" / "report.html"
    written = write_html_report(out, EVAL_DOC, kind="eval")
    assert written == out
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_load_result_document_reports_a_bad_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ReportError) as exc:
        load_result_document(bad)
    assert "not valid JSON" in str(exc.value)

    with pytest.raises(ReportError) as exc:
        load_result_document(tmp_path / "missing.json")
    assert "No such result file" in str(exc.value)

    listy = tmp_path / "list.json"
    listy.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ReportError):
        load_result_document(listy)


# ---------------------------------------------------------------------------
# `effgen report` end-to-end through the CLI
# ---------------------------------------------------------------------------

def _run_cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "effgen.cli", *argv],
        capture_output=True, text=True, timeout=180,
    )


@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_report_command_renders_each_saved_result(tmp_path, kind):
    src = tmp_path / f"{kind}.json"
    src.write_text(json.dumps(DOCS[kind]), encoding="utf-8")
    out = tmp_path / f"{kind}.html"
    proc = _run_cli("report", str(src), "-o", str(out))
    assert proc.returncode == 0, proc.stderr
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "http://" not in html and "https://" not in html


def test_report_command_defaults_the_output_path(tmp_path):
    src = tmp_path / "results.json"
    src.write_text(json.dumps(EVAL_DOC), encoding="utf-8")
    proc = _run_cli("report", str(src))
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "results.html").exists()


def test_report_command_reports_an_unrecognized_document(tmp_path):
    src = tmp_path / "other.json"
    src.write_text(json.dumps({"unrelated": True}), encoding="utf-8")
    proc = _run_cli("report", str(src))
    assert proc.returncode == 2
    combined = proc.stdout + proc.stderr
    assert "--kind" in combined


def test_report_command_honors_an_explicit_kind(tmp_path):
    """An ambiguous document still renders when the kind is named."""
    src = tmp_path / "amb.json"
    src.write_text(json.dumps({"rows": [], "period": "Lifetime"}), encoding="utf-8")
    out = tmp_path / "amb.html"
    proc = _run_cli("report", str(src), "--kind", "cost", "-o", str(out))
    assert proc.returncode == 0, proc.stderr
    assert "Spend Report" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# `-o` extension handling across the commands
# ---------------------------------------------------------------------------

def test_output_extension_chooses_the_format():
    from effgen.cli._main import _artifact_format
    assert _artifact_format("out.html") == "html"
    assert _artifact_format("out.HTM") == "html"
    assert _artifact_format("out.md") == "markdown"
    assert _artifact_format("out.markdown") == "markdown"
    assert _artifact_format("out.json") == "json"
    assert _artifact_format("out") == "json"


def test_compare_output_html_writes_a_report_not_json(tmp_path):
    """`-o result.html` renders the report rather than writing JSON to an .html file."""
    from effgen.cli import CLIInterface
    from effgen.cli._main import _write_result_artifact

    cli = CLIInterface()
    cli.console = None
    out = tmp_path / "result.html"
    _write_result_artifact(
        str(out), cli=cli, data=COMPARISON_DOC, kind="comparison",
        json_text=json.dumps(COMPARISON_DOC), markdown_text="# md",
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert '"scores"' not in text


def test_eval_output_markdown_writes_markdown(tmp_path):
    from effgen.cli import CLIInterface
    from effgen.cli._main import _write_result_artifact

    cli = CLIInterface()
    cli.console = None
    out = tmp_path / "result.md"
    _write_result_artifact(
        str(out), cli=cli, data=EVAL_DOC, kind="eval",
        json_text=json.dumps(EVAL_DOC), markdown_text="# Evaluation Results",
    )
    assert out.read_text(encoding="utf-8").startswith("# Evaluation Results")


def test_unknown_extension_still_writes_json(tmp_path):
    from effgen.cli import CLIInterface
    from effgen.cli._main import _write_result_artifact

    cli = CLIInterface()
    cli.console = None
    out = tmp_path / "result.txt"
    _write_result_artifact(
        str(out), cli=cli, data=EVAL_DOC, kind="eval",
        json_text=json.dumps(EVAL_DOC), markdown_text="# md",
    )
    assert json.loads(out.read_text(encoding="utf-8"))["suite"] == "math"


# ---------------------------------------------------------------------------
# The result objects the reports read
# ---------------------------------------------------------------------------

def test_comparison_matrix_carries_report_metadata():
    from effgen.eval.comparison import ComparisonMatrix, ModelScore

    matrix = ComparisonMatrix(
        scores=[ModelScore(model_name="m", suite_name="math", accuracy=1.0)],
        recommendations={"math": "m"},
        recommendation_rationale={"math": "highest accuracy at 100%"},
        num_cases=4, scoring="contains", suite_sizes={"math": 4},
        generated_at="2026-07-18T14:00:00+00:00",
    )
    doc = matrix.to_dict()
    for key in ("num_cases", "scoring", "suite_sizes", "generated_at",
                "recommendation_rationale"):
        assert key in doc
    assert doc["num_cases"] == 4
    # The rationale also reaches the Markdown output.
    assert "highest accuracy at 100%" in matrix.to_markdown()


def test_recommendation_rationale_states_the_numbers():
    from effgen.eval.comparison import ModelScore, _recommendation_rationale

    cheap = ModelScore(model_name="cheap", suite_name="s", accuracy=1.0,
                       avg_latency=0.3, avg_cost_usd=5e-06)
    dear = ModelScore(model_name="dear", suite_name="s", accuracy=1.0,
                      avg_latency=1.3, avg_cost_usd=4.1e-05)
    why = _recommendation_rationale(cheap, [cheap, dear], "cost")
    assert "cheapest" in why and "$0.000005/run" in why and "dear" in why

    why_latency = _recommendation_rationale(cheap, [cheap, dear], "latency")
    assert "fastest" in why_latency and "0.300s" in why_latency


def test_unpriced_winner_rationale_does_not_claim_a_zero_price():
    from effgen.eval.comparison import ModelScore, _recommendation_rationale

    local = ModelScore(model_name="local", suite_name="s", accuracy=1.0,
                       avg_latency=0.4, avg_cost_usd=None)
    cloud = ModelScore(model_name="cloud", suite_name="s", accuracy=1.0,
                       avg_latency=0.3, avg_cost_usd=5e-06)
    why = _recommendation_rationale(local, [local, cloud], "cost")
    assert "no published per-token price" in why
    assert "$0.000000" not in why
    assert "cloud" in why


def test_tied_rationale_only_claims_a_speed_margin_when_there_is_one():
    from effgen.eval.comparison import ModelScore, _recommendation_rationale

    quick = ModelScore(model_name="quick", suite_name="s", accuracy=1.0,
                       avg_latency=0.09, avg_cost_usd=4e-06)
    slow = ModelScore(model_name="slow", suite_name="s", accuracy=1.0,
                      avg_latency=1.50, avg_cost_usd=4e-05)
    why = _recommendation_rationale(quick, [quick, slow], "accuracy")
    assert "faster at 0.090s/run" in why
    # The latency is stated once, not repeated in a trailing parenthetical.
    assert why.count("0.090s/run") == 1

    # Equal accuracy and equal latency: the speed claim must not be made.
    even_a = ModelScore(model_name="a", suite_name="s", accuracy=1.0,
                        avg_latency=0.5, avg_cost_usd=1e-06)
    even_b = ModelScore(model_name="b", suite_name="s", accuracy=1.0,
                        avg_latency=0.5, avg_cost_usd=2e-06)
    tied_why = _recommendation_rationale(even_a, [even_a, even_b], "accuracy")
    assert "faster" not in tied_why
    assert "tied with 1 other model at 0.500s/run" in tied_why


def test_rationale_ignores_a_model_that_failed_to_run():
    from effgen.eval.comparison import ModelScore, _recommendation_rationale

    winner = ModelScore(model_name="winner", suite_name="s", accuracy=1.0,
                        avg_latency=0.09, avg_cost_usd=4e-06)
    broken = ModelScore(model_name="broken", suite_name="s", accuracy=0.0,
                        avg_latency=0.0, avg_cost_usd=None, error="load failed")
    # A failed model's zeroed metrics must never be quoted as the margin.
    for optimize in ("latency", "cost", "accuracy"):
        why = _recommendation_rationale(winner, [winner, broken], optimize)
        assert "broken" not in why
        assert "0.000s" not in why


@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_report_escapes_markup_from_result_text(kind):
    """Text carried in a result document is rendered as inert text, not markup."""
    payload = '</script><script>alert(1)</script><img src=x onerror=alert(2)>'
    doc = json.loads(json.dumps(DOCS[kind]))
    if kind == "eval":
        doc["suite"] = payload
        doc["results"][0]["query"] = payload
        doc["results"][0]["agent_output"] = payload
    elif kind == "comparison":
        doc["scores"][0]["model"] = payload
    elif kind == "cost":
        doc["rows"][0]["model"] = payload
    else:
        doc["scenario"] = payload
        doc["error_breakdown"] = {payload: 3}

    html_text = build_html_report(doc, kind=kind, command=payload)

    class _Tags(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []
            self.attrs = []

        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)
            self.attrs += [name for name, _ in attrs]

    parser = _Tags()
    parser.feed(html_text)
    # Only the page's own theme script exists; nothing was injected.
    assert parser.tags.count("script") == 1
    assert not [t for t in parser.tags if t in ("img", "iframe", "object", "embed")]
    assert not [a for a in parser.attrs if a.startswith("on")]


def test_suite_results_to_markdown():
    from effgen.eval.evaluator import EvalResult, SuiteResults, TestCase

    results = SuiteResults(
        suite_name="math",
        results=[
            EvalResult(test_case=TestCase(query="2+2 | plus", expected_output="4"),
                       agent_output="4", score=1.0, passed=True, latency=0.5,
                       tokens_used=90, cost_usd=2.6e-05),
            EvalResult(test_case=TestCase(query="7*6", expected_output="42"),
                       agent_output="40", score=0.0, passed=False, latency=0.6,
                       tokens_used=88, cost_usd=None),
        ],
        accuracy=0.5, avg_latency=0.55, total_tokens=178, total_cost_usd=2.6e-05,
        metadata={"scoring": "contains"},
    )
    md = results.to_markdown()
    assert md.startswith("# Evaluation Results — math")
    assert "50.0% (1/2 cases)" in md
    assert "| PASS |" in md and "| FAIL |" in md
    # A case with no published price is labeled, not priced at zero.
    assert "unpriced" in md
    # A pipe inside a query does not break the table.
    assert r"2+2 \| plus" in md
