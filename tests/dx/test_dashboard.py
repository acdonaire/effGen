"""Tests for the local dashboard — static files and server routes.

Validates:
1. Static files exist and contain the required panel IDs.
2. ``/dashboard`` serves HTML containing the panel IDs.
3. ``/dashboard/data.json`` serves valid JSON with the required keys.
4. Dashboard data builder returns the expected structure.
5. Static asset routes work.
6. SSE spans endpoint responds.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

STATIC_DIR = Path(__file__).parents[2] / "effgen" / "dashboard" / "static"

# Required panel IDs that must be present in the HTML
REQUIRED_PANEL_IDS = [
    "summary-cards",
    "panel-slo",
    "panel-latency-chart",
    "panel-agent-runs",
    "panel-spans",
    "panel-metrics",
]

# Required card value IDs
REQUIRED_CARD_IDS = [
    "val-requests",
    "val-errors",
    "val-latency",
    "val-cost",
    "val-tokens",
]


# ---------------------------------------------------------------------------
# Static file checks (no server required)
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_index_html_exists(self):
        assert (STATIC_DIR / "index.html").exists()

    def test_app_js_exists(self):
        assert (STATIC_DIR / "app.js").exists()

    def test_style_css_exists(self):
        assert (STATIC_DIR / "style.css").exists()

    def test_index_contains_panel_ids(self):
        html = (STATIC_DIR / "index.html").read_text()
        for pid in REQUIRED_PANEL_IDS:
            assert pid in html, f"Panel ID '{pid}' missing from index.html"

    def test_index_contains_card_ids(self):
        html = (STATIC_DIR / "index.html").read_text()
        for cid in REQUIRED_CARD_IDS:
            assert cid in html, f"Card ID '{cid}' missing from index.html"

    def test_app_js_polls_data_json(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "/dashboard/data.json" in js

    def test_app_js_has_render_functions(self):
        js = (STATIC_DIR / "app.js").read_text()
        for fn in ("renderCards", "renderSLO", "renderRuns", "renderMetrics"):
            assert fn in js, f"Function '{fn}' missing from app.js"

    def test_app_js_has_sse_support(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "EventSource" in js

    def test_style_css_has_variables(self):
        css = (STATIC_DIR / "style.css").read_text()
        assert "--bg:" in css
        assert "--accent:" in css

    def test_index_loads_chartjs(self):
        html = (STATIC_DIR / "index.html").read_text()
        assert "chart.js" in html.lower() or "Chart.js" in html

    def test_index_contains_auth_banner_element(self):
        html = (STATIC_DIR / "index.html").read_text()
        assert 'id="auth-banner"' in html

    def test_app_js_shows_banner_on_401(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "showAuthBanner" in js
        assert "status === 401" in js

    def test_app_js_hides_banner_on_success(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "hideAuthBanner" in js


# ---------------------------------------------------------------------------
# Dashboard data builder
# ---------------------------------------------------------------------------


class TestDashboardDataBuilder:
    def test_build_dashboard_data_structure(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert isinstance(data, dict)

    def test_required_top_level_keys(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        for key in ("ts", "metrics", "slo", "recent_runs", "recent_spans", "raw_metrics"):
            assert key in data, f"Key '{key}' missing from dashboard data"

    def test_verify_spec_aliases_present(self):
        """The documented ``spans``/``slos`` aliases must also be present."""
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        for key in ("metrics", "spans", "slos", "recent_runs"):
            assert key in data, f"Verify-spec key '{key}' missing from dashboard data"
        # Aliases mirror their canonical keys.
        assert data["slos"] == data["slo"]
        assert data["spans"] == data["recent_spans"]

    def test_buffered_span_appears_in_data(self):
        """A buffered span should surface in recent_spans / spans."""
        from effgen.observability.tracing import _buffer_span, get_recent_spans
        from effgen.server.app import _build_dashboard_data

        _buffer_span("unit.test.span", 1.5)
        assert any(s["name"] == "unit.test.span" for s in get_recent_spans())
        data = _build_dashboard_data()
        assert any(s["name"] == "unit.test.span" for s in data["recent_spans"])

    def test_metrics_sub_keys(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        m = data["metrics"]
        for key in ("total_requests", "total_errors", "avg_latency_s", "total_tokens", "daily_cost_usd"):
            assert key in m, f"Metric key '{key}' missing"

    def test_slo_sub_keys(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        slo = data["slo"]
        for key in ("p99_latency_burn", "error_rate_burn", "availability"):
            assert key in slo, f"SLO key '{key}' missing"

    def test_recent_runs_is_list(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert isinstance(data["recent_runs"], list)

    def test_recent_spans_is_list(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert isinstance(data["recent_spans"], list)

    def test_raw_metrics_is_dict(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert isinstance(data["raw_metrics"], dict)

    def test_ts_is_iso_string(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        ts = data["ts"]
        assert isinstance(ts, str)
        assert "T" in ts  # ISO 8601-ish

    def test_data_is_json_serializable(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        # Should not raise
        serialized = json.dumps(data)
        assert len(serialized) > 2

    def test_run_log_records_appear(self):
        """Records written to run_log should appear in dashboard data."""
        from effgen.observability.run_log import clear, record_run
        from effgen.server.app import _build_dashboard_data

        clear()
        record_run(model="cerebras:llama3.1-8b", duration_s=0.5, input_tokens=10, output_tokens=20)
        data = _build_dashboard_data()
        clear()

        runs = data["recent_runs"]
        assert len(runs) >= 1
        assert runs[0]["model"] == "cerebras:llama3.1-8b"

    def test_agent_run_is_recorded_for_dashboard(self):
        """Agent.run should populate the dashboard run log automatically."""
        from effgen.core.agent import Agent, AgentConfig, AgentResponse
        from effgen.models.base import (
            BaseModel,
            GenerationConfig,
            GenerationResult,
            ModelType,
            TokenCount,
        )
        from effgen.observability.run_log import clear, get_recent_runs

        class DummyModel(BaseModel):
            def __init__(self):
                super().__init__("dummy-model", ModelType.OPENAI)

            def load(self) -> None:
                self._is_loaded = True

            def unload(self) -> None:
                self._is_loaded = False

            def get_context_length(self) -> int:
                return 8192

            def generate(
                self,
                prompt: str,
                config: GenerationConfig | None = None,
                **kwargs,
            ) -> GenerationResult:
                return GenerationResult(
                    text="Final Answer: ok",
                    tokens_used=3,
                    finish_reason="stop",
                    model_name=self.model_name,
                )

            def generate_stream(
                self,
                prompt: str,
                config: GenerationConfig | None = None,
                **kwargs,
            ):
                yield "ok"

            def count_tokens(self, text: str) -> TokenCount:
                return TokenCount(count=len(text.split()), model_name=self.model_name)

        clear()
        agent = Agent(AgentConfig(name="dashboard_test", model=DummyModel()))
        with patch.object(
            Agent,
            "_run_single_agent",
            return_value=AgentResponse(output="ok", tokens_used=7, metadata={"prompt_tokens": 4}),
        ):
            agent.run("hello")
        runs = get_recent_runs()
        clear()

        assert runs
        assert runs[0]["model"] == "dummy-model"
        assert runs[0]["input_tokens"] == 4
        assert runs[0]["output_tokens"] == 7

    def test_effgen_native_metrics_are_summarized(self):
        from effgen.observability.metrics import record_model_call, record_tokens, reset_all
        from effgen.server.app import _build_dashboard_data

        reset_all()
        record_model_call(provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.5)
        record_model_call(provider="cerebras", model="llama3.1-8b", outcome="error", latency=1.5)
        record_tokens(provider="cerebras", model="llama3.1-8b", input_tokens=10, output_tokens=5)
        data = _build_dashboard_data()
        reset_all()

        assert data["metrics"]["total_requests"] >= 2
        assert data["metrics"]["total_errors"] >= 1
        assert data["metrics"]["total_tokens"] >= 15
        assert data["metrics"]["avg_latency_s"] == pytest.approx(1.0)

    def test_prompt_templates_exposed_for_editor_integrations(self):
        from effgen.server.app import _build_dashboard_data

        data = _build_dashboard_data()
        assert "prompt_templates" in data
        assert isinstance(data["prompt_templates"], list)
        if data["prompt_templates"]:
            first = data["prompt_templates"][0]
            for key in ("name", "description", "template", "category"):
                assert key in first


# ---------------------------------------------------------------------------
# FastAPI server route tests (TestClient)
# ---------------------------------------------------------------------------


def _make_dev_app():
    """Create a dev-mode FastAPI app for testing."""
    try:
        from effgen.server.app import create_app

        return create_app(dev_mode=True)
    except Exception as exc:
        pytest.skip(f"Cannot create test app: {exc}")


class TestDashboardRoutes:
    @pytest.fixture(autouse=True)
    def client(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        app = _make_dev_app()
        self._client = TestClient(app, raise_server_exceptions=False)
        return self._client

    def test_dashboard_index_200(self):
        resp = self._client.get("/dashboard")
        assert resp.status_code == 200

    def test_dashboard_index_html_content_type(self):
        resp = self._client.get("/dashboard")
        assert "html" in resp.headers.get("content-type", "")

    def test_dashboard_index_contains_panel_ids(self):
        resp = self._client.get("/dashboard")
        html = resp.text
        for pid in REQUIRED_PANEL_IDS:
            assert pid in html, f"Panel ID '{pid}' not in served HTML"

    def test_dashboard_slash_redirects_or_200(self):
        resp = self._client.get("/dashboard/")
        assert resp.status_code in (200, 301, 307, 308)

    def test_dashboard_data_json_200(self):
        resp = self._client.get("/dashboard/data.json")
        assert resp.status_code == 200

    def test_dashboard_data_json_content_type(self):
        resp = self._client.get("/dashboard/data.json")
        assert "json" in resp.headers.get("content-type", "")

    def test_dashboard_data_json_valid(self):
        resp = self._client.get("/dashboard/data.json")
        data = resp.json()
        assert isinstance(data, dict)
        assert "metrics" in data
        assert "slo" in data

    def test_dashboard_data_json_required_keys(self):
        resp = self._client.get("/dashboard/data.json")
        data = resp.json()
        for key in ("ts", "metrics", "slo", "recent_runs", "recent_spans", "raw_metrics"):
            assert key in data

    def test_dashboard_static_app_js(self):
        resp = self._client.get("/dashboard/app.js")
        assert resp.status_code == 200

    def test_dashboard_static_style_css(self):
        resp = self._client.get("/dashboard/style.css")
        assert resp.status_code == 200

    def test_dashboard_spans_sse(self, monkeypatch):
        """SSE endpoint should return 200 with text/event-stream content type
        and terminate promptly once the client disconnects (no hang)."""
        # Short heartbeat so the disconnect-check loop exits quickly when the
        # stream context closes (guards against the unbounded-loop regression).
        monkeypatch.setenv("EFFGEN_DASHBOARD_SSE_HEARTBEAT_SECONDS", "0.05")
        monkeypatch.setenv("EFFGEN_DASHBOARD_SSE_MAX_SECONDS", "2")

        # Buffer a span so the stream has immediate data to emit.
        from effgen.observability.tracing import _buffer_span

        _buffer_span("test.span", 12.3)

        with self._client.stream("GET", "/dashboard/spans") as resp:
            assert resp.status_code == 200
            ct = resp.headers.get("content-type", "")
            assert "text/event-stream" in ct or "event-stream" in ct
            # Read the first buffered span line, then disconnect.
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    assert "test.span" in line
                    break


class TestDashboardAuth:
    def test_dashboard_lookalike_path_is_not_public(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from effgen.server.app import create_app

        client = TestClient(create_app(dev_mode=False), raise_server_exceptions=False)
        resp = client.get("/dashboardevil")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# `effgen serve` startup banner + --help: the dashboard gap must be
# discoverable without opening a browser and hitting a 401.
# ---------------------------------------------------------------------------

class TestServeDashboardDiscoverability:
    def _run_serve(self, monkeypatch, capsys, **env):
        from types import SimpleNamespace

        import effgen.cli._main as _main

        for var in ("EFFGEN_API_KEY", "EFFGEN_DEV_MODE", "EFFGEN_PUBLIC_DASHBOARD"):
            monkeypatch.delenv(var, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

        args = SimpleNamespace(host="127.0.0.1", port=8000, rate_limit=None, verbose=False)
        cli = _main.CLIInterface()
        rc = cli.serve_api(args)
        assert rc == 0
        return capsys.readouterr().out

    def test_dashboard_line_flags_auth_by_default(self, monkeypatch, capsys):
        pytest.importorskip("fastapi")
        out = self._run_serve(monkeypatch, capsys)
        assert "/dashboard" in out
        assert "EFFGEN_PUBLIC_DASHBOARD" in out

    def test_dashboard_line_omits_flag_when_public(self, monkeypatch, capsys):
        pytest.importorskip("fastapi")
        out = self._run_serve(monkeypatch, capsys, EFFGEN_PUBLIC_DASHBOARD="1")
        dashboard_lines = [ln for ln in out.splitlines() if "/dashboard" in ln and "v1" not in ln.lower()]
        assert dashboard_lines
        assert "EFFGEN_PUBLIC_DASHBOARD" not in dashboard_lines[0]

    def test_serve_help_documents_public_dashboard_var(self):
        import subprocess
        import sys
        from pathlib import Path

        scripts = Path(sys.executable).parent
        effgen_bin = str(scripts / "effgen") if (scripts / "effgen").exists() else "effgen"
        result = subprocess.run(
            [effgen_bin, "serve", "--help"], capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "EFFGEN_PUBLIC_DASHBOARD" in result.stdout
