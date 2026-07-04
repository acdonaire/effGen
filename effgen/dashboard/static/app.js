/**
 * effGen Dashboard — frontend app
 *
 * Polls /dashboard/data.json every 5 seconds and renders:
 *  - summary cards (requests, errors, latency, cost, tokens)
 *  - SLO burn-rate bars
 *  - latency trend chart (Chart.js)
 *  - recent agent runs table
 *  - live span stream (SSE from /dashboard/spans if available)
 *  - raw Prometheus metrics table
 */

(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Constants
  // ------------------------------------------------------------------
  const POLL_MS = 5000;
  const MAX_SPANS = 200;
  const MAX_CHART_POINTS = 30;

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  let latencyChart = null;
  let latencyHistory = [];
  let spanCount = 0;
  let spanPaused = false;
  let eventSource = null;

  // ------------------------------------------------------------------
  // DOM helpers
  // ------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }

  function setText(id, text) {
    const el = $(id);
    if (el) el.textContent = text;
  }

  function setStatus(online) {
    const dot = $("status-dot");
    const txt = $("status-text");
    if (dot) { dot.className = "status-dot " + (online ? "ok" : "err"); }
    if (txt) { txt.textContent = online ? "Connected" : "Offline"; }
  }

  function showAuthBanner() {
    const el = $("auth-banner");
    if (!el) return;
    el.textContent = "Dashboard data requires authentication. Restart the server with "
      + "EFFGEN_PUBLIC_DASHBOARD=1 for local viewing, or supply an API key "
      + "(Authorization: Bearer <key> or X-API-Key: <key>).";
    el.hidden = false;
  }

  function hideAuthBanner() {
    const el = $("auth-banner");
    if (el) el.hidden = true;
  }

  // ------------------------------------------------------------------
  // Chart initialisation
  // ------------------------------------------------------------------
  function initChart() {
    const ctx = $("latency-chart");
    if (!ctx || !window.Chart) return;
    latencyChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [{
          label: "avg latency (s)",
          data: [],
          borderColor: "#6366f1",
          backgroundColor: "rgba(99,102,241,0.15)",
          borderWidth: 2,
          tension: 0.35,
          pointRadius: 2,
        }],
      },
      options: {
        animation: false,
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: {
            beginAtZero: true,
            ticks: { color: "#8892a4" },
            grid:  { color: "#1e2236" },
          },
        },
      },
    });
  }

  function pushLatency(ts, value) {
    latencyHistory.push({ ts, value });
    if (latencyHistory.length > MAX_CHART_POINTS) {
      latencyHistory.shift();
    }
    if (!latencyChart) return;
    latencyChart.data.labels = latencyHistory.map(p => p.ts);
    latencyChart.data.datasets[0].data = latencyHistory.map(p => p.value);
    latencyChart.update();
  }

  // ------------------------------------------------------------------
  // Render helpers
  // ------------------------------------------------------------------
  function renderCards(data) {
    const m = data.metrics || {};
    setText("val-requests", fmt(m.total_requests));
    setText("val-errors",   fmt(m.total_errors));
    setText("val-latency",  m.avg_latency_s != null ? m.avg_latency_s.toFixed(3) + "s" : "—");
    setText("val-cost",     m.daily_cost_usd != null ? "$" + m.daily_cost_usd.toFixed(4) : "—");
    setText("val-tokens",   fmt(m.total_tokens));

    const ts = new Date().toLocaleTimeString([], { hour12: false });
    if (m.avg_latency_s != null) {
      pushLatency(ts, m.avg_latency_s);
    }
  }

  function fmt(v) {
    if (v == null) return "—";
    if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(v);
  }

  function renderSLO(data) {
    const slo = data.slo || {};

    // p99 latency (burn rate 0→1, 1 = at threshold)
    const p99 = Math.min((slo.p99_latency_burn || 0), 1);
    setSLOBar("slo-p99", "slo-p99-pct", p99);

    // error rate (burn rate 0→1)
    const errRate = Math.min((slo.error_rate_burn || 0), 1);
    setSLOBar("slo-err", "slo-err-pct", errRate);

    // availability (0→1, 1 = fully available)
    const avail = Math.min((slo.availability || 1), 1);
    setSLOBar("slo-avail", "slo-avail-pct", avail, true);
  }

  function setSLOBar(barId, pctId, ratio, invert) {
    const bar = $(barId);
    const pct = $(pctId);
    const percent = (ratio * 100).toFixed(1) + "%";
    if (bar) bar.style.width = percent;
    if (pct) pct.textContent = percent;
    if (bar && !invert) {
      bar.style.background = ratio > 0.9 ? "#ef4444" : ratio > 0.5 ? "#f59e0b" : "#6366f1";
    }
  }

  function renderRuns(data) {
    const runs = data.recent_runs || [];
    const tbody = $("runs-tbody");
    if (!tbody) return;
    if (!runs.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No runs yet</td></tr>';
      return;
    }
    tbody.innerHTML = runs.map(r => `
      <tr>
        <td>${r.ts || "—"}</td>
        <td>${esc(r.model || "—")}</td>
        <td>${fmt(r.input_tokens)} / ${fmt(r.output_tokens)}</td>
        <td>${r.cost_usd != null ? "$" + r.cost_usd.toFixed(5) : "—"}</td>
        <td>${r.duration_s != null ? r.duration_s.toFixed(3) + "s" : "—"}</td>
        <td><span class="badge ${r.error ? "badge-err" : "badge-ok"}">${r.error ? "error" : "ok"}</span></td>
      </tr>`).join("");
  }

  function renderMetrics(data) {
    const raw = data.raw_metrics || {};
    const tbody = $("metrics-tbody");
    if (!tbody) return;
    const entries = Object.entries(raw);
    if (!entries.length) {
      tbody.innerHTML = '<tr><td colspan="2" class="empty-row">No metrics</td></tr>';
      return;
    }
    tbody.innerHTML = entries
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${typeof v === "number" ? v.toLocaleString() : esc(String(v))}</td></tr>`)
      .join("");
  }

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // ------------------------------------------------------------------
  // Span stream
  // ------------------------------------------------------------------
  function appendSpan(span) {
    if (spanPaused) return;
    const stream = $("span-stream");
    if (!stream) return;

    spanCount++;
    setText("span-count", spanCount + " spans");

    const el = document.createElement("div");
    el.className = "span-entry";
    const ts   = span.ts ? `<span class="span-ts">${esc(span.ts)}</span> ` : "";
    const name = `<span class="span-name">${esc(span.name || "span")}</span>`;
    const dur  = span.duration_ms != null
      ? ` <span class="span-dur">${span.duration_ms.toFixed(1)}ms</span>` : "";
    const err  = span.error ? ` <span class="span-err">[${esc(span.error)}]</span>` : "";
    el.innerHTML = `${ts}${name}${dur}${err}`;
    stream.appendChild(el);

    // Trim old spans
    while (stream.children.length > MAX_SPANS) {
      stream.removeChild(stream.firstChild);
    }
    stream.scrollTop = stream.scrollHeight;
  }

  function startSSE() {
    if (eventSource) return;
    try {
      eventSource = new EventSource("/dashboard/spans");
      eventSource.onmessage = (e) => {
        try { appendSpan(JSON.parse(e.data)); } catch { /* ignore */ }
      };
      eventSource.onerror = () => {
        eventSource.close();
        eventSource = null;
        // Retry after 10s
        setTimeout(startSSE, 10000);
      };
    } catch {
      // SSE not supported or path not found
    }
  }

  // ------------------------------------------------------------------
  // Data polling
  // ------------------------------------------------------------------
  async function fetchData() {
    try {
      const resp = await fetch("/dashboard/data.json", { cache: "no-store" });
      if (resp.status === 401) {
        setStatus(false);
        showAuthBanner();
        return;
      }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const data = await resp.json();
      hideAuthBanner();
      setStatus(true);
      renderCards(data);
      renderSLO(data);
      renderRuns(data);
      renderMetrics(data);
      // Inject demo spans if none came in via SSE
      if (!eventSource && data.recent_spans) {
        data.recent_spans.forEach(appendSpan);
      }
    } catch (err) {
      setStatus(false);
      console.warn("[effGen dashboard] fetch failed:", err);
    }
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  function init() {
    initChart();

    // Wire up buttons
    const refreshBtn = $("refresh-btn");
    if (refreshBtn) refreshBtn.addEventListener("click", fetchData);

    const clearBtn = $("span-clear-btn");
    if (clearBtn) clearBtn.addEventListener("click", () => {
      const s = $("span-stream");
      if (s) s.innerHTML = "";
      spanCount = 0;
      setText("span-count", "0 spans");
    });

    const pauseCb = $("span-pause-cb");
    if (pauseCb) pauseCb.addEventListener("change", () => { spanPaused = pauseCb.checked; });

    // Initial load + poll
    fetchData();
    setInterval(fetchData, POLL_MS);

    // Try SSE
    startSSE();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
