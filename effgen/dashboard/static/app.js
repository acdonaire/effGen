/**
 * effGen Dashboard — frontend app
 *
 * Polls /dashboard/data.json every 5 seconds and renders:
 *  - summary cards (requests, model-call errors, latency, real cost, tokens)
 *  - SLO burn-rate bars (true p99 from the latency histogram)
 *  - a latency trend chart drawn on a canvas (no external chart library)
 *  - per-model breakdown (calls, error rate, p95, tokens, cost)
 *  - HTTP responses by status code
 *  - recent agent runs table
 *  - live span stream (SSE from /dashboard/spans if available)
 *  - raw Prometheus metrics table
 *
 * All assets are served locally; the page has no external network dependency.
 */

(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Constants
  // ------------------------------------------------------------------
  const POLL_MS = 5000;
  const MAX_SPANS = 200;
  const MAX_CHART_POINTS = 30;
  const THEME_KEY = "effgen-dashboard-theme";

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
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

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function fmt(v) {
    if (v == null) return "—";
    if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(v);
  }

  function fmtCost(v) {
    if (v == null) return "—";
    if (v === 0) return "$0.00";
    if (v < 0.01) return "$" + v.toFixed(6);
    return "$" + v.toFixed(4);
  }

  function fmtSeconds(v) {
    return v == null ? "—" : v.toFixed(3) + "s";
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // ------------------------------------------------------------------
  // Theme
  // ------------------------------------------------------------------
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const icon = $("theme-icon");
    if (icon) icon.textContent = theme === "dark" ? "☾" : "☀";
    const btn = $("theme-btn");
    if (btn) btn.setAttribute("aria-label",
      theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
    drawChart();
  }

  function initTheme() {
    let stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch { /* storage blocked */ }
    if (stored !== "dark" && stored !== "light") {
      const prefersLight = window.matchMedia
        && window.matchMedia("(prefers-color-scheme: light)").matches;
      stored = prefersLight ? "light" : "dark";
    }
    applyTheme(stored);
    const btn = $("theme-btn");
    if (btn) btn.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark"
        ? "light" : "dark";
      try { localStorage.setItem(THEME_KEY, next); } catch { /* storage blocked */ }
      applyTheme(next);
    });
  }

  // ------------------------------------------------------------------
  // Latency chart (canvas 2D — no external library)
  // ------------------------------------------------------------------
  function pushLatency(ts, value) {
    latencyHistory.push({ ts, value });
    if (latencyHistory.length > MAX_CHART_POINTS) {
      latencyHistory.shift();
    }
    drawChart();
  }

  function drawChart() {
    const canvas = $("latency-chart");
    const empty = $("chart-empty");
    if (!canvas || !canvas.getContext) return;

    const points = latencyHistory.map(p => p.value).filter(v => v != null);
    if (empty) empty.hidden = points.length > 0;
    if (!points.length) {
      const ctx0 = canvas.getContext("2d");
      ctx0.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    // Scale the backing store to device pixels for a crisp line.
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    const cssH = parseInt(canvas.getAttribute("height"), 10) || 150;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const padL = 44, padR = 8, padT = 10, padB = 20;
    const plotW = cssW - padL - padR;
    const plotH = cssH - padT - padB;
    const maxV = Math.max.apply(null, points) * 1.15 || 1;
    const n = points.length;

    const grid = cssVar("--border") || "#2c3150";
    const axis = cssVar("--text-muted") || "#8892a4";
    const line = cssVar("--accent") || "#818cf8";

    // Horizontal grid lines + y labels
    ctx.strokeStyle = grid;
    ctx.fillStyle = axis;
    ctx.lineWidth = 1;
    ctx.font = "10px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    const rows = 4;
    for (let i = 0; i <= rows; i++) {
      const y = padT + (plotH * i) / rows;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      const val = maxV * (1 - i / rows);
      ctx.fillText(val.toFixed(2) + "s", 4, y);
    }

    const xFor = (i) => n <= 1 ? padL + plotW : padL + (plotW * i) / (n - 1);
    const yFor = (v) => padT + plotH * (1 - v / maxV);

    // Filled area under the curve
    ctx.beginPath();
    ctx.moveTo(xFor(0), yFor(points[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(xFor(i), yFor(points[i]));
    ctx.lineTo(xFor(n - 1), padT + plotH);
    ctx.lineTo(xFor(0), padT + plotH);
    ctx.closePath();
    ctx.fillStyle = hexToRgba(line, 0.15);
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(xFor(0), yFor(points[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(xFor(i), yFor(points[i]));
    ctx.strokeStyle = line;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Points
    ctx.fillStyle = line;
    for (let i = 0; i < n; i++) {
      ctx.beginPath();
      ctx.arc(xFor(i), yFor(points[i]), 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function hexToRgba(hex, alpha) {
    const h = hex.replace("#", "");
    if (h.length !== 6) return "rgba(129,140,248," + alpha + ")";
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // ------------------------------------------------------------------
  // Render helpers
  // ------------------------------------------------------------------
  function renderCards(data) {
    const m = data.metrics || {};
    setText("val-requests", fmt(m.total_requests));
    setText("val-errors",   fmt(m.total_errors));
    setText("val-latency",  fmtSeconds(m.avg_latency_s));
    setText("val-tokens",   fmt(m.total_tokens));

    // Real cost — sum of per-run cost_usd; never a fabricated flat rate.
    if (m.cost_usd == null) {
      setText("val-cost", "—");
      setText("sub-cost", (m.unpriced_runs ? m.unpriced_runs + " unpriced run(s)" : "no priced runs"));
    } else {
      setText("val-cost", fmtCost(m.cost_usd));
      const parts = [];
      if (m.priced_runs) parts.push(m.priced_runs + " priced");
      if (m.unpriced_runs) parts.push(m.unpriced_runs + " unpriced");
      setText("sub-cost", parts.join(" · "));
    }

    // HTTP error context for the errors card.
    const http4 = m.http_client_errors || 0;
    const http5 = m.http_server_errors || 0;
    setText("sub-errors", (http4 || http5)
      ? `HTTP ${http4} 4xx · ${http5} 5xx` : "");

    const version = data.version;
    if (version) setText("header-version", "v" + version);

    const ts = new Date().toLocaleTimeString([], { hour12: false });
    if (m.avg_latency_s != null) {
      pushLatency(ts, m.avg_latency_s);
    }
  }

  function renderSLO(data) {
    const slo = data.slo || {};

    // p99 latency burn (0→1, 1 = at the latency threshold), driven by the true p99.
    const p99 = Math.min((slo.p99_latency_burn || 0), 1);
    setSLOBar("slo-p99", "slo-p99-pct", "slo-p99-img", p99, false, "p99 latency burn");

    const errRate = Math.min((slo.error_rate_burn || 0), 1);
    setSLOBar("slo-err", "slo-err-pct", "slo-err-img", errRate, false, "error rate burn");

    const avail = Math.min((slo.availability != null ? slo.availability : 1), 1);
    setSLOBar("slo-avail", "slo-avail-pct", "slo-avail-img", avail, true, "availability");

    const detail = [];
    if (slo.p50_latency_s != null) detail.push("p50 " + slo.p50_latency_s.toFixed(2) + "s");
    if (slo.p95_latency_s != null) detail.push("p95 " + slo.p95_latency_s.toFixed(2) + "s");
    if (slo.p99_latency_s != null) detail.push("p99 " + slo.p99_latency_s.toFixed(2) + "s");
    if (slo.latency_threshold_s != null) detail.push("target " + slo.latency_threshold_s + "s");
    setText("slo-detail", detail.length ? detail.join("  ·  ") : "No latency samples yet.");
  }

  function setSLOBar(barId, pctId, imgId, ratio, invert, label) {
    const bar = $(barId);
    const pct = $(pctId);
    const img = $(imgId);
    const percent = (ratio * 100).toFixed(1) + "%";
    if (bar) bar.style.width = percent;
    if (pct) pct.textContent = percent;
    if (img) img.setAttribute("aria-label", label + ": " + percent);
    if (bar && !invert) {
      bar.style.background = ratio > 0.9
        ? cssVar("--err") : ratio > 0.5 ? cssVar("--warn") : cssVar("--accent");
    }
  }

  function renderByModel(data) {
    const rows = data.by_model || [];
    const tbody = $("by-model-tbody");
    if (!tbody) return;
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty-row">No model calls yet</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td>${esc(r.model || "—")}</td>
        <td>${esc(r.provider || "—")}</td>
        <td class="num">${fmt(r.calls)}</td>
        <td class="num">${(r.error_rate != null ? (r.error_rate * 100).toFixed(1) + "%" : "—")}</td>
        <td class="num">${fmtSeconds(r.p95_latency_s)}</td>
        <td class="num">${fmt(r.input_tokens)} / ${fmt(r.output_tokens)}</td>
        <td class="num">${fmtCost(r.cost_usd)}</td>
      </tr>`).join("");
  }

  function renderByStatus(data) {
    const chips = $("status-chips");
    if (!chips) return;
    const byStatus = data.by_status || {};
    const codes = Object.keys(byStatus);
    if (!codes.length) {
      chips.innerHTML = '<span class="empty-row">No HTTP responses recorded yet</span>';
      return;
    }
    chips.innerHTML = codes.sort().map(code => {
      const cls = "s-" + (code.charAt(0) || "2") + "xx";
      return `<span class="status-chip ${cls}">`
        + `<span class="chip-code">${esc(code)}</span>`
        + `<span class="chip-count">${fmt(byStatus[code])}</span></span>`;
    }).join("");
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
        <td>${esc(r.ts || "—")}</td>
        <td>${esc(r.model || "—")}</td>
        <td>${fmt(r.input_tokens)} / ${fmt(r.output_tokens)}</td>
        <td>${fmtCost(r.cost_usd)}</td>
        <td>${fmtSeconds(r.duration_s)}</td>
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

  // ------------------------------------------------------------------
  // Span stream
  // ------------------------------------------------------------------
  function spanKind(name) {
    const n = String(name || "");
    if (n.indexOf("model.call") !== -1) return { kind: "model", label: n.replace(/^effgen\.model\.call\s*/, "") };
    if (n.indexOf("tool.call") !== -1) return { kind: "tool", label: n.replace(/^effgen\.tool\.call\s*/, "") };
    if (n.indexOf("router") !== -1) return { kind: "router", label: n.replace(/^effgen\.router\.\w+\s*/, "") };
    if (n.indexOf("agent.run") !== -1) return { kind: "run", label: n.replace(/^effgen\.agent\.run\s*/, "") };
    if (n.indexOf("agent.iteration") !== -1) return { kind: "iter", label: n };
    return { kind: "span", label: n };
  }

  function appendSpan(span) {
    if (spanPaused) return;
    const stream = $("span-stream");
    if (!stream) return;

    spanCount++;
    setText("span-count", spanCount + " spans");

    const parsed = spanKind(span.name);
    const el = document.createElement("div");
    el.className = "span-entry" + (span.error ? " is-error" : "");
    const ts   = span.ts ? `<span class="span-ts">${esc(span.ts)}</span>` : "";
    const kind = `<span class="span-kind">${esc(parsed.kind)}</span>`;
    const name = `<span class="span-name">${esc(parsed.label || "span")}</span>`;
    const dur  = span.duration_ms != null
      ? `<span class="span-dur">${span.duration_ms.toFixed(1)}ms</span>` : "";
    const err  = span.error ? `<span class="span-err">[${esc(span.error)}]</span>` : "";
    el.innerHTML = `${ts}${kind}${name}${dur}${err}`;
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
      renderByModel(data);
      renderByStatus(data);
      renderRuns(data);
      renderMetrics(data);
      // Seed the span stream from the payload if SSE has not connected.
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
    initTheme();

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

    window.addEventListener("resize", drawChart);

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
