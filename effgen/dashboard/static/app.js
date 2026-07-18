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
  // Spans grouped by run id for the per-run waterfall. Insertion order is the
  // order runs first appeared; the newest few are drawn.
  const runs = new Map();
  const seenSpans = new Set();
  const MAX_RUNS = 8;

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
  // The effective theme: an explicit choice (data-theme attribute) if set,
  // otherwise the OS preference the CSS is already following.
  function currentTheme() {
    const attr = document.documentElement.getAttribute("data-theme");
    if (attr === "dark" || attr === "light") return attr;
    const prefersLight = window.matchMedia
      && window.matchMedia("(prefers-color-scheme: light)").matches;
    return prefersLight ? "light" : "dark";
  }

  function syncThemeButton() {
    const theme = currentTheme();
    const icon = $("theme-icon");
    if (icon) icon.textContent = theme === "dark" ? "☾" : "☀";
    const btn = $("theme-btn");
    if (btn) btn.setAttribute("aria-label",
      theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }

  // Set an explicit theme (overriding the OS scheme). Persisted only on a click.
  function applyTheme(theme, persist) {
    document.documentElement.setAttribute("data-theme", theme);
    if (persist) {
      try { localStorage.setItem(THEME_KEY, theme); } catch { /* storage blocked */ }
    }
    syncThemeButton();
    drawChart();
  }

  function initTheme() {
    let stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch { /* storage blocked */ }
    if (stored === "dark" || stored === "light") {
      // Honor the user's saved choice.
      applyTheme(stored, false);
    } else {
      // No stored choice: leave first paint to the CSS/OS scheme, just label
      // the toggle to match. Follow live OS-scheme changes until a choice is made.
      syncThemeButton();
      if (window.matchMedia) {
        window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
          if (!document.documentElement.getAttribute("data-theme")) {
            syncThemeButton();
            drawChart();
          }
        });
      }
    }
    const btn = $("theme-btn");
    if (btn) btn.addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      applyTheme(next, true);
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
  // Model catalog browser
  // ------------------------------------------------------------------
  const CAT_PAGE_SIZE = 25;
  let catalogAll = [];        // every record from /dashboard/catalog.json
  let catalogView = [];       // filtered + sorted
  let catalogPage = 0;
  let catalogLoaded = false;

  function fmtPrice(v) {
    // A published nonzero rate shows as $<n>; null is unknown; 0 with no rate
    // is unpriced rather than a fabricated $0 (mirrors the CLI label).
    if (v == null) return null;
    if (v === 0) return null;
    return "$" + (Math.round(v * 1000) / 1000);
  }

  function priceCells(rec) {
    const pin = fmtPrice(rec.price_in_per_1m);
    const pout = fmtPrice(rec.price_out_per_1m);
    if (pin || pout) return [pin || "?", pout || "?"];
    const label = rec.free_tier ? "free" : (rec.is_priced ? "metered" : "unpriced");
    return [label, label];
  }

  function catalogFilterSort() {
    const q = ($("cat-search").value || "").toLowerCase().trim();
    const prov = $("cat-provider").value || "";
    const sort = $("cat-sort").value || "provider";
    const wantTools = $("cat-tools").checked;
    const wantVision = $("cat-vision").checked;
    const wantAudio = $("cat-audio").checked;
    const wantFree = $("cat-free").checked;
    const minCtx = parseInt($("cat-min-context").value, 10);

    let rows = catalogAll.filter((r) => {
      if (prov && r.provider !== prov) return false;
      if (wantTools && !r.supports_tools) return false;
      if (wantVision && !r.supports_vision) return false;
      if (wantAudio && !r.supports_audio) return false;
      if (wantFree && !r.free_tier) return false;
      if (!isNaN(minCtx) && (r.context_window || 0) < minCtx) return false;
      if (q) {
        const hay = (r.id + " " + (r.family || "") + " " + r.provider).toLowerCase();
        if (hay.indexOf(q) === -1) return false;
      }
      return true;
    });

    const big = Number.POSITIVE_INFINITY;
    const numIn = (r) => (r.price_in_per_1m == null ? big : r.price_in_per_1m);
    const numOut = (r) => (r.price_out_per_1m == null ? big : r.price_out_per_1m);
    const cmp = {
      provider: (a, b) => (a.provider + a.id).localeCompare(b.provider + b.id),
      id: (a, b) => a.id.localeCompare(b.id),
      context: (a, b) => (a.context_window || 0) - (b.context_window || 0),
      "max-out": (a, b) => (a.max_output || 0) - (b.max_output || 0),
      "price-in": (a, b) => numIn(a) - numIn(b),
      "price-out": (a, b) => numOut(a) - numOut(b),
    }[sort] || ((a, b) => 0);
    rows.sort(cmp);

    catalogView = rows;
    catalogPage = 0;
    renderCatalogPage();
  }

  function renderCatalogPage() {
    const tbody = $("catalog-tbody");
    if (!tbody) return;
    const total = catalogView.length;
    const pages = Math.max(1, Math.ceil(total / CAT_PAGE_SIZE));
    if (catalogPage >= pages) catalogPage = pages - 1;
    const start = catalogPage * CAT_PAGE_SIZE;
    const slice = catalogView.slice(start, start + CAT_PAGE_SIZE);

    if (!slice.length) {
      tbody.innerHTML = '<tr><td colspan="9" class="empty-row">'
        + (catalogAll.length ? "No models match those filters"
                             : "Catalog unavailable") + "</td></tr>";
    } else {
      tbody.innerHTML = slice.map((r) => {
        const [pin, pout] = priceCells(r);
        const check = (on) => on ? '<span class="cat-yes" aria-label="yes">✓</span>' : "";
        return `<tr>
          <td>${esc(r.provider)}</td>
          <td class="cat-id">${esc(r.id)}</td>
          <td class="num">${r.context_window ? r.context_window.toLocaleString() : "—"}</td>
          <td class="num">${r.max_output ? r.max_output.toLocaleString() : "—"}</td>
          <td class="num">${esc(pin)}</td>
          <td class="num">${esc(pout)}</td>
          <td>${check(r.supports_tools)}</td>
          <td>${check(r.supports_vision)}</td>
          <td>${check(r.free_tier)}</td>
        </tr>`;
      }).join("");
    }

    const from = total ? start + 1 : 0;
    const to = Math.min(start + CAT_PAGE_SIZE, total);
    setText("cat-page-info", total
      ? `${from}–${to} of ${total} (of ${catalogAll.length})`
      : `0 of ${catalogAll.length}`);
    const prev = $("cat-prev");
    const next = $("cat-next");
    if (prev) prev.disabled = catalogPage <= 0;
    if (next) next.disabled = catalogPage >= pages - 1;
  }

  function populateProviderSelect() {
    const sel = $("cat-provider");
    if (!sel) return;
    const provs = Array.from(new Set(catalogAll.map((r) => r.provider))).sort();
    // Keep the leading "all" option; append the discovered providers once.
    provs.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p;
      sel.appendChild(opt);
    });
  }

  async function loadCatalog() {
    if (catalogLoaded) return;
    try {
      const resp = await fetch("/dashboard/catalog.json", { cache: "no-store" });
      if (resp.status === 401) {
        setText("catalog-sub", "Catalog requires authentication (see the banner above).");
        return;
      }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const data = await resp.json();
      catalogAll = (data.data || []).slice();
      catalogLoaded = true;
      populateProviderSelect();
      const provCount = (data.providers || []).length;
      const verified = (data.providers || [])
        .map((p) => p.verified_on).filter(Boolean).sort();
      const asOf = verified.length ? verified[0] : "unknown";
      setText("catalog-sub",
        `${catalogAll.length} models across ${provCount} providers · `
        + `pricing from catalog snapshot (verified ${asOf}) · `
        + `run “effgen models refresh” to update`);
      catalogFilterSort();
    } catch (err) {
      setText("catalog-sub", "Catalog unavailable.");
      console.warn("[effGen dashboard] catalog fetch failed:", err);
    }
  }

  function initCatalog() {
    const ids = ["cat-search", "cat-provider", "cat-sort", "cat-tools",
                 "cat-vision", "cat-audio", "cat-free", "cat-min-context"];
    ids.forEach((id) => {
      const el = $(id);
      if (!el) return;
      const evt = (el.tagName === "INPUT" && (el.type === "search" || el.type === "number"))
        ? "input" : "change";
      el.addEventListener(evt, catalogFilterSort);
    });
    const prev = $("cat-prev");
    if (prev) prev.addEventListener("click", () => {
      if (catalogPage > 0) { catalogPage--; renderCatalogPage(); }
    });
    const next = $("cat-next");
    if (next) next.addEventListener("click", () => {
      catalogPage++; renderCatalogPage();
    });
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

    collectForWaterfall(span);
  }

  // ------------------------------------------------------------------
  // Per-run waterfall
  // ------------------------------------------------------------------
  function spanSignature(span) {
    return [span.run_id, span.name, span.offset_ms, span.duration_ms].join("|");
  }

  function collectForWaterfall(span) {
    const runId = span.run_id;
    if (!runId) return;
    // A run that streams live may replay buffered spans on reconnect; dedupe.
    const sig = spanSignature(span);
    if (seenSpans.has(sig)) return;
    seenSpans.add(sig);

    if (!runs.has(runId)) {
      runs.set(runId, { id: runId, ts: span.ts, spans: [] });
      // Bound memory: drop the oldest run once we exceed the cap.
      while (runs.size > MAX_RUNS) {
        const oldest = runs.keys().next().value;
        runs.delete(oldest);
      }
    }
    runs.get(runId).spans.push({
      kind: spanKind(span.name).kind,
      label: spanKind(span.name).label || span.name,
      offset: Number(span.offset_ms) || 0,
      duration: Number(span.duration_ms) || 0,
      error: !!span.error,
    });
    renderWaterfall();
  }

  function renderWaterfall() {
    const host = $("waterfall");
    if (!host) return;
    const list = Array.from(runs.values()).reverse();  // newest first
    if (!list.length) {
      host.innerHTML = '<p class="empty-row" id="waterfall-empty">No runs recorded yet.</p>';
      return;
    }
    const rows = list.map((run) => {
      // A run's total span is the agent.run bar; scale everything to it.
      const total = Math.max(
        1,
        ...run.spans.map((s) => s.offset + s.duration)
      );
      const bars = run.spans
        .slice()
        .sort((a, b) => a.offset - b.offset)
        .map((s) => {
          const left = (s.offset / total) * 100;
          const width = Math.max(0.6, (s.duration / total) * 100);
          const cls = s.error ? "wf-bar is-error" : "wf-bar wf-" + s.kind;
          const title = `${s.kind}: ${esc(s.label)} — ${s.duration.toFixed(1)}ms @ +${s.offset.toFixed(0)}ms`;
          return `<div class="wf-track"><span class="wf-track-label">${esc(s.kind)}</span>`
            + `<div class="wf-lane"><div class="${cls}" style="left:${left}%;width:${width}%" `
            + `title="${title}"><span class="wf-bar-label">${esc(s.label)} · ${s.duration.toFixed(0)}ms</span></div></div></div>`;
        })
        .join("");
      return `<div class="wf-run"><div class="wf-run-head">`
        + `<span class="wf-run-id">run ${esc(run.id)}</span>`
        + `<span class="wf-run-total">${total.toFixed(0)}ms · ${run.spans.length} spans</span></div>`
        + `${bars}</div>`;
    }).join("");
    host.innerHTML = rows;
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
      // The catalog shares the dashboard's access rule; load it once data
      // access is confirmed (a 401 there shows an inline note instead).
      loadCatalog();
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
    initCatalog();

    // Wire up buttons
    const refreshBtn = $("refresh-btn");
    if (refreshBtn) refreshBtn.addEventListener("click", fetchData);

    const clearBtn = $("span-clear-btn");
    if (clearBtn) clearBtn.addEventListener("click", () => {
      const s = $("span-stream");
      if (s) s.innerHTML = "";
      spanCount = 0;
      setText("span-count", "0 spans");
      runs.clear();
      seenSpans.clear();
      renderWaterfall();
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
