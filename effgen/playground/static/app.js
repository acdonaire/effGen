/**
 * effGen Playground — frontend app
 *
 * Drives the existing POST /v1/chat/completions endpoint:
 *  - fetches /playground/bootstrap for presets, tool options, defaults, and (in
 *    local-view mode) a session key
 *  - fetches /v1/models/catalog to populate the model picker with real ids,
 *    pricing, and local/free flags
 *  - runs a prompt (streamed or complete), rendering the answer, token/cost
 *    stats, and the tool step trace
 *  - generates copy-as-curl / CLI / Python snippets from the form state
 *
 * All assets are served locally; the page has no external network dependency.
 * The API key, when the user pastes one, is held in memory for the tab only and
 * is never written to disk.
 */
(function () {
  "use strict";

  const THEME_KEY = "effgen-playground-theme";

  // In-memory config/state (no persistence of the key to disk).
  let apiKey = "";
  let bootstrap = { presets: [], tools: [], defaults: {} };
  let catalog = { data: [], local: [], providers: [] };
  let presetsByName = {};
  let snippetKind = "curl";
  let lastRun = null; // {model, prompt, tools, temperature, maxTokens, system}

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ------------------------------------------------------------------
  // Theme (mirrors the dashboard: OS scheme by default, toggle overrides)
  // ------------------------------------------------------------------
  function currentTheme() {
    const attr = document.documentElement.getAttribute("data-theme");
    if (attr === "dark" || attr === "light") return attr;
    const prefersLight = window.matchMedia
      && window.matchMedia("(prefers-color-scheme: light)").matches;
    return prefersLight ? "light" : "dark";
  }
  function syncThemeButton() {
    const icon = $("theme-icon");
    if (icon) icon.textContent = currentTheme() === "dark" ? "☾" : "☀";
  }
  function applyTheme(theme, persist) {
    document.documentElement.setAttribute("data-theme", theme);
    if (persist) { try { localStorage.setItem(THEME_KEY, theme); } catch { /* blocked */ } }
    syncThemeButton();
  }
  function initTheme() {
    let stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch { /* blocked */ }
    if (stored === "dark" || stored === "light") applyTheme(stored, false);
    else syncThemeButton();
    const btn = $("theme-btn");
    if (btn) btn.addEventListener("click", () =>
      applyTheme(currentTheme() === "dark" ? "light" : "dark", true));
  }

  // ------------------------------------------------------------------
  // Banners
  // ------------------------------------------------------------------
  function showAuthBanner(msg) {
    const el = $("auth-banner");
    if (!el) return;
    el.textContent = msg || "This run needs an API key. Paste it above, or restart the "
      + "server with EFFGEN_PUBLIC_PLAYGROUND=1 for local viewing.";
    el.hidden = false;
  }
  function hideAuthBanner() { const el = $("auth-banner"); if (el) el.hidden = true; }
  function showError(msg) {
    const el = $("error-banner");
    if (!el) return;
    el.textContent = msg;
    el.hidden = false;
  }
  function hideError() { const el = $("error-banner"); if (el) el.hidden = true; }

  // ------------------------------------------------------------------
  // Formatting
  // ------------------------------------------------------------------
  function fmtCost(v) {
    if (v == null) return "unpriced";
    if (v === 0) return "$0.00";
    if (v < 0.01) return "$" + v.toFixed(6);
    return "$" + v.toFixed(4);
  }
  function fmtPrice(rec) {
    if (rec.free_tier) return "free";
    if (rec.price_in_per_1m == null && rec.price_out_per_1m == null) return "unpriced";
    const i = rec.price_in_per_1m == null ? "?" : "$" + rec.price_in_per_1m;
    const o = rec.price_out_per_1m == null ? "?" : "$" + rec.price_out_per_1m;
    return i + " / " + o + " per 1M";
  }

  function authHeaders() {
    return apiKey ? { Authorization: "Bearer " + apiKey } : {};
  }

  // ------------------------------------------------------------------
  // Bootstrap + catalog
  // ------------------------------------------------------------------
  async function loadBootstrap() {
    try {
      const resp = await fetch("/playground/bootstrap", { cache: "no-store" });
      if (resp.status === 401) { showAuthBanner(); revealKeyField(); return; }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      bootstrap = await resp.json();
    } catch (err) {
      // Bootstrap is behind auth by default; the user pastes a key and we retry.
      revealKeyField();
      return;
    }
    // A local-view session key means Run works without a paste.
    if (bootstrap.session_key) { apiKey = bootstrap.session_key; hideAuthBanner(); }
    else { revealKeyField(); }

    applyDefaults();
    populatePresets();
    populateTools();
  }

  function revealKeyField() {
    const f = $("key-field");
    if (f) f.hidden = false;
    const input = $("api-key");
    if (input && !input.dataset.wired) {
      input.dataset.wired = "1";
      input.addEventListener("input", () => {
        apiKey = input.value.trim();
        if (apiKey) { hideAuthBanner(); loadCatalog(); }
      });
    }
  }

  function applyDefaults() {
    const d = bootstrap.defaults || {};
    if (d.temperature != null) $("temperature").value = d.temperature;
    if (d.max_tokens != null) $("max-tokens").value = d.max_tokens;
    if (d.stream != null) $("stream-toggle").checked = !!d.stream;
  }

  function populatePresets() {
    const sel = $("preset-select");
    (bootstrap.presets || []).forEach((p) => {
      presetsByName[p.name] = p;
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name + " — " + (p.description || "").slice(0, 60);
      sel.appendChild(opt);
    });
    sel.addEventListener("change", onPresetChange);
  }

  function onPresetChange() {
    const name = $("preset-select").value;
    const hint = $("preset-hint");
    if (!name || !presetsByName[name]) {
      if (hint) hint.textContent = "";
      syncToolChecks([]);
      return;
    }
    const p = presetsByName[name];
    // Apply the preset client-side: its system prompt + tool selection + temp.
    if (p.temperature != null) $("temperature").value = p.temperature;
    syncToolChecks(p.tools || []);
    if (hint) {
      const known = (p.tools || []).filter((t) => (bootstrap.tools || []).includes(t));
      const extra = (p.tools || []).filter((t) => !(bootstrap.tools || []).includes(t));
      let msg = "System prompt applied.";
      if (known.length) msg += " Tools: " + known.join(", ") + ".";
      if (extra.length) msg += " (" + extra.join(", ") + " run server-side if hosted.)";
      hint.textContent = msg;
    }
  }

  function populateTools() {
    const host = $("tool-list");
    host.innerHTML = "";
    (bootstrap.tools || []).forEach((name) => {
      const id = "tool-" + name;
      const label = document.createElement("label");
      label.className = "tool-chip";
      label.innerHTML = `<input type="checkbox" id="${id}" value="${esc(name)}" /> ${esc(name)}`;
      host.appendChild(label);
    });
  }

  function syncToolChecks(names) {
    const set = new Set(names || []);
    (bootstrap.tools || []).forEach((name) => {
      const cb = $("tool-" + name);
      if (cb) cb.checked = set.has(name);
    });
  }

  function selectedTools() {
    // Union of the checkbox tools and any preset-only tools (hosted server-side).
    const checked = (bootstrap.tools || [])
      .filter((name) => { const cb = $("tool-" + name); return cb && cb.checked; });
    const presetName = $("preset-select").value;
    const presetTools = presetName && presetsByName[presetName]
      ? (presetsByName[presetName].tools || []) : [];
    const out = new Set(checked);
    presetTools.forEach((t) => out.add(t));
    return Array.from(out);
  }

  async function loadCatalog() {
    const sel = $("model-select");
    try {
      const resp = await fetch("/v1/models/catalog", {
        cache: "no-store", headers: authHeaders(),
      });
      if (resp.status === 401) { showAuthBanner(); return; }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      catalog = await resp.json();
    } catch (err) {
      sel.innerHTML = '<option value="">Catalog unavailable — type a model id below</option>';
      makeModelFreeText();
      return;
    }
    renderModelOptions();
  }

  function renderModelOptions() {
    const sel = $("model-select");
    sel.innerHTML = "";

    // Locally-cached models first — the zero-config, no-key-needed entry.
    const locals = (catalog.local || []).filter((m) => m.complete !== false);
    if (locals.length) {
      const grp = document.createElement("optgroup");
      grp.label = "Local (no cloud key)";
      locals.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.id + "  ·  local";
        opt.dataset.local = "1";
        grp.appendChild(opt);
      });
      sel.appendChild(grp);
    }

    const byProvider = {};
    (catalog.data || []).forEach((rec) => {
      if (rec.deprecated) return;
      (byProvider[rec.provider] = byProvider[rec.provider] || []).push(rec);
    });
    Object.keys(byProvider).sort().forEach((prov) => {
      const grp = document.createElement("optgroup");
      grp.label = prov;
      byProvider[prov].forEach((rec) => {
        const opt = document.createElement("option");
        opt.value = rec.provider + ":" + rec.id;
        const tag = rec.free_tier ? "free" : (rec.is_priced ? fmtPrice(rec) : "unpriced");
        opt.textContent = rec.id + "  ·  " + tag;
        opt.dataset.rec = JSON.stringify(rec);
        grp.appendChild(opt);
      });
      sel.appendChild(grp);
    });

    // Default the picker to the server's configured default when present.
    const def = bootstrap.default_model;
    if (def) {
      for (const o of sel.options) {
        if (o.value === def || o.value.endsWith(":" + def) || o.value === "openai:" + def) {
          o.selected = true; break;
        }
      }
    }
    sel.addEventListener("change", onModelChange);
    onModelChange();
  }

  function makeModelFreeText() {
    // Fallback when the catalog can't be read: let the user type any id.
    const sel = $("model-select");
    sel.insertAdjacentHTML("beforeend",
      '<option value="' + esc(bootstrap.default_model || "") + '">'
      + esc(bootstrap.default_model || "effgen-default") + "</option>");
  }

  function onModelChange() {
    const opt = $("model-select").selectedOptions[0];
    const hint = $("model-hint");
    if (!opt || !hint) return;
    if (opt.dataset.local) { hint.textContent = "Local model — runs on this server, unpriced."; return; }
    if (!opt.dataset.rec) { hint.textContent = ""; return; }
    try {
      const rec = JSON.parse(opt.dataset.rec);
      const bits = [fmtPrice(rec)];
      if (rec.context_window) bits.push((rec.context_window / 1000).toFixed(0) + "K ctx");
      if (rec.supports_tools) bits.push("tools");
      if (rec.supports_vision) bits.push("vision");
      if (rec.verified_on) bits.push("verified " + rec.verified_on);
      hint.textContent = bits.join("  ·  ");
    } catch { hint.textContent = ""; }
  }

  // ------------------------------------------------------------------
  // Run
  // ------------------------------------------------------------------
  function buildMessages(system, prompt) {
    const msgs = [];
    if (system && system.trim()) msgs.push({ role: "system", content: system });
    msgs.push({ role: "user", content: prompt });
    return msgs;
  }

  function currentSystemPrompt() {
    const name = $("preset-select").value;
    return name && presetsByName[name] ? (presetsByName[name].system_prompt || "") : "";
  }

  function toolSpecs(names) {
    return names.map((n) => ({ type: "function", function: { name: n } }));
  }

  async function run() {
    hideError();
    const prompt = $("prompt").value.trim();
    if (!prompt) { showError("Enter a prompt first."); return; }
    const model = $("model-select").value || bootstrap.default_model || "effgen-default";
    if (!apiKey && !bootstrap.dev_mode) { showAuthBanner(); revealKeyField(); return; }

    const tools = selectedTools();
    const temperature = parseFloat($("temperature").value);
    const maxTokens = parseInt($("max-tokens").value, 10);
    const stream = $("stream-toggle").checked;
    const system = currentSystemPrompt();

    lastRun = { model, prompt, tools, temperature, maxTokens, system };
    renderSnippet();
    $("snippets").hidden = false;

    const body = {
      model,
      messages: buildMessages(system, prompt),
      temperature: isNaN(temperature) ? undefined : temperature,
      max_tokens: isNaN(maxTokens) ? undefined : maxTokens,
      stream,
    };
    if (tools.length) body.tools = toolSpecs(tools);
    if (stream) body.stream_options = { include_usage: true };

    const btn = $("run-btn");
    btn.disabled = true;
    btn.textContent = "Running…";
    const answerEl = $("answer");
    answerEl.className = "answer";
    answerEl.textContent = "";
    $("stats").hidden = true;
    $("trace-wrap").hidden = true;

    const started = performance.now();
    try {
      if (stream) await runStreaming(body, answerEl);
      else await runComplete(body, answerEl);
    } catch (err) {
      showError(String(err && err.message ? err.message : err));
      answerEl.className = "answer placeholder";
      answerEl.textContent = "Run failed.";
    } finally {
      const ms = performance.now() - started;
      const lat = $("stat-latency");
      if (lat && lat.textContent === "—") lat.textContent = (ms / 1000).toFixed(2) + "s";
      btn.disabled = false;
      btn.textContent = "Run";
    }
  }

  async function runComplete(body, answerEl) {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
      body: JSON.stringify(body),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      if (resp.status === 401) { showAuthBanner(); revealKeyField(); }
      const emsg = data && data.error ? data.error.message : ("HTTP " + resp.status);
      throw new Error(emsg);
    }
    const choice = (data.choices || [])[0] || {};
    answerEl.textContent = (choice.message && choice.message.content) || "(no content)";
    renderStats(data);
    renderTrace(data.effgen || {}, body.tools);
  }

  async function runStreaming(body, answerEl) {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, authHeaders()),
      body: JSON.stringify(body),
    });
    if (!resp.ok || !resp.body) {
      const data = await resp.json().catch(() => ({}));
      if (resp.status === 401) { showAuthBanner(); revealKeyField(); }
      throw new Error(data && data.error ? data.error.message : ("HTTP " + resp.status));
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    let usage = null;
    let model = body.model;
    let cursor = '<span class="cursor">▋</span>';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const evt of events) {
        const line = evt.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let obj;
        try { obj = JSON.parse(payload); } catch { continue; }
        if (obj.error) throw new Error(obj.error.message || "stream error");
        if (obj.model) model = obj.model;
        if (obj.usage) usage = obj.usage;
        const delta = (obj.choices && obj.choices[0] && obj.choices[0].delta) || {};
        if (delta.content) {
          text += delta.content;
          answerEl.innerHTML = esc(text) + cursor;
          answerEl.scrollTop = answerEl.scrollHeight;
        }
      }
    }
    answerEl.textContent = text || "(no content)";
    renderStats({ model, usage, effgen: {} });
    // The streamed path returns tokens, not the tool trace; a non-streamed run
    // surfaces the step trace.
    if (body.tools && body.tools.length) {
      $("trace-wrap").hidden = false;
      $("trace").innerHTML = '<p class="trace-empty">Turn off streaming to see the tool step trace.</p>';
    }
  }

  function renderStats(data) {
    const u = data.usage || {};
    const eff = data.effgen || {};
    $("stat-model").textContent = data.model || eff.resolved_model || "—";
    const pin = u.prompt_tokens == null ? "?" : u.prompt_tokens;
    const pout = u.completion_tokens == null ? "?" : u.completion_tokens;
    $("stat-tokens").textContent = pin + " / " + pout;
    $("stat-total").textContent = u.total_tokens == null ? "—" : u.total_tokens;
    $("stat-cost").textContent = fmtCost(eff.cost_usd);
    $("stat-latency").textContent = "—";
    $("stats").hidden = false;
  }

  function renderTrace(eff, toolsSent) {
    const wrap = $("trace-wrap");
    const host = $("trace");
    const steps = eff.trace || [];
    if (!steps.length) {
      if (toolsSent && toolsSent.length) {
        wrap.hidden = false;
        host.innerHTML = '<p class="trace-empty">No tool was invoked for this prompt.</p>';
      } else {
        wrap.hidden = true;
      }
      return;
    }
    wrap.hidden = false;
    host.innerHTML = steps.map((s) => {
      const cls = s.ok === false ? "err" : "ok";
      const dur = s.duration_ms != null ? s.duration_ms.toFixed(0) + "ms" : "";
      return `<div class="trace-step ${cls}">`
        + `<span class="trace-tool">${esc(s.tool)}</span>`
        + `<span class="trace-args">${esc(s.args || "")}</span>`
        + `<span class="trace-arrow">→</span>`
        + `<span class="trace-result">${esc(s.result_summary || "")}</span>`
        + `<span class="trace-dur">${dur}</span></div>`;
    }).join("");
  }

  // ------------------------------------------------------------------
  // Snippets
  // ------------------------------------------------------------------
  function shellQuote(s) { return "'" + String(s).replace(/'/g, "'\\''") + "'"; }
  function pyQuote(s) { return JSON.stringify(String(s)); }

  function renderSnippet() {
    if (!lastRun) return;
    const box = $("snippet-code");
    box.textContent = buildSnippet(snippetKind, lastRun);
  }

  function buildSnippet(kind, r) {
    const preset = $("preset-select").value;
    if (kind === "curl") {
      const body = {
        model: r.model,
        messages: buildMessages(r.system, r.prompt),
      };
      if (!isNaN(r.temperature)) body.temperature = r.temperature;
      if (!isNaN(r.maxTokens)) body.max_tokens = r.maxTokens;
      if (r.tools.length) body.tools = toolSpecs(r.tools);
      return "curl http://127.0.0.1:8000/v1/chat/completions \\\n"
        + "  -H 'Authorization: Bearer $EFFGEN_API_KEY' \\\n"
        + "  -H 'Content-Type: application/json' \\\n"
        + "  -d " + shellQuote(JSON.stringify(body));
    }
    if (kind === "cli") {
      let cmd = "effgen run " + shellQuote(r.prompt) + " -m " + shellQuote(r.model);
      if (preset) cmd += " --preset " + preset;
      if (r.tools.length) cmd += " -t " + r.tools.join(" ");
      if (!isNaN(r.temperature)) cmd += " --temperature " + r.temperature;
      if (!isNaN(r.maxTokens)) cmd += " --max-tokens " + r.maxTokens;
      return cmd;
    }
    // python
    const hasTools = r.tools.length > 0;
    const lines = ["from effgen import Agent", "from effgen.core.agent import AgentConfig"];
    if (hasTools) lines.push("from effgen.tools import get_registry");
    lines.push("");
    if (hasTools) { lines.push("reg = get_registry()"); lines.push(""); }
    const cfg = ["    model=" + pyQuote(r.model)];
    // AgentConfig(tools=...) takes Tool instances, so resolve each name from the
    // registry rather than passing bare strings.
    if (hasTools) {
      const resolved = r.tools.map((n) => "reg.get_tool_sync(" + pyQuote(n) + ")").join(", ");
      cfg.push("    tools=[" + resolved + "]");
    }
    if (!isNaN(r.temperature)) cfg.push("    temperature=" + r.temperature);
    if (!isNaN(r.maxTokens)) cfg.push("    max_tokens=" + r.maxTokens);
    if (r.system && r.system.trim()) cfg.push("    system_prompt=" + pyQuote(r.system));
    lines.push("agent = Agent(AgentConfig(");
    lines.push(cfg.join(",\n"));
    lines.push("))");
    lines.push("print(agent.run(" + pyQuote(r.prompt) + "))");
    return lines.join("\n");
  }

  function initSnippetTabs() {
    document.querySelectorAll(".snippet-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".snippet-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        snippetKind = tab.dataset.kind;
        renderSnippet();
      });
    });
    const copy = $("copy-btn");
    if (copy) copy.addEventListener("click", async () => {
      const text = $("snippet-code").textContent;
      try {
        await navigator.clipboard.writeText(text);
        copy.textContent = "Copied";
        setTimeout(() => { copy.textContent = "Copy"; }, 1500);
      } catch {
        // Clipboard blocked (non-secure context): select the text as a fallback.
        const range = document.createRange();
        range.selectNodeContents($("snippet-code"));
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
    });
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  async function init() {
    initTheme();
    initSnippetTabs();
    $("run-btn").addEventListener("click", run);
    $("prompt").addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run();
    });
    await loadBootstrap();
    await loadCatalog();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
