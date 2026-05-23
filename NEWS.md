# effGen Release Notes

## v0.2.9 — May 23, 2026

**effGen v0.2.9** ships the **Observability & Reliability** layer — everything you need to run effGen agents confidently in production. Structured JSON logs with automatic secret redaction, OpenTelemetry traces with configurable sampling, Prometheus histograms, SLO burn-rate tracking, circuit breakers, bulkheads, jittered retries, a deterministic chaos harness, a Hypothesis-based fuzz suite, a load-testing CLI (`effgen loadtest`), and six Alertmanager-compatible alert rules. All telemetry is async/non-blocking — a failed export never fails inference.

### What's new at a glance

**Structured logging with secret redaction.** `get_logger(__name__)` emits JSON lines with `{ts, level, module, event, attributes, trace_id, span_id}`. A built-in `Redactor` catches OpenAI, Anthropic, Cerebras, Google, HuggingFace, Groq, Bearer token, Slack, and Discord webhook patterns at the encoder — secrets can't slip through any log path.

**Prometheus histograms.** Four new metrics with full label dimensions: `effgen_model_call_latency_seconds{provider,model,outcome}`, `effgen_tool_call_latency_seconds{tool,outcome}`, `effgen_agent_iteration_latency_seconds{preset}`, and `effgen_tokens_total{provider,model,kind}`. The existing `/metrics` endpoint now emits histogram buckets in valid Prometheus text format.

**SLO tracking.** `SLOTracker` maintains a rolling window per SLO (name, target%, window duration). `burn_rate(name)` returns the current error-budget consumption rate. Results are exposed at the `/slo` FastAPI endpoint.

**Configurable tracing samplers.** Choose `AlwaysOn`, `AlwaysOff`, `ParentBased(TraceIdRatio(p))`, or `ParentBased(RateLimited(per_second))` via `ObservabilityConfig`. A canonical span-attribute spec (`effgen/observability/spans.py`) is the single source of truth for all attribute names — no more scattered string literals.

**Reliability primitives.** Four building blocks now wrap every adapter call:
- **Timeouts** — `ReliabilityConfig.default_timeouts` with `{model_call: 60, tool_call: 30, http: 20}`. Explicit timeouts on every adapter httpx client; audit guard raises if any are missing.
- **Retries** — `@retryable(Retry(...))` with jittered exponential backoff. Handles 5xx, 429 + Retry-After, transient network errors. Emits OTel `effgen.retry.attempt` events.
- **Circuit breaker** — `CircuitBreaker` (CLOSED → OPEN → HALF_OPEN) per provider via `CircuitBreakerRegistry`. Isolates a misbehaving provider automatically.
- **Bulkhead** — `Bulkhead(max_concurrency, queue_size, queue_timeout)` per provider via `BulkheadRegistry`. Prevents one provider from starving others.

**Deterministic chaos harness.** `Chaos(seed)` injects `NetworkTimeout`, `Http5xx`, `Http429`, `SlowResponse`, `PartialResponse`, or `MalformedJSON` faults into the provider middleware. Four canonical scenarios (fallback on 5xx, Retry-After honoured, timeout fires cleanly, AllProvidersFailed no silent empty string) each pass across 10 seeds — 273 tests, all deterministic.

**Fuzz suite.** Hypothesis-based fuzz tests cover all 66 `BaseTool` subclasses (500 examples each), random `ContentPart` message sequences, and the router's provider-availability logic. No unhandled exceptions, no secret leaks across 164 test/500-example combinations.

**Load-testing CLI.** `effgen loadtest --concurrency 10 --duration 30 --scenario fixed` runs a mock or live load test and writes a JSON report with throughput, p50/p95/p99 latency, and error rate.

**Alerting.** `docs/observability/alert_rules.yaml` contains six Alertmanager-compatible rules (error rate, p95 latency, cost burn, SLO fast-burn, SLO slow-burn, circuit-breaker open). `AlertWebhook(url).fire(alert)` posts to Slack or Discord; never raises even if delivery fails.

### CLI quick-start

```bash
# Run a load test against the mock model
effgen loadtest --concurrency 10 --duration 30

# Run against Cerebras live
effgen loadtest --provider cerebras --model llama3.1-8b --concurrency 5 --duration 15

# Check SLOs
curl http://localhost:8000/slo

# Check Prometheus metrics
curl http://localhost:8000/metrics | grep effgen_model_call_latency
```

### Python API quick-start

```python
from effgen.observability import get_logger
from effgen.reliability.retry import Retry, retryable
from effgen.reliability.circuit import CircuitBreaker

log = get_logger(__name__)
log.event("demo.started")

breaker = CircuitBreaker("my_provider", failure_threshold=5, recovery_timeout=30)

@retryable(Retry(max_attempts=3, base_delay=1.0, jitter=True))
def call_api():
    if not breaker.is_call_permitted():
        raise RuntimeError("circuit open for my_provider")
    try:
        result = ...  # your adapter call here
        breaker.on_success()
        return result
    except Exception as exc:
        breaker.on_failure(exc)
        raise
```

### Upgrading from v0.2.8

No breaking API changes. All new modules are additive; existing code is unaffected.

```bash
pip install --upgrade effgen
```

---

## v0.2.8 — May 21, 2026

**effGen v0.2.8** ships first-class **multimodal input** — send images, audio, and video to any capable provider through a single, unified `Message` schema. Six providers (Gemini, OpenAI, Groq, Anthropic, Together, HuggingFace) gain structured multimodal routing with automatic preprocessing, capability-gated error surfaces, a new `multimodal` preset, and five end-to-end cookbook walkthroughs. No breaking API changes.

### What's new at a glance

**Unified `Message` schema.** `Message.content` is now a typed `List[ContentPart]` — a union of `TextPart`, `ImagePart`, `AudioPart`, `VideoPart`, `ToolCallPart`, and `ToolResultPart`. The old string constructor still works: `Message(role, "hello")` auto-wraps in a `TextPart`. Validation fires on construction, not at send time.

**Three `_from` helpers.** `image_from(source)`, `audio_from(source)`, and `video_from(source, fps=1)` accept `bytes`, a local path, a URL, a `PIL.Image`, or an `np.ndarray` — whichever is convenient. MIME type is inferred automatically.

**Preprocessing is explicit and loggable.** `image_pre.prepare()` enforces per-provider pixel/byte limits and Lanczos-downscales when needed, logging every action to `part.meta["preprocessing"]`. `audio_pre` downsamples to 16 kHz and chunks long clips. `video_pre` samples keyframes via ffmpeg (raising `MissingSystemDependency` with OS-specific install hints when absent).

**Image input across 6 providers.** Gemini, OpenAI gpt-4o, Groq Llama 4 / Llama 3.2-vision, Anthropic (code only), Together, and HF BLIP/LLaVA all accept `ImagePart`. Every adapter raises `CapabilityNotSupportedError` cleanly when the selected model doesn't support vision — no silent text fallback.

**Audio input across 3 providers.** Gemini native audio, OpenAI Whisper (`/audio/transcriptions`) + gpt-4o audio, and HF ASR. Anthropic raises `CapabilityNotSupportedError(Capability.audio_input)`.

**Video input — native + frame-sampling.** Gemini 2.x/3.x accepts raw video natively. All other adapters decompose a `VideoPart` into a sequence of `ImagePart`s (frame sampling) plus an optional `AudioPart` from the audio track.

**`multimodal` preset.** `create_agent("multimodal", model)` wires Gemini Flash-Lite as primary (vision + audio + video) with OpenAI gpt-4o-mini as vision fallback. The preset ships with `ImageInfoTool`, `ImageCaptionTool`, `OCRTool`, `AudioTranscribeTool`, `PDFTool`, `WeatherTool`, and the new `MultimodalDescribeTool` — which automatically chooses the right tool based on the input part type.

**MLX-VLM adapter.** `effgen/models/mlx_vlm_engine.py` wraps `mlx-vlm` for Apple Silicon vision-language inference. Raises `MissingSystemDependency` on non-Apple hardware or missing library. Live tests skipped on Linux; 28 unit tests with fakes pass.

**5 cookbook walkthroughs.** Image Q&A, audio transcribe + reason, video summarize, OCR + LLM structured extraction, chart reading from an image. Each is a runnable Python snippet with prose. See `docs/cookbook/README.md`.

### CLI quick-start

```bash
# Image Q&A via multimodal preset
effgen run --preset multimodal "What is in this image?" --image /tmp/photo.jpg

# Check which providers support vision
python -c "
from effgen.models.capabilities import Capability
from effgen import list_models
print([m for m in list_models('gemini') if Capability.vision in m.get('capabilities', [])][:3])
"
```

### Python API quick-start

```python
from effgen import image_from, audio_from, video_from, load_model
from effgen.core.messages import Message, Role
from effgen.presets import create_agent

model = load_model("gemini-2.0-flash", provider="gemini")
agent = create_agent("multimodal", model)

# Image
img = image_from("https://example.com/photo.jpg")
result = agent.run_message(Message(role=Role.USER, content=[img, "Describe this."]))

# Audio
aud = audio_from("/tmp/interview.mp3")
result = agent.run_message(Message(role=Role.USER, content=[aud, "Summarize in one line."]))

# Video (requires ffmpeg for frame-sampling fallback)
vid = video_from("/tmp/clip.mp4", fps=1)
result = agent.run_message(Message(role=Role.USER, content=[vid, "What happens in the first 5 seconds?"]))
```

### Upgrading from v0.2.7

No breaking API changes. The old string-based `Message` constructor is unchanged.

```bash
pip install --upgrade effgen
```

---

## v0.2.7 — May 20, 2026

**effGen v0.2.7** ships the **Prompt Library** — a curated, domain-organized catalog of **31 reusable prompt templates** covering research, coding, data/SQL, legal, medical, creative writing, and business. Every template is a Python callable that renders deterministically for fixed inputs, ships with a fixture and golden evaluation test, and is accessible through a rich CLI and an interactive playground.

### What's new at a glance

**31 templates across 7 domains.** Research (literature review, paper summary, citation extraction, methodology critique), Coding (code review, bug diagnosis, refactoring plan, test generation, docstring fill), Data (NL-to-SQL, SQL explain, SQL optimize, data profile, ETL plan), Legal (contract summary, clause classify, research brief), Medical (symptom triage, drug interaction, medical literature), Creative (story continuation ×2, poetry forms, character bio, world building), and Business (meeting summary, email draft, OKR generation, SWOT analysis, elevator pitch).

**Golden + live eval harness.** `effgen prompts eval` renders every template with its fixture and compares against a stored golden. Add `--live --model <name>` to run prompts through a real model and validate output shape — including `sqlglot.parse()` for SQL templates and `ast.parse()` for generated Python.

**Interactive playground.** `effgen prompts playground` opens a REPL where you can select any template, set its inputs, render a preview, run it against a model, and save the session to JSON. Non-interactive `effgen prompts render` and `effgen prompts run` modes are also available for scripts.

**Legal and medical safety.** Every legal and medical template renders the required non-advice disclaimer verbatim in the system prompt — enforced by unit tests, not convention.

**Auto-generated gallery.** `docs/prompts/gallery.md` lists all 31 templates with their variant and one-line description. Regenerate it any time with `effgen prompts list --format markdown`.

### CLI quick-start

```bash
# Discover templates
effgen prompts list
effgen prompts list --domain research --variant cot
effgen prompts list --format markdown

# Inspect a template
effgen prompts show research.literature_review.v1.cot

# Run golden evaluations (no model needed)
effgen prompts eval

# Run live evaluations (requires API key)
effgen prompts eval --domain coding --live --model llama3.1-8b

# Interactive playground
effgen prompts playground

# Non-interactive render
effgen prompts render data.sql_from_nl.v1 --input '{"schema_ddl": "CREATE TABLE orders (id INT, total FLOAT)", "question": "Total orders this month", "dialect": "sqlite"}'
```

### Python API quick-start

```python
from effgen.prompts.library import registry

# Browse all templates
for p in registry.all():
    print(p.name, p.variant, p.domain)

# Get and render a specific template
p = registry.get("research.literature_review.v1.cot")
prompt_text = p.template(
    topic="diffusion models",
    years_range="2022-2025",
    max_papers=10
)

# Search by domain and variant
sql_prompts = registry.search(domain="data", variant="structured")
```

### Upgrading from v0.2.6

No breaking API changes. All prompt library classes are opt-in additions.

```bash
pip install --upgrade effgen
```

---

## v0.2.6 — May 19, 2026

**effGen v0.2.6** is a document, media, and communication tools release that adds **14 new built-in tools** — OCR, audio transcription, image analysis, document parsing (PDF/DOCX/Excel), geo/weather, and email/webhook — raising the total built-in tool count from 44 to **58+**. Two new presets (`media`, `notify`) join the existing roster. No breaking API changes.

### New Tools at a Glance

**OCR** — `OCRTool` extracts text from images using Tesseract locally, with OCR.space as a free API fallback. Raises `OCRBackendUnavailable` with per-OS install instructions when neither backend is available. Added to `general` preset.

**Audio Transcription** — `AudioTranscribeTool` transcribes audio files locally via `faster-whisper` (CPU/GPU auto-detected), falling back to HuggingFace Inference when `HF_TOKEN` is set. Warns on CPU when a large model size is selected. Added to new `media` preset.

**Image Analysis** — `ImageInfoTool` extracts image metadata and performs local resize/thumbnail operations entirely via Pillow (zero network). `ImageCaptionTool` uses the effGen model router to select a vision-capable provider (Gemini / OpenAI / MLX-VLM) and generate a natural-language caption or description. `ImageInfoTool` is in `general`; `ImageCaptionTool` is in `media`.

**Document Parsing** — `PDFTool` (pypdf + pdfplumber), `DOCXTool` (python-docx), and `ExcelTool` (openpyxl + pandas) round-trip local documents with full text, table, and metadata extraction. All three added to both `research` and `general` presets.

**Geo / Weather** — `WeatherTool` fetches current, forecast, and historical weather from Open-Meteo (free, no auth). `GeocodeTool` forward/reverse geocodes via Nominatim (OSM) with 1 req/s token-bucket rate limiting and proper User-Agent header. `MapsTool` renders static PNG maps from OSM tiles via the `staticmap` library. All three added to `general`.

**Email** — `EmailSMTPTool` sends email via SMTP (stdlib `smtplib`, TLS on by default). `EmailIMAPTool` reads email via IMAP (stdlib `imaplib`). Both raise `MissingCredentialsError` when env vars are absent. Added to new `notify` preset.

**Webhooks** — `SlackWebhookTool` and `DiscordWebhookTool` post messages to Slack and Discord via incoming webhook URLs (no OAuth). Webhook URLs are redacted in all logs. Both added to `notify` preset.

### New Presets

```python
from effgen.presets import create_agent
from effgen import load_model

model = load_model("llama3.1-8b", provider="cerebras")

# Media processing agent
media_agent = create_agent("media", model)   # AudioTranscribeTool + ImageCaptionTool

# Notification/alert agent
notify_agent = create_agent("notify", model) # EmailSMTP + EmailIMAP + Slack + Discord
```

### Upgrading from v0.2.5

No breaking API changes. All new tools are opt-in extras.

```bash
pip install --upgrade "effgen[all]"
# or selectively:
pip install --upgrade "effgen[documents]"   # PDFTool, DOCXTool, ExcelTool
pip install --upgrade "effgen[audio]"       # AudioTranscribeTool
pip install --upgrade "effgen[tools]"       # OCRTool, ImageInfoTool, and more
```

**System dependencies** (only needed for the relevant tool's primary path):
- `OCRTool` Tesseract: `apt-get install tesseract-ocr` / `brew install tesseract`
- `AudioTranscribeTool` ffmpeg (for non-WAV): `apt-get install ffmpeg` / `brew install ffmpeg`

---

## v0.2.5 — May 18, 2026

**effGen v0.2.5** is a tools-focused release that adds **13 free, no-auth-required tools** across six new categories — academic research, news aggregation, YouTube, social media, translation/language detection, and QR codes — bringing the total built-in tool count above 44. Every new tool ships with structured `{success, data, error}` output, preset integration, and a dedicated doc page.

### New Tools at a Glance

**Academic Research** — `PubMedTool` (NCBI E-utilities, 3 operations, built-in token-bucket rate limiter), `ArXivTool` (Atom feed search + PDF download), `SemanticScholarTool` (paper search + citations + references with polite backoff). All three are now part of the `research` preset.

**News & RSS** — `RSSFeedTool` fetches and full-text-searches any RSS/Atom feed; `NewsTool` aggregates top headlines across a curated list of reputable sources (Reuters, BBC, Hacker News, NPR, Al Jazeera, and more) with an optional NewsAPI.org key for better relevance. Both are in the `research` and `general` presets.

**YouTube** — `YouTubeTranscriptTool` pulls captions from public videos without a Google API key (via `youtube-transcript-api`), with URL extraction for watch?v=, youtu.be/, and shorts/ formats. `YouTubeMetadataTool` retrieves video and channel metadata via `yt-dlp` in metadata-only mode. Both added to `research`.

**Social Media** — `RedditTool` reads top/hot posts, user submissions, and thread comments from Reddit's public JSON endpoints (no OAuth). `HackerNewsTool` covers top/new stories, items, and user profiles from HN's Firebase API. Both added to `research` and `general`.

**Translation & Language Detection** — `TranslateTool` translates text between languages using LibreTranslate (configurable endpoint) with an offline `argostranslate` fallback; language packs are cached in `~/.effgen/argos/`. `LanguageDetectTool` detects language in text or batches, fully offline via `langdetect` (55+ languages). Both added to `general`.

**QR Codes** — `QRGenerateTool` generates QR codes locally from any text or URL, returning a base64 PNG or saving to a file path. `QRReadTool` decodes QR codes and barcodes from image files or base64 PNG using `pyzbar` + Pillow, with an OpenCV QR fallback when `libzbar` is unavailable. Both added to `general`.

### Tool Gallery

A new `docs/tools/gallery.md` file provides a one-line description and a working quickstart snippet for every tool in the effGen ecosystem — useful for discovering what's available at a glance.

### Upgrading from v0.2.4

No breaking API changes. All new tools are opt-in; existing code is unaffected. Install the new tool extras:

```bash
pip install --upgrade "effgen[tools]"   # feedparser, youtube-transcript-api, yt-dlp, pyzbar, qrcode, opencv, langdetect
# or grab everything:
pip install --upgrade "effgen[all]"
```

---

## v0.2.4 — May 14, 2026

**effGen v0.2.4** makes multi-provider AI inference production-grade with a composable **ModelRouter** — a new opt-in layer that sits between your application code and the 9 cloud providers effGen already supports. Instead of hard-coding a provider, you describe what you need (cheapest call within budget, fastest that meets an SLA, prefer free tier, fall back to paid) and the router picks the right provider, records its reasoning, and transparently retries or fails over when things go wrong.

### Top Highlights

1. **Three composable routing policies** — mix and match to build exactly the routing logic you need:

   ```python
   from effgen import PolicyBasedRouter, RoutingContext, CostBasedPolicy, LatencyBasedPolicy
   from effgen.models.capabilities import Capability

   router = PolicyBasedRouter(
       policies=[LatencyBasedPolicy(), CostBasedPolicy()],
   )
   context = RoutingContext(
       prompt_tokens_estimate=500,
       user_budget_usd=0.01,
       latency_budget_ms=3000,
       required_capabilities={Capability.chat, Capability.tools},
   )
   decision = router.route(context)
   print(decision.chosen)        # e.g., ProviderModelPair("cerebras", "llama3.1-8b")
   print(decision.eliminated)    # list of (pair, reason) — fully explainable
   ```

2. **Transparent failover** — `route_and_execute(context, fn)` automatically retries on `RateLimitExceeded`, 5xx errors, or timeouts and moves to the next-best provider. Each failover fires a `RouterEvent` to any registered subscribers so you can log or alert in real time.

3. **Cross-process rate-limit coordination** — `SQLiteRateLimitStore` (WAL-mode, `BEGIN IMMEDIATE`) lets multiple workers share a single rate-limit budget at `~/.effgen/rate_limits.sqlite`. Pass it into `RateLimitCoordinator(storage=store)` — the default in-memory mode is unchanged.

4. **Persistent cost tracking + `effgen cost` CLI** — every API call writes a row to `~/.effgen/costs.sqlite`. Query it instantly:

   ```bash
   effgen cost today          # per-provider per-model table
   effgen cost week           # rolling 7-day view
   effgen cost by-provider    # lifetime totals
   effgen cost set-budget 1.0 # set $1/day cap
   ```

   When cumulative daily spend hits 80% of your cap, effGen emits a warning; at 100% it raises `BudgetExceededError` — which the router treats as retriable and automatically fails over to a free-tier provider.

5. **Fully explainable decisions** — every `RouterDecision` carries the chosen provider, a list of eliminated candidates with per-provider reasons (`"rate_limited"`, `"no_key"`, `"cost_exceeds_budget"`, `"latency_exceeds_sla"`), the winning policy name, and a numeric score. Nothing is a black box.

### Upgrading from v0.2.3

No breaking API changes. All existing `load_model`, `Agent`, and direct adapter paths work without modification. The `ModelRouter` is a completely opt-in new layer.

```bash
pip install --upgrade effgen
```

`RateLimitCoordinator` and `CostTracker` both retain their existing in-memory defaults — existing code that constructs them without a `storage=` argument is unaffected.

---

## v0.2.3 — May 4, 2026

**effGen v0.2.3** grows the provider roster from 4 to **9 cloud inference backends** — Groq, Together AI, Fireworks, Replicate, and HuggingFace Inference join the existing OpenAI, Anthropic, Gemini, and Cerebras adapters. Every new backend ships with streaming, native tool-calling where the provider supports it, automatic rate-limit coordination, and per-call cost tracking. A new `ProviderRegistry` consolidates all providers for clean introspection and the `effgen doctor` command tells you at a glance which API keys are wired up. A backend parity matrix proves that the canonical "What is (17 × 23) + sqrt(144)?" agentic task returns the correct answer (403) across every provider, with identical `ModelAuthError` raised on bad credentials.

### Top Highlights

1. **5 new cloud backends** — `GroqAdapter` (16 models, RPM/TPD windows), `TogetherAdapter` (163-model catalog with drift detection), `FireworksAdapter` (80 chat models), `ReplicateAdapter` (async run-poll + SSE streaming + timeout handling), `HFInferenceAdapter` (124-model HuggingFace Router catalog + custom Endpoint URL support). Each supports streaming and native tools.

   ```python
   from effgen import load_model

   # Groq — ultra-fast inference
   model = load_model("llama-3.1-8b-instant", provider="groq")

   # Together AI
   model = load_model("meta-llama/Llama-3.3-70B-Instruct-Turbo", provider="together")

   # Fireworks
   model = load_model("accounts/fireworks/models/llama-v3p1-8b-instruct", provider="fireworks")

   # HuggingFace Inference Router
   model = load_model("Qwen/Qwen2.5-72B-Instruct", provider="hf")
   ```

2. **Unified ProviderRegistry** — `list_providers()`, `list_models(provider)`, `lookup(model_id)` in one place. All 9 adapters self-register on import. Duplicate model IDs across providers raise `AmbiguousModelError` with disambiguation instructions.

3. **`effgen doctor`** — new CLI command that prints a table of all 9 providers and whether their API key is available, with setup instructions for missing keys.

4. **Backend parity matrix** — 7/8 providers passed the canonical agentic task (Anthropic skipped — no key in dev env; Replicate xfail — billing credits). All 9 raise `ModelAuthError` uniformly on bad credentials. Full report in `docs/providers/parity.md`.

5. **HuggingFace Router support** — `HFInferenceAdapter` routes via `provider="auto"` (the new HF Inference Router), supports 124 bundled models with live `refresh_models()` + `check_drift()`, and raises helpful `ModelUnavailableError` with `suggest_alternatives()` when a model is temporarily offline.

### Installing New Backends

```bash
pip install "effgen[groq]"       # Groq: GROQ_API_KEY
pip install "effgen[together]"   # Together AI: TOGETHER_API_KEY
pip install "effgen[fireworks]"  # Fireworks: FIREWORKS_API_KEY
pip install "effgen[replicate]"  # Replicate: REPLICATE_API_TOKEN
pip install "effgen[hf]"         # HuggingFace: HF_TOKEN
```

Or grab everything at once:

```bash
pip install "effgen[all]"
```

### Upgrading from v0.2.2

No breaking API changes. All new providers are opt-in extras. Existing `load_model`, `Agent`, and tool calls work without modification.

```bash
pip install --upgrade effgen
```

---

## v0.2.2 — April 28, 2026

**effGen v0.2.2** brings Gemini's latest thinking and grounding capabilities to effGen, adds the Gemini Files API and three Gemini-native tools, and modernizes Anthropic support for the full Claude 4.x lineup.

### Top Highlights

1. **Gemini 3.x / 2.5 / 2.0 + Gemma 3/4 model registry** — `gemini-3.1-flash-lite`, `gemini-3.0-pro`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`, and Gemma families all recognized with correct context windows, output limits, and feature flags. SDK migrated to `google-genai>=1.0.0`.

2. **Gemini `thinking_budget`** — pass `thinking_budget=8192` (or any token count) in `GenerationConfig` to activate Gemini's internal reasoning. Set `include_thoughts=True` to surface the thinking trace in `ModelResponse.metadata["thinking"]`.

   ```python
   from effgen import load_model
   from effgen.models.base import GenerationConfig
   model = load_model("gemini-3.1-flash-lite", provider="gemini")
   result = model.generate("Explain why π is irrational.", config=GenerationConfig(thinking_budget=8192, include_thoughts=True))
   ```

3. **Gemini Google Search grounding** — set `grounding=True` in `GenerationConfig` and the adapter injects Google Search; grounding attributions (URLs + snippets) arrive in `ModelResponse.metadata["grounding_chunks"]`.

4. **Gemini Files API** — `effgen.models.gemini_files.upload_file(path)` returns a `FileRef`; pass it in `generate(prompt, files=[...])` to give the model access to PDFs, images, and other documents (2 GiB limit enforced before upload).

5. **Gemini native tools** — `GoogleSearchTool`, `GeminiUrlContextTool`, `GeminiCodeExecutionTool` in `effgen.tools.builtin.gemini_native`. Use them directly in Agent — they activate Gemini's server-side capabilities with no extra API calls. Pairing with a non-Gemini model raises `ToolIncompatibleError` at init.

6. **Anthropic Claude 4.x registry** — claude-opus-4-7 (1M ctx), claude-sonnet-4-6, claude-haiku-4-5, and the full legacy 3.x / 4.x lineup in `effgen/models/anthropic_models.py`.

7. **Anthropic extended thinking** — `GenerationConfig.thinking = {"type": "enabled", "budget_tokens": N}` activates Claude's extended thinking; `redacted_thinking` blocks are preserved across multi-turn conversations.

8. **Anthropic prompt caching** — `mark_cached(block)` + `AgentConfig.cache_system_prompt=True` / `cache_tools=True` wire `cache_control` automatically; cache hit/creation tokens surfaced in `ModelResponse.usage`.

9. **Anthropic streaming polish** — `generate_stream_full()` handles thinking deltas, redacted-thinking, and parallel `tool_use` blocks in a unified `StreamChunk` API.

10. **Experimental Anthropic native tools** — `AnthropicBashTool`, `AnthropicTextEditorTool`, `AnthropicComputerTool` stubs in `effgen/tools/builtin/anthropic_native.py` (flag-gated, not registered by default).


### Upgrading from v0.2.1

No breaking API changes. All new fields (`thinking_budget`, `include_thoughts`, `grounding`, `thinking`, `cache_system_prompt`, `cache_tools`) default to safe backward-compatible values.

```bash
pip install --upgrade effgen
```

---

## v0.2.1 — April 25, 2026

**effGen v0.2.1** brings **Cerebras** to effGen as a first-class inference backend and modernizes the **OpenAI** adapter for the latest reasoning models.

### Top Highlights

1. **Cerebras backend** — All 4 free-tier Cerebras models (`gpt-oss-120b`, `llama3.1-8b`, `qwen-3-235b-a22b-instruct-2507`, `zai-glm-4.7`) with streaming, native function-calling, automatic rate-limit coordination (RPM/RPH/RPD + TPM/TPH/TPD sliding windows), and per-call cost tracking. `pip install effgen[cerebras]` and set `CEREBRAS_API_KEY`.

   ```python
   from effgen import load_model
   model = load_model("llama3.1-8b", provider="cerebras")
   ```

2. **OpenAI: gpt-5, gpt-5.4-nano, and o-series reasoning models** — full registry coverage with `reasoning_effort` (`minimal`/`low`/`medium`/`high`) and `max_reasoning_tokens` on `GenerationConfig`. Reasoning-only payloads are routed only to reasoning-capable models; chat models silently drop the field.

3. **OpenAI prompt caching** — `cached_input_tokens` is now surfaced in `ModelResponse.usage` and metadata. `AgentConfig.stable_system_prompt=True` keeps your system prompt anchored at position 0 so OpenAI's automatic ≥1024-token prefix cache stays warm.

4. **Structured outputs v2** — `OpenAIAdapter.generate_structured()` with strict JSON Schema; `to_openai_schema(pydantic_model)` inlines `$ref`s and forces `additionalProperties: false`. Refusals raise `ModelRefusalError` with the model's refusal text preserved.

5. **OpenAI native tools** — `OpenAIWebSearchTool`, `OpenAICodeInterpreterTool`, and `OpenAIFileSearchTool` route through OpenAI's Responses API and compose with effGen's local tools in the same agent. Pairing one with a non-OpenAI model raises `ToolIncompatibleError` at Agent init (no surprise mid-run failures).

### Other Improvements
- `load_model(..., provider="openai"/"anthropic"/"gemini"/"cerebras")` now routes correctly (was previously HF-only)
- HF-only kwargs are stripped before reaching API adapters
- `transformers` engine `unload()` removes accelerate hooks + syncs CUDA, eliminating cross-test GPU state leaks
- Stability sweep: ruff clean, mypy lenient-clean, multi-Python-version verified (3.10/3.11/3.12/3.13)

### Upgrading from v0.2.0

No breaking API changes. New parameters (`reasoning_effort`, `max_reasoning_tokens`, `stable_system_prompt`) all default to safe values. To use Cerebras:

```bash
pip install --upgrade "effgen[cerebras]"
export CEREBRAS_API_KEY=...
```

## v0.2.0 — April 9, 2026

**effGen v0.2.0** is a major release that transforms the framework into a production-grade agentic AI platform. 15 development phases deliver powerful new capabilities — all optimized for Small Language Models.

### Top 5 Features

1. **Native Tool Calling & Structured Output** — Models like Qwen, Llama, and Mistral can now use their built-in function calling instead of text-based ReAct parsing. Set `tool_calling_mode="native"` or `"hybrid"` in AgentConfig. JSON schema and Pydantic model output validation included.

2. **Guardrails & Safety** — Protect your agents with `PIIGuardrail`, `PromptInjectionGuardrail`, `ToxicityGuardrail`, `ToolPermissionGuardrail`, and more. Use presets: `get_guardrail_preset("strict")` for instant configuration.

3. **Advanced RAG Pipeline** — Full document ingestion (PDF, DOCX, HTML, Markdown, CSV, JSON), semantic/code/table/hierarchical chunking, hybrid search (dense + BM25 + keyword), reranking, source attribution with inline citations. One-liner: `create_agent("rag", model, knowledge_base="./docs/")`.

4. **Production API Server** — OpenAI-compatible `/v1/chat/completions` endpoint, request queuing with priority, agent pooling, multi-tenancy with API key management, CORS, GZip, graceful shutdown. Drop-in replacement for OpenAI API with local SLMs.

5. **Apple Silicon Native (MLX)** — Community-contributed MLX and MLX-VLM backends for Apple Silicon. Native Metal GPU acceleration with unified memory. `pip install effgen[mlx]` — no CUDA required.

### What's New

- **31 built-in tools** (up from 14) — finance (stock/currency/crypto), data science (DataFrame/Plot/Stats), DevOps (Git/Docker/SystemInfo/HTTP), knowledge (Arxiv/StackOverflow/GitHub/Wolfram), communication (EmailDraft/SlackDraft/Notification)
- **Multi-agent orchestration** — MessageBus pub/sub, DAG-based workflows (YAML), shared state, agent lifecycle management with pools and registries
- **Model router** — automatic model selection based on query complexity; multi-model agents with speculative execution; model pool with LRU eviction
- **Checkpointing & sessions** — save/restore agent state mid-task; persistent conversation sessions across processes; background task runner with pause/resume/cancel
- **Evaluation framework** — 5 built-in test suites (270 test cases), regression tracking, model comparison matrix; `effgen eval` and `effgen compare` CLI
- **Observability** — full OpenTelemetry tracing, structured JSON logging with correlation IDs, Prometheus metrics with percentiles, Grafana dashboard template, interactive debug mode
- **Human-in-the-loop** — approval workflows for dangerous tools, clarification requests, feedback collection
- **Performance** — prompt caching (LRU + TTL), result caching with semantic similarity, token budget management, lazy model loading, GGUF/AWQ/GPTQ quantization, continuous batching, speculative decoding hints
- **Python & TypeScript SDKs** — `EffGenClient` with sync/async, streaming, retries; TypeScript client for Node/Deno/Bun/browser
- **Local embedding API** — `/v1/embeddings` endpoint with sentence-transformers + TF-IDF fallback, LRU + SQLite caching
- **Domain keyword expansion** — 5 built-in domains (Tech/Science/Finance/Health/Legal) with WordNet/template/LLM-based expansion

### Upgrading from v0.1.x

No breaking API changes. All existing `Agent`, `AgentConfig`, `load_model`, and tool APIs work without modification. New features are opt-in. See the [migration guide](docs/migration.md) for details.

```bash
pip install --upgrade effgen==0.2.0
```

### New Optional Dependencies

```bash
pip install effgen[rag]       # RAG pipeline (sentence-transformers, faiss-cpu)
pip install effgen[finance]   # Finance tools (yfinance)
pip install effgen[data]      # Data science tools (matplotlib, plotly)
pip install effgen[eval]      # Evaluation extras (rouge-score, nltk)
pip install effgen[gguf]      # GGUF model support (llama-cpp-python)
pip install effgen[mlx]       # Apple Silicon MLX support
pip install effgen[mlx-vlm]   # Apple Silicon vision-language models
```

---

## v0.1.3 — March 25, 2026

v0.1.3 addresses 19 issues discovered during v0.1.2 verification, hardening the framework for real-world SLM agent usage.

### Highlights

- **Smarter loop detection** — allows 1 retry before flagging exact loops, raises threshold for data-processing tools, and normalizes inputs before comparison. Fewer false positives in multi-step pipelines.
- **"Skip the tool" prompting** — ReAct prompt now explicitly tells SLMs they can answer directly without tools. Reduces unnecessary tool calls for greetings, jokes, and recall tasks.
- **Model-aware token counting** — ShortTermMemory uses the loaded model's tokenizer instead of the `len//4` heuristic, improving summarization trigger accuracy.
- **Sub-agent depth limit** — configurable `max_sub_agent_depth` (default 3) prevents infinite sub-agent recursion.
- **Circuit breaker persistence** — optional JSON file persistence so breaker state survives agent restarts.

### What's Improved

- Partial answer extraction now finds day names and numeric results in tool observations
- Model-family prompt formatters differentiated (Qwen `<|tools|>` tags, Llama header/EOT tags)
- Removed `\n\n\n` stop sequence that truncated multi-paragraph output
- Streaming examples hardened with SIGALRM timeouts
- Integration test fixtures gracefully fall back to fp16 when bitsandbytes is missing
- NotImplementedError stubs in MCP and Retrieval now include descriptive messages

### What's Fixed

- Loop detection false positives on JSON data pipelines
- SLMs over-using tools for tasks that don't need them
- DateTimeTool date queries more reliable (better answer extraction)
- Silent model loading failures now logged with clear warning

---

## v0.1.2 — March 12, 2026

v0.1.2 is a test-driven hardening release. Every feature was built by creating a real agent, testing it across multiple models (0.5B to 8B), watching what breaks, and fixing the framework.

### Highlights

- **10 comprehensive example agents** — Q&A, calculator, multi-tool, file operations, code execution, conversational memory, error recovery, data processing, streaming, and multi-agent pipeline orchestration
- **19 framework bugs fixed** — discovered through real inference testing, not unit tests. Fixes cover tool parsing, answer extraction, memory management, and model-specific edge cases
- **Cross-model compatibility matrix** — 11 models tested across all 10 agents. 73% pass rate (80 PASS, 23 PARTIAL, 7 FAIL out of 110 combinations)
- **Top models (10/10 PASS):** Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Phi-4-mini-instruct

### What's New

- 10 example agents in `examples/` with full documentation, model recommendations, and interactive modes
- Compatibility matrix at `examples/compatibility_matrix.md` with per-agent model recommendations
- User-explicit sub-agent trigger detection (e.g., "use 3 agents to parallelize this")
- Sweep runner (`examples/sweep_model.py`) for automated cross-model testing

### What's Improved

- ReAct loop is more robust — better loop detection, answer extraction, and error recovery
- Tool input parsing handles single-quoted JSON, non-JSON inputs, and markdown fences
- Conversation history is better managed — configurable turn limits, auto-summarization, response truncation
- Tool results are properly formatted for the model (no more raw dicts)

### What's Fixed

- 4-bit quantization now works correctly with TransformersEngine
- gemma-3 context length detection fixed for nested config
- DateTimeTool `now` operation respects date parameter
- PythonREPL no longer double-prints output
- Absolute file paths no longer get their leading slash stripped
- Many more — see [CHANGELOG.md](CHANGELOG.md) for the full list

### Model Recommendations

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| Q&A (no tools) | 0.5B | 1.5B+ |
| Tool calling | 1.5B | 3B |
| Multi-turn conversation | 1.5B | 3B |
| Multi-agent pipeline | 1.5B | 3B |

---

## v0.1.1 — March 6, 2026

v0.1.1 is a stabilization release that fixes metadata inconsistencies, improves error handling, adds 6 new examples, and expands the test suite.

### What's Fixed
- License references now consistently say Apache-2.0 everywhere (was MIT in some files)
- `setup.py` entry points, Development Status, and dependency versions now match `pyproject.toml`
- 5 bare `except:` handlers in GPU monitoring replaced with specific exception types
- 15+ stray `print()` calls converted to structured logging

### What's New
- 6 example scripts: presets, streaming, memory, multi-tool, weather, and plugin usage
- 50+ new tests covering CLI, API server, plugins, presets, fallback chains, and circuit breakers
- Top-level convenience imports for `ToolFallbackChain`, `CircuitBreaker`, `ToolPromptGenerator`, `AgentSystemPromptBuilder`
- `NEWS.md` for user-friendly release summaries

### What's Changed
- Error handlers across execution modules now log exceptions instead of silently swallowing them
- Comprehensive lint cleanup via ruff (2200+ auto-fixes)

---

# effGen v0.1.0 Release Notes

**Release Date:** March 1, 2026

effGen v0.1.0 is the first feature-complete release, upgrading the framework from Alpha to Beta status. This release transforms effGen into a full-featured agentic AI framework optimized for Small Language Models (1B-7B parameters).

## Highlights

- **14 Built-in Tools** — 7 new tools added: BashTool, WeatherTool, JSONTool, DateTimeTool, TextProcessingTool, URLFetchTool, and WikipediaTool
- **Protocol Support** — Complete MCP, A2A, and ACP protocol implementations for tool and agent interoperability
- **Real Token Streaming** — True streaming via `generate_stream()` with callbacks for thoughts, tool calls, observations, and answers
- **Memory System** — ShortTermMemory, LongTermMemory, and VectorMemoryStore integrated into the Agent lifecycle
- **Agent Presets** — One-line agent creation with `create_agent("math", model)` for math, research, coding, general, and minimal configurations
- **Plugin System** — Extend effGen with custom tools via entry points or directory-based discovery
- **CLI Enhancements** — Rich progress display, `--preset`, `--explain`, `--verbose` flags, tab completion for bash/zsh/fish, and persistent chat history
- **API Server** — WebSocket streaming, API key authentication, rate limiting, and OpenAPI documentation
- **CI/CD & Testing** — 6 GitHub Actions workflows, 67 unit tests, health monitoring, OpenTelemetry tracing, and Prometheus metrics

## What's Changed

- Structured tool descriptions with parameter types and usage examples
- `stream()` now uses real token streaming (previously character-by-character)
- `run_async()` is natively async (previously wrapped sync in executor)
- Memory uses proper ShortTermMemory/LongTermMemory classes
- Development status upgraded from Alpha to Beta

## What's Fixed

- All `NotImplementedError` paths in retrieval tool
- ACP JSON Schema validation (was checking required fields only)
- Streaming placeholder removed (`time.sleep(0.01)`)
- Direct inference now retains multi-turn conversation context

## Upgrading from v0.0.2

No breaking API changes. Existing `Agent(config=AgentConfig(...))` and `load_model()` calls work without modification. New features are opt-in.

```bash
pip install --upgrade effgen
```
