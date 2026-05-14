# effGen Release Notes

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
