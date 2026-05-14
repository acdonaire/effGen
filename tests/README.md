# effGen Test Suite

A concise map of `tests/` so contributors can find the right place for a new test, run a targeted slice, and understand which tests need network/API credentials.

## Layout

```
tests/
├── conftest.py                  # Root fixtures + secret scrubbing
├── benchmarks/                  # Performance & eval baselines
├── cli/                         # CLI-surface tests
├── e2e/                         # End-to-end workflow tests
├── fixtures/                    # Shared test data (configs, KB docs, mocks)
├── integration/                 # Live-provider integration tests
│   └── parity/                  # Cross-provider parity matrix
├── models/                      # Adapter / router unit tests
├── tools/                       # Native-tool (provider-native) tests
└── unit/                        # Pure unit tests (no network)
```

## Directories

### `unit/`
Pure unit tests. No network, no API keys required. Covers core modules: agent, memory, tools registry, RAG, eval, presets, prompt/result caches, rate limiting, circuit breaker, plugins, configs, MLX/MLX-VLM engines.

### `models/`
Adapter and router tests. The router-related files (`test_router_core.py`, `test_router_cost.py`, `test_router_latency.py`, `test_router_failover.py`, `test_latency_tracker.py`, `test_cost_store.py`, `test_rate_limit_store.py`) cover the v0.2.4 `ModelRouter`. Adapter test files test request-shape and parameter handling without hitting live endpoints (the live counterparts live in `integration/`).

### `integration/`
Live-API tests. Each provider has at least one `test_<provider>_live.py` file that hits the real endpoint. These tests skip cleanly when the corresponding API key is not present in the environment.
- `parity/test_backend_parity.py` — every provider answers the canonical "What is (17 × 23) + sqrt(144)?" task with the correct result (403).
- `test_rate_limit_multiproc.py` — exercises `SQLiteRateLimitStore` under multi-process contention.

### `cli/`
Tests for the `effgen` CLI surface (e.g. `effgen cost today`, `effgen cost set-budget`).

### `e2e/`
Workflow tests that exercise full agent loops: math, research, coding, calculator. Slower than unit; faster than the parity matrix.

### `tools/`
Provider-native tool surface tests (Anthropic / Gemini native tools — search, code execution, etc.).

### `benchmarks/`
- `baseline.json` — agent-init / latency baselines.
- `eval_baseline_math*.json` — math eval baselines used by `effgen eval`.
- `test_performance.py` — regression guards against the baselines.

### `fixtures/`
Shared inputs:
- `configs/` — sample agent configs.
- `knowledge_base/` — small text corpus for RAG tests.
- `mock_models.py` — in-memory model stand-ins used **only** by unit tests where a real model would add no signal. Live-provider tests never use mocks.

## Running tests

Quick unit pass (no network):
```bash
pytest tests/unit tests/models tests/cli -q
```

Full suite (network calls hit real providers if the keys are set):
```bash
pytest tests/ --timeout=180 -q
```

Pin to one provider's live tests:
```bash
pytest tests/integration/test_cerebras_live.py -v
```

Cross-provider parity matrix:
```bash
pytest tests/integration/parity/ -v
```

## API-key skip behavior

Live tests use the standard `pytest.skip(...)` pattern when the matching key (`CEREBRAS_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`, `REPLICATE_API_TOKEN`, `HF_TOKEN`, `ANTHROPIC_API_KEY`) is absent. Keys are loaded from `.env` via `python-dotenv` at session start; never read inline.

## Markers

- `integration` — needs network/keys.
- `slow` — > 10 s expected.

Run `pytest --markers` for the full list.
