# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.10] - 2026-05-27

### Highlights

**effGen v0.2.10** is the **Security, Edge & Developer Experience** release — hardening effGen end-to-end with secret scanning, dependency auditing, a SBOM pipeline, supply-chain integrity verification, a sandboxed `CodeExecutor`, OAuth2/OIDC auth with RBAC and a per-request audit log, Docker and Helm production deployments, AWS Lambda (Mangum adapter), a Cloudflare Worker edge proxy, a VSCode extension with prompt-template completion, Jupyter magics, and a live local dashboard. No breaking API changes; every security and DX feature is additive.

### Added

#### Security — Secret Scanning (`effgen/security/`, `.gitleaks.toml`, `.pre-commit-config.yaml`)

- **`.gitleaks.toml`** — tuned rule set covering OpenAI, Anthropic, Cerebras, Google, HuggingFace, Groq, Slack, Discord, and Bearer-token patterns. Allowlist for test fixtures with obviously fake keys.
- **`.pre-commit-config.yaml`** — gitleaks pre-commit hook; blocks commits containing secret-like strings.
- **`.github/workflows/secret-scan.yml`** — CI secret-scan workflow; scans both working tree and full git history. Fails on any detected real secret.
- **`tests/security/test_secret_patterns.py`** — plants fake secrets in a temp file, runs gitleaks, asserts detected; repo-history scan asserts clean.

#### Security — SBOM (`sbom.cdx.json`, `.github/workflows/sbom.yml`)

- **`sbom.cdx.json`** — CycloneDX 1.5 SBOM; every runtime dependency listed with name, version, and PURL.
- **`.github/workflows/sbom.yml`** — CI SBOM workflow; generates `sbom.cdx.json` on each push; validates against CycloneDX schema; uploads as release artifact.
- **`tests/security/test_sbom.py`** — asserts every runtime dep from `pyproject.toml` appears in the SBOM.

#### Security — Dependency Pinning (`requirements-lock.txt`, `requirements-all-lock.txt`)

- **`requirements-lock.txt`** — hash-verified lock for base dependencies. `pip install -e . -c requirements-lock.txt` is reproducible.
- **`requirements-all-lock.txt`** — hash-verified lock for `.[all]` extras bucket (uv-generated; `google-protobuf` floors + `fireworks-ai<0.18` cap resolve deep-resolution issues).

#### Security — Vulnerability Audit (`docs/security/`)

- **`.github/workflows/deps-audit.yml`** — `pip-audit` CI; fails on HIGH/CRITICAL; reports MEDIUM. Excludes `fastapi==0.136.3` (known malicious, pinned away in lock).
- **`.github/workflows/sbom.yml`** — SBOM generation and CycloneDX schema validation.
- **Startup hash verification** — `EFFGEN_VERIFY_HASHES=1` on startup compares installed-wheel hashes against the lockfile; logs `hash_verification: ok` or `hash_verification: drift` with the first drifted package.
- **`tests/security/test_vuln_audit.py`** — runs `pip-audit --format json`; asserts no HIGH/CRITICAL in the current environment.
- **`tests/security/test_supply_chain.py`** — asserts `pyproject.toml` contains required fields (`license`, `authors`, `urls`).
- **`docs/security/secrets.md`** — secret-scanning guide and pre-commit setup.
- **`docs/security/sbom.md`** — SBOM generation and validation guide.
- **`docs/security/supply_chain.md`** — supply-chain hardening, hash verification, and Dependabot configuration.

#### Security — Sandbox for CodeExecutor (`effgen/security/sandbox.py`)

- **`SubprocessSandbox`** — rootless user-namespace isolation via `unshare --map-root-user --net --pid --mount`; separate `/tmp` bind-mount; network blocked; no `CAP_SYS_ADMIN` required. Loud warning when falling back from Docker.
- **`DockerSandbox`** — `--read-only --network=none --cap-drop=ALL --pids-limit=100 --memory=256m`; non-root user; automatically selected when Docker daemon is available.
- **`FirecrackerSandbox`** — stub interface; `NotImplementedError` with install instructions for v0.3 roadmap.
- **`OffSandbox`** — `EFFGEN_SANDBOX_BACKEND=off`; executes on host with a loud startup warning; never auto-selected.
- **`SandboxConfig`** — env-driven: `EFFGEN_SANDBOX_BACKEND=docker|subprocess|off`, `EFFGEN_SANDBOX_TIMEOUT=10`.
- **`CodeExecutor.run(code, language)`** — dispatches to the configured sandbox backend.
- **`tests/security/test_sandbox.py`** — network-block test (subprocess path: `urllib.request.urlopen` → `OSError`); filesystem isolation test (`rm -rf /tmp/evil` leaves host untouched). Docker paths skip when daemon unavailable.
- **`docs/security/codeexecutor.md`** — threat model, sandbox architecture, and configuration reference.

#### Auth — OAuth2/OIDC, RBAC, Audit Log (`effgen/server/auth.py`, `effgen/server/rbac.py`, `effgen/server/budget.py`, `effgen/server/audit.py`)

- **OIDC JWT validation** (`effgen/server/auth.py`) — Bearer JWT validation on every non-public endpoint via `authlib`. Configurable issuer, `client_id`, JWKS endpoint via env vars (`EFFGEN_OIDC_ISSUER`, `EFFGEN_OIDC_CLIENT_ID`, `EFFGEN_OIDC_JWKS_URI`). Public endpoints: `/health`, `/metrics` (configurable).
- **RBAC** (`effgen/server/rbac.py`) — `Role(name, allowed_tools, allowed_models, max_cost_per_day)`. JWT claim `roles: [...]` resolved per-request. `RBACBudgetMiddleware` (pure-ASGI, no body-consumption issue) enforces `allowed_tools` (403) and daily cost cap (429 `BudgetExceeded`).
- **Budget tracking** (`effgen/server/budget.py`) — per-principal daily cost accumulation; 429 on cap breach.
- **Audit log** (`effgen/server/audit.py`) — every request/response pair appended to `~/.effgen/audit/<date>.jsonl`. Fields: `ts, principal, role, endpoint, request_summary, response_summary, outcome`. Content redacted via `Redactor`.
- **Dev mode** — `EFFGEN_DEV_MODE=1` disables auth with a loud startup warning. Default off in CI/prod.
- **`tests/server/test_auth.py`** — JWT verify, role matrix, unauthenticated rejected, reader/deny_tools 403, cost-cap 429.
- **`tests/server/test_audit.py`** — audit log fields present, no secrets in log.
- **`docs/server/auth.md`** — OIDC configuration guide (Auth0 walkthrough).
- **`docs/server/rbac.md`** — role definition, JWT claims, tool/model allow-lists.
- **`docs/server/audit.md`** — audit log format, location, rotation.

#### Deploy — Docker (`deploy/docker/Dockerfile`, `docs/deploy/docker.md`)

- **Multi-stage Dockerfile** — `python:3.11-slim` builder + slim runtime; non-root user (`uid=1000`); read-only filesystem; `HEALTHCHECK` on `/health`; regular (non-editable) install with `EXTRAS=server`.
- **`prometheus-client`** added to `server` extras (required for `/metrics`).
- **`tests/deploy/test_dockerfile.py`** — Dockerfile structure: 19 checks (non-root, HEALTHCHECK, EXPOSE, multi-stage); integration path: `docker run --health-cmd` (skipped without Docker group access).
- **`docs/deploy/docker.md`** — one-liner quickstart, environment variable reference, multi-platform build instructions.

#### Deploy — Kubernetes / Helm (`deploy/k8s/helm/effgen/`, `docs/deploy/kubernetes.md`)

- **Helm chart** — `Chart.yaml`, `values.yaml`; templates: `Deployment`, `Service`, `Ingress`, `ConfigMap`, `Secret`, `ServiceAccount`, `NetworkPolicy`, `PodDisruptionBudget`, `HPA` (CPU + `effgen_model_call_latency_seconds`), `PersistentVolumeClaim`.
- Default replicas=2; resource requests `cpu:100m / memory:256Mi`; HPA min=2, max=10.
- **`kubeconform`** validates all rendered manifests against Kubernetes 1.29 strict schema.
- **`tests/deploy/test_helm_lint.py`** — `helm lint` (40 tests); `helm template` produces valid YAML; all resource kinds present; HPA custom metric configured.
- **`docs/deploy/kubernetes.md`** — minikube walkthrough, OIDC secret wiring, HPA tuning guide.

#### Deploy — AWS Lambda (`deploy/aws_lambda/`, `docs/deploy/lambda.md`)

- **`deploy/aws_lambda/handler.py`** — Mangum-adapter wrapping the FastAPI app; `lifespan="off"`; `ProviderRegistry` preloaded at module level for cold-start optimisation (first call < 3 s, warm call < 100 ms); per-invocation timeout budget enforced → 504 on overrun.
- **`deploy/aws_lambda/sam-template.yaml`** — AWS SAM template: HTTP API + Lambda function + CloudWatch Log Group + Outputs. SecretsManager reference for API keys.
- **`deploy/aws_lambda/_smoke_runner.py`** — local smoke runner for live Cerebras calls through the handler.
- **`tests/deploy/test_lambda_handler.py`** — 41 tests: v1 + v2 APIGateway events, 4xx invalid body, 404 unknown path, cold-start timing, timeout 504, SAM struct + cfn-lint.
- **`docs/deploy/lambda.md`** — build zip, deploy via SAM, expected cold-start times, SecretsManager wiring.

#### Deploy — Cloudflare Worker (`deploy/cloudflare/`, `docs/deploy/cloudflare.md`)

- **`deploy/cloudflare/worker.js`** — thin edge proxy: CORS headers, Bearer JWT auth, fixed-window rate limiting (KV-backed), upstream forward with `duplex:"half"` for streaming bodies, security response headers.
- **`deploy/cloudflare/wrangler.toml`** — routes, KV `RATE_LIMIT` namespace, env vars, staging and production environments.
- **`tests/deploy/test_cloudflare_worker.py`** — 30 tests: 11 structural + 9 unit (stubbed) + 2 real-fetch round-trip regression guards (file:// → worker.js → live origin).
- **`docs/deploy/cloudflare.md`** — wrangler deploy guide, KV setup, environment variable reference, JWT configuration.

#### DX — VSCode Extension (`tools/vscode-effgen/`, `docs/dx/vscode.md`)

- **TypeScript extension** (`tools/vscode-effgen/src/extension.ts`) — prompt-template completion from the registry, inline "Run" code lens on `LibraryPrompt` definitions, hover docs with template description + input schema.
- **`npm run compile`** — TypeScript 5.3 strict, 0 errors.
- **`tests/dx/test_vscode_build.py`** — asserts `npm ci && npm run compile` succeed; compiled JS present.
- **`docs/dx/vscode.md`** — install from `.vsix`, feature walkthrough, development guide.

#### DX — Jupyter Magics (`effgen/jupyter/magics.py`, `docs/dx/jupyter.md`)

- **`%effgen_chat <message>`** — one-shot chat; displays formatted response.
- **`%%effgen_agent <preset>`** — cell body as task; displays final answer + tool trace.
- **`%effgen_metrics`** — snapshot of current Prometheus counters inline.
- **`effgen[jupyter]` extra** — `ipython` dependency added.
- **`tests/dx/test_jupyter_magics.py`** — 31 tests: magic loading, `%effgen_chat` with mock model, `%%effgen_agent`, `%effgen_metrics`.
- **`docs/dx/jupyter.md`** — `%load_ext effgen.jupyter`, magic reference, Cerebras live example.

#### DX — Local Dashboard (`effgen/dashboard/`, `docs/dx/dashboard.md`)

- **Static SPA** (`effgen/dashboard/static/`) — served at `/dashboard` (public, no auth required). Panels: live span stream (SSE), `/metrics` summary, recent agent runs with token counts and cost, SLO burn rates. Chart.js from CDN.
- **`/dashboard/data.json`** — `{ts, metrics, slo, recent_runs, recent_spans, raw_metrics}` JSON snapshot.
- **SSE endpoint** at `/dashboard/spans` — pushes new spans in real time.
- **Auth exemption** — `/dashboard` and `/dashboard/*` paths bypassed by `AuthMiddleware`.
- **`tests/dx/test_dashboard.py`** — 46 tests: `/dashboard` 200, panel IDs present, `/dashboard/data.json` schema, SSE stream, auth boundary.
- **`docs/dx/dashboard.md`** — panel reference, SSE protocol, customisation guide.

### Tests Added

| File | Tests | Coverage |
|------|-------|----------|
| `tests/security/test_secret_patterns.py` | 4 | gitleaks planted-secret detection, repo-history clean |
| `tests/security/test_sbom.py` | 4 | SBOM structure, CycloneDX schema, dep coverage |
| `tests/security/test_vuln_audit.py` | 6 | pip-audit JSON, no HIGH/CRITICAL |
| `tests/security/test_supply_chain.py` | 30 | pyproject required fields, hash verification startup |
| `tests/security/test_sandbox.py` | 35 | SubprocessSandbox net+fs isolation; Docker paths skip |
| `tests/server/test_auth.py` | 57 | JWT, RBAC, unauthenticated rejected, 429 budget |
| `tests/server/test_audit.py` | 28 | Audit fields, no secrets, anonymisation |
| `tests/deploy/test_dockerfile.py` | 19 | Dockerfile structure; integration skipped w/o Docker |
| `tests/deploy/test_helm_lint.py` | 40 | helm lint + template, resource kinds, HPA metric |
| `tests/deploy/test_lambda_handler.py` | 41 | v1+v2 events, 4xx, 404, timing, 504, SAM + cfn-lint |
| `tests/deploy/test_cloudflare_worker.py` | 30 | Structural + unit + real-fetch round-trip |
| `tests/dx/test_vscode_build.py` | 20 | npm compile, compiled JS, manifest |
| `tests/dx/test_jupyter_magics.py` | 31 | Magic loading, chat, agent, metrics |
| `tests/dx/test_dashboard.py` | 46 | /dashboard routes, panels, data.json schema, SSE, auth |

### Validation Results

| Check | Result |
|-------|--------|
| `effgen.__version__` | **0.2.10** |
| gitleaks pre-commit | Detects planted secrets ✓ |
| gitleaks CI (dir + history) | Both exit 0 on clean repo ✓ |
| CycloneDX SBOM | Generates + validates against 1.5 schema ✓ |
| pip-audit | 0 HIGH/CRITICAL ✓ |
| `EFFGEN_VERIFY_HASHES=1` | ok/drift logged correctly ✓ |
| SubprocessSandbox network block | `OSError` on `urlopen` ✓ |
| SubprocessSandbox filesystem isolation | Host `/tmp` unchanged ✓ |
| DockerSandbox tests | Skip without Docker group (not an error) ✓ |
| API server (unauthenticated) | 401 in non-dev mode ✓ |
| RBAC deny_tools | 403 ✓ |
| Budget exceeded | 429 `BudgetExceeded` ✓ |
| Audit log | Fields correct, no secrets ✓ |
| Dockerfile | Builds; /health 200 via uvicorn smoke ✓ |
| Helm chart | `helm lint` clean; `kubeconform` strict K8s 1.29 ✓ |
| Lambda handler | 41/41 tests pass; live Cerebras call through handler ✓ |
| Cloudflare Worker | wrangler --dry-run validates; live round-trip via worker ✓ |
| VSCode extension | `npm run compile` 0 errors; .vsix buildable ✓ |
| Jupyter magics | Live Cerebras llama3.1-8b smoke ✓ |
| Dashboard | /dashboard 200 + /dashboard/data.json valid JSON ✓ |
| Full regression suite | **3721 passed, 0 failed** (88 skipped, 14 xfailed) ✓ |
| Wheel build | `effgen-0.2.10-py3-none-any.whl` built cleanly ✓ |
| Wheel smoke | `python -c "import effgen; assert effgen.__version__ == '0.2.10'"` ✓ |

### Upgrading from v0.2.9

No breaking API changes. All security and DX features are additive.

```bash
pip install --upgrade effgen
```

#### Security Quick Start

```python
# CodeExecutor now sandboxed by default (DockerSandbox if available, else SubprocessSandbox)
import asyncio
from effgen.security.sandbox import get_sandbox, SandboxConfig

async def main():
    config = SandboxConfig(backend="subprocess", timeout=10)
    sandbox = await get_sandbox(config)
    result = await sandbox.run('print("hello")', "python", config)
    print(result.stdout)  # hello

asyncio.run(main())

# Or use EFFGEN_SANDBOX_BACKEND=docker for Docker isolation
```

#### Auth Quick Start

```bash
# Start server with OIDC auth
export EFFGEN_OIDC_ISSUER=https://your-auth0-domain.auth0.com/
export EFFGEN_OIDC_CLIENT_ID=your-client-id
export EFFGEN_OIDC_JWKS_URI=https://your-auth0-domain.auth0.com/.well-known/jwks.json
effgen serve --port 8000

# Dev mode (auth disabled — for local development only)
EFFGEN_DEV_MODE=1 effgen serve --port 8000
```

#### Deploy Quick Start

```bash
# Docker
docker build -f deploy/docker/Dockerfile -t effgen:0.2.10 .
docker run -p 8000:8000 --env-file .env effgen:0.2.10

# Helm (Kubernetes)
helm install effgen deploy/k8s/helm/effgen/ -f deploy/k8s/helm/effgen/values.yaml

# AWS Lambda
cd deploy/aws_lambda && sam build && sam deploy

# Cloudflare Worker edge proxy
cd deploy/cloudflare && wrangler deploy
```

#### DX Quick Start

```python
# Jupyter
%load_ext effgen.jupyter
%effgen_chat "What is 17 * 23?"
%%effgen_agent general
Summarise the top HackerNews stories today.

# Python API
from effgen.jupyter.magics import EffgenMagics
```

---

## [0.2.9] - 2026-05-23

### Highlights

**effGen v0.2.9** is the **Observability & Reliability** release — turning effGen into something you can operate in production. Structured JSON logs with secret redaction, OpenTelemetry tracing with configurable samplers, Prometheus histograms, SLO tracking, circuit breakers, bulkheads, jittered retries, timeout propagation, a deterministic chaos harness, a Hypothesis fuzz suite, a load-testing harness with CLI, and Alertmanager-compatible alert rules ship in this release. No breaking API changes; every telemetry path is async/non-blocking.

### Added

#### Observability — Structured Logging (`effgen/observability/logs.py`)

- **`StructuredFormatter`** — emits JSON lines `{ts, level, module, event, attributes, trace_id, span_id}` with OTel span-context integration. Every log line is structured; no ad-hoc `print()` in critical paths.
- **`get_logger(__name__)` helper** — `log.event("model.call.started", model=..., cached_tokens=128)` API; module-level structured logger with trace-context injection.
- **Migration pass** — agent loop, model adapters, router, and tool call sites migrated from ad-hoc `print`/`logger.info` to the structured logger.

#### Observability — Secret Redaction (`effgen/observability/redact.py`)

- **`Redactor`** — built-in patterns for OpenAI (`sk-`), Anthropic (`sk-ant-`), Cerebras (`csk-`), Google (`AIza`), HuggingFace (`hf_`), Groq (`gsk_`), Bearer tokens, Slack webhook URLs, Discord webhook URLs. Replaces with `<REDACTED:openai_key>` etc. User-extensible via `Redactor.add_pattern(name, regex)`.
- **Applied at the log encoder** — every path is covered; secrets never appear in log output.

#### Observability — Metrics: Histograms + SLO (`effgen/observability/metrics.py`, `effgen/observability/slo.py`)

- **`effgen_model_call_latency_seconds{provider,model,outcome}`** — Histogram (buckets 50 ms–60 s).
- **`effgen_tool_call_latency_seconds{tool,outcome}`** — Histogram.
- **`effgen_agent_iteration_latency_seconds{preset}`** — Histogram.
- **`effgen_tokens_total{provider,model,kind}`** — Counter (kind ∈ input, output, cached).
- **`SLO(name, target_pct, window_seconds, query)`** and **`SLOTracker`** — rolling-window error-budget tracking. `burn_rate(name)` returns the ratio against target. `/slo` endpoint on the FastAPI server.

#### Observability — Tracing: Sampling + Span Spec (`effgen/observability/tracing.py`, `effgen/observability/spans.py`)

- **Samplers** — `AlwaysOn`, `AlwaysOff`, `ParentBased(TraceIdRatio(p))`, `ParentBased(RateLimited(per_second))` configurable via `ObservabilityConfig.tracing.sampler`. No implicit `head=1.0` in production.
- **Canonical span-attribute spec** (`effgen/observability/spans.py`) — single source of truth for every attribute name: `effgen.agent.*`, `effgen.model.*`, `effgen.tool.*`, `effgen.router.*`, `effgen.retry.*`.
- **Span emission** — all adapters, tools, and router decisions emit correct spans with declared attributes; multimodal `effgen.model.parts_count` where relevant.

#### Reliability — Timeouts (`effgen/reliability/timeouts.py`)

- **`ReliabilityConfig.default_timeouts`** — `{model_call: 60, tool_call: 30, http: 20}`. Propagated into adapter `httpx` clients + tool executions.
- **`with_timeout`, `async_timeout`, `apply_timeout`** wrappers; `audit_no_none_timeouts()` guard. Every adapter missing an explicit timeout fails the timeout audit test.

#### Reliability — Retries (`effgen/reliability/retry.py`)

- **`Retry(max_attempts, base_delay, max_delay, jitter, retryable)`** — configurable policy.
- **`@retryable(Retry(...))`** decorator with jittered exponential backoff.
- Emits `effgen.retry.attempt` OTel span event per retry attempt.
- Default policy: transient network / 5xx / 429 after Retry-After header.

#### Reliability — Circuit Breaker (`effgen/reliability/circuit.py`)

- **`CircuitBreaker(name, failure_threshold, recovery_timeout, half_open_probes)`** — three-state (CLOSED → OPEN → HALF_OPEN) per-provider breaker.
- **`CircuitBreakerRegistry`** — one breaker per provider, wired into `ProviderRegistry`.

#### Reliability — Bulkhead (`effgen/reliability/bulkhead.py`)

- **`Bulkhead(name, max_concurrency, queue_size, queue_timeout)`** — semaphore-based concurrency limiter with bounded queue, sync + async variants.
- **`BulkheadRegistry`** — one bulkhead per provider so one misbehaving provider can't starve the others.

#### Chaos Harness (`effgen/reliability/chaos.py`)

- **`Chaos(seed)`** — deterministic fault injection with reproducible outcomes across seeds.
- **Fault types**: `NetworkTimeout`, `Http5xx`, `Http429(retry_after)`, `SlowResponse(ms)`, `PartialResponse`, `MalformedJSON`.
- **`registry.with_chaos(Chaos(...))`** — attaches as middleware to `ProviderRegistry`.
- **4 canonical scenarios** validated: A (5xx fallback), B (429 + Retry-After), C (SlowResponse timeout), D (AllProvidersFailed — no silent empty string).

#### Load-Testing Harness (`effgen/tools/loadgen.py`, `effgen/cli/loadtest.py`)

- **`effgen loadtest`** CLI — concurrency, duration, scenario (fixed/synthetic/multi_tool).
- Reports throughput, p50/p95/p99 latency, error rate to stdout (JSON) or file.
- Runs against local mock model by default; `--provider` switches to live inference.
- Live smoke: 30 s, c=10, mock model → ~69 k req, 0% error, p95 ≈ 4.3 ms.

#### Alerting (`effgen/observability/alerting.py`, `docs/observability/alert_rules.yaml`)

- **6 Alertmanager-compatible rules**: `HighErrorRate` (>5% for 10 min), `HighP95Latency` (>10 s for 5 min), `CostBurnHigh` (>$10/day), `SLOFastBurn` (>14.4× error budget), `SLOSlowBurn`, `CircuitBreakerOpen`.
- **`AlertWebhook(url).fire(alert)`** — posts to Slack/Discord via Phase v0.2.6 webhook tools; generic `httpx` fallback; non-raising (fire never throws).
- Webhook URL redacted in logs.

#### Documentation (`docs/observability/`)

- `overview.md` — architecture, quickstart, configuration reference.
- `metrics.md` — all metrics with label dimensions and bucket definitions.
- `tracing.md` — sampler selection guide, span attribute spec.
- `alerting.md` — Alertmanager integration, webhook configuration.
- `loadtest.md` — load-testing harness guide with examples.

### Tests Added

| File | Coverage |
|------|----------|
| `tests/observability/test_logs.py` | JSON shape, trace_id propagation, 26 tests |
| `tests/observability/test_redact.py` | Every pattern on known fixtures, 37 tests |
| `tests/observability/test_metrics.py` | Histograms, counters, Prometheus text format |
| `tests/observability/test_slo.py` | Rolling-window math, burn-rate formula |
| `tests/observability/test_tracing.py` | In-memory span exporter, 3-tool agent span tree |
| `tests/reliability/test_timeouts.py` | Timeout wrappers, adapter audit (0 `timeout=None` violations) |
| `tests/reliability/test_retry.py` | Jitter, backoff, Retry-After, OTel event emission |
| `tests/reliability/test_circuit.py` | CLOSED→OPEN→HALF_OPEN state machine |
| `tests/reliability/test_bulkhead.py` | Concurrency limits, queue overflow, sync + async |
| `tests/reliability/test_chaos.py` | 4 scenarios × 10 seeds, 273 tests — all deterministic |
| `tests/fuzz/test_tool_fuzz.py` | All 66 BaseTool subclasses × 500 examples, no secret leaks |
| `tests/fuzz/test_message_fuzz.py` | Random ContentPart sequences, no unhandled exceptions |
| `tests/fuzz/test_router_fuzz.py` | Random availability + capabilities → valid decision or NoEligibleProvider |
| `tests/tools/test_loadgen.py` | Loadgen library + CLI, 47 tests |
| `tests/observability/test_alerting.py` | Alert rules, AlertWebhook, 34 tests |

### Validation Results

| Check | Result |
|-------|--------|
| `effgen.__version__` | **0.2.9** |
| Every log line in `agent.run()` | JSON + no raw secrets ✓ |
| `/metrics` scrape | Prometheus-valid histograms ✓ |
| SLO burn-rate math | Spot-checked against rolling-window fixtures ✓ |
| Span tree (Calculator+WebSearch, 3-tool) | All declared attributes present ✓ |
| Timeout audit | 0 `timeout=None` violations in source ✓ |
| Breaker CLOSED→OPEN→HALF_OPEN | Verified across synthetic faults ✓ |
| Bulkhead concurrency limit | Verified sync + async ✓ |
| Chaos Scenario A–D × 10 seeds | 273/273 pass, deterministic ✓ |
| Fuzz × 500 examples per test | 164/164 pass, no unhandled exceptions ✓ |
| Load harness mock smoke (c=10, 30 s) | ~69 k req, 0% error, p95 ≈ 4.3 ms ✓ |
| Alert rules YAML | Syntactically valid, 6 rules ✓ |
| Regression suite | All prior tests pass (p=1300+, f=0) ✓ |
| Wheel build | `effgen-0.2.9-py3-none-any.whl` built cleanly ✓ |
| Wheel smoke | `python -c "import effgen; assert effgen.__version__ == '0.2.9'"` ✓ |

### Upgrading from v0.2.8

No breaking API changes. All observability and reliability features are additive.

```bash
pip install --upgrade effgen
```

#### Observability Quick Start

```python
from effgen.observability import get_logger
from effgen.observability import record_model_call, export_metrics
from effgen.observability.slo import SLOTracker, SLO

log = get_logger(__name__)
log.event("agent.started", preset="general", model="llama3.1-8b")

# Histograms auto-record on agent/model/tool calls; you can also record directly:
record_model_call(provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.42)
print(export_metrics())  # Prometheus text format

tracker = SLOTracker()
tracker.register(SLO("model_success", target_pct=99.0, window_seconds=3600))
tracker.record("model_success", ok=True)
print(tracker.burn_rate("model_success"))  # e.g. 0.0 when all calls succeed
```

#### Reliability Quick Start

```python
from effgen.reliability.retry import Retry, retryable
from effgen.reliability.circuit import CircuitBreaker
from effgen.reliability.bulkhead import Bulkhead

@retryable(Retry(max_attempts=3, base_delay=0.5, jitter=True))
def call_model(prompt):
    ...

breaker = CircuitBreaker("cerebras", failure_threshold=5, recovery_timeout=30)
if breaker.is_call_permitted():
    try:
        result = call_model("hello")
        breaker.on_success()
    except Exception as exc:
        breaker.on_failure(exc)
        raise

bulkhead = Bulkhead("cerebras", max_concurrency=10, queue_size=50)
with bulkhead.acquire():
    call_model("hello")
```

---

## [0.2.8] - 2026-05-21

### Highlights

**effGen v0.2.8** is the **Multimodal Input** release — image, audio, and video are now first-class citizens across 6 cloud providers (Gemini, OpenAI, Groq, Anthropic, Together, HF). A unified `Message` content schema, provider-specific adapters, automatic preprocessing (resize, downsample, frame-sampling), capability-gating errors, a new `multimodal` preset, `MultimodalDescribeTool`, a local MLX-VLM adapter (Apple Silicon), and 5 cookbook walkthroughs ship in this release. No breaking API changes.

### Added

#### Core — Unified Message Schema (`effgen/core/messages.py`)

- **Structured `ContentPart` union** — `TextPart`, `ImagePart`, `AudioPart`, `VideoPart`, `ToolCallPart`, `ToolResultPart` form the typed `ContentPart` union. `Message.content` is always `List[ContentPart]`.
- **Backwards-compatible constructor** — `Message(role, "text string")` auto-wraps in `TextPart`; `Message.text` property joins all text parts; `Message.from_str(text)` classmethod.
- **Validation on construction** — `ImagePart` validates MIME ∈ {image/png, image/jpeg, image/gif, image/webp}; `AudioPart` validates MIME ∈ {audio/mp3, audio/wav, audio/flac, audio/ogg, audio/m4a}; `VideoPart.frames` must be non-empty. Raises `InvalidMultimodalContent` on failure.

#### Core — Multimodal Helpers (`effgen/core/multimodal.py`)

- **`image_from(source) → ImagePart`** — accepts `bytes`, local path, URL, `PIL.Image`, `np.ndarray`. MIME sniffed automatically.
- **`audio_from(source) → AudioPart`** — accepts `bytes`, local path, URL. Duration extracted from metadata when available.
- **`video_from(source, fps=1) → VideoPart`** — accepts `bytes`, local path, URL; samples keyframes at `fps` via ffmpeg; raises `MissingSystemDependency` with install hints if ffmpeg is absent.

#### Multimodal Preprocessing

- **`effgen/multimodal/image_pre.py`** — `prepare(part, provider, model) → ImagePart`. Applies per-provider constraints (max bytes, max pixel dims, supported MIMEs); PIL Lanczos downscale when needed; records preprocessing steps in `part.meta["preprocessing"]` for observability.
- **`effgen/multimodal/audio_pre.py`** — Downsamples to 16 kHz mono if provider requires; chunks audio longer than provider max duration into sequential requests and concatenates results. Uses `pydub`.
- **`effgen/multimodal/video_pre.py`** — `VideoSource(path_or_url).sample_frames(fps, max_frames) → List[ImagePart]`; `VideoSource.extract_audio() → AudioPart | None`. Raises `MissingSystemDependency("ffmpeg", ...)` when ffmpeg is absent.

#### Provider Adapters — Image Input

- **Gemini** — native `inline_data` image parts (base64 + mime_type); all Gemini 2.x/3.x vision models.
- **OpenAI** — `content: [{type: "image_url"}]` format for gpt-4o family; base64 data-URL encoding.
- **Anthropic** — base64 media blocks in content list (code-only; live tests skipped — no key in dev env).
- **Groq** — Llama 4 / Llama 3.2 vision model support via image_url content blocks.
- **Together** — vision-capable Together models via image_url content blocks.
- **HuggingFace Inference** — BLIP / LLaVA family via multimodal inference payload.
- **Capability gating** — every adapter raises `CapabilityNotSupportedError(Capability.vision)` when the selected model doesn't support images; no silent text downcast.

#### Provider Adapters — Audio Input

- **Gemini** — native `Part.from_bytes(audio_bytes, mime_type)` inline audio; full conversation with audio context.
- **OpenAI** — Whisper via `/audio/transcriptions` (`transcribe_audio()` method) + gpt-4o audio in chat completions.
- **HuggingFace Inference** — `automatic_speech_recognition` task endpoint.
- **Anthropic** — raises `CapabilityNotSupportedError(Capability.audio_input)` (no audio support).

#### Provider Adapters — Video Input

- **Gemini** — native video inline data for Gemini 2.x/3.x; video MIME passed directly.
- **All others** — `VideoPart` converted to sequence of `ImagePart`s (frame sampling) + optional `AudioPart` (from audio track); sent as multi-image message.

#### New Preset: `multimodal` (`effgen/presets/multimodal.py`)

- **Primary model** — Gemini Flash-Lite (vision + audio + video).
- **Fallback** — OpenAI gpt-4o-mini (vision), HF BLIP (vision-only).
- **Tools** — `ImageInfoTool`, `ImageCaptionTool`, `OCRTool`, `AudioTranscribeTool`, `PDFTool`, `WeatherTool`, `MultimodalDescribeTool`.
- **`MultimodalDescribeTool`** — auto-selects between `ImageCaption`, `OCR`, and `AudioTranscribe` based on the input part type; returns structured description.
- **`create_agent("multimodal", model=...)`** — factory wires the preset end-to-end.

#### Local MLX-VLM Adapter (`effgen/models/mlx_vlm_engine.py`)

- Thin wrapper around the `mlx-vlm` library for Apple Silicon vision-language inference.
- Raises `MissingSystemDependency` on non-Apple-Silicon / missing `mlx-vlm`. Live tests skipped on Linux; fully unit-tested with fakes.

#### Cookbook (`docs/cookbook/`)

- **`multimodal_01_image_qa.md`** — image Q&A walk-through with Gemini and OpenAI.
- **`multimodal_02_audio_transcribe_reason.md`** — audio → transcript → sentiment analysis.
- **`multimodal_03_video_summarize.md`** — video → keyframes → narrative summary.
- **`multimodal_04_ocr_plus_llm.md`** — OCR text extraction then structured extraction via `contract_summarize_v1` prompt.
- **`multimodal_05_bullet_chart_read.md`** — read a bar chart from an image and answer comparison questions.
- **`docs/cookbook/README.md`** — index of all cookbook walkthroughs with quick-start links.

#### Documentation

- **`docs/multimodal/overview.md`** — unified Message schema, ContentPart types, capability gating, provider support matrix, preprocessing pipeline, and quick-start examples.
- **`docs/multimodal/images.md`** — per-provider image input guide.
- **`docs/multimodal/audio.md`** — per-provider audio input guide.
- **`docs/multimodal/video.md`** — video frame-sampling and native video path guide.

### Tests Added

| File | Coverage |
|------|----------|
| `tests/core/test_message_schema.py` | ContentPart construction, validation, back-compat |
| `tests/core/test_multimodal_helpers.py` | `image_from`, `audio_from`, `video_from` on all source types |
| `tests/core/test_image_input.py` | Adapter translation per provider (fakes) |
| `tests/core/test_audio_input.py` | Audio adapter translation per provider (fakes) |
| `tests/core/test_video_input.py` | VideoPart → ImagePart fallback; Gemini native path (fakes) |
| `tests/multimodal/test_image_pre.py` | Resize, MIME, size constraints |
| `tests/multimodal/test_audio_pre.py` | Chunking, downsample |
| `tests/multimodal/test_video_pre.py` | ffmpeg missing → clean error; frame sampling rates |
| `tests/presets/test_multimodal.py` | Preset construction, tool wiring, MultimodalDescribeTool |
| `tests/models/test_mlx_vlm.py` | MLX-VLM adapter unit tests (28 tests) |
| `tests/cookbook/test_cookbook_runs.py` | Cookbook snippet extraction, `pytest.mark.live` gating |

### Validation Results

| Check | Result |
|-------|--------|
| `effgen.__version__` | **0.2.8** |
| Image input — live | Gemini ✓, OpenAI ✓, Groq ✓ (≥3 providers) |
| Audio input — live | Gemini ✓, OpenAI Whisper ✓ |
| Video input — live | Gemini native ✓, OpenAI frame-sampling ✓ |
| Multimodal preset — live | image ✓, audio ✓, video ✓ (all 3 modalities) |
| Cookbook live runs | 7/8 pass (1 skipped — ffmpeg not installed in test env) |
| `CapabilityNotSupportedError` | Raised cleanly on vision-incapable provider/model |
| `MissingSystemDependency("ffmpeg")` | Raised with install hints when ffmpeg absent |
| Wheel build | `effgen-0.2.8-py3-none-any.whl` built cleanly |
| Wheel smoke | `python -c "import effgen; assert effgen.__version__ == '0.2.8'"` ✓ |
| Regression suite | All prior tests pass (p=2489+, f=0) |

### Upgrading from v0.2.7

No breaking API changes. The old `Message(role, content: str)` constructor still works.

```bash
pip install --upgrade effgen
```

#### Quick Start

```python
from effgen import image_from, audio_from
from effgen.core.messages import Message, Role
from effgen.presets import create_agent
from effgen import load_model

model = load_model("gemini-2.0-flash", provider="gemini")
agent = create_agent("multimodal", model)

# Image Q&A
img = image_from("https://example.com/photo.jpg")
msg = Message(role=Role.USER, content=[img, "Describe this image."])
result = agent.run_message(msg)

# Audio
aud = audio_from("/tmp/recording.mp3")
msg = Message(role=Role.USER, content=[aud, "Transcribe and give the sentiment."])
result = agent.run_message(msg)
```

---

## [0.2.7] - 2026-05-20

### Highlights

**effGen v0.2.7** is the **Prompt Library** release — a curated, domain-organized catalog of **31 reusable prompt templates** across 7 domains, paired with a golden evaluation harness, a rich CLI, and an interactive playground. No breaking API changes.

### Added

#### Prompt Library (`effgen/prompts/library/`)

- **`LibraryPrompt` dataclass** (`base.py`) — structured prompt definition with `name`, `domain`, `variant`, `description`, `template` (callable), `input_schema` (JSON Schema), `fixture`, `expected_shape`, and `tags`. Fully validated on registration.
- **`PromptRegistry` singleton** (`registry.py`) — auto-discovers all domain packages under `effgen/prompts/library/domains/` at startup; `register`, `get`, `search`, `all`, `domains`, `__len__`.
- **`PromptEval` harness** (`eval.py`) — `eval_golden` (renders with fixture, compares against `.txt` golden, writes on first run); `eval_live` (renders + runs via model, checks `expected_shape`); `eval_all_golden` with pass/fail table.
- **CLI** — `effgen prompts list [--domain X] [--variant Y] [--format table|json|markdown]`, `effgen prompts show <name>`, `effgen prompts eval [--domain X] [--live --model M]`.

#### Research Domain (`effgen/prompts/library/domains/research/`)

- **`research.literature_review.v1.zero_shot`** — zero-shot literature review; inputs: `topic`, `years_range`, `max_papers`.
- **`research.literature_review.v1.cot`** — chain-of-thought literature review with step-by-step reasoning.
- **`research.paper_summary.v1`** — structured output: `{abstract_summary, key_findings, limitations, future_work}`.
- **`research.citation_extract.v1`** — tool-augmented; instructs agent to retrieve live ArXiv/PubMed metadata.
- **`research.methodology_critique.v1`** — CoT critique covering design, sampling, measurement, analysis, generalizability.

#### Coding Domain (`effgen/prompts/library/domains/coding/`)

- **`coding.code_review.v1`** — structured output: `{issues: [{severity, location, suggestion}]}`.
- **`coding.bug_diagnose.v1`** — CoT diagnosis; inputs: `code`, `error_message`, `repro_steps`.
- **`coding.refactor_plan.v1`** — tool-augmented; reads the source file then produces a structured plan with risk assessment.
- **`coding.test_generate.v1`** — few-shot; two exemplar pytest suites; live eval asserts `ast.parse()` passes on generated Python.
- **`coding.docstring_fill.v1`** — zero-shot; adds Google/NumPy/Sphinx-style docstrings to undocumented functions.

#### Data Domain (`effgen/prompts/library/domains/data/`)

- **`data.sql_from_nl.v1`** — structured output: `{sql, warnings[]}`; inputs: `schema_ddl`, `question`, `dialect`; live eval validates via `sqlglot.parse()`.
- **`data.sql_explain.v1`** — zero-shot; explains SQL in plain English for developer or business audience.
- **`data.sql_optimize.v1`** — CoT; identifies anti-patterns, explains execution impact, produces rewritten query and index hints.
- **`data.data_profile.v1`** — tool-augmented; takes ExcelTool/CSV column stats, produces structured data-quality report.
- **`data.etl_plan.v1`** — few-shot; two exemplar ETL designs covering Extract → Transform → Load → Validate → Cleanup.

#### Legal Domain (`effgen/prompts/library/domains/legal/`)

> All legal prompts include the verbatim disclaimer: *"This output is for informational purposes only and does not constitute legal advice. Consult a qualified attorney for guidance specific to your situation."*

- **`legal.contract_summarize.v1`** — structured output: `{parties, term, obligations, termination, risks}`.
- **`legal.clause_classify.v1`** — zero-shot clause classification with characteristic flags.
- **`legal.legal_research_brief.v1`** — tool-augmented; produces structured research brief grounded in pre-retrieved sources.

#### Medical Domain (`effgen/prompts/library/domains/medical/`)

> All medical prompts include the verbatim disclaimer: *"This output is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional."*

- **`medical.symptom_triage.v1`** — structured output with mandatory `disclaimer` field and `see_doctor_if` list.
- **`medical.drug_interaction_query.v1`** — structured output with severity levels and recommendations.
- **`medical.medical_literature.v1`** — tool-augmented; synthesizes retrieved PubMed abstracts into a clinical evidence brief.

#### Creative Domain (`effgen/prompts/library/domains/creative/`)

- **`creative.story_continuation.v1.zero_shot`** — zero-shot story continuation maintaining genre and tone.
- **`creative.story_continuation.v1.few_shot`** — few-shot with craft exemplars from multiple genres.
- **`creative.poetry_forms.v1`** — few-shot with exemplars for haiku, sonnet, and free verse; inputs: `theme`, `form`, `mood`.
- **`creative.character_bio.v1`** — structured output: `{name, age, background, personality_traits, goals, flaws, relationships}`.
- **`creative.world_building.v1`** — CoT; develops geography, politics, magic/tech, culture, and story hooks step by step.

#### Business Domain (`effgen/prompts/library/domains/business/`)

- **`business.meeting_summary.v1`** — structured output: `{decisions, action_items[{owner, item, due}], risks}`; inputs: `transcript`, `meeting_title`, `attendees`.
- **`business.email_draft.v1`** — few-shot; two tone exemplars (formal, casual); inputs: `purpose`, `recipient`, `key_points`, `tone`.
- **`business.okr_generate.v1`** — CoT; produces aligned objectives and measurable key results from mission and strategic priorities.
- **`business.swot_analysis.v1`** — structured output: `{strengths, weaknesses, opportunities, threats, strategic_insights}`; perspective-aware.
- **`business.elevator_pitch.v1`** — zero-shot; strict ≤150-word constraint; live eval asserts word count.

#### Playground CLI (`effgen prompts playground`)

- **Interactive REPL** — `select`, `set`, `render`, `run`, `save`, `list`, `show`, `help`, `quit` commands.
- **Non-interactive mode** — `effgen prompts render <name> [--input input.json]` and `effgen prompts run <name> [--input input.json] [--model M]`.
- **Session persistence** — sessions saved to `~/.effgen/playground/<timestamp>.json`; `effgen prompts playground --load <session>` reloads.
- **Hot-reload** — template edits are picked up without REPL restart (importlib-based re-import).

#### Gallery Doc

- **`docs/prompts/gallery.md`** — auto-generated from registry; one row per template with name, domain, variant, and description. Regenerate with `effgen prompts list --format markdown`.

### Tests

- `tests/prompts/test_registry.py` — discovery, search, validation.
- `tests/prompts/test_eval.py` — golden and live eval harness.
- `tests/prompts/test_research.py`, `test_coding.py`, `test_data.py`, `test_legal.py`, `test_medical.py`, `test_creative.py`, `test_business.py` — domain golden + live checks.
- `tests/prompts/test_playground.py` — scripted non-interactive walk-through.

### Documentation

- `docs/prompts/library.md` — framework overview, key classes, CLI reference, adding-new-domain guide.
- `docs/prompts/research.md`, `coding.md`, `data.md`, `legal.md`, `medical.md`, `creative.md`, `business.md`, `playground.md` — per-domain guides.

---

## [0.2.6] - 2026-05-19

### Highlights

**effGen v0.2.6** is a document, media, and communication tools release adding **14 new built-in tools** across six categories — OCR, audio transcription, image analysis, document parsing, geo/weather, and email/webhook communication — raising the total built-in tool count from 44 to **58+**. Two new presets (`media`, `notify`) are introduced. Every tool follows the established `BaseTool` pattern with structured `{success, data, error}` output, async `_execute()`, unit + integration tests, a dedicated doc page, and preset integration. No breaking API changes.

### Added

#### OCR Tools
- **`OCRTool`** (`effgen/tools/builtin/ocr.py`) — extract text from images using Tesseract (local, primary) with OCR.space free API as fallback (`OCR_SPACE_API_KEY`). Raises `OCRBackendUnavailable` with per-OS install instructions when no backend is available. Operations: `extract`, `extract_regions`. Added to `general` preset.

  ```python
  from effgen.tools.builtin.ocr import OCRTool
  result = OCRTool().execute({"operation": "extract", "image_path": "/tmp/scan.png", "lang": "eng"})
  print(result["data"]["text"])
  ```

  **System dep install:**
  ```bash
  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr
  # macOS
  brew install tesseract
  # Windows
  choco install tesseract
  ```

#### Audio Transcription Tools
- **`AudioTranscribeTool`** (`effgen/tools/builtin/audio_transcribe.py`) — transcribe audio files locally via `faster-whisper` (CPU/GPU auto-detected) with HuggingFace Inference fallback (`HF_TOKEN`). Detects GPU via `nvidia-smi`; warns when `model_size > "base"` on CPU. Operations: `transcribe`. Added to `media` preset.

  ```python
  from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
  result = AudioTranscribeTool().execute({"operation": "transcribe", "audio_path": "/tmp/clip.mp3", "model_size": "base"})
  print(result["data"]["text"])
  ```

  **System dep install (for non-WAV formats):**
  ```bash
  sudo apt-get install ffmpeg   # Ubuntu/Debian
  brew install ffmpeg           # macOS
  ```

#### Image Analysis Tools
- **`ImageInfoTool`** (`effgen/tools/builtin/image_info.py`) — extract image metadata (size, format, mode, EXIF, color histogram) and perform local resize/thumbnail operations using Pillow. Zero network calls. Operations: `info`, `resize`, `thumbnail`. Added to `general` preset.

  ```python
  from effgen.tools.builtin.image_info import ImageInfoTool
  result = ImageInfoTool().execute({"operation": "info", "image_path": "/tmp/photo.jpg"})
  print(result["data"]["size"], result["data"]["format"])
  ```

- **`ImageCaptionTool`** (`effgen/tools/builtin/image_caption.py`) — generate natural-language descriptions of images via the effGen model router (selects a vision-capable provider: Gemini, OpenAI, or MLX-VLM). Raises `NoVisionProviderAvailable` when no vision-capable provider is configured. Operations: `caption`, `describe`. Added to `media` preset.

  ```python
  from effgen.tools.builtin.image_caption import ImageCaptionTool
  result = ImageCaptionTool().execute({"operation": "caption", "image_path": "/tmp/photo.jpg"})
  print(result["data"]["caption"])
  ```

#### Document Parsing Tools
- **`PDFTool`** (`effgen/tools/builtin/pdf.py`) — extract text, tables, and metadata from PDF files using `pypdf` (primary) with `pdfplumber` for structured table extraction. Operations: `text`, `metadata`, `tables`, `extract_images`. Added to `research` and `general` presets.

  ```python
  from effgen.tools.builtin.pdf import PDFTool
  result = PDFTool().execute({"operation": "text", "path": "/tmp/paper.pdf"})
  print(result["data"]["text"][:500])
  ```

- **`DOCXTool`** (`effgen/tools/builtin/docx.py`) — parse Word documents (`.docx`) using `python-docx`. Operations: `text`, `paragraphs`, `tables`, `metadata`. Added to `research` and `general` presets.

  ```python
  from effgen.tools.builtin.docx import DOCXTool
  result = DOCXTool().execute({"operation": "text", "path": "/tmp/report.docx"})
  print(result["data"]["text"])
  ```

- **`ExcelTool`** (`effgen/tools/builtin/excel.py`) — read Excel workbooks (`.xlsx`) using `openpyxl` with tabular DataFrame output via `pandas`. Operations: `sheets`, `read_sheet`, `headers`. Added to `research` and `general` presets.

  ```python
  from effgen.tools.builtin.excel import ExcelTool
  result = ExcelTool().execute({"operation": "read_sheet", "path": "/tmp/data.xlsx", "sheet_name": "Sheet1"})
  print(result["data"]["rows"][:3])
  ```

#### Geo / Weather Tools
- **`WeatherTool`** (`effgen/tools/builtin/weather.py`) — fetch current conditions, forecasts, and historical weather data from Open-Meteo (free, no auth required). Integrates with `GeocodeTool` for place-name → lat/lon resolution. Operations: `current`, `forecast`, `historical`. Added to `general` preset.

  ```python
  from effgen.tools.builtin.weather import WeatherTool
  result = WeatherTool().execute({"operation": "current", "lat": 37.42, "lon": -122.08})
  print(result["data"]["temperature_c"], result["data"]["weather_description"])
  ```

- **`GeocodeTool`** (`effgen/tools/builtin/geocode.py`) — forward/reverse geocoding using Nominatim (OpenStreetMap). Sets `effGen/<version>` User-Agent as required; built-in 1 req/s token-bucket rate limiter. Operations: `geocode`, `reverse`. Added to `general` preset.

  ```python
  from effgen.tools.builtin.geocode import GeocodeTool
  result = GeocodeTool().execute({"operation": "geocode", "address": "1600 Amphitheatre Pkwy, Mountain View, CA"})
  print(result["data"]["lat"], result["data"]["lon"])
  ```

- **`MapsTool`** (`effgen/tools/builtin/maps.py`) — render static PNG maps from OpenStreetMap tiles using the `staticmap` library. Operations: `render`, `bounding_box`. Added to `general` preset.

  ```python
  from effgen.tools.builtin.maps import MapsTool
  result = MapsTool().execute({"operation": "render", "lat": 37.42, "lon": -122.08, "zoom": 13, "dest": "/tmp/map.png"})
  print(result["data"]["path"])
  ```

#### Email Tools
- **`EmailSMTPTool`** (`effgen/tools/builtin/email_smtp.py`) — send email via SMTP using stdlib `smtplib`. TLS-on by default. Config: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`. Raises `MissingCredentialsError` when config is absent. Operations: `send`. Added to `notify` preset.

  ```python
  from effgen.tools.builtin.email_smtp import EmailSMTPTool
  result = EmailSMTPTool().execute({"operation": "send", "to": "alice@example.com", "subject": "Hello", "body": "Hi there!"})
  ```

- **`EmailIMAPTool`** (`effgen/tools/builtin/email_imap.py`) — read email via IMAP using stdlib `imaplib`. Config: `IMAP_HOST`, `IMAP_PORT`, `IMAP_USER`, `IMAP_PASSWORD`. Operations: `list_folders`, `fetch_recent`, `search`, `get`. Added to `notify` preset.

  ```python
  from effgen.tools.builtin.email_imap import EmailIMAPTool
  result = EmailIMAPTool().execute({"operation": "fetch_recent", "folder": "INBOX", "n": 5})
  for msg in result["data"]["messages"]:
      print(msg["subject"], msg["from"])
  ```

#### Webhook Tools
- **`SlackWebhookTool`** (`effgen/tools/builtin/slack_webhook.py`) — post messages to Slack via incoming webhook URL (no OAuth required). Config: `SLACK_WEBHOOK_URL`. URL is redacted in all logs. Operations: `post`. Added to `notify` preset.

  ```python
  from effgen.tools.builtin.slack_webhook import SlackWebhookTool
  result = SlackWebhookTool().execute({"operation": "post", "text": "Deploy complete!"})
  ```

- **`DiscordWebhookTool`** (`effgen/tools/builtin/discord_webhook.py`) — post messages to Discord via webhook URL. Config: `DISCORD_WEBHOOK_URL`. URL is redacted in all logs. Operations: `post`. Added to `notify` preset.

  ```python
  from effgen.tools.builtin.discord_webhook import DiscordWebhookTool
  result = DiscordWebhookTool().execute({"operation": "post", "content": "Deployment succeeded!"})
  ```

#### New Presets
- **`media` preset** — bundles `AudioTranscribeTool` and `ImageCaptionTool` for media-processing agents.
- **`notify` preset** — bundles `EmailSMTPTool`, `EmailIMAPTool`, `SlackWebhookTool`, and `DiscordWebhookTool` for notification/alert agents.

#### Documentation
- **`docs/tools/gallery.md`** — updated with all 14 new tools (OCR, AudioTranscribe, ImageInfo, ImageCaption, PDF, DOCX, Excel, Weather, Geocode, Maps, EmailSMTP, EmailIMAP, SlackWebhook, DiscordWebhook).
- **`docs/tools/ocr.md`** — OCRTool reference with per-OS Tesseract install instructions.
- **`docs/tools/audio_transcribe.md`** — AudioTranscribeTool reference with ffmpeg install notes.
- **`docs/tools/image.md`** — ImageInfoTool + ImageCaptionTool reference.
- **`docs/tools/documents.md`** — PDFTool + DOCXTool + ExcelTool reference.
- **`docs/tools/weather.md`** — WeatherTool reference.
- **`docs/tools/geocode.md`** — GeocodeTool reference.
- **`docs/tools/maps.md`** — MapsTool reference.
- **`docs/tools/email.md`** — EmailSMTPTool + EmailIMAPTool reference.
- **`docs/tools/webhooks.md`** — SlackWebhookTool + DiscordWebhookTool reference (with security note: webhook URLs are secrets).

### Changed
- **`general` preset** — now includes OCRTool, ImageInfoTool, PDFTool, DOCXTool, ExcelTool, WeatherTool, GeocodeTool, MapsTool, EmailSMTPTool, EmailIMAPTool, SlackWebhookTool, DiscordWebhookTool in addition to existing tools.
- **`research` preset** — now includes PDFTool, DOCXTool, ExcelTool for document parsing alongside existing academic/web tools.
- **`effgen/__init__.py`** — version bumped to `0.2.6`.

### New Errors
- **`OCRBackendUnavailable`** — raised when neither Tesseract nor OCR.space is available; includes per-OS install instructions.
- **`MissingSystemDependency`** — raised by audio/document tools when a required system binary (ffmpeg, tesseract) is absent.
- **`NoVisionProviderAvailable`** — raised by `ImageCaptionTool` when no vision-capable provider is configured.
- **`MissingCredentialsError`** — raised by email/webhook tools when required env vars are absent.
- **`CorruptDocumentError`** — raised by PDF/DOCX/Excel tools on unreadable files.

---

## [0.2.5] - 2026-05-18

### Highlights

**effGen v0.2.5** adds **13 new free/no-auth tools** spanning academic research, news & RSS, YouTube, social media, translation, language detection, and QR codes — bringing the total built-in tool count to **44+**. All tools are `BaseTool` subclasses with structured `{success, data, error}` output, integrated into the `research` and `general` presets, and covered by unit + integration tests.

### Added

#### Academic Research Tools
- **`PubMedTool`** (`effgen/tools/builtin/pubmed.py`) — search PubMed via NCBI E-utilities, fetch article metadata, retrieve abstracts. Operations: `search`, `fetch`, `abstract`. Built-in token-bucket rate limiter (3 req/s without key, 10/s with `NCBI_API_KEY`). Added to `research` preset.
- **`ArXivTool`** (`effgen/tools/builtin/arxiv.py`) — search arXiv Atom feed, fetch paper metadata by ID, download PDFs. Operations: `search`, `fetch`, `download_pdf`. Added to `research` preset.
- **`SemanticScholarTool`** (`effgen/tools/builtin/semantic_scholar.py`) — search papers, fetch paper details, retrieve citations and references via Semantic Scholar Graph API. Operations: `search`, `paper`, `citations`, `references`. Built-in backoff (100 req/5 min unauth). Added to `research` preset.

#### News & RSS Tools
- **`RSSFeedTool`** (`effgen/tools/builtin/rss.py`) — fetch, browse, and full-text search any RSS/Atom feed by URL. Operations: `fetch`, `latest`, `search_in_feed`. Handles malformed feeds gracefully. Added to `research` and `general` presets.
- **`NewsTool`** (`effgen/tools/builtin/news.py`) — aggregate top headlines and search news across curated reputable RSS sources (Reuters, BBC, HN, NPR, etc.); optional `NEWS_API_KEY` for NewsAPI.org. Operations: `top_headlines`, `search`. Added to `research` and `general` presets.

#### YouTube Tools
- **`YouTubeTranscriptTool`** (`effgen/tools/builtin/youtube_transcript.py`) — fetch YouTube captions/transcripts without a Google API key via `youtube-transcript-api`. Operations: `get_transcript`, `list_available_languages`, `translated`. Handles watch?v=, youtu.be/, and shorts/ URL formats. Added to `research` preset.
- **`YouTubeMetadataTool`** (`effgen/tools/builtin/youtube_metadata.py`) — fetch video/channel metadata using yt-dlp in metadata-only mode. Operations: `metadata`, `channel`. No auth required for public content. Added to `research` preset.

#### Social Media Tools
- **`RedditTool`** (`effgen/tools/builtin/reddit.py`) — access Reddit top/hot posts, user submissions, and thread comments via public JSON endpoints (no OAuth for reads). Operations: `subreddit_top`, `subreddit_hot`, `user_submissions`, `thread_comments`. Sets `effGen/<version>` User-Agent; exponential backoff on 429. Added to `research` and `general` presets.
- **`HackerNewsTool`** (`effgen/tools/builtin/hackernews.py`) — fetch top/new stories, story details, and user profiles from HN Firebase API. Operations: `top_stories`, `new_stories`, `story`, `user`. No auth required. Added to `research` and `general` presets.

#### Translation & Language Detection Tools
- **`TranslateTool`** (`effgen/tools/builtin/translate.py`) — translate text between languages with LibreTranslate as primary backend (configurable via `LIBRE_TRANSLATE_URL`) and `argostranslate` as an offline fallback. Operations: `translate`, `available_pairs`. Language pack cache at `~/.effgen/argos/`. Added to `general` preset.
- **`LanguageDetectTool`** (`effgen/tools/builtin/language_detect.py`) — detect language of text or a batch of texts, fully offline via `langdetect` (55+ languages). Operations: `detect`, `detect_batch`. Added to `general` preset.

#### QR Code Tools
- **`QRGenerateTool`** (`effgen/tools/builtin/qr_generate.py`) — generate QR codes locally from any text or URL; returns base64 PNG or file path. Operations: `generate`. Supports `data_url_return=True` for inline embedding. No network required. Added to `general` preset.
- **`QRReadTool`** (`effgen/tools/builtin/qr_read.py`) — decode QR codes and barcodes from image files or base64 PNG using `pyzbar` + Pillow, with OpenCV QR fallback when `libzbar` is unavailable. Operations: `read`. Fully local. Added to `general` preset.

#### Documentation
- **`docs/tools/gallery.md`** — tool gallery with one-line description and quickstart snippet for every built-in tool (all 44+).
- **`docs/tools/index.md`** — updated with all 13 new tools.
- Per-tool docs: `pubmed.md`, `arxiv.md`, `semantic_scholar.md`, `rss.md`, `news.md`, `youtube.md`, `reddit.md`, `hackernews.md`, `translate.md`, `language_detect.md`, `qr.md`.

### Changed
- **Preset registry** — `research` preset now includes PubMed, ArXiv, SemanticScholar, RSS, News, YouTubeTranscript, YouTubeMetadata, Reddit, HackerNews tools. `general` preset now includes RSS, News, Reddit, HackerNews, Translate, LanguageDetect, QRGenerate, QRRead tools.
- **`effgen/__init__.py`** — version bumped to `0.2.5`.

---

## [0.2.4] - 2026-05-14

### Highlights

**effGen v0.2.4** introduces a production-ready **ModelRouter** with three composable routing policies (FirstAvailable, CostBased, LatencyBased), transparent provider failover with retry logic, persisted cross-process rate-limit coordination via SQLite, and a persistent cost tracker with a `effgen cost` CLI dashboard.

### Added

#### ModelRouter + Routing Policies
- **`PolicyBasedRouter`** (`effgen/models/router.py`) — composable policy engine; `route(context)` returns an explainable `RouterDecision` recording which providers were eliminated and why; `route_and_execute(context, fn)` wraps any callable with transparent failover across `failover_hops` (default 3).
- **`RoutingPolicy` ABC** — base class for all policies; implement `select(candidates, context) → RouterDecision`.
- **`RoutingContext`** — carries `prompt_tokens_estimate`, `user_budget_usd`, `latency_budget_ms`, `required_capabilities`.
- **`RouterDecision`** — records `chosen`, `eliminated` (with per-provider reasons), `policy_name`, and `score`. Every routing decision is fully explainable.
- **`RouterEvent`** — emitted on failover; subscribers register via `PolicyBasedRouter.subscribe(callback)`.
- **`FirstAvailablePolicy`** (`effgen/models/routing/first_available.py`) — returns the first provider with a valid API key that meets the required capabilities.
- **`CostBasedPolicy`** (`effgen/models/routing/cost.py`) — estimates cost per call from pricing registry; ranks cheapest-first; free-tier providers rank ahead of equally-priced paid providers; raises `NoCandidateWithinBudgetError` when no candidate fits `user_budget_usd`.
- **`LatencyBasedPolicy`** (`effgen/models/routing/latency.py`) — picks the fastest provider by observed p50 latency; eliminates candidates exceeding `latency_budget_ms`; warm-up probe seeds empty-history tiebreaks.
- **`RetryPolicy`** (`effgen/models/routing/retry.py`) — configurable `max_retries`, exponential backoff with jitter; retries `RateLimitExceeded`, `ProviderTransientError`, `ModelTimeoutError`; does **not** retry `ModelAuthError`, `ModelRefusalError`, `InvalidRequestError`.

#### Capability Model
- **`Capability` enum** (`effgen/models/capabilities.py`) — `{chat, tools, streaming, vision, grounding, thinking, json_schema}`; all 9 adapters register their capability sets in `ProviderRegistry`.
- **`ProviderRegistry.register(..., capabilities=..., pricing=...)`** — extended with capability and pricing fields; all 9 adapters updated with current published pricing (2026-05-14).

#### Latency Tracker
- **`LatencyTracker`** (`effgen/models/latency_tracker.py`) — rolling window (last 50 calls) per `(provider, model)`; records total latency and time-to-first-token (TTFT); `p50(provider, model)` returns `float | None`; all 9 adapters instrumented.

#### Cross-Process Rate-Limit Coordination (SQLite)
- **`SQLiteRateLimitStore`** (`effgen/models/_rate_limit_store.py`) — WAL-mode SQLite at `~/.effgen/rate_limits.sqlite`; `BEGIN IMMEDIATE` row-locking prevents double-spend across processes; schema: `rate_events(provider, model, kind, timestamp, tokens)`.
- **`RateLimitCoordinator`** gains `storage=` parameter (default in-memory for back-compat); pass `SQLiteRateLimitStore` for cross-process coordination.
- Background housekeeping removes events older than 24 h × 1.1.

#### Cost Tracker Persistence + Dashboard
- **`SQLiteCostStore`** (`effgen/models/_cost_store.py`) — WAL-mode SQLite at `~/.effgen/costs.sqlite`; schema: `cost_events(provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp)`.
- **`CostTracker`** gains `storage=` parameter (default in-memory for back-compat); every `record()` writes a row; 80% daily budget → warning; 100% → `BudgetExceededError`.
- **`effgen cost` CLI** — `today`, `week`, `by-provider`, `set-budget`, `clear-budget` subcommands; rich table output.
- **`effgen config set budget.daily <USD>`** — configure daily spend cap.

#### New Errors
- `AllCandidatesExhaustedError` — raised when failover exhausts all providers.
- `BudgetExceededError` — raised when cumulative spend exceeds the configured cap; `RetryPolicy` treats this as retriable (failover to free tier).
- `ProviderTransientError` — base class for 5xx / transient provider failures.
- `InvalidRequestError` — bad request (4xx); not retried.

### Changed
- **`RateLimitCoordinator`** — gains `storage: RateLimitStore` parameter; default behavior (in-memory) unchanged.
- **`CostTracker`** — gains `storage: CostStore` parameter; default behavior (in-memory) unchanged.
- **`ProviderRegistry.register()`** — extended with `capabilities=set[Capability]` and `pricing=dict` optional fields; existing callers unaffected.
- **All 9 adapters** — instrumented with `LatencyTracker.record()` on every `generate()` call; TTFT recorded in streaming adapters on first yielded chunk.
- **Top-level `effgen` namespace** — new exports: `PolicyBasedRouter`, `RoutingPolicy`, `RoutingContext`, `RouterDecision`, `RouterEvent`, `ProviderModelPair`, `FirstAvailablePolicy`, `CostBasedPolicy`, `LatencyBasedPolicy`, `RetryPolicy`, `LatencyTracker`, `CostTracker`, `SQLiteCostStore`, `AllCandidatesExhaustedError`, `BudgetExceededError`, `ProviderTransientError`, `InvalidRequestError`.

### Fixed
- **Stability sweep** — all pre-existing ruff / mypy warnings resolved at the v0.2.3 baseline.
- **Back-compat guarantee** — `load_model(...)`, `Agent(config)`, direct adapter paths all unaffected by the router layer.

---

## [0.2.3] - 2026-05-04

### Highlights

**effGen v0.2.3** expands the provider ecosystem from 4 to **9 inference backends** — adding Groq, Together AI, Fireworks, Replicate, and HuggingFace Inference — each with full streaming, native tool-calling, cost tracking, and rate-limit coordination. A unified `ProviderRegistry` consolidates all adapters for first-class introspection, and a backend parity matrix proves cross-provider correctness on a canonical agentic task.

### Added

#### New Backends (5 providers)
- **`GroqAdapter`** — 16 chat models (llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b, gemma2-9b-it, qwen3-32b, and the full free-tier roster); native tool-calling; streaming with timestamp-verified chunk delivery; RPM/RPD/TPM/TPD rate-limit windows; free-tier `CostTracker` ($0). `pip install effgen[groq]`.
- **`TogetherAdapter`** — 163-model catalog (149 chat + 13 language + 1 embedding) with live `refresh_models()` drift detection; native tools; streaming; per-model pricing. `pip install effgen[together]`.
- **`FireworksAdapter`** — 80 chat models (54 tool-capable); live catalog via `refresh_models()`; OpenAI-compatible interface; streaming; per-model pricing. `pip install effgen[fireworks]`.
- **`ReplicateAdapter`** — 38 models (25 tool-capable, 34 streaming); async run-then-poll with exponential backoff; SSE streaming; configurable prediction timeout (default 300 s); `ModelTimeoutError` with prediction cancellation; `compute_seconds` in metadata. `pip install effgen[replicate]`.
- **`HFInferenceAdapter`** — 124-model dynamic registry from HuggingFace Router (refresh + drift detection + `~/.effgen/cache` hot-reload); `chat_completion` + `text_generation` + streaming + native tools; custom Inference Endpoint URL support; `ModelUnavailableError` with `suggest_alternatives()`; `ModelNotFoundError`. `pip install effgen[hf]`.

#### Unified ProviderRegistry + Auth
- **`ProviderRegistry`** (`effgen/models/registry.py`) — singleton with `register`, `list_providers`, `list_models`, `lookup`; handles duplicate model IDs across providers; `AmbiguousModelError` on bare ambiguous IDs.
- **Adapter self-registration** — all 9 adapters register on import; idempotent.
- **`check_keys()`** (`effgen/models/auth.py`) — `provider → {available, env_key, env_keys_checked}` map.
- **`effgen doctor`** CLI command — prints provider auth table; `--json` and `--provider` filter flags; loads `.env` from `~/.effgen/.env` + project root.

#### Backend Parity Matrix
- **`tests/integration/parity/canonical_task.py`** — shared Calculator + ReAct task: "What is (17 × 23) + sqrt(144)?" Expected answer: 403.
- **`tests/integration/parity/test_backend_parity.py`** — parametrized across (provider, model) pairs; per-parametrization skip on missing key.
- **Parity reports** — `outputs/7-parity-matrix.md` (7/8 providers correct; Anthropic=no key; Replicate=billing), `outputs/7-stream-parity.md` (7/7 providers streaming), `outputs/7-error-parity.md` (9/9 providers raise `ModelAuthError`).
- **`docs/providers/parity.md`** — full provider capability table.

### Changed
- **`load_model`** now uses `ProviderRegistry` for `provider:model_id` prefix parsing; existing per-provider branches unchanged (no behavior change for callers).
- **Provider table in README expanded** to 9 providers with `effgen[groq]` / `[together]` / `[fireworks]` / `[replicate]` / `[hf]` install instructions.

### Fixed
- **`cli.py`** — `BatchConfig` variable reference from the stability sweep.
- **`aggregation.py`** — `sources` variable shadowing.
- **`CostTracker._rate`** — Fireworks pricing path added.
- **`GroqAdapter.supports_tool_calling()`** — was missing; added.
- **`ModelAuthError`** — unified across all 9 adapters for consistent error surface.
- **`AmbiguousModelError`** raised by `ModelLoader` on bare IDs shared across multiple providers (previously fell through to HF download path).

---

## [0.2.2] - 2026-04-28

### Highlights

**effGen v0.2.2** expands the **Gemini** adapter with the latest model families (3.x, 2.5, 2.0, Gemma 3/4), `thinking_budget`, Google Search grounding, the Files API, and three Gemini-native tools (`GoogleSearchTool`, `GeminiUrlContextTool`, `GeminiCodeExecutionTool`). It also modernizes the **Anthropic** adapter — Claude 4.7 / 4.x registry, extended thinking, prompt caching via `cache_control`, streaming polish, and experimental native tools — implemented and unit-tested; Anthropic live tests are skipped (no key in dev env).

### Added

#### Gemini — New Model Families
- **Expanded model registry** (`effgen/models/gemini_models.py`) — Gemini 3.1-flash-lite, 3.0-pro, 2.5-flash, 2.5-pro, 2.0-flash, Gemma 3 and Gemma 4 families; `available_models()`, `free_tier_models()`, `recommended_models()`, `model_info()` helpers; context/output/feature flags per model
- **Migrated SDK** from `google.generativeai` (legacy) to `google.genai` (`google-genai>=1.0.0`)

#### Gemini — Thinking
- **`GenerationConfig.thinking_budget: int | None`** — pass tokens for Gemini's internal reasoning; wired through `ThinkingConfig` in the adapter
- **`GenerationConfig.include_thoughts: bool`** (default `False`) — surface thinking trace in `ModelResponse.metadata["thinking"]`

#### Gemini — Grounding
- **`GenerationConfig.grounding: bool`** (default `False`) — injects Google Search tool when model supports it; grounding attributions surfaced in `ModelResponse.metadata["grounding_chunks"]`

#### Gemini — File Upload
- **`effgen.models.gemini_files.upload_file(path) → FileRef`** — wraps `genai` Files API; 2 GiB pre-upload guard; accepts `FileRef` objects in `generate(prompt, files=[...])`

#### Gemini — Native Tools
- **`GoogleSearchTool`** — activates Gemini's built-in search; `ToolIncompatibleError` at Agent init with non-Gemini model
- **`GeminiUrlContextTool`** — server-side URL content fetching
- **`GeminiCodeExecutionTool`** — server-side Python execution; output surfaced in `generated_text`
- All three in `effgen/tools/builtin/gemini_native.py`; re-exported from `effgen`
- Parallel function calls handled: adapter encodes all parallel `functionCall` parts; `metadata["tool_calls"]` is a list

#### Anthropic — Claude 4.x Registry
- **`effgen/models/anthropic_models.py`** — claude-opus-4-7 (1M ctx / 128K out), claude-sonnet-4-6 (1M ctx / 64K out), claude-haiku-4-5, legacy 4.x and 3.x lineup; `supports_thinking`, `supports_native_tools`, `supports_prompt_caching`, pricing fields

#### Anthropic — Extended Thinking
- **`GenerationConfig.thinking: dict | None`** — accepts `{"type": "enabled", "budget_tokens": N}`; temperature is forced to 1.0 when thinking is active
- Thinking trace surfaced in `ModelResponse.metadata["thinking"]`
- **`redacted_thinking` multi-turn preservation** — `raw_content_blocks` in metadata; `build_assistant_message()` helper re-inserts redacted blocks on next turn

#### Anthropic — Prompt Caching
- **`effgen/models/anthropic_cache.py`** — `mark_cached(block, ttl="5m"|"1h")`, `apply_cache_to_system()`, `apply_cache_to_tools()`, `validate_breakpoint_count()` (max 4; raises `ValueError` on 5th)
- **`AgentConfig.cache_system_prompt: bool = True`** — auto-inserts `cache_control` on the system prompt's final block
- **`AgentConfig.cache_tools: bool = True`** — auto-inserts `cache_control` on the last tool spec
- **Usage surfacing** — `ModelResponse.usage.cached_input_tokens` + `ModelResponse.usage.cache_creation_tokens`

#### Anthropic — Streaming + Native Tools
- **`generate_stream_full()`** — returns typed `StreamChunk` objects; thinking, tool-use, redacted-thinking, and text deltas all handled; parallel `tool_use` blocks accumulated per-index
- **Experimental native tools** — `AnthropicBashTool`, `AnthropicTextEditorTool`, `AnthropicComputerTool` in `effgen/tools/builtin/anthropic_native.py`; `IS_ANTHROPIC_NATIVE` sentinel; `ToolIncompatibleError` at Agent init with non-Anthropic model; marked `experimental=True`

### Changed
- **`GenerationConfig`** gains `thinking_budget`, `include_thoughts`, `grounding`, `thinking` (all `None`/`False` by default — fully back-compat)
- **`AgentConfig`** gains `cache_system_prompt: bool = True`, `cache_tools: bool = True` (additive, safe defaults)
- **Gemini adapter** migrated to `google-genai` SDK; existing `GeminiAdapter` public API unchanged

### Fixed
- **`cli.py`** — `ToolMetadata.input_schema` missing field from the stability sweep
- **`aggregation.py`** — `sources` variable shadowing / redefinition from the stability sweep
- **Gemini mixed native + function-calling** — `tool_config.include_server_side_tool_invocations=True` set when mixing built-in and user-defined tools (Gemini API requirement)
- **Anthropic `top_k`** — removed unsupported parameter that caused 400 errors
- **Gemini model aliases** — short aliases (e.g. `gemini-3.1-flash-lite`) now resolve to canonical registry IDs (e.g. `gemini-3.1-flash-lite-preview`) at `GeminiAdapter` init, so API calls succeed when callers pass the friendly short form

---

## [0.2.1] - 2026-04-25

### Highlights

**effGen v0.2.1** adds the **Cerebras** inference backend (4 free-tier models with streaming, native tool-calling, and cost tracking) and modernizes the **OpenAI** adapter (gpt-5/gpt-5.4-nano + o-series reasoning models, `reasoning_effort`, prompt caching surfacing, structured outputs v2, and OpenAI native tools — web_search, code_interpreter, file_search).

### Added

#### Cerebras Backend (new provider)
- **`CerebrasAdapter`** — full async adapter on top of `cerebras-cloud-sdk>=1.0`; `load`/`generate`/`generate_stream`/`generate_with_tools`/`unload`; OpenAI-compatible message format
- **All 4 free-tier models** in `effgen.models.cerebras_models` — `gpt-oss-120b`, `llama3.1-8b`, `qwen-3-235b-a22b-instruct-2507`, `zai-glm-4.7`; `available_models()`, `free_tier_models()`, `model_info()` helpers
- **`RateLimitCoordinator`** (`effgen/models/_rate_limit.py`) — sliding-window per-(provider,model) RPM/RPH/RPD + TPM/TPH/TPD throttling; `asyncio.Lock`-guarded; raises `RateLimitExceeded` on daily-budget exhaustion; wired into both sync and async Cerebras paths
- **Streaming** — `generate_stream()` yields token deltas via SDK `stream=True`; preserves usage on the terminal chunk
- **Native function-calling** — `generate_with_tools()` integrates with the Agent loop in `hybrid` mode; `supports_native_tools` flag per model
- **`CostTracker`** (`effgen/models/_cost.py`) — thread-safe singleton with per-provider rate table (Cerebras free-tier $0); `record/total_cost/summary/reset`
- **Loader integration** — `load_model(..., provider="cerebras")` and `effgen.CerebrasAdapter` re-exported
- **Docs** — `docs/models/cerebras.md` (all 4 models, rate-limit table, streaming + tool examples)

#### OpenAI Modernization
- **Expanded model registry** (`effgen/models/openai_models.py`) — gpt-5, gpt-5.4-nano, gpt-4.1, gpt-4o family, o1/o1-mini/o3/o3-mini/o4-mini reasoning models; pricing + `supports_reasoning`/`supports_native_tools`/`supports_prompt_caching` flags
- **`reasoning_effort`** + **`max_reasoning_tokens`** on `GenerationConfig` — `Literal["minimal","low","medium","high"]`; routed only to reasoning models; chat models silently drop with debug log; unknown values raise `ValueError`
- **`_pick_default_max_output()`** — family-aware default output budget (reasoning models default to 100k)
- **Prompt caching surfacing** — `AgentConfig.stable_system_prompt` (default True); `cached_input_tokens` exposed via `ModelResponse.usage` and metadata
- **Structured outputs v2** — `OpenAIAdapter.generate_structured()`; `to_openai_schema()` helper inlines `$ref`s and forces `additionalProperties: false`; `ModelRefusalError` raised on refusal
- **OpenAI native tools** — `OpenAIWebSearchTool`, `OpenAICodeInterpreterTool`, `OpenAIFileSearchTool` in `effgen.tools.builtin.openai_native`; routed through Responses API; `ToolIncompatibleError` at Agent init when paired with non-OpenAI models
- **Docs** — `docs/models/openai.md`, `docs/models/openai_advanced.md`, `docs/tools/openai_native.md`

### Changed
- **`GenerationConfig`** gains `reasoning_effort` and `max_reasoning_tokens` (both `None` by default — fully back-compat)
- **`AgentConfig`** gains `stable_system_prompt: bool = True`
- **OpenAI adapter** uses `max_completion_tokens` (replaces deprecated `max_tokens`); drops unsupported `stop`, `temperature`, `top_p` params for reasoning/gpt-5 models
- **`load_model(..., provider=...)`** routes correctly to OpenAI/Anthropic/Gemini/Cerebras (previously HF-only); HF-specific kwargs are stripped before reaching API adapters; OpenAI auto-detection prefix list extended to gpt-5/o-series

### Fixed
- **Stability sweep** — ruff cleanup (F821, B023, F841, C401, C408, I001, F401, F541); 2 real bug fixes (missing `Any` import; loop-closure variable capture)
- **`load_model(..., provider="openai"/"anthropic"/"gemini")`** — was silently treated as HF-only
- **OpenAI** — warns rather than errors on unknown model ids
- **Transformers engine** — `unload()` now removes `accelerate` hooks and syncs CUDA, eliminating cross-test CUDA-state bleed
- **GPU e2e/integration test fixtures** — disabled bitsandbytes 4-bit (CUDA state leak) and narrowed scope to class-level, fixing intermittent Qwen2 RMSNorm aborts
- **Cerebras streaming test** — retries on Cerebras `429`/`queue_exceeded`

---

## [0.2.0] - 2026-04-09

### Highlights

**effGen v0.2.0** is a major release that transforms the framework from a capable agent toolkit into a **production-grade agentic AI platform** — with native tool calling, guardrails, multi-agent orchestration, RAG pipelines, evaluation, and a production API server — all optimized for Small Language Models.

### Added

#### Critical Bug Fixes & Foundation Repairs
- **ReAct parser hardening** — improved `Final Answer:` extraction with `Observation:`/`Human:` boundary splitting; `_clean_json_input()` handles trailing commas, markdown fences, unquoted keys; 28-case parser test suite
- **Async/sync race condition fix** — replaced direct `asyncio.run()` with `_run_coroutine_sync()` for sub-agent parallel execution; works inside Jupyter/FastAPI/async contexts; configurable timeout (120s default)
- **Memory performance fix** — `get_token_count()` uses cached `_current_token_count` instead of O(n) recalculation; structured summary format preserving facts/decisions/pending items
- **Agent resource cleanup** — `Agent.close()` + sync context manager (`with Agent(config) as agent:`)
- **MCP transport fix** — correlation-ID-based pending request tracking; SSE exponential backoff reconnection (max 5 retries)
- **Tool security hardening** — BashTool blocks `${VAR:-$(cmd)}`, heredoc injection, process substitution; PythonREPL blocks `__import__`, `importlib`, `__builtins__`, `__subclasses__`; standardized 30s timeout / 100KB output limits
- **Sub-agent depth tracking** — try/finally cleanup; reset on run() start
- **Vision pass-through** — OpenAI, Anthropic adapters now support image_url/image blocks
- **New examples** — `async_concurrent_agent.py`, Docker Compose deployment, `agent_communication.py`

#### Native Tool Calling & Structured Output
- **`ToolCallingStrategy`** — abstract strategy with `ReActStrategy`, `NativeFunctionCallingStrategy`, `HybridStrategy` implementations
- **Native function calling** — `supports_tool_calling()` on all model backends; Qwen/Llama/Mistral/generic format parsers; tool JSON Schema definitions passed via chat template `tools` parameter
- **`tool_calling_mode`** in `AgentConfig` — `"auto"`, `"native"`, `"react"`, `"hybrid"` modes
- **Structured output** — `StructuredOutputConfig`, `constrain_output()`, `validate_json_schema()`; `output_schema` and `output_model` (Pydantic) parameters on `Agent.run()`; `output_format` and `output_schema` on `AgentConfig`
- **`ToolDefinition`** — with OpenAI/Anthropic format converters and `tools_to_definitions()` utility

#### Guardrails, Safety & Input/Output Validation
- **`effgen.guardrails`** module — `Guardrail` ABC, `GuardrailChain`, `GuardrailPosition` enum
- **Content guardrails** — `ToxicityGuardrail`, `PIIGuardrail` (SSN/email/phone/CC with Luhn/IP), `LengthGuardrail`, `TopicGuardrail`
- **`PromptInjectionGuardrail`** — low/medium/high sensitivity with zero false positives on normal queries
- **Tool safety** — `ToolInputGuardrail`, `ToolOutputGuardrail` (PII stripping, size limit), `ToolPermissionGuardrail` (allow/deny/require_approval)
- **Agent integration** — `AgentConfig.guardrails` param; pre-run input check, pre/post-tool checks, pre-return output check
- **Presets** — `get_guardrail_preset("strict"|"standard"|"minimal"|"none")`

#### Advanced Multi-Agent Orchestration
- **`MessageBus`** — pub/sub, mailbox, broadcast inter-agent communication with topic-based wildcard subscriptions and optional persistence
- **`WorkflowDAG`** — DAG-based workflow engine with cycle detection (Kahn's topological sort), conditional branching, auto-parallelization via `asyncio.gather`; YAML workflow definitions; `effgen workflow run/validate` CLI
- **`SharedState`** — thread-safe namespaced key-value store with per-namespace RLock, snapshots for rollback, event-sourced mutation log
- **Agent lifecycle management** — `AgentLifecycleState` (8 states), `AgentEntry` state machine, `AgentPool` (pre-warmed), `AgentRegistry` (thread-safe); per-agent timeout and cancellation

#### Batch Execution & Domain Scaling
- **`BatchRunner`** — asyncio-based concurrent batch execution with semaphore, retry, timeout; JSONL/CSV/JSON/text I/O; `Agent.run_batch()` convenience; `effgen batch` CLI
- **`ResultAggregator`** — exact hash + fuzzy Jaccard deduplication, ranking (confidence/relevance/speed/custom), merge strategies (first/best/consensus/union)
- **`ToolResultCache`** — thread-safe LRU + TTL for cross-query tool result sharing
- **`effgen.domains`** module — `Domain` base class, `KeywordExpander` (WordNet/template/LLM expansion); 5 built-in domains: `TechDomain`, `ScienceDomain`, `FinanceDomain`, `HealthDomain`, `LegalDomain`

#### Observability, Tracing & Debugging
- **OpenTelemetry upgrade** — full OTel SDK with Resource, BatchSpanProcessor, configurable exporters (OTLP/Jaeger/Zipkin/console); cross-agent trace propagation; no-op fallback
- **Structured logging** — `EffGenJSONFormatter`, `StructuredLogger` with agent/tool/model/iteration events; `LogRunContext` with run_id/workflow_id/agent_name/session_id correlation
- **Prometheus metrics upgrade** — response_latency/token_usage/tool_execution_time histograms with percentiles; GPU memory gauge; labels support
- **Grafana dashboard** — 12 panels: latency p50/p95/p99, throughput, error rate, tool breakdown
- **`effgen.debug`** module — `DebugAgent` wrapper with rich TUI step-through; `Agent.run(debug=True)` captures `DebugTrace` with per-iteration raw_prompt, raw_response, thought, action, observation, tokens, latency; `effgen debug` CLI

#### Model Router & Auto-Selection
- **`ModelRouter`** — routing by complexity, capabilities, loaded state, model size; `RoutingConfig`, `RoutingDecision`
- **`estimate_complexity()`** — heuristic keyword analysis (code/math/reasoning/multilingual), query length, structural patterns; < 1ms execution
- **`MODEL_CAPABILITIES`** — registry with pre-populated profiles for 12 models (Qwen 0.5B-7B, Llama 1B-3B, Phi-3/3.5/4, Mistral 7B, Gemma 2B/9B)
- **Multi-model agent** — `models` list and `speculative_execution` in AgentConfig; ModelRouter auto-created; `_generate_speculative()` runs on 2 models via `asyncio.wait(FIRST_COMPLETED)`
- **`ModelPool`** — LRU eviction, GPU memory-based eviction, hot-swap; `effgen models load|unload|status` CLI

#### Community Contribution: MLX & MLX-VLM Backends (PR #4, commit e5b54f5)
- **`MLXEngine`** — MLX (mlx-lm) text generation engine with streaming/batch support for Apple Silicon
- **`MLXVLMEngine`** — MLX-VLM vision-language engine with image support (30+ architectures)
- **`effgen.hardware`** module — `platform.py` with Apple Silicon/CUDA/MLX detection helpers and backend recommendation
- **Model loader integration** — MLX/MLX-VLM auto-selection on Apple Silicon; `ModelType.MLX` and `ModelType.MLX_VLM`
- **Optional deps** — `pip install effgen[mlx]` and `pip install effgen[mlx-vlm]` (darwin/arm64 only)
- **5 new GUI examples** — `chat_gui_mlx.py`, `agent_viz_mlx.py`, `tool_builder_gui.py`, `tool_tester_gui.py`, `basic_agent_mlx.py` (Gradio-based)
- **Unit tests** — `test_hardware_platform.py`, `test_mlx_engine.py`, `test_mlx_vlm_engine.py`

#### Persistent Agent State & Checkpointing
- **`CheckpointManager`** — save/restore full agent state (scratchpad, memory, tool states, iteration count, partial results); filesystem + SQLite backends
- **Agent checkpoint/resume** — `agent.run("...", checkpoint_interval=3)` for periodic checkpointing; `agent.resume(checkpoint_id="...")` to resume; CLI: `effgen run --checkpoint-dir --checkpoint-interval`, `effgen resume --checkpoint`
- **`Session`** / `SessionManager` — persistent conversation sessions with UUID management, expiry, cleanup; `Agent(config, session_id="user-123")` auto-loads/persists per turn; CLI: `effgen sessions list|delete|export|cleanup`
- **`BackgroundTaskRunner`** — priority queue, pause/resume/cancel, threading workers; `Agent.run_background()` / `get_task_status()` / `get_task_result()` / `cancel_task()`

#### Advanced RAG Pipeline
- **`effgen.rag`** module — complete RAG pipeline
- **`DocumentIngester`** — txt/md/json/jsonl/csv/html built-in loaders; pdf/docx/epub optional; SHA-256 deduplication; progress tracking
- **Advanced chunking** — `SemanticChunker`, `CodeChunker` (py/js/ts/go/rust/java), `TableChunker`, `HierarchicalChunker`
- **`HybridSearchEngine`** — dense + BM25 + keyword + metadata filter fused via Reciprocal Rank Fusion
- **Reranking** — `CrossEncoderReranker` (optional), `LLMReranker` (free default), `RuleBasedReranker` (recency/authority/keyword/title)
- **`ContextBuilder`** — token budget management, source deduplication, relevance/chronological ordering, inline `[N]` citations
- **Source attribution** — `Citation` dataclass, `CitationTracker` with verify/extract; `AgentResponse.citations` and `.sources` fields
- **RAG preset** — `create_agent("rag", model, knowledge_base="./docs/")`

#### Human-in-the-Loop & Approval Workflows
- **Human interaction points** — `HumanApproval`, `HumanInput`, `HumanChoice` (all with timeout via ThreadPoolExecutor)
- **Tool approval** — `requires_approval` on `ToolMetadata`; `approval_callback`, `approval_mode` (`always`/`first_time`/`never`/`dangerous_only`), `approval_timeout` in `AgentConfig`; `ApprovalManager` wired into tool execution path
- **Clarification** — `ClarificationRequest` (options + free-text), `ClarificationDetector` with heuristic ambiguity detection (short query, vague words, multiple-tool-match)
- **Feedback collection** — `FeedbackCollector` (thumbs/rate/comment), `FeedbackEntry`, export to JSONL

#### New Domain Tools (17 New Tools — 31 Total)
- **Finance** — `StockPriceTool` (yfinance + Yahoo Finance v8 fallback), `CurrencyConverterTool` (frankfurter.app/ECB), `CryptoTool` (CoinGecko); all include "not financial advice" disclaimer
- **Data Science** — `DataFrameTool` (pandas: load/head/describe/filter/aggregate), `PlotTool` (matplotlib: line/bar/scatter/hist → PNG), `StatsTool` (numpy: mean/median/std/correlation/regression)
- **DevOps** — `GitTool` (read-only: status/log/diff/branch/show), `DockerTool` (read-only: ps/images/logs), `SystemInfoTool` (psutil: cpu/memory/disk/network), `HTTPTool` (urllib GET/POST)
- **Knowledge** — `ArxivTool` (Atom feed), `StackOverflowTool` (SE API), `GitHubTool` (public search API), `WolframAlphaTool` (optional, requires API key)
- **Communication** — `EmailDraftTool` (draft only, does NOT send), `SlackDraftTool` (draft only), `NotificationTool` (plyer desktop notifications, optional)
- All external libraries handled as optional with clear install hints

#### Evaluation, Benchmarking & Regression Testing
- **`effgen.eval`** module — `AgentEvaluator`, `EvalResult`, `SuiteResults`, `TestCase`, `TestSuite`
- **Scoring modes** — `EXACT_MATCH`, `CONTAINS`, `REGEX`, `SEMANTIC_SIMILARITY` (sentence-transformers optional), `LLM_JUDGE`
- **5 built-in test suites** — `MathSuite` (77 cases), `ToolUseSuite` (93), `ReasoningSuite` (40), `SafetySuite` (40), `ConversationSuite` (20)
- **`RegressionTracker`** — save/load/compare baselines; severity levels (warning/high/critical); thresholds: >5% accuracy drop, >20% latency increase
- **`ModelComparison`** — multi-model matrix comparison with recommendations; markdown/JSON export
- **CLI** — `effgen eval --suite <name>` and `effgen compare --models "a,b,c" --suite <name>`
- **Nightly CI** — eval-regression job compares against stored baselines, opens GitHub issue on failure

#### API Server v2 — Production Gateway
- **OpenAI-compatible API** — `/v1/chat/completions` and `/v1/completions` with `tools` param and `stream: true` (SSE); model aliases (gpt-4 → Qwen2.5-7B, gpt-3.5-turbo → Qwen2.5-3B)
- **`RequestQueue`** — priority queue with fair scheduling, deadlines, backpressure (`QueueFullError`)
- **`AgentPool`** — min/max size, factory, idle TTL, health checking, acquire/release
- **Multi-tenancy** — `TenantManager` (rate limits, model restrictions, tool permissions); `APIKey` management with hashed storage and constant-time resolution
- **Production middleware** — CORS, request ID injection (X-Request-ID), GZip compression, graceful shutdown

#### SDK, Client Libraries & Embedding API
- **Python client SDK** — `EffGenClient` with sync + async via httpx; `chat()`, `embed()`, `health()`, `chat_stream_sync()`, `achat()`, `chat_stream()` (async iterator); retries with exponential backoff; 7 typed exception classes
- **TypeScript/JavaScript client** — `clients/typescript/` with fetch-based `EffGenClient`; chat/embed/health/streaming; works in Node 18+/Deno/Bun/browser
- **Local embedding API** — `/v1/embeddings` endpoint (OpenAI-compatible); `SentenceTransformerEmbedder` + `TFIDFEmbedder` fallback; model aliases; `LRUCache` + `SQLiteCache` for embedding caching

#### Performance Optimization & Caching
- **`effgen.cache`** module — `PromptCache` (LRU + TTL, sha256 fingerprint, thread-safe, hit/miss stats); `ResultCache` (LRU + per-tool TTL, optional semantic similarity via embed_fn + cosine)
- **`TokenBudget`** — smart context window allocation (system 20% / tools 30% / history 40% / response 10%); `smart_truncate()` preserves head+tail; `fit_to_budget()` per-section truncation
- **`LazyModel`** — defers `.load()` until first generate/count_tokens; idle_timeout-based eviction (default 600s)
- **GGUF support** — `GGUFEngine` via optional llama-cpp-python; auto-routed by model_loader for `.gguf` files
- **AWQ / GPTQ quantization** — `quantization="awq"` and `quantization="gptq"` in model_loader; optional deps with friendly install hints
- **Speculative decoding** — `GenerationConfig.draft_model` field for backends that support draft-model decoding
- **`ContinuousBatcher`** — coalesces concurrent submit() calls in background worker; max_batch_size / max_wait_ms flush; `BatchModel` fast path + sequential fallback

### Changed
- **7 inference backends** (was 5) — added MLX and MLX-VLM for Apple Silicon
- **31 built-in tools** (was 14) — added 17 domain tools (finance, data science, DevOps, knowledge, communication)
- Model backends now support `supports_tool_calling()` for native function calling
- `AgentConfig` extended with `tool_calling_mode`, `output_format`, `output_schema`, `guardrails`, `models`, `speculative_execution`, `approval_mode`, `approval_callback`, `session_id`, `checkpoint_interval`, `checkpoint_dir`
- `AgentResponse` extended with `citations` and `sources` fields
- `Agent.run()` accepts `output_schema`, `output_model`, `debug`, `checkpoint_interval` parameters
- Prometheus metrics now include histograms with percentiles, GPU memory gauge, and labels

### Fixed
- `asyncio.run()` crash when Agent used inside existing event loops (Jupyter, FastAPI)
- ShortTermMemory `get_token_count()` O(n) recalculation on every call (now O(1))
- MCP HTTP transport race condition with concurrent requests
- Sub-agent `_current_depth` not reset on completion/failure
- BashTool vulnerable to nested command substitution (`${VAR:-$(cmd)}`)
- PythonREPL sandbox escape via `__import__`, `importlib`, `__builtins__`

### Internal
- 487+ unit tests passing (up from 157 in v0.1.3)
- Real GPU integration tests (A40 GPUs)
- Fresh isolated environment validation for each feature set
- Nightly CI with eval regression detection and automated GitHub issue creation

---

## [0.1.3] - 2026-03-25

### Added
- **Sub-agent depth limiting** — `max_sub_agent_depth` config option (default 3) prevents unbounded sub-agent recursion (ISSUE-005)
- **"No tool needed" guidance** in ReAct prompt — explicit instruction and example for direct answers, reducing unnecessary tool calls by SLMs (ISSUE-016)
- **Model-aware token counting** — `ShortTermMemory` now accepts an optional `model` parameter for accurate tokenization instead of the `len(text)//4` heuristic (ISSUE-009)
- **Circuit breaker persistence** — optional JSON file persistence for circuit breaker state via `persist_path` parameter (ISSUE-012)
- **Streaming timeout safety** — all streaming examples now use `signal.SIGALRM` timeouts to prevent indefinite hangs (ISSUE-013)
- **`pytest-timeout`** added to dev dependencies with 120s default timeout (ISSUE-001)
- **`bitsandbytes`** added to dev dependencies for 4-bit quantization testing (ISSUE-002)

### Improved
- **Loop detection** — exact loop now allows 1 retry before triggering (was zero-tolerance); fuzzy loop threshold raised to 7 for `DATA_PROCESSING` category tools; action inputs normalized (JSON key sorting, whitespace stripping) before comparison (ISSUE-004, ISSUE-019)
- **Partial answer extraction** — observations now scanned for day names and numeric results; multiple valid observations combined for multi-tool tasks (ISSUE-017)
- **"Answer now" nudge** — when iterations are running low and a tool returned successfully, the scratchpad hints the model to emit `Final Answer:` (ISSUE-017)
- **Model-family prompt formatters** — Qwen format uses `<|tools|>` section markers; Llama format uses `<|begin_of_text|>` header/EOT tags (ISSUE-010)
- **Stop sequences** — removed overly aggressive `\n\n\n` stop sequence that could truncate legitimate multi-paragraph output (ISSUE-015)
- **System prompt** — added "Do NOT use tools for greetings, jokes, opinions, or recalling information" to mistakes section (ISSUE-016)
- **Model loading warning** — logs a clear warning when `require_model=False` and loading fails, instead of silently setting `self.model = None` (ISSUE-011)
- **Integration test robustness** — `real_model` fixture falls back to fp16 if bitsandbytes is not installed (ISSUE-002)

### Fixed
- **NotImplementedError messages** — MCP transport stubs and Retrieval tool stubs now include descriptive messages instead of bare `raise NotImplementedError` (ISSUE-006, ISSUE-008)

### Internal
- 19 issues from v0.1.2 verification addressed across 12 files
- Streaming examples hardened with timeout handling
- Conversational agent example tuned for memory summarization

## [0.1.2] - 2026-03-12

### Added
- **10 comprehensive example agents** covering Q&A, calculator, multi-tool, file operations, code execution, conversational memory, error recovery, data processing, streaming, and multi-agent pipeline orchestration
- **Cross-model compatibility matrix** — 11 models tested across all 10 agents (110 combinations), 73% pass rate ([compatibility_matrix.md](examples/compatibility_matrix.md))
- **User-explicit sub-agent trigger detection** in `SubAgentRouter` — regex-based fuzzy matching for phrases like "use sub-agents", "launch 3 agents", "spawn agents" (router.py)
- **Compatibility sweep runner** (`examples/sweep_model.py`) for automated cross-model testing

### Improved
- **ReAct loop robustness** — loop detection breaks repeated identical actions (BUG-003), fuzzy loop detection for 5+ calls with different inputs (BUG-017)
- **Tool input parsing** — single-quoted JSON via `ast.literal_eval` fallback (BUG-016), non-JSON input mapping, markdown fence stripping for code params
- **Conversation history** — `max_turns` increased from 5 to 25, summary inclusion, assistant response truncation (300 chars), configurable `keep_recent_messages` (BUG-014, BUG-015)
- **Answer extraction** — line-start anchor for "Answer:" regex with `re.MULTILINE` (BUG-004), trailing text trimming (BUG-005), newline boundary fix for `Action: Final Answer` (BUG-008)
- **Tool result formatting** — proper extraction of `data`/`message` keys from FileOperations dict results (BUG-010), stderr extraction for CodeExecutor errors (BUG-013), stdout preference over None for PythonREPL (BUG-012)
- **Default max_tokens** increased from 512 to 1024 for long tool data (BUG-018)

### Fixed
- **BUG-001:** `quantization="4bit"` silently ignored by TransformersEngine — now properly passed through (model_loader.py)
- **BUG-002:** gemma-3 context length detection fails when config uses nested `text_config` (transformers_engine.py)
- **BUG-006:** DateTimeTool `now` operation ignores `date` parameter (datetime_tool.py)
- **BUG-007:** `validate_parameters` rejects unknown parameters hallucinated by SLMs — now warns instead of failing (base_tool.py)
- **BUG-009:** `_map_input_to_parameters` strips leading slash from absolute paths via `lstrip('/')` (agent.py)
- **BUG-011:** PythonREPL `_execute()` re-evaluates last `ast.Call` expression causing double `print()` output (python_repl.py)

### Internal
- Test-driven development with real GPU inference across 12 development iterations
- 19 framework bugs discovered and fixed through systematic agent testing
- Compatibility testing across 11 model families (0.5B to 8B parameters)
- Verification sweep: 116 unit tests pass, all integration tests pass

## [0.1.1] - 2026-03-06

### Fixed
- Fixed license inconsistency: all files now correctly reference Apache-2.0 (was MIT in some files)
- Fixed `setup.py` entry point mismatch: `effgen-agent` now correctly points to `agent_main` (was `main`)
- Fixed `setup.py` Development Status: now correctly says Beta (was Alpha)
- Fixed `setup.py` dependency version mismatches with `pyproject.toml` (duckduckgo-search, cloud-secrets, monitoring groups)
- Fixed missing `fastapi` and `uvicorn` in `pyproject.toml` dependencies (`effgen serve` now works out of the box with `pip install effgen`)
- Replaced 5 bare `except:` in `gpu/monitor.py` with specific exception handlers
- Replaced 15+ `print()` calls with proper logger calls in `docker_sandbox`, `decomposition_engine`, `router`, `complexity_analyzer`, `gpu/utils`
- Added logging to silent `except Exception:` handlers in execution modules (`docker_sandbox.py`, `sandbox.py`, `code_executor.py`)

### Added
- `NEWS.md` with user-friendly release summaries
- 6 new example scripts: `preset_agents`, `streaming_agent`, `memory_agent`, `multi_tool_agent`, `weather_agent`, `plugin_example`
- Updated `examples/README.md` with descriptions for all examples
- Top-level imports for `ToolFallbackChain`, `CircuitBreaker`, `ToolPromptGenerator`, `AgentSystemPromptBuilder`
- CLI smoke tests (`tests/integration/test_cli.py`)
- API server tests (`tests/integration/test_api_server.py`)
- Plugin system tests (`tests/unit/test_plugin.py`)
- Preset tests (`tests/unit/test_presets.py`)
- Fallback chain tests (`tests/unit/test_fallback.py`)
- Circuit breaker tests (`tests/unit/test_circuit_breaker.py`)
- Benchmark baseline (`tests/benchmarks/baseline.json`)

### Changed
- All error handlers in `gpu/monitor.py` now catch specific exceptions instead of bare `except:`
- Diagnostic output in `docker_sandbox`, `decomposition_engine`, `router`, `complexity_analyzer`, and `gpu/utils` now uses structured logging

### Internal
- Lint cleanup via ruff (2200+ auto-fixes)
- mypy fixes on modified files
- Validated all GitHub Actions YAML files

---

## [0.1.0] - 2026-03-01

### Added

#### Foundation Hardening
- **ToolPromptGenerator**: Dynamic system prompts with exact tool usage examples for SLMs
- **Model-Specific Prompts**: Optimized prompt formatting for Qwen, Llama, and Phi model families
- **Tool Fallback Chains**: Automatic fallback when tools fail (e.g., calculator → python_repl → code_executor)
- **CircuitBreaker**: Tracks tool failure rates and temporarily disables failing tools
- **Enhanced Tool Descriptions**: Structured format with parameter types, defaults, and usage examples
- **Retry Logic**: Exponential backoff for empty model responses with temperature adjustment
- **Partial Answer Extraction**: Extracts best answer from scratchpad when max iterations reached
- **Input Sanitization**: Validates and sanitizes all tool inputs before execution
- **Async Context Manager**: `async with Agent(config) as agent:` support
- **True Async**: `run_async()` is now natively async (not executor-wrapped)

#### Tool Ecosystem (7 New Tools — 14 Total)
- **BashTool**: Shell command execution with security controls (command whitelist/blacklist)
- **WeatherTool**: Weather data via Open-Meteo API (free, no API key required)
- **JSONTool**: Parse, query (JSONPath), transform, and validate JSON
- **DateTimeTool**: Current time, timezone conversion, date arithmetic
- **TextProcessingTool**: Word count, regex operations, text comparison
- **URLFetchTool**: Fetch and extract text from web pages
- **WikipediaTool**: Search and retrieve Wikipedia articles (free API)
- **Enhanced Retrieval**: Document loaders (txt, md, pdf, csv, json), chunking strategies, hybrid search (vector + BM25)
- **Enhanced AgenticSearch**: ripgrep backend, multi-query, file-type awareness, summarization
- **AgentSystemPromptBuilder**: Auto-generates tool-aware system prompts per agent configuration

#### Protocols & Streaming
- **ACP Protocol Complete**: Full JSON Schema validation, server/client modes, async task polling
- **MCP Client Enhanced**: Auto-reconnection, MCP→effGen tool bridge, resource→context bridge, health monitoring
- **Real Streaming**: True token streaming via generate_stream() (replaces placeholder)
- **Streaming Callbacks**: on_thought, on_tool_call, on_observation, on_answer
- **SSE Streaming**: Server-Sent Events endpoint for real-time API streaming
- **Memory Integration**: ShortTermMemory, LongTermMemory, VectorMemoryStore connected to Agent
- **Memory Configuration**: Configurable backends, persistence paths, auto-summarization

#### Infrastructure
- **CI/CD Pipelines**: GitHub Actions for CI, releases, docs, nightly tests, health checks, PR gates
- **Health Monitoring**: Website, DNS, SSL checks for effgen.org and docs.effgen.org
- **Test Suite**: 67 unit tests, 8 benchmarks, integration and e2e tests with MockModel and fixtures
- **Observability**: OpenTelemetry tracing (no-op fallback), Prometheus metrics
- **`effgen health` Command**: CLI health checker for all infrastructure
- **Code Quality**: Pre-commit hooks (black, isort, flake8, mypy, bandit), CONTRIBUTING.md

#### Developer Experience
- **Plugin System**: ToolPlugin base class with entry point and directory discovery
- **Agent Presets**: Ready-to-use configs — math, research, coding, general, minimal
- **`create_agent()` Factory**: One-line agent creation from presets
- **CLI Enhancements**: Rich progress, verbose/explain modes, tab completion (bash/zsh/fish), session persistence
- **API Server**: WebSocket streaming, API key authentication, rate limiting, OpenAPI docs, /health, /metrics
- **Documentation**: API reference, 6 tutorials, architecture guide, configuration reference, FAQ, migration guide
- **Packaging**: py.typed (PEP 561), Dockerfile, conda-forge recipe, optional dependency groups

### Changed
- `_get_tools_description()` now outputs structured format with parameter details
- `stream()` now uses real token streaming (previously character-by-character placeholder)
- `run_async()` is now truly async (previously wrapped sync in executor)
- Memory system uses proper ShortTermMemory/LongTermMemory classes (previously plain list)
- ACP `validate_request()` now does full JSON Schema validation (previously only checked required fields)
- User-Agent strings now use dynamic version from `effgen.__version__`
- Development status upgraded from Alpha to Beta

### Fixed
- All `NotImplementedError` paths in retrieval tool
- ACP TODO for JSON schema validation
- Streaming placeholder (`time.sleep(0.01)`)
- Memory as plain list (`self.short_term_memory = []`)
- Direct inference path now includes conversation history for multi-turn context retention

---

## [0.0.2] - 2026-02-03

### Added
- **Retrieval Tool**: RAG-based semantic search tool for knowledge base Q&A
- **Agentic Search Tool**: Grep-based exact match search with async support

### Fixed
- **vLLM Backend**: Fixed automatic chat template support for instruction-tuned models
- **GPU Memory Control**: Improved `gpu_memory_utilization` parameter handling
- **OOM Error Handling**: Better error messages and suggestions for CUDA out-of-memory errors
- **Tensor Parallel Auto-Selection**: Fixed auto-detection of tensor parallel size for small models (1.7B, 4B, etc.)
- **vLLM Cache Directory**: Resolved issues with vLLM cache directory handling

### Changed
- **Model Loader**: Improved small model detection for tensor parallel size selection
- **Version Management**: Consolidated `__version__` to single source in main `effgen/__init__.py`

### Compatibility
- Tested with multiple model families:
  - Qwen (Qwen3-1.7B, Qwen2.5-3B-Instruct)
  - Meta Llama (Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct)
  - Microsoft Phi (Phi-4-mini-instruct)
  - HuggingFace SmolLM (SmolLM2-1.7B-Instruct, SmolLM3-3B)
  - Google Gemma (Gemma-3-4b-it)

---

## [0.0.1] - 2026-01-31

### Added

#### Core Framework
- **Agent System**: Complete agentic framework optimized for Small Language Models (1B-7B parameters)
- **Task Management**: Task and SubTask classes with priority levels and status tracking
- **Agent State**: Comprehensive state management for agent execution
- **ReAct Pattern**: Reasoning and Acting pattern implementation for structured problem-solving

#### Model Support
- **Multi-Backend Support**:
  - HuggingFace Transformers (local models)
  - vLLM (fast inference with 5-10x speedup)
  - OpenAI API adapter
  - Anthropic API adapter
  - Google Gemini API adapter
- **Model Loader**: Automatic model detection and loading with intelligent fallback
- **Generation Configuration**: Flexible configuration for temperature, tokens, sampling, etc.

#### Tool System
- **Built-in Tools**:
  - Calculator (basic math, conversions, financial calculations)
  - Web Search (DuckDuckGo integration with caching)
  - Code Executor (Python, JavaScript, Bash in sandboxed environment)
  - File Operations (read, write, list, search)
  - Python REPL (interactive Python execution)
- **Tool Registry**: Dynamic tool registration and discovery
- **Protocol Support**:
  - MCP (Model Context Protocol) - Official Anthropic SDK integration
  - A2A (Agent-to-Agent) protocol
  - ACP (Agent Communication Protocol)

#### Prompt Engineering
- **Template Manager**: Jinja2-based template system with versioning
- **Chain Manager**: Multi-step prompt chaining with conditional execution
- **Prompt Optimizer**: SLM-specific optimization techniques
- **Few-Shot Learning**: Dynamic example selection for improved performance

#### Memory Systems
- **Short-Term Memory**: Conversation history and context management
- **Long-Term Memory**: Persistent storage with importance-based retrieval
- **Vector Store**: Semantic search with FAISS, ChromaDB, and Qdrant support
- **Storage Backends**: JSON and SQLite storage options

#### Task Decomposition
- **Complexity Analysis**: Automatic task complexity assessment
- **Decomposition Engine**: Break complex tasks into manageable subtasks
- **Sub-Agent Manager**: Specialized sub-agents for different task types
- **Orchestrator**: Coordinate multi-agent execution with parallel/sequential strategies

#### GPU Management
- **GPU Allocator**: Intelligent GPU allocation with memory requirements
- **GPU Monitor**: Real-time monitoring of utilization, temperature, and power
- **Multi-GPU Support**: Automatic distribution across available GPUs

#### Code Execution
- **Sandboxed Execution**: Safe code execution with Docker containers
- **Code Validator**: Static analysis and security checks
- **Multiple Languages**: Support for Python, JavaScript, Bash, and more
- **Resource Limits**: Configurable CPU, memory, and timeout limits

#### Configuration
- **YAML Configuration**: Hierarchical configuration with validation
- **JSON Schema Validation**: Type-safe configuration with comprehensive schemas
- **Environment Variables**: Secure secret management with .env support
- **Cloud Secrets**: AWS Secrets Manager, HashiCorp Vault, Azure Key Vault integration

#### CLI Interface
- **Interactive Chat**: Real-time chat interface with rich formatting
- **One-Shot Execution**: Direct task execution from command line
- **API Server**: FastAPI-based REST API server
- **Web Agent**: Autonomous web browsing and interaction
- **Tool Management**: List, inspect, and test tools

#### Utilities
- **Logging System**: Rich, structured logging with multiple levels and formats
- **Metrics Tracking**: Performance metrics, token usage, and cost tracking
- **Error Handling**: Comprehensive error handling with retry logic
- **Async Support**: Full async/await support for concurrent operations

#### Examples & Documentation
- **Basic Agent Example**: Simple agent with calculator and web search
- **Web Agent Example**: Agent that can browse and extract information
- **Installation Script**: Interactive installer with animations
- **Security Policy**: Comprehensive security guidelines and vulnerability reporting

### Configuration Files
- `pyproject.toml`: Modern Python packaging with build system configuration
- `setup.py`: Traditional setuptools configuration for compatibility
- `.gitignore`: Comprehensive ignore patterns for Python, IDEs, and system files
- `requirements.txt`: Core dependencies with version specifications

### Package Metadata
- **License**: MIT License
- **Python Support**: 3.10, 3.11, 3.12, 3.13
- **Development Status**: Alpha
- **Keywords**: ai, agents, llm, slm, language-models, tool-use, multi-agent

### Optional Dependencies
- `dev`: Development tools (pytest, black, isort, flake8, mypy)
- `vllm`: Fast inference engine
- `flash-attn`: Flash Attention for faster transformer inference
- `vector-db`: Vector database backends (FAISS, ChromaDB, Qdrant)
- `search`: Advanced search engines (Google, DuckDuckGo)
- `cloud-secrets`: Cloud secret management (AWS, Azure, Vault)
- `monitoring`: Experiment tracking (Weights & Biases, TensorBoard)
- `all`: All optional dependencies combined

### Entry Points
- `effgen`: Main CLI entry point
- `effgen-agent`: Agent-specific commands
- `effgen-web`: Web agent interface

---

## Version History

### Version Naming Convention
- **Major.Minor.Patch** (Semantic Versioning)
- **Major**: Breaking changes, major new features
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, minor improvements

### Release Schedule
- **Patch releases**: As needed for critical bugs
- **Minor releases**: Monthly feature updates
- **Major releases**: Quarterly for significant changes

---

## Links

- **GitHub**: https://github.com/ctrl-gaurav/effGen
- **PyPI**: https://pypi.org/project/effgen/
- **Documentation**: https://effgen.org/docs/
- **Issues**: https://github.com/ctrl-gaurav/effGen/issues

---

## Contributors

Thank you to all contributors who helped make effGen possible!

- Gaurav Srivastava (@ctrl-gaurav) - Creator and maintainer

---

[Unreleased]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.7...HEAD
[0.2.7]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/ctrl-gaurav/effGen/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ctrl-gaurav/effGen/compare/v0.0.2...v0.1.0
[0.0.2]: https://github.com/ctrl-gaurav/effGen/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ctrl-gaurav/effGen/releases/tag/v0.0.1
