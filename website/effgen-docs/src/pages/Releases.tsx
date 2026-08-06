import React from 'react';
import { Rocket } from 'lucide-react';
import DocPage, { ApiTable, InfoBox, QuickLinks } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Releases() {
  return (
    <DocPage
      title="Release Notes"
      subtitle="Everything added after v0.2.0: Cerebras and OpenAI in v0.2.1, Gemini and Anthropic in v0.2.2, the expanded provider ecosystem in v0.2.3, the policy-based ModelRouter, transparent failover, and effgen cost CLI in v0.2.4, 13 new free / no-auth tools in v0.2.5, 14 more in v0.2.6 plus media + notify presets (58+ built-in tools total), the Prompt Library in v0.2.7 (31 templates, golden + live eval, CLI, playground), multimodal image / audio / video input across 6 providers in v0.2.8, a full observability & reliability stack in v0.2.9, security, edge & DX in v0.2.10 (sandboxed CodeExecutor, OIDC auth + RBAC + audit log, Docker / Helm / Lambda / Cloudflare deploys, VSCode extension, Jupyter magics, local dashboard), the v0.3.0 stabilization & hardening release (fail-closed errors, a self-updating model catalog, real GPU support, a fail-closed server, sandboxed built-in tools, a near-instant import effgen), and — most recently — the v0.3.1 real-world usability & polish release: grounded response.sources / .citations, reasoning models that finish token-heavy tasks, custom personas honored on every path, one-call domain agents, honest multi-agent teams and an honest OpenAI-compatible server, physical GPU memory in models status, and effgen run --json."
      icon={<Rocket size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Release Notes' },
      ]}
    >
      <InfoBox type="success" title="Current release">
        <p>
          The current documented release is <strong>effGen v0.3.1</strong>, released on
          <strong> June 29, 2026</strong>. It is a real-world usability &amp; polish release that
          is fully backward-compatible with v0.3.0 — it adds no new providers or subsystems and
          instead makes grounded results carry their sources, reasoning models finish token-heavy
          work, custom personas steer every path, multi-agent teams fail honestly, the
          OpenAI-compatible server stop silently downgrading, and a knowledge domain become a
          runnable agent in one call. Every improvement is additive; there are no breaking API changes.
        </p>
      </InfoBox>

      <h2>Release Train Summary</h2>
      <ApiTable
        headers={['Version', 'Date', 'Primary scope', 'Compatibility']}
        rows={[
          ['0.2.1', 'April 25, 2026', 'Cerebras backend plus modern OpenAI reasoning-model controls, structured outputs, prompt caching, and native tools', 'No breaking changes'],
          ['0.2.2', 'April 28, 2026', 'Gemini thinking/grounding/files/native tools plus Anthropic Claude 4.x, thinking, caching, and experimental adapter specs', 'No breaking changes'],
          ['0.2.3', 'May 4, 2026', 'Groq, Together AI, Fireworks, Replicate, HuggingFace Inference, ProviderRegistry, doctor, and parity matrix', 'No breaking changes'],
          ['0.2.4', 'May 14, 2026', 'Policy-based ModelRouter (FirstAvailable/Cost/Latency) with transparent failover, SQLite cost + rate-limit stores, effgen cost CLI, BudgetExceededError', 'No breaking changes'],
          ['0.2.5', 'May 18, 2026', '13 new free / no-auth tools — academic (PubMed, ArXiv, SemanticScholar), news & RSS (RSS, News), YouTube (Transcript, Metadata), social (Reddit, HackerNews), translation (Translate, LanguageDetect), QR (Generate, Read). 44+ built-in tools total. research + general presets expanded.', 'No breaking changes'],
          ['0.2.6', 'May 19, 2026', '14 new tools — OCR, AudioTranscribe, ImageInfo, ImageCaption, PDF, DOCX, Excel, Weather, Geocode, Maps, EmailSMTP, EmailIMAP, SlackWebhook, DiscordWebhook — plus media + notify presets. 58+ built-in tools total. research + general presets expanded with docs.', 'No breaking changes'],
          ['0.2.7', 'May 20, 2026', 'Prompt Library — 31 curated templates across research / coding / data-SQL / legal / medical / creative / business, with PromptRegistry auto-discovery, PromptEval golden + live harness, effgen prompts CLI, and interactive playground REPL', 'No breaking changes'],
          ['0.2.8', 'May 21, 2026', 'Multimodal Input — image / audio / video as first-class types across Gemini, OpenAI, Groq, Anthropic, Together, HF + local MLX-VLM. Unified ContentPart Message schema, per-provider preprocessing, capability gating, multimodal preset, MultimodalDescribeTool, 5 cookbook walkthroughs', 'No breaking changes'],
          ['0.2.9', 'May 23, 2026', 'Observability & Reliability — structured JSON logging + secret redaction, Prometheus histograms, SLO burn-rate tracking, OTel tracing with samplers, timeouts / jittered retries / circuit breakers / bulkheads, deterministic chaos harness, Hypothesis fuzz suite, effgen loadtest CLI, 6 Alertmanager rules', 'No breaking changes'],
          ['0.2.10', 'May 27, 2026', 'Security, Edge & DX — sandboxed CodeExecutor (Docker / unprivileged-namespace subprocess), OIDC auth + RBAC + per-request audit log, gitleaks + CycloneDX SBOM + pip-audit + hash verification, Docker / Helm / AWS Lambda / Cloudflare Worker deploys, VSCode extension, Jupyter magics, local dashboard', 'No breaking changes'],
          ['0.3.0', 'June 19, 2026', 'Stabilization & Hardening — fail-closed typed errors (no silent success), self-updating drift-aware model catalog (effgen models refresh), real GPU support (driver-compatible torch, temperature=0 greedy, NVML allocator), server fails closed (forged JWTs rejected, 502 not 401), hardened tools (shared SSRF guard, out-of-process PythonREPL timeout, path confinement, no eval/pickle), instant import (~7.5s → ~20ms), faster streaming + early-stopping agent loop, quiet --json CLI, live thinking UX', 'No breaking changes (additive aliases only)'],
          ['0.3.1', 'June 29, 2026', 'Real-World Usability & Polish — grounded response.sources / .citations, reasoning models (gpt-5 / o-series) finish token-heavy tasks, cost + latency on every result, custom personas honored on every path, one-call domain agents (LegalDomain().to_agent()), honest multi-agent teams + workflow DAGs, honest OpenAI-compatible server (no silent tool/embedding downgrade), physical GPU memory in models status, grammar-constrained local structured output, no MCP deadlock on sync run(), tool-plugin auto-discovery, effgen run --json to stdout, REPL sandbox out of the model’s hands', 'No breaking changes (additive only)'],
        ]}
      />

      <h2>v0.3.1 — Real-World Usability &amp; Polish</h2>
      <p>
        <strong>effGen v0.3.1</strong> is a real-world usability &amp; polish release, driven by
        living with the framework as eleven professionals do — a finance analyst, a journalist, a
        researcher, a founder, a backend engineer, a support lead, an ML engineer, an educator, a
        security reviewer, an integration engineer, and a legal knowledge manager. It adds no new
        providers or subsystems; it seals the sharp edges those users hit first.{' '}
        <strong>No breaking API changes.</strong>
      </p>
      <ApiTable
        headers={['Area', 'What changed', 'Where']}
        rows={[
          ['Traceable evidence', 'response.sources and response.citations are populated from the URLs a run actually retrieved (web_search / url_fetch / news / wikipedia) and from provider-native grounding (OpenAI url_citation annotations, surfaced as metadata["grounding_chunks"]; Gemini search grounding) — never from the model’s prose. The research preset cites only a URL one of its tools returned this run.', <code>effgen.core.agent</code>],
          ['Reasoning models finish the job', 'gpt-5 family and o-series no longer return empty, billed results on token-heavy tasks: they get a larger default output budget across every path, a finish_reason="length" empty is treated as truncation (grown once and retried, or an actionable error), and a starved budget is never retried three times. effgen batch gained --max-tokens.', <code>effgen.core.agent</code>],
          ['Costed, measurable results', 'cost_usd, prompt/completion/total tokens, and latency_ms / duration_s land on every AgentResponse and raw GenerationResult metadata (local stays honestly cost-free). Teams and workflows report summed cost. A shared adaptive formatter shows real sub-cent costs instead of $0.0000. str(GenerationResult) returns the text and a notebook card renders model.generate() cleanly.', 'package-wide'],
          ['Personas honored everywhere', 'A custom system_prompt now steers every response — it was silently dropped on the no-tool direct path, the streaming path chat uses, and the native/hybrid tool path. New chat --system-prompt / --persona, an education.* prompt set, and prompts list --json.', <code>effgen.core.agent</code>],
          ['One-call domain agents', 'A knowledge domain becomes a runnable agent in one call: LegalDomain().to_agent("gpt-5-nano") (or create_agent(domain=...)) wires the domain’s prompt + recommended tools + guardrails. A RAG agent accepts a pre-built VectorMemoryStore as its knowledge_base, and the everyday guardrail classes are exported at the top level.', <code>effgen.domains</code>],
          ['Honest teams & workflows', 'Collaborative teams fail closed with a per-agent error; hierarchical teams route each subtask to the worker the manager named and run every subtask; a workflow DAG marks a node downstream of a failure as skipped (so an internal error never becomes a customer-facing reply); failed runs no longer echo the input back as the answer.', <code>effgen.core.orchestrator</code>],
          ['Honest OpenAI-compatible server', 'No silent client-tool drop — an unhosted function tool is rejected with a clear 400 (unknown_tool). /v1/embeddings strips a provider: prefix and reflects its real backend (or fails closed under EFFGEN_EMBEDDINGS_STRICT=1) instead of quietly serving TF-IDF vectors. Auth / validation / rate-limit errors share the OpenAI error envelope; empty messages → 400; per-call cost in the effgen extension; serve --help documents env knobs + --rate-limit.', <code>effgen.server</code>],
          ['Local-first truth', 'models status shows physical GPU memory across all processes (with a utilization column); models info recognizes a model in the local cache instead of routing to the cloud; incomplete downloads are flagged. Local Transformers batch is thread-safe. The optional effgen[grammar] extra lets small local models emit schema-valid JSON in one constrained pass. compare breaks accuracy ties on latency then tokens.', <code>effgen.cli / effgen.gpu</code>],
          ['Dependable automation', 'Synchronous Agent.run() no longer hangs forever on a loop-bound MCP tool — it is bounded by the timeout and raises a clear TimeoutError pointing at run_async(). Installed tool plugins (an effgen.plugins entry point) auto-discover. effgen run --json emits the full result document to stdout; --json is added to eval / compare / workflow / sessions list.', <code>effgen.tools / effgen.cli</code>],
          ['Hardened code execution', 'The Python REPL restricted_mode toggle is out of the model-facing schema — unrestricted execution is a developer-only opt-in. The bash env scrub strips every provider credential and refuses common secret files, and bash is no longer bundled in the general preset. Broader prompt-injection detection, credential-aware PII redaction, and an IPv4 that ends a sentence is now redacted.', <code>effgen.tools.builtin / effgen.guardrails</code>],
          ['RAG ingestion & ergonomics', 'PDFs ingest out of the box (fallback to pypdf / pdfplumber) and a skipped file names why. extra_tools accepts tool-name strings, tools= aliases extra_tools, bare Agent() teaches, create_agent reports an unknown preset before demanding a model, and effgen run with no -m mirrors quickstart and says which model it chose and why.', 'package-wide'],
        ]}
      />

      <h2>v0.3.0 — Stabilization &amp; Hardening</h2>
      <p>
        <strong>effGen v0.3.0</strong> is a major stabilization release. It adds no new providers,
        tools, prompt templates, or subsystems — instead it makes everything already in effGen{' '}
        <strong>robust, predictable, fast, secure, and pleasant to use</strong>. Failures are now
        loud and typed instead of silently succeeding; the model catalog updates itself and warns
        when it drifts; local GPUs work out of the box; the API server fails closed; the built-in
        tools are sandboxed and SSRF-safe; streaming and the agent loop are dramatically faster;
        the CLI is quiet and scriptable; and <code>import effgen</code> is effectively instant.
        No breaking API changes — every ergonomic addition is an additive alias.
      </p>
      <ApiTable
        headers={['Area', 'What changed', 'Where']}
        rows={[
          ['Fail-closed errors', 'Agent.run() never returns success=True with empty output; the direct and tool paths return the same failure shape (success=False, a coarse metadata["reason"] stage label, and a typed redacted metadata["error"] dict). classify_provider_error() populates metadata["error"]["category"] with a stable taxonomy: auth / not_found / rate_limited / transient / timeout / fatal. AgentConfig.raise_on_error opts into exceptions.', <code>effgen.core.agent</code>],
          ['Smart retries', 'Retries fire only on retryable / rate-limited errors; auth and not-found fast-stop with one clear message. A 404 model id suggests the nearest live alternative.', <code>effgen.core.agent</code>],
          ['Self-updating catalog', 'Every provider ships a local snapshot (id, context, max output, input/output price, tool/vision/audio support, free-tier flag, rate limits) with a count and a "verified on" date. effgen models refresh diffs the live API; check_drift() warns once when stale. Filters to chat/text base models; never persists ft: / embeddings / audio / image ids.', <code>effgen.models</code>],
          ['Real GPU support', 'Documented install selects a driver-compatible torch; a runtime guard warns once when NVML sees GPUs but torch.cuda cannot. temperature<=0 is greedy decoding (do_sample=False) across local backends. The allocator no longer deadlocks on reset() and reads real free memory via NVML / torch.cuda.mem_get_info.', <code>effgen.gpu</code>],
          ['Server fails closed', 'With no configured issuer/JWKS outside dev mode, the server rejects all bearer tokens — a forged HS256 JWT cannot reach /whoami or /v1/chat/completions; /v1/* return 401 without credentials. CORS no longer combines wildcard origin with credentials; metrics + dashboard require auth; viewer cannot run tools; budget reserves then reconciles; upstream/auth/missing-key failures map to 502/503, not 401.', <code>effgen.server</code>],
          ['Hardened tools', 'PythonREPL runs user code in a worker subprocess with an out-of-process wall-clock timeout, process-group kill, and memory/output caps. One shared SSRF guard (tools/builtin/_net.py) re-validates on every redirect for every URL tool. Path confinement (tools/builtin/_fs.py) on every file tool; the un-gated pickle.load path was removed (JSON-only state); prompt-chain conditions use an AST-whitelist comparator, not eval().', <code>effgen.tools.builtin</code>],
          ['Performance', 'import effgen dropped from ~7.5 s / ~800 MB to ~0.02 s / ~12 MB via lazy resolution. Faster, truly incremental streaming with typed mid-stream errors. The agent loop short-circuits repeated tool calls and stops once a confident answer exists (6 tool calls / 66 s → 1). Native JSON modes preferred for structured output.', 'package-wide'],
          ['Scriptable CLI', 'Quiet by default (clean tables, no INFO spam); --json on doctor / models / tools / cost; --provider on run / chat / debug; catalog-backed models list/info/refresh; doctor distinguishes "key present" from "key usable" and gains doctor --live --cheap; user errors exit non-zero.', <code>effgen.cli</code>],
          ['Ergonomic aliases', 'The obvious calls work: create_agent(preset, model, name="X"), TemplateManager() populated by default, ConfigLoader.load, ShortTermMemory.get_messages, TestCase(input=, expected=), and a @tool / Tool.from_function() helper. A bare/invalid constructor raises a clear accepted-kwargs error.', 'package-wide'],
          ['Packaging', 'Reconciled extras with a single source of truth ([api], [server], [rag], [tools-web], [tools-docs], [local], [vllm], [all]); a constraints-cu1xx.txt flow that protects a working torch; one canonical installer; pip-audit clean across documented extras; pypdf bumped to >=6.13.3 (GHSA-jm82-fx9c-mx94).', <code>pyproject.toml</code>],
        ]}
      />
      <CodeBlock
        code={`# v0.3.0 — failures are loud and typed, never silent
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL

model = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")
agent = Agent(config=AgentConfig(name="math_agent", model=model, tools=[Calculator(), PythonREPL()]))

result = agent.run("What is 24344 * 334?")
if not result.success:
    print(result.metadata["error"]["category"])  # stable taxonomy: auth / not_found / rate_limited / transient / timeout / fatal
    print(result.metadata["reason"])             # coarse stage label, e.g. "generation_failed"
else:
    print(result.output)`}
        language="python"
        filename="v030_failclosed.py"
      />
      <CodeBlock
        code={`# v0.3.0 — a model catalog that updates itself, and a quiet scriptable CLI
effgen models refresh                 # pull the live list, report added / removed / changed
effgen models refresh --provider cerebras
effgen models list --json             # catalog-backed: status, price, context, deprecation, verified-on
effgen doctor --live --cheap          # which provider keys are actually usable (not just present)
effgen run --provider groq "Summarize the theory of relativity in two sentences."`}
        language="bash"
        filename="terminal"
      />
      <InfoBox type="info" title="Upgrading from v0.2.x">
        <p>
          No code changes are required. If you previously relied on <code>Agent.run()</code>{' '}
          returning <code>success=True</code> with an empty answer, you should now branch on{' '}
          <code>result.success</code> and read the typed{' '}
          <code>result.metadata["error"]["category"]</code> (with a coarse{' '}
          <code>result.metadata["reason"]</code> stage label) — the failure shape is consistent
          across the direct and tool paths. <code>temperature=0</code>{' '}
          now decodes greedily instead of raising. If you self-host the API server, confirm your
          OIDC issuer/JWKS are configured (the server now rejects bearer tokens otherwise).
        </p>
      </InfoBox>

      <h2>v0.2.10 — Security, Edge &amp; Developer Experience</h2>
      <p>
        <strong>effGen v0.2.10</strong> hardens the framework end-to-end and adds production
        deployment targets plus three developer-experience surfaces. Every security and DX
        feature is additive — no breaking API changes.
      </p>
      <ApiTable
        headers={['Area', 'What shipped', 'Where']}
        rows={[
          ['Sandboxed CodeExecutor', 'Docker by default (--read-only, --network=none, --cap-drop=ALL, 256m); unprivileged-namespace subprocess fallback; Firecracker/Off stubs', <code>effgen.security.sandbox</code>],
          ['Auth', 'OIDC/JWT validation (authlib) on every non-public endpoint; EFFGEN_DEV_MODE bypass with loud warning', <code>effgen.server.auth</code>],
          ['RBAC + budget', 'Union-of-roles policy; pure-ASGI RBACBudgetMiddleware enforces tool allow-lists (403) and daily cost cap (429 BudgetExceeded)', <code>effgen.server.rbac</code>],
          ['Audit log', 'Per-request JSONL at ~/.effgen/audit/<date>.jsonl; redacted; principal/role/endpoint/outcome', <code>effgen.server.audit</code>],
          ['Supply chain', 'gitleaks pre-commit + CI, CycloneDX SBOM, pip-audit CI, hash-verified locks, EFFGEN_VERIFY_HASHES startup check', <code>.gitleaks.toml</code>, <code>sbom.cdx.json</code>],
          ['Deploy', 'Multi-stage Dockerfile, Helm chart (HPA/PDB/NetworkPolicy), AWS Lambda (Mangum + SAM), Cloudflare Worker edge proxy', <code>deploy/</code>],
          ['DX', 'VSCode extension (template completion + run code lens), Jupyter magics, live /dashboard SPA', <code>tools/vscode-effgen</code>, <code>effgen.jupyter</code>, <code>effgen.dashboard</code>],
        ]}
      />
      <CodeBlock
        code={`# v0.2.10 — CodeExecutor is sandboxed by default
import asyncio
from effgen.security.sandbox import get_sandbox, SandboxConfig

async def main():
    config = SandboxConfig(backend="subprocess", timeout=10)  # or backend="docker"
    sandbox = await get_sandbox(config)
    result = await sandbox.run('print("hello")', "python", config)
    print(result.stdout)  # hello

asyncio.run(main())`}
        language="python"
        filename="v0210_sandbox.py"
      />
      <CodeBlock
        code={`# v0.2.10 — serve with OIDC auth + RBAC, then deploy
export EFFGEN_OIDC_ISSUER=https://your-tenant.auth0.com/
export EFFGEN_OIDC_CLIENT_ID=https://effgen.api
export EFFGEN_OIDC_JWKS_URI=https://your-tenant.auth0.com/.well-known/jwks.json
effgen serve --port 8000

# Docker
docker build -f deploy/docker/Dockerfile --build-arg EXTRAS=server -t effgen:0.2.10 .
docker run -p 8000:8000 --env-file .env effgen:0.2.10

# Helm / Lambda / Cloudflare
helm install effgen deploy/k8s/helm/effgen/ -f deploy/k8s/helm/effgen/values.yaml
cd deploy/aws_lambda && sam build && sam deploy
cd deploy/cloudflare && wrangler deploy`}
        language="bash"
        filename="terminal"
      />
      <CodeBlock
        code={`# v0.2.10 — Jupyter magics
%load_ext effgen.jupyter
%effgen_chat "What is 17 * 23?"
%%effgen_agent general
Summarise the top HackerNews stories today.

# Local dashboard at /dashboard (v0.2.10 shipped it auth-exempt; v0.3.0 protects it by default)
# EFFGEN_DEV_MODE=1 uvicorn effgen.server.app:create_app --factory --port 8080`}
        language="python"
        filename="v0210_dx.py"
      />
      <InfoBox type="info" title="Validation (v0.2.10)">
        <p>
          Full regression suite: <strong>3721 passed, 0 failed</strong> (88 skipped, 14 xfailed).
          gitleaks detects planted secrets and exits clean on the repo; CycloneDX SBOM validates
          against the 1.5 schema; pip-audit reports 0 HIGH/CRITICAL; the SubprocessSandbox blocks
          network and isolates <code>/tmp</code>; the API server returns 401 unauthenticated, 403
          on denied tools, and 429 on budget; Docker, Helm (kubeconform K8s 1.29), Lambda
          (live Cerebras call), and the Cloudflare Worker (live round-trip) all pass.
        </p>
      </InfoBox>
      <p>
        See <a href="/docs/security">Security</a>, <a href="/docs/deployment">Deployment</a>, and{' '}
        <a href="/docs/dx">Developer Experience</a> for full reference docs.
      </p>

      <h2>v0.2.9 — Observability &amp; Reliability</h2>
      <p>
        <strong>effGen v0.2.9</strong> turns effGen into something you can operate in production.
        Every telemetry path is async / non-blocking — a failed export never fails inference.
      </p>
      <ApiTable
        headers={['Area', 'What shipped']}
        rows={[
          ['Structured logging', 'StructuredFormatter emits JSON {ts, level, module, event, attributes, trace_id, span_id}; agent loop / adapters / router / tools migrated off ad-hoc print.'],
          ['Secret redaction', 'Redactor strips OpenAI/Anthropic/Cerebras/Google/HF/Groq keys, Bearer tokens, and Slack/Discord webhook URLs at the log encoder — every path covered.'],
          ['Metrics', 'Prometheus histograms (model/tool/agent latency) + token counters; GET /metrics.'],
          ['SLO tracking', 'SLO + SLOTracker rolling-window error budgets; burn_rate(); GET /slo.'],
          ['Tracing', 'OTel samplers (AlwaysOn/Off, TraceIdRatio, RateLimited, ParentBased); canonical span-attribute spec in spans.py.'],
          ['Reliability', 'Explicit timeouts (no timeout=None), jittered retries, three-state circuit breaker per provider, bulkheads.'],
          ['Chaos + fuzz', 'Deterministic Chaos(seed) with 6 fault types & 4 canonical scenarios (273 tests); Hypothesis fuzz over 66 tools, messages, router.'],
          ['Load testing', 'effgen loadtest CLI → throughput + p50/p95/p99 + error rate.'],
          ['Alerting', '6 Alertmanager rules + AlertWebhook (Slack/Discord, redacted, never throws).'],
        ]}
      />
      <CodeBlock
        code={`from effgen.observability import get_logger, record_model_call, export_metrics
from effgen.observability.slo import SLOTracker, SLO

log = get_logger(__name__)
log.event("agent.started", preset="general", model="llama3.1-8b")

record_model_call(provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.42)
print(export_metrics())  # Prometheus text format

tracker = SLOTracker()
tracker.register(SLO("model_success", target_pct=99.0, window_seconds=3600))
tracker.record("model_success", ok=True)
print(tracker.burn_rate("model_success"))`}
        language="python"
        filename="v029_observability.py"
      />
      <CodeBlock
        code={`from effgen.reliability import Retry, retryable, CircuitBreaker, Bulkhead

@retryable(Retry(max_attempts=3, base_delay=0.5, jitter=True))
def call_model(prompt):
    ...

breaker = CircuitBreaker("cerebras", failure_threshold=5, recovery_timeout=30)
if breaker.is_call_permitted():
    try:
        result = call_model("hello"); breaker.on_success()
    except Exception as exc:
        breaker.on_failure(exc); raise

bulkhead = Bulkhead("cerebras", max_concurrency=10, queue_size=50)
with bulkhead.acquire():
    call_model("hello")`}
        language="python"
        filename="v029_reliability.py"
      />
      <p>
        See <a href="/docs/observability">Observability</a> and{' '}
        <a href="/docs/reliability">Reliability</a> for full reference docs.
      </p>

      <h2>v0.2.8 — Multimodal Input</h2>
      <p>
        <strong>effGen v0.2.8</strong> makes image, audio, and video first-class input types.
        A unified <code>ContentPart</code> <code>Message</code> schema, per-provider adapters,
        automatic preprocessing, capability gating, a new <code>multimodal</code> preset, the{' '}
        <code>MultimodalDescribeTool</code>, a local MLX-VLM adapter (Apple Silicon), and 5
        cookbook walkthroughs ship in this release. The old{' '}
        <code>Message(role, "text")</code> constructor still works.
      </p>
      <ApiTable
        headers={['Provider', 'Image', 'Audio', 'Video (native)', 'Video (frames)']}
        rows={[
          ['Gemini 2.x/3.x', '✅', '✅', '✅', '✅'],
          ['OpenAI gpt-4o family', '✅', '✅ (Whisper)', '❌', '✅'],
          ['Groq (Llama 4 / 3.2-vision)', '✅', '❌', '❌', '✅'],
          ['Anthropic (code-only)', '✅', '❌', '❌', '❌'],
          ['Together (vision)', '✅', '❌', '❌', '✅'],
          ['HuggingFace (BLIP/LLaVA)', '✅', '✅ (ASR)', '❌', '✅'],
          ['MLX-VLM (Apple Silicon)', '✅', '❌', '❌', '✅'],
        ]}
      />
      <CodeBlock
        code={`from effgen import load_model, image_from
from effgen.core.messages import Message, Role, TextPart, ImagePart
from effgen.presets import create_agent

# Structured multimodal message
msg = Message(role=Role.USER, content=[
    ImagePart(image=open("/tmp/chart.png", "rb").read(), mime="image/png"),
    TextPart(text="What is the largest bar in this chart?"),
])

# Or use the multimodal preset (auto-routes via MultimodalDescribeTool)
model = load_model("gemini-2.0-flash", provider="gemini")
agent = create_agent("multimodal", model)
print(agent.run("Describe /tmp/chart.png and read off the largest bar.").output)`}
        language="python"
        filename="v028_multimodal.py"
      />
      <InfoBox type="info" title="Capability gating, not silent downcast">
        <p>
          Every adapter raises <code>CapabilityNotSupportedError</code> when the selected model
          lacks <code>vision</code> / <code>audio_input</code> / <code>video_input</code> — no
          image is ever silently replaced with <code>"[image not supported]"</code>. Video on
          non-native providers is sampled to frames via <code>ffmpeg</code> (raises{' '}
          <code>MissingSystemDependency</code> when ffmpeg is absent).
        </p>
      </InfoBox>
      <p>
        See <a href="/docs/multimodal">Multimodal</a> for the full schema, preprocessing
        pipeline, and per-provider guides.
      </p>

      <h2>v0.2.7 — Prompt Library · 31 Templates · CLI + Playground</h2>
      <p>
        <strong>effGen v0.2.7</strong> ships the new <strong>Prompt Library</strong> at{' '}
        <code>effgen.prompts.library</code> — a curated, domain-organized catalog of{' '}
        <strong>31 reusable prompt templates</strong> across research, coding, data/SQL, legal,
        medical, creative, and business. Every template is a Python callable that renders
        deterministically for fixed inputs, ships with a fixture and a golden evaluation test,
        and is accessible through a rich CLI and an interactive playground. No breaking API
        changes — all classes are opt-in additions.
      </p>
      <ApiTable
        headers={['Domain', 'Count', 'Templates', 'Notes']}
        rows={[
          ['Research', '5', 'literature_review.v1.zero_shot, literature_review.v1.cot, paper_summary.v1, citation_extract.v1, methodology_critique.v1', 'CoT + structured + tool-augmented variants.'],
          ['Coding', '5', 'code_review.v1, bug_diagnose.v1, refactor_plan.v1, test_generate.v1, docstring_fill.v1', 'test_generate.v1 live eval asserts ast.parse() passes; refactor_plan reads source via tool.'],
          ['Data / SQL', '5', 'sql_from_nl.v1, sql_explain.v1, sql_optimize.v1, data_profile.v1, etl_plan.v1', 'sql_from_nl.v1 live eval validates via sqlglot.parse(); data_profile is tool-augmented.'],
          ['Legal', '3', 'contract_summarize.v1, clause_classify.v1, legal_research_brief.v1', 'Non-advice disclaimer rendered verbatim in every template, enforced by unit tests.'],
          ['Medical', '3', 'symptom_triage.v1, drug_interaction_query.v1, medical_literature.v1', 'Non-advice disclaimer rendered verbatim in every template, enforced by unit tests.'],
          ['Creative', '5', 'story_continuation.v1.zero_shot, story_continuation.v1.few_shot, poetry_forms.v1, character_bio.v1, world_building.v1', 'Zero-shot, few-shot, and structured creative variants.'],
          ['Business', '5', 'meeting_summary.v1, email_draft.v1, okr_generate.v1, swot_analysis.v1, elevator_pitch.v1', 'Structured + zero-shot variants suitable for ops/sales workflows.'],
        ]}
      />
      <CodeBlock
        code={`# v0.2.7 — browse and render templates from the Prompt Library
from effgen.prompts.library import registry

# Browse every registered template
for p in registry.all():
    print(p.name, p.variant, p.domain)

# Get and render a specific template
p = registry.get("research.literature_review.v1.cot")
prompt_text = p.template(
    topic="diffusion models",
    years_range="2022-2025",
    max_papers=10,
)

# Search by domain and variant
sql_prompts = registry.search(domain="data", variant="structured")`}
        language="python"
        filename="v027_prompt_library.py"
      />
      <CodeBlock
        code={`# v0.2.7 — CLI quick-start
# Discover templates
effgen prompts list
effgen prompts list --domain research --variant cot
effgen prompts list --format markdown          # regenerate docs/prompts/gallery.md

# Inspect a template
effgen prompts show research.literature_review.v1.cot

# Run golden evaluations (no model needed)
effgen prompts eval

# Run live evaluations against a model
effgen prompts eval --domain coding --live --model llama3.1-8b

# Interactive REPL
effgen prompts playground

# Non-interactive render / run
effgen prompts render data.sql_from_nl.v1 \\
    --input '{"schema_ddl": "CREATE TABLE orders (id INT, total FLOAT)", "question": "Total orders this month", "dialect": "sqlite"}'

effgen prompts run    data.sql_from_nl.v1 \\
    --input ... --model groq:llama-3.3-70b-versatile`}
        language="bash"
        filename="terminal"
      />
      <CodeBlock
        code={`# v0.2.7 — golden + live eval harness
from effgen.prompts.library import registry, PromptEval

# Optional kwargs: goldens_dir=Path(...), live_retries=1, live_retry_delay=65.0
evaluator = PromptEval()

# Compare every template's render(fixture) against its stored .txt golden.
# A missing golden is written on first run and counted as a pass.
report = evaluator.eval_all_golden(list(registry))
print(report.as_table())
for r in report.failed():
    print(r.name, r.message)

# Live eval — pass a LibraryPrompt instance (not a name string). Runs the
# rendered prompt through the model and validates expected_shape:
#   sqlglot.parse() for SQL templates, ast.parse() for generated Python,
#   regex / JSON-schema / callable for the rest.
prompt = registry.get("data.sql_from_nl.v1")
live = evaluator.eval_live(prompt, model="groq:llama-3.3-70b-versatile")
print(live.passed, live.message)
print(live.model_output[:200])`}
        language="python"
        filename="v027_prompt_eval.py"
      />
      <InfoBox type="info" title="Variants and registry validation">
        <p>
          Allowed <code>variant</code> values are <code>zero_shot</code>, <code>cot</code>,{' '}
          <code>few_shot</code>, <code>tool</code>, and <code>structured</code> — the registry
          validates this on <code>register()</code> alongside JSON-Schema correctness of{' '}
          <code>input_schema</code> and fixture conformance. Domain packages under{' '}
          <code>effgen/prompts/library/domains/</code> are auto-discovered on first access to
          <code>registry</code>.
        </p>
      </InfoBox>
      <InfoBox type="warning" title="Legal + medical safety">
        <p>
          Every legal and medical template renders a verbatim non-advice disclaimer in its system
          prompt. The disclaimers are unit-tested so they cannot be silently dropped by a future
          refactor — they are policy, not convention.
        </p>
      </InfoBox>
      <p>
        Upgrade with <code>pip install --upgrade effgen</code>. All prompt library classes are
        opt-in; existing <code>TemplateManager</code>, <code>PromptChain</code>, and{' '}
        <code>PromptOptimizer</code> code from v0.2.0–v0.2.6 keeps working unchanged.
      </p>

      <h2>v0.2.6 — Docs · Media · Comms · 58+ Total</h2>
      <p>
        <strong>effGen v0.2.6</strong> adds <strong>14 new built-in tools</strong> across six categories — OCR, audio
        transcription, image analysis, document parsing, geo/weather, and email/webhook communication — raising the total
        built-in tool count from 44 to <strong>58+</strong>. Two new presets (<code>media</code>, <code>notify</code>) join
        the existing roster. Every tool follows the established <code>BaseTool</code> pattern with structured{' '}
        <code>{'{success, data, error}'}</code> output, async <code>_execute()</code>, unit + integration tests, a dedicated
        doc page, and preset integration. No breaking API changes.
      </p>
      <ApiTable
        headers={['Category', 'Tools', 'Notes']}
        rows={[
          ['OCR', 'OCRTool', 'Tesseract (local, primary) + OCR.space fallback (OCR_SPACE_API_KEY). Raises OCRBackendUnavailable with per-OS install. general preset.'],
          ['Audio transcription', 'AudioTranscribeTool', 'faster-whisper auto-detect CPU/GPU + HuggingFace Inference fallback (HF_TOKEN). media preset.'],
          ['Image analysis', 'ImageInfoTool, ImageCaptionTool', 'ImageInfoTool: Pillow info/resize/thumbnail (zero network). ImageCaptionTool: vision-router (Gemini/OpenAI/MLX-VLM); raises NoVisionProviderAvailable.'],
          ['Documents', 'PDFTool, DOCXTool, ExcelTool', 'pypdf + pdfplumber (text/metadata/tables/images), python-docx, openpyxl + pandas. research + general presets.'],
          ['Geo / Weather', 'WeatherTool, GeocodeTool, MapsTool', 'Open-Meteo (no auth), Nominatim (1 req/s token bucket + User-Agent), staticmap (OSM static PNG). general preset.'],
          ['Email (live)', 'EmailSMTPTool, EmailIMAPTool', 'stdlib smtplib (TLS-on) + imaplib. Env: SMTP_* / IMAP_*. Raises MissingCredentialsError. notify preset.'],
          ['Webhooks', 'SlackWebhookTool, DiscordWebhookTool', 'Incoming webhook URLs; redacted in all logs. Env: SLACK_WEBHOOK_URL / DISCORD_WEBHOOK_URL. notify preset.'],
        ]}
      />
      <CodeBlock
        code={`# v0.2.6 — new media preset (audio transcription + vision captioning)
from effgen import load_model
from effgen.presets import create_agent

model = load_model("llama3.1-8b", provider="cerebras")
media_agent = create_agent("media", model)
# Tools wired in: AudioTranscribeTool + ImageCaptionTool

result = media_agent.run("Transcribe /tmp/meeting.mp3 and list the action items.")
print(result.output)`}
        language="python"
        filename="v026_media_preset.py"
      />
      <CodeBlock
        code={`# v0.2.6 — new notify preset (email + Slack + Discord)
from effgen.presets import create_agent
from effgen import load_model

model = load_model("llama3.1-8b", provider="cerebras")
notify_agent = create_agent("notify", model)
# Tools wired in: EmailSMTPTool + EmailIMAPTool + SlackWebhookTool + DiscordWebhookTool

# Credentials are read from env vars:
#   SMTP_HOST/SMTP_USER/SMTP_PASSWORD/SMTP_FROM, IMAP_HOST/IMAP_USER/IMAP_PASSWORD,
#   SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL
notify_agent.run("Send a deploy summary to alice@example.com and post a status to #ops.")`}
        language="python"
        filename="v026_notify_preset.py"
      />
      <CodeBlock
        code={`# v0.2.6 — document + OCR pipeline
from effgen.tools.builtin.ocr import OCRTool
from effgen.tools.builtin.pdf import PDFTool

text = await OCRTool().execute({"operation": "extract", "image_path": "/tmp/scan.png", "lang": "eng"})
tables = await PDFTool().execute({"operation": "tables", "path": "/tmp/paper.pdf"})
print(text["data"]["text"], tables["data"]["tables"][:1])`}
        language="python"
        filename="v026_docs_pipeline.py"
      />
      <InfoBox type="info" title="Install extras">
        <p>
          Selective: <code>pip install -U "effgen[documents]"</code> (PDF/DOCX/Excel),{' '}
          <code>pip install -U "effgen[audio]"</code> (AudioTranscribe),{' '}
          <code>pip install -U "effgen[tools]"</code> (OCR/Image + more). Or grab everything:{' '}
          <code>pip install -U "effgen[all]"</code>.
        </p>
        <p>
          System dependencies (for the relevant tool's primary path):{' '}
          <code>apt-get install tesseract-ocr</code> / <code>brew install tesseract</code> (OCR);{' '}
          <code>apt-get install ffmpeg</code> / <code>brew install ffmpeg</code> (non-WAV audio).
        </p>
      </InfoBox>
      <p>
        <strong>New errors:</strong> <code>OCRBackendUnavailable</code>, <code>MissingSystemDependency</code>,{' '}
        <code>NoVisionProviderAvailable</code>, <code>MissingCredentialsError</code>, <code>CorruptDocumentError</code>.
      </p>

      <h2>v0.2.5 — 13 New Free Tools · 44+ Total</h2>
      <p>
        <strong>effGen v0.2.5</strong> adds <strong>13 new free / no-auth tools</strong> spanning
        academic research, news &amp; RSS, YouTube, social media, translation, language detection,
        and QR codes — bringing the total built-in tool count to <strong>44+</strong>. All tools are
        <code> BaseTool</code> subclasses with the structured <code>{'{success, data, error}'}</code> output shape,
        integrated into the <code>research</code> and <code>general</code> presets, and covered by
        unit + integration tests.
      </p>
      <ApiTable
        headers={['Category', 'Tools', 'Notes']}
        rows={[
          ['Academic research', 'PubMedTool, ArXivTool, SemanticScholarTool', 'NCBI E-utilities (3 req/s, 10/s with NCBI_API_KEY), arXiv Atom feed, Semantic Scholar Graph API (100 req/5 min unauth)'],
          ['News & RSS', 'RSSFeedTool, NewsTool', 'Any RSS/Atom feed; curated reputable sources (Reuters, BBC, HN, NPR, …); optional NEWS_API_KEY'],
          ['YouTube', 'YouTubeTranscriptTool, YouTubeMetadataTool', 'youtube-transcript-api (no Google API key); yt-dlp metadata-only'],
          ['Social', 'RedditTool, HackerNewsTool', 'Public Reddit JSON (no OAuth for reads); HN Firebase API'],
          ['Translation & Language', 'TranslateTool, LanguageDetectTool', 'LibreTranslate + argostranslate offline fallback; langdetect 55+ languages'],
          ['QR Codes', 'QRGenerateTool, QRReadTool', 'Fully local — qrcode lib; pyzbar + Pillow with OpenCV QR fallback'],
        ]}
      />
      <CodeBlock
        code={`# v0.2.5 research preset — academic + news + social + video
from effgen import load_model
from effgen.presets import create_agent

model = load_model("llama3.1-8b", provider="cerebras")
agent = create_agent("research", model)
# Tools wired in: WebSearch, URLFetch, Wikipedia, ArXiv, PubMed, SemanticScholar,
#                 RSSFeed, News, YouTubeTranscript, YouTubeMetadata, Reddit, HackerNews

result = agent.run(
    "Find recent arXiv papers on small language models for agents "
    "and summarise the top HN discussion on the topic."
)
print(result.output)`}
        language="python"
        filename="v025_research_agent.py"
      />
      <CodeBlock
        code={`# v0.2.5 — generate a QR and translate offline (fully local)
from effgen.tools.builtin.translate import TranslateTool
from effgen.tools.builtin.qr_generate import QRGenerateTool

translate = TranslateTool()
qr = QRGenerateTool()

translated = await translate.execute({
    "operation": "translate",
    "text": "Hello, world!",
    "source": "en",
    "target": "fr",
})  # → "Bonjour le monde!"

qr_png = await qr.execute({
    "operation": "generate",
    "data": "https://effgen.org",
    "data_url_return": True,
})  # data:image/png;base64,...`}
        language="python"
        filename="v025_local_tools.py"
      />

      <h2>v0.2.4 - Policy ModelRouter + Cost CLI</h2>
      <p>
        v0.2.4 layers an opt-in <code>PolicyBasedRouter</code> on top of the 9 cloud providers
        and adds persistent cost + rate-limit coordination. Existing <code>load_model</code>,
        <code> Agent</code>, and direct adapter paths are unchanged.
      </p>
      <ApiTable
        headers={['Area', 'Capability', 'API surface']}
        rows={[
          ['Router', 'PolicyBasedRouter runs an ordered list of RoutingPolicy instances over all ProviderRegistry pairs and returns an explainable RouterDecision', <code>PolicyBasedRouter</code>],
          ['Routing context', 'prompt_tokens_estimate, user_budget_usd, latency_budget_ms, required_capabilities (Capability enum)', <code>RoutingContext</code>],
          ['Capability flags', 'chat, tools, streaming, vision, grounding, thinking, json_schema', <code>effgen.models.capabilities.Capability</code>],
          ['Policies', 'FirstAvailablePolicy (key + capability check), CostBasedPolicy (cheapest within budget, raises NoCandidateWithinBudgetError), LatencyBasedPolicy (fastest within SLA using LatencyTracker p50)', <code>effgen.models.routing.*</code>],
          ['Failover', 'route_and_execute(context, fn) retries RateLimitExceeded, ProviderTransientError, ModelTimeoutError, BudgetExceededError; non-retriable errors re-raise immediately', <code>PolicyBasedRouter.route_and_execute</code>],
          ['Events', 'RouterEvent(from_provider, from_model, to_provider, to_model, reason, hop, exception) per failover hop', <code>router.subscribe(callback)</code>],
          ['Cost store', 'SQLiteCostStore writes every paid call to ~/.effgen/costs.sqlite; query_today / query_week / query_all', <code>effgen.models._cost_store.SQLiteCostStore</code>],
          ['Cost CLI', 'effgen cost today / week / by-provider / set-budget &lt;usd&gt; / clear-budget', <code>effgen cost ...</code>],
          ['Budget guard', '>= 80% emits UserWarning; >= 100% raises BudgetExceededError (period="daily" or "monthly"); router classifies it as retriable', <code>effgen.models.errors.BudgetExceededError</code>],
          ['Rate-limit store', 'SQLiteRateLimitStore (WAL mode, BEGIN IMMEDIATE) at ~/.effgen/rate_limits.sqlite for cross-process coordination', <code>effgen.models._rate_limit_store.SQLiteRateLimitStore</code>],
          ['Latency tracking', 'LatencyTracker records p50 total + p50 time-to-first-token per (provider, model) for LatencyBasedPolicy', <code>effgen.models.latency_tracker.LatencyTracker</code>],
          ['Retry policy', 'RetryPolicy(max_retries) handles per-provider retries before failover; classifies retriable vs terminal exceptions', <code>effgen.models.routing.retry.RetryPolicy</code>],
        ]}
      />
      <CodeBlock
        code={`# Policy chain: prefer fastest, fall back to cheapest
import effgen.models  # registers all 9 cloud adapters
from effgen import (
    PolicyBasedRouter, RoutingContext,
    CostBasedPolicy, LatencyBasedPolicy,
    load_model,
)
from effgen.models.capabilities import Capability

router = PolicyBasedRouter(
    policies=[LatencyBasedPolicy(), CostBasedPolicy()],
    failover_hops=3,
)
router.subscribe(lambda ev: print("failover:", ev.as_dict()))

context = RoutingContext(
    prompt_tokens_estimate=500,
    user_budget_usd=0.01,
    latency_budget_ms=3000,
    required_capabilities={Capability.chat, Capability.tools},
)

decision = router.route(context)
print(decision.chosen.provider, decision.chosen.model_id)
for pair, reason in decision.eliminated:
    print(" -", pair.provider, pair.model_id, "->", reason)

def call_model(pair):
    model = load_model(f"{pair.provider}:{pair.model_id}")
    return model.generate("Summarize the v0.2.4 release notes.")

answer = router.route_and_execute(context, call_model)`}
        language="python"
        filename="v024_router.py"
      />
      <CodeBlock
        code={`# Persistent cost tracking (v0.2.4+)
effgen cost today
effgen cost week
effgen cost by-provider

# Set a $1/day cap. 80% emits UserWarning; 100% raises BudgetExceededError,
# which route_and_execute treats as retriable failover to a free-tier provider.
effgen cost set-budget 1.0
effgen cost clear-budget`}
        language="bash"
        filename="terminal"
      />
      <InfoBox type="info" title="Provider capability matrix (v0.2.4)">
        <p>
          The router consults a per-provider capability matrix. Capabilities are
          declared in <code>effgen/models/registry.py</code>: cerebras / openai /
          groq / anthropic / gemini / together / fireworks all support
          <code> chat</code>, <code>tools</code>, and <code>streaming</code>;
          <code> openai</code>, <code>anthropic</code>, <code>gemini</code> add
          <code> vision</code>, and <code>openai</code>, <code>anthropic</code>,
          <code> gemini</code> add <code>thinking</code>; <code>gemini</code> is the
          only provider with <code>grounding</code>; <code>replicate</code> supports
          <code> chat</code> + <code>streaming</code> + <code>vision</code>, and
          <code> hf</code> supports <code>chat</code> + <code>streaming</code> only.
          <code> CostBasedPolicy</code> eliminates models marked
          <code> requires_endpoint</code> and uses per-model registry prices when
          available (falling back to per-provider defaults documented in
          <code> docs/models/router.md</code>).
        </p>
      </InfoBox>
      <InfoBox type="info" title="Upgrade note from v0.2.3">
        <p>
          No breaking changes. <code>RateLimitCoordinator</code> and
          <code> CostTracker</code> retain their in-memory defaults; construct them
          with <code>storage=SQLiteRateLimitStore()</code> or
          <code> storage=SQLiteCostStore()</code> to opt in. The
          <code> PolicyBasedRouter</code> is an entirely new layer — existing
          <code> load_model</code> and <code>Agent</code> code continues to work
          without any modification.
        </p>
      </InfoBox>

      <h2>v0.2.3 - Provider Ecosystem + Parity</h2>
      <p>
        v0.2.3 grows the cloud provider roster from 4 to 9 and introduces the
        <code> ProviderRegistry</code> as the shared lookup/auth/introspection layer.
      </p>
      <ApiTable
        headers={['Provider', 'Extra', 'Env key', 'Notes']}
        rows={[
          [<code>groq</code>, <code>effgen[groq]</code>, <code>GROQ_API_KEY</code>, '16 chat models, native tools, streaming, RPM/RPD/TPM/TPD windows, free-tier CostTracker'],
          [<code>together</code>, <code>effgen[together]</code>, <code>TOGETHER_API_KEY</code>, '163-model catalog, live refresh_models() drift detection, native tools, streaming, per-model pricing'],
          [<code>fireworks</code>, <code>effgen[fireworks]</code>, <code>FIREWORKS_API_KEY</code>, '80 chat models, 54 tool-capable, OpenAI-compatible interface, streaming, per-model pricing'],
          [<code>replicate</code>, <code>effgen[replicate]</code>, <code>REPLICATE_API_TOKEN</code>, '38 models, 34 streaming-capable, async run/poll, SSE streaming, prediction timeouts, compute_seconds metadata'],
          [<code>hf</code>, <code>effgen[hf]</code>, <code>HF_TOKEN</code>, '124-model HuggingFace Router catalog, custom endpoint URLs, ModelUnavailableError suggestions'],
        ]}
      />
      <CodeBlock
        code={`from effgen import load_model
from effgen.models.registry import list_providers, list_models, lookup
from effgen.models.auth import check_keys

print(list_providers())
print(list_models("groq")[:3])

model = load_model("groq:llama-3.3-70b-versatile")
provider, adapter_cls, info = lookup("groq:llama-3.3-70b-versatile")
keys = check_keys()`}
        language="python"
        filename="v023_registry.py"
      />
      <CodeBlock
        code={`effgen doctor
effgen doctor --provider groq
effgen doctor --json`}
        language="bash"
        filename="terminal"
      />
      <InfoBox type="info" title="Parity reports">
        <p>
          The v0.2.3 validation reports recorded 7/8 providers correct on the canonical
          calculator task, with Anthropic skipped because no key was available and Replicate
          marked as billing-credit xfail. Streaming parity passed 7/7 validated providers,
          and 9/9 providers raised <code>ModelAuthError</code> on invalid credentials.
        </p>
      </InfoBox>

      <h2>v0.2.2 - Gemini + Anthropic</h2>
      <p>
        v0.2.2 modernizes the Gemini and Anthropic adapters without requiring
        existing agent code to change.
      </p>
      <InfoBox type="info" title="v0.2.1 to v0.2.2 upgrade note">
        <p>
          All new fields default to backward-compatible values:
          <code> thinking_budget=None</code>, <code>include_thoughts=False</code>,
          <code>grounding=False</code>, and <code>thinking=None</code>. Existing v0.2.1
          Gemini and Anthropic calls keep the same behavior until these controls are enabled.
        </p>
      </InfoBox>
      <ApiTable
        headers={['Area', 'Capability', 'API surface']}
        rows={[
          ['Gemini models', 'Gemini 3.x, 2.5, 2.0, Gemma 3/4 registry; google-genai>=1.0.0 SDK added alongside legacy google-generativeai', <code>effgen.models.gemini_models</code>],
          ['Gemini thinking', 'Budgeted internal reasoning and optional thought traces', <code>GenerationConfig(thinking_budget=8192, include_thoughts=True)</code>],
          ['Gemini grounding', 'Google Search grounding and attribution metadata', <code>GenerationConfig(grounding=True)</code>],
          ['Gemini Files API', 'effgen.models.gemini_files.upload_file(path) returns FileRef; pass generate(..., files=[...]); 2 GiB pre-upload guard', <code>upload_file(path)</code>],
          ['Gemini native tools', 'Google Search, URL Context, Code Execution; parallel function calls surface in metadata["tool_calls"]', <code>GoogleSearchTool</code>],
          ['Anthropic models', 'Claude 4.7/4.x/3.x registry with feature flags', <code>effgen.models.anthropic_models</code>],
          ['Anthropic thinking', 'Extended thinking and redacted_thinking preservation through raw_content_blocks and build_assistant_message()', <code>GenerationConfig(thinking={"{...}"})</code>],
          ['Anthropic caching', 'cache_control helpers; AgentConfig.cache_system_prompt and cache_tools default to True so the last system block and last tool spec are cached automatically; max 4 cache breakpoints per request', <code>mark_cached()</code>],
          ['Anthropic streaming', 'generate_stream_full() returns typed StreamChunk objects for thinking, redacted-thinking, tool-use, and text deltas', <code>generate_stream_full()</code>],
          ['Anthropic tool specs', 'Experimental bash, text editor, computer-use adapter specs', <code>AnthropicBashTool</code>],
        ]}
      />
      <CodeBlock
        code={`from effgen import load_model
from effgen.models.base import GenerationConfig
from effgen.models.gemini_files import upload_file

model = load_model("gemini-2.5-pro", provider="gemini")
doc = upload_file("brief.pdf")  # returns FileRef
response = model.generate(
    "Explain why a parity matrix matters.",
    config=GenerationConfig(thinking_budget=4096, include_thoughts=True, grounding=True),
    files=[doc],
)
print(response.metadata.get("tool_calls"))`}
        language="python"
        filename="v022_gemini.py"
      />

      <h2>v0.2.1 - Cerebras + Modern OpenAI</h2>
      <p>
        v0.2.1 adds Cerebras as a first-class backend and updates OpenAI for newer
        reasoning models and server-side tools.
      </p>
      <ApiTable
        headers={['Area', 'Capability', 'API surface']}
        rows={[
          ['Cerebras backend', '4 registered models; 2 reliably free-tier callable; streaming, model-dependent native tool-calling, RateLimitCoordinator, RateLimitExceeded, and CostTracker', <code>load_model("llama3.1-8b", provider="cerebras")</code>],
          ['OpenAI registry', 'gpt-5, gpt-5.4-nano, gpt-4.1, gpt-4o family, o-series reasoning models', <code>effgen.models.openai_models</code>],
          ['Reasoning controls', 'none/minimal/low/medium/high/xhigh effort values and token budgets; v0.2.1 validates membership and passes the value through', <code>GenerationConfig(reasoning_effort="medium")</code>],
          ['Prompt caching', 'cached_input_tokens surfaced; generate_with_system_prompt() handles explicit cache-friendly placement, and stable_system_prompt keeps the system prompt anchored at position 0 across the Agent loop and the OpenAI native-tools path so OpenAI prefix cache stays warm', <code>generate_with_system_prompt()</code>],
          ['Structured outputs v2', 'OpenAIAdapter.generate_structured(), to_openai_schema(), $ref inlining, required defaults when absent, additionalProperties=false, ModelRefusalError', <code>OpenAIAdapter.generate_structured()</code>],
          ['OpenAI native tools', 'web_search, code_interpreter, file_search via Responses API', <code>OpenAIWebSearchTool</code>],
        ]}
      />
      <ApiTable
        headers={['OpenAI v0.2.1 registry', 'Family', 'Context / max output', 'Reasoning', 'Native tools', 'Prompt cache', 'Input / cached / output $ per 1M']}
        rows={[
          ['gpt-5.4, gpt-5.4-mini, gpt-5.4-nano', 'chat', '1,047,576 / 32,768', 'No', 'Yes', 'Yes', '2.50/0.25/15.00; 0.75/0.075/4.50; 0.20/0.02/1.25'],
          ['gpt-5.4-pro', 'chat', '1,047,576 / 32,768', 'No', 'Yes', 'No', '30.00/-/180.00'],
          ['gpt-5, gpt-5-mini, gpt-5-nano', 'chat', '1,047,576 / 32,768', 'No', 'Yes', 'Yes', '1.25/0.125/10.00; 0.25/0.025/2.00; 0.05/0.005/0.40'],
          ['gpt-5-pro', 'chat', '1,047,576 / 32,768', 'No', 'Yes', 'No', '15.00/-/120.00'],
          ['gpt-5.2, gpt-5.1', 'chat', '1,047,576 / 32,768', 'No', 'Yes', 'Yes', '1.75/0.175/14.00; 1.25/0.125/10.00'],
          ['gpt-4.1, gpt-4.1-mini, gpt-4.1-nano', 'chat', '1,047,576 / 32,768', 'No', 'Yes', 'Yes', '2.00/0.50/8.00; 0.40/0.10/1.60; 0.10/0.025/0.40'],
          ['gpt-4o, gpt-4o-mini, gpt-4o-2024-11-20', 'chat', '128,000 / 16,384', 'No', 'Yes', 'Yes', '2.50/1.25/10.00; 0.15/0.075/0.60; 2.50/1.25/10.00'],
          ['gpt-4o-2024-05-13', 'chat', '128,000 / 4,096', 'No', 'Yes', 'No', '5.00/-/15.00'],
          ['gpt-4-turbo, gpt-4-turbo-preview, gpt-4-turbo-2024-04-09', 'chat', '128,000 / 4,096', 'No', 'Yes', 'No', '10.00/-/30.00'],
          ['gpt-4, gpt-4-0613', 'chat', '8,192 / 4,096', 'No', 'Yes', 'No', '30.00/-/60.00'],
          ['gpt-4-32k', 'chat', '32,768 / 4,096', 'No', 'Yes', 'No', '60.00/-/120.00'],
          ['gpt-3.5-turbo, gpt-3.5-turbo-0125', 'chat', '16,385 / 4,096', 'No', 'Yes', 'No', '0.50/-/1.50'],
          ['gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613', 'chat', '16,385 / 4,096', 'No', 'Yes', 'No', '3.00/-/4.00'],
          ['o1', 'reasoning', '200,000 / 100,000', 'Yes', 'Yes', 'Yes', '15.00/7.50/60.00'],
          ['o1-pro', 'reasoning', '200,000 / 100,000', 'Yes', 'Yes', 'No', '150.00/-/600.00'],
          ['o1-mini', 'reasoning', '128,000 / 65,536', 'Yes', 'Yes', 'Yes', '1.10/0.55/4.40'],
          ['o1-preview', 'reasoning', '128,000 / 32,768', 'Yes', 'No', 'Yes', '15.00/7.50/60.00'],
          ['o3, o3-mini', 'reasoning', '200,000 / 100,000', 'Yes', 'Yes', 'Yes', '2.00/0.50/8.00; 1.10/0.55/4.40'],
          ['o3-pro', 'reasoning', '200,000 / 100,000', 'Yes', 'Yes', 'No', '20.00/-/80.00'],
          ['o4-mini', 'reasoning', '200,000 / 100,000', 'Yes', 'Yes', 'Yes', '1.10/0.275/4.40'],
        ]}
      />
      <ApiTable
        headers={['Cerebras model', 'Free tier', 'Native tools', 'Context / output', 'RPM/RPH/RPD', 'TPM/TPH/TPD', 'Deprecation / notes']}
        rows={[
          [<code>gpt-oss-120b</code>, 'Restricted', 'Yes (paid tier)', '65,536 / 32,768', '30/900/14,400', '64k/1M/1M', 'Registered with native tool support but commonly 404s on the free tier; tools are reachable only on paid access'],
          [<code>llama3.1-8b</code>, 'Yes', 'Yes', '8,192 / 8,192', '30/900/14,400', '60k/1M/1M', 'Deprecated May 27, 2026'],
          [<code>qwen-3-235b-a22b-instruct-2507</code>, 'Yes', 'Yes', '65,536 / 32,768', '30/900/14,400', '60k/1M/1M', 'Deprecated May 27, 2026'],
          [<code>zai-glm-4.7</code>, 'Restricted', 'No', '65,536 / 40,960', '10/100/100', '60k/1M/1M', 'Registered; stricter request window'],
        ]}
      />
      <CodeBlock
        code={`from effgen import load_model
from effgen.models.base import GenerationConfig
from effgen.models.cerebras_models import available_models, deprecated_models, free_tier_models, model_info
from effgen.models._cost import CostTracker

chat_model = load_model("gpt-5.4-nano", provider="openai")
chat_response = chat_model.generate_with_system_prompt(
    prompt="Explain automatic prompt caching briefly.",
    system_prompt="You are a concise OpenAI platform explainer.",
)
print(chat_response.metadata["cached_input_tokens"])

reasoning_model = load_model("o4-mini", provider="openai")
reasoning_response = reasoning_model.generate(
    "Solve the problem and explain your reasoning briefly.",
    config=GenerationConfig(reasoning_effort="medium", max_reasoning_tokens=4096),
)
print(reasoning_response.text)
print(model_info("llama3.1-8b"))
print(free_tier_models())
print(deprecated_models())
print(CostTracker.get().summary())`}
        language="python"
        filename="v021_openai.py"
      />

      <h2>Upgrade Path</h2>
      <p>
        Every release from v0.2.1 through v0.2.10 is additive. Existing <code>Agent</code>,
        <code> AgentConfig</code>, <code>load_model</code>, local tools, RAG, guardrails,
        workflows, and API server code from v0.2.0 continue to work. The v0.2.4{' '}
        <code>PolicyBasedRouter</code>, v0.2.7 Prompt Library, v0.2.8 multimodal schema, v0.2.9
        observability / reliability layer, and v0.2.10 sandbox / auth / deploy / DX features are
        all opt-in. The only behavioural change to watch for: as of v0.2.10 the{' '}
        <code>CodeExecutor</code> runs inside a sandbox by default (Docker if available, else an
        unprivileged-namespace subprocess) — set{' '}
        <code>EFFGEN_SANDBOX_BACKEND=off</code> to restore host execution (with a loud warning).
      </p>
      <CodeBlock
        code={`pip install --upgrade effgen

# Optional provider extras:
pip install "effgen[cerebras]"
pip install "effgen[groq]"
pip install "effgen[together]"
pip install "effgen[fireworks]"
pip install "effgen[replicate]"
pip install "effgen[hf]"`}
        language="bash"
        filename="terminal"
      />

      <QuickLinks
        links={[
          { icon: 'M', title: 'Multimodal', description: 'v0.2.8 image / audio / video input', path: '/multimodal' },
          { icon: 'O', title: 'Observability', description: 'v0.2.9 logging, metrics, SLOs, tracing, alerting', path: '/observability' },
          { icon: 'R', title: 'Reliability', description: 'v0.2.9 timeouts, retries, breakers, bulkheads', path: '/reliability' },
          { icon: 'S', title: 'Security', description: 'v0.2.10 sandbox, OIDC auth, RBAC, audit, supply chain', path: '/security' },
          { icon: 'D', title: 'Deployment', description: 'v0.2.10 Docker, Helm, Lambda, Cloudflare', path: '/deployment' },
          { icon: 'X', title: 'Developer Experience', description: 'v0.2.10 VSCode, Jupyter, dashboard', path: '/dx' },
        ]}
      />
    </DocPage>
  );
}
