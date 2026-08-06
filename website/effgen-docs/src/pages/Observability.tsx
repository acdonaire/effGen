import React from 'react';
import { Activity } from 'lucide-react';
import DocPage, { ApiTable, InfoBox, QuickLinks } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Observability() {
  return (
    <DocPage
      title="Observability"
      subtitle="effGen v0.2.9 turns the framework into something you can operate in production: structured JSON logging with secret redaction, OpenTelemetry tracing with configurable samplers, Prometheus histograms, SLO tracking with burn-rate endpoints, a load-testing harness, and an Alertmanager-compatible rule pack. Every telemetry path is async / non-blocking — a failed export never fails inference."
      icon={<Activity size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Observability' },
      ]}
    >
      <InfoBox type="success" title="New in v0.2.9">
        <p>
          The <strong>Observability &amp; Reliability</strong> release adds the{' '}
          <code>effgen.observability</code> package: structured logging, secret redaction,
          Prometheus metrics, SLO tracking, OTel tracing, an alerting rule pack, and the{' '}
          <code>effgen loadtest</code> CLI. Reliability primitives (timeouts, retries, circuit
          breakers, bulkheads, chaos + fuzz harnesses) live alongside it — see the{' '}
          <a href="/docs/reliability">Reliability</a> page. No breaking API changes.
        </p>
      </InfoBox>

      <InfoBox type="info" title="v0.3.0 — /metrics now requires auth by default">
        <p>
          As of v0.3.0 the API server fails closed: <code>GET /metrics</code> and the{' '}
          <code>/dashboard</code> data endpoints <strong>require auth by default</strong>. Open them
          explicitly with <code>EFFGEN_PUBLIC_METRICS=1</code> /{' '}
          <code>EFFGEN_PUBLIC_DASHBOARD=1</code> (or run in dev mode) when scraping from an
          unauthenticated Prometheus. See <a href="/docs/security">Security</a> and{' '}
          <a href="/docs/api-server">API Server</a>.
        </p>
      </InfoBox>

      <h2>Structured Logging</h2>
      <p>
        <code>StructuredFormatter</code> emits JSON lines{' '}
        <code>{'{ts, level, module, event, attributes, trace_id, span_id}'}</code> with OTel
        span-context integration. The agent loop, model adapters, router, and tool call sites
        were all migrated from ad-hoc <code>print</code>/<code>logger.info</code> to the
        structured logger.
      </p>
      <CodeBlock
        code={`from effgen.observability import get_logger, configure_logging

configure_logging(level="INFO")  # call once at startup
log = get_logger(__name__)
log.event("model.call.started", provider="cerebras", model="llama3.1-8b", cached_tokens=0)
# → {"ts": "...", "level": "INFO", "event": "model.call.started", "attributes": {...}, ...}`}
        language="python"
        filename="logging.py"
      />

      <h2>Secret Redaction</h2>
      <p>
        The <code>Redactor</code> matches and strips secrets <strong>at the log encoder</strong>,
        so every path is covered — an API key passed in a log attribute never reaches the log
        file. Built-in patterns cover OpenAI (<code>sk-</code>), Anthropic (
        <code>sk-ant-</code>), Cerebras (<code>csk-</code>), Google (<code>AIza</code>),
        HuggingFace (<code>hf_</code>), Groq (<code>gsk_</code>), Bearer tokens, and Slack /
        Discord webhook URLs. It is user-extensible via{' '}
        <code>Redactor.add_pattern(name, regex)</code>.
      </p>
      <InfoBox type="info" title="Redaction is code, not configuration">
        <p>
          Because redaction runs at the encoder, you can't accidentally disable it by forgetting
          a config flag. Matches are replaced with markers like{' '}
          <code>&lt;REDACTED:openai_key&gt;</code>.
        </p>
      </InfoBox>

      <h2>Metrics</h2>
      <p>
        Prometheus histograms and counters auto-record on agent / model / tool calls and are
        exposed by the FastAPI server at <code>GET /metrics</code>.
      </p>
      <ApiTable
        headers={['Metric', 'Type', 'Labels']}
        rows={[
          [<code>effgen_model_call_latency_seconds</code>, 'Histogram (50 ms–60 s buckets)', 'provider, model, outcome'],
          [<code>effgen_tool_call_latency_seconds</code>, 'Histogram', 'tool, outcome'],
          [<code>effgen_agent_iteration_latency_seconds</code>, 'Histogram', 'preset'],
          [<code>effgen_tokens_total</code>, 'Counter', 'provider, model, kind ∈ {input, output, cached}'],
        ]}
      />
      <CodeBlock
        code={`from effgen.observability import record_model_call, export_metrics

# Histograms auto-record on agent/model/tool calls; you can also record directly:
record_model_call(provider="cerebras", model="llama3.1-8b", outcome="ok", latency=0.42)
print(export_metrics())  # Prometheus text format`}
        language="python"
        filename="metrics.py"
      />

      <h2>SLO Tracking</h2>
      <p>
        <code>SLO(name, target_pct, window_seconds, query)</code> and{' '}
        <code>SLOTracker</code> implement rolling-window error-budget tracking.{' '}
        <code>burn_rate(name)</code> returns the ratio against target (1.0 = on budget,{' '}
        &gt; 1.0 = burning the budget). Exposed at <code>GET /slo</code>.
      </p>
      <CodeBlock
        code={`from effgen.observability.slo import SLOTracker, SLO

tracker = SLOTracker()
tracker.register(SLO("model_call_success", target_pct=99.0, window_seconds=3600))
tracker.record("model_call_success", ok=True)
print(tracker.burn_rate("model_call_success"))  # 1.0 = on budget, >1.0 = alert`}
        language="python"
        filename="slo.py"
      />

      <h2>Tracing</h2>
      <p>
        OpenTelemetry tracing ships with explicit samplers — there is no implicit{' '}
        <code>head=1.0</code> in production. Every span attribute name lives in a single
        source of truth, <code>effgen/observability/spans.py</code>.
      </p>
      <ApiTable
        headers={['Sampler', 'Behaviour']}
        rows={[
          [<code>AlwaysOnSampler</code>, 'Record every trace.'],
          [<code>AlwaysOffSampler</code>, 'Record nothing.'],
          [<code>TraceIdRatioSampler(p)</code>, 'Sample a fixed ratio p of traces.'],
          [<code>RateLimitedSampler(per_second)</code>, 'Cap recorded traces per second.'],
          [<code>ParentBasedSampler(child)</code>, 'Honour the parent span’s decision, else delegate to the child sampler.'],
        ]}
      />
      <CodeBlock
        code={`from effgen.observability import (
    setup_tracing, ParentBasedSampler, TraceIdRatioSampler, start_agent_run,
)

setup_tracing(sampler=ParentBasedSampler(TraceIdRatioSampler(0.1)))  # sample 10%
with start_agent_run(preset="my_agent", task="hello"):
    ...`}
        language="python"
        filename="tracing.py"
      />
      <p>
        The canonical span-attribute spec covers <code>effgen.agent.*</code>,{' '}
        <code>effgen.model.*</code>, <code>effgen.tool.*</code>,{' '}
        <code>effgen.router.*</code>, and <code>effgen.retry.*</code> (plus{' '}
        <code>effgen.model.parts_count</code> for multimodal requests).
      </p>

      <h2>Load Testing</h2>
      <p>
        The <code>effgen loadtest</code> CLI (backed by{' '}
        <code>effgen/tools/loadgen.py</code>) drives concurrent traffic and reports
        throughput plus p50/p95/p99 latency and error rate as JSON.
      </p>
      <CodeBlock
        code={`# Run against the local mock model (default), concurrency 10 for 30s
effgen loadtest --concurrency 10 --duration 30 --scenario fixed

# Switch to live inference
effgen loadtest --concurrency 5 --duration 60 --provider cerebras --scenario multi_tool

# Mock smoke result: ~69k req, 0% error, p95 ≈ 4.3 ms`}
        language="bash"
        filename="terminal"
      />

      <h2>Alerting</h2>
      <p>
        Six Alertmanager-compatible rules ship in{' '}
        <code>docs/observability/alert_rules.yaml</code>, and{' '}
        <code>AlertWebhook(url).fire(alert)</code> posts to Slack / Discord (with a generic
        httpx fallback). The webhook URL is redacted in logs, and <code>fire</code> never
        throws.
      </p>
      <ApiTable
        headers={['Rule', 'Fires when']}
        rows={[
          [<code>HighErrorRate</code>, 'Error rate &gt; 5% for 10 min'],
          [<code>HighP95Latency</code>, 'p95 latency &gt; 10 s for 5 min'],
          [<code>CostBurnHigh</code>, 'Spend &gt; $10/day'],
          [<code>SLOFastBurn</code>, 'Error budget burning &gt; 14.4×'],
          [<code>SLOSlowBurn</code>, 'Sustained slow error-budget burn'],
          [<code>CircuitBreakerOpen</code>, 'A provider circuit breaker is OPEN'],
        ]}
      />

      <h2>Design Principles</h2>
      <ApiTable
        headers={['Principle', 'What it means']}
        rows={[
          ['Never block on observability', 'All telemetry is async / non-blocking. A failed export is not a failed inference.'],
          ['Redaction is code', 'Secret patterns are stripped at the encoder, so every path is covered.'],
          ['Explicit sampling', 'The tracing sampler is declared explicitly — no implicit head=1.0 in production.'],
          ['Bound every wait', 'Timeouts are declared in config and propagated into adapter httpx clients and tool executions — no default timeout=None.'],
        ]}
      />

      <QuickLinks
        links={[
          { icon: 'R', title: 'Reliability', description: 'Timeouts, retries, circuit breakers, bulkheads, chaos & fuzz', path: '/reliability' },
          { icon: 'S', title: 'API Server', description: '/metrics, /slo, and the auth boundary', path: '/api-server' },
          { icon: 'D', title: 'Debugging', description: 'Tracing-driven debugging workflows', path: '/debug' },
          { icon: 'N', title: 'Release Notes', description: 'v0.2.9 observability & reliability release', path: '/releases' },
        ]}
      />
    </DocPage>
  );
}
