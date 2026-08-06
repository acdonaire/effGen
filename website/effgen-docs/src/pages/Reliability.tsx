import React from 'react';
import { ShieldCheck } from 'lucide-react';
import DocPage, { ApiTable, InfoBox, QuickLinks } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Reliability() {
  return (
    <DocPage
      title="Reliability"
      subtitle="effGen v0.2.9 ships four production-grade reliability primitives under effgen.reliability — explicit timeouts, jittered retries, per-provider circuit breakers, and bulkheads — plus a deterministic chaos harness and a Hypothesis fuzz suite to test them. Every I/O boundary carries an explicit timeout; no default timeout=None."
      icon={<ShieldCheck size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Reliability' },
      ]}
    >
      <InfoBox type="success" title="New in v0.2.9">
        <p>
          The reliability layer wires timeouts, retries, circuit breakers, and bulkheads into
          the model adapters and <code>ProviderRegistry</code>. A deterministic chaos harness and
          a Hypothesis-based fuzz suite validate the stack. All additive — see{' '}
          <a href="/docs/observability">Observability</a> for the telemetry side of the release.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.0 — fail-closed errors &amp; smarter retries">
        <p>
          v0.3.0 makes failures loud and typed. <code>Agent.run()</code> never returns{' '}
          <code>success=True</code> with empty output; on failure it returns <code>success=False</code>{' '}
          with a coarse <code>metadata["reason"]</code> stage label and a typed redacted{' '}
          <code>metadata["error"]</code> dict. A new <code>classify_provider_error()</code>{' '}
          populates <code>metadata["error"]["category"]</code> with a stable taxonomy —{' '}
          <code>auth</code>, <code>not_found</code>, <code>rate_limited</code>,{' '}
          <code>transient</code>, <code>timeout</code>, <code>fatal</code> — and retries now fire{' '}
          <strong>only</strong> on transient / rate-limited errors. Auth and not-found errors fast-stop with one clear
          message instead of a retry storm, and a 404 model id suggests the nearest live
          alternative. Set <code>AgentConfig(raise_on_error=True)</code> to opt into exceptions.
          Internally there is now a single canonical <code>CircuitBreaker</code> implementation.
          See <a href="/docs/agents">Agents → Error Handling</a>.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.1 — honest failures across teams, workflows &amp; reasoning models">
        <p>
          v0.3.1 extends fail-closed behavior beyond the single agent. <strong>Collaborative teams
          fail closed</strong> — a failed collaborator sets the team <code>success=False</code> with
          a discoverable per-agent error instead of always reporting success. <strong>Hierarchical
          teams route by name</strong> to the worker the manager named, and every subtask runs.{' '}
          <strong>Workflow DAGs don't run downstream of a failed or skipped node</strong>, so an
          internal error is never rewritten into a customer-facing answer. <strong>Reasoning
          models</strong> (the <code>gpt-5</code> family, <code>o</code>-series) no longer return
          empty, billed results on token-heavy tasks — an empty{' '}
          <code>finish_reason="length"</code> is treated as truncation, the budget grows once and
          retries, and a starved budget is never retried three times. Local Transformers batch is
          now thread-safe (no more <code>"Already borrowed"</code> under default concurrency), and
          sync <code>Agent.run()</code> no longer hangs forever on an MCP stdio tool — it raises a
          clear <code>TimeoutError</code> pointing at <code>run_async()</code>.
        </p>
      </InfoBox>

      <h2>Primitives</h2>
      <ApiTable
        headers={['Primitive', 'Module', 'Purpose']}
        rows={[
          ['Timeouts', <code>timeouts.py</code>, 'Explicit wall-clock limits on every I/O call'],
          ['Retries', <code>retry.py</code>, 'Jittered exponential backoff with span events'],
          ['Circuit Breaker', <code>circuit.py</code>, 'Fast-fail / recovery state machine per provider'],
          ['Bulkhead', <code>bulkhead.py</code>, 'Concurrency cap + bounded queue per provider'],
        ]}
      />

      <h2>Configuration</h2>
      <CodeBlock
        code={`from effgen.reliability import ReliabilityConfig, TimeoutConfig

cfg = ReliabilityConfig(
    timeouts=TimeoutConfig(
        model_call=60,   # seconds
        tool_call=30,
        http=20,
        agent_loop=600,
        queue=5,
    )
)
print(cfg.to_dict())   # ReliabilityConfig.defaults() returns the defaults above`}
        language="python"
        filename="config.py"
      />
      <ApiTable
        headers={['Timeout', 'Default', 'Applies to']}
        rows={[
          [<code>model_call</code>, '60 s', 'Model provider calls'],
          [<code>tool_call</code>, '30 s', 'Tool executions'],
          [<code>http</code>, '20 s', 'Raw HTTP requests'],
        ]}
      />

      <h2>Timeouts</h2>
      <p>
        Every I/O boundary in effGen must carry an explicit timeout.{' '}
        <code>timeout=None</code> is detected at test time by{' '}
        <code>audit_no_none_timeouts()</code> — any adapter missing an explicit timeout fails
        the audit test.
      </p>
      <CodeBlock
        code={`from effgen.reliability import with_timeout, apply_timeout, EffGenTimeoutError
from effgen.reliability.timeouts import async_timeout, make_httpx_timeout

# Sync context manager
with with_timeout(30.0, "tool_call"):
    result = tool.run(...)

# Async context manager
async with async_timeout(60.0, "model_call"):
    result = await adapter.generate(...)

# Wrap a callable (sync or async)
wrapped = apply_timeout(my_fn, seconds=30.0, operation="my_op")

# httpx timeout helper → httpx.Timeout(timeout=20.0, connect=5.0)
timeout = make_httpx_timeout(http=20.0, connect=5.0)`}
        language="python"
        filename="timeouts.py"
      />
      <p>
        <code>EffGenTimeoutError</code> subclasses the built-in <code>TimeoutError</code> and
        carries <code>.operation</code> and <code>.limit</code> attributes.
      </p>

      <h2>Retries</h2>
      <p>
        <code>Retry(max_attempts, base_delay, max_delay, jitter, retryable)</code> with the{' '}
        <code>@retryable(...)</code> decorator applies jittered exponential backoff. The default
        policy retries transient network errors, 5xx responses, and 429s (honouring the{' '}
        <code>Retry-After</code> header), and emits an <code>effgen.retry.attempt</code> OTel
        span event per attempt.
      </p>
      <CodeBlock
        code={`from effgen.reliability import Retry, retryable

@retryable(Retry(max_attempts=3, base_delay=0.5, jitter=True))
def call_model(prompt):
    ...`}
        language="python"
        filename="retry.py"
      />

      <h2>Circuit Breaker</h2>
      <p>
        <code>CircuitBreaker(name, failure_threshold, recovery_timeout, half_open_probes)</code>{' '}
        is a three-state (CLOSED → OPEN → HALF_OPEN) per-provider breaker. The{' '}
        <code>CircuitBreakerRegistry</code> keeps one breaker per provider, wired into{' '}
        <code>ProviderRegistry</code>.
      </p>
      <CodeBlock
        code={`from effgen.reliability import CircuitBreaker

breaker = CircuitBreaker("cerebras", failure_threshold=5, recovery_timeout=30)
if breaker.is_call_permitted():
    try:
        result = call_model("hello")
        breaker.on_success()
    except Exception as exc:
        breaker.on_failure(exc)
        raise`}
        language="python"
        filename="circuit.py"
      />

      <h2>Bulkhead</h2>
      <p>
        <code>Bulkhead(name, max_concurrency, queue_size, queue_timeout)</code> is a
        semaphore-based concurrency limiter with a bounded queue (sync + async variants). The{' '}
        <code>BulkheadRegistry</code> keeps one bulkhead per provider so a misbehaving provider
        can't starve the others.
      </p>
      <CodeBlock
        code={`from effgen.reliability import Bulkhead

bulkhead = Bulkhead("cerebras", max_concurrency=10, queue_size=50)
with bulkhead.acquire():
    call_model("hello")`}
        language="python"
        filename="bulkhead.py"
      />

      <h2>Chaos Harness</h2>
      <p>
        <code>Chaos(seed)</code> injects deterministic, reproducible faults so reliability
        behaviour can be tested without flaky live failures. Attach it to the registry with{' '}
        <code>registry.with_chaos(Chaos(...))</code>.
      </p>
      <ApiTable
        headers={['Fault type', 'Effect']}
        rows={[
          [<code>NetworkTimeout</code>, 'Simulates a hung connection'],
          [<code>Http5xx</code>, 'Returns a 5xx server error'],
          [<code>Http429(retry_after)</code>, 'Returns 429 with a Retry-After header'],
          [<code>SlowResponse(ms)</code>, 'Delays the response by ms'],
          [<code>PartialResponse</code>, 'Truncates the response body'],
          [<code>MalformedJSON</code>, 'Returns invalid JSON'],
        ]}
      />
      <p>
        Four canonical scenarios are validated end-to-end: <strong>A</strong> (5xx fallback),{' '}
        <strong>B</strong> (429 + Retry-After), <strong>C</strong> (SlowResponse timeout), and{' '}
        <strong>D</strong> (AllProvidersFailed — no silent empty string). The chaos suite runs
        4 scenarios × 10 seeds (273 tests), all deterministic.
      </p>

      <h2>Fuzz Harness</h2>
      <p>
        A Hypothesis-based fuzz suite exercises the framework's input surface for crashes and
        secret leaks:
      </p>
      <ApiTable
        headers={['Harness', 'Coverage']}
        rows={[
          [<code>test_tool_fuzz.py</code>, 'All 66 BaseTool subclasses × 500 examples — no unhandled exceptions, no secret leaks'],
          [<code>test_message_fuzz.py</code>, 'Random ContentPart sequences — no unhandled exceptions'],
          [<code>test_router_fuzz.py</code>, 'Random availability + capabilities → a valid decision or NoEligibleProvider'],
        ]}
      />

      <QuickLinks
        links={[
          { icon: 'O', title: 'Observability', description: 'Logging, metrics, SLOs, tracing, alerting', path: '/observability' },
          { icon: 'M', title: 'Models', description: 'Adapters and the PolicyBasedRouter failover layer', path: '/models' },
          { icon: 'P', title: 'Providers', description: 'ProviderRegistry, where breakers and bulkheads attach', path: '/providers' },
          { icon: 'R', title: 'Release Notes', description: 'v0.2.9 observability & reliability release', path: '/releases' },
        ]}
      />
    </DocPage>
  );
}
