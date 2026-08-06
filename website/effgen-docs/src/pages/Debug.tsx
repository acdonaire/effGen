import React from 'react';
import { Link } from 'react-router-dom';
import { Activity } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Debug() {
  return (
    <DocPage
      title="Debugging &amp; Observability"
      subtitle="OpenTelemetry tracing, Prometheus metrics, structured logging, and a step-through DebugAgent for inspecting ReAct loops iteration-by-iteration."
      icon={<Activity size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Advanced', path: '/multi-agent' },
        { label: 'Debugging' },
      ]}
    >
      <h2>DebugAgent — Step-Through Inspection</h2>
      <p>
        Wrap any agent in <code>DebugAgent</code> (or just pass <code>debug=True</code> to
        <code> Agent.run()</code>) to capture a rich per-iteration trace:
      </p>

      <CodeBlock
        code={`from effgen.debug import DebugAgent

agent = DebugAgent(config)
result = agent.run("What is 24344 * 334?")

trace = result.metadata["debug_trace"]     # DebugTrace
for it in trace.iterations:                # list[DebugIteration]
    print(it.iteration)
    print("  raw_prompt:  ", it.raw_prompt[:120])
    print("  raw_response:", it.raw_response[:120])
    print("  thought:     ", it.thought)
    print("  action:      ", it.action, it.action_input)
    print("  observation: ", it.observation)
    print("  tokens_used: ", it.tokens_used)
    print("  latency (s): ", it.latency)

print(trace.total_tokens, trace.total_latency)`}
        language="python"
        filename="debug_agent.py"
      />

      <h3>DebugIteration fields</h3>
      <ApiTable
        headers={['Field', 'Type', 'Description']}
        rows={[
          [<code>iteration</code>, 'int', 'Index in the ReAct loop (starting at 1)'],
          [<code>raw_prompt</code>, 'str', 'Exact prompt sent to the model'],
          [<code>raw_response</code>, 'str', 'Exact model output (pre-parsing)'],
          [<code>thought</code>, 'str', 'Extracted "Thought:" text'],
          [<code>action</code>, 'str', 'Tool name chosen'],
          [<code>action_input</code>, 'str | None', 'Raw tool argument string (parsed by the tool)'],
          [<code>observation</code>, 'str | None', 'Tool output fed back to the model'],
          [<code>final_answer</code>, 'str | None', 'Final answer if this iteration completed the run'],
          [<code>tokens_used</code>, 'int', 'Tokens consumed this iteration'],
          [<code>latency</code>, 'float', 'Wall-clock for this iteration (seconds)'],
          [<code>scratchpad_snapshot</code>, 'str', 'Scratchpad state captured for step-through debugging'],
          [<code>memory_snapshot</code>, 'list[dict]', 'Memory state snapshot when available'],
          [<code>metadata</code>, 'dict', 'Additional debug metadata'],
        ]}
      />

      <h3>Interactive TUI</h3>
      <CodeBlock
        code={`effgen debug "What is sqrt(144)?" --model Qwen/Qwen2.5-3B-Instruct`}
        language="bash"
        filename="terminal"
      />
      <p>
        Steps through iterations one at a time, showing the raw prompt, thought, action, and
        observation. Press <code>n</code> to advance, <code>q</code> to quit.
      </p>

      <h2>OpenTelemetry Tracing</h2>
      <p>
        effGen emits OpenTelemetry spans for every agent run, iteration, tool call, and model
        generation. Cross-agent propagation flows through <code>LogRunContext</code>.
      </p>

      <CodeBlock
        code={`# Simply set the standard OTel environment variables:
export OTEL_SERVICE_NAME=my-effgen-app
export OTEL_EXPORTER_TYPE=otlp                # "otlp" | "jaeger" | "zipkin" | "console"
export OTEL_EXPORTER_ENDPOINT=http://otel-collector:4318

# Or configure programmatically:
from effgen.utils.tracing import setup_tracing

setup_tracing(
    service_name="my-effgen-app",
    exporter_type="otlp",            # "otlp" | "jaeger" | "zipkin" | "console"
    endpoint="http://otel-collector:4318",
)`}
        language="python"
        filename="otel_setup.py"
      />

      <InfoBox type="info" title="No-op fallback">
        <p>
          If <code>opentelemetry</code> is not installed, effGen silently no-ops all tracing
          calls. You can deploy without OTel and add it later without code changes.
        </p>
      </InfoBox>

      <h2>Prometheus Metrics</h2>
      <p>
        When <code>prometheus_client</code> is installed, <code>/metrics</code> exposes:
      </p>
      <FeatureList
        features={[
          { icon: '⏱️', title: 'effgen_response_latency_seconds', description: 'Histogram for agent response latency.' },
          { icon: '🔢', title: 'effgen_token_usage', description: 'Histogram of token usage per request.' },
          { icon: '🛠️', title: 'effgen_tool_execution_seconds', description: 'Histogram for tool execution duration.' },
          { icon: '💾', title: 'effgen_gpu_memory_used_bytes', description: 'Gauge for live GPU memory usage.' },
          { icon: '❌', title: 'effgen_errors_total', description: 'Counter for errors.' },
        ]}
      />

      <h3>Grafana Dashboard</h3>
      <p>
        A 12-panel Grafana dashboard (latency p50/p95/p99, throughput, error rate, tool
        breakdown, GPU memory, queue depth) ships in the effGen repo under
        <code> assets/grafana/dashboard.json</code>. Import it directly into Grafana.
      </p>

      <h2>Structured Logging</h2>
      <p>
        <code>EffGenJSONFormatter</code> emits one JSON object per log line with correlation
        fields so log aggregators (Loki, Elasticsearch, Datadog) can group a single agent run
        together.
      </p>
      <CodeBlock
        code={`from effgen.utils.structured_logging import (
    get_structured_logger, LogRunContext, generate_run_id,
)

log = get_structured_logger(__name__)
run_id = generate_run_id()

with LogRunContext(run_id=run_id, agent_name="researcher", session_id="user-123"):
    log.info("starting_research", query="SLM agents", num_sources=5)
    # ... correlated automatically through all downstream logs ...`}
        language="python"
        filename="structured_logging.py"
      />

      <h3>Fields always attached</h3>
      <ApiTable
        headers={['Field', 'Source']}
        rows={[
          [<code>run_id</code>, 'Generated per agent.run()'],
          [<code>workflow_id</code>, 'WorkflowDAG.run() / run_async()'],
          [<code>agent_name</code>, 'AgentConfig.name'],
          [<code>session_id</code>, 'Agent(..., session_id="...") if using Sessions'],
          [<code>iteration</code>, 'Current ReAct loop index'],
          [<code>tool_name</code>, 'During tool execution spans'],
        ]}
      />

      <h2>Caching</h2>
      <p>
        <code>effgen.cache</code> offers two caches relevant to debugging slow runs:
      </p>

      <FeatureList
        features={[
          { icon: '📝', title: 'PromptCache', description: 'LRU + TTL over SHA-256 prompt fingerprints. Thread-safe, tracks hit / miss stats.' },
          { icon: '🛠️', title: 'ResultCache', description: 'LRU + per-tool TTL for tool results. Optional semantic similarity via embed_fn + cosine.' },
        ]}
      />

      <CodeBlock
        code={`from effgen.cache import PromptCache, ResultCache

pc = PromptCache(max_size=1000, default_ttl=3600)
rc = ResultCache(max_size=500, default_ttl=600)
rc.set_tool_ttl("web_search", 60)        # per-tool TTL override

# Hit / miss stats
print(pc.stats())      # {"hits": 42, "misses": 7, "hit_rate": 0.857, ...}`}
        language="python"
        filename="caching.py"
      />

      <h2>TokenBudget</h2>
      <p>
        <code>TokenBudget</code> allocates context window by section (system 20% / tools 30% /
        history 40% / response 10%) and truncates with <code>smart_truncate()</code>
        (head + tail) or <code>fit_to_budget()</code>.
      </p>

      <h2>See Also</h2>
      <p>
        <Link to="/agents">Agents</Link> · <Link to="/evaluation">Evaluation</Link> ·
        {' '}<Link to="/api-server">API Server</Link>
      </p>
    </DocPage>
  );
}
