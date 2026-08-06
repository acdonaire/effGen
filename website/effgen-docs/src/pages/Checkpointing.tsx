import React from 'react';
import { Link } from 'react-router-dom';
import { Database } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Checkpointing() {
  return (
    <DocPage
      title="Checkpointing &amp; Sessions"
      subtitle="Persist agent state — scratchpad, memory, tool states, iterations — to JSON or SQLite. Resume multi-hour runs across processes. Session management and background tasks included."
      icon={<Database size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Advanced', path: '/multi-agent' },
        { label: 'Checkpointing' },
      ]}
    >
      <h2>Why?</h2>
      <p>
        Long-running agents (research, batch pipelines, 100-iteration ReAct loops) need two
        things: the ability to <strong>resume</strong> if a process dies, and
        <strong> persistent conversations</strong> across user sessions. v0.2.0 introduces three
        layers for this:
      </p>
      <ul>
        <li><strong><code>CheckpointManager</code></strong> — per-run state snapshots (JSON or SQLite).</li>
        <li><strong><code>Session</code> / <code>SessionManager</code></strong> — per-user conversational state.</li>
        <li><strong><code>BackgroundTaskRunner</code></strong> — priority queue with pause / resume / cancel.</li>
      </ul>

      <InfoBox type="success" title="v0.3.0 — durable sessions &amp; resume">
        <p>
          As of v0.3.0, sessions, checkpoints, and <code>effgen resume</code> are durable with{' '}
          <strong>clear errors on corrupt or absent files</strong> instead of a silent failure or a
          cryptic traceback. State remains JSON-only (no <code>pickle</code>), consistent with the
          broader v0.3.0 hardening — see <a href="/docs/security">Security</a>.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.1 — resume a session interactively">
        <p>
          <code>effgen chat --session-id &lt;id&gt;</code> (or <code>--resume &lt;id&gt;</code>) now
          continues a persisted conversation — the same store that{' '}
          <code>effgen run --session-id</code> and <code>effgen sessions</code> already use — so an
          operator can pick a customer&apos;s conversation back up interactively. Streamed turns are
          saved as they happen, and the session-vs-checkpoint help text is aligned so the two
          persistence mechanisms no longer read as unrelated features.
        </p>
      </InfoBox>

      <h2>Checkpoints</h2>
      <p>
        A <code>Checkpoint</code> captures <code>scratchpad</code>, <code>iteration</code>,
        <code> partial_output</code>, <code>tool_calls</code>, <code>tokens_used</code>,
        <code> memory</code>, and <code>tool_states</code>. Checkpoints are JSON-serialisable
        only — <strong>no pickle</strong> — so they are safe to store and ship.
      </p>

      <CodeBlock
        code={`# Save every 3 iterations during a run
result = agent.run(
    "Do deep research on X, Y, Z and write a report",
    checkpoint_interval=3,
    checkpoint_dir="./checkpoints",
)

# Later (maybe a different process):
resumed = agent.resume(checkpoint_id=result.metadata["checkpoint_id"])
# Or — pass nothing to load the most recent checkpoint in checkpoint_dir
# resumed = agent.resume()`}
        language="python"
        filename="checkpoint_basic.py"
      />

      <h3>Manual use of <code>CheckpointManager</code></h3>
      <CodeBlock
        code={`from effgen.core.checkpoint import CheckpointManager, Checkpoint

mgr = CheckpointManager("./checkpoints")                      # filesystem (JSON)
# mgr = CheckpointManager("./checkpoints", backend="sqlite") # SQLite-backed

cp = Checkpoint(
    checkpoint_id="",                             # auto-populated on save
    agent_name="researcher",
    task="Summarise recent SLM papers",
    iteration=5,
    scratchpad="...",
    partial_output="...",
)
cp_id = mgr.save(cp)

latest = mgr.load_latest()                        # most recent
specific = mgr.load(cp_id)                        # by id (or path)
for entry in mgr.list_checkpoints():              # list metadata for all
    print(entry["checkpoint_id"], entry["agent_name"], entry["created_at"])`}
        language="python"
        filename="checkpoint_manager.py"
      />

      <h3>CLI</h3>
      <CodeBlock
        code={`effgen run "Task" --checkpoint-dir ./ckpt --checkpoint-interval 3
effgen resume --checkpoint <id>`}
        language="bash"
        filename="terminal"
      />

      <h2>Sessions</h2>
      <p>
        A <code>Session</code> is a persistent conversation history (and optional memory
        snapshot) keyed by <code>session_id</code> (UUID by default, but any string works —
        e.g. <code>"user-123"</code>). Stored as JSON under
        <code> ~/.effgen/sessions/&lt;session_id&gt;.json</code>.
      </p>

      <CodeBlock
        code={`from effgen import Agent, AgentConfig, load_model

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# Auto-loads existing session or creates a new one, then auto-persists every turn
agent = Agent(AgentConfig(
    name="chat",
    model=model,
    tools=[...],
), session_id="user-123")

agent.run("What did we talk about yesterday?")   # has access to past turns`}
        language="python"
        filename="session_agent.py"
      />

      <h3>SessionManager</h3>
      <CodeBlock
        code={`from effgen.core.session import SessionManager

sm = SessionManager()                          # ~/.effgen/sessions/ by default
sm.list_sessions()                             # list session metadata dicts
sm.get("user-123").messages                    # conversation history
sm.delete("user-123")                          # delete
sm.cleanup(older_than_days=30)                 # GC sessions older than N days

# export() returns the serialised string — write it yourself
text = sm.export("user-123", format="json")    # or format="text"
open("backup.json", "w").write(text)`}
        language="python"
        filename="session_manager.py"
      />

      <CodeBlock
        code={`effgen sessions list
effgen sessions delete <session_id>
effgen sessions export <session_id> --format json
effgen sessions cleanup --days 30`}
        language="bash"
        filename="terminal"
      />

      <h2>Background Tasks</h2>
      <p>
        <code>BackgroundTaskRunner</code> is a priority queue with pause / resume / cancel and
        threading workers. Agents expose <code>run_background()</code> /
        <code> get_task_status()</code> / <code>get_task_result()</code> /
        <code> cancel_task()</code> over it.
      </p>

      <CodeBlock
        code={`task_id = agent.run_background(
    "Long multi-hour research run",
    priority=5,
    checkpoint_interval=3,
)

# Later:
status = agent.get_task_status(task_id)
# "queued" | "running" | "paused" | "completed" | "cancelled" | "failed"

if status == "completed":
    result = agent.get_task_result(task_id)

# Cancellation is cooperative — applied at iteration boundaries
agent.cancel_task(task_id)`}
        language="python"
        filename="background_tasks.py"
      />

      <h2>Batch Execution</h2>
      <p>
        For embarrassingly-parallel workloads the <code>BatchRunner</code> (via
        <code> Agent.run_batch()</code>) runs many tasks concurrently with a semaphore, retry,
        and timeout. Supports JSONL / CSV / JSON / plain-text I/O.
      </p>

      <CodeBlock
        code={`results = agent.run_batch([
    "Summarise doc 1",
    "Summarise doc 2",
    "Summarise doc 3",
], max_concurrency=8, timeout_per_item=60.0)

# results is a BatchResult; AgentResponse objects come back in input order
for resp in results.results:
    print(resp.output)

# Or use the CLI — streams input from JSONL, writes results alongside
# effgen batch --input tasks.jsonl --output results.jsonl --concurrency 8`}
        language="python"
        filename="batch_run.py"
      />

      <h3>ResultAggregator &amp; ToolResultCache</h3>
      <p>
        Two helpers complement <code>BatchRunner</code> for high-volume workloads:
      </p>
      <FeatureList
        features={[
          { icon: '🧮', title: 'ResultAggregator', description: 'Exact-hash + fuzzy Jaccard deduplication, ranking by confidence/relevance/speed/custom, and merge strategies (first / best / consensus / union).' },
          { icon: '⚡', title: 'ToolResultCache', description: 'Thread-safe LRU + TTL across queries — share tool results between agents, runs, and batch items to avoid redundant work.' },
        ]}
      />
      <CodeBlock
        code={`from effgen.core.aggregation import ResultAggregator, ToolResultCache

# Cross-query tool result cache (shared between batch items)
cache = ToolResultCache(max_size=1024, ttl=600)
cache.put("web_search", "effGen agents", result)
hit = cache.get("web_search", "effGen agents")

# Aggregate batch results — deduplicate then rank
agg = ResultAggregator(fuzzy_threshold=0.85, tool_cache=cache)
unique = agg.deduplicate(results.results, fuzzy=True)
ranked = agg.rank(results.results, key="confidence")`}
        language="python"
        filename="aggregation_and_cache.py"
      />

      <h2>What gets persisted?</h2>
      <ApiTable
        headers={['Field', 'Purpose']}
        rows={[
          [<code>scratchpad</code>, 'Full ReAct thought / action / observation history'],
          [<code>iteration</code>, 'Current loop index (resume picks up here)'],
          [<code>partial_output</code>, 'Best-so-far answer for early-exit scenarios'],
          [<code>tool_calls</code>, 'Count of tool invocations so far'],
          [<code>tokens_used</code>, 'Cumulative token count'],
          [<code>memory</code>, 'ShortTermMemory / LongTermMemory snapshots'],
          [<code>tool_states</code>, 'Per-tool serialisable state (e.g. PythonREPL globals)'],
          [<code>created_at</code>, 'ISO timestamp'],
        ]}
      />

      <InfoBox type="warning" title="No pickle">
        <p>
          Checkpoints are strictly JSON. Tool state must be JSON-serialisable — if a tool holds
          a non-serialisable object (an open file handle, a socket), exclude it from
          <code> tool_states</code> or implement a custom <code>__getstate__</code>-style
          hook that returns a plain dict.
        </p>
      </InfoBox>

      <h2>See Also</h2>
      <p>
        <Link to="/memory">Memory</Link> · <Link to="/agents">Agents</Link> ·
        {' '}<Link to="/api-server">API Server</Link>
      </p>
    </DocPage>
  );
}
