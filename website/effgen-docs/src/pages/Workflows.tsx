import React from 'react';
import { Link } from 'react-router-dom';
import { GitBranch } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';

export default function Workflows() {
  const dagDiagram = `
flowchart LR
    R[research] --> A[analyze]
    R --> S[summarise]
    A --> W[write_report]
    S --> W
    W --> P[publish]
`;

  return (
    <DocPage
      title="DAG Workflows"
      subtitle="Define multi-agent pipelines as a directed acyclic graph with cycle detection, conditional branching, and auto-parallel execution."
      icon={<GitBranch size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Advanced', path: '/multi-agent' },
        { label: 'Workflows' },
      ]}
    >
      <h2>Overview</h2>
      <p>
        <code>WorkflowDAG</code> models multi-agent execution as a directed acyclic graph.
        Edges carry typed data between nodes. Independent nodes run in parallel via
        <code> asyncio.gather</code>. Cycles are rejected at construction via
        <strong> Kahn's topological sort</strong>.
      </p>

      <InfoBox type="success" title="v0.3.0 — workflows fail honestly">
        <p>
          As of v0.3.0, multi-agent teams, workflows, and the DAG are exported top-level, reconciled
          to consistent shapes, and <strong>fail honestly</strong>: a failing node or sub-agent
          yields <code>success=False</code> with a typed per-node error rather than a silent
          partial success, and an empty workflow returns{' '}
          <code>success=False, reason="empty_workflow"</code>. Branch on{' '}
          <code>result.success</code> and inspect the per-node errors instead of assuming a run
          succeeded.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.1 — no work downstream of a failure">
        <p>
          v0.3.1 stops a failed branch from poisoning a customer-facing answer. A node whose
          required upstream <strong>failed or was skipped</strong> is now itself marked{' '}
          <code>skipped</code> with a reason, instead of being executed on the upstream&apos;s error
          text — so an internal error is never rewritten into a downstream reply. The run still
          reports <code>success=False</code>, and (as with teams) the result carries summed{' '}
          <code>cost_usd</code> and tokens.
        </p>
      </InfoBox>

      <MermaidDiagram chart={dagDiagram} title="Example DAG" />

      <h2>Building a Workflow in Python</h2>
      <CodeBlock
        code={`from effgen.core.workflow import WorkflowDAG, WorkflowNode, WorkflowEdge
from effgen.presets import create_agent
from effgen import load_model

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

dag = WorkflowDAG(name="research_to_report")

dag.add_node(WorkflowNode(id="research",   agent=create_agent("research", model), output_key="facts"))
dag.add_node(WorkflowNode(id="analyze",    agent=create_agent("math",     model), input_keys=["facts"]))
dag.add_node(WorkflowNode(id="write",      agent=create_agent("general",  model),
                          input_keys=["facts", "analysis"], output_key="report"))

# Convenience: connect(source, target, key)
dag.connect("research", "analyze",  key="facts")
dag.connect("research", "write",    key="facts")
dag.connect("analyze",  "write",    key="analysis")

# Execute (sync wrapper around run_async, safe inside or outside an event loop)
result = dag.run(initial_inputs={"research": "Write a report on SLMs"})

# Or, from inside an existing event loop:
# result = await dag.run_async(initial_inputs={"research": "Write a report on SLMs"})

print(result.success)
print(result.outputs["write"])         # final node output, keyed by node id
print(result.execution_time)`}
        language="python"
        filename="workflow.py"
      />

      <h2>Conditional Branching</h2>
      <p>
        Edges can carry a <code>condition</code> predicate that receives the source output —
        if it returns <code>False</code>, the target node is skipped (status =
        <code> SKIPPED</code>).
      </p>
      <CodeBlock
        code={`dag.connect(
    "classify", "human_escalation",
    condition=lambda out: out.get("confidence", 1.0) < 0.6,
)`}
        language="python"
        filename="conditional_edge.py"
      />

      <h2>YAML Workflows</h2>
      <p>
        Workflows can also be declared in YAML and executed via the CLI — no Python required
        for the pipeline definition.
      </p>
      <CodeBlock
        code={`# workflow.yaml
name: research_to_report
nodes:
  - id: research
    preset: research
    output_key: facts
  - id: analyze
    preset: math
    input_keys: [facts]
  - id: write
    preset: general
    input_keys: [facts, analysis]
    output_key: report

edges:
  - { source: research, target: analyze, key: facts }
  - { source: research, target: write,   key: facts }
  - { source: analyze,  target: write,   key: analysis }`}
        language="yaml"
        filename="workflow.yaml"
      />

      <CodeBlock
        code={`# CLI
effgen workflow validate workflow.yaml
effgen workflow run workflow.yaml`}
        language="bash"
        filename="terminal"
      />

      <h2>Node Statuses</h2>
      <ApiTable
        headers={['Status', 'Meaning']}
        rows={[
          [<code>PENDING</code>, 'Not yet scheduled'],
          [<code>RUNNING</code>, 'Currently executing'],
          [<code>COMPLETED</code>, 'Finished successfully'],
          [<code>SKIPPED</code>, 'Conditional edge predicate returned False'],
          [<code>FAILED</code>, 'Raised an exception'],
        ]}
      />

      <h2>MessageBus — Pub/Sub Between Agents</h2>
      <p>
        Complementing DAGs, <code>MessageBus</code> provides pub/sub, mailbox, and broadcast
        communication between agents with topic-based wildcard subscriptions and optional
        persistence.
      </p>
      <CodeBlock
        code={`from effgen.core.message_bus import MessageBus, AgentMessage, MessageType

bus = MessageBus(persist_path="./bus.jsonl")

# Topic-based pub/sub with wildcard subscriptions
bus.subscribe("alerts.*", handler=lambda msg: print("ALERT:", msg.payload))
bus.publish(AgentMessage(
    sender="watchdog",
    topic="alerts.gpu",
    msg_type=MessageType.STATUS_UPDATE,
    payload={"level": "critical", "usage": 0.98},
))

# Mailbox semantics — per-agent inbox
bus.send(AgentMessage(
    sender="planner",
    recipient="agent_b",
    msg_type=MessageType.TASK_ASSIGNMENT,
    payload={"task": "summarise"},
))
for msg in bus.receive("agent_b"):
    print(msg.sender, msg.payload)`}
        language="python"
        filename="message_bus.py"
      />

      <h2>SharedState — Thread-Safe KV Store</h2>
      <CodeBlock
        code={`from effgen.core.shared_state import SharedState

state = SharedState()
state.set("session_42", "plan", {"steps": [...]}, agent_id="planner")
plan = state.get("session_42", "plan")

# Snapshot returns a snapshot_id; rollback() restores the entire state to it
snap_id = state.snapshot()
# ... mutate ...
state.rollback(snap_id)

# Event-sourced mutation log
for event in state.get_mutations(namespace="session_42"):
    print(event)`}
        language="python"
        filename="shared_state.py"
      />

      <h2>Agent Lifecycle &amp; Pool</h2>
      <FeatureList
        features={[
          { icon: '♻️', title: 'AgentLifecycleState', description: '8-state machine (created → initialising → ready → running → …). Backed by AgentEntry.' },
          { icon: '🏊', title: 'AgentPool', description: 'Pre-warmed pool with acquire / release semantics; min / max size, idle TTL, health checking.' },
          { icon: '📒', title: 'AgentRegistry', description: 'Thread-safe global registry keyed by agent name; per-agent timeout and cancellation.' },
        ]}
      />

      <InfoBox type="success" title="When to use which?">
        <p>
          Reach for <strong>WorkflowDAG</strong> when you know the shape of the pipeline up
          front. Reach for <strong>MessageBus</strong> when agents need dynamic, unstructured
          communication. Use <strong>SharedState</strong> to synchronise mutable state between
          them. Use <strong>AgentPool</strong> when serving workloads under the <Link to="/api-server">API server</Link>.
        </p>
      </InfoBox>

      <h2>See Also</h2>
      <p>
        <Link to="/multi-agent">Multi-Agent patterns</Link> ·
        {' '}<Link to="/api-server">API Server v2</Link> · <Link to="/agents">Agents</Link>
      </p>
    </DocPage>
  );
}
