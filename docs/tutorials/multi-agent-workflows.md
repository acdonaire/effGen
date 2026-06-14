# Multi-Agent Workflows

effGen provides multi-agent orchestration with a message bus, DAG-based
workflows, shared state, and agent lifecycle management. The main entry points
are importable from the top level:

```python
from effgen import (
    MultiAgentOrchestrator, OrchestrationPattern, TeamConfig,
    WorkflowDAG, WorkflowNode, SubAgentRouter,
)
```

## Teams — MultiAgentOrchestrator

Coordinate several agents with a chosen pattern (sequential, parallel,
hierarchical, collaborative, competitive, pipeline):

```python
from effgen import Agent, AgentConfig, MultiAgentOrchestrator, OrchestrationPattern, load_model

m = load_model("gpt-5-nano")
writer = Agent(AgentConfig(name="writer", model=m))
editor = Agent(AgentConfig(name="editor", model=m))

orch = MultiAgentOrchestrator()
orch.create_team("blog", [writer, editor], pattern=OrchestrationPattern.SEQUENTIAL)

# Pass the team object or just its name:
result = orch.assign_task("Write one sentence about the ocean.", "blog")
print(result.success, result.output)
```

`assign_task` returns a `TeamResponse` whose `success` is `False` (never a
silent success) if the team is empty or any agent fails; per-agent outputs and
errors are kept in `result.agent_responses`.

## MessageBus — Agent Communication

```python
from effgen.core.message_bus import MessageBus, AgentMessage, MessageType

bus = MessageBus()

# Subscribe to a topic pattern (supports * / ? wildcards)
bus.subscribe("results.*", lambda msg: print(f"Got result: {msg.payload}"))

# Publish a message to its topic
bus.publish(AgentMessage(
    sender="math_agent",
    recipient="coordinator",
    type=MessageType.RESULT,
    payload={"answer": 42},
    topic="results.math",
))

# Mailbox-based (direct agent-to-agent)
bus.send(AgentMessage(
    sender="coordinator",
    recipient="research_agent",
    type=MessageType.TASK_ASSIGNMENT,
    payload={"task": "Search for quantum computing papers"},
))
messages = bus.receive("research_agent")
```

## DAG-Based Workflows

Define complex workflows as directed acyclic graphs:

### Python API

```python
from effgen import WorkflowDAG, WorkflowNode

dag = WorkflowDAG("report_pipeline")
dag.add_node(WorkflowNode(id="research", agent=research_agent))
dag.add_node(WorkflowNode(id="summarize", agent=summary_agent))
dag.add_node(WorkflowNode(id="format", agent=format_agent))

dag.connect("research", "summarize")   # add an edge (validates against cycles)
dag.connect("summarize", "format")

# A single string is routed to the entry node(s); or pass {node_id: task}.
result = dag.run("Find recent papers on small language models and write a report.")
# Runs in topological order and parallelizes independent nodes.
print(result.success, result.outputs["format"])
```

### YAML Workflow Definitions

```yaml
# workflow.yaml
nodes:
  - id: fetch_data
    preset: research
    task: "Fetch latest data on {topic}"
  - id: analyze
    preset: coding
    task: "Analyze the data and compute statistics"
    depends_on: [fetch_data]
  - id: visualize
    preset: coding
    task: "Create charts from the analysis"
    depends_on: [analyze]
  - id: report
    preset: general
    task: "Write a summary report"
    depends_on: [analyze, visualize]
```

```bash
# CLI
effgen workflow validate workflow.yaml
effgen workflow run workflow.yaml
```

### Conditional Branching

Edges can have conditions based on previous node outputs:

```python
dag.add_edge("classify", "handle_urgent", condition=lambda result: "urgent" in result.output)
dag.add_edge("classify", "handle_normal", condition=lambda result: "urgent" not in result.output)
```

## Shared State

Thread-safe key-value store shared across agents in a workflow:

```python
from effgen.core.shared_state import SharedState

state = SharedState()

# Namespaced access
state.set("research", "papers_found", 42)
state.set("analysis", "top_topic", "quantum computing")

count = state.get("research", "papers_found")  # 42

# Snapshots for rollback
snapshot = state.snapshot()
# ... do work ...
state.rollback(snapshot)  # Restore to the snapshot
```

## Agent Lifecycle Management

```python
from effgen.core.lifecycle import AgentRegistry, AgentPool

# Registry — track agents by id
registry = AgentRegistry()
registry.register("agent-1", agent, timeout=300)  # 5-minute timeout

# Pool — reuse pre-built agents for fast allocation
pool = AgentPool(max_size=10)
pool.add(create_agent("general", model))  # pre-warm
agent = pool.acquire()
try:
    result = agent.run("task")
finally:
    pool.release(agent)

# Timeout and cancellation
registry.check_timeouts()    # Cancel agents that exceeded their timeout
registry.cancel("agent-1")   # Cancel a specific agent
```

## CLI Commands

```bash
# Workflow operations
effgen workflow run pipeline.yaml --model "Qwen/Qwen2.5-3B-Instruct"
effgen workflow validate pipeline.yaml

# Batch execution
effgen batch --input queries.jsonl --output results.jsonl --concurrency 5 --preset research
```
