"""
DAG-Based Workflow Engine for effGen multi-agent orchestration.

Define agent execution as a directed acyclic graph (DAG):
- WorkflowNode: wraps an agent with input/output specs
- WorkflowEdge: data flow between nodes
- WorkflowDAG: validates, topologically sorts, and executes the graph
- Automatic parallelisation of independent nodes
- Conditional branching based on agent output
- YAML workflow definition support
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..observability.tracing import (
    execution_scope,
    new_execution_id,
    record_skipped_step,
)

logger = logging.getLogger(__name__)


def _redact(text: str) -> str:
    """Scrub secrets from an error string before it is surfaced/logged.

    Mirrors the agent's failure path so orchestrated errors never leak keys.
    Redaction must never mask the underlying error, so any failure falls back
    to the raw text.
    """
    try:
        from ..observability.redact import get_redactor
        return get_redactor().scrub(text)
    except Exception:  # pragma: no cover - redaction is best-effort
        return text


class NodeStatus(Enum):
    """Execution status of a workflow node."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class WorkflowEdge:
    """
    Data-flow edge between two workflow nodes.

    Attributes:
        source: Source node ID
        target: Target node ID
        key: Optional key to extract from source output
        condition: Optional callable ``(source_output) -> bool``; if it
                   returns False the target node is skipped.
    """
    source: str
    target: str
    key: str | None = None
    condition: Callable[[Any], bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "source": self.source,
            "target": self.target,
            "key": self.key,
            "has_condition": self.condition is not None,
        }


@dataclass
class WorkflowNode:
    """
    A single node in a workflow DAG.

    Attributes:
        id: Unique node identifier
        agent: The agent instance to execute (or None for placeholder)
        tools: Tool names to enable for this node
        input_keys: Expected input keys from upstream edges
        output_key: Key under which this node's output is stored
        metadata: Arbitrary metadata (e.g., description)
    """
    id: str
    agent: Any = None  # Agent instance — kept as Any to avoid circular import
    tools: list[str] = field(default_factory=list)
    input_keys: list[str] = field(default_factory=list)
    output_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime state (populated during execution)
    status: NodeStatus = NodeStatus.PENDING
    output: Any = None
    error: str | None = None
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "id": self.id,
            "tools": self.tools,
            "input_keys": self.input_keys,
            "output_key": self.output_key or self.id,
            "status": self.status.value,
            "execution_time": round(self.execution_time, 3),
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowResult:
    """
    Result of executing a workflow DAG.

    Attributes:
        success: True if all required nodes completed
        outputs: Mapping of node_id -> output
        node_results: Per-node detailed results
        execution_time: Total wall-clock time
        metadata: Extra info
    """
    success: bool = True
    outputs: dict[str, Any] = field(default_factory=dict)
    node_results: list[dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (output values truncated to 200 chars)."""
        return {
            "success": self.success,
            "outputs": {k: str(v)[:200] for k, v in self.outputs.items()},
            "node_results": self.node_results,
            "execution_time": round(self.execution_time, 3),
            "metadata": self.metadata,
        }


class WorkflowDAG:
    """
    Directed acyclic graph of agent execution nodes.

    Validates for cycles at construction time (topological sort).
    Executes independent nodes in parallel via ``asyncio.gather``.
    """

    def __init__(self, name: str = "workflow") -> None:
        self.name = name
        self._nodes: dict[str, WorkflowNode] = {}
        self._edges: list[WorkflowEdge] = []
        # Adjacency: node_id -> list of edge
        self._forward: dict[str, list[WorkflowEdge]] = defaultdict(list)
        self._reverse: dict[str, list[WorkflowEdge]] = defaultdict(list)
        self._sorted: list[str] | None = None  # cached topo order

    # -- Construction --

    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the DAG."""
        if node.id in self._nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self._nodes[node.id] = node
        self._sorted = None  # invalidate cache

    def add_edge(self, edge: WorkflowEdge) -> None:
        """
        Add an edge and validate no cycle is introduced.

        Raises ``ValueError`` if the edge would create a cycle.
        """
        if edge.source not in self._nodes:
            raise ValueError(f"Source node '{edge.source}' not in DAG")
        if edge.target not in self._nodes:
            raise ValueError(f"Target node '{edge.target}' not in DAG")

        self._forward[edge.source].append(edge)
        self._reverse[edge.target].append(edge)
        self._edges.append(edge)
        self._sorted = None

        # Validate — will raise on cycle
        try:
            self._topological_sort()
        except ValueError:
            # Roll back
            self._forward[edge.source].remove(edge)
            self._reverse[edge.target].remove(edge)
            self._edges.remove(edge)
            self._sorted = None
            raise

    def connect(self, source: str, target: str,
                key: str | None = None,
                condition: Callable[[Any], bool] | None = None) -> WorkflowEdge:
        """Convenience: create and add an edge between two node IDs."""
        edge = WorkflowEdge(source=source, target=target, key=key, condition=condition)
        self.add_edge(edge)
        return edge

    # -- Topological sort --

    def _topological_sort(self) -> list[str]:
        """
        Kahn's algorithm. Returns topological order or raises
        ``ValueError`` if a cycle exists.
        """
        in_degree: dict[str, int] = dict.fromkeys(self._nodes, 0)
        for edge in self._edges:
            in_degree[edge.target] += 1

        queue = deque(nid for nid, d in in_degree.items() if d == 0)
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for edge in self._forward.get(nid, []):
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(order) != len(self._nodes):
            raise ValueError(
                "Cycle detected in workflow DAG — cannot topologically sort"
            )

        self._sorted = order
        return order

    def topological_order(self) -> list[str]:
        """Return cached topological order (recomputes if stale)."""
        if self._sorted is None:
            self._topological_sort()
        return list(self._sorted)  # type: ignore[arg-type]

    # -- Execution --

    def entry_nodes(self) -> list[str]:
        """Return the ids of nodes with no incoming edges (the DAG's roots)."""
        return [nid for nid in self.topological_order() if not self._reverse.get(nid)]

    def _normalize_initial_inputs(
        self, initial_inputs: dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        """
        Coerce the ``initial_inputs`` argument into a ``{node_id: input}`` dict.

        Accepts the same shapes a user would naturally try:
        - ``None`` -> ``{}`` (nodes get their upstream/empty input).
        - a ``dict`` -> used as-is (the canonical ``{node_id: input}`` form).
        - a bare ``str`` -> routed to every entry (root) node, so the obvious
          ``dag.run("do the task")`` works like ``agent.run("do the task")``.

        Any other type raises a one-line ``TypeError`` naming the expected shape.
        """
        if initial_inputs is None:
            return {}
        if isinstance(initial_inputs, dict):
            return initial_inputs
        if isinstance(initial_inputs, str):
            roots = self.entry_nodes()
            if not roots:
                # No nodes (or — impossible for a DAG — no roots): nothing to seed.
                return {}
            return dict.fromkeys(roots, initial_inputs)
        raise TypeError(
            "WorkflowDAG.run() expects the task as a {node_id: input} dict "
            "(e.g. {'search': 'find recent news'}) or a single string routed to "
            f"the entry node(s) {self.entry_nodes()!r}; got {type(initial_inputs).__name__}."
        )

    def run(self, initial_inputs: dict[str, Any] | str | None = None,
            context: dict[str, Any] | None = None) -> WorkflowResult:
        """
        Execute the workflow synchronously.

        Args:
            initial_inputs: Either a ``{node_id: task}`` mapping, or a single
                task string that is routed to the workflow's entry node(s).
            context: Shared context dict passed to each agent

        Returns:
            WorkflowResult

        Example::

            from effgen import Agent, AgentConfig, WorkflowDAG, WorkflowNode, load_model

            m = load_model("gpt-5-nano")
            dag = WorkflowDAG("pipeline")
            dag.add_node(WorkflowNode(id="draft", agent=Agent(AgentConfig(name="w", model=m))))
            dag.add_node(WorkflowNode(id="polish", agent=Agent(AgentConfig(name="e", model=m))))
            dag.connect("draft", "polish")
            result = dag.run("Write one sentence about the sea.")
            print(result.outputs["polish"])
        """
        initial_inputs = self._normalize_initial_inputs(initial_inputs)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop — use thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.run_async(initial_inputs, context))
                return future.result()
        else:
            return asyncio.run(self.run_async(initial_inputs, context))

    def execute(self, initial_inputs: dict[str, Any] | str | None = None,
                context: dict[str, Any] | None = None) -> WorkflowResult:
        """Alias for :meth:`run` — executes the workflow synchronously."""
        return self.run(initial_inputs, context)

    async def execute_async(self, initial_inputs: dict[str, Any] | str | None = None,
                            context: dict[str, Any] | None = None) -> WorkflowResult:
        """Alias for :meth:`run_async` — executes the workflow asynchronously."""
        return await self.run_async(initial_inputs, context)

    async def run_async(self, initial_inputs: dict[str, Any] | str | None = None,
                        context: dict[str, Any] | None = None) -> WorkflowResult:
        """
        Execute the workflow asynchronously.

        Accepts the same ``initial_inputs`` shapes as :meth:`run` (a
        ``{node_id: task}`` dict or a single string routed to the entry nodes).
        Independent nodes at the same topological level are run in parallel
        via ``asyncio.gather``.
        """
        start = time.time()
        initial_inputs = self._normalize_initial_inputs(initial_inputs)
        context = context or {}
        execution_id = new_execution_id()

        # A zero-node workflow has nothing to run: report it explicitly instead
        # of an empty success (all([]) is True). Mirrors the empty-team contract
        # in MultiAgentOrchestrator.assign_task.
        if not self._nodes:
            return WorkflowResult(
                success=False,
                outputs={},
                node_results=[],
                execution_time=time.time() - start,
                metadata={
                    "name": self.name,
                    "node_count": 0,
                    "reason": "empty_workflow",
                    "error": "Workflow has no nodes. Add nodes with add_node() "
                    "before running.",
                },
            )

        # Reset node state
        for node in self._nodes.values():
            node.status = NodeStatus.PENDING
            node.output = None
            node.error = None
            node.execution_time = 0.0

        order = self.topological_order()

        # Group nodes into levels for parallel execution
        levels = self._compute_levels(order)

        outputs: dict[str, Any] = {}

        for level_nodes in levels:
            tasks = []
            for nid in level_nodes:
                node = self._nodes[nid]

                # Decide whether to skip this node. Two reasons:
                #  1. A required upstream did NOT complete (it failed or was
                #     itself skipped) — running on its error text would turn an
                #     internal failure into a (often customer-facing) answer, so
                #     we skip explicitly instead.
                #  2. A conditional edge's predicate returned False.
                skip = False
                skip_reason: str | None = None
                for edge in self._reverse.get(nid, []):
                    src = self._nodes.get(edge.source)
                    if src is not None and src.status in (
                        NodeStatus.FAILED, NodeStatus.SKIPPED
                    ):
                        skip = True
                        skip_reason = (
                            f"upstream '{edge.source}' did not complete "
                            f"({src.status.value})"
                        )
                        break
                    if edge.condition is not None:
                        source_output = outputs.get(edge.source)
                        if not edge.condition(source_output):
                            skip = True
                            skip_reason = f"condition on edge from '{edge.source}' was not met"
                            break

                if skip:
                    node.status = NodeStatus.SKIPPED
                    if skip_reason:
                        node.metadata = {**node.metadata, "skip_reason": skip_reason}
                    # Record the skip against the workflow so a consumer reading
                    # telemetry sees a skipped node, not a missing one.
                    upstream_ids = [e.source for e in self._reverse.get(nid, [])]
                    with execution_scope(
                        kind="workflow", name=self.name,
                        execution_id=execution_id, role=f"node:{node.id}",
                        parent_agent=upstream_ids[0] if upstream_ids else None,
                    ):
                        record_skipped_step(
                            node.id, reason=skip_reason or "upstream did not complete",
                        )
                    continue

                # Build input for this node from upstream outputs + initial
                node_input = initial_inputs.get(nid, "")
                upstream_data: dict[str, Any] = {}
                for edge in self._reverse.get(nid, []):
                    src_out = outputs.get(edge.source)
                    if edge.key and isinstance(src_out, dict):
                        upstream_data[edge.key] = src_out.get(edge.key, src_out)
                    else:
                        upstream_data[edge.source] = src_out

                if upstream_data:
                    # Append upstream data as context to the task
                    context_str = "\n".join(
                        f"[{k}]: {v}" for k, v in upstream_data.items()
                    )
                    if node_input:
                        node_input = f"{node_input}\n\nContext from previous steps:\n{context_str}"
                    else:
                        node_input = context_str

                tasks.append(self._run_node(
                    node, node_input, context,
                    execution_id=execution_id,
                ))

            if tasks:
                await asyncio.gather(*tasks)

            # Collect outputs from this level
            for nid in level_nodes:
                node = self._nodes[nid]
                out_key = node.output_key or node.id
                outputs[out_key] = node.output
                # Also store under node id if output_key differs
                if out_key != node.id:
                    outputs[node.id] = node.output

        elapsed = time.time() - start
        success = all(
            n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
            for n in self._nodes.values()
        )

        # Fold a running cost/token tab onto the result so a budget owner can read
        # workflow spend without summing node_results by hand. The tab sums the
        # nodes that reported a cost; when none did (every model unpriced or
        # local) it is ``None`` rather than ``$0``.
        total_cost = 0.0
        priced_nodes = 0
        total_tokens = 0
        for n in self._nodes.values():
            raw_node_cost = n.metadata.get("cost_usd")
            if raw_node_cost is not None:
                try:
                    total_cost += float(raw_node_cost)
                    priced_nodes += 1
                except (TypeError, ValueError):
                    pass
            try:
                total_tokens += int(n.metadata.get("tokens_used") or 0)
            except (TypeError, ValueError):
                pass

        return WorkflowResult(
            success=success,
            outputs=outputs,
            node_results=[n.to_dict() for n in self._nodes.values()],
            execution_time=elapsed,
            metadata={
                "name": self.name,
                "node_count": len(self._nodes),
                "cost_usd": round(total_cost, 6) if priced_nodes else None,
                "tokens_used": total_tokens,
                # Carry the topology so any consumer can rebuild the graph from
                # the result alone (the per-node dicts don't record edges).
                "edges": [e.to_dict() for e in self._edges],
                "topological_order": order,
                "levels": levels,
                "execution_id": execution_id,
            },
        )

    async def _run_node(self, node: WorkflowNode, task: str,
                        context: dict[str, Any],
                        *, execution_id: str | None = None) -> None:
        """Execute a single workflow node.

        A node is COMPLETED only when its agent returns a real, successful
        answer. If the agent reports ``success=False`` (e.g. a bad model id or
        an auth error inside the node) the node is marked FAILED and carries a
        typed, redacted error — never a silent success — matching the failure
        contract used everywhere else.

        The node's run is tagged with the workflow's execution id and the node
        id, so its spans and its stored run record group with the rest of the
        workflow rather than standing alone.
        """
        node.status = NodeStatus.RUNNING
        t0 = time.time()
        try:
            if node.agent is None:
                raise ValueError(f"Node '{node.id}' has no agent assigned")

            upstream = [e.source for e in self._reverse.get(node.id, [])]
            scope_kwargs = {
                "kind": "workflow",
                "name": self.name,
                "execution_id": execution_id,
                "role": f"node:{node.id}",
                "parent_agent": upstream[0] if upstream else None,
            }

            # Use async if available, else run in executor
            if hasattr(node.agent, "run_async"):
                with execution_scope(**scope_kwargs):
                    response = await node.agent.run_async(task, context=context)
            else:
                loop = asyncio.get_running_loop()

                def _call() -> Any:
                    # A thread pool does not inherit the caller's context, so
                    # the scope is entered inside the worker.
                    with execution_scope(**scope_kwargs):
                        return node.agent.run(task, context=context)

                response = await loop.run_in_executor(None, _call)

            node.output = response.output if hasattr(response, "output") else str(response)

            # Record this node's spend so the workflow can report a running tab
            # (mirrors the per-run AgentResponse cost surface). A node whose
            # model publishes no price — a local engine, or a catalog id with no
            # rate — records ``None``, not a zero the reader would take for a
            # free call.
            r_meta = getattr(response, "metadata", None) or {}
            raw_cost = r_meta.get("cost_usd", r_meta.get("cost"))
            try:
                node.metadata["cost_usd"] = float(raw_cost) if raw_cost is not None else None
            except (TypeError, ValueError):
                node.metadata["cost_usd"] = None
            node.metadata["tokens_used"] = int(getattr(response, "tokens_used", 0) or 0)

            # Honour the agent's own success flag: a sub-agent that failed must
            # fail the node too (its error is already typed + redacted upstream).
            if getattr(response, "success", True) is False:
                detail = (getattr(response, "metadata", None) or {}).get("error")
                if isinstance(detail, dict):
                    node.error = (
                        f"{detail.get('type', 'AgentError')}: "
                        f"{detail.get('message', node.output)}"
                    )
                    node.metadata["error_detail"] = detail
                else:
                    node.error = _redact(str(node.output))
                node.status = NodeStatus.FAILED
                logger.error("Workflow node '%s' failed: %s", node.id, node.error)
            else:
                node.status = NodeStatus.COMPLETED
        except Exception as e:
            node.error = f"{type(e).__name__}: {_redact(str(e))}"
            node.status = NodeStatus.FAILED
            logger.error("Workflow node '%s' failed: %s", node.id, node.error)
        finally:
            node.execution_time = time.time() - t0

    def _compute_levels(self, order: list[str]) -> list[list[str]]:
        """
        Group topologically-sorted nodes into parallel levels.

        Nodes in the same level have no edges between them and can
        be executed concurrently.
        """
        node_level: dict[str, int] = {}
        for nid in order:
            deps = [node_level[e.source] for e in self._reverse.get(nid, [])]
            node_level[nid] = (max(deps) + 1) if deps else 0

        levels: dict[int, list[str]] = defaultdict(list)
        for nid, lvl in node_level.items():
            levels[lvl].append(nid)

        return [levels[i] for i in sorted(levels)]

    # -- YAML loading --

    @classmethod
    def from_yaml(cls, path: str, agent_factory: Callable[[dict[str, Any]], Any] | None = None) -> WorkflowDAG:
        """
        Load a workflow from a YAML file.

        Dependencies can be declared per-node with ``depends_on`` or in a
        top-level ``edges`` list; both build the same graph. Each edge is either
        a ``[source, target]`` pair or a mapping with ``source``/``target``
        (aliases ``from``/``to``) and an optional ``key``::

            workflow:
              name: my_pipeline
              nodes:
                - id: search
                  agent: research_agent
                  tools: [web_search]
                - id: summarize
                  agent: summary_agent
                  depends_on: [search]

            # equivalent wiring via a top-level edges block:
            #   edges:
            #     - [search, summarize]

        Args:
            path: Path to the YAML file
            agent_factory: Optional callable that receives a node dict and
                           returns an Agent instance. If None, nodes are
                           created without agents (must be assigned later).
                           The CLI's ``effgen workflow run`` supplies a
                           factory that reads a node's ``model:`` key first,
                           then falls back to treating ``agent:`` as a model
                           id (e.g. ``agent: gpt-5-nano``); pass ``-m`` to
                           override every node's model at once.

        Returns:
            A validated WorkflowDAG

        An unrecognized top-level key is reported with a warning rather than
        dropped silently, so a mis-keyed file (e.g. ``edge:`` instead of
        ``edges:``) does not validate as a workflow with all its declared wiring.
        """
        import yaml  # pyyaml is an existing dependency

        with open(path) as f:
            data = yaml.safe_load(f)

        wf_data = data.get("workflow", data)
        name = wf_data.get("name", "workflow")

        # Surface unknown top-level keys instead of dropping the wiring silently.
        _known_top = {"name", "nodes", "edges", "description", "metadata"}
        unknown = [k for k in wf_data if k not in _known_top]
        if unknown:
            logger.warning(
                "Workflow '%s': ignoring unrecognized top-level key(s) %s "
                "(recognized: %s). Declare dependencies with per-node "
                "'depends_on' or a top-level 'edges' list.",
                name, sorted(unknown), sorted(_known_top),
            )

        dag = cls(name=name)

        node_defs = wf_data.get("nodes", [])
        for nd in node_defs:
            agent = None
            if agent_factory:
                agent = agent_factory(nd)

            node = WorkflowNode(
                id=nd["id"],
                agent=agent,
                tools=nd.get("tools", []),
                input_keys=nd.get("input_keys", []),
                output_key=nd.get("output_key", nd["id"]),
                metadata={k: v for k, v in nd.items()
                          if k not in ("id", "tools", "input_keys",
                                       "output_key", "depends_on", "agent")},
            )
            dag.add_node(node)

        # Create edges from per-node depends_on ...
        for nd in node_defs:
            for dep in nd.get("depends_on", []):
                dag.connect(dep, nd["id"])

        # ... and from a top-level edges list (same graph; both may be present).
        for edge in wf_data.get("edges", []):
            src, tgt, key = cls._parse_yaml_edge(edge)
            dag.connect(src, tgt, key=key)

        return dag

    @staticmethod
    def _parse_yaml_edge(edge: Any) -> tuple[str, str, str | None]:
        """Parse one entry of a YAML ``edges`` list into ``(source, target, key)``.

        Accepts a ``[source, target]`` pair or a mapping with ``source``/``target``
        (aliases ``from``/``to``) and an optional ``key``.
        """
        if isinstance(edge, list | tuple):
            if len(edge) < 2:
                raise ValueError(
                    f"Workflow edge {edge!r} must be [source, target]."
                )
            return str(edge[0]), str(edge[1]), (str(edge[2]) if len(edge) > 2 else None)
        if isinstance(edge, dict):
            src = edge.get("source", edge.get("from"))
            tgt = edge.get("target", edge.get("to"))
            if not src or not tgt:
                raise ValueError(
                    f"Workflow edge {edge!r} needs 'source'/'target' "
                    "(aliases 'from'/'to')."
                )
            key = edge.get("key")
            return str(src), str(tgt), (str(key) if key is not None else None)
        raise ValueError(
            f"Unsupported workflow edge {edge!r}: use [source, target] or "
            "{source, target}."
        )

    # -- Introspection --

    def get_node(self, node_id: str) -> WorkflowNode | None:
        """Return the node with id *node_id*, or ``None`` when absent."""
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> list[WorkflowNode]:
        """All nodes in insertion order."""
        return list(self._nodes.values())

    @property
    def edges(self) -> list[WorkflowEdge]:
        """All edges in insertion order."""
        return list(self._edges)

    def to_dict(self) -> dict[str, Any]:
        """Return the DAG (nodes, edges, topological order) as a serializable dict."""
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "topological_order": self.topological_order(),
        }

    def __repr__(self) -> str:
        return f"WorkflowDAG(name={self.name!r}, nodes={len(self._nodes)}, edges={len(self._edges)})"
