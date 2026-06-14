"""Unit tests for multi-agent orchestration & workflow correctness.

Covers the doorway and the failure semantics of the coordination surface:
- WorkflowDAG.run() accepts a bare string (routed to entry nodes) or raises a
  clear typed error; topological order + cycle rejection.
- A node / sub-agent that reports success=False fails the workflow/team (never
  a silent success); partial results are labelled.
- assign_task accepts a TeamConfig or a registered team name; empty team fails.
- MessageBus pub/sub, ApprovalManager approve/deny/timeout, cooperative
  cancellation, the consensus score, and the public-surface exports.

These use an in-process fake agent (no network / no model) — orchestration glue
is what's under test, not live API behaviour (which is exercised separately in
tests/integration/test_multi_agent_live.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest


# --------------------------------------------------------------------------- #
# Fake agent: matches the small slice of the Agent contract the orchestrator and
# workflow engine actually use (name, run/run_async, success/output/tokens, close).
# --------------------------------------------------------------------------- #
@dataclass
class _FakeResponse:
    output: str
    success: bool = True
    tokens_used: int = 7
    tool_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeAgent:
    def __init__(self, name: str, *, succeed: bool = True, reply: str | None = None,
                 delay: float = 0.0):
        self.name = name
        self._succeed = succeed
        self._reply = reply
        self._delay = delay
        self.closed = False
        self.runs = 0

    def run(self, task: str, mode: Any = None, context: Any = None, **kw) -> _FakeResponse:
        self.runs += 1
        if self._delay:
            time.sleep(self._delay)
        if not self._succeed:
            return _FakeResponse(
                output="boom: sk-secret-key-12345 leaked",
                success=False,
                metadata={"error": {"type": "ModelNotFoundError",
                                    "message": "model 'x' not found", "provider": "test"}},
            )
        return _FakeResponse(output=self._reply or f"{self.name}:{task}")

    async def run_async(self, task: str, mode: Any = None, context: Any = None, **kw):
        return self.run(task, mode=mode, context=context, **kw)

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------- #
# Public surface (GA2)
# --------------------------------------------------------------------------- #
def test_entry_points_exported_top_level():
    import effgen
    for name in [
        "MultiAgentOrchestrator", "TeamConfig", "TeamResponse", "OrchestrationPattern",
        "WorkflowDAG", "WorkflowNode", "WorkflowEdge", "WorkflowResult",
        "SubAgentRouter", "RoutingStrategy", "RoutingDecision",
    ]:
        assert name in effgen.__all__, f"{name} missing from __all__"
        assert getattr(effgen, name) is not None


# --------------------------------------------------------------------------- #
# WorkflowDAG type-guard + topology (GA1)
# --------------------------------------------------------------------------- #
def _two_node_dag(a_ok=True, b_ok=True):
    from effgen.core.workflow import WorkflowDAG, WorkflowNode
    dag = WorkflowDAG("t")
    dag.add_node(WorkflowNode(id="a", agent=FakeAgent("a", succeed=a_ok)))
    dag.add_node(WorkflowNode(id="b", agent=FakeAgent("b", succeed=b_ok)))
    dag.connect("a", "b")
    return dag


def test_run_accepts_bare_string():
    dag = _two_node_dag()
    res = dag.run("do the task")
    assert res.success is True
    assert {n["id"]: n["status"] for n in res.node_results} == {
        "a": "completed", "b": "completed"}


def test_run_rejects_bad_type_with_clear_error():
    dag = _two_node_dag()
    with pytest.raises(TypeError) as ei:
        dag.run(123)
    assert "node_id" in str(ei.value)


def test_entry_nodes_and_topological_order():
    dag = _two_node_dag()
    assert dag.entry_nodes() == ["a"]
    assert dag.topological_order() == ["a", "b"]


def test_cycle_rejected():
    dag = _two_node_dag()
    with pytest.raises(ValueError):
        dag.connect("b", "a")


def test_node_failure_fails_workflow_and_redacts():
    # b fails -> workflow not success; a stays completed (labelled partial).
    dag = _two_node_dag(b_ok=False)
    res = dag.run("go")
    statuses = {n["id"]: n["status"] for n in res.node_results}
    assert statuses == {"a": "completed", "b": "failed"}
    assert res.success is False
    berr = next(n["error"] for n in res.node_results if n["id"] == "b")
    assert "sk-secret-key-12345" not in str(res.node_results)  # redacted/structured
    assert "ModelNotFoundError" in berr


# --------------------------------------------------------------------------- #
# Orchestrator shapes + honest failure (GA2 + Phase-0 bar)
# --------------------------------------------------------------------------- #
def _orch_with_team(name="team", agents=None, pattern=None):
    from effgen.core.orchestrator import MultiAgentOrchestrator, OrchestrationPattern
    orch = MultiAgentOrchestrator()
    pattern = pattern or OrchestrationPattern.SEQUENTIAL
    agents = agents if agents is not None else [FakeAgent("a1"), FakeAgent("a2")]
    orch.create_team(name, agents, pattern=pattern)
    return orch


def test_assign_task_by_name_and_by_config():
    from effgen.core.orchestrator import OrchestrationPattern
    orch = _orch_with_team()
    by_name = orch.assign_task("hi", "team")
    assert by_name.success is True
    by_cfg = orch.assign_task("hi", orch.get_team("team"))
    assert by_cfg.success is True
    assert by_cfg.pattern == OrchestrationPattern.SEQUENTIAL


def test_assign_task_unknown_name_raises_keyerror():
    orch = _orch_with_team()
    with pytest.raises(KeyError):
        orch.assign_task("hi", "nope")


def test_assign_task_bad_type_raises_typeerror():
    orch = _orch_with_team()
    with pytest.raises(TypeError):
        orch.assign_task("hi", 123)


def test_empty_team_is_honest_failure():
    from effgen.core.orchestrator import MultiAgentOrchestrator, OrchestrationPattern, TeamConfig
    orch = MultiAgentOrchestrator()
    orch.teams["empty"] = TeamConfig(name="empty", pattern=OrchestrationPattern.SEQUENTIAL, agents=[])
    res = orch.assign_task("hi", "empty")
    assert res.success is False
    assert res.metadata.get("reason") == "empty_team"


def test_failing_agent_makes_team_fail_with_redacted_error():
    orch = _orch_with_team(agents=[FakeAgent("ok"), FakeAgent("bad", succeed=False)])
    res = orch.assign_task("hi", "team")
    assert res.success is False
    assert res.metadata.get("reason") == "sub_agent_failed"
    # the failing agent's response carries an error; secrets are not leaked raw
    assert "sk-secret-key-12345" not in str(res.metadata.get("error"))


def test_parallel_all_fail_is_failure():
    from effgen.core.orchestrator import OrchestrationPattern
    orch = _orch_with_team(
        agents=[FakeAgent("p1", succeed=False), FakeAgent("p2", succeed=False)],
        pattern=OrchestrationPattern.PARALLEL,
    )
    res = orch.assign_task("hi", "team")
    assert res.success is False


# --------------------------------------------------------------------------- #
# Consensus, message bus, approval, cancellation
# --------------------------------------------------------------------------- #
def test_consensus_score_range():
    from effgen.core.orchestrator import MultiAgentOrchestrator
    orch = MultiAgentOrchestrator()
    assert orch._calculate_consensus([{"output": "the cat sat"},
                                      {"output": "the cat sat"}]) == 1.0
    assert orch._calculate_consensus([{"output": "the cat sat"},
                                      {"output": "dogs run far"}]) == 0.0
    assert orch._calculate_consensus([{"output": "alone"}]) == 1.0  # trivial


def test_message_bus_pubsub_wildcard():
    from effgen.core.message_bus import AgentMessage, MessageBus, MessageType
    bus = MessageBus(persist=True)
    got = []
    bus.subscribe("team.*.result", lambda m: got.append(m.payload))
    bus.send(AgentMessage(sender="a", recipient="o", type=MessageType.RESULT,
                          payload="match", topic="team.x.result"))
    bus.send(AgentMessage(sender="a", recipient="o", type=MessageType.STATUS_UPDATE,
                          payload="nomatch", topic="team.x.status"))
    assert got == ["match"]
    assert len(bus.get_history()) == 2


def test_approval_manager_approve_deny_timeout():
    from effgen.core.human_loop import ApprovalDecision, ApprovalManager, ApprovalMode
    yes = ApprovalManager(mode=ApprovalMode.ALWAYS, callback=lambda n, a: True)
    no = ApprovalManager(mode=ApprovalMode.ALWAYS, callback=lambda n, a: False)
    to = ApprovalManager(mode=ApprovalMode.ALWAYS, timeout=0.2,
                         callback=lambda n, a: (time.sleep(2) or True))
    assert yes.request_approval("bash", "ls") == ApprovalDecision.APPROVED
    assert no.request_approval("bash", "rm") == ApprovalDecision.DENIED
    assert to.request_approval("bash", "x") == ApprovalDecision.TIMEOUT


def test_cancel_stops_in_flight_sequential():
    # Cancel after the first agent's RESULT is published (synchronous delivery),
    # so the loop stops before launching the next agent.
    orch = _orch_with_team(agents=[FakeAgent("c1"), FakeAgent("c2"), FakeAgent("c3")])
    fired = {"v": False}

    def hook(_msg):
        if not fired["v"]:
            fired["v"] = True
            orch.cancel_workflow("team")

    orch.message_bus.subscribe("team.team.result", hook)
    res = orch.assign_task("count", "team")
    assert res.success is False
    assert res.metadata.get("reason") == "cancelled"
    assert len(res.agent_responses) == 1  # only the first ran


# --------------------------------------------------------------------------- #
# SubAgentManager: real execution requirement (no fabrication)
# --------------------------------------------------------------------------- #
def test_sub_agent_requires_real_parent_model():
    from effgen.core.sub_agent_manager import SubAgentConfig, SubAgentManager
    from effgen.core.task import SubTask
    mgr = SubAgentManager(parent_agent=None)  # no model available
    st = SubTask(id="st_1", description="do something", expected_output="x")
    cfg = SubAgentConfig.get_default_config("general")
    with pytest.raises(RuntimeError):
        mgr._run_real_sub_agent(st, cfg)


def test_sub_agent_manager_has_no_simulation_placeholder():
    # The fabrication trap must be gone for good.
    from effgen.core import sub_agent_manager as sam
    assert not hasattr(sam.SubAgentManager, "_simulate_execution")
    assert hasattr(sam.SubAgentManager, "_run_real_sub_agent")


# --------------------------------------------------------------------------- #
# SubAgentRouter type guard
# --------------------------------------------------------------------------- #
def test_router_rejects_non_string_task():
    from effgen.core.router import SubAgentRouter
    with pytest.raises(TypeError):
        SubAgentRouter().route({"not": "a string"})
