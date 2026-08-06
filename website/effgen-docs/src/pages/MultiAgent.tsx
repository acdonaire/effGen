import React from 'react';
import { Link } from 'react-router-dom';
import { GitBranch } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';

export default function MultiAgent() {
  const orchestrationDiagram = `
flowchart TB
    subgraph Orchestrator["Multi-Agent Orchestrator"]
        O[Orchestrator]
    end

    subgraph Patterns["Orchestration Patterns"]
        SEQ[Sequential]
        PAR[Parallel]
        HIER[Hierarchical]
        COLLAB[Collaborative]
    end

    subgraph Agents["Specialized Agents"]
        R[Researcher]
        A[Analyst]
        W[Writer]
        C[Coder]
    end

    O --> Patterns
    SEQ --> R
    R --> A
    A --> W

    PAR --> R
    PAR --> A
    PAR --> W

    HIER --> M[Manager]
    M --> R
    M --> A
    M --> W

    COLLAB --> R
    COLLAB --> A
    R <--> A
`;

  return (
    <DocPage
      title="Multi-Agent Orchestration"
      subtitle="Coordinate multiple specialized agents to tackle complex tasks. effGen provides DAG workflows, MessageBus pub/sub, SharedState, AgentPool + lifecycle, plus five classic orchestration patterns for backward compatibility."
      icon={<GitBranch size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Advanced', path: '/multi-agent' },
        { label: 'Multi-Agent' },
      ]}
    >
      <h2>Overview</h2>
      <p>
        Complex tasks often benefit from multiple specialized agents working together. The
        Multi-Agent Orchestrator coordinates agent interactions using various patterns.
      </p>

      <InfoBox type="success" title="v0.3.0 — teams &amp; workflows fail honestly">
        <p>
          As of v0.3.0, multi-agent teams, workflows, and the DAG are exported top-level, reconciled
          to consistent shapes, and <strong>fail honestly</strong>: a failing sub-agent or node
          yields <code>success=False</code> with a typed per-node error instead of a silent partial
          success (an empty workflow returns <code>reason="empty_workflow"</code>). Each sub-agent
          run also follows the framework-wide fail-closed contract — see{' '}
          <Link to="/agents">Agents → Error Handling</Link> and <Link to="/workflows">Workflows</Link>.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.1 — honest routing &amp; costed teams">
        <p>
          v0.3.1 makes the orchestration patterns trustworthy for real handoffs.{' '}
          <strong>Collaborative</strong> teams now fail closed — a failed collaborator sets the
          team <code>success=False</code> with a discoverable per-agent error instead of always
          returning <code>success=True</code>. <strong>Hierarchical</strong> teams route each
          subtask to the worker the manager <em>named</em> (round-robin only as a fallback), run
          every subtask, and let a failed worker fail the team — making hierarchical the real
          triage&nbsp;&rarr;&nbsp;specialist handoff. On failure, sequential / hierarchical no
          longer echo the caller&apos;s own input back as the answer, and team / workflow results
          report summed <code>cost_usd</code> and tokens in their metadata.
        </p>
      </InfoBox>

      <MermaidDiagram chart={orchestrationDiagram} title="Multi-Agent Orchestration Patterns" />

      <h2>Orchestration Patterns</h2>

      <ApiTable
        headers={['Pattern', 'Description', 'Best For']}
        rows={[
          [<code>SEQUENTIAL</code>, 'Agents work one after another, passing results', 'Pipelines, step-by-step processing'],
          [<code>PARALLEL</code>, 'Agents work simultaneously on different aspects', 'Independent sub-tasks'],
          [<code>HIERARCHICAL</code>, 'Manager agent delegates to worker agents', 'Complex coordinated tasks'],
          [<code>COLLABORATIVE</code>, 'Agents discuss and reach consensus', 'Decision making, analysis'],
          [<code>COMPETITIVE</code>, 'Agents compete for best solution', 'Creative tasks, optimization'],
        ]}
      />

      <h2>Basic Usage</h2>

      <CodeBlock
        code={`from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.core.orchestrator import (
    MultiAgentOrchestrator,
    OrchestrationPattern
)

from effgen.tools.builtin import WebSearch, Calculator, PythonREPL

model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

# Create specialized agents
researcher = Agent(config=AgentConfig(
    name="researcher",
    model=model,
    system_prompt="You are an expert researcher. Gather comprehensive information on topics.",
    tools=[WebSearch()]
))

analyst = Agent(config=AgentConfig(
    name="analyst",
    model=model,
    system_prompt="You are a data analyst. Analyze information and identify patterns.",
    tools=[Calculator(), PythonREPL()]
))

writer = Agent(config=AgentConfig(
    name="writer",
    model=model,
    system_prompt="You are a professional writer. Create clear, engaging content."
))

# Create orchestrator
orchestrator = MultiAgentOrchestrator()

# Register agents
orchestrator.register_agent(researcher)
orchestrator.register_agent(analyst)
orchestrator.register_agent(writer)`}
        language="python"
        filename="multi_agent_setup.py"
      />

      <h2>Sequential Pattern</h2>
      <p>
        Agents work in sequence, each building on the previous agent's output:
      </p>

      <CodeBlock
        code={`# Create a sequential team
team = orchestrator.create_team(
    name="research_pipeline",
    agents=[researcher, analyst, writer],
    pattern=OrchestrationPattern.SEQUENTIAL
)

# Execute task
result = orchestrator.assign_task(
    task="Research quantum computing trends, analyze the data, and write a report",
    team=team
)

print(result.output)
print(f"Agents involved: {[r['agent'] for r in result.agent_responses]}")
print(f"Total time: {result.execution_time}s")`}
        language="python"
        filename="sequential_pattern.py"
      />

      <h2>Parallel Pattern</h2>
      <p>
        Agents work simultaneously on different aspects of a task:
      </p>

      <CodeBlock
        code={`# Create parallel team
team = orchestrator.create_team(
    name="parallel_research",
    agents=[researcher, analyst, writer],
    pattern=OrchestrationPattern.PARALLEL
)

# Execute - all agents work at the same time
result = orchestrator.assign_task(
    task="""
    Researcher: Find information about AI trends
    Analyst: Analyze market data for tech sector
    Writer: Draft introduction for tech report
    """,
    team=team
)

# Results from all agents — agent_responses is a list of {agent, output, ...} dicts
for entry in result.agent_responses:
    print(f"\\n{entry['agent']}:")
    print(entry.get('output', ''))`}
        language="python"
        filename="parallel_pattern.py"
      />

      <h2>Hierarchical Pattern</h2>
      <p>
        A manager agent coordinates worker agents:
      </p>

      <CodeBlock
        code={`# Create manager agent
manager = Agent(config=AgentConfig(
    name="manager",
    model=model,
    system_prompt="""You are a project manager. Your role is to:
1. Break down complex tasks into subtasks
2. Assign subtasks to appropriate team members
3. Review and synthesize their outputs
4. Ensure quality and completeness"""
))

# Create hierarchical team
team = orchestrator.create_team(
    name="managed_team",
    agents=[researcher, analyst, writer],
    pattern=OrchestrationPattern.HIERARCHICAL,
    manager_agent=manager
)

# Execute - manager coordinates the work
result = orchestrator.assign_task(
    task="Create a comprehensive market analysis report for AI startups",
    team=team
)

# Manager's final synthesized output
print(result.output)`}
        language="python"
        filename="hierarchical_pattern.py"
      />

      <h2>Collaborative Pattern</h2>
      <p>
        Agents discuss and iterate to reach consensus:
      </p>

      <CodeBlock
        code={`# Create collaborative team
team = orchestrator.create_team(
    name="discussion_team",
    agents=[researcher, analyst],
    pattern=OrchestrationPattern.COLLABORATIVE,
    max_rounds=5,                  # max discussion rounds (default 3)
    voting_strategy="majority",    # "majority" | "unanimous" | "weighted"
)

# Execute collaborative task
result = orchestrator.assign_task(
    task="Evaluate whether investing in quantum computing startups is advisable",
    team=team
)

print(f"Rounds: {result.rounds}")
print(f"Consensus score: {result.consensus_score:.2f}")
for entry in result.agent_responses:
    print(f"  {entry.get('agent')}: {str(entry.get('output', ''))[:100]}...")

print(f"\\nFinal consensus: {result.output}")`}
        language="python"
        filename="collaborative_pattern.py"
      />

      <h2>Task Decomposition</h2>
      <p>
        Automatically decompose complex tasks and route to appropriate agents:
      </p>

      <CodeBlock
        code={`from effgen.core.decomposition_engine import DecompositionEngine
from effgen.core.router import SubAgentRouter

# Create decomposition engine
decomposer = DecompositionEngine()

# Analyze task complexity
task = """
Build a data pipeline that:
1. Fetches data from 3 APIs
2. Cleans and transforms the data
3. Performs statistical analysis
4. Generates visualizations
5. Creates a PDF report
"""

analysis = decomposer.analyze_task_structure(task)
print(f"Has data gathering: {analysis.has_data_gathering}")
print(f"Has analysis: {analysis.has_analysis}")
print(f"Has synthesis: {analysis.has_synthesis}")
print(f"Parallelizable: {analysis.parallelizable}")

# Create router
router = SubAgentRouter(orchestrator=orchestrator)

# Route task to agents
routing_decision = router.route(task)
print(f"Strategy: {routing_decision.strategy}")
print(f"Num agents: {routing_decision.num_sub_agents}")
print(f"Specializations: {routing_decision.specializations}")`}
        language="python"
        filename="task_decomposition.py"
      />

      <h2>Dynamic Agent Spawning</h2>
      <p>
        Spawn specialized sub-agents based on task requirements:
      </p>

      <CodeBlock
        code={`from effgen.core.agent import AgentConfig

# Enable sub-agent spawning
config = AgentConfig(
    name="main_agent",
    model=model,
    enable_sub_agents=True,
    max_sub_agent_depth=3,          # bounds recursion depth (default 3)
    sub_agent_config={
        "auto_specialize": True,    # Auto-create specialized agents
        "share_memory": True,       # Share memory with sub-agents
        "inherit_tools": True       # Sub-agents inherit parent's tools
    }
)

main_agent = Agent(config=config)

# For complex tasks, main agent can spawn helpers
result = main_agent.run("""
Perform comprehensive analysis:
1. Research latest AI papers on transformers
2. Analyze code from top 3 implementations
3. Compare performance benchmarks
4. Write technical summary
""")

# See spawned sub-agents
print(f"Sub-agents spawned: {result.metadata.get('sub_agents', [])}")`}
        language="python"
        filename="dynamic_spawning.py"
      />

      <h2>Communication Protocols</h2>

      <CodeBlock
        code={`from effgen.core.message_bus import MessageBus, AgentMessage, MessageType

# Create message bus for agent communication
bus = MessageBus(persist_path="./bus.jsonl")  # optional persistence

# Send messages between agents (mailbox semantics)
bus.send(AgentMessage(
    sender="researcher",
    recipient="analyst",
    msg_type=MessageType.RESULT,
    payload={"findings": [...]},
))

# Broadcast to multiple agents
bus.broadcast(
    sender="manager",
    agent_ids=["researcher", "analyst", "writer"],
    msg_type=MessageType.STATUS_UPDATE,
    payload={"announcement": "Deadline extended"},
)

# Topic-based pub/sub with wildcard subscriptions
bus.subscribe("alerts.*", handler=lambda msg: print("ALERT:", msg.payload))
bus.publish(AgentMessage(
    sender="watchdog",
    topic="alerts.gpu",
    msg_type=MessageType.STATUS_UPDATE,
    payload={"level": "critical", "usage": 0.98},
))`}
        language="python"
        filename="communication.py"
      />

      <h2>Best Practices</h2>

      <FeatureList
        features={[
          { icon: '🎯', title: 'Specialize Agents', description: 'Give each agent a clear, focused role for better results' },
          { icon: '📊', title: 'Choose Right Pattern', description: 'Sequential for pipelines, parallel for independent tasks, hierarchical for coordination' },
          { icon: '🔄', title: 'Limit Iterations', description: 'Set reasonable limits on collaborative rounds to prevent infinite loops' },
          { icon: '💾', title: 'Share Context', description: 'Enable memory sharing for agents working on related tasks' },
          { icon: '📝', title: 'Clear Prompts', description: 'Write clear system prompts defining each agent\'s expertise' },
        ]}
      />

      <InfoBox type="success" title="Next Steps">
        <p>
          Explore <Link to="/protocols">Communication Protocols</Link> for MCP and A2A integration,
          or see <Link to="/examples">Examples</Link> for real-world multi-agent patterns.
        </p>
      </InfoBox>
    </DocPage>
  );
}
