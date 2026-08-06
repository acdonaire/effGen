import React from 'react';
import { Link } from 'react-router-dom';
import { Bot } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';

export default function Agents() {
  const agentLifecycleDiagram = `
stateDiagram-v2
    [*] --> Created: Agent(config)
    Created --> Running: agent.run(task)
    Running --> Thinking: Start ReAct
    Thinking --> Acting: Select action
    Acting --> Observing: Tool result
    Observing --> Thinking: Continue
    Acting --> Complete: Final answer
    Complete --> [*]
    Running --> Error: Exception
    Error --> [*]
`;

  return (
    <DocPage
      title="Agents"
      subtitle="The core building blocks of effGen. Learn how to create, configure, and manage AI agents."
      icon={<Bot size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Core Concepts', path: '/agents' },
        { label: 'Agents' },
      ]}
    >
      <h2>What is an Agent?</h2>
      <p>
        An agent in effGen is an autonomous AI system that combines a language model with tools,
        memory, and reasoning capabilities to accomplish tasks. Agents can:
      </p>
      <ul>
        <li>Reason about problems using the ReAct loop</li>
        <li>Use tools to interact with external systems</li>
        <li>Maintain context through memory systems</li>
        <li>Decompose complex tasks into subtasks</li>
        <li>Spawn sub-agents for specialized work</li>
      </ul>

      <MermaidDiagram chart={agentLifecycleDiagram} title="Agent Lifecycle" />

      <InfoBox type="success" title="New in v0.3.1 — grounded sources, honored personas, and measurable results">
        <p>
          Every run now carries its evidence and cost. <code>response.sources</code> and{' '}
          <code>response.citations</code> are populated from the URLs the run actually retrieved
          (and provider-native grounding) — never scraped from the model&apos;s prose —{' '}
          and <code>response.metadata</code> carries <code>cost_usd</code>, token counts, and{' '}
          <code>latency_ms</code> (local models stay honestly cost-free). A custom{' '}
          <code>system_prompt</code> now steers <em>every</em> path (direct, streaming, and the
          native/hybrid tool path), not just text-ReAct, so a persona is never silently dropped.
          Reasoning models (the <code>gpt-5</code> family, <code>o</code>-series) get a larger
          default output budget so they finish token-heavy tasks instead of returning an empty,
          billed result.
        </p>
      </InfoBox>
      <CodeBlock
        code={`agent = create_agent("research", "openai:gpt-5-nano")
r = agent.run("What is the capital of France? Cite a source.")
print(r.text)                    # "...Paris (Source: https://en.wikipedia.org/wiki/Paris)."
print(r.sources)                 # ['https://en.wikipedia.org/wiki/Paris']
print(r.metadata["cost_usd"], r.metadata["latency_ms"])`}
      />

      <h2>Creating an Agent</h2>
      <p>
        Agents are created using <code>AgentConfig</code> and the <code>Agent</code> class:
      </p>

      <CodeBlock
        code={`from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL

# Load model
model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

# Create configuration
config = AgentConfig(
    name="data_analyst",
    model=model,
    tools=[Calculator(), PythonREPL()],
    system_prompt="""You are an expert data analyst.
You can perform calculations and write Python code for data analysis.
Always explain your reasoning step by step.""",
    temperature=0.3,
    max_iterations=15,
    enable_memory=True,
    enable_sub_agents=True
)

# Create agent
agent = Agent(config=config)

# Run tasks
result = agent.run("Analyze the trend in [1, 2, 4, 8, 16, 32]")`}
        language="python"
        filename="create_agent.py"
      />

      <h2>AgentConfig Reference</h2>
      <p>
        The <code>AgentConfig</code> class accepts the following parameters:
      </p>

      <ApiTable
        headers={['Parameter', 'Type', 'Default', 'Description']}
        rows={[
          [<code>name</code>, 'str', 'Required', 'Unique identifier for the agent'],
          [<code>model</code>, 'BaseModel', 'Required', 'Loaded model instance'],
          [<code>tools</code>, 'List[BaseTool]', '[]', 'Available tools for the agent'],
          [<code>system_prompt</code>, 'str', 'Default prompt', 'System-level instructions'],
          [<code>temperature</code>, 'float', '0.7', 'Generation temperature (0-1)'],
          [<code>max_iterations</code>, 'int', '10', 'Maximum ReAct loop iterations'],
          [<code>max_context_length</code>, 'int | None', 'None', 'Max context tokens (auto-detect from model when None)'],
          [<code>enable_sub_agents</code>, 'bool', 'True', 'Allow spawning sub-agents'],
          [<code>enable_memory</code>, 'bool', 'True', 'Enable memory systems'],
          [<code>enable_streaming</code>, 'bool', 'False', 'Stream output tokens'],
          [<code>router_config</code>, 'dict', '{}', 'Configuration for sub-agent router'],
          [<code>sub_agent_config</code>, 'dict', '{}', 'Configuration for sub-agent manager'],
          [<code>model_config</code>, 'dict | None', 'None', 'Optional model engine configuration'],
          [<code>require_model</code>, 'bool', 'False', 'Whether model loading is required (raise error on failure)'],
          [<code>system_prompt_template</code>, 'str | None', 'None', 'Template for system prompt generation'],
          [<code>verbose_tools</code>, 'bool | None', 'None', 'Enable verbose tool descriptions'],
          [<code>fallback_chain</code>, 'dict | None', 'None', 'Tool fallback chain configuration'],
          [<code>enable_fallback</code>, 'bool', 'True', 'Enable tool fallback chains'],
          [<code>memory_config</code>, 'dict', '{...}', 'Memory system configuration (short-term, long-term settings)'],
        ]}
      />

      <h2>Agent Response</h2>
      <p>
        The <code>AgentResponse</code> object contains comprehensive information about task execution:
      </p>

      <CodeBlock
        code={`result = agent.run("Calculate fibonacci(10)")

# Access response attributes
class AgentResponse:
    output: str              # Final output text
    success: bool            # Whether task completed successfully
    mode: AgentMode          # SINGLE, SUB_AGENTS, or AUTO
    iterations: int          # Number of ReAct iterations
    tool_calls: int          # Number of tool calls made
    tokens_used: int         # Total tokens consumed
    execution_time: float    # Time in seconds
    execution_trace: List    # Detailed step-by-step trace
    execution_tree: dict     # Hierarchical execution tree
    routing_decision: RoutingDecision | None  # Routing decision (if sub-agents used)
    metadata: dict           # Additional metadata

# Example usage
print(f"Output: {result.output}")
print(f"Success: {result.success}")
print(f"Iterations: {result.iterations}")
print(f"Tokens: {result.tokens_used}")
print(f"Tool calls (count): {result.tool_calls}")

# Inspect each tool invocation through the execution_trace list
for step in result.execution_trace:
    if step.get("tool"):
        print(f"Tool: {step['tool']}")
        print(f"Input: {step.get('tool_input')}")
        print(f"Output: {step.get('tool_output')}")`}
        language="python"
        filename="agent_response.py"
      />

      <h2>Execution Modes</h2>

      <p>
        <code>AgentMode</code> picks how an agent runs the task: <code>SINGLE</code> (no
        sub-agents), <code>SUB_AGENTS</code> (force sub-agent decomposition), or <code>AUTO</code>
        (router decides; the default). <code>AgentResponse.mode</code> reflects the mode that
        was actually used.
      </p>

      <h3>Single Mode</h3>
      <p>
        For simple questions, run the agent directly without sub-agent decomposition:
      </p>

      <CodeBlock
        code={`from effgen.core.agent import AgentMode

# Single mode — no sub-agents, no router
result = agent.run("What is the capital of France?", mode=AgentMode.SINGLE)
print(result.mode)        # AgentMode.SINGLE`}
        language="python"
      />

      <h3>Sub-Agents Mode</h3>
      <p>
        For complex tasks, force sub-agent decomposition:
      </p>

      <CodeBlock
        code={`from effgen.core.agent import AgentMode

result = agent.run(
    "Research quantum computing, analyse the trends, and write a report",
    mode=AgentMode.SUB_AGENTS,
)
print(result.mode)         # AgentMode.SUB_AGENTS
print(result.iterations)   # Number of think-act-observe cycles
print(result.routing_decision)  # which sub-agents fired`}
        language="python"
      />

      <h2>Advanced Agent Patterns</h2>

      <h3>Streaming Output</h3>
      <CodeBlock
        code={`config = AgentConfig(
    name="streaming_agent",
    model=model,
    enable_streaming=True
)

agent = Agent(config=config)

# Stream tokens as they're generated
for chunk in agent.stream("Explain quantum computing"):
    print(chunk, end="", flush=True)`}
        language="python"
        filename="streaming.py"
      />

      <h3>Conversation History</h3>
      <CodeBlock
        code={`config = AgentConfig(
    name="conversational_agent",
    model=model,
    enable_memory=True
)

agent = Agent(config=config)

# Multi-turn conversation
agent.run("My name is Alice")
agent.run("I'm interested in machine learning")
result = agent.run("What topics should I study?")
# Agent remembers context from previous turns

# Access conversation history via the short-term memory buffer
for msg in agent.short_term_memory.messages:
    print(f"[{msg.role.value}]: {msg.content}")`}
        language="python"
        filename="conversation.py"
      />

      <h3>Sub-Agent Spawning</h3>
      <CodeBlock
        code={`config = AgentConfig(
    name="manager_agent",
    model=model,
    enable_sub_agents=True
)

agent = Agent(config=config)

# For complex tasks, the agent can spawn sub-agents
result = agent.run("""
Research the latest AI trends, analyze market data,
and create a comprehensive report with visualizations.
""")

# The manager agent may spawn:
# - Research agent for gathering information
# - Analysis agent for data processing
# - Writer agent for report generation`}
        language="python"
        filename="sub_agents.py"
      />

      <h2>Error Handling</h2>

      <p>
        <code>Agent.run()</code> catches internal failures and reports them through the returned
        <code> AgentResponse</code> rather than raising. Inspect <code>success</code>,
        <code> iterations</code>, and <code>metadata</code> to recover gracefully.
      </p>

      <InfoBox type="success" title="Fail-closed behavior (v0.3.0)">
        <p>
          As of v0.3.0, <code>Agent.run()</code> <strong>never returns <code>success=True</code>{' '}
          with empty output</strong>. The direct and tool paths return the <em>same</em> shape on
          failure: <code>success=False</code>, a coarse <code>metadata["reason"]</code> stage label
          (e.g. <code>generation_failed</code>), and a typed redacted{' '}
          <code>metadata["error"]</code> dict. A new <code>classify_provider_error()</code>{' '}
          populates <code>metadata["error"]["category"]</code> with a stable taxonomy —{' '}
          <code>auth</code>, <code>not_found</code>, <code>rate_limited</code>,{' '}
          <code>transient</code>, <code>timeout</code>, <code>fatal</code> — so retries fire
          only when retrying could help (auth and not-found fast-stop with one clear message instead
          of a retry storm). A wrong or 404 model id suggests the nearest live alternative. Set{' '}
          <code>AgentConfig(raise_on_error=True)</code> to opt into exceptions instead.
        </p>
      </InfoBox>
      <CodeBlock
        code={`result = agent.run("What is 24344 * 334?")

if not result.success:
    # Consistent across the direct and tool paths
    err = result.metadata["error"]            # {type, category, provider, model, message, ...}
    category = err["category"]                # auth | not_found | rate_limited | transient | timeout | fatal
    stage = result.metadata["reason"]         # coarse stage, e.g. "generation_failed" / "run_failed"
    print("Failed:", category, "-", err["message"])
else:
    print(result.output)

# Or opt into exceptions
from effgen.core.agent import AgentConfig
agent = Agent(config=AgentConfig(name="strict", model=model, raise_on_error=True))`}
        language="python"
        filename="v030_fail_closed.py"
      />

      <CodeBlock
        code={`result = agent.run("Complex task here")

# Did the agent succeed?
if not result.success:
    print("Failed:", result.metadata.get("error") or result.output)

# Did it hit the iteration cap?
if result.iterations >= agent.config.max_iterations:
    print("Hit max_iterations — partial output:", result.output)

# Was it blocked by a guardrail?
if result.metadata.get("guardrail_blocked"):
    print("Blocked:", result.metadata["guardrail_reason"])

# Per-tool errors are visible in execution_trace
for step in result.execution_trace:
    if step.get("tool_error"):
        print("Tool", step["tool"], "failed:", step["tool_error"])`}
        language="python"
        filename="error_handling.py"
      />

      <p>
        For typed exceptions when calling the API server, see
        {' '}<Link to="/clients">Clients &amp; SDKs</Link> — the Python client raises
        <code> EffGenAPIError</code>, <code>EffGenAuthError</code>, <code>EffGenRateLimitError</code>,
        <code> EffGenTimeoutError</code>, and friends from <code>effgen.client</code>.
      </p>

      <h2>Agent Presets</h2>
      <p>
        Create agents with a single line using built-in presets. Each preset comes with optimized
        tool selections for common use cases:
      </p>

      <CodeBlock
        code={`from effgen import load_model
from effgen.presets import create_agent

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# One-line agent creation
agent = create_agent("math", model)
result = agent.run("What is 24344 * 334?")
print(result.output)  # 8,130,896

# Available presets (9 total):
# "math"     → Calculator + PythonREPL
# "research" → WebSearch + URLFetch + Wikipedia + ArXiv + PubMed + SemanticScholar
#               + RSSFeed + News + YouTubeTranscript + YouTubeMetadata + Reddit + HackerNews
#               + PDF + DOCX + Excel (v0.2.6 adds documents)
# "coding"   → CodeExecutor + PythonREPL + FileOps + Bash
# "general"  → 31 tools (calc, repl, search, code, file, json, datetime, text, url, wiki,
#               rss, news, reddit, hackernews, translate, language_detect, qr_*, ocr,
#               audio_transcribe, image_info, pdf, docx, excel, weather, geocode, maps,
#               email_smtp, email_imap, slack_webhook, discord_webhook)
# "rag"      → Retrieval over a knowledge_base directory (v0.2.0)
# "media"    → AudioTranscribeTool + ImageCaptionTool (v0.2.6)
# "notify"   → EmailSMTPTool + EmailIMAPTool + SlackWebhookTool + DiscordWebhookTool (v0.2.6)
# "multimodal" → MultimodalDescribe + ImageInfo + ImageCaption + OCR + AudioTranscribe + PDF + Weather (v0.2.8)
# "minimal"  → Direct inference, no tools`}
        language="python"
        filename="presets.py"
      />

      <CodeBlock
        code={`from effgen.presets import list_presets

# List all available presets (returns name → description dict)
for name, description in list_presets().items():
    print(f"{name}: {description}")`}
        language="python"
        filename="list_presets.py"
      />

      <h2>Construction Ergonomics &amp; Input Robustness (v0.3.0)</h2>
      <p>
        v0.3.0 makes the obvious calls just work — every addition is an additive alias, so existing
        code is unchanged. <code>create_agent(preset, model, name="X")</code> accepts a name;{' '}
        <code>Agent.run()</code> accepts <code>str | Message | list[ContentPart]</code> plus a
        first-class <code>inputs=</code> kwarg for media (and raises a clear <code>TypeError</code>{' '}
        otherwise). A bare or invalid constructor now raises a clear accepted-kwargs error instead
        of a cryptic dataclass <code>TypeError</code>. Related conveniences: <code>TemplateManager()</code>{' '}
        is populated by default, <code>ConfigLoader.load</code>, <code>ShortTermMemory.get_messages</code>,{' '}
        <code>TestCase(input=, expected=)</code>, and a <code>@tool</code> / <code>Tool.from_function()</code>{' '}
        helper for turning a plain function into a tool.
      </p>
      <CodeBlock
        code={`from effgen import Agent, load_model, image_from
from effgen.presets import create_agent
from effgen.tools import tool

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# create_agent accepts name=
agent = create_agent("general", model, name="assistant")

# run() accepts str | Message | list[ContentPart], plus inputs= for media
agent.run("Summarize this page")
agent.run(inputs=[image_from("/tmp/chart.png"), "What trend does this show?"])

# Turn a plain function into a tool
@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b`}
        language="python"
        filename="v030_ergonomics.py"
      />

      <h2>Streaming</h2>
      <p>
        effGen provides true token-by-token streaming with callbacks for thoughts, tool calls,
        observations, and answers:
      </p>

      <CodeBlock
        code={`from effgen import load_model
from effgen.presets import create_agent

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = create_agent("math", model)

# Token-by-token streaming
for chunk in agent.stream("Explain the Pythagorean theorem"):
    print(chunk, end="", flush=True)

# Streaming with callbacks
def on_thought(text):
    print(f"[Thinking] {text}")

def on_tool_call(tool_name, args):
    print(f"[Tool] {tool_name}({args})")

def on_observation(result):
    print(f"[Observe] {result}")

def on_answer(text):
    print(f"[Answer] {text}")

agent.stream(
    "Calculate sqrt(144) + 8",
    on_thought=on_thought,
    on_tool_call=on_tool_call,
    on_observation=on_observation,
    on_answer=on_answer
)`}
        language="python"
        filename="streaming_callbacks.py"
      />

      <h2>Async Support</h2>
      <p>
        effGen supports native async for non-blocking agent execution:
      </p>

      <CodeBlock
        code={`import asyncio
from effgen import load_model
from effgen.presets import create_agent

async def main():
    model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

    agent = create_agent("research", model)

    # Native async execution
    result = await agent.run_async("What are the latest AI trends?")
    print(result.output)

    # Run multiple tasks concurrently
    tasks = [
        agent.run_async("Research quantum computing"),
        agent.run_async("Research climate change"),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r.output)

asyncio.run(main())`}
        language="python"
        filename="async_support.py"
      />

      <h2>Best Practices</h2>

      <FeatureList
        features={[
          {
            icon: '🎯',
            title: 'Clear System Prompts',
            description: 'Write specific, concise system prompts that define the agent\'s role and capabilities',
          },
          {
            icon: '🌡️',
            title: 'Appropriate Temperature',
            description: 'Use low temperature (0.1-0.3) for factual tasks, higher (0.7-0.9) for creative tasks',
          },
          {
            icon: '🔄',
            title: 'Reasonable Iterations',
            description: 'Set max_iterations based on task complexity. 5-10 for simple, 15-20 for complex tasks',
          },
          {
            icon: '🔧',
            title: 'Minimal Tools',
            description: 'Only provide tools the agent actually needs. Too many tools can confuse smaller models',
          },
          {
            icon: '💾',
            title: 'Enable Memory When Needed',
            description: 'Use memory for multi-turn conversations or when context persistence is important',
          },
        ]}
      />

      <InfoBox type="success" title="Next Steps">
        <p>
          Learn about the <Link to="/models">Model system</Link> to understand different backends,
          or explore <Link to="/tools">Tools</Link> to extend your agent's capabilities.
        </p>
      </InfoBox>
    </DocPage>
  );
}
