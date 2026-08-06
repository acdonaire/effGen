import React from 'react';
import { Link } from 'react-router-dom';
import { Rocket } from 'lucide-react';
import DocPage, { InfoBox, QuickLinks } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';

export default function QuickStart() {
  const reactLoopDiagram = `
flowchart LR
    T["💭 Think"] --> A["⚡ Act"]
    A --> O["👁️ Observe"]
    O --> T
    A --> F["✅ Final Answer"]

    style T fill:#00c96e,color:#fff
    style A fill:#009950,color:#fff
    style O fill:#00b8d4,color:#fff
    style F fill:#22c55e,color:#fff
`;

  return (
    <DocPage
      title="Quick Start"
      subtitle="Build your first AI agent in minutes. This guide walks you through creating, configuring, and running agents with effGen."
      icon={<Rocket size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Quick Start' },
      ]}
    >
      <h2>Your First Agent</h2>
      <p>
        Let's create a simple agent that can perform calculations. This demonstrates the core concepts
        of model loading, agent creation, and task execution.
      </p>

      <CodeBlock
        code={`from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Step 1: Load a model
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    engine="transformers",
    quantization="4bit"  # Use 4-bit quantization for efficiency
)

# Step 2: Create agent configuration
config = AgentConfig(
    name="calculator_agent",
    model=model,
    tools=[Calculator()],
    system_prompt="You are a helpful math assistant.",
    temperature=0.1,
    max_iterations=5
)

# Step 3: Create and run the agent
agent = Agent(config=config)
result = agent.run("What is 25 * 17?")

# Step 4: Access the results
print(f"Answer: {result.output}")
print(f"Success: {result.success}")
print(f"Mode: {result.mode}")
print(f"Tool calls: {result.tool_calls}")`}
        language="python"
        filename="first_agent.py"
      />

      <InfoBox type="success" title="New in v0.3.0">
        <p>
          <code>import effgen</code> is now effectively instant (~7.5 s → ~20 ms) thanks to lazy
          loading, so a bare import or <code>--version</code> check no longer pulls in torch. Also,{' '}
          <code>Agent.run()</code> <strong>never returns <code>success=True</code> with empty
          output</strong> — on failure you get <code>success=False</code> with a typed{' '}
          <code>result.metadata["error"]["category"]</code> taxonomy (<code>auth</code> /{' '}
          <code>not_found</code> / <code>rate_limited</code> / <code>transient</code> /{' '}
          <code>timeout</code> / <code>fatal</code>) plus a coarse{' '}
          <code>result.metadata["reason"]</code> stage label, so always branch on{' '}
          <code>result.success</code>. See <Link to="/agents">Agents → Error Handling</Link>.
        </p>
      </InfoBox>

      <InfoBox type="success" title="New in v0.3.1">
        <p>
          Every result now carries its <strong>evidence and its cost</strong>:{' '}
          <code>result.sources</code> / <code>result.citations</code> are filled from the URLs the
          run actually retrieved (never the model's prose), and{' '}
          <code>result.metadata</code> carries <code>cost_usd</code>, token counts, and{' '}
          <code>latency_ms</code>. Reasoning models (the <code>gpt-5</code> family,{' '}
          <code>o</code>-series) now finish token-heavy tasks instead of returning empty, billed
          output. A knowledge domain becomes a runnable agent in one call —{' '}
          <code>LegalDomain().to_agent("openai:gpt-5-nano")</code> — and{' '}
          <code>create_agent(extra_tools=["calculator"])</code> accepts tool <strong>name</strong>{' '}
          strings (with "did you mean" on a typo). For CI, <code>effgen run --json -q</code> emits a
          pure-JSON result document to stdout for piping to <code>jq</code>.
        </p>
      </InfoBox>

      <InfoBox type="info" title="How It Works">
        <p>
          The agent uses the <strong>ReAct loop</strong> (Reasoning + Acting) to solve tasks:
        </p>
      </InfoBox>

      <MermaidDiagram chart={reactLoopDiagram} title="ReAct Loop" />

      <h2>Quick Start with Presets</h2>
      <p>
        The fastest way to create an agent is with the preset API — one-line agent creation with
        optimized tool configurations:
      </p>

      <CodeBlock
        code={`from effgen import load_model
from effgen.presets import create_agent

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# One-line agent creation with presets (9 built-in)
math_agent = create_agent("math", model)        # Calculator + PythonREPL
research_agent = create_agent("research", model)  # 15 research tools (web, academic, news, social, video)
coding_agent = create_agent("coding", model)      # CodeExecutor + PythonREPL + FileOps + Bash
general_agent = create_agent("general", model)    # 31 general-purpose tools
rag_agent = create_agent("rag", model, knowledge_base="./docs/")  # v0.2.0 RAG preset
media_agent = create_agent("media", model)        # v0.2.6 — audio transcription + vision captioning
notify_agent = create_agent("notify", model)      # v0.2.6 — email + Slack + Discord
multimodal_agent = create_agent("multimodal", model)  # v0.2.8 — image / audio / video
minimal_agent = create_agent("minimal", model)    # No tools, direct inference

result = math_agent.run("What is 24344 * 334?")
print(result.output)  # 8,130,896

# RAG preset auto-ingests docs, retrieves with hybrid search, returns citations:
answer = rag_agent.run("How do I configure guardrails?")
print(answer.output)
for c in answer.citations:
    print(f"  [{c.index}] {c.source}  (score={c.relevance_score:.3f})")`}
        language="python"
        filename="preset_quickstart.py"
      />

      <h2>CLI Quick Start</h2>
      <p>
        effGen also provides a command-line interface for quick tasks:
      </p>

      <CodeBlock
        code={`# Run a one-off task
effgen run "What is the capital of France?"

# Use a specific preset
effgen run --preset math "What is sqrt(144)?"

# Start interactive chat mode
effgen chat

# v0.2.3+: check cloud provider API keys
effgen doctor

# Start the API server
effgen serve --port 8000`}
        language="bash"
        filename="terminal"
      />

      <h2>Agent with Multiple Tools</h2>
      <p>
        Agents become more powerful when equipped with multiple tools. Here's an agent that can
        do math and execute Python code:
      </p>

      <CodeBlock
        code={`from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# Create agent with multiple tools
config = AgentConfig(
    name="multi_tool_agent",
    model=model,
    tools=[Calculator(), PythonREPL()],
    system_prompt="""You are a helpful assistant with math and coding capabilities.
Use the calculator for mathematical operations.
Use Python REPL for complex computations or data processing.""",
    temperature=0.1,
    max_iterations=10
)

agent = Agent(config=config)

# Complex task requiring multiple tools
result = agent.run("""
Calculate the square root of 144, then write a Python function
that generates the first 10 Fibonacci numbers.
""")

print(result.output)`}
        language="python"
        filename="multi_tool_agent.py"
      />

      <h2>Using Different Model Backends</h2>

      <h3>vLLM (Production)</h3>
      <p>
        For production deployments, use vLLM for maximum performance:
      </p>

      <CodeBlock
        code={`from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, CodeExecutor

# Load model with vLLM (5-10x faster)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="vllm",
    tensor_parallel_size=2  # Use 2 GPUs
)

config = AgentConfig(
    name="production_agent",
    model=model,
    tools=[Calculator(), CodeExecutor()],
    enable_streaming=True
)

agent = Agent(config=config)`}
        language="python"
        filename="vllm_agent.py"
      />

      <h3>OpenAI API</h3>
      <CodeBlock
        code={`import os
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, WebSearch
from effgen.models.base import GenerationConfig

model = load_model(
    "gpt-5.4-nano",
    provider="openai",
    api_key=os.environ.get("OPENAI_API_KEY")
)

config = AgentConfig(
    name="openai_reasoning_agent",
    model=model,
    tools=[Calculator(), WebSearch()]
)

agent = Agent(config=config)`}
        language="python"
        filename="openai_agent.py"
      />

      <h3>Apple Silicon (MLX)</h3>
      <CodeBlock
        code={`from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Auto-routes to MLXEngine on darwin/arm64
model = load_model("mlx-community/Qwen2.5-3B-Instruct-4bit", engine="mlx")

config = AgentConfig(name="mlx_agent", model=model, tools=[Calculator()])
agent = Agent(config=config)`}
        language="python"
        filename="mlx_agent.py"
      />

      <h3>Gemini Thinking, Grounding, and Files (v0.2.2)</h3>
      <CodeBlock
        code={`import os
from pathlib import Path
from effgen import load_model
from effgen.models import GenerationConfig, upload_file

Path("brief.txt").write_text("Summarize the v0.2.2 release note.")
doc = upload_file("brief.txt", api_key=os.environ.get("GOOGLE_API_KEY"))

model = load_model(
    "gemini-2.5-pro",
    provider="gemini",
    api_key=os.environ.get("GOOGLE_API_KEY")
)

result = model.generate(
    "Use this file and Google Search grounding to summarize the release.",
    config=GenerationConfig(thinking_budget=4096, include_thoughts=True, grounding=True),
    files=[doc],
)
print(result.metadata.get("thinking"))
print(result.metadata.get("grounding_chunks"))`}
        language="python"
        filename="gemini_v022.py"
      />

      <h3>Anthropic Thinking and Prompt Cache (v0.2.2)</h3>
      <CodeBlock
        code={`import os
from effgen import load_model
from effgen.models import GenerationConfig, mark_cached

model = load_model(
    "claude-sonnet-4-6",
    provider="anthropic",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

long_context = "Reusable architecture policy or repository summary."
result = model.generate(
    "Reason about migration risks.",
    system_prompt=[mark_cached({"type": "text", "text": long_context}, ttl="1h")],
    config=GenerationConfig(thinking={"type": "enabled", "budget_tokens": 4096}),
)
print(result.metadata.get("cached_input_tokens"))
print(result.metadata.get("cache_creation_tokens"))`}
        language="python"
        filename="anthropic_v022.py"
      />

      <h3>Cerebras / Groq / Provider Prefixes</h3>
      <CodeBlock
        code={`from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import Calculator

# v0.2.1 Cerebras
cerebras = load_model("llama3.1-8b", provider="cerebras")

# v0.2.3 provider-prefixed model ID
# groq = load_model("groq:llama-3.3-70b-versatile")

agent = Agent(AgentConfig(
    name="provider_agent",
    model=cerebras,
    tools=[Calculator()],
    tool_calling_mode="hybrid",
))
result = agent.run("What is (17 * 23) + sqrt(144)?")
print(result.output)  # 403`}
        language="python"
        filename="provider_agent.py"
      />

      <h3>Policy-Based ModelRouter (v0.2.4+)</h3>
      <p>
        v0.2.4 introduces an opt-in router that picks the best
        <code> (provider, model_id)</code> pair for a request and transparently
        fails over on retriable errors. Compose
        <code> FirstAvailablePolicy</code>, <code>CostBasedPolicy</code>, or
        <code> LatencyBasedPolicy</code>:
      </p>
      <CodeBlock
        code={`# v0.2.4+
import effgen.models  # registers all 9 cloud adapters
from effgen import (
    PolicyBasedRouter, RoutingContext,
    CostBasedPolicy, LatencyBasedPolicy,
    load_model,
)
from effgen.models.capabilities import Capability

router = PolicyBasedRouter(
    policies=[LatencyBasedPolicy(), CostBasedPolicy()],
    failover_hops=3,
)

context = RoutingContext(
    prompt_tokens_estimate=500,
    user_budget_usd=0.01,
    latency_budget_ms=3000,
    required_capabilities={Capability.chat, Capability.tools},
)

def call_model(pair):
    return load_model(f"{pair.provider}:{pair.model_id}").generate("Hello")

answer = router.route_and_execute(context, call_model)`}
        language="python"
        filename="v024_router_quickstart.py"
      />

      <h2>Understanding Agent Responses</h2>
      <p>
        The <code>AgentResponse</code> object contains detailed information about the execution:
      </p>

      <CodeBlock
        code={`result = agent.run("Complex task here")

# Basic information
print(f"Output: {result.output}")
print(f"Success: {result.success}")

# Execution details
print(f"Mode: {result.mode}")          # AgentMode.SINGLE | SUB_AGENTS | AUTO
print(f"Iterations: {result.iterations}")
print(f"Execution time: {result.execution_time}s")

# Tool usage — tool_calls is the COUNT of tool invocations
print(f"Tool calls: {result.tool_calls}")

# Token usage
print(f"Tokens used: {result.tokens_used}")

# Detailed execution trace — list of per-step dicts
for step in result.execution_trace:
    if step.get("tool"):
        print(f"  - {step['tool']}: {step.get('tool_input')} -> {step.get('tool_output')}")
    if step.get("thought"):
        print(f"Step {step.get('iteration')}: {step['thought']}")`}
        language="python"
        filename="response_details.py"
      />

      <h2>Configuration Options</h2>
      <p>
        <code>AgentConfig</code> provides many options to customize agent behavior:
      </p>

      <CodeBlock
        code={`from effgen.core.agent import AgentConfig

config = AgentConfig(
    # Required
    name="my_agent",
    model=model,

    # Tools
    tools=[Calculator(), PythonREPL()],

    # Prompts
    system_prompt="You are a helpful assistant.",

    # Generation settings
    temperature=0.7,         # Higher = more creative
    max_iterations=10,       # Max ReAct loop iterations
    max_context_length=4096, # Max context window

    # Advanced features
    enable_sub_agents=True,  # Allow spawning sub-agents
    enable_memory=True,      # Enable memory systems
    enable_streaming=False,  # Stream output tokens
)`}
        language="python"
        filename="config_options.py"
      />

      <h2>Creating Custom Tools</h2>
      <p>
        Extend your agent's capabilities by creating custom tools:
      </p>

      <CodeBlock
        code={`from effgen import Agent, AgentConfig, load_model
from effgen.tools import BaseTool, ParameterSpec, ParameterType, ToolCategory, ToolMetadata
from effgen.tools.builtin import Calculator

class WeatherTool(BaseTool):
    """Get current weather for a location."""

    def __init__(self):
        super().__init__(metadata=ToolMetadata(
            name="weather",
            description="Get current weather information for a city",
            category=ToolCategory.EXTERNAL_API,
            parameters=[
                ParameterSpec(
                    name="city",
                    type=ParameterType.STRING,
                    description="City name",
                    required=True,
                ),
                ParameterSpec(
                    name="units",
                    type=ParameterType.STRING,
                    description="Temperature units",
                    required=False,
                    default="celsius",
                    enum=["celsius", "fahrenheit"],
                ),
            ],
        ))

    async def _execute(self, city: str, units: str = "celsius") -> dict:
        # Your weather API logic here
        # This is a mock implementation
        return {
            "city": city,
            "temperature": 22,
            "units": units,
            "condition": "Sunny",
            "humidity": 45
        }

# Use custom tool
model = load_model("Qwen/Qwen2.5-7B-Instruct")
weather_tool = WeatherTool()
config = AgentConfig(
    name="weather_agent",
    model=model,
    tools=[weather_tool, Calculator()]
)

agent = Agent(config=config)
result = agent.run("What's the weather in Tokyo?")`}
        language="python"
        filename="custom_tool.py"
      />

      <h2>Guardrails (v0.2.0)</h2>
      <p>
        Offline, ML-free input/output validation. Wire in a preset or a custom
        <code>GuardrailChain</code>:
      </p>
      <CodeBlock
        code={`from effgen.core.agent import AgentConfig
from effgen.guardrails import get_guardrail_preset, PIIGuardrail, PromptInjectionGuardrail, GuardrailChain

# Option A — preset
config = AgentConfig(
    name="safe_agent",
    model=model,
    tools=[...],
    guardrails=get_guardrail_preset("standard"),   # "strict" | "standard" | "minimal" | "none"
)

# Option B — custom chain
chain = GuardrailChain([
    PromptInjectionGuardrail(sensitivity="medium"),
    PIIGuardrail(action="redact"),   # "block" | "redact"
])
config = AgentConfig(name="custom_safe", model=model, guardrails=chain)`}
        language="python"
        filename="guardrails_quickstart.py"
      />

      <h2>Structured Output (v0.2.0-v0.2.1)</h2>
      <CodeBlock
        code={`from pydantic import BaseModel

class Answer(BaseModel):
    result: float
    explanation: str

# Via schema
result = agent.run(
    "What is 25 * 17?",
    output_schema={
        "type": "object",
        "properties": {"result": {"type": "number"}, "explanation": {"type": "string"}},
        "required": ["result"],
    },
)

# Via Pydantic model — result.metadata["parsed"] will be an Answer instance
result = agent.run("What is 25 * 17?", output_model=Answer)`}
        language="python"
        filename="structured_output.py"
      />
      <p>
        v0.2.1 also adds OpenAI structured outputs v2 for strict provider-enforced JSON.
      </p>
      <CodeBlock
        code={`from typing import Literal
from pydantic import BaseModel
from effgen.models.errors import ModelRefusalError
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.openai_schema import to_openai_schema

class Classification(BaseModel):
    label: Literal["bug", "feature", "question"]
    confidence: float

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "Classification",
        "schema": to_openai_schema(Classification),
        "strict": True,
    },
}

model = OpenAIAdapter(model_name="gpt-5.4-nano")
model.load()

try:
    result = model.generate_structured(
        "Classify this ticket: Export button fails on Safari.",
        response_format=response_format,
    )
    print(result.text)
except ModelRefusalError as exc:
    print("Structured output refused:", exc)`}
        language="python"
        filename="openai_structured_output.py"
      />

      <h2>Checkpointing &amp; Resume (v0.2.0)</h2>
      <CodeBlock
        code={`# Save every 3 iterations
result = agent.run("Long multi-step research task", checkpoint_interval=3)

# Later — maybe a different process:
resumed = agent.resume(checkpoint_id=result.metadata["checkpoint_id"])
# Or pass nothing to load the latest checkpoint in ./checkpoints
# resumed = agent.resume()`}
        language="python"
        filename="checkpoint_quickstart.py"
      />

      <h2>Production API Server (v0.2.0)</h2>
      <p>
        The API server exposes OpenAI-compatible endpoints. Point any OpenAI SDK at it.
      </p>
      <CodeBlock
        code={`# Start the server
effgen serve --host 0.0.0.0 --port 8000

# Use any OpenAI client against it:
#   POST /v1/chat/completions  (with tools, stream=True)
#   POST /v1/completions
#   POST /v1/embeddings`}
        language="bash"
        filename="terminal"
      />
      <CodeBlock
        code={`from effgen.client import EffGenClient

client = EffGenClient(base_url="http://localhost:8000", api_key="sk-...")

# Sync
print(client.chat("What is 2+2?", tools=["calculator"]).content)

# Async streaming
import asyncio
async def main():
    async for chunk in client.chat_stream("Tell me a story"):
        print(chunk, end="")
asyncio.run(main())`}
        language="python"
        filename="client_quickstart.py"
      />

      <h2>CLI Additions (v0.2.0-v0.2.4)</h2>
      <CodeBlock
        code={`# Evaluation
effgen eval --suite math --model Qwen/Qwen2.5-3B-Instruct
effgen compare --models "Qwen/Qwen2.5-3B-Instruct,microsoft/Phi-3.5-mini-instruct" --suite reasoning

# Workflows (YAML-defined)
effgen workflow run my_pipeline.yaml
effgen workflow validate my_pipeline.yaml

# Checkpoints & sessions
effgen run --checkpoint-dir ./checkpoints --checkpoint-interval 3
effgen resume --checkpoint <id>
effgen sessions list | delete | export | cleanup

# Batch
effgen batch --input tasks.jsonl --output results.jsonl --concurrency 8

# Model pool
effgen models list
effgen models info Qwen/Qwen2.5-3B-Instruct
effgen models load Qwen/Qwen2.5-3B-Instruct
effgen models unload Qwen/Qwen2.5-3B-Instruct
effgen models status

# Config, tools, examples, and plugins
effgen config show
effgen config validate config.yaml
effgen config init
effgen tools list
effgen tools info calculator
effgen tools test calculator
effgen examples list
effgen examples run basic_agent
effgen presets
effgen health
effgen create-plugin my-tool

# Debug
effgen debug "What is sqrt(144)?"          # rich step-through TUI

# Provider registry / auth readiness (v0.2.3+)
effgen doctor
effgen doctor --provider groq
effgen doctor --json

# Persistent cost dashboard + budgets (v0.2.4+)
effgen cost today
effgen cost week
effgen cost by-provider
effgen cost set-budget 1.0
effgen cost clear-budget`}
        language="bash"
        filename="terminal"
      />

      <h2>Next Steps</h2>
      <p>
        Now that you've created your first agents, explore more advanced features:
      </p>

      <QuickLinks
        links={[
          {
            icon: '🛡️',
            title: 'Guardrails',
            description: 'Toxicity, PII, injection, and tool safety',
            path: '/guardrails',
          },
          {
            icon: '📖',
            title: 'RAG Pipeline',
            description: 'Hybrid search, rerankers, and citations',
            path: '/rag',
          },
          {
            icon: '🕸️',
            title: 'DAG Workflows',
            description: 'Multi-agent pipelines with auto-parallel',
            path: '/workflows',
          },
          {
            icon: '📊',
            title: 'Evaluation',
            description: 'Test suites, LLM-judge, regression tracking',
            path: '/evaluation',
          },
          {
            icon: 'P',
            title: 'Providers',
            description: 'ProviderRegistry, doctor, and backend parity',
            path: '/providers',
          },
          {
            icon: 'T',
            title: 'Native Provider Tools',
            description: 'OpenAI, Gemini, and Anthropic server-side tools',
            path: '/native-provider-tools',
          },
        ]}
      />

      <InfoBox type="success" title="Congratulations!">
        <p>
          You've created your first effGen agent! Check out the <Link to="/examples">Examples</Link> page
          for more real-world use cases, or dive into the <Link to="/agents">Core Concepts</Link> for
          deeper understanding.
        </p>
      </InfoBox>
    </DocPage>
  );
}
