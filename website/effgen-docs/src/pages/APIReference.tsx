import React, { useState, useRef } from 'react';
import { Code, Bot, Cpu, Wrench, Brain, Network, ChevronRight, Search, Server, Layers, Plug, GitBranch } from 'lucide-react';
import PageNavbar from '../components/PageNavbar';
import CodeBlock from '../components/CodeBlock';
import { ApiTable } from '../components/DocPage';
import './APIReference.css';

interface ApiSection {
  id: string;
  title: string;
  icon: React.ReactNode;
  content: React.ReactNode;
}

const apiSections: ApiSection[] = [
  {
    id: 'agent',
    title: 'Agent',
    icon: <Bot size={20} />,
    content: (
      <>
        <h3>Agent Class</h3>
        <p>The main agent class for executing tasks.</p>
        <CodeBlock
          code={`class Agent:
    """
    Main agent executor for running tasks with tools and memory.

    Attributes:
        config: AgentConfig - Agent configuration
        name: str - Agent name
        model: BaseModel - Loaded model instance
        tools: List[BaseTool] - Available tools
        history: List[Message] - Conversation history
    """

    def __init__(self, config: AgentConfig, session_id: str | None = None) -> None:
        """Initialize agent with configuration. Pass session_id to load/persist
        a multi-turn conversation under ~/.effgen/sessions/."""

    def run(
        self,
        task: str,
        mode: AgentMode = AgentMode.AUTO,
        context: dict | None = None,
        output_schema: dict | None = None,
        output_model: type | None = None,      # Pydantic model
        **kwargs,                              # debug, checkpoint_interval, checkpoint_dir, ...
    ) -> AgentResponse:
        """
        Execute a task and return the response.

        Args:
            task: The task description to execute
            mode: AgentMode.SINGLE | SUB_AGENTS | AUTO
            context: Optional shared context dict
            output_schema: JSON Schema to validate output against
            output_model: Pydantic model; output parsed into this type
            kwargs: debug=True (DebugTrace), checkpoint_interval=N, checkpoint_dir="..."

        Returns:
            AgentResponse with output, success status, citations, and metadata
        """

    def resume(
        self,
        checkpoint_id: str | None = None,
        checkpoint_dir: str = "./checkpoints",
        **kwargs,
    ) -> AgentResponse:
        """Resume a previously checkpointed run (or the latest if id omitted)."""

    def run_batch(
        self,
        queries: list[str],
        max_concurrency: int = 5,
        batch_size: int = 0,
        retry_failed: int = 1,
        timeout_per_item: float = 120.0,
        **run_kwargs,
    ) -> BatchResult:
        """Concurrent batch execution via BatchRunner.
        Returns BatchResult with .results (List[AgentResponse]), .succeeded, .failed,
        .total_time, .per_query_times."""

    def run_background(self, task: str, **kwargs) -> str:
        """Submit task to BackgroundTaskRunner; returns task_id."""

    def close(self) -> None:
        """Release resources (model, memory, caches). Also available via 'with' / 'async with'."""

    async def run_async(
        self,
        task: str,
        mode: AgentMode = AgentMode.AUTO,
        context: dict | None = None,
        **kwargs,
    ) -> AgentResponse:
        """Async wrapper around run() that runs in a thread executor."""

    def stream(
        self,
        task: str,
        mode: AgentMode = AgentMode.AUTO,
        context: dict | None = None,
        on_thought: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
        on_observation: Callable[[str], None] | None = None,
        on_answer: Callable[[str], None] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream the ReAct loop in real-time, yielding tokens as they are generated."""

    def get_task_status(self, task_id: str): ...
    def get_task_result(self, task_id: str, wait: bool = False, timeout: float | None = None): ...
    def cancel_task(self, task_id: str) -> bool: ...`}
          language="python"
          filename="agent_api.py"
        />

        <h4>AgentConfig</h4>
        <ApiTable
          headers={['Parameter', 'Type', 'Required', 'Default', 'Description']}
          rows={[
            [<code>name</code>, 'str', 'Yes', '-', 'Unique agent identifier'],
            [<code>model</code>, 'BaseModel | str', 'Yes', '-', 'Loaded model instance, or model name string (auto-loaded)'],
            [<code>tools</code>, 'List[BaseTool]', 'No', '[]', 'Available tools'],
            [<code>system_prompt</code>, 'str', 'No', 'default', 'System instructions'],
            [<code>temperature</code>, 'float', 'No', '0.7', 'Generation temperature'],
            [<code>max_iterations</code>, 'int', 'No', '10', 'Max ReAct iterations'],
            [<code>max_context_length</code>, 'int | None', 'No', 'None', 'Max context tokens (auto-detect from model when None)'],
            [<code>enable_sub_agents</code>, 'bool', 'No', 'True', 'Allow spawning sub-agents'],
            [<code>enable_memory</code>, 'bool', 'No', 'True', 'Enable memory'],
            [<code>enable_streaming</code>, 'bool', 'No', 'False', 'Enable streaming'],
            [<code>router_config</code>, 'dict', 'No', '{}', 'Sub-agent router config'],
            [<code>sub_agent_config</code>, 'dict', 'No', '{}', 'Sub-agent manager config'],
            [<code>model_config</code>, 'dict | None', 'No', 'None', 'Model engine config overrides'],
            [<code>require_model</code>, 'bool', 'No', 'False', 'Raise error on model load failure'],
            [<code>system_prompt_template</code>, 'str | None', 'No', 'None', 'Template for system prompt'],
            [<code>verbose_tools</code>, 'bool | None', 'No', 'None', 'Verbose tool descriptions'],
            [<code>fallback_chain</code>, 'dict | None', 'No', 'None', 'Tool fallback chain config'],
            [<code>enable_fallback</code>, 'bool', 'No', 'True', 'Enable tool fallback chains'],
            [<code>memory_config</code>, 'dict', 'No', '{...}', 'Memory system configuration'],
            [<code>max_sub_agent_depth</code>, 'int', 'No', '3', 'Max sub-agent recursion depth'],
            [<code>tool_calling_mode</code>, 'str', 'No', '"auto"', '"auto" | "native" | "react" | "hybrid"'],
            [<code>output_format</code>, 'str | None', 'No', 'None', '"json" | "yaml" | "csv" | None'],
            [<code>output_schema</code>, 'dict | None', 'No', 'None', 'JSON Schema for structured output'],
            [<code>guardrails</code>, 'GuardrailChain | str | None', 'No', 'None', 'Chain, preset name, or None'],
            [<code>models</code>, 'list | None', 'No', 'None', 'Additional models for routing'],
            [<code>speculative_execution</code>, 'bool', 'No', 'False', 'Run on 2 models, first success wins'],
            [<code>approval_mode</code>, 'str', 'No', '"never"', '"always" | "first_time" | "never" | "dangerous_only"'],
            [<code>approval_callback</code>, 'Callable | None', 'No', 'None', '(tool_name, tool_args) -> bool'],
            [<code>approval_timeout</code>, 'float', 'No', '0.0', 'Seconds (0 = wait forever)'],
            [<code>clarification_callback</code>, 'Callable | None', 'No', 'None', '(question, options) -> int'],
            [<code>input_callback</code>, 'Callable | None', 'No', 'None', '(prompt) -> str'],
            [<code>stable_system_prompt</code>, 'bool', 'No', 'True', 'v0.2.1: pass system prompt separately on the OpenAI native-tools path for cache-friendly placement'],
            [<code>cache_system_prompt</code>, 'bool', 'No', 'True', 'v0.2.2+: apply Anthropic cache_control to the final system block'],
            [<code>cache_tools</code>, 'bool', 'No', 'True', 'v0.2.2+: apply Anthropic cache_control to the final tool spec'],
          ]}
        />

        <h4>AgentResponse</h4>
        <ApiTable
          headers={['Attribute', 'Type', 'Description']}
          rows={[
            [<code>output</code>, 'str', 'Final output text'],
            [<code>success</code>, 'bool', 'Whether task completed successfully'],
            [<code>mode</code>, 'AgentMode', 'SINGLE, SUB_AGENTS, or AUTO'],
            [<code>iterations</code>, 'int', 'Number of ReAct iterations'],
            [<code>tool_calls</code>, 'int', 'Number of tool calls made'],
            [<code>tokens_used</code>, 'int', 'Total tokens consumed'],
            [<code>execution_time</code>, 'float', 'Time in seconds'],
            [<code>execution_trace</code>, 'List[dict]', 'Full execution trace'],
            [<code>execution_tree</code>, 'dict', 'Hierarchical execution tree'],
            [<code>routing_decision</code>, 'RoutingDecision | None', 'Routing decision if sub-agents used'],
            [<code>metadata</code>, 'dict', 'Additional metadata'],
            [<code>citations</code>, 'list[Citation]', 'RAG citations (v0.2.0) with index, source, chunk_id, relevance_score, quote, page, section'],
            [<code>sources</code>, 'list[str]', 'De-duplicated source identifiers backing the answer'],
          ]}
        />
      </>
    ),
  },
  {
    id: 'model',
    title: 'Model',
    icon: <Cpu size={20} />,
    content: (
      <>
        <h3>Model Loading</h3>
        <CodeBlock
          code={`def load_model(
    model_name: str,
    engine: str | None = None,                  # auto-detected from model_name when None
    engine_config: dict | None = None,
    tensor_parallel_size: int | None = None,    # vLLM only
    gpu_memory_utilization: float | None = None,# vLLM only (0.0-1.0, default 0.90)
    apply_chat_template: bool = True,           # vLLM instruction-tuned models
    provider: str | None = None,                # explicit cloud provider
    **kwargs                                    # quantization="4bit|8bit|awq|gptq",
                                                # device_map, api_key, trust_remote_code, ...
) -> BaseModel:
    """
    Load a language model.

    Args:
        model_name: HuggingFace ID, local path, .gguf path, API model name, or
                    v0.2.3+ provider-prefixed model ID (e.g. 'groq:llama-3.3-70b-versatile')
        engine: Force a local HuggingFace engine. Common values:
                'transformers' | 'vllm' | 'mlx' | 'mlx_vlm'.
                GGUF and API adapters are auto-detected by model_name/provider.
        provider: Explicit cloud provider override when model IDs overlap.
                  v0.2.1: openai | anthropic | gemini | cerebras.
                  v0.2.3+: also groq | together | fireworks | replicate | hf.
        kwargs: quantization='4bit' | '8bit' | 'awq' | 'gptq', device_map='auto',
                api_key='...' (cloud), torch_dtype='float16', trust_remote_code=True

    Returns:
        Loaded model instance
    """`}
          language="python"
          filename="model_api.py"
        />

        <h4>GenerationConfig</h4>
        <ApiTable
          headers={['Parameter', 'Type', 'Default', 'Description']}
          rows={[
            [<code>max_tokens</code>, 'int | None', 'None', 'Maximum tokens to generate'],
            [<code>temperature</code>, 'float', '0.7', 'Sampling temperature'],
            [<code>top_p</code>, 'float', '0.9', 'Nucleus sampling probability'],
            [<code>top_k</code>, 'int', '50', 'Top-k sampling'],
            [<code>presence_penalty</code>, 'float', '0.0', 'Presence penalty'],
            [<code>frequency_penalty</code>, 'float', '0.0', 'Frequency penalty'],
            [<code>repetition_penalty</code>, 'float', '1.0', 'Repetition penalty'],
            [<code>stop_sequences</code>, 'List[str]', 'None', 'Stop generation sequences'],
            [<code>seed</code>, 'int | None', 'None', 'Random seed'],
            [<code>draft_model</code>, 'Any', 'None', 'Optional draft model for speculative decoding'],
            [<code>reasoning_effort</code>, '"none" | "minimal" | "low" | "medium" | "high" | "xhigh" | None', 'None', 'v0.2.1: OpenAI reasoning effort; effGen validates membership and passes the value through to reasoning models'],
            [<code>max_reasoning_tokens</code>, 'int | None', 'None', 'v0.2.1: OpenAI o-series reasoning token budget'],
            [<code>thinking_budget</code>, 'int | None', 'None', 'v0.2.2+: Gemini internal reasoning token budget'],
            [<code>include_thoughts</code>, 'bool', 'False', 'v0.2.2+: surface Gemini thinking trace in metadata'],
            [<code>grounding</code>, 'bool', 'False', 'v0.2.2+: enable Gemini Google Search grounding'],
            [<code>thinking</code>, 'dict | None', 'None', 'v0.2.2+: Anthropic extended thinking config'],
          ]}
        />

        <h4>GenerationResult Metadata</h4>
        <p>
          Provider adapters return <code>GenerationResult</code>. Token, cache, tool,
          grounding, and provider-specific fields are surfaced in <code>metadata</code>.
        </p>
        <ApiTable
          headers={['Field', 'Providers', 'Description']}
          rows={[
            [<code>metadata["prompt_tokens"]</code>, 'API providers', 'Prompt/input tokens reported by the provider'],
            [<code>metadata["completion_tokens"]</code>, 'API providers', 'Completion/output tokens reported by the provider'],
            [<code>metadata["total_tokens"]</code>, 'API providers', 'Prompt plus completion tokens'],
            [<code>metadata["cached_input_tokens"]</code>, 'OpenAI; Anthropic v0.2.2+', 'Prompt-cache hit tokens; 0 when caching is not used'],
            [<code>metadata["cache_creation_tokens"]</code>, 'Anthropic v0.2.2+', 'Tokens written into a new prompt-cache entry'],
            [<code>metadata["thinking"]</code>, 'Gemini v0.2.2+, Anthropic v0.2.2+', 'Reasoning/thinking trace when requested'],
            [<code>metadata["grounding_chunks"]</code>, 'Gemini v0.2.2+', 'Google Search grounding chunk records with URL/title fields when available'],
            [<code>metadata["tool_calls"]</code>, 'OpenAI, Cerebras, Gemini v0.2.2+', 'Structured native or function tool calls returned by the model'],
            [<code>metadata["tool_uses"]</code>, 'Anthropic v0.2.2+', 'Tool-use records returned by AnthropicAdapter.generate_with_tools()'],
            [<code>metadata["raw_content_blocks"]</code>, 'Anthropic v0.2.2+', 'Raw Claude content blocks, including redacted_thinking, for multi-turn replay'],
            [<code>metadata["compute_seconds"]</code>, 'Replicate v0.2.3+', 'Prediction runtime used for Replicate cost tracking'],
          ]}
        />

        <h4>OpenAI Prompt Caching (v0.2.1)</h4>
        <p>
          Normal <code>OpenAIAdapter.generate()</code> sends a user message only.
          For explicit cache-friendly system prompt placement, use
          <code> generate_with_system_prompt()</code>. Inside an Agent,
          <code> stable_system_prompt</code> is applied only when the OpenAI
          native-tools path calls the Responses API.
        </p>
        <CodeBlock
          code={`from effgen import load_model

model = load_model("gpt-5.4-nano", provider="openai")
result = model.generate_with_system_prompt(
    prompt="Summarize the policy update.",
    system_prompt="You are the support policy assistant. Use the fixed policy below...",
)
print(result.metadata["cached_input_tokens"])`}
          language="python"
          filename="openai_prompt_cache.py"
        />

        <h4>Supported Backends</h4>
        <ApiTable
          headers={['Backend', 'How load_model selects it', 'Best For']}
          rows={[
            [<code>transformers</code>, 'Default local HuggingFace path, or engine="transformers"', 'Development, quantization (4bit/8bit/AWQ/GPTQ)'],
            [<code>vllm</code>, 'engine="vllm"', 'Production CUDA, high throughput'],
            [<code>mlx</code>, 'engine="mlx" or mlx-community model auto-detection', 'Native M1/M2/M3/M4 text generation'],
            [<code>mlx_vlm</code>, 'engine="mlx_vlm"', '30+ vision-language architectures on Apple Silicon'],
            [<code>gguf</code>, '.gguf model path', 'CPU / Metal quantized models'],
            [<code>openai</code>, 'provider="openai"; v0.2.3+ also supports openai:gpt-5.4-nano prefix', 'gpt-5/gpt-5.4-nano/o-series, structured outputs, native tools'],
            [<code>anthropic</code>, 'provider="anthropic"; v0.2.3+ also supports anthropic:claude-sonnet-4-6 prefix', 'v0.2.1 adapter; v0.2.2+ Claude 4.x thinking, prompt caching, experimental tool specs'],
            [<code>gemini</code>, 'provider="gemini"; v0.2.3+ also supports gemini:gemini-2.5-flash prefix', 'v0.2.1 adapter; v0.2.2+ Gemini thinking, grounding, Files API, native tools'],
            [<code>cerebras</code>, 'provider="cerebras"; v0.2.3+ also supports cerebras:llama3.1-8b prefix', '4 registered models; 2 reliably free-tier callable; model-dependent native tools'],
            [<code>groq</code>, 'v0.2.3+: provider="groq" or groq:llama-3.3-70b-versatile prefix', 'Fast cloud inference and native tools'],
            [<code>together</code>, 'v0.2.3+: provider="together"', 'Together AI catalog and model refresh'],
            [<code>fireworks</code>, 'v0.2.3+: provider="fireworks"', 'Fireworks OpenAI-compatible hosted models; 54 tool-capable'],
            [<code>replicate</code>, 'v0.2.3+: provider="replicate"', 'Replicate predictions, 34 streaming-capable models, timeout cancellation'],
            [<code>hf</code>, 'v0.2.3+: provider="hf"', 'HuggingFace Inference Router and custom endpoints'],
          ]}
        />

        <h4>ProviderRegistry API (v0.2.3+)</h4>
        <CodeBlock
          code={`from effgen.models.registry import ProviderRegistry, list_providers, list_models, lookup
from effgen.models.auth import check_keys

providers: list[str] = list_providers()
models: list[dict] = list_models("groq")
provider, adapter_cls, model_info = lookup("groq:llama-3.1-8b-instant")
keys: dict = check_keys()

# CLI equivalent:
# effgen doctor
# effgen doctor --provider groq --json`}
          language="python"
          filename="provider_registry_api.py"
        />

        <h4>OpenAI Structured Outputs v2</h4>
        <p>
          <code>to_openai_schema()</code> converts Pydantic models into strict
	          OpenAI JSON Schema by inlining <code>$ref</code> definitions, forcing
	          <code> additionalProperties: false</code>, and adding a
	          <code> required</code> list when an object schema does not already
	          define one.
        </p>
        <CodeBlock
          code={`from typing import Literal
from pydantic import BaseModel
from effgen.models.errors import ModelRefusalError
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.openai_schema import to_openai_schema

class Answer(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "Answer",
        "schema": to_openai_schema(Answer),
        "strict": True,
    },
}

model = OpenAIAdapter(model_name="gpt-5.4-nano")
model.load()

try:
    result = model.generate_structured(
        "Classify the customer note.",
        response_format=response_format,
    )
    print(result.text)
    print(result.metadata["cached_input_tokens"])
except ModelRefusalError as exc:
    print("Model refused:", exc)`}
          language="python"
          filename="openai_structured_outputs.py"
        />

        <h4>OpenAI Registry Helpers (v0.2.1)</h4>
        <CodeBlock
          code={`from effgen.models.openai_models import (
    available_models,
    chat_models,
    reasoning_models,
    model_info,
    supports_reasoning,
    get_context_length,
    get_max_output,
    get_pricing,
)

print(available_models())
print(chat_models())
print(reasoning_models())
print(model_info("gpt-5.4-nano"))
print(supports_reasoning("o4-mini"))
print(get_context_length("o4-mini"))
print(get_max_output("o4-mini"))
print(get_pricing("gpt-5.4-nano"))`}
          language="python"
          filename="openai_registry_helpers.py"
        />

        <h4>Provider Helpers, Rate Limits, and Cost</h4>
        <CodeBlock
          code={`import asyncio
from effgen.models._cost import CostTracker
from effgen.models._rate_limit import RateLimitCoordinator, RateLimitExceeded
from effgen.models.cerebras_models import available_models, free_tier_models, model_info

print(available_models())
print(free_tier_models())
limits = model_info("llama3.1-8b")

coordinator = RateLimitCoordinator(
    "cerebras",
    "llama3.1-8b",
    rpm=limits["rpm"],
    rph=limits["rph"],
    rpd=limits["rpd"],
    tpm=limits["tpm"],
    tph=limits["tph"],
    tpd=limits["tpd"],
)

async def schedule_call(tokens_estimate: int) -> None:
    try:
        await coordinator.acquire(tokens_estimate=tokens_estimate)
        coordinator.record(actual_tokens=tokens_estimate)
    except RateLimitExceeded as exc:
        print("Daily provider limit exhausted:", exc)

asyncio.run(schedule_call(512))
print(CostTracker.get().summary())`}
          language="python"
          filename="provider_limits_cost.py"
        />

        <h4>Gemini Files and Parallel Tool Calls (v0.2.2+)</h4>
        <CodeBlock
          code={`from effgen.models.base import GenerationConfig
from effgen.models.gemini_adapter import GeminiAdapter
from effgen.models.gemini_files import FileRef, upload_file
from effgen.tools.builtin.gemini_native import GoogleSearchTool

doc: FileRef = upload_file("brief.pdf")

with GeminiAdapter(model_name="gemini-2.5-pro") as model:
    result = model.generate(
        "Use this file and search grounding to summarize the latest requirements.",
        config=GenerationConfig(thinking_budget=4096, include_thoughts=True, grounding=True),
        files=[doc],
        tools=[GoogleSearchTool().to_gemini_tool()],
    )

print(result.metadata["thinking"])
print(result.metadata["grounding_chunks"])
print(result.metadata["tool_calls"])`}
          language="python"
          filename="gemini_files_tools.py"
        />

        <h4>Anthropic Streaming, Caching, and Redacted Thinking (v0.2.2+)</h4>
        <CodeBlock
          code={`from effgen.models import AnthropicAdapter, mark_cached
from effgen.models.base import GenerationConfig

long_context = "..."  # Large reusable system context.

with AnthropicAdapter(model_name="claude-sonnet-4-6") as model:
    result = model.generate(
        "Reason about this architecture.",
        system_prompt=[mark_cached({"type": "text", "text": long_context}, ttl="1h")],
        config=GenerationConfig(thinking={"type": "enabled", "budget_tokens": 4096}),
    )
    print(result.metadata["cached_input_tokens"])
    print(result.metadata["cache_creation_tokens"])

    assistant_message = model.build_assistant_message(result)
    print(assistant_message["content"])  # Preserves raw_content_blocks/redacted_thinking.

    for chunk in model.generate_stream_full("Continue with implementation risks."):
        if chunk.type == "thinking":
            print("[thinking]", chunk.text)
        elif chunk.type == "redacted_thinking":
            print("[redacted]", chunk.data)
        elif chunk.type == "tool_use":
            print("[tool]", chunk.data)
        elif chunk.type == "text":
            print(chunk.text, end="")`}
          language="python"
          filename="anthropic_stream_cache.py"
        />
      </>
    ),
  },
  {
    id: 'tools',
    title: 'Tools',
    icon: <Wrench size={20} />,
    content: (
      <>
        <h3>Tool API</h3>
        <CodeBlock
          code={`class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, metadata: ToolMetadata):
        self._metadata = metadata

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @abstractmethod
    async def _execute(self, **kwargs) -> dict:
        """Execute the tool with given arguments."""

    async def execute(self, **kwargs) -> ToolResult:
        """Public execute method with validation."""

class ToolResult:
    """Result from tool execution."""
    success: bool         # Execution success
    output: Any           # Tool output
    error: str = None     # Error message if failed
    execution_time: float # Time taken
    metadata: dict        # Extra metadata
    timestamp: str        # ISO timestamp

class ToolMetadata:
    """Tool metadata and schema."""
    name: str
    description: str
    category: ToolCategory
    parameters: list[ParameterSpec]
    examples: List[dict] = None`}
          language="python"
          filename="tool_api.py"
        />

        <h4>Built-in Tools (31)</h4>
        <ApiTable
          headers={['Tool', 'Description', 'Parameters']}
          rows={[
            [<code>Calculator</code>, 'Math, unit conversions, statistics', 'expression, operation, from_unit, to_unit, precision'],
            [<code>PythonREPL</code>, 'Interactive Python with state persistence', 'code, session_id, reset_session, return_variables, restricted_mode'],
            [<code>WebSearch</code>, 'Search the web (DuckDuckGo)', 'query, num_results, backend, time_range, language, region'],
            [<code>CodeExecutor</code>, 'Execute code in secure sandbox', 'code, language, timeout, memory_limit, network_enabled, files, env_vars'],
            [<code>FileOperations</code>, 'Read, write, list files', 'operation, path, content, format, encoding, pattern, recursive, target_format'],
            [<code>BashTool</code>, 'Shell commands with security controls', 'command: str'],
            [<code>Retrieval</code>, 'RAG-based semantic search', 'query, top_k, score_threshold, filter_metadata'],
            [<code>AgenticSearch</code>, 'ripgrep-based exact search', 'query, context_lines, case_sensitive, use_regex, max_results, search_mode, file_type'],
            [<code>JSONTool</code>, 'Parse, query, validate JSON', 'data: str, operation: str'],
            [<code>DateTimeTool</code>, 'Time, timezone, date math', 'operation, timezone, date, days, hours, to_timezone'],
            [<code>TextProcessingTool</code>, 'Word count, regex, comparison', 'operation, text, pattern, replacement, transform_type, num_sentences'],
            [<code>URLFetchTool</code>, 'Fetch and extract web pages', 'url, extract_links'],
            [<code>WeatherTool</code>, 'Weather via Open-Meteo API', 'location: str'],
            [<code>WikipediaTool</code>, 'Wikipedia search and articles', 'query: str'],
            [<code>StockPriceTool</code>, 'Stock quotes (yfinance + Yahoo v8)', 'symbol: str'],
            [<code>CurrencyConverterTool</code>, 'FX conversion (frankfurter.app)', 'amount: float, from_currency: str, to_currency: str'],
            [<code>CryptoTool</code>, 'Crypto prices (CoinGecko)', 'coin: str, vs_currency: str (default "usd")'],
            [<code>DataFrameTool</code>, 'Pandas DataFrames', 'file_path, operation, n, column, op, value, agg, group_by'],
            [<code>PlotTool</code>, 'Matplotlib plots → PNG', 'chart_type, x, y, title, xlabel, ylabel, output_path'],
            [<code>StatsTool</code>, 'NumPy statistics', 'operation, data, y'],
            [<code>GitTool</code>, 'Read-only Git', 'operation, cwd, n, ref'],
            [<code>DockerTool</code>, 'Read-only Docker', 'operation, container, tail'],
            [<code>SystemInfoTool</code>, 'psutil CPU/mem/disk/net', 'kind: str (cpu/memory/disk/network/all)'],
            [<code>HTTPTool</code>, 'urllib GET/POST', 'url: str, method: str, headers, params, json_body, timeout'],
            [<code>ArxivTool</code>, 'arXiv Atom feed', 'query: str, max_results: int'],
            [<code>StackOverflowTool</code>, 'Stack Exchange API', 'query: str, max_results: int'],
            [<code>GitHubTool</code>, 'Public GitHub search', 'query: str, kind: str (repositories/code/issues), max_results: int'],
            [<code>WolframAlphaTool</code>, 'WolframAlpha (optional AppID)', 'query: str (set WOLFRAM_ALPHA_APPID env var)'],
            [<code>EmailDraftTool</code>, 'Draft email (does NOT send)', 'to, subject, body, cc, bcc, from_address'],
            [<code>SlackDraftTool</code>, 'Draft Slack message (does NOT send)', 'channel, text, thread_ts, mentions'],
            [<code>NotificationTool</code>, 'Local desktop notification', 'title, message, timeout'],
          ]}
        />

        <h4>Tool Categories</h4>
        <ApiTable
          headers={['Category', 'Description']}
          rows={[
            [<code>INFORMATION_RETRIEVAL</code>, 'Search and information gathering'],
            [<code>CODE_EXECUTION</code>, 'Code execution tools'],
            [<code>FILE_OPERATIONS</code>, 'File system operations'],
            [<code>COMPUTATION</code>, 'Math, calculations'],
            [<code>COMMUNICATION</code>, 'Communication tools'],
            [<code>DATA_PROCESSING</code>, 'Data manipulation'],
            [<code>SYSTEM</code>, 'System-level operations'],
            [<code>EXTERNAL_API</code>, 'External API calls'],
          ]}
        />
      </>
    ),
  },
  {
    id: 'memory',
    title: 'Memory',
    icon: <Brain size={20} />,
    content: (
      <>
        <h3>Memory API</h3>
        <CodeBlock
          code={`class ShortTermMemory:
    """Conversation context memory."""

    def __init__(
        self,
        max_tokens: int = 4096,
        max_messages: int = 100,
        summarization_threshold: float = 0.8
    ): ...

    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: dict | None = None,
        tokens: int | None = None,
    ) -> Message: ...
    def add_user_message(self, content: str) -> Message: ...
    def add_assistant_message(self, content: str) -> Message: ...
    def get_recent_messages(self, n: int) -> List[Message]: ...
    def get_token_count(self) -> int: ...
    def clear(self) -> None: ...

class LongTermMemory:
    """Persistent memory storage."""

    def __init__(
        self,
        backend: StorageBackend,
        consolidation_interval: int = 100,
        max_memories: int = 10000,
        min_importance_to_keep: ImportanceLevel = ImportanceLevel.LOW,
    ): ...

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: ImportanceLevel = ImportanceLevel.MEDIUM,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryEntry: ...

    def search(
        self,
        query: str,
        tags: List[str] = None,
        min_importance: ImportanceLevel = None,
        limit: int = 10
    ) -> List[Memory]: ...

    def get_memory(self, memory_id: str, update_access: bool = True) -> MemoryEntry | None: ...
    def consolidate(self) -> int: ...`}
          language="python"
          filename="memory_api.py"
        />

        <h4>Memory Types</h4>
        <ApiTable
          headers={['Type', 'Description', 'Use Case']}
          rows={[
            [<code>CONVERSATION</code>, 'Conversation summaries / dialogue snapshots', 'Context recall across sessions'],
            [<code>FACT</code>, 'Discrete facts learned about user/world', 'User preferences, profile data'],
            [<code>OBSERVATION</code>, 'Facts observed by the agent', 'Environmental signals'],
            [<code>TASK</code>, 'Task completions and outcomes', 'Task history'],
            [<code>TOOL_RESULT</code>, 'Notable tool invocation results', 'Reusing past computations'],
            [<code>REFLECTION</code>, 'Agent reflections / learning notes', 'Learning patterns'],
          ]}
        />

        <h4>Importance Levels</h4>
        <ApiTable
          headers={['Level', 'Value', 'Description']}
          rows={[
            [<code>LOW</code>, '1', 'Routine information'],
            [<code>MEDIUM</code>, '2', 'Useful context'],
            [<code>HIGH</code>, '3', 'Important facts'],
            [<code>CRITICAL</code>, '4', 'Essential information'],
          ]}
        />
      </>
    ),
  },
  {
    id: 'orchestrator',
    title: 'Orchestrator',
    icon: <Network size={20} />,
    content: (
      <>
        <h3>Orchestration API</h3>
        <CodeBlock
          code={`class MultiAgentOrchestrator:
    """Coordinate multiple agents."""

    def __init__(self, config: dict | None = None): ...

    def register_agent(self, agent: Agent) -> None: ...

    def create_team(
        self,
        name: str,
        agents: list[Agent],
        pattern: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL,
        manager_agent: Agent | None = None,
        **kwargs,                          # voting_strategy, timeout, max_rounds, metadata
    ) -> TeamConfig: ...

    def assign_task(
        self,
        task: str,
        team: TeamConfig,
        context: dict | None = None,
    ) -> TeamResponse: ...

class OrchestrationPattern(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    PIPELINE = "pipeline"

@dataclass
class TeamResponse:
    output: str
    success: bool = True
    pattern: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL
    agent_responses: list[dict] = field(default_factory=list)  # per-agent dict
    execution_time: float = 0.0
    rounds: int = 1                          # collaborative
    selected_response: dict | None = None    # competitive
    consensus_score: float = 0.0
    metadata: dict = field(default_factory=dict)`}
          language="python"
          filename="orchestration_api.py"
        />

        <h4>Orchestration Patterns</h4>
        <ApiTable
          headers={['Pattern', 'Description', 'Use Case']}
          rows={[
            [<code>SEQUENTIAL</code>, 'Agents run one after another', 'Pipelines, chains'],
            [<code>PARALLEL</code>, 'Agents run simultaneously', 'Independent subtasks'],
            [<code>HIERARCHICAL</code>, 'Manager delegates to workers', 'Complex coordination'],
            [<code>COLLABORATIVE</code>, 'Agents discuss and refine', 'Creative tasks'],
            [<code>COMPETITIVE</code>, 'Best response wins', 'Quality selection'],
            [<code>PIPELINE</code>, 'Output of each agent feeds next, with explicit data passing', 'Data transformation pipelines'],
          ]}
        />
      </>
    ),
  },
  {
    id: 'exceptions',
    title: 'Exceptions',
    icon: <Code size={20} />,
    content: (
      <>
        <h3>Errors via AgentResponse</h3>
        <p>
          <code>Agent.run()</code> does not raise on internal failures — failures are surfaced
          through the returned <code>AgentResponse</code> so callers can inspect partial output,
          guardrail blocks, and per-tool errors. Inspect <code>success</code>,
          <code> iterations</code>, <code>metadata</code>, and <code>execution_trace</code>.
        </p>

        <CodeBlock
          code={`result = agent.run("Complex task")

if not result.success:
    print("Failed:", result.metadata.get("error") or result.output)

if result.iterations >= agent.config.max_iterations:
    print("Hit max_iterations — partial:", result.output)

if result.metadata.get("guardrail_blocked"):
    print("Blocked:", result.metadata["guardrail_reason"])

for step in result.execution_trace:
    if step.get("tool_error"):
        print("Tool", step["tool"], "failed:", step["tool_error"])`}
          language="python"
          filename="agent_errors.py"
        />

        <h3>Client SDK Exceptions</h3>
        <p>
          The Python API client (<code>effgen.client</code>) raises typed exceptions for HTTP-level
          failures. All inherit from <code>EffGenClientError</code>.
        </p>

        <ApiTable
          headers={['Exception', 'When']}
          rows={[
            [<code>EffGenClientError</code>, 'Base class for all client errors'],
            [<code>EffGenAPIError</code>, 'Server returned a non-2xx response'],
            [<code>EffGenAuthError</code>, '401 / 403 from the server'],
            [<code>EffGenRateLimitError</code>, '429 — server payload accessible via .payload'],
            [<code>EffGenServerError</code>, '5xx — automatically retried up to max_retries'],
            [<code>EffGenConnectionError</code>, 'Network / DNS / TLS failure'],
            [<code>EffGenTimeoutError</code>, 'Request exceeded the configured timeout'],
          ]}
        />

        <CodeBlock
          code={`from effgen.client import (
    EffGenClient,
    EffGenAuthError,
    EffGenRateLimitError,
    EffGenServerError,
)

client = EffGenClient(base_url="http://localhost:8000", api_key="sk-...")

try:
    resp = client.chat("Hello")
except EffGenAuthError:
    print("Invalid API key")
except EffGenRateLimitError as e:
    print("Rate-limited:", e)
except EffGenServerError as e:
    print("Server error after retries:", e)`}
          language="python"
          filename="client_error_handling.py"
        />
      </>
    ),
  },
  {
    id: 'server',
    title: 'API Server',
    icon: <Server size={20} />,
    content: (
      <>
        <h3>Server Endpoints</h3>
        <p>effGen provides a FastAPI-based REST server with the following endpoints:</p>

        <ApiTable
          headers={['Method', 'Endpoint', 'Description']}
          rows={[
            [<code>POST</code>, '/run', 'Execute a task and return the result'],
            [<code>WS</code>, '/ws', 'WebSocket endpoint for bidirectional streaming'],
            [<code>GET</code>, '/health', 'Health check endpoint'],
            [<code>GET</code>, '/metrics', 'JSON server metrics endpoint'],
            [<code>GET</code>, '/tools', 'List available built-in tools'],
            [<code>GET</code>, '/openapi.json', 'OpenAPI specification'],
          ]}
        />

        <h4>Starting the Server</h4>
        <CodeBlock
          code={`# CLI
effgen serve --port 8000 --model Qwen/Qwen2.5-3B-Instruct

# OpenAI-compatible routers are available separately:
from effgen.api import create_openai_router, create_embeddings_router`}
          language="python"
          filename="server.py"
        />

        <h4>POST /run</h4>
        <CodeBlock
          code={`import httpx

async def run_task():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/run",
            json={
                "task": "What is sqrt(144)?",
                "preset": "math",
                "max_iterations": 10
            },
            headers={"Authorization": "Bearer YOUR_API_KEY"}
        )
        result = response.json()
        print(result["output"])
        print(result["success"])`}
          language="python"
          filename="api_run.py"
        />

        <h4>WS /ws</h4>
        <CodeBlock
          code={`import asyncio
import websockets

async def stream_task():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send("Explain quantum computing")
        async for chunk in ws:
            print(chunk, end="")

asyncio.run(stream_task())`}
          language="python"
          filename="api_websocket.py"
        />
      </>
    ),
  },
  {
    id: 'presets',
    title: 'Presets',
    icon: <Layers size={20} />,
    content: (
      <>
        <h3>Preset API Reference</h3>

        <h4>create_agent()</h4>
        <CodeBlock
          code={`from effgen.presets import create_agent

def create_agent(
    preset: str,           # "math" | "research" | "coding" | "general" | "rag" | "minimal"
    model: BaseModel | str,
    *,
    agent_name: str | None = None,
    extra_tools: list | None = None,
    knowledge_base: str | None = None,   # RAG preset: directory ingested at construction
    system_prompt: str | None = None,
    max_iterations: int | None = None,
    temperature: float | None = None,
    enable_memory: bool | None = None,
    **config_overrides,    # Override any AgentConfig parameter
) -> Agent:
    """Create an agent from a preset configuration."""`}
          language="python"
          filename="create_agent_api.py"
        />

        <ApiTable
          headers={['Preset', 'Tools Included', 'Description']}
          rows={[
            [<code>math</code>, 'Calculator, PythonREPL', 'Mathematical computations'],
            [<code>research</code>, '15 tools (WebSearch, URLFetch, Wikipedia, ArXiv, PubMed, SemanticScholar, RSS, News, YouTube, Reddit, HackerNews, PDF/DOCX/Excel)', 'Web + academic + news + social research'],
            [<code>coding</code>, 'CodeExecutor, PythonREPL, FileOps, Bash', 'Code development tasks'],
            [<code>general</code>, '31 tools (computation, code, web, research, news, social, translation, QR, OCR, audio, image, documents, geo/weather, email, webhooks)', 'General purpose with all capabilities'],
            [<code>rag</code>, 'Retrieval (hybrid search + citations)', 'Retrieval-Augmented Generation over a knowledge_base directory'],
            [<code>media</code>, 'AudioTranscribe, ImageCaption', 'v0.2.6 — audio transcription + vision captioning'],
            [<code>notify</code>, 'EmailSMTP, EmailIMAP, SlackWebhook, DiscordWebhook', 'v0.2.6 — email + Slack + Discord notifications'],
            [<code>multimodal</code>, 'MultimodalDescribe, ImageInfo, ImageCaption, OCR, AudioTranscribe, PDF, Weather', 'v0.2.8 — image / audio / video understanding'],
            [<code>minimal</code>, 'None', 'Direct LLM inference without tools'],
          ]}
        />

        <h4>list_presets()</h4>
        <CodeBlock
          code={`from effgen.presets import list_presets, get_preset

def list_presets() -> Dict[str, str]:
    """Return a mapping of preset name → description."""

# Usage
for name, description in list_presets().items():
    print(f"{name}: {description}")

# Get full preset config
preset = get_preset("math")
print(f"Tools: {preset.tool_names}")
print(f"Temperature: {preset.temperature}")`}
          language="python"
          filename="list_presets_api.py"
        />
      </>
    ),
  },
  {
    id: 'plugins',
    title: 'Plugins',
    icon: <Plug size={20} />,
    content: (
      <>
        <h3>Plugin API Reference</h3>

        <h4>ToolPlugin Base Class</h4>
        <CodeBlock
          code={`from effgen.tools.plugin import ToolPlugin

class ToolPlugin:
    name: str              # Plugin name
    version: str           # Plugin version
    description: str       # Plugin description
    tools: list[type[BaseTool]]

    def __init__(self, name=None, version=None, description=None, tools=None): ...

    def register(self, registry: ToolRegistry) -> List[str]:
        """Register tool classes from self.tools. Returns registered tool names."""`}
          language="python"
          filename="tool_plugin_api.py"
        />

        <h4>PluginManager</h4>
        <CodeBlock
          code={`from effgen.tools.plugin import PluginManager

class PluginManager:
    def __init__(self, registry: ToolRegistry | None = None): ...

    def discover_entry_points(self) -> List[str]:
        """Discover plugins via Python entry points."""

    def discover_user_dir(self) -> List[str]:
        """Discover plugins from ~/.effgen/plugins/."""

    def discover_env_dir(self) -> List[str]:
        """Discover plugins from EFFGEN_PLUGINS_DIR."""

    def discover_all(self) -> List[str]:
        """Discover from all sources."""`}
          language="python"
          filename="plugin_manager_api.py"
        />

        <ApiTable
          headers={['Discovery Method', 'Source', 'Description']}
          rows={[
            [<code>discover_entry_points()</code>, 'Python entry points', 'Loads plugins registered in pyproject.toml/setup.py'],
            [<code>discover_user_dir()</code>, '~/.effgen/plugins/', 'Loads plugins from user directory'],
            [<code>discover_env_dir()</code>, 'EFFGEN_PLUGINS_DIR', 'Loads plugins from environment variable path'],
            [<code>discover_all()</code>, 'All sources', 'Combines all discovery methods'],
          ]}
        />
      </>
    ),
  },
  {
    id: 'router',
    title: 'Router (v0.2.4+)',
    icon: <GitBranch size={20} />,
    content: (
      <>
        <h3>Policy-Based ModelRouter</h3>
        <p>
          v0.2.4 ships an opt-in policy-based router that selects a cloud
          <code> (provider, model_id)</code> pair using any combination of
          first-available, cost, and latency policies.
        </p>
        <CodeBlock
          code={`from effgen import (
    PolicyBasedRouter, RoutingContext,
    FirstAvailablePolicy, CostBasedPolicy, LatencyBasedPolicy,
    RouterDecision, RouterEvent,
)
from effgen.models.capabilities import Capability

class PolicyBasedRouter:
    def __init__(
        self,
        policies: list[RoutingPolicy],
        fallback: RoutingPolicy | None = None,  # default: FirstAvailablePolicy()
        failover_hops: int = 3,
        retry_policy: RetryPolicy | None = None,
    ) -> None: ...

    def subscribe(self, callback: Callable[[RouterEvent], None]) -> None: ...
    def route(self, context: RoutingContext) -> RouterDecision: ...
    def route_and_execute(
        self,
        context: RoutingContext,
        fn: Callable[[ProviderModelPair], Any],
    ) -> Any: ...`}
          language="python"
          filename="router_api.py"
        />

        <h4>RoutingContext</h4>
        <ApiTable
          headers={['Field', 'Type', 'Default', 'Description']}
          rows={[
            [<code>prompt_tokens_estimate</code>, 'int', '0', 'Estimated input token count used by CostBasedPolicy'],
            [<code>user_budget_usd</code>, 'float | None', 'None', 'Max cost per call in USD; None = unlimited'],
            [<code>latency_budget_ms</code>, 'int | None', 'None', 'Max acceptable p50 latency in ms; None = unlimited'],
            [<code>required_capabilities</code>, 'set[Capability]', 'set()', 'Capability flags the chosen provider must support'],
          ]}
        />

        <h4>Capability flags</h4>
        <ApiTable
          headers={['Flag', 'Meaning']}
          rows={[
            [<code>Capability.chat</code>, 'Basic text completion / chat'],
            [<code>Capability.tools</code>, 'Function / tool calling'],
            [<code>Capability.streaming</code>, 'Token streaming via generate_stream()'],
            [<code>Capability.vision</code>, 'Image inputs'],
            [<code>Capability.grounding</code>, 'Web grounding / Google Search'],
            [<code>Capability.thinking</code>, 'Extended thinking / chain-of-thought'],
            [<code>Capability.json_schema</code>, 'Structured output with JSON schema constraint'],
          ]}
        />

        <h4>RouterDecision</h4>
        <ApiTable
          headers={['Field', 'Type', 'Description']}
          rows={[
            [<code>chosen</code>, 'ProviderModelPair', 'NamedTuple(provider, model_id) selected by the winning policy'],
            [<code>eliminated</code>, 'list[tuple[ProviderModelPair, str]]', 'Every rejected candidate with the reason (no_key, cost_exceeds_budget, latency_exceeds_sla, missing capabilities: …, requires dedicated endpoint, model inactive)'],
            [<code>policy_name</code>, 'str', 'Name of the policy that produced the decision'],
            [<code>score</code>, 'float', 'Policy-defined numeric score (e.g. CostBasedPolicy returns USD cost; LatencyBasedPolicy returns p50 ms)'],
          ]}
        />

        <h4>Policies</h4>
        <ApiTable
          headers={['Policy', 'Selection rule', 'Raises']}
          rows={[
            [<code>FirstAvailablePolicy</code>, 'First provider (alphabetical) with key + required capabilities + serverless availability', <code>NoCandidateError</code>],
            [<code>CostBasedPolicy</code>, 'Cheapest provider/model pair within user_budget_usd; free tier ranks ahead of paid when prices tie; remaining ties broken by published price, provider priority list, provider name, model id', <><code>NoCandidateWithinBudgetError</code> (exposes <code>cheapest_pair</code>, <code>cheapest_cost_usd</code>)</>],
            [<code>LatencyBasedPolicy</code>, 'Fastest provider/model pair within latency_budget_ms using LatencyTracker p50 (seed values only when no measurements exist)', <code>NoCandidateWithinLatencyError</code>],
          ]}
        />

        <h4>RouterEvent</h4>
        <CodeBlock
          code={`@dataclass
class RouterEvent:
    from_provider: str
    from_model: str
    to_provider: str
    to_model: str
    reason: str               # rate_limited | transient_error_5xx | timeout
                              # | budget_exceeded_daily | budget_exceeded_monthly
                              # | <ExceptionClassName>.lower()
    hop: int                  # 1-indexed failover attempt
    exception: Exception | None

    def as_dict(self) -> dict: ...

router.subscribe(lambda ev: print(ev.as_dict()))`}
          language="python"
          filename="router_event.py"
        />

        <h4>Failover semantics</h4>
        <p>
          <code>route_and_execute(context, fn)</code> calls <code>fn</code> with the
          chosen <code>ProviderModelPair</code> and classifies exceptions:
        </p>
        <ApiTable
          headers={['Exception', 'Classification', 'Behaviour']}
          rows={[
            [<code>RateLimitExceeded</code>, 'retriable', 'Blocklist the provider, re-route, emit RouterEvent(reason="rate_limited")'],
            [<code>ProviderTransientError</code>, 'retriable', 'reason="transient_error_&lt;status_code&gt;"'],
            [<code>ModelTimeoutError</code>, 'retriable', 'reason="timeout"'],
            [<code>BudgetExceededError</code>, 'retriable', 'reason="budget_exceeded_&lt;period&gt;"; failover to a free-tier provider'],
            [<code>ModelAuthError</code>, 'terminal', 'Re-raised immediately'],
            [<code>ModelRefusalError</code>, 'terminal', 'Re-raised immediately'],
            [<code>InvalidRequestError</code>, 'terminal', 'Re-raised immediately'],
            [<code>AllCandidatesExhaustedError</code>, 'final', 'Raised when failover_hops is exhausted; carries the per-hop failures list'],
          ]}
        />

        <h3>Persistent Cost &amp; Rate-Limit Stores (v0.2.4+)</h3>
        <CodeBlock
          code={`from effgen.models._cost import CostTracker
from effgen.models._cost_store import SQLiteCostStore
from effgen.models._rate_limit import RateLimitCoordinator
from effgen.models._rate_limit_store import SQLiteRateLimitStore
from effgen.models.latency_tracker import LatencyTracker

# SQLite-backed cost tracker (default in v0.2.4 when CostTracker.get() is used)
store = SQLiteCostStore()              # ~/.effgen/costs.sqlite
tracker = CostTracker.get()            # singleton, persists every call
print(store.query_today())
print(store.query_week())
print(store.query_all())

# Cross-process rate-limit coordination (WAL mode, BEGIN IMMEDIATE)
rl_store = SQLiteRateLimitStore()      # ~/.effgen/rate_limits.sqlite
coord = RateLimitCoordinator(storage=rl_store)

# Latency tracker — populated automatically by every generate()/generate_stream()
lat = LatencyTracker()
print(lat.p50(provider="groq", model="llama-3.1-8b-instant"))`}
          language="python"
          filename="stores_v024.py"
        />

        <h3>effgen cost CLI (v0.2.4+)</h3>
        <ApiTable
          headers={['Command', 'Description']}
          rows={[
            [<code>effgen cost today</code>, 'Per-provider/model summary for the last 24 hours'],
            [<code>effgen cost week</code>, 'Rolling 7-day spend summary'],
            [<code>effgen cost by-provider</code>, 'Lifetime totals grouped by provider'],
            [<code>effgen cost set-budget &lt;usd&gt;</code>, 'Set a daily budget in USD (also via effgen config set budget.daily / budget.monthly)'],
            [<code>effgen cost clear-budget</code>, 'Remove configured budget limits'],
          ]}
        />
        <p>
          When configured, every paid call checks cumulative daily and monthly
          spend. At <code>≥ 80%</code> a <code>UserWarning</code> is emitted; at
          <code> ≥ 100%</code> a <code>BudgetExceededError(period="daily" |
          "monthly")</code> is raised. Zero-cost calls remain allowed so the
          router can fail over to free-tier providers.
        </p>
        <CodeBlock
          code={`from effgen.models.errors import BudgetExceededError, NoCandidateWithinBudgetError
from effgen import PolicyBasedRouter, RoutingContext, CostBasedPolicy
from effgen.models.capabilities import Capability

router = PolicyBasedRouter(policies=[CostBasedPolicy()], failover_hops=3)
try:
    decision = router.route(RoutingContext(
        prompt_tokens_estimate=1_000_000,
        user_budget_usd=0.001,
        required_capabilities={Capability.chat},
    ))
except NoCandidateWithinBudgetError as exc:
    print("Budget too tight. Cheapest:", exc.cheapest_pair, f"\${exc.cheapest_cost_usd:.6f}")`}
          language="python"
          filename="budget_errors.py"
        />
      </>
    ),
  },
];

export default function APIReference() {
  const [selectedSection, setSelectedSection] = useState<ApiSection>(apiSections[0]);
  const [searchQuery, setSearchQuery] = useState('');
  const contentRef = useRef<HTMLDivElement>(null);

  const handleSectionClick = (section: ApiSection) => {
    setSelectedSection(section);
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
  };

  const filteredSections = searchQuery
    ? apiSections.filter(s =>
        s.title.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : apiSections;

  return (
    <div className="api-page">
      <PageNavbar
        title="API Reference"
        items={[]}
        rightContent={
          <div className="api-search">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search API..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        }
      />

      <div className="api-layout">
        <nav className="api-nav">
          <div className="api-nav-header">
            <Code size={18} />
            <span>API Sections</span>
          </div>
          {filteredSections.map(section => (
            <button
              key={section.id}
              className={`api-nav-item ${selectedSection.id === section.id ? 'active' : ''}`}
              onClick={() => handleSectionClick(section)}
            >
              <div className="api-nav-icon">{section.icon}</div>
              <span className="api-nav-title">{section.title}</span>
              <ChevronRight size={16} className="api-nav-arrow" />
            </button>
          ))}
        </nav>

        <article className="api-content" ref={contentRef}>
          <div className="api-content-header">
            <div className="api-content-icon">{selectedSection.icon}</div>
            <div>
              <h2>{selectedSection.title} API</h2>
              <p className="api-content-desc">Complete reference for the {selectedSection.title} module.</p>
            </div>
          </div>
          <div className="api-content-body">
            {selectedSection.content}
          </div>
        </article>
      </div>
    </div>
  );
}
