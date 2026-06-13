# Architecture Guide

## System Overview

```
                    ┌─────────────────────────┐
                    │       User Input         │
                    │  (CLI / API / Python)    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │         Agent            │
                    │    (ReAct Loop)          │
                    │                         │
                    │  Thought → Action →     │
                    │  Observation → ...      │
                    └──┬──────┬──────┬────────┘
                       │      │      │
              ┌────────▼┐  ┌──▼───┐ ┌▼────────┐
              │  Model   │  │Tools │ │ Memory  │
              │ Backend  │  │      │ │         │
              └────────┬┘  └──┬───┘ └┬────────┘
                       │      │      │
    ┌──────────────────┼──────┼──────┼──────────┐
    │ Backends:        │      │      │          │
    │ - Transformers   │  Built-in   │ Short    │
    │ - vLLM           │  + Plugins  │ + Long   │
    │ - OpenAI API     │  + MCP      │ + Vector │
    │ - Anthropic API  │  + A2A/ACP  │          │
    │ - Gemini API     │             │          │
    └──────────────────┘─────────────┘──────────┘
```

## Core Components

### Agent (`effgen/core/agent.py`)

The central class. Implements the ReAct reasoning loop:

1. Receives a task from the user
2. Generates a **Thought** (reasoning about what to do)
3. Selects an **Action** (tool to call) with parameters
4. Receives an **Observation** (tool result)
5. Repeats until a **Final Answer** is reached or max iterations hit

Key features:
- Sub-agent decomposition for complex tasks
- Streaming output support
- Memory integration (short-term, long-term, vector)
- Configurable via `AgentConfig` dataclass

### Model Backends (`effgen/models/`)

Abstraction over multiple LLM backends:

| Backend | File | Use Case |
|---------|------|----------|
| `TransformersEngine` | `transformers_engine.py` | Local GPU inference (default) |
| `VLLMEngine` | `vllm_engine.py` | High-throughput serving |
| `OpenAIAdapter` | `openai_adapter.py` | OpenAI API models |
| `AnthropicAdapter` | `anthropic_adapter.py` | Claude models |
| `GeminiAdapter` | `gemini_adapter.py` | Google Gemini models |
| `MLXEngine` | `mlx_engine.py` | Apple Silicon (MLX) |
| `MLXVLMEngine` | `mlx_vlm_engine.py` | Apple Silicon vision-language |
| `GGUFEngine` | `gguf_engine.py` | GGUF quantized models (llama-cpp) |

All implement `BaseModel` with: `generate()`, `generate_stream()`, `count_tokens()`, `get_context_length()`, `load()`, `unload()`, `supports_tool_calling()`

Additional model infrastructure:
- `router.py`: `ModelRouter` — automatic model selection by query complexity
- `capabilities.py`: `MODEL_CAPABILITIES` — pre-populated profiles for 12+ models
- `pool.py`: `ModelPool` — LRU eviction, GPU memory management, hot-swap
- `lazy.py`: `LazyModel` — deferred loading until first use
- `batching.py`: `ContinuousBatcher` — coalesces concurrent requests

### Tools (`effgen/tools/`)

- `base_tool.py`: `BaseTool` abstract class with metadata and validation
- `registry.py`: `ToolRegistry` for discovery, lazy loading, dependency management
- `builtin/`: 66 built-in tools (core, finance, data science, DevOps, knowledge, communication)
- `plugin.py`: External plugin loading via entry points
- `protocols/`: MCP, A2A, ACP protocol implementations

### Guardrails (`effgen/guardrails/`)

Safety and validation framework:
- `base.py`: `Guardrail` ABC, `GuardrailChain`, `GuardrailPosition`
- `content.py`: `ToxicityGuardrail`, `PIIGuardrail`, `LengthGuardrail`, `TopicGuardrail`
- `injection.py`: `PromptInjectionGuardrail` (low/medium/high sensitivity)
- `tool_safety.py`: `ToolInputGuardrail`, `ToolOutputGuardrail`, `ToolPermissionGuardrail`
- `presets.py`: `get_guardrail_preset()` — strict/standard/minimal/none

### RAG (`effgen/rag/`)

Production RAG pipeline:
- `ingest.py`: `DocumentIngester` — multi-format document loading with deduplication
- `chunking.py`: `SemanticChunker`, `CodeChunker`, `TableChunker`, `HierarchicalChunker`
- `search.py`: `HybridSearchEngine` — dense + BM25 + keyword + metadata via Reciprocal Rank Fusion
- `reranker.py`: `CrossEncoderReranker`, `LLMReranker`, `RuleBasedReranker`
- `context_builder.py`: `ContextBuilder` — token budget management with citations
- `attribution.py`: `Citation`, `CitationTracker`

### Evaluation (`effgen/eval/`)

- `evaluator.py`: `AgentEvaluator`, `TestCase`, `EvalResult`, `SuiteResults`
- `suites.py`: 5 built-in suites (math, tool_use, reasoning, safety, conversation)
- `regression.py`: `RegressionTracker` — baseline comparison with severity alerts
- `comparison.py`: `ModelComparison` — multi-model matrix benchmarking

### Memory (`effgen/memory/`)

Three tiers:
1. **ShortTermMemory**: Recent conversation context (token-limited)
2. **LongTermMemory**: Persistent facts across sessions (SQLite)
3. **VectorMemoryStore**: Semantic search over past interactions
- `token_budget.py`: `TokenBudget` — smart context window allocation

### Cache (`effgen/cache/`)

- `prompt_cache.py`: `PromptCache` — LRU + TTL with sha256 fingerprinting
- `result_cache.py`: `ResultCache` — per-tool TTL, optional semantic similarity

### Orchestration (`effgen/core/`)

- `message_bus.py`: `MessageBus` — pub/sub inter-agent communication
- `workflow.py`: `WorkflowDAG` — DAG execution with conditional branching
- `shared_state.py`: `SharedState` — thread-safe namespaced key-value store
- `lifecycle.py`: `AgentRegistry`, `AgentPool` — lifecycle management
- `checkpoint.py`: `CheckpointManager` — save/restore agent state
- `session.py`: `Session`, `SessionManager` — persistent conversations
- `human_loop.py`: `HumanApproval`, `HumanInput`, `HumanChoice`
- `batch.py`: `BatchRunner` — concurrent batch execution

### Hardware (`effgen/hardware/`)

- `platform.py`: Apple Silicon/CUDA/MLX detection and backend recommendation

### Debug (`effgen/debug/`)

- `inspector.py`: `DebugAgent` — rich TUI step-through with `DebugTrace`

### API Server (`effgen/api/`)

- `openai_compat.py`: OpenAI-compatible `/v1/chat/completions` and `/v1/completions`
- `embeddings.py`: `/v1/embeddings` endpoint with caching
- `queue.py`: `RequestQueue` — priority-based with backpressure
- `pool.py`: `AgentPool` — pre-warmed agent instances
- `tenancy.py`: `TenantManager`, `APIKey` management
- `middleware.py`: CORS, request ID, GZip, graceful shutdown

### Client SDK (`effgen/client/`)

- `client.py`: `EffGenClient` — sync/async with retries, streaming, typed exceptions

### Prompts (`effgen/prompts/`)

- `TemplateManager`: Prompt template management
- `ChainManager`: Prompt chaining
- `PromptOptimizer`: SLM-specific prompt optimization
- `AgentSystemPromptBuilder`: Auto-generates system prompts from tools

### Domains (`effgen/domains/`)

- `base.py`: `Domain` — keywords, system_prompt, tool_names
- `expander.py`: `KeywordExpander` — WordNet/template/LLM expansion
- 5 built-in: `TechDomain`, `ScienceDomain`, `FinanceDomain`, `HealthDomain`, `LegalDomain`

### Configuration (`effgen/config/`)

YAML/JSON configuration loading with validation and defaults.

## Data Flow

```
User Query
    │
    ▼
AgentConfig (model, tools, prompts, memory)
    │
    ▼
Agent.__init__() → loads model, initializes tools & memory
    │
    ▼
Agent.run(task) → enters ReAct loop
    │
    ├──▶ Model.generate(prompt) → raw LLM output
    │        │
    │        ▼
    │    Parse: extract Thought, Action, Action Input
    │        │
    │        ▼
    │    Tool._execute(**params) → observation
    │        │
    │        ▼
    │    Append to context, check for Final Answer
    │        │
    │        └──▶ (loop back to generate)
    │
    ▼
AgentResponse (output, stats, trace)
```

## Plugin Architecture

Plugins are discovered from three sources:
1. Python entry points (`effgen.plugins` group)
2. User plugin directory (`~/.effgen/plugins/`)
3. Environment variable (`EFFGEN_PLUGINS_DIR`)

Each plugin provides a `ToolPlugin` subclass that registers `BaseTool` implementations into the global `ToolRegistry`.
