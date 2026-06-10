# Migration Guide

## v0.1.x → v0.2.0

### Breaking Changes

**None.** All existing `Agent`, `AgentConfig`, `load_model`, and tool APIs work without modification. v0.2.0 is fully backwards compatible.

### New AgentConfig Parameters (All Optional)

```python
config = AgentConfig(
    name="my_agent",
    model=model,
    tools=[Calculator()],

    # New in v0.2.0 (all optional, defaults preserve v0.1.x behavior):
    tool_calling_mode="auto",        # "auto", "native", "react", "hybrid"
    output_format=None,              # "json", "text", or None
    output_schema=None,              # JSON Schema dict
    guardrails=None,                 # GuardrailChain instance
    models=None,                     # List of additional models for routing
    speculative_execution=False,     # Run on 2 models, take fastest
    approval_mode="never",           # "never", "always", "first_time", "dangerous_only"
    approval_callback=None,          # Callable for human approval
    approval_timeout=60,             # Seconds to wait for approval
    session_id=None,                 # Persistent session ID
    checkpoint_interval=None,        # Checkpoint every N iterations
    checkpoint_dir=None,             # Directory for checkpoints
)
```

### New Agent.run() Parameters (All Optional)

```python
result = agent.run(
    "What is 2+2?",
    output_schema={"type": "object", ...},  # Per-call JSON schema
    output_model=MyPydanticModel,            # Per-call Pydantic model
    debug=True,                              # Capture DebugTrace
    checkpoint_interval=3,                   # Per-call checkpoint interval
)
```

### New AgentResponse Fields

```python
result = agent.run("query")
result.citations    # List[Citation] — RAG source citations (empty if no RAG)
result.sources      # List[str] — deduplicated source names
result.metadata["debug_trace"]  # DebugTrace (when debug=True)
result.metadata["parsed_output"]  # Pydantic model (when output_model used)
```

### New Modules

| Module | Import | Purpose |
|--------|--------|---------|
| Guardrails | `from effgen.guardrails import ...` | Safety & validation |
| RAG | `from effgen.rag import ...` | Retrieval Augmented Generation |
| Evaluation | `from effgen.eval import ...` | Benchmarking & regression |
| Domains | `from effgen.domains import ...` | Domain keyword expansion |
| Cache | `from effgen.cache import ...` | Prompt & result caching |
| Debug | `from effgen.debug import ...` | Interactive debugging |
| Hardware | `from effgen.hardware import ...` | Platform detection |
| Client SDK | `from effgen.client import ...` | API client |

### New Tools (17 Added)

Finance: `StockPriceTool`, `CurrencyConverterTool`, `CryptoTool`
Data Science: `DataFrameTool`, `PlotTool`, `StatsTool`
DevOps: `GitTool`, `DockerTool`, `SystemInfoTool`, `HTTPTool`
Knowledge: `ArxivTool`, `StackOverflowTool`, `GitHubTool`, `WolframAlphaTool`
Communication: `EmailDraftTool`, `SlackDraftTool`, `NotificationTool`

All imported from `effgen.tools.builtin`.

### New Optional Dependencies

```bash
pip install effgen[rag]       # sentence-transformers, faiss-cpu
pip install effgen[finance]   # yfinance
pip install effgen[data]      # matplotlib, plotly
pip install effgen[eval]      # rouge-score, nltk
pip install effgen[gguf]      # llama-cpp-python
pip install effgen[mlx]       # MLX for Apple Silicon
pip install effgen[mlx-vlm]   # MLX vision-language models
```

### New CLI Commands

```bash
# Workflows
effgen workflow run pipeline.yaml
effgen workflow validate pipeline.yaml

# Batch execution
effgen batch --input queries.jsonl --output results.jsonl

# Evaluation
effgen eval --suite math --model "Qwen/Qwen2.5-3B-Instruct"
effgen compare --models "model_a,model_b" --suite math

# Model management
effgen models load "Qwen/Qwen2.5-3B-Instruct"
effgen models status
effgen models unload "Qwen/Qwen2.5-3B-Instruct"

# Sessions
effgen sessions list
effgen sessions delete <id>
effgen sessions export <id>

# Debugging
effgen debug --preset math "What is 2+2?"

# Checkpointing
effgen run "Long task" --checkpoint-dir ./checkpoints
effgen resume --checkpoint ./checkpoints/latest.json
```

### API Server v2 Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible chat (new) |
| `POST /v1/completions` | OpenAI-compatible text completion (new) |
| `POST /v1/embeddings` | OpenAI-compatible embeddings (new) |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |
| `WS /ws` | WebSocket streaming |

Model aliases map OpenAI model names to local SLMs (e.g., `gpt-3.5-turbo` → `Qwen2.5-3B-Instruct`).

---

## 0.0.2 → 0.1.0

### New Features
- **Presets**: Use `create_agent("math", model)` for instant agent setup
- **Plugin system**: Distribute tools as installable packages
- **CLI**: `--preset`, `--explain`, `--completion`, `create-plugin` commands
- **API server**: WebSocket streaming, API key auth, rate limiting, metrics
- **Tab completion**: `eval "$(effgen --completion bash)"`

### Breaking Changes
None. All existing `Agent`, `AgentConfig`, and `load_model` APIs remain unchanged.

### New Imports
```python
# Presets (new)
from effgen.presets import create_agent, list_presets

# Plugin system (new)
from effgen.tools.plugin import ToolPlugin, PluginManager, discover_plugins
```

### CLI Changes
```bash
# New commands
effgen presets                              # List available presets
effgen run --preset math "What is 2+2?"     # Use preset
effgen run --explain "..."                  # Show tool reasoning
effgen create-plugin my_tools               # Generate plugin scaffold
effgen --completion bash                    # Print completion script
```

### API Server Changes
- New endpoints: `WS /ws`, `GET /metrics`
- Auth: Set `EFFGEN_API_KEY` environment variable
- Rate limiting: Set `EFFGEN_RATE_LIMIT` to a requests/min cap per client (unset or `0` = disabled)
- `POST /run` now accepts `preset` field
