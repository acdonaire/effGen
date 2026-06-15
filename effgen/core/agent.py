"""
Core Agent implementation for effGen.

The main Agent class with:
- ReAct loop (Reason + Act)
- Tool selection and execution
- Sub-agent integration via router
- Memory management
- Streaming support
- State persistence
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..memory.long_term import (
    ImportanceLevel,
    JSONStorageBackend,
    LongTermMemory,
    MemoryType,
    SQLiteStorageBackend,
)
from ..memory.short_term import MessageRole, ShortTermMemory
from ..models.base import BaseModel, GenerationConfig
from ..models.model_loader import ModelLoader
from ..observability import get_logger as _get_obs_logger
from ..observability.spans import AgentAttrs
from ..observability.tracing import (
    set_span_attribute,
    set_span_error,
    start_agent_run,
)
from ..prompts.agent_system_prompt import AgentSystemPromptBuilder
from ..prompts.tool_prompt_generator import ToolPromptGenerator
from ..tools.base_tool import BaseTool
from ..tools.fallback import ToolFallbackChain
from ..utils.circuit_breaker import CircuitBreaker
from ..utils.prometheus_metrics import metrics as prom_metrics
from ..utils.structured_logging import (
    LogRunContext,
    generate_run_id,
    get_structured_logger,
)
from .execution_tracker import EventType, ExecutionEvent, ExecutionTracker
from .router import RoutingDecision, SubAgentRouter
from .state import AgentState
from .sub_agent_manager import SubAgentManager
from .tool_calling import (
    get_strategy,
)

if TYPE_CHECKING:
    from .messages import Message

logger = logging.getLogger(__name__)
_slog = get_structured_logger(__name__)
# Canonical structured observability logger — emits redacted JSON lines with OTel context
_obs_log = _get_obs_logger(__name__)
from .agent_runtime import (  # noqa: E402
    AgentRuntimeMixin,
    _safe_float_or_none,
    _safe_int_or_none,
    sanitize_final_answer,
)


class AgentMode(Enum):
    """Agent execution modes."""
    SINGLE = "single"  # Single agent execution
    SUB_AGENTS = "sub_agents"  # Use sub-agents for complex tasks
    AUTO = "auto"  # Automatically decide based on router


@dataclass
class AgentConfig:
    """
    Agent configuration.

    Attributes:
        name: Agent name/identifier
        model: Model instance or name
        tools: List of available tools
        system_prompt: System-level instructions
        max_iterations: Maximum tool-use loop iterations
        temperature: Generation temperature
        enable_sub_agents: Enable sub-agent spawning
        enable_memory: Enable memory systems
        enable_streaming: Enable response streaming
        max_context_length: Maximum context window
        router_config: Configuration for sub-agent router
        sub_agent_config: Configuration for sub-agent manager
        model_config: Optional model engine configuration
        require_model: Whether a string model must load at construction time.
            Defaults to True so a typo'd id / missing key fails immediately
            instead of building a working-looking agent that only crashes on
            the first run(). Set False to defer loading (advanced use).
        provider: Optional explicit provider for a string ``model`` (e.g.
            "openai", "cerebras"). Equivalent to the "provider:model" prefix
            and the CLI ``--provider`` flag; resolves bare ids that exist on
            multiple providers.
        raise_on_error: When True, run() raises the typed error on failure
            instead of returning an AgentResponse with success=False. The same
            failure raises regardless of which internal path (direct or tool
            loop) produced it.
    """
    name: str
    model: BaseModel | str
    tools: list[BaseTool] = field(default_factory=list)
    system_prompt: str = "You are a helpful AI assistant."
    max_iterations: int = 10
    temperature: float = 0.7
    enable_sub_agents: bool = True
    enable_memory: bool = True
    enable_streaming: bool = False
    max_context_length: int | None = None
    router_config: dict[str, Any] = field(default_factory=dict)
    sub_agent_config: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] | None = None
    require_model: bool = True
    provider: str | None = None
    raise_on_error: bool = False
    system_prompt_template: str | None = None
    verbose_tools: bool | None = None
    fallback_chain: dict[str, list] | None = None
    enable_fallback: bool = True
    max_sub_agent_depth: int = 3
    tool_calling_mode: str = "auto"  # "auto", "native", "react", "hybrid"
    output_format: str | None = None  # Global default: "json", "yaml", "csv", or None
    output_schema: dict[str, Any] | None = None  # Global default JSON Schema
    guardrails: Any = None  # GuardrailChain, preset name (str), or None
    memory_config: dict[str, Any] = field(default_factory=lambda: {
        "short_term_max_tokens": 4096,
        "short_term_max_messages": 100,
        "long_term_backend": "sqlite",
        "long_term_persist_path": None,
        "auto_summarize": True,
    })
    # Multi-model support
    models: list[BaseModel | str] | None = None  # Additional models for routing
    speculative_execution: bool = False  # Run on 2 models, return first success
    # Human-in-the-loop
    approval_callback: Callable[[str, str], bool] | None = None
    approval_mode: str = "never"  # "always", "first_time", "never", "dangerous_only"
    approval_timeout: float = 0.0  # seconds; 0 = wait forever
    clarification_callback: Callable[[str, list[str]], int] | None = None
    input_callback: Callable[[str], str] | None = None
    # Prompt caching: keep the system prompt at a fixed position so OpenAI
    # can cache the prefix automatically across sequential calls.
    stable_system_prompt: bool = True
    # Anthropic explicit prompt caching via cache_control markers.
    # cache_system_prompt=True: Agent marks the last block of the system message
    #   with cache_control so it is cached across requests.
    # cache_tools=True: Agent marks the last tool spec with cache_control.
    # These flags have no effect when the model is not an AnthropicAdapter.
    cache_system_prompt: bool = True
    cache_tools: bool = True


@dataclass
class AgentResponse:
    """
    Response from agent execution.

    Attributes:
        output: Final output text
        success: Whether execution succeeded
        mode: Execution mode used
        iterations: Number of iterations performed
        tool_calls: Number of tool calls made
        tokens_used: Total tokens consumed
        execution_time: Time taken in seconds
        execution_trace: Full execution trace
        execution_tree: Hierarchical execution tree
        routing_decision: Routing decision (if sub-agents used)
        metadata: Additional metadata. Always includes ``reason``, one of:

            - ``"final_answer"`` — the model produced an answer (``success=True``).
              A finer ``answer_source`` may also be present (e.g.
              ``loop_detected``, ``direct_calculator_result``) for heuristically
              recovered answers.
            - ``"max_iterations_partial"`` — the tool loop hit its iteration cap
              but a usable partial answer was recovered (``success=True``,
              ``partial=True``).
            - ``"max_iterations_exhausted"`` — the loop gave up with no answer
              (``success=False``).
            - ``"generation_failed"`` — the model/provider call failed
              (``success=False``); ``metadata["error"]`` is a structured dict
              ``{type, category, provider, model, message, retryable}`` and is
              identical whether the failure happened on the direct or tool path.

            Success rule: ``success`` is ``True`` only when a real answer was
            produced (``final_answer`` / ``max_iterations_partial``); it is
            never ``True`` with empty output.
    """
    output: str
    success: bool = True
    mode: AgentMode = AgentMode.SINGLE
    iterations: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    execution_time: float = 0.0
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    execution_tree: dict[str, Any] = field(default_factory=dict)
    routing_decision: RoutingDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    citations: list[Any] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """The answer text — so ``print(result)`` shows the answer, not a repr.

        The full structured view stays available via ``repr(result)`` and
        :meth:`to_dict`.
        """
        return self.output if self.output is not None else ""

    def __repr__(self) -> str:
        """A detailed-but-compact developer view.

        The default dataclass repr dumps the entire execution trace and tree,
        which is unreadable. This keeps the useful structured fields and a
        truncated answer preview, and summarizes the trace by length.
        """
        preview = (self.output or "").replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "…"
        reason = self.metadata.get("reason") if isinstance(self.metadata, dict) else None
        return (
            f"AgentResponse(success={self.success}, output={preview!r}, "
            f"mode={self.mode.value!r}, iterations={self.iterations}, "
            f"tool_calls={self.tool_calls}, tokens_used={self.tokens_used}, "
            f"execution_time={self.execution_time:.2f}s, "
            f"trace_steps={len(self.execution_trace)}, reason={reason!r})"
        )

    def _repr_html_(self) -> str:
        """Rich HTML card for Jupyter/IPython (answer + metrics + step trace)."""
        from effgen.ui import response_html

        return response_html(self)

    def show(self, console: Any = None) -> "AgentResponse":
        """Print a compact human view: the answer plus a one-line metric footer.

        Renders markdown, fenced code, and tables in the answer. Returns
        ``self`` so it can be chained. Use :meth:`trace` for the full reasoning.
        """
        from effgen.ui import response_show

        response_show(self, console=console)
        return self

    def trace(self, console: Any = None) -> "AgentResponse":
        """Print the full step-by-step reasoning trace. Returns ``self``."""
        from effgen.ui import response_trace

        response_trace(self, console=console)
        return self

    @property
    def text(self) -> str:
        """Read-only alias for :attr:`output` (familiar from other SDKs)."""
        return self.output

    @property
    def content(self) -> str:
        """Read-only alias for :attr:`output` (familiar from other SDKs)."""
        return self.output

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "success": self.success,
            "mode": self.mode.value,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "execution_time": round(self.execution_time, 2),
            "execution_trace": self.execution_trace,
            "execution_tree": self.execution_tree,
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision else None,
            "metadata": self.metadata,
            "citations": [c.to_dict() if hasattr(c, "to_dict") else c for c in self.citations],
            "sources": self.sources,
        }


from .agent_generation import AgentGenerationMixin  # noqa: E402
from .agent_react import AgentReActMixin  # noqa: E402


class Agent(AgentGenerationMixin, AgentReActMixin, AgentRuntimeMixin):
    """
    Main Agent implementation with ReAct loop and sub-agent support.

    The agent can:
    - Execute tasks using ReAct (Reason + Act) pattern
    - Intelligently spawn sub-agents for complex tasks
    - Use tools to interact with external systems
    - Manage conversation memory
    - Stream responses
    - Save/load state
    """

    # Default ReAct prompt template
    REACT_PROMPT_TEMPLATE = """You are a helpful AI assistant that can reason step-by-step and use tools.
{conversation_history}
Available tools:
{tools_description}

IMPORTANT: If there is previous conversation context above, use that information to answer questions about past interactions.

Use the following format:

Question: the input question or task
Thought: think step-by-step about what to do next
Action: the tool to use (or "Final Answer" when ready to respond)
Action Input: the input for the tool
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the complete response to the original question

IMPORTANT: You do NOT have to use a tool for every question.
If you can answer directly from your knowledge or from the conversation history above, skip the Action step entirely:
Thought: I can answer this directly without any tools.
Final Answer: [your answer here]
Only use tools when you NEED external computation, data, or system access.

Example (no tool needed):
Question: Tell me a joke about programming.
Thought: This is a creative request. I can answer directly without tools.
Final Answer: Why do programmers prefer dark mode? Because light attracts bugs!

Begin!

Question: {task}
{scratchpad}"""

    def __init__(self, config: AgentConfig, session_id: str | None = None):
        """
        Initialize agent.

        Args:
            config: Agent configuration
            session_id: Optional persistent session id. If provided, the
                agent loads/creates a Session in ~/.effgen/sessions/ and
                appends each run() turn to it.
        """
        self.config = config
        self.name = config.name
        self._closed = False

        # Persistent session
        self._session_id = session_id
        self.session = None
        if session_id:
            from .session import Session as _Session
            self.session = _Session.load_or_create(session_id, agent_name=self.name)

        # Background task runner, loaded lazily.
        self._bg_runner = None

        # Last checkpoint info
        self._last_checkpoint_id: str | None = None

        # Model initialization
        self.model_loader = ModelLoader()
        if isinstance(config.model, BaseModel):
            # Model instance provided directly
            self.model = config.model
            self.model_name = getattr(config.model, 'model_name', 'custom')
        elif isinstance(config.model, str):
            # Model name provided - load it
            self.model_name = config.model
            load_kwargs: dict[str, Any] = {}
            if config.provider:
                load_kwargs["provider"] = config.provider
            try:
                logger.debug(f"Loading model: {self.model_name}")
                self.model = self.model_loader.load_model(
                    self.model_name,
                    engine_config=config.model_config,
                    **load_kwargs,
                )
                logger.debug(f"Model loaded successfully: {self.model_name}")
            except Exception as e:
                self.model = None
                if config.require_model:
                    # Fail fast on a typo'd id / missing key rather than
                    # returning a working-looking agent (see AgentConfig docs).
                    logger.error(f"Failed to load model '{self.model_name}': {e}")
                    raise RuntimeError(
                        f"Failed to load model '{self.model_name}': {e}"
                    ) from e
                logger.warning(
                    f"Model loading failed for '{self.model_name}' and "
                    "require_model=False; the agent will fail on first inference. "
                    "Set require_model=True (the default) to fail fast at construction."
                )
        else:
            # No model provided
            self.model_name = None
            self.model = None
            if config.require_model and config.model is None:
                # Fail at construction (not on the first run()) so a model-less
                # agent doesn't look usable. Set require_model=False to defer
                # model assignment (advanced/sub-agent use).
                raise ValueError(
                    f"Agent '{config.name}' was created without a model. "
                    "Pass a model id or instance, e.g. "
                    "AgentConfig(name=..., model='gpt-5-nano') or "
                    "model='Qwen/Qwen2.5-1.5B-Instruct'. "
                    "Run `effgen models list` to see options, or `effgen doctor` "
                    "to check which providers are usable. "
                    "(Set require_model=False to defer loading.)"
                )

        # Multi-model router
        self._model_router = None
        self._all_models: list[BaseModel] = []
        if self.model is not None:
            self._all_models.append(self.model)
        if config.models:
            for m in config.models:
                if isinstance(m, BaseModel):
                    self._all_models.append(m)
                elif isinstance(m, str):
                    try:
                        loaded = self.model_loader.load_model(
                            m, engine_config=config.model_config
                        )
                        self._all_models.append(loaded)
                    except Exception as e:
                        logger.warning("Failed to load additional model '%s': %s", m, e)

        if len(self._all_models) > 1:
            from ..models.router import ModelRouter
            self._model_router = ModelRouter(models=self._all_models)
            logger.info(
                "Model router enabled with %d models: %s",
                len(self._all_models),
                [getattr(m, 'model_name', str(m)) for m in self._all_models],
            )

        self._speculative_execution = config.speculative_execution

        # Tools
        self.tools = {tool.name: tool for tool in config.tools}

        # Validate OpenAI native tool compatibility at init time
        self._validate_native_tool_compatibility(config.tools)

        # Tool calling strategy
        self._tool_calling_strategy = get_strategy(
            mode=config.tool_calling_mode,
            model=self.model,
        )
        logger.info(f"Tool calling strategy: {self._tool_calling_strategy.name}")

        # Tool prompt generator for enhanced ReAct prompts
        self._tool_prompt_generator = ToolPromptGenerator(
            tools=config.tools,
            model_name=self.model_name or "",
        )

        # Determine verbose_tools setting: auto-detect from model size if not set
        if config.verbose_tools is not None:
            self._verbose_tools = config.verbose_tools
        else:
            self._verbose_tools = self._auto_detect_verbose()

        # Tool fallback chain
        self._fallback_chain = ToolFallbackChain(
            custom_chains=config.fallback_chain
        )
        self._enable_fallback = config.enable_fallback

        # Circuit breaker for tool failures
        self._circuit_breaker = CircuitBreaker()

        # Human-in-the-loop approval manager
        from .human_loop import ApprovalManager, ApprovalMode
        try:
            _approval_mode = ApprovalMode(config.approval_mode)
        except ValueError:
            _approval_mode = ApprovalMode.NEVER
        self._approval_manager = ApprovalManager(
            mode=_approval_mode,
            callback=config.approval_callback,
            timeout=config.approval_timeout,
        )

        # Auto-generate system prompt if tools are present and using default prompt
        self._system_prompt_builder = AgentSystemPromptBuilder(
            model_name=self.model_name or "",
        )
        if config.tools and config.system_prompt == "You are a helpful AI assistant.":
            self.config.system_prompt = self._build_system_prompt()

        # State management
        self.state = AgentState(agent_id=self.name)

        # Per-run accumulator of retrieved evidence (RAG/search). Populated by
        # _execute_tool_once when a retrieval/search tool returns ranked passages,
        # then surfaced on AgentResponse.sources / .citations at the end of run().
        self._collected_citations: list[dict[str, Any]] = []

        # Sub-agent components
        self._current_depth = 0
        self.router = None
        self.sub_agent_manager = None
        if config.enable_sub_agents:
            self.router = SubAgentRouter(
                config=config.router_config,
                llm_client=self  # Pass self as LLM client
            )
            self.sub_agent_manager = SubAgentManager(
                parent_agent=self,
                config=config.sub_agent_config
            )

        # Execution tracker
        self.execution_tracker = ExecutionTracker()

        # Memory system
        mem_cfg = config.memory_config or {}
        stm_max_tokens = mem_cfg.get("short_term_max_tokens", 4096)
        stm_max_messages = mem_cfg.get("short_term_max_messages", 100)
        self.short_term_memory = ShortTermMemory(
            max_tokens=stm_max_tokens,
            max_messages=stm_max_messages,
            summarization_threshold=mem_cfg.get("summarization_threshold", 0.8),
            keep_recent_messages=mem_cfg.get("keep_recent_messages", 4),
            model=self.model,
        )

        # Long-term memory (optional, requires persist path)
        self.long_term_memory: LongTermMemory | None = None
        if config.enable_memory:
            persist_path = mem_cfg.get("long_term_persist_path")
            if persist_path:
                import os
                persist_path = os.path.expanduser(persist_path)
                backend_type = mem_cfg.get("long_term_backend", "sqlite")
                if backend_type == "sqlite":
                    backend = SQLiteStorageBackend(
                        os.path.join(persist_path, "long_term.db")
                    )
                else:
                    backend = JSONStorageBackend(
                        os.path.join(persist_path, "long_term.json")
                    )
                self.long_term_memory = LongTermMemory(backend=backend)
                self.long_term_memory.start_session(name=self.name)

        # Guardrails
        self._guardrail_chain = self._resolve_guardrails(config.guardrails)

        # Hydrate short-term memory from persistent session if loaded
        if self.session and self.session.messages:
            for m in self.session.messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    self.short_term_memory.add_user_message(content)
                elif role == "assistant":
                    self.short_term_memory.add_assistant_message(content)

    def _validate_native_tool_compatibility(self, tools: list) -> None:
        """Raise ToolIncompatibleError for provider-specific tools used with the wrong model."""
        from ..models.errors import ToolIncompatibleError

        # --- OpenAI native tools ---
        try:
            from ..models.openai_adapter import OpenAIAdapter
            from ..tools.builtin.openai_native import OpenAINativeTool

            openai_native = [t for t in tools if isinstance(t, OpenAINativeTool)]
            if openai_native:
                is_openai = isinstance(self.model, OpenAIAdapter)
                if not is_openai and not isinstance(self.model, BaseModel):
                    # Only guess from the model name when the object is not a
                    # known effGen adapter (e.g. a duck-typed custom model). A
                    # concrete non-OpenAI adapter is authoritative: do NOT match
                    # on the "gpt-" prefix, or Cerebras' ``gpt-oss-*`` (and other
                    # OpenAI-compatible providers) would be mistaken for OpenAI
                    # and silently skip this fail-closed guard.
                    model_name_str = getattr(self.model, "model_name", "") or ""
                    provider = getattr(self.model, "_provider", "") or ""
                    is_openai = (
                        "openai" in provider.lower()
                        or model_name_str.startswith(("gpt-", "o1", "o3", "o4"))
                    )
                if not is_openai:
                    bad = openai_native[0]
                    current_model = getattr(self.model, "model_name", str(self.model)) if self.model else "None"
                    raise ToolIncompatibleError(
                        tool_name=bad.name,
                        model_name=current_model,
                        reason=(
                            "OpenAI native tools (web_search, code_interpreter, file_search) "
                            "are executed server-side by OpenAI and require an OpenAIAdapter. "
                            f"Current model: '{current_model}'. "
                            "Switch to an OpenAI model or remove the native tool."
                        ),
                    )
        except ImportError:
            pass

        # --- Gemini native tools ---
        try:
            from ..models.gemini_adapter import GeminiAdapter
            from ..tools.builtin.gemini_native import GeminiNativeTool

            gemini_native = [t for t in tools if isinstance(t, GeminiNativeTool)]
            if gemini_native:
                is_gemini = isinstance(self.model, GeminiAdapter)
                if not is_gemini and not isinstance(self.model, BaseModel):
                    # Name guess only for unknown duck-typed objects; a concrete
                    # non-Gemini adapter is authoritative (see OpenAI note above).
                    model_name_str = getattr(self.model, "model_name", "") or ""
                    is_gemini = model_name_str.startswith("gemini")
                if not is_gemini:
                    bad = gemini_native[0]
                    current_model = getattr(self.model, "model_name", str(self.model)) if self.model else "None"
                    raise ToolIncompatibleError(
                        tool_name=bad.name,
                        model_name=current_model,
                        reason=(
                            "Gemini native tools (google_search, url_context, code_execution) "
                            "are executed server-side by Google and require a GeminiAdapter. "
                            f"Current model: '{current_model}'. "
                            "Switch to a Gemini model or remove the native tool."
                        ),
                    )
        except ImportError:
            pass

        # --- Anthropic native tools (computer-use) ---
        try:
            from ..models.anthropic_adapter import AnthropicAdapter
            from ..tools.builtin.anthropic_native import AnthropicNativeTool

            anthropic_native = [t for t in tools if isinstance(t, AnthropicNativeTool)]
            if anthropic_native:
                is_anthropic = isinstance(self.model, AnthropicAdapter)
                if not is_anthropic and not isinstance(self.model, BaseModel):
                    # Name guess only for unknown duck-typed objects; a concrete
                    # non-Anthropic adapter is authoritative (see OpenAI note above).
                    model_name_str = getattr(self.model, "model_name", "") or ""
                    is_anthropic = model_name_str.startswith("claude")
                if not is_anthropic:
                    bad = anthropic_native[0]
                    current_model = getattr(self.model, "model_name", str(self.model)) if self.model else "None"
                    raise ToolIncompatibleError(
                        tool_name=bad.name,
                        model_name=current_model,
                        reason=(
                            "Anthropic native computer-use tools (bash, text_editor, computer) "
                            "are executed server-side by Anthropic and require an AnthropicAdapter. "
                            f"Current model: '{current_model}'. "
                            "Switch to an Anthropic model or remove the native tool."
                        ),
                    )
        except ImportError:
            pass


    def _get_anthropic_system(self) -> str | list | None:
        """
        Return the system prompt for Anthropic requests.

        When ``AgentConfig.cache_system_prompt=True`` and the model is an
        ``AnthropicAdapter``, the system prompt is returned as a list of
        content blocks with ``cache_control`` on the last block so that it is
        cached across sequential requests.
        """
        system = self.config.system_prompt if self.config.stable_system_prompt else None
        if system is None:
            return None
        try:
            from ..models.anthropic_adapter import AnthropicAdapter
            from ..models.anthropic_cache import apply_cache_to_system
        except ImportError:
            return system
        if isinstance(self.model, AnthropicAdapter) and self.config.cache_system_prompt:
            return apply_cache_to_system(system)
        return system

    def _get_anthropic_tools(self, tools: list[dict]) -> list[dict]:
        """
        Apply ``cache_control`` to the last tool spec when appropriate.

        Only active when ``AgentConfig.cache_tools=True`` and the model is an
        ``AnthropicAdapter``.
        """
        if not tools:
            return tools
        try:
            from ..models.anthropic_adapter import AnthropicAdapter
            from ..models.anthropic_cache import apply_cache_to_last_tool
        except ImportError:
            return tools
        if isinstance(self.model, AnthropicAdapter) and self.config.cache_tools:
            return apply_cache_to_last_tool(tools)
        return tools

    def _build_system_prompt(self) -> str:
        """Build a dynamic system prompt based on agent configuration and tools."""
        return self._system_prompt_builder.build(
            tools=self.config.tools,
            agent_name=self.name,
            base_system_prompt=None,  # Will generate default role
            enable_fallback=self._enable_fallback,
            verbose=self._verbose_tools,
        )

    def _auto_detect_verbose(self) -> bool:
        """Auto-detect whether to use verbose tool descriptions based on model size."""
        name = (self.model_name or "").lower()
        # Check for known small models (< 3B) -> full verbose with examples
        # Check for medium models (3B-7B) -> verbose without examples
        # Check for large models (> 7B) or API models -> compact
        for indicator in ["0.5b", "1b", "1.5b", "2b"]:
            if indicator in name:
                return True
        for indicator in ["3b", "4b", "5b", "7b"]:
            if indicator in name:
                return True
        # API models
        for indicator in ["gpt", "claude", "gemini"]:
            if indicator in name:
                return False
        # Default: verbose (safe for SLMs)
        return True

    def run(self,
            task: "str | Message | list[Any]",
            mode: AgentMode = AgentMode.AUTO,
            context: dict[str, Any] | None = None,
            output_schema: dict[str, Any] | None = None,
            output_model: Any = None,
            inputs: list[Any] | None = None,
            **kwargs) -> AgentResponse:
        """
        Execute a task.

        Args:
            task: Task description. Accepts a plain ``str``, a multimodal
                ``Message``, or a ``list[ContentPart]``; text is extracted and
                any image/audio/video parts are routed through the multimodal
                path.
            mode: Execution mode (single, sub_agents, auto)
            context: Optional context
            output_schema: JSON Schema dict — when provided, the final output
                is guaranteed to be valid JSON matching this schema.
            output_model: Pydantic BaseModel class — when provided, output is
                validated and the parsed instance is stored in
                ``response.metadata["parsed"]``.
            inputs: Optional list of multimodal content parts created by
                ``image_from``, ``audio_from``, or ``video_from``. When present,
                the agent sends a structured Message directly to the model.
            **kwargs: Additional arguments (debug=True for DebugTrace)

        Returns:
            AgentResponse with results
        """
        start_time = time.time()
        context = context or {}

        # Accept str | Message | list[ContentPart]; route media to the
        # multimodal `inputs` path. Raises a clear TypeError otherwise.
        task, inputs = self._coerce_task_input(task, inputs)
        if inputs is not None:
            kwargs["inputs"] = inputs
        self._current_depth = 0  # Reset depth at the start of each top-level run()
        self._collected_citations = []  # Reset retrieved-evidence accumulator
        debug = kwargs.pop("debug", False)
        run_id = generate_run_id()
        # Capture checkpoint args here so the outer run() can use
        # them for the final-checkpoint write even after _run_single_agent
        # consumes them from kwargs.
        _outer_ckpt_dir = kwargs.get("checkpoint_dir") or context.get("checkpoint_dir")

        # Metrics: track request
        labels = {"agent_name": self.name}
        prom_metrics.total_requests.inc(labels=labels)
        prom_metrics.active_agents.inc(labels=labels)

        # Pre-run input guardrail check
        if self._guardrail_chain is not None:
            from ..guardrails.base import GuardrailPosition
            gr = self._guardrail_chain.check(task, position=GuardrailPosition.INPUT)
            if not gr.passed:
                prom_metrics.active_agents.dec(labels=labels)
                return AgentResponse(
                    output=f"Input blocked by guardrail: {gr.reason}",
                    success=False,
                    execution_time=time.time() - start_time,
                    metadata={"guardrail_blocked": True, "guardrail_reason": gr.reason},
                )
            if gr.modified_content is not None:
                task = gr.modified_content

        # Resolve structured output schema
        effective_schema = output_schema or self.config.output_schema
        if output_model is not None and effective_schema is None:
            from .structured_output import pydantic_model_to_schema
            effective_schema = pydantic_model_to_schema(output_model)

        # Track task start
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.TASK_START,
            agent_id=self.name,
            message=f"Starting task: {self._extract_task_preview(task, 100)}...",
            data={"task": task, "mode": mode.value}
        ))

        # Wrap entire run in tracing span + structured log context
        _task_preview = self._extract_task_preview(task, 200)
        with start_agent_run(preset=self.name, task=task, run_id=run_id) as _span, \
             LogRunContext(run_id=run_id, agent_name=self.name):
            _slog.agent_event(self.name, "task_start", task=_task_preview, mode=mode.value, run_id=run_id)
            _obs_log.agent_event("run.started", agent=self.name, task=_task_preview, mode=mode.value, run_id=run_id)

            try:
                # Pass debug flag through kwargs
                if debug:
                    kwargs["_debug"] = True
                    kwargs["_run_id"] = run_id

                # Determine execution mode
                if mode == AgentMode.AUTO and self.config.enable_sub_agents:
                    # Use router to decide
                    routing_decision = self.router.route(task, context)

                    if routing_decision.use_sub_agents:
                        response = self._run_with_sub_agents(task, routing_decision, context, **kwargs)
                    else:
                        response = self._run_single_agent(task, context, **kwargs)
                elif mode == AgentMode.SUB_AGENTS and self.config.enable_sub_agents:
                    # Force sub-agent mode
                    routing_decision = self.router.route(task, context)
                    response = self._run_with_sub_agents(task, routing_decision, context, **kwargs)
                else:
                    # Single agent mode
                    response = self._run_single_agent(task, context, **kwargs)

                # Strip any internal scaffolding from the answer before it is
                # used downstream (structured output, guardrails, memory). This
                # is the single funnel covering every execution path; individual
                # assembly sites also sanitize so direct callers stay clean.
                if response.success and isinstance(response.output, str):
                    response.output = sanitize_final_answer(response.output)
                    # If this run used a retrieval tool and the model echoed the
                    # raw result dict as its answer (small models sometimes paste
                    # the tool observation), render it as readable passage text.
                    if self._collected_citations:
                        response.output = self._humanize_observation(response.output)

                # Apply structured output constraint if requested
                if effective_schema and response.success and response.output:
                    response = self._apply_structured_output(
                        response, effective_schema, output_model, task,
                    )

                # Post-run output guardrail check
                if self._guardrail_chain is not None and response.success and response.output:
                    from ..guardrails.base import GuardrailPosition as _GP
                    gr = self._guardrail_chain.check(response.output, position=_GP.OUTPUT)
                    if not gr.passed:
                        response.output = f"Output blocked by guardrail: {gr.reason}"
                        response.success = False
                        response.metadata["guardrail_blocked"] = True
                        response.metadata["guardrail_reason"] = gr.reason
                    elif gr.modified_content is not None:
                        response.output = gr.modified_content

                # Add execution metadata
                response.execution_time = time.time() - start_time
                response.execution_trace = self.execution_tracker.get_trace()
                response.execution_tree = self.execution_tracker.generate_execution_tree()
                response.metadata["run_id"] = run_id

                # Surface retrieved evidence: if the run consulted a knowledge
                # base / search tool, expose its passages as sources + inline
                # citations on the response (fills only what's still empty).
                self._attach_citations(response)

                # Track completion
                self.execution_tracker.track_event(ExecutionEvent(
                    type=EventType.TASK_COMPLETE,
                    agent_id=self.name,
                    message=f"Task completed in {response.execution_time:.2f}s",
                    data={
                        "execution_time": response.execution_time,
                        "tokens_used": response.tokens_used,
                        "tool_calls": response.tool_calls
                    }
                ))

                # Metrics: record latency and tokens
                prom_metrics.response_latency.observe(response.execution_time, labels=labels)
                if response.tokens_used:
                    prom_metrics.token_usage.observe(response.tokens_used, labels=labels)
                    prom_metrics.tokens_used.inc(response.tokens_used, labels=labels)

                # Tracing span attributes (using span constants)
                set_span_attribute(AgentAttrs.RUN_ID, run_id or "")
                set_span_attribute("effgen.tokens_used", response.tokens_used)
                set_span_attribute("effgen.tool_calls", response.tool_calls)
                set_span_attribute("effgen.success", response.success)
                set_span_attribute("effgen.latency", response.execution_time)

                _slog.agent_event(
                    self.name, "task_complete",
                    latency=response.execution_time,
                    tokens=response.tokens_used,
                    tool_calls=response.tool_calls,
                    success=response.success,
                )
                _obs_log.agent_event(
                    "run.completed",
                    agent=self.name,
                    run_id=run_id,
                    latency_ms=round(response.execution_time * 1000, 1),
                    tokens=response.tokens_used,
                    tool_calls=response.tool_calls,
                    success=response.success,
                )
                self._record_dashboard_run(response)

                # Store conversation in short-term memory for context retention
                if response.success and response.output:
                    self.short_term_memory.add_user_message(task)
                    self.short_term_memory.add_assistant_message(response.output)
                    logger.debug(
                        f"Stored conversation turn in memory "
                        f"(total: {self.short_term_memory.total_messages_added} messages)"
                    )

                    # Persist important facts to long-term memory if available
                    if self.long_term_memory and response.tool_calls > 0:
                        self.long_term_memory.add_memory(
                            content=f"Q: {task}\nA: {response.output}",
                            memory_type=MemoryType.CONVERSATION,
                            importance=ImportanceLevel.MEDIUM,
                            tags=["conversation"],
                        )

                    # Persist to session
                    if self.session is not None:
                        self.session.add_user_message(task)
                        self.session.add_assistant_message(response.output)
                        try:
                            self.session.save()
                        except Exception as _e:
                            logger.warning("Failed to save session: %s", _e)

                # Final checkpoint
                ckpt_dir = _outer_ckpt_dir
                if ckpt_dir:
                    try:
                        from .checkpoint import CheckpointManager
                        mgr = CheckpointManager(ckpt_dir)
                        cp = CheckpointManager.snapshot_agent(
                            self,
                            task=task,
                            iteration=getattr(response, "iterations", 0),
                            scratchpad="",
                            partial_output=response.output,
                            tool_calls=response.tool_calls,
                            tokens_used=response.tokens_used,
                            metadata={"final": True, "success": response.success},
                        )
                        self._last_checkpoint_id = mgr.save(cp)
                        response.metadata["checkpoint_id"] = self._last_checkpoint_id
                    except Exception as _e:
                        logger.warning("Failed to save final checkpoint: %s", _e)

                # raise_on_error contract: surface a typed error on any failure
                # instead of returning success=False (same on both run paths).
                if not response.success and self.config.raise_on_error:
                    raise self._reconstruct_error(response.metadata)

                return response

            except Exception as e:
                # When raise_on_error is set, propagate the typed error rather
                # than swallowing it into a success=False response.
                if self.config.raise_on_error:
                    prom_metrics.errors.inc(labels=labels)
                    set_span_error(e)
                    raise
                # Track failure
                self.execution_tracker.track_event(ExecutionEvent(
                    type=EventType.TASK_FAILED,
                    agent_id=self.name,
                    message=f"Task failed: {str(e)}",
                    data={"error": str(e)}
                ))
                prom_metrics.errors.inc(labels=labels)
                set_span_error(e)
                # Build a structured, redacted error record so this catch-all
                # surfaces the same metadata["error"]={type,provider,model,...}
                # shape (and redacted message) as the inner failure paths.
                detail = self._build_error_detail(e, self.model)
                redacted_msg = detail["message"]
                _slog.agent_event(self.name, "task_failed", level=logging.ERROR, error=redacted_msg)
                _obs_log.agent_event("run.failed", level=logging.ERROR, agent=self.name, run_id=run_id, error=redacted_msg)
                response = AgentResponse(
                    output=f"Error: {redacted_msg}",
                    success=False,
                    execution_time=time.time() - start_time,
                    execution_trace=self.execution_tracker.get_trace(),
                    metadata={"reason": "run_failed", "error": detail, "run_id": run_id}
                )
                self._record_dashboard_run(response, error=redacted_msg)
                return response

            finally:
                prom_metrics.active_agents.dec(labels=labels)

    def _record_dashboard_run(
        self,
        response: AgentResponse,
        *,
        error: str | None = None,
    ) -> None:
        """Best-effort process-local run log used by the dashboard."""
        try:
            from effgen.observability.run_log import record_run

            metadata = response.metadata or {}
            cost = metadata.get("cost_usd", metadata.get("cost"))
            output_tokens = metadata.get("output_tokens", metadata.get("completion_tokens"))
            if output_tokens is None and response.tokens_used:
                output_tokens = response.tokens_used
            input_tokens = metadata.get("input_tokens", metadata.get("prompt_tokens"))
            record_run(
                model=str(getattr(self, "model_name", None) or "unknown"),
                input_tokens=_safe_int_or_none(input_tokens),
                output_tokens=_safe_int_or_none(output_tokens),
                duration_s=response.execution_time,
                cost_usd=_safe_float_or_none(cost),
                error=error if error is not None else (None if response.success else response.output[:200]),
            )
        except Exception:  # noqa: BLE001 - dashboard logging must not break runs
            logger.debug("Dashboard run logging failed", exc_info=True)

    def run_batch(
        self,
        queries: list[str],
        max_concurrency: int = 5,
        batch_size: int = 0,
        retry_failed: int = 1,
        timeout_per_item: float = 120.0,
        progress_callback: Callable[[int, int], None] | None = None,
        **run_kwargs: Any,
    ) -> Any:
        """Run multiple queries in parallel through this agent.

        Convenience wrapper around :class:`~effgen.core.batch.BatchRunner`.

        Args:
            queries: List of query strings to execute.
            max_concurrency: Maximum number of concurrent agent runs.
            batch_size: Process queries in batches of this size (0 = all at once).
            retry_failed: Number of retries for failed queries.
            timeout_per_item: Timeout per query in seconds.
            progress_callback: Called with (completed, total) after each query.
            **run_kwargs: Extra keyword arguments forwarded to ``self.run()``.

        Returns:
            BatchResult containing all AgentResponse objects in input order.
        """
        from .batch import BatchConfig, BatchRunner

        config = BatchConfig(
            max_concurrency=max_concurrency,
            batch_size=batch_size,
            retry_failed=retry_failed,
            timeout_per_item=timeout_per_item,
            progress_callback=progress_callback,
        )
        runner = BatchRunner(self)
        return runner.run(queries, config=config, **run_kwargs)




















    # Tool tags / categories whose results carry retrievable evidence we can
    # turn into AgentResponse sources + inline citations.
    _RETRIEVAL_TOOL_TAGS = frozenset({
        "retrieval", "rag", "knowledge-base", "search", "web-search",
        "wikipedia", "documents", "semantic",
    })

















    def _get_tools_description(self, verbose: bool | None = None) -> str:
        """
        Get formatted description of available tools.

        Args:
            verbose: Override verbosity. If None, uses self._verbose_tools.

        Returns:
            Formatted tools description string.
        """
        if not self.tools:
            return "No tools available."

        use_verbose = verbose if verbose is not None else self._verbose_tools
        return self._tool_prompt_generator.generate_tools_section(verbose=use_verbose)

    def _format_conversation_history(self, max_turns: int = 25) -> str:
        """
        Format conversation history for inclusion in prompt.

        Uses ShortTermMemory to retrieve recent messages, including
        summaries of older messages when available.

        Args:
            max_turns: Maximum number of previous turns (user+assistant pairs)

        Returns:
            Formatted conversation history string
        """
        # Include summaries of older messages first
        summaries = self.short_term_memory.summaries
        messages = self.short_term_memory.get_recent_messages(n=max_turns * 2)
        if not messages and not summaries:
            return ""

        history = "\n\n=== Previous Conversation Context ===\n"

        # Add summaries if they exist (these cover older, summarized turns)
        if summaries:
            for summary in summaries:
                history += f"[Earlier context summary: {summary.summary}]\n\n"

        turn_num = 0
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == MessageRole.USER:
                turn_num += 1
                history += f"[Turn {turn_num}]\n"
                history += f"User: {msg.content}\n"
                # Check if next message is assistant
                if i + 1 < len(messages) and messages[i + 1].role == MessageRole.ASSISTANT:
                    # Truncate long assistant responses to save tokens
                    assistant_content = messages[i + 1].content
                    if len(assistant_content) > 300:
                        assistant_content = assistant_content[:300] + "..."
                    history += f"Assistant: {assistant_content}\n\n"
                    i += 2
                    continue
                else:
                    history += "\n"
            i += 1

        history += "=== End of Previous Context ===\n"
        return history if turn_num > 0 or summaries else ""

    def add_tool(self, tool: BaseTool):
        """
        Add a tool to the agent.

        Args:
            tool: Tool instance to add
        """
        self.tools[tool.name] = tool

    def remove_tool(self, tool_name: str):
        """
        Remove a tool from the agent.

        Args:
            tool_name: Name of tool to remove
        """
        if tool_name in self.tools:
            del self.tools[tool_name]

    def reset_memory(self):
        """Clear conversation and tool history."""
        self.state.clear_history()
        self.short_term_memory.clear()
        if self.long_term_memory:
            self.long_term_memory.end_session()
            self.long_term_memory.start_session(name=self.name)

    def save_state(self, filepath: str, format: str = "json"):
        """
        Save agent state.

        Args:
            filepath: Path to save to
            format: Format (json or pickle)
        """
        self.state.save(filepath, format)

    def load_state(self, filepath: str, format: str = "json"):
        """
        Load agent state.

        Args:
            filepath: Path to load from
            format: Format (json or pickle)
        """
        self.state = AgentState.load(filepath, format)

    # ------------------------------------------------------------------ Resume
    def resume(self, checkpoint_id: str | None = None, checkpoint_dir: str = "./checkpoints", **kwargs) -> "AgentResponse":
        """
        Resume execution from a checkpoint.

        Args:
            checkpoint_id: Checkpoint id (or path to a JSON file). If None,
                loads the most recent checkpoint in ``checkpoint_dir``.
            checkpoint_dir: Directory containing checkpoints.
            **kwargs: Additional run() kwargs.

        Returns:
            AgentResponse from continuing the task.
        """
        from .checkpoint import CheckpointManager
        mgr = CheckpointManager(checkpoint_dir)
        cp = mgr.load(checkpoint_id) if checkpoint_id else mgr.load_latest()
        CheckpointManager.restore_to_agent(self, cp)
        # Seed the next run with the saved scratchpad
        kwargs.setdefault("_resume_scratchpad", cp.scratchpad)
        kwargs.setdefault("checkpoint_dir", checkpoint_dir)
        return self.run(cp.task, **kwargs)

    def run_background(self, task: str, priority: int = 5, **run_kwargs) -> str:
        """
        Submit a task to the background runner and return its id.
        """
        if self._bg_runner is None:
            from .background import BackgroundTaskRunner
            self._bg_runner = BackgroundTaskRunner(self, max_workers=1)
        return self._bg_runner.submit(task, priority=priority, **run_kwargs)

    def get_task_status(self, task_id: str):
        """Return the status of a background task."""
        if self._bg_runner is None:
            raise RuntimeError("No background runner active")
        return self._bg_runner.get_status(task_id)

    def get_task_result(self, task_id: str, wait: bool = False, timeout: float | None = None):
        """Return the result of a background task (optionally blocking)."""
        if self._bg_runner is None:
            raise RuntimeError("No background runner active")
        return self._bg_runner.get_result(task_id, wait=wait, timeout=timeout)

    def cancel_task(self, task_id: str) -> bool:
        if self._bg_runner is None:
            return False
        return self._bg_runner.cancel(task_id)

    def pause_task(self, task_id: str) -> bool:
        if self._bg_runner is None:
            return False
        return self._bg_runner.pause(task_id)

    def resume_task(self, task_id: str) -> bool:
        if self._bg_runner is None:
            return False
        return self._bg_runner.resume(task_id)

    def synthesize(self, synthesis_data: dict[str, Any]) -> str:
        """
        Synthesize results from sub-agents.

        Args:
            synthesis_data: Data to synthesize

        Returns:
            Synthesized output
        """
        # Build synthesis prompt
        results_text = []
        for result in synthesis_data.get("results", []):
            output = result.get("output", {})
            if isinstance(output, dict):
                results_text.append(output.get("output", str(output)))
            else:
                results_text.append(str(output))

        prompt = f"""Synthesize the following results into a comprehensive answer for: {synthesis_data['original_task']}

Results from sub-agents:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(results_text))}

Provide a well-structured, comprehensive response that integrates all findings."""

        # Generate synthesis
        response = self._generate(prompt, temperature=0.6)
        return response.get("text", "").strip()

    async def run_async(self,
                       task: "str | Message | list[Any]",
                       mode: AgentMode = AgentMode.AUTO,
                       context: dict[str, Any] | None = None,
                       inputs: list[Any] | None = None,
                       **kwargs) -> AgentResponse:
        """
        Truly asynchronous version of run().

        Runs the synchronous run() method in a thread executor so it
        doesn't block the event loop, while remaining compatible with
        async callers.

        Args:
            task: Task description (str, Message, or list[ContentPart])
            mode: Execution mode
            context: Optional context
            inputs: Optional multimodal content parts (see ``run``).
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        import functools
        loop = asyncio.get_running_loop()
        func = functools.partial(self.run, task, mode, context, inputs=inputs, **kwargs)
        return await loop.run_in_executor(None, func)

    # ── Resource management ─────────────────────────────────────────────

    def close(self) -> None:
        """
        Release resources held by the agent.

        Closes SQLite connections (long-term memory), resets circuit
        breakers, and clears memory references.  Safe to call multiple
        times.
        """
        if getattr(self, '_closed', False):
            return
        self._closed = True
        self._circuit_breaker.reset_all()
        # Stop background worker threads so they never outlive the agent.
        if getattr(self, '_bg_runner', None) is not None:
            try:
                self._bg_runner.shutdown(wait=False)
            except Exception:
                logger.debug("Failed to shut down background runner", exc_info=True)
        if self.long_term_memory is not None:
            try:
                self.long_term_memory.close()
            except Exception:
                logger.debug("Failed to close long-term memory", exc_info=True)
        logger.debug(f"Agent '{self.name}' closed")

    def __enter__(self):
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit — clean up resources."""
        self.close()
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit — clean up resources."""
        self.close()
        return False

    def __del__(self):
        """Warn if agent was garbage-collected without close()."""
        if getattr(self, '_closed', True):
            return
        # During interpreter shutdown module globals (including ``logger``) may
        # already be torn down to None — guard so __del__ never raises.
        if logger is None:
            return
        try:
            logger.warning(
                f"Agent '{getattr(self, 'name', '?')}' was garbage-collected "
                "without calling close(). Use 'with Agent(config) as agent:' "
                "or call agent.close() explicitly."
            )
        except Exception:
            pass

    def _stream_direct(self, task: str, on_answer: Callable[[str], None] | None = None,
                       **kwargs) -> Iterator[str]:
        """Stream a model answer directly, without the ReAct scaffold.

        Used by ``stream()`` when the agent has no tools. The prompt mirrors
        ``_run_direct_inference`` so streamed and non-streamed answers match.
        Tokens are yielded as they arrive (true incrementality); the assembled
        answer is sanitized before it is stored in memory and handed to
        ``on_answer``. A mid-stream provider error is raised
        (typed + redacted) rather than yielded as a chunk, so a consumer can
        tell success from failure.
        """
        conversation_history = self._format_conversation_history()
        if conversation_history:
            prompt = (
                f"{conversation_history}\n\n"
                f"Based on the conversation above, answer this question directly "
                f"and concisely:\n\n{task}\n\nAnswer:"
            )
        else:
            prompt = f"Answer this question directly and concisely:\n\n{task}\n\nAnswer:"

        # No ReAct stop sequences here: there is no scaffold to trim, and the
        # GPT-5/reasoning families reject `stop`. reasoning_effort is threaded
        # through so callers can request "minimal" for trivial prompts.
        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", 1024),
            top_p=kwargs.get("top_p", 0.9),
            stop_sequences=kwargs.get("stop_sequences"),
            reasoning_effort=kwargs.get("reasoning_effort"),
        )

        accumulated = ""
        stream_iter = self.model.generate_stream(prompt, config=gen_config)
        try:
            for token in stream_iter:
                accumulated += token
                yield token
        except Exception:
            logger.debug("Streaming generation failed", exc_info=True)
            raise
        finally:
            close_stream = getattr(stream_iter, "close", None)
            if close_stream is not None:
                close_stream()

        answer = sanitize_final_answer(accumulated) or accumulated.strip()
        if answer:
            self.short_term_memory.add_user_message(task)
            self.short_term_memory.add_assistant_message(answer)
        if on_answer:
            on_answer(answer)

    def stream(self,
               task: "str | Message | list[Any]",
               mode: AgentMode = AgentMode.AUTO,
               context: dict[str, Any] | None = None,
               on_thought: Callable[[str], None] | None = None,
               on_tool_call: Callable[[str, str], None] | None = None,
               on_observation: Callable[[str], None] | None = None,
               on_answer: Callable[[str], None] | None = None,
               inputs: list[Any] | None = None,
               **kwargs) -> Iterator[str]:
        """
        Stream a response incrementally using real model streaming.

        Streaming contract (stable):

        - Iterating yields successive **text-delta** ``str`` chunks. Joining
          every chunk (``"".join(agent.stream(task))``) reconstructs the answer.
        - On a no-tool agent the chunks are the model's answer tokens directly
          (no ReAct scaffolding). With tools, intermediate ``Observation:``
          lines are interleaved as the loop runs.
        - The iterator simply **ending** is the terminal "done" signal; there is
          no sentinel value to test for.
        - A provider/model failure raises a typed error from the iterator (it is
          not silently swallowed into an empty stream).

        The richer typed step-event stream (per-tool ticks, reasoning markers)
        is a presentation layer built on top of these deltas; the text-delta
        contract here is the stable one to code against.

        Args:
            task: Task description. Accepts a ``str``, a ``Message``, or a
                ``list[ContentPart]`` (text is extracted).
            mode: Execution mode
            context: Optional context
            on_thought: Callback for thought tokens
            on_tool_call: Callback(tool_name, tool_input) when a tool is called
            on_observation: Callback for tool observation text
            on_answer: Callback for final answer tokens
            inputs: Multimodal content parts. Streaming is text-only today; if
                media parts are supplied a clear error points to ``run()``.
            **kwargs: Additional arguments

        Yields:
            str: Successive text-delta chunks (see the streaming contract above).
        """
        # Accept str | Message | list[ContentPart]; streaming is text-only, so
        # surface a clear error if media is supplied rather than dropping it.
        task, _stream_inputs = self._coerce_task_input(task, inputs)
        if _stream_inputs is not None:
            raise TypeError(
                "Agent.stream() is text-only; multimodal inputs are not "
                "supported while streaming. Use agent.run(task, inputs=[...]) "
                "for image/audio/video input."
            )

        if self.model is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no model loaded. "
                "Provide a model in AgentConfig or use a mock for testing."
            )

        context = context or {}

        # Fast path: with no tools there is nothing for the ReAct loop to do, so
        # stream the model's answer directly. The ReAct scaffold otherwise forces
        # the model to emit Thought/Action/Final Answer bookkeeping that wastes
        # latency (acute on reasoning models) and leaks into the streamed output —
        # and small models that write "Action: Final Answer" instead of
        # "Final Answer:" loop to max-iterations and never surface an answer.
        if not self.tools:
            yield from self._stream_direct(task, on_answer=on_answer, **kwargs)
            return

        max_iterations = self.config.max_iterations
        scratchpad = ""
        iterations = 0
        tool_calls = 0

        # Build conversation history
        conversation_history = self._format_conversation_history()

        default_stop_sequences = [
            "\nObservation:",
            "\nQuestion:",
            "\nHuman:",
            "\nUser:",
        ]

        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", 1024),
            top_p=kwargs.get("top_p", 0.9),
            stop_sequences=kwargs.get("stop_sequences", default_stop_sequences),
        )

        while iterations < max_iterations:
            iterations += 1

            # Build prompt
            tools_desc = self._get_tools_description()
            if self.config.system_prompt_template:
                prompt = self.config.system_prompt_template.format(
                    tools_description=tools_desc,
                    task=task,
                    scratchpad=scratchpad,
                    conversation_history=conversation_history,
                )
            else:
                prompt = self._tool_prompt_generator.generate_react_prompt(
                    task=task,
                    scratchpad=scratchpad,
                    conversation_history=conversation_history,
                    system_prompt=self.config.system_prompt,
                    verbose=self._verbose_tools,
                )

            # Stream tokens from model
            accumulated = ""
            stream_iter = self.model.generate_stream(prompt, config=gen_config)
            try:
                for token in stream_iter:
                    accumulated += token

                    # Check for stop sequences
                    hit_stop = False
                    for stop_seq in default_stop_sequences:
                        if stop_seq in accumulated:
                            # Trim at stop sequence — only yield text before it
                            idx = accumulated.index(stop_seq)
                            # Calculate how much of this token to yield
                            pre_stop = accumulated[:idx]
                            already_yielded = accumulated[:len(accumulated) - len(token)]
                            remaining = pre_stop[len(already_yielded):]
                            if remaining:
                                yield remaining
                            accumulated = pre_stop
                            hit_stop = True
                            break

                    # Break early when a complete Final Answer is detected to avoid
                    # runaway generation — transformers streaming ignores stop_sequences.
                    if not hit_stop and "Final Answer:" in accumulated:
                        fa_pos = accumulated.index("Final Answer:")
                        after_fa = accumulated[fa_pos + len("Final Answer:"):]
                        if "\n" in after_fa or len(after_fa.strip()) >= 5:
                            hit_stop = True

                    if hit_stop:
                        break

                    yield token

            except Exception:
                # Fail honestly: raise the typed (already-redacted) provider error
                # at the iterator boundary so a consumer iterating stream() can tell
                # success from failure, instead of receiving the error text as a
                # normal chunk that looks like model output.
                logger.debug("Streaming generation failed", exc_info=True)
                raise
            finally:
                close_stream = getattr(stream_iter, "close", None)
                if close_stream is not None:
                    close_stream()

            # Parse the accumulated response
            parsed = self._parse_react_response(accumulated)
            thought = parsed.get("thought", "")
            scratchpad += f"\nThought: {thought}"

            if on_thought and thought:
                on_thought(thought)

            # Check for final answer
            if parsed.get("final_answer"):
                answer = sanitize_final_answer(parsed["final_answer"]) or parsed["final_answer"]
                if on_answer:
                    on_answer(answer)
                # Store in memory
                if answer:
                    self.short_term_memory.add_user_message(task)
                    self.short_term_memory.add_assistant_message(answer)
                return

            # Execute tool if present
            if parsed.get("action") and parsed.get("action_input"):
                action = parsed["action"]
                action_input = parsed["action_input"]

                if on_tool_call:
                    on_tool_call(action, action_input)

                if action in self.tools:
                    tool_result = self._execute_tool(action, action_input)
                    tool_calls += 1

                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {tool_result}"

                    # Yield observation
                    obs_text = f"\nObservation: {tool_result}\n"
                    yield obs_text
                    if on_observation:
                        on_observation(str(tool_result))
                else:
                    no_tool_msg = f"\nObservation: Tool '{action}' not found. Use 'Final Answer:' to respond directly.\n"
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: Tool '{action}' not found."
                    yield no_tool_msg
            else:
                scratchpad += "\nAction: (continue reasoning)"

        yield "\n[Max iterations reached]"

    def get_execution_summary(self) -> dict[str, Any]:
        """
        Get summary of execution.

        Returns:
            Summary dictionary
        """
        return self.execution_tracker.get_summary()

    def __repr__(self) -> str:
        """String representation."""
        return f"Agent(name={self.name}, tools={len(self.tools)}, sub_agents={self.config.enable_sub_agents})"
