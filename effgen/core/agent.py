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
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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
from ..observability.spans import AgentAttrs, ModelAttrs, ToolAttrs
from ..observability.tracing import (
    set_span_attribute,
    set_span_error,
    start_agent_iteration,
    start_agent_run,
    start_model_call,
    start_tool_call,
)
from ..prompts.agent_system_prompt import AgentSystemPromptBuilder
from ..prompts.tool_prompt_generator import ToolPromptGenerator
from ..tools.base_tool import BaseTool, ToolCategory
from ..tools.fallback import ToolFallbackChain
from ..utils.circuit_breaker import CircuitBreaker
from ..utils.prometheus_metrics import metrics as prom_metrics
from ..utils.structured_logging import (
    LogRunContext,
    generate_run_id,
    get_structured_logger,
)
from .execution_tracker import EventType, ExecutionEvent, ExecutionTracker
from .router import RoutingDecision, RoutingStrategy, SubAgentRouter
from .state import AgentState
from .sub_agent_manager import SubAgentManager
from .tool_calling import (
    ToolCallResult,
    get_strategy,
)

logger = logging.getLogger(__name__)
_slog = get_structured_logger(__name__)
# Canonical structured observability logger — emits redacted JSON lines with OTel context
_obs_log = _get_obs_logger(__name__)


_PROVIDER_BY_CLASS_PREFIX: dict[str, str] = {
    "OpenAI": "openai",
    "Cerebras": "cerebras",
    "Gemini": "google",
    "Anthropic": "anthropic",
    "Groq": "groq",
    "Together": "together",
    "Fireworks": "fireworks",
    "HFInference": "hf_inference",
    "Replicate": "replicate",
    "MLXVLM": "mlx_vlm",
}


def _infer_provider_from_model(model: Any, model_name: str | None = None) -> str:
    """
    Best-effort provider inference from a model instance.

    Order of preference:
    1. ``model.provider`` / ``model.provider_name`` attribute if present.
    2. Class-name prefix mapping (``OpenAIAdapter`` → ``openai``).
    3. Heuristic mapping from the model identifier string.
    4. ``"unknown"``.
    """
    if model is None:
        return "unknown"
    for attr in ("provider", "provider_name"):
        val = getattr(model, attr, None)
        if isinstance(val, str) and val:
            return val
    cls_name = type(model).__name__
    for prefix, provider in _PROVIDER_BY_CLASS_PREFIX.items():
        if cls_name.startswith(prefix):
            return provider
    m = (model_name or "").lower()
    if m.startswith(("gpt-", "o1", "o3", "o4", "text-")):
        return "openai"
    if m.startswith(("gemini", "models/gemini")):
        return "google"
    if m.startswith(("claude", "anthropic")):
        return "anthropic"
    if "llama" in m or "qwen" in m or m.startswith("cerebras"):
        return "cerebras"
    if m.startswith(("mixtral", "mistral")):
        return "groq"
    return "unknown"


def _safe_int_or_none(value: Any) -> int | None:
    """Convert numeric dashboard metadata values without raising."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float_or_none(value: Any) -> float | None:
    """Convert numeric dashboard metadata values without raising."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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
        require_model: Whether model loading is required (raise error on failure)
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
    require_model: bool = False
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
        metadata: Additional metadata
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


class Agent:
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
            try:
                logger.info(f"Loading model: {self.model_name}")
                self.model = self.model_loader.load_model(
                    self.model_name,
                    engine_config=config.model_config
                )
                logger.info(f"Model loaded successfully: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load model '{self.model_name}': {e}")
                self.model = None
                if config.require_model:
                    raise RuntimeError(f"Failed to load required model: {e}")
                else:
                    logger.warning(
                        f"Model loading failed for '{self.model_name}'. "
                        "Agent will crash on first inference call. "
                        "Set require_model=True to fail fast."
                    )
        else:
            # No model provided
            self.model_name = None
            self.model = None

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
                if not is_openai:
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
                if not is_gemini:
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
                if not is_anthropic:
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

    @staticmethod
    def _resolve_guardrails(guardrails: Any):
        """Resolve guardrails config to a GuardrailChain or None."""
        if guardrails is None:
            return None
        # Already a GuardrailChain
        from ..guardrails.base import GuardrailChain
        if isinstance(guardrails, GuardrailChain):
            return guardrails
        # Preset name string
        if isinstance(guardrails, str):
            from ..guardrails.presets import get_guardrail_preset
            return get_guardrail_preset(guardrails)
        return None

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
            task: str,
            mode: AgentMode = AgentMode.AUTO,
            context: dict[str, Any] | None = None,
            output_schema: dict[str, Any] | None = None,
            output_model: Any = None,
            **kwargs) -> AgentResponse:
        """
        Execute a task.

        Args:
            task: Task description
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
        self._current_depth = 0  # Reset depth at the start of each top-level run()
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
            message=f"Starting task: {task[:100]}...",
            data={"task": task, "mode": mode.value}
        ))

        # Wrap entire run in tracing span + structured log context
        with start_agent_run(preset=self.name, task=task, run_id=run_id) as _span, \
             LogRunContext(run_id=run_id, agent_name=self.name):
            _slog.agent_event(self.name, "task_start", task=task[:200], mode=mode.value, run_id=run_id)
            _obs_log.agent_event("run.started", agent=self.name, task=task[:200], mode=mode.value, run_id=run_id)

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

                return response

            except Exception as e:
                # Track failure
                self.execution_tracker.track_event(ExecutionEvent(
                    type=EventType.TASK_FAILED,
                    agent_id=self.name,
                    message=f"Task failed: {str(e)}",
                    data={"error": str(e)}
                ))
                prom_metrics.errors.inc(labels=labels)
                set_span_error(e)
                _slog.agent_event(self.name, "task_failed", level=logging.ERROR, error=str(e))
                _obs_log.agent_event("run.failed", level=logging.ERROR, agent=self.name, run_id=run_id, error=str(e))
                response = AgentResponse(
                    output=f"Error: {str(e)}",
                    success=False,
                    execution_time=time.time() - start_time,
                    execution_trace=self.execution_tracker.get_trace(),
                    metadata={"error": str(e), "run_id": run_id}
                )
                self._record_dashboard_run(response, error=str(e))
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
            pass

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

    def _apply_structured_output(
        self,
        response: AgentResponse,
        schema: dict[str, Any],
        output_model: Any | None,
        task: str,
    ) -> AgentResponse:
        """Post-process response to ensure structured output matches schema.

        If the agent's free-text output already contains valid JSON matching
        the schema, it is extracted and returned. Otherwise, the model is
        re-prompted to produce conforming JSON.

        Args:
            response: The original AgentResponse.
            schema: JSON Schema dict.
            output_model: Optional Pydantic model class for parsing.
            task: Original task (used for re-prompting).

        Returns:
            AgentResponse with validated structured output.
        """
        from .structured_output import (
            StructuredOutputConfig,
            constrain_output,
            extract_json_from_text,
            validate_json_schema,
        )

        # First, try to extract and validate JSON from existing output
        json_str = extract_json_from_text(response.output)
        if json_str:
            try:
                parsed = json.loads(json_str)
                valid, err = validate_json_schema(parsed, schema)
                if valid:
                    response.output = json_str
                    response.metadata["structured_output"] = True
                    if output_model is not None:
                        response.metadata["parsed"] = self._parse_with_pydantic(
                            output_model, parsed,
                        )
                    return response
            except (json.JSONDecodeError, TypeError):
                pass

        # Existing output doesn't match — use constrain_output to re-prompt
        if self.model is not None:
            try:
                logger.info("Re-prompting model for structured output")
                config = StructuredOutputConfig(schema=schema)
                structured_prompt = (
                    f"Based on this task and result, produce structured output.\n"
                    f"Task: {task}\n"
                    f"Result: {response.output}"
                )
                json_str, parsed = constrain_output(
                    self.model, structured_prompt, schema, config,
                )
                response.output = json_str
                response.metadata["structured_output"] = True
                response.metadata["structured_output_reprompted"] = True
                if output_model is not None:
                    response.metadata["parsed"] = self._parse_with_pydantic(
                        output_model, parsed,
                    )
            except ValueError as e:
                logger.warning(f"Structured output constraint failed: {e}")
                response.metadata["structured_output"] = False
                response.metadata["structured_output_error"] = str(e)

        return response

    @staticmethod
    def _parse_with_pydantic(model_class: Any, data: Any) -> Any:
        """Parse data into a Pydantic model instance.

        Supports both Pydantic v1 and v2.
        """
        try:
            if hasattr(model_class, 'model_validate'):
                # Pydantic v2
                return model_class.model_validate(data)
            else:
                # Pydantic v1
                return model_class(**data)
        except Exception as e:
            logger.warning(f"Pydantic parsing failed: {e}")
            return None

    def _run_single_agent(self,
                         task: str,
                         context: dict[str, Any],
                         **kwargs) -> AgentResponse:
        """
        Execute task using single agent with ReAct loop or direct inference.

        Args:
            task: Task description
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # Extract debug flags (set by run())
        debug = kwargs.pop("_debug", False)
        run_id = kwargs.pop("_run_id", "")
        # Pop custom kwargs so they don't leak to the model layer.
        # We re-read them locally below before they would propagate further.
        _ckpt_interval_arg = kwargs.pop("checkpoint_interval", 0) or 0
        _ckpt_dir_arg = kwargs.pop("checkpoint_dir", None)
        _resume_scratchpad_arg = kwargs.pop("_resume_scratchpad", None)

        # Structured multimodal inputs must reach adapters as Message parts.
        # The ReAct prompt is text-only, so use direct inference for these calls
        # even when the preset includes tools.
        if kwargs.get("inputs") is not None:
            return self._run_direct_inference(task, context, **kwargs)

        # If no tools available, use direct inference instead of ReAct
        if not self.tools:
            return self._run_direct_inference(task, context, **kwargs)

        # If any native OpenAI tools are present and the model supports it,
        # route through the Responses API directly (not the ReAct loop).
        if self._has_native_tools():
            return self._run_with_native_tools(task, context, **kwargs)

        # If any Gemini native tools are present, route through the Gemini
        # native-tool path which passes tool objects directly to the adapter.
        if self._has_gemini_native_tools():
            return self._run_with_gemini_native_tools(task, context, **kwargs)

        iterations = 0
        tool_calls = 0
        tokens_used = 0
        scratchpad = ""
        max_iterations = kwargs.get("max_iterations", self.config.max_iterations)

        # Debug trace collector
        debug_trace = None
        if debug:
            from ..debug.inspector import DebugTrace
            debug_trace = DebugTrace(
                task=task, agent_name=self.name, run_id=run_id,
            )

        # Format conversation history
        conversation_history = self._format_conversation_history()

        # ReAct loop
        previous_actions: list[tuple[str, str]] = []  # Track (action, input) pairs for loop detection
        _batch_tool_runs = 0  # Count of batch native-tool runs; cap at 2 to prevent infinite loops
        # Optional periodic checkpointing
        _ckpt_interval = _ckpt_interval_arg
        _ckpt_dir = _ckpt_dir_arg
        _ckpt_mgr = None
        if _ckpt_interval and _ckpt_dir:
            try:
                from .checkpoint import CheckpointManager as _CM
                _ckpt_mgr = _CM(_ckpt_dir)
            except Exception as _e:
                logger.warning("Failed to init CheckpointManager: %s", _e)
        # Allow resuming with a seeded scratchpad
        if _resume_scratchpad_arg:
            scratchpad = _resume_scratchpad_arg
        while iterations < max_iterations:
            iterations += 1
            iter_start = time.time()
            if _ckpt_mgr is not None and iterations > 1 and (iterations - 1) % _ckpt_interval == 0:
                try:
                    from .checkpoint import CheckpointManager as _CM2
                    cp = _CM2.snapshot_agent(
                        self,
                        task=task,
                        iteration=iterations,
                        scratchpad=scratchpad,
                        tool_calls=tool_calls,
                        tokens_used=tokens_used,
                        metadata={"interval": _ckpt_interval},
                    )
                    self._last_checkpoint_id = _ckpt_mgr.save(cp)
                except Exception as _e:
                    logger.warning("Periodic checkpoint failed: %s", _e)

            # Determine if we should use native tool calling prompt format
            use_native_prompt = (
                self._tool_calling_strategy.name in ("native", "hybrid")
                and self.model is not None
                and hasattr(self.model, 'supports_tool_calling')
                and self.model.supports_tool_calling()
            )

            # Build prompt
            gen_kwargs = dict(kwargs)
            # After 2 multi-tool batches, stop passing tools to force synthesis.
            if _batch_tool_runs >= 2:
                use_native_prompt = False
            if use_native_prompt and not self.config.system_prompt_template:
                # Native/hybrid mode: use a simple user message and pass
                # tool definitions via the chat template's tools parameter.
                # The model will produce native tool call tokens (e.g.
                # <tool_call> for Qwen, [TOOL_CALLS] for Mistral).
                if scratchpad:
                    prompt = (
                        f"{task}\n\n"
                        f"Previous steps:\n{scratchpad}\n\n"
                        f"Continue solving the task. If you have the final answer, state it clearly."
                    )
                else:
                    prompt = task
                # Pass tool definitions for the chat template
                tool_defs = self._tool_calling_strategy.format_tools_for_prompt(
                    list(self.tools.values())
                )
                if isinstance(tool_defs, list):
                    gen_kwargs["tools"] = tool_defs
            elif self.config.system_prompt_template:
                # User-provided custom template
                tools_description = self._get_tools_description()
                prompt = self.config.system_prompt_template.format(
                    tools_description=tools_description,
                    conversation_history=conversation_history,
                    task=task,
                    scratchpad=scratchpad
                )
            else:
                # ReAct mode: use enhanced ToolPromptGenerator
                prompt = self._tool_prompt_generator.generate_react_prompt(
                    task=task,
                    scratchpad=scratchpad,
                    conversation_history=conversation_history,
                    system_prompt=self.config.system_prompt,
                    verbose=self._verbose_tools,
                )

            # Debug: log first iteration prompt to see if history is included
            if iterations == 1 and conversation_history:
                logger.info(f"[Memory] Including conversation history ({len(self.short_term_memory.messages)} messages)")

            # Track reasoning step
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.REASONING_STEP,
                agent_id=self.name,
                message=f"Iteration {iterations}: Reasoning...",
                data={"iteration": iterations}
            ))

            # Generate response inside tracing span
            with start_agent_iteration(preset=self.name, iteration=iterations):
                model_name = getattr(self, "model_name", None) or "unknown"
                provider = _infer_provider_from_model(self.model, model_name)
                with start_model_call(provider=provider, model=model_name) as _mspan:
                    response = self._generate(prompt, **gen_kwargs)
                    # Annotate span with token counts from response
                    _meta = response.get("metadata") or {}
                    _in_tok = _meta.get("prompt_tokens", 0) or 0
                    _out_tok = response.get("tokens_used", 0) or 0
                    _cached = _meta.get("cached_input_tokens", 0) or 0
                    try:
                        _mspan.set_attribute(ModelAttrs.INPUT_TOKENS, int(_in_tok))
                        _mspan.set_attribute(ModelAttrs.OUTPUT_TOKENS, int(_out_tok))
                        if _cached:
                            _mspan.set_attribute(ModelAttrs.CACHED_TOKENS, int(_cached))
                        _mspan.set_attribute(ModelAttrs.OUTCOME, "ok" if response.get("finish_reason") != "error" else "error")
                    except Exception:
                        pass
                iter_tokens = response.get("tokens_used", 0)
                tokens_used += iter_tokens

            _slog.iteration_event(iterations, "generate", tokens=iter_tokens)
            _obs_log.event("agent.iteration.generate", iteration=iterations, tokens=iter_tokens, model=getattr(self, "model_name", "unknown"))

            if response.get("finish_reason") == "error":
                metadata = response.get("metadata") or {}
                error_text = str(metadata.get("error") or "generation_error")
                meta_fail: dict[str, Any] = {
                    "reason": "generation_error",
                    "error": error_text,
                }
                if debug_trace is not None:
                    debug_trace.total_tokens = tokens_used
                    debug_trace.success = False
                    meta_fail["debug_trace"] = debug_trace
                return AgentResponse(
                    output=error_text,
                    success=False,
                    mode=AgentMode.SINGLE,
                    iterations=iterations,
                    tool_calls=tool_calls,
                    tokens_used=tokens_used,
                    metadata=meta_fail,
                )

            # Debug: Log the raw response
            logger.info(f"[Iteration {iterations}] Raw model output: {response['text'][:300]}...")
            logger.debug(f"[Iteration {iterations}] Full model output: {response['text']}")

            # Parse response using strategy. If the adapter returned a native
            # tool call (empty text + structured tool_calls in metadata), use
            # it directly — no text parsing needed.
            native_tool_calls = response.get("tool_calls") or []

            # Execute ALL native tool calls in one batch (OpenAI/Cerebras can
            # return multiple tool_calls in a single response).
            if len(native_tool_calls) > 1 and self.tools:
                batch_observations: list[str] = []
                for _tc in native_tool_calls:
                    _fn = _tc.get("function", _tc)
                    _tname = _fn.get("name", "")
                    _targs = _fn.get("arguments", {})
                    if isinstance(_targs, str):
                        try:
                            _targs = json.loads(_targs)
                        except (json.JSONDecodeError, TypeError):
                            _targs = {"__raw_input__": _targs}
                    if _tname in self.tools:
                        with start_tool_call(tool_name=_tname, tool_input=str(_targs)[:500]) as _btspan:
                            _obs = self._execute_tool(_tname, json.dumps(_targs))
                            try:
                                _btspan.set_attribute(ToolAttrs.STATUS, "ok")
                            except Exception:
                                pass
                        tool_calls += 1
                        batch_observations.append(f"[{_tname}({_targs})] → {_obs}")
                        scratchpad += f"\nAction: {_tname}\nAction Input: {json.dumps(_targs)}\nObservation: {_obs}"
                    else:
                        batch_observations.append(f"[{_tname}] → Tool not found")
                # After batch execution, nudge model to synthesize a final answer.
                scratchpad += "\n[Tool results computed above. Continue or provide Final Answer:]"
                _batch_tool_runs += 1
                parsed = {"thought": "", "action": None, "action_input": None, "final_answer": None}
                cur_observation = "\n".join(batch_observations)
                logger.info(f"[Batch native tool calls] {len(native_tool_calls)} calls executed (batch run #{_batch_tool_runs})")
            elif native_tool_calls:
                strategy_result = self._parse_native_tool_calls(native_tool_calls)
                # Convert to legacy dict format for compatibility with rest of loop
                parsed = self._tool_call_result_to_dict(strategy_result)
            else:
                strategy_result = self._tool_calling_strategy.parse_response(
                    response["text"], tools=self.tools,
                )
                # Convert to legacy dict format for compatibility with rest of loop
                parsed = self._tool_call_result_to_dict(strategy_result)

            # Debug: Log what was parsed
            logger.info(f"[Iteration {iterations}] Parsed - Action: {parsed.get('action')}, Input: {parsed.get('action_input')}, Final: {parsed.get('final_answer')}")

            # Add to scratchpad
            scratchpad += f"\nThought: {parsed.get('thought', '')}"

            # Capture debug iteration data
            cur_observation = None  # filled later if tool runs

            def _build_response(
                output: str,
                success: bool = True,
                _tokens_used: int = tokens_used,
                _iterations: int = iterations,
                _tool_calls: int = tool_calls,
                _iter_start: float = iter_start,
                **extra_meta: Any,
            ) -> AgentResponse:
                """Helper to build response and attach debug trace."""
                meta: dict[str, Any] = {"tool_calling_strategy": self._tool_calling_strategy.name}
                meta.update(extra_meta)
                if debug_trace is not None:
                    debug_trace.total_tokens = _tokens_used
                    debug_trace.total_latency = time.time() - (_iter_start - (_iterations - 1) * 0.001)
                    debug_trace.final_answer = output if success else None
                    debug_trace.success = success
                    meta["debug_trace"] = debug_trace
                return AgentResponse(
                    output=output,
                    success=success,
                    mode=AgentMode.SINGLE,
                    iterations=_iterations,
                    tool_calls=_tool_calls,
                    tokens_used=_tokens_used,
                    metadata=meta,
                )

            # Check for final answer
            final_answer = parsed.get("final_answer")
            if final_answer and tool_calls > 0 and final_answer.strip().lower() in {
                "none",
                "null",
                "n/a",
                "na",
            }:
                partial = self._extract_partial_answer(scratchpad)
                if partial:
                    logger.info(
                        "Ignoring null-like final answer after tool execution; "
                        "returning latest observation"
                    )
                    return _build_response(partial, reason="null_final_from_model", partial=True)

            if final_answer:
                # Record final debug iteration
                if debug_trace is not None:
                    from ..debug.inspector import DebugIteration
                    debug_trace.iterations.append(DebugIteration(
                        iteration=iterations,
                        raw_prompt=prompt[:2000],
                        raw_response=response["text"][:2000],
                        thought=parsed.get("thought", ""),
                        final_answer=final_answer,
                        tokens_used=iter_tokens,
                        latency=time.time() - iter_start,
                        scratchpad_snapshot=scratchpad,
                    ))
                return _build_response(final_answer)

            # Check if model is stating an answer without "Final Answer:" keyword
            # This happens when model provides result after tool execution
            if tool_calls > 0 and not parsed.get("action"):
                # No action and we've used tools - model might be stating the answer
                response_text = response["text"].strip()
                # Check for answer-like patterns
                if any(phrase in response_text.lower() for phrase in ["the answer is", "the result is", "the sum is", "equals", "="]):
                    logger.info("Detected answer statement without 'Final Answer:' keyword")
                    if debug_trace is not None:
                        from ..debug.inspector import DebugIteration
                        debug_trace.iterations.append(DebugIteration(
                            iteration=iterations,
                            raw_prompt=prompt[:2000],
                            raw_response=response_text[:2000],
                            thought=parsed.get("thought", ""),
                            final_answer=response_text,
                            tokens_used=iter_tokens,
                            latency=time.time() - iter_start,
                            scratchpad_snapshot=scratchpad,
                        ))
                    return _build_response(response_text)

            # Execute action if present
            if parsed.get("action") and parsed.get("action_input"):
                action = parsed["action"]
                action_input = parsed["action_input"]

                # Loop detection: check if we've seen this exact (action, input) before
                # Also detect fuzzy loops: same tool called 3+ times with different inputs
                # (SLMs like Llama produce slightly different formatting each time)
                # Normalize action_input for comparison
                normalized_input = action_input.strip()
                try:
                    parsed_json = json.loads(normalized_input)
                    normalized_input = json.dumps(parsed_json, sort_keys=True)
                except (json.JSONDecodeError, TypeError):
                    pass
                current_pair = (action, normalized_input)
                action_call_count = sum(1 for a, _ in previous_actions if a == action)
                exact_loop_count = sum(1 for pair in previous_actions if pair == current_pair)
                is_exact_loop = exact_loop_count >= 1 and action in self.tools
                fuzzy_threshold = 5
                if action in self.tools:
                    tool = self.tools[action]
                    if hasattr(tool, 'metadata') and hasattr(tool.metadata, 'category'):
                        if tool.metadata.category == ToolCategory.DATA_PROCESSING:
                            fuzzy_threshold = 7
                is_fuzzy_loop = action_call_count >= fuzzy_threshold and action in self.tools
                if is_exact_loop or is_fuzzy_loop:
                    loop_type = "exact" if is_exact_loop else f"fuzzy ({action_call_count + 1} calls)"
                    logger.info(
                        f"[Loop detected] Repeated action '{action}' ({loop_type}) — "
                        f"breaking loop and returning last observation"
                    )
                    # Extract the last successful observation from scratchpad
                    partial = self._extract_partial_answer(scratchpad)
                    if partial:
                        return _build_response(partial, reason="loop_detected", repeated_action=action)
                    # If no partial answer, add a hint to the scratchpad and continue
                    scratchpad += (
                        f"\nAction: {action}"
                        f"\nAction Input: {action_input}"
                        "\nObservation: You already computed this. "
                        "Please provide your final response using 'Final Answer:' now."
                    )
                    continue

                previous_actions.append(current_pair)

                # Check if tool is available (handle no-tool mode gracefully)
                if not self.tools or action not in self.tools:
                    # No tools available - model is hallucinating tools
                    # Guide it to provide direct answer
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += "\nObservation: No tools available. Please provide your answer directly using 'Final Answer:'."
                else:
                    # Execute tool inside tracing span
                    tool_start = time.time()
                    with start_tool_call(tool_name=action, tool_input=str(action_input)) as _tspan:
                        tool_result = self._execute_tool(action, action_input)
                        try:
                            _tspan.set_attribute(ToolAttrs.STATUS, "ok")
                        except Exception:
                            pass
                    tool_elapsed = time.time() - tool_start
                    tool_calls += 1
                    cur_observation = tool_result

                    # Metrics for tool execution
                    tool_labels = {"tool_name": action, "agent_name": self.name}
                    prom_metrics.tool_calls.inc(labels=tool_labels)
                    prom_metrics.tool_execution_time.observe(tool_elapsed, labels=tool_labels)
                    _slog.tool_event(action, "executed", latency=tool_elapsed)
                    _obs_log.tool_event("executed", tool=action, latency_ms=round(tool_elapsed * 1000, 1))

                    # Add observation to scratchpad
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {tool_result}"

                    # Log the observation for debugging
                    logger.info(f"Tool result added to scratchpad: {tool_result[:100]}...")

                    if self._should_return_direct_calculator_result(task, action, action_input):
                        logger.info(
                            "Returning direct calculator result for simple arithmetic task"
                        )
                        return _build_response(
                            tool_result,
                            _tool_calls=tool_calls,
                            reason="direct_calculator_result",
                        )

                    # Nudge model to answer when iterations are running low
                    if iterations >= self.config.max_iterations - 2:
                        scratchpad += "\n[You have the answer from the tool. Please respond with 'Final Answer:' now.]"

            else:
                # No action specified, prompt to continue
                scratchpad += "\nAction: (continue reasoning)"

            # Record debug iteration
            if debug_trace is not None:
                from ..debug.inspector import DebugIteration
                debug_trace.iterations.append(DebugIteration(
                    iteration=iterations,
                    raw_prompt=prompt[:2000],
                    raw_response=response["text"][:2000],
                    thought=parsed.get("thought", ""),
                    action=parsed.get("action"),
                    action_input=parsed.get("action_input"),
                    observation=cur_observation,
                    tokens_used=iter_tokens,
                    latency=time.time() - iter_start,
                    scratchpad_snapshot=scratchpad,
                ))

        # Max iterations reached — try to extract partial answer from scratchpad
        partial_answer = self._extract_partial_answer(scratchpad)
        if partial_answer:
            logger.info("Max iterations reached, returning partial answer from scratchpad")
            meta: dict[str, Any] = {"reason": "max_iterations_partial", "partial": True}
            if debug_trace is not None:
                debug_trace.total_tokens = tokens_used
                debug_trace.final_answer = partial_answer
                debug_trace.success = True
                meta["debug_trace"] = debug_trace
            return AgentResponse(
                output=partial_answer,
                success=True,
                mode=AgentMode.SINGLE,
                iterations=iterations,
                tool_calls=tool_calls,
                tokens_used=tokens_used,
                metadata=meta,
            )

        meta_fail: dict[str, Any] = {"reason": "max_iterations_reached"}
        if debug_trace is not None:
            debug_trace.total_tokens = tokens_used
            debug_trace.success = False
            meta_fail["debug_trace"] = debug_trace
        return AgentResponse(
            output="Maximum iterations reached without final answer.",
            success=False,
            mode=AgentMode.SINGLE,
            iterations=iterations,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            metadata=meta_fail,
        )

    def _extract_partial_answer(self, scratchpad: str) -> str | None:
        """
        Extract the best partial answer from the scratchpad when max iterations is reached.

        Looks for patterns like "I now know the answer", recent observations with
        answer-like content, or the last substantive thought.

        Args:
            scratchpad: The accumulated scratchpad text.

        Returns:
            A partial answer string, or None if nothing useful found.
        """
        if not scratchpad:
            return None

        # Pattern 1: "I now know" type thoughts
        know_match = re.search(
            r"Thought:\s*I (?:now )?know[^.]*\.\s*(.+?)(?=\nThought:|\nAction:|\Z)",
            scratchpad, re.IGNORECASE | re.DOTALL
        )
        if know_match:
            return know_match.group(1).strip()

        # Pattern 2: Observations with clear result values
        observations = re.findall(r"Observation:\s*(.+?)(?=\nThought:|\nAction:|\Z)", scratchpad, re.DOTALL)
        if observations:
            # If multiple observations, combine non-error ones for multi-tool tasks
            valid_obs = [o.strip() for o in observations if o.strip() and not o.strip().lower().startswith("error")]
            if len(valid_obs) > 1:
                return " | ".join(valid_obs)
            elif valid_obs:
                return valid_obs[-1]

        # Pattern 2b: Look for day names or numeric results in any observation
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for obs in reversed(observations) if observations else []:
            obs_lower = obs.strip().lower()
            for day in day_names:
                if day in obs_lower:
                    return obs.strip()

        # Pattern 3: Last substantive thought
        thoughts = re.findall(r"Thought:\s*(.+?)(?=\nAction:|\nObservation:|\Z)", scratchpad, re.DOTALL)
        if thoughts:
            last_thought = thoughts[-1].strip()
            if len(last_thought) > 20:
                return last_thought

        return None

    def _should_return_direct_calculator_result(
        self,
        task: str,
        action: str,
        action_input: str,
    ) -> bool:
        """Return calculator output directly for simple arithmetic questions."""
        if action != "calculator" or not action_input:
            return False

        task_lower = task.lower()
        if any(
            marker in task_lower
            for marker in (
                "explain",
                "show your work",
                "step by step",
                "steps",
                "why",
                "reason",
            )
        ):
            return False

        try:
            parsed_input = json.loads(action_input)
        except (json.JSONDecodeError, TypeError):
            parsed_input = {"expression": action_input}
        if not isinstance(parsed_input, dict):
            return False

        operation = str(parsed_input.get("operation", "calculate")).lower()
        if operation != "calculate":
            return False

        expression = str(parsed_input.get("expression", "")).strip()
        if not expression:
            return False

        task_numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", task)
        expr_numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", expression)
        if not task_numbers or not all(num in expr_numbers for num in task_numbers):
            return False

        arithmetic_markers = (
            "+",
            "-",
            "*",
            "/",
            "^",
            " plus ",
            " minus ",
            " times ",
            " multiplied ",
            " divide ",
            " divided ",
            " square root ",
            " sqrt",
        )
        return any(marker in f" {task_lower} " for marker in arithmetic_markers)

    def _run_with_sub_agents(self,
                            task: str,
                            routing_decision: RoutingDecision,
                            context: dict[str, Any],
                            **kwargs) -> AgentResponse:
        """
        Execute task using sub-agents based on routing decision.

        Args:
            task: Task description
            routing_decision: Router's decision
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        if self._current_depth >= self.config.max_sub_agent_depth:
            logger.warning(f"Sub-agent depth limit reached ({self.config.max_sub_agent_depth})")
            return self._run_single_agent(task, context, **kwargs)

        self._current_depth += 1

        try:
            # Track decomposition
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_DECOMPOSITION,
                agent_id=self.name,
                message=f"Decomposed into {routing_decision.num_sub_agents} subtasks using {routing_decision.strategy.value}",
                data={
                    "strategy": routing_decision.strategy.value,
                    "num_subtasks": routing_decision.num_sub_agents,
                    "specializations": routing_decision.specializations
                }
            ))

            # Execute based on strategy
            strategy = routing_decision.strategy
            subtasks = routing_decision.decomposition

            if strategy == RoutingStrategy.PARALLEL_SUB_AGENTS:
                # Execute in parallel (use helper to handle existing event loops)
                results = self._run_coroutine_sync(
                    self.sub_agent_manager.execute_parallel(subtasks)
                )
            elif strategy == RoutingStrategy.SEQUENTIAL_SUB_AGENTS:
                # Execute sequentially
                results = self.sub_agent_manager.execute_sequential(subtasks)
            elif strategy == RoutingStrategy.HYBRID:
                # Execute with hybrid approach
                results = self.sub_agent_manager.execute_hybrid(subtasks)
            else:
                # Default to sequential
                results = self.sub_agent_manager.execute_sequential(subtasks)

            # Synthesize results
            synthesis = self.sub_agent_manager.synthesize_results(
                results,
                task,
                strategy
            )

            # Calculate totals
            total_tokens = synthesis["metrics"]["total_tokens_used"]
            total_tool_calls = synthesis["metrics"]["total_tool_calls"]

            return AgentResponse(
                output=synthesis["final_output"],
                success=synthesis["successful"] > 0,
                mode=AgentMode.SUB_AGENTS,
                iterations=len(subtasks),
                tool_calls=total_tool_calls,
                tokens_used=total_tokens,
                routing_decision=routing_decision,
                metadata={
                    "synthesis": synthesis,
                    "failed_subtasks": synthesis["failed"]
                }
            )
        finally:
            self._current_depth -= 1

    def _generate(self, prompt: Any, **kwargs) -> dict[str, Any]:
        """
        Generate response from model with retry logic for empty responses.

        Retries up to 3 times on empty responses with exponential backoff
        and slightly increasing temperature.

        When a model router is configured (multi-model mode), the router
        selects the best model for the query. On failure, the agent
        automatically fails over to the next model in the pool.

        If speculative_execution is enabled, runs on two models in parallel
        and returns the first successful result.

        Args:
            prompt: Input prompt. May be a plain string or a structured
                multimodal Message/list of Messages accepted by API adapters.
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Dictionary with 'text', 'tokens_used', and other metadata

        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None and not self._all_models:
            raise RuntimeError(
                f"Agent '{self.name}' has no model loaded. "
                "Provide a model in AgentConfig or use a mock for testing."
            )

        # Speculative execution: run on 2 models, return first success
        if self._speculative_execution and len(self._all_models) >= 2:
            result = self._generate_speculative(prompt, **kwargs)
            if result is not None:
                return result
            # Fall through to normal path if speculative failed

        # Select model via router if available
        active_model = self.model
        if self._model_router is not None:
            try:
                task_hint = kwargs.pop("_task_hint", self._prompt_to_task_hint(prompt))
                tools_list = list(self.tools.values()) if self.tools else None
                decision = self._model_router.select(task_hint, tools_list)
                active_model = decision.model
                logger.info(
                    "Router selected '%s' (reason: %s)",
                    decision.model_name, decision.reason,
                )
            except Exception as e:
                logger.warning("Router selection failed, using default model: %s", e)
                active_model = self.model

        if active_model is None:
            active_model = self.model

        max_retries = 3
        backoff_delays = [0.5, 1.0, 2.0]
        base_temperature = kwargs.get('temperature', self.config.temperature)

        default_stop_sequences = [
            "\nObservation:",
            "\nQuestion:",
            "\nHuman:",
            "\nUser:",
        ]

        last_error = None
        total_tokens = 0
        # Build ordered list of models to try: selected first, then others
        failover_models = [active_model] + [
            m for m in self._all_models if m is not active_model
        ] if len(self._all_models) > 1 else [active_model]

        for model_idx, current_model in enumerate(failover_models):
            if current_model is None:
                continue

            for attempt in range(max_retries):
                try:
                    # Slightly increase temperature on retries to get different output
                    retry_temperature = min(base_temperature + (attempt * 0.1), 1.0)

                    gen_config = GenerationConfig(
                        temperature=retry_temperature,
                        max_tokens=kwargs.get('max_tokens', 1024),
                        top_p=kwargs.get('top_p', 0.9),
                        stop_sequences=kwargs.get('stop_sequences', default_stop_sequences)
                    )

                    # Pass through extra kwargs (e.g. tools for native calling)
                    extra_gen_kwargs = {}
                    if "tools" in kwargs:
                        extra_gen_kwargs["tools"] = kwargs["tools"]

                    result = current_model.generate(prompt, config=gen_config, **extra_gen_kwargs)

                    response_text = result.text if result and result.text else ""
                    tokens_used = result.tokens_used if result and hasattr(result, 'tokens_used') else 0
                    finish_reason = result.finish_reason if result and hasattr(result, 'finish_reason') else "unknown"
                    total_tokens += tokens_used
                    result_metadata = result.metadata if result and hasattr(result, 'metadata') else {}

                    # Native tool-calls can arrive with empty text (finish_reason="tool_calls").
                    # Return the call to the agent loop instead of treating it as an empty
                    # response that needs retrying.
                    native_tool_calls = (result_metadata or {}).get("tool_calls") or []

                    # If we got non-empty text OR a native tool call, return it
                    if response_text.strip() or native_tool_calls:
                        return {
                            "text": response_text,
                            "tokens_used": total_tokens,
                            "finish_reason": finish_reason,
                            "tool_calls": native_tool_calls,
                            "metadata": result_metadata or {},
                        }

                    # Empty response — retry
                    if attempt < max_retries - 1:
                        logger.info(
                            f"Empty response on attempt {attempt + 1}/{max_retries}, "
                            f"retrying in {backoff_delays[attempt]}s with temperature={retry_temperature:.2f}"
                        )
                        time.sleep(backoff_delays[attempt])
                    else:
                        logger.warning(f"Empty response after {max_retries} attempts")

                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Generation error on attempt {attempt + 1}/{max_retries}: {e}, "
                            f"retrying in {backoff_delays[attempt]}s"
                        )
                        time.sleep(backoff_delays[attempt])
                    else:
                        logger.error(f"Generation failed after {max_retries} attempts: {e}")

            # If we have more models to try, failover
            if model_idx < len(failover_models) - 1:
                next_name = getattr(failover_models[model_idx + 1], 'model_name', '?')
                logger.warning(
                    "Failing over to model '%s' after '%s' exhausted retries",
                    next_name, getattr(current_model, 'model_name', '?'),
                )

        # All models and retries exhausted
        if last_error:
            logger.warning(f"Returning empty response due to generation failure: {last_error}")
        return {
            "text": "",
            "tokens_used": total_tokens,
            "finish_reason": "error",
            "metadata": {"error": str(last_error) if last_error else "empty_response"}
        }

    def _generate_speculative(self, prompt: str, **kwargs) -> dict[str, Any] | None:
        """Run generation on 2 models concurrently, return first success.

        Uses asyncio.gather with return_when=FIRST_COMPLETED semantics via
        asyncio.wait. Returns None if both fail.
        """
        if len(self._all_models) < 2:
            return None

        models_to_run = self._all_models[:2]
        base_temperature = kwargs.get('temperature', self.config.temperature)

        default_stop_sequences = [
            "\nObservation:", "\nQuestion:", "\nHuman:", "\nUser:",
        ]

        gen_config = GenerationConfig(
            temperature=base_temperature,
            max_tokens=kwargs.get('max_tokens', 1024),
            top_p=kwargs.get('top_p', 0.9),
            stop_sequences=kwargs.get('stop_sequences', default_stop_sequences),
        )

        extra_gen_kwargs = {}
        if "tools" in kwargs:
            extra_gen_kwargs["tools"] = kwargs["tools"]

        async def _run_model(model: BaseModel) -> dict[str, Any]:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: model.generate(prompt, config=gen_config, **extra_gen_kwargs)
            )
            text = result.text if result and result.text else ""
            if not text.strip():
                raise RuntimeError("Empty response")
            return {
                "text": text,
                "tokens_used": result.tokens_used if result else 0,
                "finish_reason": result.finish_reason if result else "unknown",
                "metadata": result.metadata if result and hasattr(result, 'metadata') else {},
            }

        async def _speculate() -> dict[str, Any] | None:
            tasks = [asyncio.create_task(_run_model(m)) for m in models_to_run]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Cancel remaining
            for t in pending:
                t.cancel()

            for t in done:
                if not t.cancelled() and t.exception() is None:
                    return t.result()

            # All failed
            return None

        try:
            return self._run_coroutine_sync(_speculate())
        except Exception as e:
            logger.warning("Speculative execution failed: %s", e)
            return None

    def _parse_react_response(self, text: str) -> dict[str, Any]:
        """
        Parse ReAct formatted response with robust error handling.

        Args:
            text: Response text

        Returns:
            Dictionary with parsed components

        Notes:
            This parser handles various formats and edge cases:
            - Case-insensitive matching
            - Multiple thought/action patterns
            - Malformed responses
            - Missing fields
        """
        parsed = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None
        }

        if not text or not isinstance(text, str):
            logger.warning(f"Invalid response text for parsing: {type(text)}")
            return parsed

        try:
            # Check for final answer first (highest priority)
            # NOTE: "Answer:" must be at the start of a line to avoid greedy
            # mid-text matches (e.g. "The answer is 42" should NOT match here).
            final_patterns = [
                r"Final Answer:\s*(.+)",
                r"^Answer:\s*(.+)",
                r"^The answer is:\s*(.+)"
            ]

            for pattern in final_patterns:
                try:
                    final_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    if final_match:
                        answer = final_match.group(1).strip()
                        # Stop at next section marker, observation, or human turn
                        answer = re.split(r'\n(?:Question|Thought|Action|Observation|Human):', answer, maxsplit=1)[0].strip()
                        # Strip trailing unrelated content (e.g. Phi-4 generating
                        # follow-up questions after the answer like "...is 42.What year...")
                        # Match sentence boundary (.!?) followed by a new sentence
                        # that itself contains a question mark — likely a hallucinated follow-up.
                        trailing = re.search(r'([.!?])[\s]*[A-Z][^.!?]*\?', answer)
                        if trailing:
                            answer = answer[:trailing.start() + 1].strip()
                        parsed["final_answer"] = answer
                        logger.debug(f"Extracted final answer: {answer[:100]}...")
                        return parsed
                except Exception as e:
                    logger.warning(f"Error matching final answer pattern '{pattern}': {e}")
                    continue

            # Extract thought
            thought_patterns = [
                r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer|Question):|$)",
                r"Thought:\s*(.+?)(?:\n\n|\n[A-Z]|$)"
            ]

            for pattern in thought_patterns:
                try:
                    thought_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        parsed["thought"] = thought
                        logger.debug(f"Extracted thought: {thought[:100]}...")
                        break
                except Exception as e:
                    logger.warning(f"Error matching thought pattern '{pattern}': {e}")
                    continue

            # Extract action
            action_patterns = [
                r"Action:\s*([^\n]+)",
                r"Tool:\s*([^\n]+)",
                r"Use tool:\s*([^\n]+)"
            ]

            for pattern in action_patterns:
                try:
                    action_match = re.search(pattern, text, re.IGNORECASE)
                    if action_match:
                        action = action_match.group(1).strip()
                        # Clean up common artifacts
                        action = action.replace('"', '').replace("'", "")

                        # Check if action is actually "Final Answer" - treat it as final answer, not tool
                        if action.lower() in ["final answer", "finalanswer", "answer"]:
                            logger.debug(f"Action '{action}' detected as Final Answer indicator")
                            # When model writes "Action: Final Answer", the answer may be:
                            # 1. On the same line: "Action: Final Answer: The answer is 42"
                            # 2. On the Action Input line: "Action Input: The answer is 42"
                            # But NOT if Action Input is JSON (model repeating tool input)

                            # Try same-line first — [: \t]+ excludes newlines
                            same_line = re.search(
                                r"Action:\s*Final\s*Answer[: \t]+([^\n]+)", text, re.IGNORECASE,
                            )
                            if same_line:
                                answer_text = same_line.group(1).strip()
                                if answer_text:
                                    parsed["final_answer"] = answer_text
                                    logger.debug("Extracted final answer from Action line")
                                    return parsed

                            # Try Action Input line (only if it's natural language, not JSON)
                            ai_match = re.search(
                                r"Action\s*Input:\s*(.+?)(?:\n|$)",
                                text, re.IGNORECASE,
                            )
                            if ai_match:
                                answer_text = ai_match.group(1).strip()
                                if answer_text and not answer_text.startswith(("{", "[")):
                                    parsed["final_answer"] = answer_text
                                    logger.debug("Extracted final answer from Action Input line")
                                    return parsed

                            # If we get here, the model wrote "Action: Final Answer"
                            # but didn't provide a proper answer text. Don't extract
                            # anything — let the loop continue for another iteration.
                            logger.debug("Action: Final Answer detected but no answer text found")
                            break

                        # Handle function-call format: tool_name(args) or tool_name("args")
                        # Extract just the tool name and put args into action_input
                        func_call_match = re.match(r'^(\w+)\s*\((.+)\)$', action, re.DOTALL)
                        if func_call_match:
                            tool_name = func_call_match.group(1).strip()
                            embedded_args = func_call_match.group(2).strip()
                            # Remove surrounding quotes if present
                            embedded_args = embedded_args.strip('"\'')
                            parsed["action"] = tool_name
                            # Only set action_input if not already set
                            if "action_input" not in parsed or not parsed["action_input"]:
                                parsed["action_input"] = embedded_args
                            logger.debug(f"Extracted function-call style: action={tool_name}, input={embedded_args[:100]}...")
                        else:
                            parsed["action"] = action
                            logger.debug(f"Extracted action: {action}")
                        break
                except Exception as e:
                    logger.warning(f"Error matching action pattern '{pattern}': {e}")
                    continue

            # Extract action input (only if not already set from function-call style)
            # Skip if we already have embedded args from tool_name(args) format
            if "action_input" not in parsed or not parsed.get("action_input"):
                input_patterns = [
                    r"Action Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question|Final Answer):|$)",
                    r"Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
                    r"Parameters?:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)"
                ]

                for pattern in input_patterns:
                    try:
                        input_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                        if input_match:
                            action_input = input_match.group(1).strip()
                            # Remove trailing observation text if present
                            action_input = re.split(r'\nObservation:', action_input, maxsplit=1)[0].strip()
                            parsed["action_input"] = action_input
                            logger.debug(f"Extracted action input: {action_input[:100]}...")
                            break
                    except Exception as e:
                        logger.warning(f"Error matching action input pattern '{pattern}': {e}")
                        continue

        except Exception as e:
            logger.error(f"Critical error in parse_react_response: {e}", exc_info=True)
            # Return partial parse results even if there was an error

        return parsed

    @staticmethod
    def _parse_native_tool_calls(native_tool_calls: list[dict[str, Any]]) -> ToolCallResult:
        """Convert provider-native tool_calls (OpenAI/Cerebras format) into ToolCallResult.

        Providers return:
            [{"id": "...", "type": "function",
              "function": {"name": "...", "arguments": {...} | "json-string"}}]
        """
        result = ToolCallResult(raw_text="")
        if not native_tool_calls:
            return result
        tc = native_tool_calls[0]
        fn = tc.get("function", tc)
        tool_name = fn.get("name")
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {"__raw_input__": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        if tool_name:
            result.tool_name = tool_name
            result.arguments = arguments
            result.is_tool_call = True
        return result

    @staticmethod
    def _tool_call_result_to_dict(result: ToolCallResult) -> dict[str, Any]:
        """Convert a ToolCallResult to the legacy dict format used by the ReAct loop.

        This bridges the new strategy-based parsing with the existing loop
        logic that expects ``{'thought', 'action', 'action_input', 'final_answer'}``.
        """
        parsed: dict[str, Any] = {
            "thought": result.thought,
            "action": None,
            "action_input": None,
            "final_answer": result.final_answer,
        }
        if result.is_tool_call and result.tool_name:
            parsed["action"] = result.tool_name
            # Convert arguments dict back to the string form the loop expects
            if result.arguments:
                raw = result.arguments.get("__raw_input__")
                if raw:
                    parsed["action_input"] = raw
                else:
                    parsed["action_input"] = json.dumps(result.arguments)
            else:
                parsed["action_input"] = "{}"
        return parsed

    @staticmethod
    def _run_coroutine_sync(coro, timeout: float = 120.0):
        """
        Run an async coroutine from synchronous code.

        Uses a simple strategy: try asyncio.run() first. If an event loop
        is already running, fall back to a thread-based approach.

        Args:
            coro: The coroutine to run.
            timeout: Maximum seconds to wait (default 120s).
        """
        try:
            # No running loop — simplest path
            return asyncio.run(coro)
        except RuntimeError:
            # Event loop already running (Jupyter, FastAPI, etc.) —
            # run in a dedicated thread with its own event loop.
            import concurrent.futures
            logger.info(
                "Event loop already running — falling back to "
                "ThreadPoolExecutor for coroutine execution"
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=timeout)

    @staticmethod
    def _clean_json_input(raw: str) -> str:
        """
        Clean malformed JSON commonly produced by SLMs before parsing.

        Handles:
        - Markdown-wrapped JSON (```json ... ```)
        - Trailing commas  ({"key": "val",})
        - Unquoted keys    ({expression: "2+2"})

        Returns the cleaned string (still needs json.loads).
        """
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            # Remove opening fence (with optional language tag) and closing fence
            text = re.sub(r'^```(?:json|JSON)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)
            text = text.strip()

        # Remove trailing commas before } or ]
        # e.g. {"key": "val",} -> {"key": "val"}
        text = re.sub(r',\s*([}\]])', r'\1', text)

        # Quote unquoted keys:  {expression: "2+2"} -> {"expression": "2+2"}
        # Match word characters at the start of a key position (after { or ,)
        # but only if they aren't already quoted
        text = re.sub(
            r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:',
            r' "\1":',
            text
        )

        return text

    @staticmethod
    def _sanitize_tool_input(tool_input: str, max_length: int = 10000) -> str:
        """
        Sanitize tool input by stripping control characters and limiting length.

        Args:
            tool_input: Raw input string.
            max_length: Maximum allowed input length.

        Returns:
            Sanitized input string.
        """
        if not tool_input:
            return tool_input
        # Strip control characters except newline and tab
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', tool_input)
        # Limit length
        if len(sanitized) > max_length:
            logger.warning(f"Tool input truncated from {len(sanitized)} to {max_length} chars")
            sanitized = sanitized[:max_length]
        return sanitized

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a tool with circuit breaker, fallback support, and input sanitization.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input for the tool (JSON string or plain text)

        Returns:
            Tool output as string
        """
        # Sanitize input
        tool_input = self._sanitize_tool_input(tool_input)

        # Pre-tool guardrail check (TOOL_INPUT)
        if self._guardrail_chain is not None:
            from ..guardrails.base import GuardrailPosition as _GP
            tool_obj = self.tools.get(tool_name)
            gr = self._guardrail_chain.check(
                tool_input, position=_GP.TOOL_INPUT,
                tool_name=tool_name, tool=tool_obj,
            )
            if not gr.passed:
                logger.info(f"Guardrail blocked tool '{tool_name}': {gr.reason}")
                return f"Error executing tool '{tool_name}': blocked by guardrail — {gr.reason}"
            if gr.modified_content is not None:
                tool_input = gr.modified_content

        # Human-in-the-loop approval check
        tool_obj = self.tools.get(tool_name)
        _requires_approval = getattr(
            getattr(tool_obj, '_metadata', None), 'requires_approval', False
        ) if tool_obj else False
        if self._approval_manager.should_request_approval(tool_name, _requires_approval):
            from .human_loop import ApprovalDecision
            decision = self._approval_manager.request_approval(tool_name, tool_input)
            if decision != ApprovalDecision.APPROVED:
                logger.info("Tool '%s' denied by human approval (%s)", tool_name, decision.value)
                return f"Error executing tool '{tool_name}': execution denied by human approval ({decision.value})"

        # Circuit breaker check
        if not self._circuit_breaker.is_available(tool_name):
            logger.info(f"Circuit breaker OPEN for '{tool_name}', skipping execution")
            return f"Error executing tool '{tool_name}': tool temporarily disabled due to repeated failures"

        result_str = self._execute_tool_once(tool_name, tool_input)

        # Update circuit breaker
        if result_str.startswith("Error executing tool"):
            self._circuit_breaker.record_failure(tool_name)
        else:
            self._circuit_breaker.record_success(tool_name)

        # Check if primary tool failed and fallback is enabled
        if (
            self._enable_fallback
            and result_str.startswith("Error executing tool")
            and self._fallback_chain.has_fallbacks(tool_name)
        ):
            fallbacks = self._fallback_chain.get_fallbacks(tool_name)
            for fb_name in fallbacks:
                if fb_name not in self.tools:
                    continue
                logger.info(f"Tool '{tool_name}' failed, trying fallback: {fb_name}")
                fb_result = self._execute_tool_once(fb_name, tool_input)
                if not fb_result.startswith("Error executing tool"):
                    logger.info(f"Fallback '{fb_name}' succeeded for '{tool_name}'")
                    return f"[Fallback: used {fb_name} instead of {tool_name}] {fb_result}"
            logger.info(f"All fallbacks exhausted for '{tool_name}'")

        # Post-tool guardrail check (TOOL_OUTPUT)
        if self._guardrail_chain is not None and not result_str.startswith("Error executing tool"):
            from ..guardrails.base import GuardrailPosition as _GP
            gr = self._guardrail_chain.check(
                result_str, position=_GP.TOOL_OUTPUT,
                tool_name=tool_name,
            )
            if not gr.passed:
                logger.info(f"Guardrail blocked output from '{tool_name}': {gr.reason}")
                return f"Error executing tool '{tool_name}': output blocked by guardrail — {gr.reason}"
            if gr.modified_content is not None:
                result_str = gr.modified_content

        return result_str

    def _execute_tool_once(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a single tool with robust error handling (no fallback).

        Args:
            tool_name: Name of tool to execute
            tool_input: Input for the tool (JSON string or plain text)

        Returns:
            Tool output as string
        """
        # Track tool call start
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.TOOL_CALL_START,
            agent_id=self.name,
            message=f"Calling tool: {tool_name}",
            data={"tool_name": tool_name, "tool_input": tool_input}
        ))

        try:
            # Validate tool exists
            if not tool_name:
                raise ValueError("Tool name cannot be empty")

            if tool_name not in self.tools:
                available_tools = ", ".join(self.tools.keys())
                raise ValueError(
                    f"Tool '{tool_name}' not available. "
                    f"Available tools: {available_tools}"
                )

            tool = self.tools[tool_name]

            # OpenAI native tools cannot be executed locally — they run server-side
            # via the Responses API.  If we end up here it means the ReAct loop
            # tried to call a native tool directly (e.g. "Action: openai_web_search").
            # Return a helpful note so the loop can recover gracefully.
            try:
                from ..tools.builtin.openai_native import OpenAINativeTool
                if isinstance(tool, OpenAINativeTool):
                    return (
                        f"Tool '{tool_name}' is an OpenAI server-side tool and cannot be "
                        "executed locally in the ReAct loop.  Use tool_calling_mode='native' "
                        "so the adapter routes it through the OpenAI Responses API."
                    )
            except ImportError:
                pass

            # Parse input intelligently
            input_dict = {}
            if tool_input:
                try:
                    # Try parsing as JSON first (after cleaning SLM artifacts)
                    cleaned = self._clean_json_input(tool_input)
                    input_dict = json.loads(cleaned)
                    if not isinstance(input_dict, dict):
                        # JSON parsed but not a dict - need to intelligently map to tool parameters
                        input_dict = self._map_input_to_parameters(tool, input_dict)
                except json.JSONDecodeError:
                    # Not valid JSON — SLMs often produce Python-style dicts with
                    # single-quoted strings (e.g. {"data": '{"key": "val"}'}).
                    # Try ast.literal_eval as a fallback before plain-text mapping.
                    try:
                        import ast
                        parsed = ast.literal_eval(tool_input)
                        if isinstance(parsed, dict):
                            input_dict = parsed
                        else:
                            input_dict = self._map_input_to_parameters(tool, tool_input)
                    except (ValueError, SyntaxError):
                        # Not valid Python either — use plain text mapping
                        input_dict = self._map_input_to_parameters(tool, tool_input)
                except Exception as e:
                    logger.warning(f"Error parsing tool input, using as plain text: {e}")
                    input_dict = self._map_input_to_parameters(tool, tool_input)

            # Strip markdown code fences from 'code' param even after JSON parse
            if isinstance(input_dict, dict) and 'code' in input_dict and isinstance(input_dict['code'], str):
                code_val = input_dict['code']
                if '```' in code_val:
                    import re as _re
                    code_val = _re.sub(r'^```(?:python|py|javascript|js|bash|sh)?\n?', '', code_val, flags=_re.MULTILINE)
                    code_val = _re.sub(r'\n?```$', '', code_val, flags=_re.MULTILINE)
                    input_dict['code'] = code_val.strip()

            logger.debug(f"Executing tool '{tool_name}' with input: {input_dict}")

            # Execute tool (handle both sync and async)
            try:
                result = tool.execute(**input_dict)

                # Handle async results (coroutines) cleanly
                if asyncio.iscoroutine(result):
                    result = self._run_coroutine_sync(result)

            except TypeError as e:
                logger.error(f"Tool parameter error: {e}")
                raise ValueError(
                    f"Tool '{tool_name}' parameter error: {str(e)}. "
                    f"Input provided: {input_dict}"
                )

            # Convert result to string safely
            if result is None:
                result_str = "No result returned"
            elif hasattr(result, 'output'):
                # ToolResult object - extract output
                if hasattr(result, 'success') and not result.success:
                    error_msg = getattr(result, 'error', 'Unknown error')
                    result_str = f"Tool execution failed: {error_msg}"
                else:
                    output = result.output
                    if isinstance(output, dict):
                        # Try common result keys: result, output, data, message
                        # BUG-012 fix: PythonREPL returns {result: None, stdout: "..."}
                        # when code uses print(). Prefer stdout over a None result.
                        if 'result' in output and output['result'] is not None:
                            result_str = str(output['result'])
                        elif 'stdout' in output and output['stdout']:
                            # PythonREPL/CodeExecutor: stdout has the printed output
                            parts = []
                            parts.append(output['stdout'].rstrip())
                            if output.get('stderr'):
                                parts.append(f"stderr: {output['stderr'].rstrip()}")
                            if output.get('error'):
                                parts.append(f"Error: {output['error']}")
                            result_str = '\n'.join(parts)
                        elif 'stderr' in output and output['stderr']:
                            # CodeExecutor error: stdout empty but stderr has traceback
                            result_str = f"Error: {output['stderr'].rstrip()}"
                            if output.get('exit_code'):
                                result_str += f"\n(exit code: {output['exit_code']})"
                        elif 'error' in output and output['error']:
                            result_str = f"Error: {output['error']}"
                        elif 'exit_code' in output:
                            # CodeExecutor: ran successfully but no output
                            result_str = f"Code executed successfully (exit code {output['exit_code']})"
                        elif 'output' in output:
                            result_str = str(output['output'])
                        elif 'data' in output and 'success' in output:
                            # FileOperations-style: {success, data, message}
                            if output.get('success'):
                                result_str = str(output['data']) if output['data'] is not None else output.get('message', str(output))
                            else:
                                result_str = f"Operation failed: {output.get('message', str(output))}"
                        elif 'data' in output:
                            result_str = str(output['data'])
                        elif 'message' in output:
                            result_str = str(output['message'])
                        else:
                            result_str = str(output)
                    else:
                        result_str = str(output)
            elif hasattr(result, 'result'):
                result_str = str(result.result)
            else:
                result_str = str(result)

            # Check if result indicates a tool-level failure
            if result_str.startswith("Tool execution failed:"):
                raise ValueError(result_str)

            # Track success
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_COMPLETE,
                agent_id=self.name,
                message=f"Tool {tool_name} completed",
                data={"tool_name": tool_name, "result": result_str[:200]}
            ))

            logger.info(f"Tool '{tool_name}' executed successfully")
            return result_str

        except ValueError as e:
            error_msg = str(e)
            logger.debug(f"Tool '{tool_name}' execution failed: {error_msg}")

            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_FAILED,
                agent_id=self.name,
                message=f"Tool {tool_name} failed: {error_msg}",
                data={"tool_name": tool_name, "error": error_msg, "input": tool_input}
            ))

            return f"Error executing tool '{tool_name}': {error_msg}"

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Tool '{tool_name}' execution failed: {error_msg}", exc_info=True)

            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_FAILED,
                agent_id=self.name,
                message=f"Tool {tool_name} failed: {error_msg}",
                data={"tool_name": tool_name, "error": error_msg, "input": tool_input}
            ))

            return f"Error executing tool '{tool_name}': {error_msg}"

    def _map_input_to_parameters(self, tool, input_value):
        """
        Intelligently map input value to tool parameters.

        This handles cases where the model provides plain text or non-dict JSON
        and we need to map it to the tool's expected parameter names.

        Args:
            tool: The tool object with metadata
            input_value: The input value (string or other type)

        Returns:
            Dict mapping parameter names to values
        """
        # Get tool parameters
        if not hasattr(tool, 'metadata') or not hasattr(tool.metadata, 'parameters'):
            # No metadata - use generic "input"
            return {"input": input_value}

        params = tool.metadata.parameters
        required_params = [p for p in params if p.required]

        # Case 1: Single parameter tool
        if len(params) == 1:
            param_name = params[0].name
            return {param_name: input_value}

        # Case 2: Multiple parameters - need to be smarter
        # Common patterns for specific tools

        # Code executor pattern: expects "code" and "language"
        if any(p.name == "code" for p in params):
            # Clean markdown code fences from code input
            code_str = str(input_value)
            # Remove markdown code fences (```python, ```py, ```javascript, etc.)
            import re
            code_str = re.sub(r'^```(?:python|py|javascript|js|bash|sh)?\n?', '', code_str, flags=re.MULTILINE)
            code_str = re.sub(r'\n?```$', '', code_str, flags=re.MULTILINE)
            code_str = code_str.strip()

            result = {"code": code_str}
            # Add default language if required
            if any(p.name == "language" and p.required for p in params):
                result["language"] = "python"  # Default to Python
            return result

        # Calculator pattern: expects "expression"
        if any(p.name == "expression" for p in params):
            return {"expression": str(input_value)}

        # Python REPL pattern: expects "code"
        if any(p.name == "code" for p in required_params):
            return {"code": str(input_value)}

        # File ops pattern: expects "operation" and "path"
        if any(p.name == "operation" for p in required_params):
            import re
            input_str = str(input_value).lower()

            # Try to extract operation and path from the input
            result = {}

            # Detect operation from keywords
            operation = None
            if any(word in input_str for word in ["read", "reading", "show", "display", "cat", "get content"]):
                operation = "read"
            elif any(word in input_str for word in ["write", "writing", "create", "save"]):
                operation = "write"
            elif any(word in input_str for word in ["list", "listing", "ls", "dir", "show files"]):
                operation = "list"
            elif any(word in input_str for word in ["search", "find", "grep"]):
                operation = "search"
            elif any(word in input_str for word in ["metadata", "info", "stat"]):
                operation = "metadata"
            elif any(word in input_str for word in ["convert", "transform"]):
                operation = "convert"

            if operation:
                result["operation"] = operation
            else:
                # Default to read if unclear
                result["operation"] = "read"

            # Extract path - look for file paths or filenames
            path_str = str(input_value)

            # Remove file:/// prefix if present
            path_str = re.sub(r'file://+', '', path_str)

            # Try to find filenames with extensions (prefer last occurrence to avoid "path/to/file")
            # Look for patterns like: file.txt, /path/file.txt, ./file.txt
            path_matches = re.findall(r'[\w\-\.\/]+\.\w+', path_str)
            if path_matches:
                # Use the last match (most likely the actual filename)
                path_candidate = path_matches[-1]
                # If it contains slashes, prefer just the filename part unless it starts with / or ./
                if '/' in path_candidate and not path_candidate.startswith(('/', './')):
                    # Extract just the filename
                    result["path"] = path_candidate.split('/')[-1]
                else:
                    result["path"] = path_candidate
            else:
                # Look for any path-like string
                path_match = re.search(r'(?:file|path)[\s:=]+([^\s]+)', path_str, re.IGNORECASE)
                if path_match:
                    result["path"] = path_match.group(1)
                else:
                    # Just use the input as path (remove operation keywords)
                    cleaned_path = path_str
                    for op_word in ["read", "write", "list", "search", "metadata", "convert", "operation="]:
                        cleaned_path = cleaned_path.replace(op_word, "").replace(op_word.upper(), "")
                    result["path"] = cleaned_path.strip()

            # Clean up path - remove trailing whitespace but preserve absolute paths
            if result.get("path"):
                result["path"] = result["path"].strip()
                # Remove file:// prefix if still present after earlier cleanup
                if result["path"].startswith("file://"):
                    result["path"] = result["path"][7:]
                    # Ensure we keep at least one leading /
                    if not result["path"].startswith("/"):
                        result["path"] = "/" + result["path"]

            return result

        # Search pattern: expects "query"
        if any(p.name == "query" for p in params):
            return {"query": str(input_value)}

        # Default: Use first required parameter name, or first parameter name
        if required_params:
            return {required_params[0].name: str(input_value)}
        elif params:
            return {params[0].name: str(input_value)}
        else:
            # Fallback
            return {"input": str(input_value)}

    def _has_native_tools(self) -> bool:
        """Return True if any OpenAI native tool is in the tools list and the model supports Responses API."""
        try:
            from ..models.openai_adapter import OpenAIAdapter
            from ..tools.builtin.openai_native import OpenAINativeTool
        except ImportError:
            return False
        has_native = any(isinstance(t, OpenAINativeTool) for t in self.tools.values())
        is_openai = isinstance(self.model, OpenAIAdapter)
        return has_native and is_openai

    def _run_with_native_tools(self, task: str, context: dict[str, Any], **kwargs) -> AgentResponse:
        """Execute a task using OpenAI native tools via the Responses API.

        Separates native tools (web_search, code_interpreter, file_search) from
        local effGen tools.  Native tools are passed directly to the Responses API
        spec; local tools are serialised as function-call specs alongside them.
        """
        from ..models.openai_adapter import OpenAIAdapter
        from ..tools.builtin.openai_native import OpenAINativeTool

        native_specs: list[dict] = []
        function_specs: list[dict] = []

        for tool in self.tools.values():
            if isinstance(tool, OpenAINativeTool):
                native_specs.append(tool.to_openai_tool_spec())
            else:
                # Regular effGen tool — include as function call spec
                schema = tool.metadata.to_json_schema()
                function_specs.append({
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema["description"],
                        "parameters": schema["parameters"],
                    },
                })

        adapter: OpenAIAdapter = self.model  # already validated in _has_native_tools

        system_prompt = self.config.system_prompt if self.config.stable_system_prompt else None

        try:
            result = adapter.generate_with_native_tools(
                prompt=task,
                native_tool_specs=native_specs,
                function_tool_specs=function_specs if function_specs else None,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error("Native tool generation failed: %s", e)
            return AgentResponse(
                output=f"Error: {e}",
                success=False,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=0,
                tokens_used=0,
                metadata={"error": str(e)},
            )

        # Handle any function tool calls that came back (local effGen tools)
        native_results = result.metadata.get("native_tool_results", [])
        local_calls = [r for r in native_results if r.get("type") == "function_call"]

        tool_calls_made = len(native_results)

        # Execute local tool calls and stitch their results into a follow-up
        if local_calls and function_specs:
            observations: list[str] = []
            for call in local_calls:
                fn_name = call.get("name", "")
                fn_args_raw = call.get("arguments", "{}")
                try:
                    fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                if fn_name in self.tools:
                    obs = self._execute_tool(fn_name, json.dumps(fn_args))
                    tool_calls_made += 1
                    observations.append(f"[{fn_name}({fn_args})] → {obs}")

            if observations and not result.text:
                # Re-prompt with the observations to get a final answer
                obs_text = "\n".join(observations)
                followup = f"Tool results:\n{obs_text}\n\nBased on these results, answer the user's question: {task}"
                try:
                    followup_result = adapter.generate(followup)
                    return AgentResponse(
                        output=followup_result.text,
                        success=True,
                        mode=AgentMode.SINGLE,
                        iterations=2,
                        tool_calls=tool_calls_made,
                        tokens_used=result.tokens_used + followup_result.tokens_used,
                        metadata=result.metadata,
                    )
                except Exception:
                    pass

        return AgentResponse(
            output=result.text or "(no output from native tools call)",
            success=bool(result.text),
            mode=AgentMode.SINGLE,
            iterations=1,
            tool_calls=tool_calls_made,
            tokens_used=result.tokens_used,
            metadata=result.metadata,
        )

    def _has_gemini_native_tools(self) -> bool:
        """Return True if any Gemini native tool is in the tools list and model is Gemini."""
        try:
            from ..models.gemini_adapter import GeminiAdapter
            from ..tools.builtin.gemini_native import GeminiNativeTool
        except ImportError:
            return False
        has_native = any(isinstance(t, GeminiNativeTool) for t in self.tools.values())
        is_gemini = isinstance(self.model, GeminiAdapter)
        return has_native and is_gemini

    def _run_with_gemini_native_tools(self, task: str, context: dict[str, Any], **kwargs) -> AgentResponse:
        """Execute a task using Gemini server-side native tools.

        Passes ``GeminiNativeTool`` objects (google_search, url_context, code_execution)
        directly to the Gemini adapter alongside any local effGen tools serialised
        as function declarations.  The adapter routes them correctly to the SDK.
        """
        from ..tools.builtin.gemini_native import GeminiNativeTool

        # Separate native vs. local tools
        native_tools: list = []
        function_tool_specs: list[dict] = []
        for tool in self.tools.values():
            if isinstance(tool, GeminiNativeTool):
                native_tools.append(tool)
            else:
                schema = tool.metadata.to_json_schema()
                function_tool_specs.append({
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema["description"],
                        "parameters": schema["parameters"],
                    },
                })

        # Combine: native tool objects first, then regular function specs
        all_tools = native_tools + function_tool_specs

        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", 2048),
        )

        system_note = ""
        if self.config.system_prompt:
            system_note = f"{self.config.system_prompt}\n\n"

        prompt = f"{system_note}{task}"

        try:
            result = self.model.generate(prompt, config=gen_config, tools=all_tools)
        except Exception as exc:
            logger.error("Gemini native tool generation failed: %s", exc)
            return AgentResponse(
                output=f"Error: {exc}",
                success=False,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=0,
                tokens_used=0,
                metadata={"error": str(exc)},
            )

        tool_calls_made = len(result.metadata.get("tool_calls") or [])

        # Execute any local effGen function calls that came back
        local_tc = result.metadata.get("tool_calls") or []
        observations: list[str] = []
        for tc in local_tc:
            fn_name = tc.get("name", "")
            fn_args = tc.get("arguments", {})
            if fn_name in self.tools and not isinstance(self.tools[fn_name], GeminiNativeTool):
                obs = self._execute_tool(fn_name, json.dumps(fn_args) if isinstance(fn_args, dict) else fn_args)
                observations.append(f"[{fn_name}({fn_args})] → {obs}")

        if observations and not result.text.strip():
            obs_text = "\n".join(observations)
            followup = f"Tool results:\n{obs_text}\n\nBased on these results, answer: {task}"
            try:
                followup_result = self.model.generate(followup, config=gen_config)
                return AgentResponse(
                    output=followup_result.text,
                    success=True,
                    mode=AgentMode.SINGLE,
                    iterations=2,
                    tool_calls=tool_calls_made,
                    tokens_used=result.tokens_used + followup_result.tokens_used,
                    metadata=result.metadata,
                )
            except Exception:
                pass

        return AgentResponse(
            output=result.text or "(no output from Gemini native tools call)",
            success=bool(result.text),
            mode=AgentMode.SINGLE,
            iterations=1,
            tool_calls=tool_calls_made,
            tokens_used=result.tokens_used,
            metadata=result.metadata,
        )

    def _run_direct_inference(self,
                               task: str,
                               context: dict[str, Any],
                               **kwargs) -> AgentResponse:
        """
        Run direct inference without ReAct loop (for when no tools are available).

        Args:
            task: Task description
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        inputs = kwargs.pop("inputs", None)
        if inputs is not None:
            prompt = self._build_multimodal_prompt(task, inputs)
        else:
            # Include conversation history from short-term memory for multi-turn context
            conversation_history = self._format_conversation_history()
            if conversation_history:
                prompt = (
                    f"{conversation_history}\n\n"
                    f"Based on the conversation above, answer this question directly and concisely:\n\n"
                    f"{task}\n\nAnswer:"
                )
            else:
                prompt = f"Answer this question directly and concisely:\n\n{task}\n\nAnswer:"

        try:
            response = self._generate(prompt, _task_hint=task, **kwargs)
            answer = response["text"].strip()
            tokens_used = response.get("tokens_used", 0)

            return AgentResponse(
                output=answer,
                success=True,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=0,
                tokens_used=tokens_used,
                metadata={"multimodal_inputs": inputs is not None},
            )

        except Exception as e:
            logger.error(f"Direct inference failed: {e}")
            return AgentResponse(
                output=f"Failed to answer: {str(e)}",
                success=False,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=0,
                tokens_used=0,
                metadata={"error": str(e)}
            )

    def _build_multimodal_prompt(self, task: str, inputs: Any) -> list[Any]:
        """Build structured Messages for adapter-native multimodal input."""
        from effgen.core.messages import ContentPart, Message, Role, TextPart

        if isinstance(inputs, tuple):
            inputs = list(inputs)
        elif not isinstance(inputs, list):
            inputs = [inputs]

        content: list[ContentPart] = [TextPart(text=task)]
        for index, part in enumerate(inputs):
            if not hasattr(part, "type"):
                raise TypeError(
                    "Agent.run(inputs=[...]) expects effGen multimodal parts "
                    f"(image_from/audio_from/video_from); item {index} is {type(part).__name__}"
                )
            content.append(part)

        messages: list[Message] = []
        if self.config.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=[TextPart(text=self.config.system_prompt)]))
        messages.append(Message(role=Role.USER, content=content))
        return messages

    @staticmethod
    def _prompt_to_task_hint(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt[:500]
        try:
            from effgen.core.messages import Message

            if isinstance(prompt, Message):
                return prompt.text[:500]
            if isinstance(prompt, list):
                text = "\n".join(p.text for p in prompt if isinstance(p, Message))
                if text:
                    return text[:500]
        except Exception:
            pass
        return str(prompt)[:500]

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
                       task: str,
                       mode: AgentMode = AgentMode.AUTO,
                       context: dict[str, Any] | None = None,
                       **kwargs) -> AgentResponse:
        """
        Truly asynchronous version of run().

        Runs the synchronous run() method in a thread executor so it
        doesn't block the event loop, while remaining compatible with
        async callers.

        Args:
            task: Task description
            mode: Execution mode
            context: Optional context
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        import functools
        loop = asyncio.get_running_loop()
        func = functools.partial(self.run, task, mode, context, **kwargs)
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
        if self.long_term_memory is not None:
            try:
                self.long_term_memory.close()
            except Exception:
                pass
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
        if not getattr(self, '_closed', True):
            logger.warning(
                f"Agent '{getattr(self, 'name', '?')}' was garbage-collected "
                "without calling close(). Use 'with Agent(config) as agent:' "
                "or call agent.close() explicitly."
            )

    def stream(self,
               task: str,
               mode: AgentMode = AgentMode.AUTO,
               context: dict[str, Any] | None = None,
               on_thought: Callable[[str], None] | None = None,
               on_tool_call: Callable[[str, str], None] | None = None,
               on_observation: Callable[[str], None] | None = None,
               on_answer: Callable[[str], None] | None = None,
               **kwargs) -> Iterator[str]:
        """
        Stream response token by token using real model streaming.

        Streams the ReAct loop in real-time:
        - Yields thought tokens as they generate
        - Pauses streaming during tool execution
        - Yields observation text after tool calls
        - Yields final answer tokens

        Args:
            task: Task description
            mode: Execution mode
            context: Optional context
            on_thought: Callback for thought tokens
            on_tool_call: Callback(tool_name, tool_input) when a tool is called
            on_observation: Callback for tool observation text
            on_answer: Callback for final answer tokens
            **kwargs: Additional arguments

        Yields:
            Response tokens (str)
        """
        if self.model is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no model loaded. "
                "Provide a model in AgentConfig or use a mock for testing."
            )

        context = context or {}
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

            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")
                yield f"\n[Error: {e}]"
                return
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
                answer = parsed["final_answer"]
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
