"""
effGen: A comprehensive framework for building agents with Small Language Models.

This framework enables SLMs to function as powerful agentic systems through:
- Tool integration (built-in tools + MCP/A2A/ACP protocols)
- Advanced prompt engineering optimized for SLMs
- Smart sub-agent decomposition for complex tasks
- Multi-GPU support with vLLM and Transformers
- Comprehensive configuration management
"""

__version__ = "0.2.1"
__author__ = "effGen Team"
__license__ = "Apache-2.0"

# Core imports
# Configuration imports
from effgen.config import Config, ConfigLoader, ConfigValidator
from effgen.core.agent import Agent, AgentConfig
from effgen.core.aggregation import ResultAggregator

# Batch & Aggregation imports
from effgen.core.batch import BatchConfig, BatchResult, BatchRunner
from effgen.core.state import AgentState
from effgen.core.task import SubTask, Task, TaskPriority, TaskStatus

# Domain imports
from effgen.domains import (
    Domain,
    FinanceDomain,
    HealthDomain,
    KeywordExpander,
    LegalDomain,
    ScienceDomain,
    TechDomain,
)

# GPU imports
from effgen.gpu import GPUAllocator, GPUMonitor, gpu_utils

# Guardrails imports
from effgen.guardrails import (
    Guardrail,
    GuardrailChain,
    GuardrailPosition,
    GuardrailResult,
    LengthGuardrail,
    PIIGuardrail,
    PromptInjectionGuardrail,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    ToolPermissionGuardrail,
    TopicGuardrail,
    ToxicityGuardrail,
    get_guardrail_preset,
)

# Memory imports
from effgen.memory import (
    ImportanceLevel,
    JSONStorageBackend,
    LongTermMemory,
    MemoryEntry,
    MemoryType,
    Message,
    MessageRole,
    ShortTermMemory,
    SQLiteStorageBackend,
    VectorMemoryStore,
)

# Model imports
from effgen.models import (
    AnthropicAdapter,
    BaseModel,
    CerebrasAdapter,
    GeminiAdapter,
    GenerationConfig,
    GenerationResult,
    ModelLoader,
    OpenAIAdapter,
    StreamChunk,
    TransformersEngine,
    VLLMEngine,
    load_model,
)
from effgen.models._rate_limit import RateLimitCoordinator, RateLimitExceeded  # noqa: I001
from effgen.models.cerebras_models import available_models as cerebras_available_models
from effgen.models.cerebras_models import free_tier_models as cerebras_free_tier_models
from effgen.models.cerebras_models import model_info as cerebras_model_info
from effgen.models.errors import ModelRefusalError, ToolIncompatibleError
from effgen.models.gemini_models import available_models as gemini_available_models
from effgen.models.gemini_models import free_tier_models as gemini_free_tier_models
from effgen.models.gemini_models import model_info as gemini_model_info
from effgen.models.gemini_models import recommended_models as gemini_recommended_models
from effgen.models.openai_models import available_models as openai_available_models
from effgen.models.openai_models import chat_models as openai_chat_models
from effgen.models.openai_models import model_info as openai_model_info
from effgen.models.openai_models import reasoning_models as openai_reasoning_models  # noqa: I001
from effgen.models.openai_schema import to_openai_schema

# Preset imports
from effgen.presets import create_agent, list_presets

# Prompt imports
from effgen.prompts import ChainManager, PromptOptimizer, TemplateManager

# Tool imports
from effgen.tools import BaseTool, ToolRegistry
from effgen.tools import get_registry as get_tool_registry

# OpenAI native tool imports
try:
    from effgen.tools.builtin.openai_native import (
        OpenAICodeInterpreterTool,
        OpenAIFileSearchTool,
        OpenAINativeTool,
        OpenAIWebSearchTool,
    )
except ImportError:
    pass

# Gemini native tool imports
try:
    from effgen.tools.builtin.gemini_native import (
        GeminiCodeExecutionTool,
        GeminiNativeTool,
        GeminiUrlContextTool,
        GoogleSearchTool,
    )
except ImportError:
    pass

# MLX engine imports (Apple Silicon only)
try:
    from effgen.models.mlx_engine import MLXEngine
    from effgen.models.mlx_vlm_engine import MLXVLMEngine
except ImportError:
    pass

# Additional convenience imports
try:
    from effgen.tools.fallback import ToolFallbackChain
except ImportError:
    pass

try:
    from effgen.utils.circuit_breaker import CircuitBreaker
except ImportError:
    pass

try:
    from effgen.prompts.tool_prompt_generator import ToolPromptGenerator
except ImportError:
    pass

try:
    from effgen.prompts.agent_system_prompt import AgentSystemPromptBuilder
except ImportError:
    pass

# Eval imports
try:
    from effgen.eval import (
        AgentEvaluator,
        EvalResult,
        ModelComparison,
        RegressionTracker,
        SuiteResults,
        TestCase,
        TestSuite,
    )
except ImportError:
    pass

# Execution imports
from effgen.execution import (
    CodeExecutor,
    CodeValidator,
    ExecutionResult,
    ExecutionStatus,
    SandboxConfig,
    ValidationResult,
    ValidationSeverity,
)

__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    "Task",
    "SubTask",
    "TaskStatus",
    "TaskPriority",
    "AgentState",

    # Models
    "load_model",
    "BaseModel",
    "VLLMEngine",
    "TransformersEngine",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "StreamChunk",
    "GeminiAdapter",
    "CerebrasAdapter",
    "ModelLoader",
    "GenerationConfig",
    "GenerationResult",
    "RateLimitCoordinator",
    "RateLimitExceeded",
    "ModelRefusalError",
    "to_openai_schema",
    # Cerebras helpers
    "cerebras_available_models",
    "cerebras_free_tier_models",
    "cerebras_model_info",
    # OpenAI helpers
    "openai_available_models",
    "openai_chat_models",
    "openai_reasoning_models",
    "openai_model_info",
    # Gemini helpers
    "gemini_available_models",
    "gemini_free_tier_models",
    "gemini_model_info",
    "gemini_recommended_models",

    # Tools
    "BaseTool",
    "ToolRegistry",
    "get_tool_registry",

    # Configuration
    "ConfigLoader",
    "Config",
    "ConfigValidator",

    # Prompts
    "TemplateManager",
    "ChainManager",
    "PromptOptimizer",

    # GPU
    "GPUAllocator",
    "GPUMonitor",
    "gpu_utils",

    # Memory
    "ShortTermMemory",
    "LongTermMemory",
    "VectorMemoryStore",
    "Message",
    "MessageRole",
    "MemoryEntry",
    "MemoryType",
    "ImportanceLevel",
    "JSONStorageBackend",
    "SQLiteStorageBackend",

    # Batch & Aggregation
    "BatchRunner",
    "BatchConfig",
    "BatchResult",
    "ResultAggregator",

    # Domains
    "Domain",
    "KeywordExpander",
    "TechDomain",
    "ScienceDomain",
    "FinanceDomain",
    "HealthDomain",
    "LegalDomain",

    # Presets
    "create_agent",
    "list_presets",

    # Additional convenience exports
    "ToolFallbackChain",
    "CircuitBreaker",
    "ToolPromptGenerator",
    "AgentSystemPromptBuilder",

    # Eval
    "AgentEvaluator",
    "EvalResult",
    "SuiteResults",
    "TestCase",
    "TestSuite",
    "ModelComparison",
    "RegressionTracker",

    # Execution
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "CodeValidator",
    "ValidationResult",
    "ValidationSeverity",
    "SandboxConfig",
]
