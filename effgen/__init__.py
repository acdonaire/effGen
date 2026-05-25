"""
effGen: A comprehensive framework for building agents with Small Language Models.

This framework enables SLMs to function as powerful agentic systems through:
- Tool integration (built-in tools + MCP/A2A/ACP protocols)
- Advanced prompt engineering optimized for SLMs
- Smart sub-agent decomposition for complex tasks
- Multi-GPU support with vLLM and Transformers
- Comprehensive configuration management
"""

# ruff: noqa: I001

__version__ = "0.2.9"
__author__ = "effGen Team"
__license__ = "Apache-2.0"

# Core imports
# Configuration imports
from effgen.config import Config, ConfigLoader, ConfigValidator
from effgen.core.agent import Agent, AgentConfig

# Multimodal message schema
from effgen.core.messages import (
    AudioPart,
    ContentPart,
    ImagePart,
    Message as MultimodalMessage,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    VideoPart,
)
from effgen.core.multimodal import audio_from, image_from, video_from
from effgen.errors import CapabilityNotSupportedError, InvalidMultimodalContent
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
    CostBasedPolicy,
    CostTracker,
    FireworksAdapter,
    FirstAvailablePolicy,
    GeminiAdapter,
    GenerationConfig,
    GenerationResult,
    GroqAdapter,
    HFInferenceAdapter,
    LatencyBasedPolicy,
    LatencyTracker,
    ModelLoader,
    OpenAIAdapter,
    PolicyBasedRouter,
    ProviderModelPair,
    ReplicateAdapter,
    RetryPolicy,
    RouterDecision,
    RouterEvent,
    RoutingContext,
    RoutingPolicy,
    SQLiteCostStore,
    StreamChunk,
    TogetherAdapter,
    TransformersEngine,
    VLLMEngine,
    load_model,
)
from effgen.models._rate_limit import RateLimitCoordinator, RateLimitExceeded  # noqa: I001
from effgen.models._rate_limit_store import SQLiteRateLimitStore  # noqa: I001
from effgen.models.auth import check_keys
from effgen.models.cerebras_models import available_models as cerebras_available_models
from effgen.models.cerebras_models import free_tier_models as cerebras_free_tier_models
from effgen.models.cerebras_models import model_info as cerebras_model_info
from effgen.models.errors import (  # noqa: I001
    AllCandidatesExhaustedError,
    AmbiguousModelError,
    BudgetExceededError,
    InvalidRequestError,
    ModelAuthError,
    ModelNotFoundError,
    ModelRefusalError,
    ModelTimeoutError,
    ModelUnavailableError,
    NoCandidateWithinBudgetError,
    ProviderTransientError,
    ToolIncompatibleError,
)
from effgen.models.fireworks_models import available_models as fireworks_available_models
from effgen.models.fireworks_models import chat_models as fireworks_chat_models
from effgen.models.fireworks_models import pricing_table as fireworks_pricing_table
from effgen.models.fireworks_models import refresh_models as fireworks_refresh_models
from effgen.models.fireworks_models import tool_capable_models as fireworks_tool_capable_models
from effgen.models.gemini_models import available_models as gemini_available_models
from effgen.models.gemini_models import free_tier_models as gemini_free_tier_models
from effgen.models.gemini_models import model_info as gemini_model_info
from effgen.models.gemini_models import recommended_models as gemini_recommended_models
from effgen.models.groq_models import available_models as groq_available_models
from effgen.models.groq_models import chat_models as groq_chat_models
from effgen.models.groq_models import tool_capable_models as groq_tool_capable_models
from effgen.models.hf_inference_models import available_models as hf_available_models
from effgen.models.hf_inference_models import catalog_summary as hf_catalog_summary
from effgen.models.hf_inference_models import chat_models as hf_chat_models
from effgen.models.hf_inference_models import cheapest_provider as hf_cheapest_provider
from effgen.models.hf_inference_models import check_drift as hf_check_drift
from effgen.models.hf_inference_models import get_model_info as hf_get_model_info
from effgen.models.hf_inference_models import list_providers_for as hf_list_providers_for
from effgen.models.hf_inference_models import refresh_models as hf_refresh_models
from effgen.models.hf_inference_models import serverless_models as hf_serverless_models
from effgen.models.hf_inference_models import suggest_alternatives as hf_suggest_alternatives
from effgen.models.hf_inference_models import tool_capable_models as hf_tool_capable_models
from effgen.models.openai_models import available_models as openai_available_models
from effgen.models.openai_models import chat_models as openai_chat_models
from effgen.models.openai_models import model_info as openai_model_info
from effgen.models.openai_models import reasoning_models as openai_reasoning_models  # noqa: I001
from effgen.models.openai_schema import to_openai_schema
from effgen.models.registry import ProviderRegistry, list_models, list_providers, lookup
from effgen.models.replicate_models import available_models as replicate_available_models
from effgen.models.replicate_models import get_model_info as replicate_get_model_info
from effgen.models.replicate_models import refresh_models as replicate_refresh_models
from effgen.models.replicate_models import streaming_models as replicate_streaming_models
from effgen.models.replicate_models import tool_capable_models as replicate_tool_capable_models
from effgen.models.together_models import available_models as together_available_models
from effgen.models.together_models import chat_models as together_chat_models
from effgen.models.together_models import pricing_table as together_pricing_table
from effgen.models.together_models import refresh_models as together_refresh_models
from effgen.models.together_models import serverless_models as together_serverless_models
from effgen.models.together_models import tool_capable_models as together_tool_capable_models

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

# Security / supply-chain exports
from effgen.security import (
    HashDriftWarning,
    VerificationResult,
    verify_installed_hashes,
    verify_on_startup,
)

__all__ = [
    # Multimodal message schema
    "Role",
    "TextPart",
    "ImagePart",
    "AudioPart",
    "VideoPart",
    "ToolCallPart",
    "ToolResultPart",
    "ContentPart",
    "MultimodalMessage",
    "image_from",
    "audio_from",
    "video_from",
    "CapabilityNotSupportedError",
    "InvalidMultimodalContent",

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
    "GroqAdapter",
    "TogetherAdapter",
    "FireworksAdapter",
    "ReplicateAdapter",
    "HFInferenceAdapter",
    "ModelLoader",
    "GenerationConfig",
    "GenerationResult",
    # Router (v0.2.4+)
    "PolicyBasedRouter",
    "RoutingPolicy",
    "RoutingContext",
    "RouterDecision",
    "RouterEvent",
    "ProviderModelPair",
    "FirstAvailablePolicy",
    "CostBasedPolicy",
    "LatencyBasedPolicy",
    "RetryPolicy",
    # Tracking (v0.2.4+)
    "LatencyTracker",
    "CostTracker",
    "SQLiteCostStore",
    "RateLimitCoordinator",
    "RateLimitExceeded",
    "SQLiteRateLimitStore",
    # Errors
    "ModelRefusalError",
    "ModelAuthError",
    "ModelTimeoutError",
    "ModelUnavailableError",
    "ModelNotFoundError",
    "AmbiguousModelError",
    "NoCandidateWithinBudgetError",
    "ToolIncompatibleError",
    "AllCandidatesExhaustedError",
    "BudgetExceededError",
    "ProviderTransientError",
    "InvalidRequestError",
    "to_openai_schema",
    # Provider registry + auth
    "ProviderRegistry",
    "list_providers",
    "list_models",
    "lookup",
    "check_keys",
    # Cerebras helpers
    "cerebras_available_models",
    "cerebras_free_tier_models",
    "cerebras_model_info",
    # Groq helpers
    "groq_available_models",
    "groq_chat_models",
    "groq_tool_capable_models",
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
    # Together helpers
    "together_available_models",
    "together_chat_models",
    "together_tool_capable_models",
    "together_pricing_table",
    "together_refresh_models",
    "together_serverless_models",
    # Fireworks helpers
    "fireworks_available_models",
    "fireworks_chat_models",
    "fireworks_tool_capable_models",
    "fireworks_pricing_table",
    "fireworks_refresh_models",
    # Replicate helpers
    "replicate_available_models",
    "replicate_streaming_models",
    "replicate_tool_capable_models",
    "replicate_refresh_models",
    "replicate_get_model_info",
    # HF Inference helpers
    "hf_available_models",
    "hf_chat_models",
    "hf_tool_capable_models",
    "hf_serverless_models",
    "hf_suggest_alternatives",
    "hf_get_model_info",
    "hf_refresh_models",
    "hf_check_drift",
    "hf_catalog_summary",
    "hf_list_providers_for",
    "hf_cheapest_provider",

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

    # Security
    "HashDriftWarning",
    "VerificationResult",
    "verify_installed_hashes",
    "verify_on_startup",
]

# Supply-chain integrity check — runs only when EFFGEN_VERIFY_HASHES=1.
# Invoked at the end of package init so it executes once on `import effgen`
# without interrupting the import block; it only reads installed metadata and
# never initialises model adapters or makes network calls.
verify_on_startup()
