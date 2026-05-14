"""
Model infrastructure for effGen framework.

This package provides a unified interface for various model backends including:
- vLLM for fast local inference
- HuggingFace Transformers as fallback
- OpenAI API adapter
- Anthropic Claude API adapter
- Google Gemini API adapter

Example:
    >>> from effgen.models import load_model
    >>>
    >>> # Load a local HuggingFace model (tries vLLM, falls back to Transformers)
    >>> model = load_model("meta-llama/Llama-2-7b-hf")
    >>>
    >>> # Load an API model
    >>> gpt4 = load_model("gpt-4")
    >>> claude = load_model("claude-3-opus-20240229")
    >>>
    >>> # Generate text
    >>> result = model.generate("What is the capital of France?")
    >>> print(result.text)
"""

from effgen.models._rate_limit import RateLimitCoordinator
from effgen.models._rate_limit_store import SQLiteRateLimitStore
from effgen.models.anthropic_adapter import AnthropicAdapter, StreamChunk
from effgen.models.anthropic_cache import (
    MAX_CACHE_BREAKPOINTS,
    apply_cache_to_last_tool,
    apply_cache_to_system,
    get_min_cache_tokens,
    mark_cached,
)
from effgen.models.anthropic_models import ANTHROPIC_MODELS
from effgen.models.anthropic_models import get_model_info as get_anthropic_model_info
from effgen.models.auth import check_keys
from effgen.models.base import (
    BaseModel,
    BatchModel,
    FunctionCallingModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)
from effgen.models.batching import ContinuousBatcher
from effgen.models.capabilities import (
    MODEL_CAPABILITIES,
    Capability,
    ModelCapability,
    estimate_capability,
    get_model_capability,
    list_registered_models,
    register_model_capability,
)
from effgen.models.cerebras_adapter import CerebrasAdapter
from effgen.models.errors import (
    AllCandidatesExhaustedError,
    AmbiguousModelError,
    InvalidRequestError,
    ModelAuthError,
    ModelNotFoundError,
    ModelRefusalError,
    ModelTimeoutError,
    ModelUnavailableError,
    NoCandidateWithinBudgetError,
    ProviderTransientError,
)
from effgen.models.fireworks_adapter import FireworksAdapter
from effgen.models.fireworks_models import FIREWORKS_MODELS
from effgen.models.gemini_adapter import GeminiAdapter
from effgen.models.gemini_files import FileRef, upload_file
from effgen.models.groq_adapter import GroqAdapter
from effgen.models.groq_models import GROQ_MODELS
from effgen.models.hf_inference_adapter import HFInferenceAdapter
from effgen.models.hf_inference_models import HF_MODELS
from effgen.models.lazy import LazyModel
from effgen.models.model_loader import ModelLoader, load_model
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.openai_schema import to_openai_schema
from effgen.models.pool import ModelPool, PoolConfig
from effgen.models.registry import ProviderRegistry, list_models, list_providers, lookup
from effgen.models.replicate_adapter import ReplicateAdapter
from effgen.models.replicate_models import REPLICATE_MODELS
from effgen.models.router import (
    ComplexityEstimate,
    ComplexityLevel,
    ModelRouter,
    NoCandidateError,
    PolicyBasedRouter,
    ProviderModelPair,
    RouterDecision,
    RouterEvent,
    RoutingConfig,
    RoutingContext,
    RoutingDecision,
    RoutingPolicy,
    estimate_complexity,
)
from effgen.models.routing.cost import CostBasedPolicy
from effgen.models.routing.first_available import FirstAvailablePolicy
from effgen.models.routing.retry import RetryPolicy
from effgen.models.together_adapter import TogetherAdapter
from effgen.models.together_models import TOGETHER_MODELS
from effgen.models.transformers_engine import TransformersEngine
from effgen.models.vllm_engine import VLLMEngine

# MLX engines (Apple Silicon only, lazy import)
try:
    from effgen.models.mlx_engine import MLXEngine
except ImportError:
    pass

try:
    from effgen.models.mlx_vlm_engine import MLXVLMEngine
except ImportError:
    pass

__all__ = [
    # Base classes
    "BaseModel",
    "BatchModel",
    "FunctionCallingModel",
    "ModelType",

    # Data classes
    "GenerationConfig",
    "GenerationResult",
    "TokenCount",

    # Engine implementations
    "VLLMEngine",
    "TransformersEngine",
    "MLXEngine",
    "MLXVLMEngine",

    # API adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "CerebrasAdapter",
    "GroqAdapter",
    "TogetherAdapter",
    "FireworksAdapter",
    "ReplicateAdapter",
    "HFInferenceAdapter",

    # Model registries
    "GROQ_MODELS",
    "TOGETHER_MODELS",
    "FIREWORKS_MODELS",
    "REPLICATE_MODELS",
    "HF_MODELS",

    # Anthropic streaming
    "StreamChunk",

    # Loader
    "ModelLoader",
    "load_model",

    # Complexity-based router (v0.2.3, back-compat)
    "ModelRouter",
    "RoutingConfig",
    "RoutingDecision",
    "ComplexityEstimate",
    "ComplexityLevel",
    "estimate_complexity",

    # Policy-based router (v0.2.4+)
    "PolicyBasedRouter",
    "RoutingPolicy",
    "RoutingContext",
    "RouterDecision",
    "RouterEvent",
    "ProviderModelPair",
    "NoCandidateError",
    "FirstAvailablePolicy",
    "CostBasedPolicy",
    "RetryPolicy",

    # Capabilities
    "Capability",
    "ModelCapability",
    "MODEL_CAPABILITIES",
    "register_model_capability",
    "get_model_capability",
    "estimate_capability",
    "list_registered_models",

    # Pool
    "ModelPool",
    "PoolConfig",

    "LazyModel",
    "ContinuousBatcher",

    # Rate-limit coordination
    "RateLimitCoordinator",
    "SQLiteRateLimitStore",

    # Errors
    "ModelRefusalError",
    "ModelAuthError",
    "ModelTimeoutError",
    "ModelUnavailableError",
    "ModelNotFoundError",
    "AmbiguousModelError",
    "NoCandidateWithinBudgetError",
    "ProviderTransientError",
    "AllCandidatesExhaustedError",
    "InvalidRequestError",

    # Provider registry + auth
    "ProviderRegistry",
    "list_providers",
    "list_models",
    "lookup",
    "check_keys",

    # Schema helpers
    "to_openai_schema",

    # Anthropic registry
    "ANTHROPIC_MODELS",
    "get_anthropic_model_info",

    # Anthropic cache helpers
    "mark_cached",
    "apply_cache_to_system",
    "apply_cache_to_last_tool",
    "get_min_cache_tokens",
    "MAX_CACHE_BREAKPOINTS",

    # Gemini Files API
    "FileRef",
    "upload_file",
]
