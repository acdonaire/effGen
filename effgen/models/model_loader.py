"""
Smart model loader with automatic detection and fallback.

This module provides intelligent model loading with:
- Automatic model type detection (HuggingFace vs API)
- Transformers-first with vLLM as optional production backend
- GPU allocation and VRAM management
- Quantization decision based on available memory
- Model validation and health checks
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from effgen.models.base import BaseModel, ModelType

if TYPE_CHECKING:
    # Cloud adapters and the heavy local-inference deps (torch, the
    # Transformers/vLLM engines) are imported lazily inside the ``_load_*``
    # methods that actually construct them. This matches the six other adapters
    # (HF/Replicate/Cerebras/Groq/Together/Fireworks) and — importantly — breaks
    # the import cycle that arises when ``effgen.models`` is imported before
    # ``effgen.core.agent`` (anthropic_adapter -> core.messages -> core.agent ->
    # model_loader). It also keeps `from effgen import Agent` / the CLI from
    # pulling torch or transformers for a pure cloud-API workflow.
    from effgen.models.anthropic_adapter import AnthropicAdapter
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.models.openai_adapter import OpenAIAdapter
    from effgen.models.transformers_engine import TransformersEngine
    from effgen.models.vllm_engine import VLLMEngine

# Cerebras import is deferred to avoid hard dependency when cerebras extra is absent.
_CerebrasAdapter = None
# Groq import is deferred to avoid hard dependency when groq extra is absent.
_GroqAdapter = None
# Together import is deferred to avoid hard dependency when together extra is absent.
_TogetherAdapter = None
# Fireworks import is deferred to avoid hard dependency when fireworks extra is absent.
_FireworksAdapter = None
# Replicate import is deferred to avoid hard dependency when replicate extra is absent.
_ReplicateAdapter = None
# HF Inference import is deferred to avoid hard dependency when hf extra is absent.
_HFInferenceAdapter = None


def _get_hf_inference_adapter():
    global _HFInferenceAdapter
    if _HFInferenceAdapter is None:
        from effgen.models.hf_inference_adapter import HFInferenceAdapter
        _HFInferenceAdapter = HFInferenceAdapter
    return _HFInferenceAdapter


def _get_replicate_adapter():
    global _ReplicateAdapter
    if _ReplicateAdapter is None:
        from effgen.models.replicate_adapter import ReplicateAdapter
        _ReplicateAdapter = ReplicateAdapter
    return _ReplicateAdapter


def _get_cerebras_adapter():
    global _CerebrasAdapter
    if _CerebrasAdapter is None:
        from effgen.models.cerebras_adapter import CerebrasAdapter
        _CerebrasAdapter = CerebrasAdapter
    return _CerebrasAdapter


def _get_groq_adapter():
    global _GroqAdapter
    if _GroqAdapter is None:
        from effgen.models.groq_adapter import GroqAdapter
        _GroqAdapter = GroqAdapter
    return _GroqAdapter


def _get_together_adapter():
    global _TogetherAdapter
    if _TogetherAdapter is None:
        from effgen.models.together_adapter import TogetherAdapter
        _TogetherAdapter = TogetherAdapter
    return _TogetherAdapter


def _get_fireworks_adapter():
    global _FireworksAdapter
    if _FireworksAdapter is None:
        from effgen.models.fireworks_adapter import FireworksAdapter
        _FireworksAdapter = FireworksAdapter
    return _FireworksAdapter


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Smart model loader with automatic detection and configuration.

    This class handles:
    1. Model type detection (local, HuggingFace, or API)
    2. Engine selection (Transformers default, vLLM optional, or API adapter)
    3. GPU allocation and memory management
    4. Automatic quantization decisions
    5. Fallback strategies
    6. Model validation

    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load_model("meta-llama/Llama-2-7b-hf")
        >>> # Uses Transformers by default, can specify vLLM with engine='vllm'

        >>> model = loader.load_model("gpt-4")
        >>> # Automatically uses OpenAI adapter
    """

    # API model prefixes for automatic detection
    OPENAI_MODELS = [
        "gpt-3.5", "gpt-4", "gpt-5", "text-davinci", "text-curie",
        "text-babbage", "text-ada",
        "o1", "o1-mini", "o1-preview",
        "o3", "o3-mini",
        "o4", "o4-mini",
    ]

    ANTHROPIC_MODELS = [
        "claude-3", "claude-2", "claude-instant", "claude-4", "claude-opus", "claude-sonnet", "claude-haiku"
    ]

    GEMINI_MODELS = [
        "gemini-pro", "gemini-ultra", "gemini-flash", "gemini-1.5",
        "gemini-2", "gemini-3",
    ]

    # Local-engine prefixes recognized in an "engine:model_id" string, mirroring
    # the cloud "provider:model_id" syntax (e.g. "transformers:Qwen/Qwen2.5-7B-Instruct").
    _LOCAL_ENGINE_PREFIXES = frozenset({"transformers", "vllm", "gguf", "mlx"})

    def __init__(
        self,
        cache_dir: str | None = None,
        default_device: str = "auto",
        force_engine: str | None = None,
    ):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory to cache downloaded models
            default_device: Default device allocation ('auto', 'cuda', 'cpu')
            force_engine: Force specific engine ('vllm', 'transformers', 'auto-fast',
                or None for auto). 'auto-fast' prefers vLLM when it is importable
                and a GPU is usable, else Transformers; None defaults to Transformers.
        """
        # Expand ~ to full path and use environment variable if set
        default_cache = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
        self.cache_dir = os.path.expanduser(cache_dir) if cache_dir else default_cache
        self.default_device = default_device
        self.force_engine = force_engine

        self.loaded_models: dict[str, BaseModel] = {}

    def load_model(
        self,
        model_name: str,
        engine_config: dict[str, Any] | None = None,
        **kwargs
    ) -> BaseModel:
        """
        Load a model with automatic detection and configuration.

        Args:
            model_name: Model identifier (HuggingFace ID, local path, or API model name)
            engine_config: Optional engine-specific configuration
            **kwargs: Additional model parameters

        Returns:
            Loaded model instance ready for inference

        Raises:
            ValueError: If model_name is invalid or unsupported
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading model: {model_name}")

        # Check if already loaded
        if model_name in self.loaded_models:
            logger.info(f"Model '{model_name}' already loaded, returning cached instance")
            return self.loaded_models[model_name]

        # Explicit provider routing (e.g. provider="cerebras", provider="openai")
        provider = kwargs.pop("provider", None)

        # An explicit but unknown provider (e.g. a typo'd "grok") must fail
        # fast and loud rather than silently falling through to a local
        # HuggingFace download.  Normalize common aliases (google -> gemini).
        if provider is not None:
            _known = {
                "openai", "anthropic", "gemini", "cerebras", "groq",
                "together", "fireworks", "replicate", "hf", "hf_inference",
            }
            _aliases = {"google": "gemini", "huggingface": "hf", "claude": "anthropic"}
            _p = str(provider).strip().lower()
            _p = _aliases.get(_p, _p)
            if _p not in _known:
                import difflib
                _close = difflib.get_close_matches(_p, sorted(_known), n=1, cutoff=0.5)
                _hint = f" Did you mean '{_close[0]}'?" if _close else ""
                raise ValueError(
                    f"Unknown provider '{provider}'.{_hint} "
                    f"Known providers: {', '.join(sorted(_known - {'hf_inference'}))}."
                )
            provider = _p

        # Support "provider:model_id" prefix syntax via ProviderRegistry
        if provider is None and isinstance(model_name, str) and ":" in model_name:
            _prefix, _rest = model_name.split(":", 1)
            try:
                from effgen.models.registry import ProviderRegistry
                if _prefix in ProviderRegistry.list_providers():
                    provider = _prefix
                    model_name = _rest
            except Exception:
                logger.debug("Provider-prefix registry lookup failed", exc_info=True)

        # Support "engine:model_id" prefix syntax for local engines, mirroring
        # the cloud "provider:model_id" syntax above (e.g.
        # "transformers:Qwen/Qwen2.5-7B-Instruct"). Without this, the whole
        # string is passed to the HuggingFace repo-id validator, which rejects
        # the colon with a cryptic HFValidationError instead of running the
        # model locally with the requested engine.
        if (
            provider is None
            and isinstance(model_name, str)
            and ":" in model_name
        ):
            _eng_prefix, _eng_rest = model_name.split(":", 1)
            if _eng_prefix in self._LOCAL_ENGINE_PREFIXES and _eng_rest:
                if self.force_engine is None:
                    self.force_engine = _eng_prefix
                model_name = _eng_rest

        # A colon-prefixed id that matched neither a provider nor a local-engine
        # prefix above is a typo'd prefix (e.g. "nonexistant:foo"), not a local
        # HuggingFace repo id — HF repo ids cannot contain a colon. Fail fast
        # with the valid prefixes instead of falling through to the local
        # loader, which would reject the whole string with an unrelated
        # "invalid repo id" error that names internal loader machinery.
        if (
            provider is None
            and isinstance(model_name, str)
            and ":" in model_name
            and not os.path.exists(model_name)
        ):
            _bad_prefix, _ = model_name.split(":", 1)
            try:
                from effgen.models.registry import ProviderRegistry
                known_providers = sorted(ProviderRegistry.list_providers())
            except Exception:
                known_providers = []
            raise ValueError(
                f"Unknown provider or engine prefix {_bad_prefix!r} in model id "
                f"{model_name!r}. Known providers: {', '.join(known_providers) or 'none configured'}. "
                f"Known local engines: {', '.join(sorted(self._LOCAL_ENGINE_PREFIXES))}."
            )

        # Route / disambiguate bare cloud model ids by consulting the model
        # catalog directly.  Without this, a documented
        # provider id such as ``gpt-oss-120b`` or ``llama-3.3-70b-versatile``
        # falls through to the local HuggingFace path and fails with a confusing
        # download error instead of calling the provider.  Bare ids that no cloud
        # catalog knows (the normal case for local HF repos / paths) are left
        # untouched so the existing local detection still runs.
        # A "/" in a bare id means an org/model HuggingFace-style repo (also how
        # Together/Fireworks/Replicate/HF list their models), so it stays on the
        # local/HF path; only slash-free cloud slugs (gpt-oss-120b,
        # llama-3.3-70b-versatile, zai-glm-4.7, …) are candidates for routing.
        if (
            provider is None
            and isinstance(model_name, str)
            and "/" not in model_name
            and not os.path.exists(model_name)
            and not model_name.lower().endswith(".gguf")
        ):
            try:
                from effgen.models import _catalog, _refresh
                from effgen.models.errors import AmbiguousModelError

                candidates = _catalog.providers_for(model_name)
                if len(candidates) > 1:
                    raise AmbiguousModelError(model_name, candidates)
                if len(candidates) == 1:
                    only = candidates[0]
                    if _refresh.has_credentials(only):
                        provider = only
                        logger.info(
                            "Routing bare model id %r to provider %r (catalog match)",
                            model_name, only,
                        )
                    else:
                        envs = _refresh._KEY_ENVS.get(only, ("<API_KEY>",))[0]
                        logger.warning(
                            "Model %r is a known %s model but %s is not set — "
                            "treating it as a local model. To call the provider, "
                            "pass provider=%r (or use '%s:%s') and set %s.",
                            model_name, only, envs, only, only, model_name, envs,
                        )
            except AmbiguousModelError:
                raise
            except Exception:
                logger.debug("Bare-id provider routing lookup failed", exc_info=True)

        if provider == "openai":
            model = self._load_openai_model(model_name, engine_config, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "anthropic":
            model = self._load_anthropic_model(model_name, engine_config, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "gemini":
            model = self._load_gemini_model(model_name, engine_config, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "cerebras":
            CerebrasAdapter = _get_cerebras_adapter()
            api_key = kwargs.pop("api_key", None)
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = CerebrasAdapter(model_name=model_name, api_key=api_key, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "groq":
            GroqAdapter = _get_groq_adapter()
            api_key = kwargs.pop("api_key", None)
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = GroqAdapter(model_name=model_name, api_key=api_key, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "together":
            TogetherAdapter = _get_together_adapter()
            api_key = kwargs.pop("api_key", None)
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = TogetherAdapter(model_name=model_name, api_key=api_key, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "fireworks":
            FireworksAdapter = _get_fireworks_adapter()
            api_key = kwargs.pop("api_key", None)
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = FireworksAdapter(model_name=model_name, api_key=api_key, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "replicate":
            ReplicateAdapter = _get_replicate_adapter()
            api_token = kwargs.pop("api_token", kwargs.pop("api_key", None))
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = ReplicateAdapter(model_name=model_name, api_token=api_token, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider in ("hf_inference", "hf"):
            # `hf:` is the remote HuggingFace Inference API, not local execution.
            # A privacy-conscious user expecting their data to stay on-box should
            # know they need a local engine instead — nudge once, don't spam.
            logger.info(
                "Loading '%s' via the HuggingFace Inference API (remote). For "
                "local GPU execution of this model, load it with "
                "engine='transformers' (or 'vllm') instead of the 'hf:' prefix.",
                model_name,
            )
            HFInferenceAdapter = _get_hf_inference_adapter()
            api_token = kwargs.pop("api_token", kwargs.pop("api_key", None))
            endpoint_url = kwargs.pop("endpoint_url", None)
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = HFInferenceAdapter(
                model_name=model_name,
                api_token=api_token,
                endpoint_url=endpoint_url,
                **kwargs,
            )
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model

        # GGUF files take a dedicated path (llama-cpp-python).
        if isinstance(model_name, str) and model_name.lower().endswith(".gguf"):
            from .gguf_engine import GGUFEngine

            gguf_params = dict(kwargs)
            gguf_params.pop("apply_chat_template", None)
            model = GGUFEngine(model_name=model_name, **gguf_params)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            logger.info(f"GGUF model '{model_name}' loaded successfully")
            return model

        # Detect model type
        model_type = self._detect_model_type(model_name)
        logger.info(f"Detected model type: {model_type.value}")

        # Load based on type
        if model_type == ModelType.OPENAI:
            model = self._load_openai_model(model_name, engine_config, **kwargs)
        elif model_type == ModelType.ANTHROPIC:
            model = self._load_anthropic_model(model_name, engine_config, **kwargs)
        elif model_type == ModelType.GEMINI:
            model = self._load_gemini_model(model_name, engine_config, **kwargs)
        elif model_type == ModelType.MLX:
            # MLX model detected (e.g., mlx-community/ prefix) — use MLX engine
            if self.force_engine is None:
                self.force_engine = "mlx"
            model = self._load_huggingface_model(model_name, engine_config, **kwargs)
        elif model_type == ModelType.MLX_VLM:
            if self.force_engine is None:
                self.force_engine = "mlx_vlm"
            model = self._load_huggingface_model(model_name, engine_config, **kwargs)
        else:
            # HuggingFace model - use Transformers by default, vLLM/MLX optional
            model = self._load_huggingface_model(model_name, engine_config, **kwargs)

        # Load the model
        model.load()

        # Validate
        self._validate_model(model)

        # Cache the loaded model
        self.loaded_models[model_name] = model

        logger.info(f"Model '{model_name}' loaded successfully")
        return model

    def _detect_model_type(self, model_name: str) -> ModelType:
        """
        Detect the type of model based on its name.

        Args:
            model_name: Model identifier

        Returns:
            ModelType enum value
        """
        model_lower = model_name.lower()

        # Check API models
        for prefix in self.OPENAI_MODELS:
            if model_lower.startswith(prefix):
                return ModelType.OPENAI

        for prefix in self.ANTHROPIC_MODELS:
            if model_lower.startswith(prefix):
                return ModelType.ANTHROPIC

        for prefix in self.GEMINI_MODELS:
            if model_lower.startswith(prefix):
                return ModelType.GEMINI

        # Check for MLX-community models (pre-converted for Apple Silicon)
        if "mlx-community/" in model_lower:
            logger.info(f"Detected MLX-community model: {model_name}")
            return ModelType.MLX

        # GGUF files are handled by a separate engine.
        if model_lower.endswith(".gguf"):
            logger.info(f"Detected GGUF model file: {model_name}")
            return ModelType.TRANSFORMERS  # routed to GGUFEngine in load path

        # Check if it's a local path
        if os.path.exists(model_name):
            logger.info(f"Detected local model path: {model_name}")
            return ModelType.TRANSFORMERS  # Default to Transformers for local models

        # Assume HuggingFace model ID
        return ModelType.TRANSFORMERS  # Default to Transformers for HuggingFace models

    def _load_openai_model(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        **kwargs
    ) -> OpenAIAdapter:
        """
        Load OpenAI model.

        Args:
            model_name: OpenAI model identifier
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            OpenAIAdapter instance
        """
        from effgen.models.openai_adapter import OpenAIAdapter

        logger.info(f"Loading OpenAI model: {model_name}")

        params = config or {}
        params.update(kwargs)
        # Drop kwargs only meaningful for local/HF engines
        for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
            params.pop(k, None)

        return OpenAIAdapter(model_name=model_name, **params)

    def _load_anthropic_model(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        **kwargs
    ) -> AnthropicAdapter:
        """
        Load Anthropic model.

        Args:
            model_name: Anthropic model identifier
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            AnthropicAdapter instance
        """
        from effgen.models.anthropic_adapter import AnthropicAdapter

        logger.info(f"Loading Anthropic model: {model_name}")

        params = config or {}
        params.update(kwargs)
        for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
            params.pop(k, None)

        return AnthropicAdapter(model_name=model_name, **params)

    def _load_gemini_model(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        **kwargs
    ) -> GeminiAdapter:
        """
        Load Gemini model.

        Args:
            model_name: Gemini model identifier
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            GeminiAdapter instance
        """
        from effgen.models.gemini_adapter import GeminiAdapter

        logger.info(f"Loading Gemini model: {model_name}")

        params = config or {}
        params.update(kwargs)
        for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
            params.pop(k, None)

        return GeminiAdapter(model_name=model_name, **params)

    def _load_huggingface_model(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        **kwargs
    ) -> "VLLMEngine | TransformersEngine | BaseModel":
        """
        Load HuggingFace model with intelligent engine selection.

        Engine selection priority:
        1. Explicitly requested engine (force_engine parameter)
        2. Auto-detect: MLX on Apple Silicon (when no CUDA), else Transformers

        Args:
            model_name: HuggingFace model ID or local path
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            Model engine instance (VLLMEngine, TransformersEngine, MLXEngine, or MLXVLMEngine)
        """
        params = config or {}
        params.update(kwargs)

        # Check if MLX engine is explicitly requested
        if self.force_engine == "mlx":
            logger.info("Using MLX engine (explicitly requested)")
            try:
                return self._load_with_mlx(model_name, params)
            except Exception as e:
                logger.warning(f"MLX loading failed: {e}")
                logger.info("Falling back to Transformers...")
                return self._load_with_transformers(model_name, params)

        if self.force_engine == "mlx_vlm":
            logger.info("Using MLX-VLM engine (explicitly requested)")
            try:
                return self._load_with_mlx_vlm(model_name, params)
            except Exception as e:
                logger.warning(f"MLX-VLM loading failed: {e}")
                logger.info("Falling back to Transformers...")
                return self._load_with_transformers(model_name, params)

        # Check if vLLM engine is explicitly requested
        if self.force_engine == "vllm":
            logger.info("Using vLLM engine (explicitly requested)")
            try:
                return self._load_with_vllm(model_name, params)
            except Exception as e:
                logger.warning(f"vLLM loading failed: {e}")
                logger.info("Falling back to Transformers...")
                return self._load_with_transformers(model_name, params)

        # Opt-in "auto-fast": prefer vLLM when it is importable and the GPU is
        # usable, otherwise fall back to Transformers. The default (force_engine
        # None) stays on Transformers — auto-fast must be requested explicitly.
        if self.force_engine == "auto-fast":
            if self._vllm_usable():
                logger.info("Using vLLM engine (auto-fast: vLLM available and GPU usable)")
                try:
                    return self._load_with_vllm(model_name, params)
                except Exception as e:
                    logger.warning(f"vLLM loading failed: {e}")
                    logger.info("Falling back to Transformers...")
                    return self._load_with_transformers(model_name, params)
            logger.info(
                "Using Transformers engine (auto-fast: vLLM unavailable or no usable GPU)"
            )
            return self._load_with_transformers(model_name, params)

        # Auto-detection: prefer MLX on Apple Silicon when no CUDA available
        if self.force_engine is None:
            try:
                import torch

                from effgen.hardware.platform import is_apple_silicon, is_mlx_available
                if is_apple_silicon() and is_mlx_available() and not torch.cuda.is_available():
                    logger.info("Apple Silicon detected with MLX available, using MLX engine")
                    try:
                        return self._load_with_mlx(model_name, params)
                    except Exception as e:
                        logger.warning(f"MLX auto-detection loading failed: {e}")
                        logger.info("Falling back to Transformers...")
            except ImportError:
                pass

        # Default to Transformers (more compatible, easier setup)
        logger.info("Using Transformers engine (default)")
        return self._load_with_transformers(model_name, params)

    @staticmethod
    def _vllm_usable() -> bool:
        """Return True only if vLLM can actually run here: the package imports
        cleanly (no CUDA/torch ABI mismatch) and a CUDA GPU is available. Used by
        the opt-in ``engine="auto-fast"`` path to decide vLLM vs Transformers
        without ever raising.
        """
        import torch

        if not torch.cuda.is_available():
            return False
        try:
            import importlib.util
            if importlib.util.find_spec("vllm") is None:
                return False
            # Probe the EXACT import VLLMEngine.load() performs. `import vllm`
            # alone succeeds even when the compiled extension is ABI-incompatible
            # (e.g. a missing libcudart.so) — only importing LLM/SamplingParams
            # surfaces that. Probing here keeps auto-fast on Transformers instead
            # of constructing an engine whose later load() would hard-fail.
            from vllm import LLM, SamplingParams  # noqa: F401
            return True
        except Exception:
            logger.debug("vLLM present but not usable; auto-fast will use Transformers",
                         exc_info=True)
            return False

    def _load_with_vllm(
        self,
        model_name: str,
        params: dict[str, Any]
    ) -> VLLMEngine:
        """
        Load model with vLLM.

        Args:
            model_name: Model identifier
            params: Configuration parameters

        Returns:
            VLLMEngine instance

        Raises:
            RuntimeError: If vLLM is unavailable or loading fails
        """
        import torch

        from effgen.models.vllm_engine import VLLMEngine

        logger.info(f"Attempting to load with vLLM: {model_name}")

        # Check CUDA availability
        if not torch.cuda.is_available():
            from effgen.gpu.cuda_compat import get_cuda_status
            status = get_cuda_status()
            if status.mismatch and status.message:
                raise RuntimeError(f"vLLM requires a usable GPU. {status.message}")
            raise RuntimeError("CUDA not available, vLLM requires GPU")

        # Determine quantization if not specified
        if "quantization" not in params:
            params["quantization"] = self._auto_select_quantization(model_name)

        # Determine tensor parallel size if not specified
        if "tensor_parallel_size" not in params:
            params["tensor_parallel_size"] = self._auto_select_tensor_parallel(model_name)

        # Only set download directory if explicitly specified (let vLLM use its default otherwise)
        # This avoids potential issues with path handling
        if "download_dir" not in params and self.cache_dir != os.path.expanduser("~/.cache/huggingface"):
            params["download_dir"] = self.cache_dir

        return VLLMEngine(model_name=model_name, **params)

    def _load_with_mlx(
        self,
        model_name: str,
        params: dict[str, Any]
    ) -> "BaseModel":
        """
        Load model with MLX (Apple Silicon).

        Args:
            model_name: Model identifier (mlx-community/ or HuggingFace ID)
            params: Configuration parameters

        Returns:
            MLXEngine instance

        Raises:
            RuntimeError: If MLX is unavailable or loading fails
        """
        from effgen.models.mlx_engine import MLXEngine

        logger.info(f"Attempting to load with MLX: {model_name}")

        # Filter out CUDA-specific params that don't apply to MLX
        mlx_params = {
            k: v for k, v in params.items()
            if k not in (
                "tensor_parallel_size", "gpu_memory_utilization", "quantization",
                "max_num_seqs", "max_num_batched_tokens", "download_dir",
                "device_map", "quantization_bits",
            )
        }

        return MLXEngine(model_name=model_name, **mlx_params)

    def _load_with_mlx_vlm(
        self,
        model_name: str,
        params: dict[str, Any]
    ) -> "BaseModel":
        """
        Load vision-language model with MLX-VLM (Apple Silicon).

        Args:
            model_name: Model identifier
            params: Configuration parameters

        Returns:
            MLXVLMEngine instance

        Raises:
            RuntimeError: If MLX-VLM is unavailable or loading fails
        """
        from effgen.models.mlx_vlm_engine import MLXVLMEngine

        logger.info(f"Attempting to load VLM with MLX-VLM: {model_name}")

        # Filter out CUDA-specific params
        mlx_params = {
            k: v for k, v in params.items()
            if k not in (
                "tensor_parallel_size", "gpu_memory_utilization", "quantization",
                "max_num_seqs", "max_num_batched_tokens", "download_dir",
                "device_map", "quantization_bits",
            )
        }

        return MLXVLMEngine(model_name=model_name, **mlx_params)

    def _load_with_transformers(
        self,
        model_name: str,
        params: dict[str, Any]
    ) -> TransformersEngine:
        """
        Load model with Transformers.

        Args:
            model_name: Model identifier
            params: Configuration parameters

        Returns:
            TransformersEngine instance
        """
        import torch

        from effgen.models.transformers_engine import TransformersEngine

        logger.info(f"Loading with Transformers: {model_name}")

        # Convert shorthand quantization="4bit"/"8bit"/"awq"/"gptq" to engine params.
        if "quantization" in params and "quantization_bits" not in params:
            q = params.pop("quantization")
            if q in ("4bit", "4"):
                params["quantization_bits"] = 4
            elif q in ("8bit", "8"):
                params["quantization_bits"] = 8
            elif q == "awq":
                # AWQ models carry their own quantization config; just verify
                # autoawq is importable so we fail with a friendly message.
                try:
                    import awq  # type: ignore  # noqa: F401
                except ImportError:
                    logger.warning(
                        "quantization='awq' requested but 'autoawq' is not installed. "
                        "Install with: pip install autoawq"
                    )
                params["quantization_method"] = "awq"
            elif q == "gptq":
                try:
                    import auto_gptq  # type: ignore  # noqa: F401
                except ImportError:
                    logger.warning(
                        "quantization='gptq' requested but 'auto-gptq' is not installed. "
                        "Install with: pip install auto-gptq"
                    )
                params["quantization_method"] = "gptq"
            elif q is not None:
                logger.warning(
                    "Unknown quantization value '%s' for Transformers engine, ignoring.", q
                )

        # Determine quantization if not specified
        if "quantization_bits" not in params:
            params["quantization_bits"] = self._auto_select_quantization_bits()

        # Set device map
        if "device_map" not in params:
            params["device_map"] = "auto" if torch.cuda.is_available() else None

        # AWQ/GPTQ quantization is encoded in the model checkpoint config; the
        # transformers engine just needs to load it as-is. Drop our internal
        # marker so it isn't forwarded to from_pretrained.
        params.pop("quantization_method", None)

        return TransformersEngine(model_name=model_name, **params)

    @staticmethod
    def _free_vram_gb() -> float:
        """Free VRAM (GB) across the visible CUDA devices.

        Uses the currently-free memory (``torch.cuda.mem_get_info``), not the
        card's total capacity, so quantization/offload decisions reflect what is
        actually available after any other tenants on the GPU.
        """
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free_bytes = 0
        for index in range(torch.cuda.device_count()):
            try:
                free_bytes += torch.cuda.mem_get_info(index)[0]
            except Exception:
                pass
        return free_bytes / (1024**3)

    def _auto_select_quantization(self, model_name: str) -> str | None:
        """
        Automatically select quantization based on free VRAM.

        Args:
            model_name: Model identifier

        Returns:
            Quantization method or None
        """
        import torch

        if not torch.cuda.is_available():
            return None

        # Base the decision on free VRAM, not total capacity, so a busy GPU
        # doesn't silently overcommit and offload to CPU.
        available_vram = self._free_vram_gb()
        logger.info(f"Free VRAM: {available_vram:.2f} GB")

        # Estimate model size (rough heuristics from the id).
        if "70b" in model_name.lower() or "65b" in model_name.lower():
            # Large models need quantization
            if available_vram < 80:
                logger.info("Using AWQ quantization for large model")
                return "awq"
        elif "13b" in model_name.lower() or "7b" in model_name.lower():
            # Medium models might benefit from quantization
            if available_vram < 24:
                logger.info("Using AWQ quantization for medium model")
                return "awq"

        # No quantization needed
        logger.info("Sufficient free VRAM, no quantization needed")
        return None

    def _auto_select_quantization_bits(self) -> int | None:
        """
        Automatically select quantization bits for Transformers.

        Returns:
            Quantization bits (4, 8) or None
        """
        import torch

        if not torch.cuda.is_available():
            return None

        # Base the decision on free VRAM, not total capacity.
        available_vram = self._free_vram_gb()

        if available_vram < 16:
            logger.info("Low free VRAM detected, using 4-bit quantization")
            return 4
        elif available_vram < 32:
            logger.info("Medium free VRAM detected, using 8-bit quantization")
            return 8

        logger.info("Sufficient free VRAM, no quantization")
        return None

    def _auto_select_tensor_parallel(self, model_name: str) -> int:
        """
        Automatically select tensor parallel size based on available GPUs and model size.

        For tensor parallelism to work, the number of attention heads must be divisible
        by the tensor parallel size. Small models often have fewer attention heads,
        so we need to be conservative.

        Args:
            model_name: Model identifier to help determine appropriate parallelism

        Returns:
            Number of GPUs to use for tensor parallelism
        """
        import torch

        if not torch.cuda.is_available():
            return 1

        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s)")

        # For small models (indicated by size in name), use fewer GPUs
        # Small models have fewer attention heads which limits parallelism options
        model_lower = model_name.lower()

        # Check for small model indicators
        # Note: We check for specific sizes like "1.7b", "3b-" etc. to handle various naming conventions
        small_model_indicators = [
            "0.5b", "1b", "1.5b", "1.7b", "2b", "3b", "4b",
            "-0.5b", "-1b", "-1.5b", "-1.7b", "-2b", "-3b", "-4b",
            "_0.5b", "_1b", "_1.5b", "_1.7b", "_2b", "_3b", "_4b",
        ]
        if any(size in model_lower for size in small_model_indicators):
            # Small models: use at most 1 GPU (attention heads typically 12-16)
            # 12 heads divisible by: 1, 2, 3, 4, 6, 12
            # 16 heads divisible by: 1, 2, 4, 8, 16
            tp_size = 1  # Conservative: use 1 GPU for small models
            logger.info(f"Small model detected, using tensor_parallel_size={tp_size}")
            return tp_size
        elif any(size in model_lower for size in ["7b", "8b"]):
            # Medium models: typically 32 heads, can use up to 4 GPUs
            # 32 heads divisible by: 1, 2, 4, 8, 16, 32
            tp_size = min(num_gpus, 4)
            logger.info(f"Medium model detected, using tensor_parallel_size={tp_size}")
            return tp_size
        elif any(size in model_lower for size in ["13b", "14b"]):
            # Larger models: typically 40 heads
            # 40 heads divisible by: 1, 2, 4, 5, 8, 10, 20, 40
            tp_size = min(num_gpus, 4)
            logger.info(f"13B+ model detected, using tensor_parallel_size={tp_size}")
            return tp_size
        elif any(size in model_lower for size in ["30b", "33b", "34b", "65b", "70b"]):
            # Large models: can benefit from more parallelism
            # 64/80 heads divisible by: 1, 2, 4, 8, 16, etc.
            tp_size = min(num_gpus, 8)
            logger.info(f"Large model detected, using tensor_parallel_size={tp_size}")
            return tp_size

        # Default: conservative approach, use 1 GPU unless we know the model
        logger.info("Unknown model size, using tensor_parallel_size=1 for safety")
        return 1

    def _validate_model(self, model: BaseModel) -> None:
        """
        Validate that the model is properly loaded and functional.

        Args:
            model: Model instance to validate

        Raises:
            RuntimeError: If validation fails
        """
        logger.info("Validating model...")

        if not model.is_loaded():
            raise RuntimeError("Model validation failed: model not loaded")

        # Test token counting
        try:
            test_text = "Hello, world!"
            token_count = model.count_tokens(test_text)
            logger.info(f"Token counting works: '{test_text}' = {token_count.count} tokens")
        except Exception as e:
            logger.warning(f"Token counting validation failed: {e}")

        # Test context length
        try:
            context_length = model.get_context_length()
            logger.info(f"Context length: {context_length} tokens")
        except Exception as e:
            logger.warning(f"Context length validation failed: {e}")

        logger.info("Model validation passed")

    def unload_model(self, model_name: str) -> None:
        """
        Unload a specific model from memory.

        Args:
            model_name: Name of model to unload
        """
        if model_name in self.loaded_models:
            logger.info(f"Unloading model: {model_name}")
            model = self.loaded_models[model_name]
            model.unload()
            del self.loaded_models[model_name]
            logger.info(f"Model '{model_name}' unloaded")
        else:
            logger.warning(f"Model '{model_name}' not found in loaded models")

    def unload_all(self) -> None:
        """
        Unload all loaded models.
        """
        logger.info("Unloading all models...")
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        logger.info("All models unloaded")

    def get_loaded_models(self) -> dict[str, BaseModel]:
        """
        Get dictionary of all loaded models.

        Returns:
            Dict mapping model names to model instances
        """
        return self.loaded_models.copy()

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """
        Get information about a loaded model.

        Args:
            model_name: Name of the model

        Returns:
            Model metadata dict or None if not loaded
        """
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            return model.get_metadata()
        return None


# Convenience function for quick model loading
def load_model(
    model_name: str,
    engine: str | None = None,
    engine_config: dict[str, Any] | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    apply_chat_template: bool = True,
    provider: str | None = None,
    **kwargs
) -> BaseModel:
    """
    Convenience function to quickly load a model.

    Provider prefixes route to a remote API (``"openai:..."``, ``"gemini:..."``,
    ``"hf:..."`` etc.). In particular ``"hf:<repo>"`` is the **remote**
    HuggingFace Inference API — to run the same repo **locally** on your GPU,
    pass ``engine="transformers"`` (or ``"vllm"``) with a bare model id instead
    of the ``hf:`` prefix.

    Args:
        model_name: Model identifier
        engine: Engine to use ('vllm', 'transformers', 'auto-fast', or None for auto).
                'auto-fast' prefers vLLM when available and the GPU is usable, else
                Transformers. None defaults to Transformers.
        engine_config: Optional engine configuration
        tensor_parallel_size: Number of GPUs for tensor parallelism (vLLM only).
                             If not specified, auto-detected based on model size.
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0, vLLM only).
                               Default is 0.90. Lower this if you get CUDA OOM errors.
        apply_chat_template: Whether to automatically apply chat templates for
                            instruction-tuned models (default: True, vLLM only).
                            This ensures proper formatting for models like Qwen-Instruct.
        **kwargs: Additional parameters (e.g., quantization="4bit", trust_remote_code=True)

    Returns:
        Loaded model instance

    Example:
        >>> from effgen.models import load_model
        >>> # Default uses Transformers engine
        >>> model = load_model("meta-llama/Llama-2-7b-hf")
        >>> result = model.generate("Hello, how are you?")

        >>> # Explicitly use vLLM for production (5-10x faster)
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm")

        >>> # With tensor parallelism for large models
        >>> model = load_model("meta-llama/Llama-2-70b-hf", engine="vllm", tensor_parallel_size=4)

        >>> # Lower GPU memory usage if getting OOM errors
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm", gpu_memory_utilization=0.7)

        >>> # Disable chat template for raw text generation
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm", apply_chat_template=False)

        >>> # Fail instead of falling back to CPU when the GPU can't hold the model
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="transformers", require_gpu=True)
    """
    # Pass tensor_parallel_size to kwargs if specified
    if tensor_parallel_size is not None:
        kwargs["tensor_parallel_size"] = tensor_parallel_size

    # Pass gpu_memory_utilization to kwargs if specified
    if gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    # Pass apply_chat_template for vLLM
    kwargs["apply_chat_template"] = apply_chat_template

    if provider is not None:
        kwargs["provider"] = provider

    loader = ModelLoader(force_engine=engine)
    return loader.load_model(model_name, engine_config, **kwargs)
