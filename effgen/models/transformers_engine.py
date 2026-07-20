"""
HuggingFace Transformers engine implementation as fallback.

This module provides a fallback inference engine using HuggingFace Transformers
with features including:
- Automatic quantization with bitsandbytes
- Flash Attention support
- Multi-GPU device mapping
- Memory optimization techniques
- CPU fallback support
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers import (
    GenerationConfig as HFGenerationConfig,
)
from transformers import (
    set_seed as _hf_set_seed,
)

from effgen.models._adapter_utils import normalize_finish_reason
from effgen.models.base import (
    BatchModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='accelerate')
warnings.filterwarnings('ignore', message='.*Some parameters are on the meta device.*')

logger = logging.getLogger(__name__)


class GPUPlacementError(RuntimeError):
    """Raised when ``require_gpu`` is set but the model can't fit on the GPU."""


class ModelNotCachedError(RuntimeError):
    """Raised when a model isn't in the local cache and offline mode is set."""


def _offline_mode_active() -> bool:
    """True if HuggingFace offline mode is set via the environment."""
    return any(
        os.environ.get(var, "").strip().lower() in ("1", "true", "yes", "on")
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _list_cached_model_repos(limit: int = 20) -> list[str]:
    """Return locally-cached HuggingFace model repo ids (best effort, sorted)."""
    try:
        from huggingface_hub import scan_cache_dir

        repos = sorted(
            r.repo_id for r in scan_cache_dir().repos if r.repo_type == "model"
        )
        return repos[:limit]
    except Exception:
        return []


def _is_cache_miss_error(exc: Exception) -> bool:
    """True if *exc* is a HuggingFace "not found locally" / offline error."""
    text = str(exc).lower()
    return (
        "couldn't find them in the cached files" in text
        or ("can't load" in text and "offline" in text)
        or "offlinemodeisenabled" in text
        or "localentrynotfound" in text
        or type(exc).__name__ in ("LocalEntryNotFoundError", "OfflineModeIsEnabled")
    )


def _reraise_if_classified(exc: Exception) -> None:
    """Re-raise *exc* unwrapped when it already carries retry classification.

    A timeout raised by ``effgen.reliability.timeouts.with_timeout()`` around
    a local generate call must propagate as-is instead of being flattened
    into a generic ``RuntimeError`` by the callers' blanket exception
    handlers below — flattening discards the type information
    ``is_transient_error()`` relies on to retry it correctly.
    """
    from effgen.reliability.timeouts import TimeoutError as EffGenTimeoutError

    if isinstance(exc, EffGenTimeoutError):
        raise exc


# Native tool-call delimiters that the downstream parser (core.tool_calling)
# needs to see; these must survive the special-token strip on the tool path.
_TOOL_CALL_DELIMITERS = frozenset({
    "<tool_call>", "</tool_call>", "<|python_tag|>", "[TOOL_CALLS]", "<function=",
})
# Markers that render as literal text on some tokenizers even though they are
# not always registered as special tokens (belt-and-suspenders).
_COMMON_END_MARKERS = frozenset({
    "<|im_end|>", "</s>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>",
})


def _strip_special_tokens_keep_tool_calls(text: str, tokenizer) -> str:
    """Remove chat-template special tokens from decoded tool-path text.

    On the native-tool-calling path the model output is decoded with
    ``skip_special_tokens=False`` so that tool-call delimiters survive for the
    parser. That also lets chat-template turn/end markers through, which must
    not reach the user-visible answer. The strip set is derived from the
    tokenizer's own ``all_special_tokens`` (minus the tool-call delimiters) so
    it stays correct across model families — e.g. Gemma's ``<end_of_turn>`` /
    ``<eos>`` / ``<start_of_turn>`` and Qwen's ``<|im_start|>``, which a fixed
    list silently let leak through.
    """
    strip = set(getattr(tokenizer, "all_special_tokens", ()) or ())
    strip -= _TOOL_CALL_DELIMITERS
    strip |= _COMMON_END_MARKERS
    # Strip longest-first so a marker that contains another is removed whole.
    for marker in sorted(strip, key=len, reverse=True):
        if marker:
            text = text.replace(marker, "")
    return text.strip()


@contextmanager
def _quiet_model_load():
    """Silence Transformers' load-time chatter (weight-loading progress bars and
    INFO reports) for the duration of a model load, restoring the previous state
    afterwards. Loading diagnostics belong in logs at debug level, not on stdout
    for every inference call.
    """
    try:
        from transformers.utils import logging as hf_logging
    except Exception:  # pragma: no cover - transformers always present here
        yield
        return

    prev_verbosity = hf_logging.get_verbosity()
    try:
        prev_progress = hf_logging.is_progress_bar_enabled()
    except Exception:
        prev_progress = True
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.set_verbosity(prev_verbosity)
        if prev_progress:
            hf_logging.enable_progress_bar()


class TransformersEngine(BatchModel):
    """
    HuggingFace Transformers-based model engine.

    This engine serves as a fallback when vLLM is unavailable or incompatible.
    It supports a wider range of models and edge cases.

    Features:
    - Automatic quantization (4-bit, 8-bit with bitsandbytes)
    - Flash Attention 2 support
    - Auto device mapping for multi-GPU
    - Memory optimization (gradient checkpointing, mixed precision)
    - CPU fallback
    - Streaming generation

    Attributes:
        model_name: HuggingFace model identifier or path
        quantization_bits: Quantization level (None, 4, 8)
        device_map: Device mapping strategy ('auto', 'balanced', or custom)
        use_flash_attention: Whether to use Flash Attention 2
        torch_dtype: Torch data type for model weights
    """

    def __init__(
        self,
        model_name: str,
        quantization_bits: int | None = None,
        device_map: str = "auto",
        use_flash_attention: bool = True,
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = True,
        max_memory: dict[int, str] | None = None,
        offload_folder: str | None = None,
        require_gpu: bool = False,
        **kwargs
    ):
        """
        Initialize Transformers engine.

        Args:
            model_name: HuggingFace model ID or local path
            quantization_bits: Quantization level (None, 4, or 8)
            device_map: Device mapping ('auto', 'balanced', 'sequential', or dict)
            use_flash_attention: Enable Flash Attention 2 if available
            torch_dtype: Data type (None for auto, or torch.float16, torch.bfloat16)
            trust_remote_code: Whether to trust remote code
            low_cpu_mem_usage: Use low CPU memory during loading
            max_memory: Maximum memory per device (e.g., {0: "20GB", "cpu": "30GB"})
            offload_folder: Folder for offloading weights
            require_gpu: Fail with a clear error if the model cannot be placed
                entirely on the GPU (rather than falling back to CPU offload).
                Use this on hardware where a silent CPU fallback is unacceptable.
            **kwargs: Additional model loading arguments
        """
        super().__init__(
            model_name=model_name,
            model_type=ModelType.TRANSFORMERS
        )

        self.quantization_bits = quantization_bits
        self.device_map = device_map
        self.use_flash_attention = use_flash_attention
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.max_memory = max_memory
        self.offload_folder = offload_folder
        self.require_gpu = require_gpu

        # Filter out parameters that shouldn't be passed to model loading
        # These are vLLM-specific or other incompatible parameters
        self.additional_kwargs = {k: v for k, v in kwargs.items()
                                  if k not in ['quantization', 'engine', 'backend', 'device',
                                               'require_gpu',
                                               'use_tqdm', 'tensor_parallel_size',
                                               'apply_chat_template', 'system_prompt',
                                               'gpu_memory_utilization', 'max_num_seqs',
                                               'max_num_batched_tokens']}

        self.model = None
        self.tokenizer = None
        self.device = None
        # HuggingFace "fast" (Rust) tokenizers are NOT thread-safe: two threads
        # encoding/decoding on the same tokenizer raise "Already borrowed". A
        # single local model can't parallelize across threads on one GPU anyway,
        # so serialize every tokenizer-touching call on this engine. Reentrant so
        # a method that internally counts tokens while holding the lock is safe.
        self._tokenizer_lock = threading.RLock()

    def load(self) -> None:
        """
        Load the model using HuggingFace Transformers.

        Raises:
            RuntimeError: If model loading fails
            ValueError: If configuration is invalid
        """
        try:
            logger.debug(f"Loading model '{self.model_name}' with Transformers...")

            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.debug(f"Using CUDA with {torch.cuda.device_count()} GPU(s)")
            else:
                self.device = "cpu"
                # If the host actually has NVIDIA GPUs but torch can't use them
                # (almost always a torch-CUDA vs driver mismatch), emit one clear,
                # actionable warning instead of a bland "CUDA not available".
                from effgen.gpu.cuda_compat import warn_cuda_mismatch_once
                if not warn_cuda_mismatch_once():
                    logger.warning("CUDA not available, using CPU (this will be slow)")

            # Setup quantization config if specified
            quantization_config = None
            if self.quantization_bits is not None:
                quantization_config = self._create_quantization_config()

            # Determine torch dtype
            if self.torch_dtype is None:
                if self.device == "cuda":
                    # Use bfloat16 if available, else float16
                    if torch.cuda.is_bf16_supported():
                        self.torch_dtype = torch.bfloat16
                    else:
                        self.torch_dtype = torch.float16
                else:
                    self.torch_dtype = torch.float32

            logger.debug(
                f"Configuration: quantization={self.quantization_bits}-bit, "
                f"dtype={self.torch_dtype}, flash_attention={self.use_flash_attention}"
            )

            # Load tokenizer (quiet: suppress Transformers progress bars / INFO
            # reports during load — these are diagnostics, not user output)
            with _quiet_model_load():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code
                )

            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Build model loading arguments
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_name,
                "trust_remote_code": self.trust_remote_code,
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
            }

            # Add quantization config if using quantization
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                # Don't set dtype when quantizing
            else:
                # transformers v5+ uses 'dtype', v4.x uses 'torch_dtype'
                import transformers
                if hasattr(transformers, 'VERSION') or int(transformers.__version__.split('.')[0]) >= 5:
                    model_kwargs["dtype"] = self.torch_dtype
                else:
                    model_kwargs["torch_dtype"] = self.torch_dtype

            # Add device map for multi-GPU or CPU offloading
            if self.device == "cuda":
                model_kwargs["device_map"] = self.device_map

                if self.max_memory:
                    model_kwargs["max_memory"] = self.max_memory

                if self.offload_folder:
                    model_kwargs["offload_folder"] = self.offload_folder

            # Add Flash Attention 2 if requested
            if self.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            # Add additional kwargs
            model_kwargs.update(self.additional_kwargs)

            # Load model (quiet: suppress weight-loading progress bars / INFO)
            try:
                # Suppress Flash Attention warnings from transformers
                import warnings
                with warnings.catch_warnings(), _quiet_model_load():
                    warnings.filterwarnings('ignore', message='.*FlashAttention.*')
                    warnings.filterwarnings('ignore', message='.*flash_attn.*')
                    self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            except Exception as e:
                # Fallback without Flash Attention if it fails
                if self.use_flash_attention and "flash" in str(e).lower():
                    logger.debug("Flash Attention not available, using standard attention")
                    model_kwargs.pop("attn_implementation", None)
                    import warnings
                    with warnings.catch_warnings(), _quiet_model_load():
                        warnings.filterwarnings('ignore', message='.*FlashAttention.*')
                        warnings.filterwarnings('ignore', message='.*flash_attn.*')
                        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                else:
                    raise

            # Move to device if not using device_map
            if "device_map" not in model_kwargs and self.device != "cpu":
                self.model = self.model.to(self.device)

            # Set model to eval mode
            self.model.eval()

            # Reconcile the requested device with where the parameters actually
            # landed. With device_map="auto", accelerate offloads layers to CPU
            # (or disk) when the GPU can't hold the model, so an intended "cuda"
            # load can end up partly or wholly on CPU. Report the real placement
            # so callers aren't told "cuda" while inference runs on CPU.
            intended_device = self.device
            self.device = self._resolve_placement()
            if intended_device == "cuda" and self.device != "cuda":
                free_gb = self._free_vram_gb()
                where = "CPU" if self.device == "cpu" else "CPU/disk (mixed placement)"
                if self.require_gpu:
                    raise GPUPlacementError(
                        f"Model '{self.model_name}' could not be placed entirely on the GPU "
                        f"(only {free_gb:.2f} GB free) and require_gpu is set. Free GPU memory, "
                        f"choose a smaller model, or enable quantization (e.g. quantization='4bit')."
                    )
                logger.warning(
                    "Model '%s' did not fit in available GPU memory (%.2f GB free); "
                    "running on %s. Inference will be slower. Free GPU memory or pass "
                    "quantization='4bit' to keep it on the GPU.",
                    self.model_name, free_gb, where,
                )

            # Store metadata
            self._context_length = self._get_max_length()
            self._metadata = {
                "model_name": self.model_name,
                "quantization": f"{self.quantization_bits}-bit" if self.quantization_bits else None,
                "dtype": str(self.torch_dtype),
                "device": str(self.device),
                "flash_attention": self.use_flash_attention,
                "max_length": self._context_length,
                "num_parameters": self.model.num_parameters(),
            }

            self._is_loaded = True
            logger.debug(f"Model '{self.model_name}' loaded successfully with Transformers")

        except GPUPlacementError:
            # An explicit require_gpu policy failure — surface it unchanged
            # rather than wrapping it as a generic load failure.
            raise
        except Exception as e:
            # In offline mode a missing/misspelled repo surfaces as a
            # connectivity error ("couldn't connect to huggingface.co"). Report
            # it as a local cache miss and name what is cached instead.
            offline = _offline_mode_active()
            if offline or _is_cache_miss_error(e):
                cached = _list_cached_model_repos()
                listed = ", ".join(cached) if cached else "none"
                reason = (
                    "offline mode is set"
                    if offline
                    else "it could not be downloaded (no network)"
                )
                raise ModelNotCachedError(
                    f"Model '{self.model_name}' is not in the local HuggingFace cache "
                    f"and {reason}. Cached models: {listed}."
                ) from e
            logger.error(f"Failed to load model with Transformers: {e}")
            raise RuntimeError(f"Transformers model loading failed: {e}") from e

    def _resolve_placement(self) -> str:
        """Return where the model parameters actually reside after loading.

        Returns 'cuda' if every parameter is on a GPU, 'cpu' if every parameter
        is on CPU (or disk), or 'mixed' if the model is split across GPU and
        CPU/disk. accelerate records the per-module placement in
        ``model.hf_device_map`` when ``device_map`` dispatch is used; otherwise
        the placement is read from the parameters directly.
        """
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            on_gpu = on_host = False
            for dev in device_map.values():
                if isinstance(dev, int):
                    on_gpu = True
                    continue
                text = str(dev).lower()
                if text.startswith("cuda") or text.isdigit():
                    on_gpu = True
                else:  # "cpu", "disk", "meta"
                    on_host = True
            if on_gpu and on_host:
                return "mixed"
            return "cuda" if on_gpu else "cpu"
        try:
            param = next(self.model.parameters())
        except StopIteration:
            return str(self.device)
        return "cuda" if param.is_cuda else "cpu"

    @staticmethod
    def _free_vram_gb() -> float:
        """Total free VRAM (GB) across the visible CUDA devices, or 0.0 if none."""
        if not torch.cuda.is_available():
            return 0.0
        free_bytes = 0
        for index in range(torch.cuda.device_count()):
            try:
                free_bytes += torch.cuda.mem_get_info(index)[0]
            except Exception:
                pass
        return free_bytes / (1024 ** 3)

    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create quantization configuration.

        Returns:
            BitsAndBytesConfig for bitsandbytes quantization

        Raises:
            ValueError: If quantization_bits is invalid
        """
        if self.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(
                f"Invalid quantization_bits: {self.quantization_bits}. "
                "Must be 4 or 8."
            )

    def _get_max_length(self) -> int:
        """
        Get maximum context length from model config.

        Returns:
            int: Maximum sequence length
        """
        config = self.model.config

        # Try different config attributes
        for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
            if hasattr(config, attr):
                return getattr(config, attr)

        # Some models (e.g. gemma-3) nest config inside text_config
        if hasattr(config, "text_config"):
            text_config = config.text_config
            for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
                if hasattr(text_config, attr):
                    return getattr(text_config, attr)

        logger.warning("Could not determine max length from config, using 2048")
        return 2048

    def _eos_token_ids(self) -> int | list[int] | None:
        """Return every token id that should end generation for this model.

        A model's own ``generation_config`` may declare several terminators
        while ``tokenizer.eos_token_id`` holds only one. Llama 3.x is the case
        that matters: the tokenizer reports ``<|eot_id|>`` (end of turn), but a
        tool call ends with ``<|eom_id|>`` (end of message). Passing only the
        tokenizer's id leaves ``<|eom_id|>`` a normal token, so after emitting a
        tool call the model keeps going and writes the assistant turn that
        should have followed the tool's result — inventing an observation it
        never received. Merging both sources stops generation where the model
        intends to stop, and leaves single-terminator models unchanged.
        """
        ids: list[int] = []

        def _add(value: Any) -> None:
            candidates = value if isinstance(value, list | tuple) else [value]
            for candidate in candidates:
                # bool is an int subclass and is never a token id.
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    if candidate not in ids:
                        ids.append(candidate)

        _add(getattr(getattr(self.model, "generation_config", None), "eos_token_id", None))
        _add(getattr(self.tokenizer, "eos_token_id", None))

        if not ids:
            return None
        return ids[0] if len(ids) == 1 else ids

    def _create_generation_config(
        self,
        config: GenerationConfig | None = None
    ) -> tuple[HFGenerationConfig, list[str]]:
        """
        Create HuggingFace GenerationConfig from our GenerationConfig.

        Args:
            config: Our generation configuration

        Returns:
            Tuple of (HuggingFace GenerationConfig object, stop_sequences list)

        Notes:
            HuggingFace Transformers doesn't support stop sequences natively like OpenAI,
            so we return them separately for post-generation processing.
        """
        if config is None:
            config = GenerationConfig()

        eos_token_id = self._eos_token_ids()

        # Normalize deterministic generation. Transformers 5.x rejects
        # temperature<=0 ("has to be a strictly positive float") and warns when
        # sampling params (temperature/top_p/top_k) are set while do_sample is
        # False. Treat temperature<=0 as greedy decoding: set do_sample=False and
        # omit the sampling params entirely so the same effGen config works
        # identically across Transformers versions and other backends.
        do_sample = config.temperature is not None and config.temperature > 0
        if do_sample:
            hf_config = HFGenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_new_tokens=config.max_tokens or 512,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
        else:
            # Greedy decoding. Set the sampling params to their no-op defaults
            # (temperature=1.0, top_p=1.0, top_k=50) explicitly: leaving them unset
            # lets Transformers merge the model's own generation_config.json
            # sampling values (e.g. Qwen's temperature=0.7) into the config, which
            # then triggers a "generation flags not valid for do_sample=False"
            # warning on every greedy call. Explicit no-op values suppress it
            # without affecting greedy output.
            hf_config = HFGenerationConfig(
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                max_new_tokens=config.max_tokens or 512,
                repetition_penalty=config.repetition_penalty,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        # Return stop sequences separately for post-processing
        # NOTE: We DON'T set them as eos_token_id because that would stop generation
        # at the first token match, not the full sequence match
        return hf_config, config.stop_sequences if config.stop_sequences else []

    # HuggingFace GenerationConfig fields effGen forwards from per-call kwargs.
    _HF_GEN_PARAMS = (
        "temperature", "top_p", "top_k", "repetition_penalty",
        "num_beams", "do_sample", "pad_token_id", "eos_token_id",
        "max_new_tokens",
    )

    def _sanitize_generation_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate per-call OpenAI-style generation kwargs to HuggingFace names.

        ``max_tokens`` becomes ``max_new_tokens``; recognised HuggingFace
        generation params pass through; a non-positive ``temperature`` collapses
        to greedy decoding; a positive ``temperature`` enables sampling. Unknown
        keys are logged and skipped so a stray kwarg never crashes generation.

        These values are folded INTO an existing ``GenerationConfig`` (see
        :meth:`_fold_into_generation_config`), so a per-call override must fully
        supersede the config's sampling fields — not merely add ``do_sample`` —
        or a config that defaulted to sampling (``temperature=0.7``) would keep
        ``temperature``/``top_p`` set alongside ``do_sample=False`` and trip the
        Transformers "generation flags are not valid" warning.
        """
        sanitized: dict[str, Any] = {}
        for key, value in kwargs.items():
            try:
                if key == "max_tokens":
                    sanitized["max_new_tokens"] = value
                    logger.debug("Converted max_tokens=%s to max_new_tokens", value)
                elif key in self._HF_GEN_PARAMS:
                    sanitized[key] = value
                else:
                    logger.warning("Skipping unknown generation parameter: %s=%s", key, value)
            except Exception as e:
                logger.error("Error processing generation parameter %s: %s", key, e)
                continue

        if "temperature" in sanitized:
            temp = sanitized["temperature"]
            if temp is None or temp <= 0:
                # Greedy decoding. Overwrite the sampling fields with their no-op
                # defaults (not just do_sample=False) so they override whatever
                # the base config carried — mirroring the greedy branch of
                # _create_generation_config and keeping the call warning-free.
                sanitized["temperature"] = 1.0
                sanitized["top_p"] = 1.0
                sanitized["top_k"] = 50
                sanitized["do_sample"] = False
            else:
                # A positive per-call temperature is an explicit request to
                # sample; enable it so the override isn't silently ignored when
                # the base config was greedy.
                sanitized.setdefault("do_sample", True)
        return sanitized

    def _fold_into_generation_config(
        self, generation_config: HFGenerationConfig, params: dict[str, Any]
    ) -> tuple[HFGenerationConfig, dict[str, Any]]:
        """Merge generation *params* INTO *generation_config*; return leftover kwargs.

        Passing a ``generation_config`` together with generation-related keyword
        arguments is deprecated in Transformers 5.x and prints a warning on every
        call. Folding the recognised parameters into the config object — and
        forwarding only genuinely non-generation kwargs (e.g. ``streamer``,
        ``stopping_criteria``) separately — keeps each call quiet with identical
        decoding behaviour (per-call values still override the config). Returns
        the mutated config and the kwargs it did not consume.
        """
        if not params:
            return generation_config, {}
        unused = generation_config.update(**params)
        return generation_config, unused or {}

    def _apply_chat_template(self, prompt: str, tools_for_template: Any = None) -> str:
        """Wrap *prompt* with the tokenizer's chat template when one exists.

        Instruct/chat models (Qwen, Llama-3, …) expect their role-tagged format;
        feeding the raw text makes them ramble and skip the stop token. Both the
        batched ``generate`` and ``generate_stream`` paths use this so streamed
        and non-streamed answers are formatted identically.
        """
        if not (hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools_for_template:
                template_kwargs["tools"] = tools_for_template
                logger.debug(
                    f"Passing {len(tools_for_template)} tool definitions "
                    "to chat template for native function calling"
                )
            formatted = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            logger.debug("Applied chat template to prompt")
            return formatted
        except TypeError as e:
            if tools_for_template:
                logger.debug(
                    f"Chat template does not accept tools param: {e}, "
                    "falling back to plain template"
                )
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            logger.warning(f"Failed to apply chat template: {e}")
            return prompt
        except Exception as e:
            logger.warning(f"Failed to apply chat template, using raw prompt: {e}")
            return prompt

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        self.validate_prompt(prompt)

        generation_config, stop_sequences = self._create_generation_config(config)

        # Serialize fast-tokenizer + generate so concurrent local calls (e.g.
        # batch at concurrency>1) never trip the tokenizer's "Already borrowed".
        self._tokenizer_lock.acquire()
        try:
            # Extract tool definitions before sanitizing kwargs — these are
            # passed to the chat template, not to HF generate()
            tools_for_template = kwargs.pop("tools", None)

            # Sanitize per-call kwargs (OpenAI-style → HuggingFace) and fold them
            # into the GenerationConfig. Passing generation params alongside a
            # generation_config is deprecated in Transformers 5.x, so we merge
            # them in and forward only the leftover non-generation kwargs.
            sanitized_kwargs = self._sanitize_generation_kwargs(kwargs)
            generation_config, extra_kwargs = self._fold_into_generation_config(
                generation_config, sanitized_kwargs
            )

            # Apply chat template if available for better model compatibility
            # Many modern models like Qwen expect a specific format
            formatted_prompt = self._apply_chat_template(prompt, tools_for_template)

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._context_length
            )

            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Seed the sampling RNGs from config.seed so run(seed=...) is
            # reproducible on-device: identical prompt + temperature + seed
            # returns identical text. set_seed() covers torch (CPU + CUDA),
            # numpy and random; it only affects sampling, so greedy decoding
            # (temperature<=0) is unchanged either way.
            _seed = getattr(config, "seed", None)
            if _seed is not None:
                _hf_set_seed(_seed)

            # Generate with sanitized kwargs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    **extra_kwargs
                )

            # Decode output
            # When native tool calling is active, preserve tool-call tokens
            # like <tool_call>, </tool_call>, [TOOL_CALLS] etc. but strip
            # chat-template end markers like <|im_end|>, </s>, <|eot_id|>.
            # clean_up_tokenization_spaces=False: the cleanup step is destructive
            # for BPE/SentencePiece tokenizers (it strips spaces before
            # punctuation) and Transformers warns + ignores it for them anyway.
            # Passing False explicitly preserves spacing and silences the warning.
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            if tools_for_template:
                # Preserve native tool-call delimiters for the parser, then
                # strip every other special token so chat-template turn/end
                # markers never leak into the answer (see helper docstring).
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                generated_text = _strip_special_tokens_keep_tool_calls(
                    generated_text, self.tokenizer,
                )
            else:
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            # Apply stop sequences post-generation
            # This is more reliable than trying to use them during generation
            finish_reason = "stop"
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        # Find first occurrence of any stop sequence
                        stop_index = generated_text.find(stop_seq)
                        if stop_index != -1:
                            generated_text = generated_text[:stop_index]
                            finish_reason = "stop_sequence"
                            logger.debug(f"Stopped generation at stop sequence: '{stop_seq}'")
                            break

            # Calculate tokens
            prompt_tokens = inputs["input_ids"].shape[1]
            completion_tokens = len(generated_ids)

            # HuggingFace `generate()` doesn't report whether decoding stopped
            # at EOS or was cut off at the token budget. Infer the budget case:
            # no stop-sequence match, the last token isn't an EOS id, and the
            # model produced the full requested budget.
            if finish_reason == "stop":
                eos_ids = generation_config.eos_token_id
                if eos_ids is None:
                    eos_ids = ()
                elif isinstance(eos_ids, int):
                    eos_ids = (eos_ids,)
                last_token = generated_ids[-1].item() if completion_tokens else None
                max_new = generation_config.max_new_tokens
                if (
                    max_new is not None
                    and completion_tokens >= max_new
                    and last_token not in eos_ids
                ):
                    finish_reason = "length"

            return GenerationResult(
                text=generated_text,
                tokens_used=completion_tokens,
                finish_reason=normalize_finish_reason(finish_reason),
                model_name=self.model_name,
                metadata={
                    "raw_finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "stop_sequences_applied": stop_sequences if stop_sequences else [],
                    "device": self.device,
                }
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            _reraise_if_classified(e)
            raise RuntimeError(f"Generation failed: {e}") from e
        finally:
            self._tokenizer_lock.release()

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Additional generation parameters

        Yields:
            str: Generated text chunks

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        self.validate_prompt(prompt)

        generation_config, stop_sequences = self._create_generation_config(config)

        try:
            # Apply the chat template (same as generate()) so instruct/chat
            # models receive their expected role-tagged format. Without this,
            # streaming fed the raw prompt and the model rambled / never emitted
            # its stop token.
            tools_for_template = kwargs.pop("tools", None)
            formatted_prompt = self._apply_chat_template(prompt, tools_for_template)

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._context_length
            )

            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Use TextIteratorStreamer for streaming
            from threading import Event, Thread

            from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
                timeout=30.0,  # prevent indefinite block if generation thread dies
                clean_up_tokenization_spaces=False,  # non-destructive for BPE; see generate()
            )

            stop_event = Event()

            class _StopOnEvent(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs) -> bool:
                    return stop_event.is_set()

            stopping_criteria = kwargs.pop("stopping_criteria", None)
            if stopping_criteria is None:
                stopping_criteria = StoppingCriteriaList([_StopOnEvent()])
            elif isinstance(stopping_criteria, StoppingCriteriaList):
                stopping_criteria.append(_StopOnEvent())
            else:
                try:
                    criteria_items = list(stopping_criteria)
                except TypeError:
                    criteria_items = [stopping_criteria]
                stopping_criteria = StoppingCriteriaList([*criteria_items, _StopOnEvent()])

            # Note: stop_sequences not fully supported in streaming mode
            # They would need to be checked in the consumer of the stream

            # Fold any per-call generation kwargs into the config (Transformers
            # 5.x deprecates passing them alongside a generation_config); forward
            # only leftover non-generation kwargs.
            generation_config, extra_kwargs = self._fold_into_generation_config(
                generation_config, self._sanitize_generation_kwargs(kwargs)
            )

            # Generate in a separate thread
            generation_kwargs = {
                **inputs,
                "generation_config": generation_config,
                "stopping_criteria": stopping_criteria,
                "streamer": streamer,
                **extra_kwargs
            }

            gen_exception: list[BaseException] = []

            def _generate_with_error_capture() -> None:
                try:
                    self.model.generate(**generation_kwargs)
                except BaseException as exc:
                    gen_exception.append(exc)
                    # Unblock the streamer queue so yield-from terminates
                    streamer.end()

            thread = Thread(target=_generate_with_error_capture, daemon=False)
            thread.start()

            try:
                # Yield tokens as they're generated
                yield from streamer
            finally:
                stop_event.set()
                try:
                    streamer.end()
                except Exception:
                    logger.debug("Failed to end streamer", exc_info=True)
                thread.join(timeout=30.0)

            if thread.is_alive():
                raise RuntimeError("Streaming generation thread did not exit cleanly")
            if gen_exception:
                raise gen_exception[0]

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            _reraise_if_classified(e)
            raise RuntimeError(f"Streaming generation failed: {e}") from e

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        **kwargs
    ) -> list[GenerationResult]:
        """
        Generate text for multiple prompts in a batch.

        Args:
            prompts: List of input prompts
            config: Generation configuration
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult objects

        Raises:
            RuntimeError: If model is not loaded or generation fails
            ValueError: If any prompt exceeds context length
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Validate all prompts
        for prompt in prompts:
            self.validate_prompt(prompt)

        generation_config, stop_sequences = self._create_generation_config(config)

        # Serialize fast-tokenizer + generate (see generate(): thread-safety).
        self._tokenizer_lock.acquire()
        try:
            # Tokenize all inputs
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._context_length
            )

            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Fold per-call generation kwargs into the config (Transformers 5.x
            # deprecates passing them alongside a generation_config); forward only
            # leftover non-generation kwargs.
            generation_config, extra_kwargs = self._fold_into_generation_config(
                generation_config, self._sanitize_generation_kwargs(kwargs)
            )

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    **extra_kwargs
                )

            # Decode outputs
            results = []
            for i, output in enumerate(outputs):
                # Get only the generated part (exclude input)
                prompt_length = inputs["input_ids"][i].shape[0]
                generated_ids = output[prompt_length:]

                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                results.append(GenerationResult(
                    text=generated_text,
                    tokens_used=len(generated_ids),
                    finish_reason="stop",
                    model_name=self.model_name,
                    metadata={
                        "prompt_tokens": prompt_length,
                        "completion_tokens": len(generated_ids),
                        "total_tokens": prompt_length + len(generated_ids),
                        "device": self.device,
                    }
                ))

            return results

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            _reraise_if_classified(e)
            raise RuntimeError(f"Batch generation failed: {e}") from e
        finally:
            self._tokenizer_lock.release()

    def count_tokens(self, text: str) -> TokenCount:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            TokenCount object

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._is_loaded or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Serialize tokenizer access (see generate(): "Already borrowed").
        with self._tokenizer_lock:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return TokenCount(count=len(tokens), model_name=self.model_name)
            except Exception as e:
                logger.error(f"Token counting failed: {e}")
                raise RuntimeError(f"Token counting failed: {e}") from e

    def get_context_length(self) -> int:
        """
        Get maximum context length.

        Returns:
            int: Maximum context length in tokens
        """
        if self._context_length is not None:
            return self._context_length
        return 2048  # Default fallback

    def get_max_batch_size(self) -> int:
        """
        Get maximum batch size.

        Returns:
            int: Maximum batch size (conservative estimate)
        """
        # Conservative batch size based on available VRAM
        if self.device == "cuda":
            return 8
        else:
            return 1  # CPU is slow, use minimal batch size

    def supports_tool_calling(self) -> bool:
        """Check if the model supports native tool calling via chat template.

        Returns True if the tokenizer's chat template accepts a ``tools``
        parameter, which is the case for Qwen2.5, Llama 3.x, Mistral, etc.
        """
        if not self._is_loaded or self.tokenizer is None:
            return False
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            return False
        # Test whether the chat template accepts a `tools` kwarg
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "test",
                        "description": "test",
                        "parameters": {"type": "object", "properties": {}},
                    }
                }],
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except (TypeError, Exception):
            return False

    def unload(self) -> None:
        """
        Unload the model and free memory.

        Removes any accelerate device-dispatch hooks before deleting the
        model — leftover hooks can corrupt the CUDA forward state of
        subsequently-loaded models in the same process (observed as
        intermittent C-level aborts inside Qwen2 RMSNorm under pytest).

        The device memory the weights occupied is returned to the GPU, so the
        next model to load sees it as free.
        """
        if self.model is not None:
            logger.debug(f"Unloading model '{self.model_name}'...")
            try:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(self.model, recurse=True)
            except Exception:
                logger.debug("Failed to remove accelerate hooks during unload", exc_info=True)
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force garbage collection
        import gc
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                logger.debug("torch.cuda.synchronize() failed during unload", exc_info=True)
            from effgen.gpu.utils import release_cached_memory

            release_cached_memory()

        self._is_loaded = False
        logger.debug(f"Model '{self.model_name}' unloaded successfully")
