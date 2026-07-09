"""
Abstract base class for all model implementations in effGen.

This module defines the interface that all model engines must implement,
ensuring consistent behavior across vLLM, Transformers, and API adapters.
"""

from __future__ import annotations

import functools
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of supported model types."""
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MLX = "mlx"
    MLX_VLM = "mlx_vlm"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    # Optional draft model for speculative decoding. Backends that
    # do not support speculative decoding should ignore this field.
    draft_model: Any = None
    # Reasoning model controls (OpenAI o-series). Both default to None
    # for full back-compat — non-reasoning models silently ignore them.
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    max_reasoning_tokens: int | None = None
    # Gemini thinking controls. thinking_budget=None means field not sent;
    # 0 disables thinking; >0 sets token budget. include_thoughts=True
    # surfaces the reasoning trace in ModelResponse.metadata["thinking"].
    thinking_budget: int | None = None
    include_thoughts: bool = False
    # Gemini grounding. When True AND the model supports_grounding, Google
    # Search grounding is activated and grounding_chunks appear in metadata.
    grounding: bool = False
    # Anthropic extended thinking. Pass {"type": "enabled", "budget_tokens": N}.
    # None means the field is not sent (standard generation).
    thinking: dict | None = None
    # Structured-output controls. When set, backends that support a native
    # JSON/structured mode constrain generation accordingly. response_mime_type
    # (e.g. "application/json") asks the provider to emit raw JSON; response_schema
    # is an optional JSON-Schema dict the provider validates against where it can.
    # Both default to None so ordinary generation is unaffected.
    response_mime_type: str | None = None
    response_schema: dict | None = None


@dataclass
class GenerationResult:
    """Result from a generation call.

    On a reasoning model, ``text`` can be empty even on a normal (non-error)
    return: the output-token budget is spent on hidden reasoning before any
    visible token is emitted. This is reported as ``finish_reason == "length"``
    and ``metadata["truncated"] is True`` — check either before trusting an
    empty ``text``/``str(result)``, or call through :class:`~effgen.core.agent.Agent`
    instead, which already raises a clear error in that case.
    """
    text: str
    tokens_used: int
    finish_reason: str
    model_name: str
    metadata: dict[str, Any] | None = None

    def __str__(self) -> str:
        """The generated text, so ``print(model.generate(...))`` shows the
        answer rather than the full dataclass repr — mirroring ``AgentResponse``.
        The unambiguous ``repr()`` still shows every field for debugging."""
        return self.text

    def _repr_html_(self) -> str:
        """Rich HTML card for Jupyter/IPython — mirrors ``AgentResponse`` so a
        raw ``load_model(...).generate(...)`` renders a tidy card (answer +
        model · time · tokens · cost) in a notebook, not a dataclass repr."""
        from effgen.ui import generation_result_html

        return generation_result_html(self)

    def _repr_markdown_(self) -> str:
        """Plain-markdown notebook view (answer + a small metric footer)."""
        from effgen.ui import generation_result_markdown

        return generation_result_markdown(self)


@dataclass
class TokenCount:
    """Token count information."""
    count: int
    model_name: str


def _stamp_latency(result: Any, elapsed_s: float) -> Any:
    """Fold per-call wall-clock latency + a truncation flag onto a
    ``GenerationResult``'s metadata.

    Adds ``latency_ms`` and ``duration_s`` (via ``setdefault``, so an engine that
    measures its own latency wins) so benchmarkers can read throughput straight off
    a raw ``generate()`` result. Also adds ``truncated`` (``finish_reason ==
    "length"``) so a caller working directly with ``model.generate()`` (no
    ``Agent`` in between) can detect a truncated/empty-from-truncation result
    without string-matching ``finish_reason`` itself. Non-``GenerationResult``
    values pass through untouched.
    """
    if isinstance(result, GenerationResult):
        meta = result.metadata
        if meta is None:
            meta = {}
            result.metadata = meta
        meta.setdefault("latency_ms", round(elapsed_s * 1000.0, 1))
        meta.setdefault("duration_s", round(elapsed_s, 4))
        meta.setdefault("truncated", result.finish_reason == "length")
    return result


def _stamp_cost(model: "BaseModel", result: Any) -> None:
    """Accumulate this call's cost onto the model instance's running total.

    Populates ``metadata["total_cost"]`` via presence check, so an adapter that
    already tracks its own cumulative total (OpenAI, Gemini, Anthropic) keeps
    its own bookkeeping untouched; every other adapter gets the same
    cumulative-cost field for free, derived from the ``cost_usd`` it already
    reports per call. Skipped when the result carries no ``cost_usd`` (e.g.
    local engines that do not price calls).
    """
    if not isinstance(result, GenerationResult):
        return
    meta = result.metadata
    if meta is None or "total_cost" in meta:
        return
    cost = meta.get("cost_usd")
    if cost is None:
        return
    model.total_cost = getattr(model, "total_cost", 0.0) + cost
    meta["total_cost"] = model.total_cost


def _warn_if_silently_empty(model: "BaseModel", result: Any) -> None:
    """Log a warning when a reasoning model's raw ``generate()`` call returns
    empty text because its output budget was spent on hidden reasoning before
    any visible token (``finish_reason == "length"``).

    :class:`~effgen.core.agent.Agent` already detects and escalates this case;
    a caller using ``model.generate()`` directly has no such safety net, so an
    empty ``GenerationResult`` would otherwise look like a working call that
    produced nothing.
    """
    if not isinstance(result, GenerationResult):
        return
    if result.text or result.finish_reason != "length":
        return
    from ._adapter_utils import needs_reasoning_headroom
    if not needs_reasoning_headroom(model):
        return
    logger.warning(
        "%s returned empty text after exhausting its token budget on internal "
        "reasoning (finish_reason='length'). Increase max_tokens, or check "
        "metadata['truncated'] before trusting an empty result.",
        getattr(model, "model_name", model.__class__.__name__),
    )


def _preflight_budget_check(model: "BaseModel") -> None:
    """Refuse to start a call when a configured budget is already at or over its cap.

    Runs before the provider call is made, so a call refused here never reaches
    the network and is never billed — unlike the check inside
    ``CostTracker.record()``, which runs after a call's tokens are already known
    and its cost already persisted. A tracker or budget-config read failure is
    swallowed (best-effort; the post-spend check still applies as a backstop).
    """
    try:
        from effgen.models._cost import CostTracker
    except ImportError:
        return
    provider = getattr(getattr(model, "model_type", None), "value", "") or ""
    model_name = getattr(model, "model_name", "") or ""
    CostTracker.get().check_preflight(provider, model_name)


def _timed_generate(func):
    """Wrap a ``generate`` method with a pre-call budget check and call latency."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        _preflight_budget_check(self)
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        result = _stamp_latency(result, time.perf_counter() - start)
        _stamp_cost(self, result)
        _warn_if_silently_empty(self, result)
        return result
    wrapper.__effgen_timed__ = True
    return wrapper


def _timed_generate_batch(func):
    """Wrap a ``generate_batch`` method with a pre-call budget check so each
    result carries call latency.

    The whole batch shares one wall-clock measurement (the per-item split isn't
    knowable here); each result gets it via ``setdefault`` so an engine that
    records true per-item latency keeps its own value.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        _preflight_budget_check(self)
        start = time.perf_counter()
        results = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        if isinstance(results, list):
            for r in results:
                _stamp_latency(r, elapsed)
                _stamp_cost(self, r)
                _warn_if_silently_empty(self, r)
        return results
    wrapper.__effgen_timed__ = True
    return wrapper


def _budget_gated_stream(func):
    """Wrap a ``generate_stream`` method with a pre-call budget check.

    The check runs synchronously when the caller invokes ``generate_stream``
    (not on first ``next()``), so a refusal happens before any token request
    reaches the provider.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        _preflight_budget_check(self)
        return func(self, *args, **kwargs)
    wrapper.__effgen_timed__ = True
    return wrapper


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.

    This class defines the interface that all model engines must implement,
    including vLLM, Transformers, OpenAI, Anthropic, and Gemini adapters.

    Attributes:
        model_name: The name or identifier of the model
        model_type: The type of model engine
        context_length: Maximum context length supported by the model
    """

    def __init_subclass__(cls, **kwargs):
        """Auto-instrument each engine's generation methods with a budget
        pre-check and call timing.

        Every concrete engine that defines ``generate``/``generate_batch``/
        ``generate_stream`` gets a pre-call check against any configured
        daily/monthly budget (refusing before the provider call is made once
        the period is already at or over its cap) and, for ``generate``/
        ``generate_batch``, its result(s) stamped with ``latency_ms``/
        ``duration_s`` (see ``_stamp_latency``) — without each engine repeating
        the bookkeeping. Abstract or already-wrapped methods are left alone, so
        this is safe across the engine hierarchy.
        """
        super().__init_subclass__(**kwargs)
        for name, wrapper in (("generate", _timed_generate),
                              ("generate_batch", _timed_generate_batch),
                              ("generate_stream", _budget_gated_stream)):
            func = cls.__dict__.get(name)
            if (
                func is not None
                and callable(func)
                and not getattr(func, "__isabstractmethod__", False)
                and not getattr(func, "__effgen_timed__", False)
            ):
                setattr(cls, name, wrapper(func))

    def __init__(
        self,
        model_name: str,
        model_type: ModelType,
        context_length: int | None = None,
        **kwargs
    ):
        """
        Initialize the base model.

        Args:
            model_name: Name or identifier of the model
            model_type: Type of model engine
            context_length: Maximum context length (auto-detected if None)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self._context_length = context_length
        self._is_loaded = False
        self._metadata: dict[str, Any] = {}

    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.

        This method should handle all initialization logic including:
        - Model weight loading
        - Device allocation
        - Tokenizer initialization
        - Configuration setup

        Raises:
            RuntimeError: If model loading fails
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt synchronously.

        Args:
            prompt: Input text prompt
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult containing generated text and metadata

        Raises:
            RuntimeError: If generation fails
            ValueError: If prompt is invalid or exceeds context length
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text from a prompt with token-by-token streaming.

        Args:
            prompt: Input text prompt
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Yields:
            str: Individual tokens or chunks of generated text

        Raises:
            RuntimeError: If generation fails
            ValueError: If prompt is invalid or exceeds context length
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> TokenCount:
        """
        Count the number of tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            TokenCount object with token count and model name

        Raises:
            RuntimeError: If tokenization fails
        """
        pass

    @abstractmethod
    def get_context_length(self) -> int:
        """
        Get the maximum context length supported by the model.

        Returns:
            int: Maximum number of tokens the model can handle
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model from memory and free resources.

        This method should:
        - Free GPU/CPU memory
        - Close any open connections
        - Clean up temporary files
        """
        pass

    def supports_tool_calling(self) -> bool:
        """
        Check if the model supports native tool/function calling.

        Native tool calling uses the model's built-in tool call format
        (e.g., chat template ``tools`` parameter, API tool_calls) instead
        of parsing free-text ReAct output.

        Returns:
            bool: True if native tool calling is supported.
        """
        return False

    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded.

        Returns:
            bool: True if model is loaded and ready for inference
        """
        return self._is_loaded

    def get_total_cost(self) -> float:
        """Cumulative cost (USD) charged to this model instance since it was
        created or since :meth:`reset_cost` was last called. Populated from
        each call's ``cost_usd``; local/unpriced engines stay at ``0.0``.
        """
        return getattr(self, "total_cost", 0.0)

    def reset_cost(self) -> None:
        """Reset this instance's cumulative cost counter (see
        :meth:`get_total_cost`) back to zero."""
        self.total_cost = 0.0

    def get_metadata(self) -> dict[str, Any]:
        """
        Get model metadata and information.

        Returns:
            Dict containing model information such as:
            - Model architecture
            - Parameter count
            - Quantization details
            - Device allocation
            - Memory usage
        """
        return self._metadata

    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate that a prompt is within context length limits.

        Args:
            prompt: Prompt to validate

        Returns:
            bool: True if prompt is valid

        Raises:
            ValueError: If prompt exceeds context length
        """
        token_count = self.count_tokens(prompt)
        max_length = self.get_context_length()

        if token_count.count > max_length:
            raise ValueError(
                f"Prompt length ({token_count.count} tokens) exceeds "
                f"model context length ({max_length} tokens)"
            )

        return True

    def __enter__(self):
        """Context manager entry."""
        if not self._is_loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"type={self.model_type.value}, "
            f"loaded={self._is_loaded})"
        )


class BatchModel(BaseModel):
    """
    Extended base class for models that support batch processing.

    This class extends BaseModel with batch generation capabilities,
    useful for high-throughput scenarios.
    """

    @abstractmethod
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
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult objects, one per prompt

        Raises:
            RuntimeError: If batch generation fails
            ValueError: If any prompt is invalid
        """
        pass

    def get_max_batch_size(self) -> int:
        """
        Get the maximum batch size supported.

        Returns:
            int: Maximum number of prompts that can be processed in one batch
        """
        return 1  # Default to no batching


class FunctionCallingModel(BaseModel):
    """
    Extended base class for models that support function/tool calling.

    This class extends BaseModel with function calling capabilities,
    primarily used by API adapters (OpenAI, Anthropic, Gemini).
    """

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        config: GenerationConfig | None = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text with tool/function calling support.

        Args:
            prompt: Input text prompt
            tools: List of tool definitions in OpenAI function format
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with potential tool calls in metadata

        Raises:
            RuntimeError: If generation fails
            ValueError: If tools are malformed
        """
        pass

    @abstractmethod
    def supports_function_calling(self) -> bool:
        """
        Check if the model supports function calling.

        Returns:
            bool: True if function calling is supported
        """
        pass
