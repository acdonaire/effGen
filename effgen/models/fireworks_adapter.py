"""
Fireworks AI SDK adapter for effGen.

Supports all Fireworks chat models with built-in rate-limit coordination,
real streaming via SSE, native function-calling on supported models, and
per-request cost tracking via CostTracker.

Fireworks uses an OpenAI-compatible API shape (chat.completions.create) with
model IDs in the format ``accounts/fireworks/models/<id>``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from effgen.models._adapter_utils import normalize_finish_reason, provider_runtime_error
from effgen.models._cost import CostTracker
from effgen.models._rate_limit import RateLimitCoordinator
from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    TokenCount,
)
from effgen.models.errors import ModelAuthError, ModelNotFoundError
from effgen.models.fireworks_models import (
    FIREWORKS_DEFAULT_MODEL,
    FIREWORKS_MODELS,
    REGISTRY_FETCH_DATE,
)
from effgen.models.latency_tracker import timed_call
from effgen.observability.spans import ModelAttrs
from effgen.observability.tracing import set_span_attribute as _set_span_attr
from effgen.utils.async_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from effgen.models._rate_limit_store import SQLiteRateLimitStore

logger = logging.getLogger(__name__)

_FIREWORKS_MODEL_TYPE_VALUE = "fireworks"
_FIREWORKS_PREFIX = "accounts/fireworks/models/"


class _FireworksModelType:
    """Sentinel so ModelType enum doesn't need patching."""
    value = _FIREWORKS_MODEL_TYPE_VALUE


class FireworksAdapter(BaseModel):
    """
    Adapter for Fireworks AI inference API.

    Wraps the ``fireworks-ai`` SDK (OpenAI-compatible) with the standard
    effGen BaseModel interface.  Supports:

    - Synchronous and async generation
    - Real token-by-token streaming (``generate_stream``)
    - Native function-calling on supported models (``generate_with_tools``)
    - Per-request cost tracking via :class:`~effgen.models._cost.CostTracker`
    - Per-model rate-limit coordination (RPM, TPM)
    - Dynamic model catalog refresh and drift detection (``refresh_models()``)

    Model IDs must use the full path format:
    ``accounts/fireworks/models/<id>``.
    Short IDs are accepted and automatically expanded.

    Args:
        model_name: Fireworks model ID.  Defaults to
            ``"accounts/fireworks/models/gpt-oss-120b"``.
        api_key: Fireworks API key.  If omitted, reads ``FIREWORKS_API_KEY``
            from the environment.
        max_retries: Maximum number of SDK retry attempts on transient errors.
        timeout: Per-request timeout in seconds.
        enable_rate_limiting: Wire built-in
            :class:`~effgen.models._rate_limit.RateLimitCoordinator`.
        enable_cost_tracking: Record token usage in the global
            :class:`~effgen.models._cost.CostTracker`.
        warn_unknown_model: Emit a warning (not an error) when the model ID is
            not in the bundled registry.  Useful for newly-released models.

    Example::

        from effgen.models.fireworks_adapter import FireworksAdapter

        adapter = FireworksAdapter("accounts/fireworks/models/gpt-oss-120b")
        adapter.load()

        result = adapter.generate("What is the capital of France?")
        print(result.text)

        for chunk in adapter.generate_stream("Count from 1 to 5."):
            print(chunk, end="", flush=True)

        adapter.unload()
    """

    #: Provider label used for metrics/error reporting (see Agent._model_provider).
    _provider = "fireworks"

    def __init__(
        self,
        model_name: str = FIREWORKS_DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 6,
        timeout: int = 120,
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
        warn_unknown_model: bool = True,
        rate_limit_storage: "SQLiteRateLimitStore | None" = None,
        **kwargs: Any,
    ) -> None:
        # Normalise short IDs → full path
        if not model_name.startswith(_FIREWORKS_PREFIX):
            model_name = f"{_FIREWORKS_PREFIX}{model_name}"

        info = FIREWORKS_MODELS.get(model_name)
        if info is None and warn_unknown_model:
            logger.warning(
                "FireworksAdapter: model '%s' is not in the bundled registry "
                "(registry date: %s).  The model may be new — call "
                "fireworks_models.refresh_models() to check for drift.  "
                "Proceeding with default parameters.",
                model_name,
                REGISTRY_FETCH_DATE,
            )
            info = {}

        super().__init__(
            model_name=model_name,
            model_type=_FireworksModelType(),  # type: ignore[arg-type]
            context_length=(info or {}).get("context", 131_072),
        )
        self._api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._extra_kwargs = kwargs
        self._client: Any = None
        self._enable_cost_tracking = enable_cost_tracking

        self._rate_limiter: RateLimitCoordinator | None = None
        if enable_rate_limiting and info:
            self._rate_limiter = RateLimitCoordinator(
                provider="fireworks",
                model=model_name,
                rpm=info.get("rpm", 10),
                rph=info.get("rpm", 10) * 60,
                rpd=10_000,
                tpm=info.get("tpm", 40_000),
                tph=info.get("tpm", 40_000) * 60,
                tpd=10_000_000,
                storage=rate_limit_storage,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the Fireworks SDK client.

        Raises:
            RuntimeError: If ``fireworks-ai`` is not installed.
            ValueError: If no API key is available.
        """
        try:
            from fireworks.client import Fireworks
        except ImportError as exc:
            raise RuntimeError(
                "fireworks-ai SDK is not installed. "
                "Install with: pip install 'effgen[fireworks]'"
            ) from exc

        if not (self._api_key or os.getenv("FIREWORKS_API_KEY")):
            raise ValueError(
                "Fireworks API key not found. Set the FIREWORKS_API_KEY "
                "environment variable or pass api_key= to FireworksAdapter."
            )

        self._client = Fireworks(
            api_key=self._api_key or os.getenv("FIREWORKS_API_KEY"),
            timeout=self.timeout,
        )
        self._is_loaded = True

        info = FIREWORKS_MODELS.get(self.model_name, {})
        self._metadata = {
            "model_name": self.model_name,
            "context_length": self.get_context_length(),
            "provider": "fireworks",
            "family": info.get("family", ""),
            "organization": info.get("organization", ""),
            "display_name": info.get("display_name", ""),
            "modality": info.get("modality", "chat"),
            "supports_native_tools": info.get("supports_native_tools", False),
            "supports_streaming": info.get("supports_streaming", True),
            "pricing_per_1m_input": info.get("pricing_per_1m_input", 0),
            "pricing_per_1m_output": info.get("pricing_per_1m_output", 0),
            "rpm": info.get("rpm"),
            "tpm": info.get("tpm"),
            "registry_fetch_date": REGISTRY_FETCH_DATE,
        }
        logger.info("FireworksAdapter loaded for model '%s'", self.model_name)

    def unload(self) -> None:
        """Release SDK client resources."""
        self._client = None
        self._is_loaded = False
        logger.info("FireworksAdapter unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response synchronously via the Fireworks AI API.

        Args:
            prompt: User prompt text.
            config: Optional generation config.
            **kwargs: Forwarded to ``chat.completions.create``.

        Returns:
            GenerationResult with text and token usage.
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("FireworksAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        try:
            est_tokens = self.count_tokens(prompt).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            run_coroutine_sync(self._rate_limiter.acquire(est_tokens))

        with timed_call("fireworks", self.model_name):
            result = self._do_generate(prompt, config, **kwargs)

        if self._rate_limiter is not None:
            actual = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(actual)

        return result

    async def async_generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Async version of :meth:`generate` — preferred inside async contexts."""
        if not self._is_loaded or self._client is None:
            raise RuntimeError("FireworksAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        try:
            est_tokens = self.count_tokens(prompt).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(est_tokens)

        result = self._do_generate(prompt, config, **kwargs)

        if self._rate_limiter is not None:
            actual = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(actual)

        return result

    def _do_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Internal: make the SDK call and return a GenerationResult."""
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        request_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        if config.temperature is not None and config.temperature != 0.7:
            request_params["temperature"] = config.temperature
        if config.top_p is not None and config.top_p != 0.9:
            request_params["top_p"] = config.top_p
        if config.max_tokens is not None:
            request_params["max_tokens"] = config.max_tokens
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        if config.seed is not None:
            request_params["seed"] = config.seed
        if config.presence_penalty:
            request_params["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty:
            request_params["frequency_penalty"] = config.frequency_penalty

        info = FIREWORKS_MODELS.get(self.model_name, {})
        if tools and info.get("supports_native_tools", False):
            openai_tools = []
            for t in tools:
                if isinstance(t, dict):
                    openai_tools.append(t if "type" in t else {"type": "function", "function": t})
                else:
                    openai_tools.append({"type": "function", "function": t.metadata.to_json_schema()})
            request_params["tools"] = openai_tools
            request_params["tool_choice"] = "auto"

        request_params.update(kwargs)

        _last_exc: Exception | None = None
        for _attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**request_params)
                break
            except Exception as exc:
                _last_exc = exc
                msg = str(exc)
                msg_lower = msg.lower()

                is_auth = (
                    "401" in msg
                    or "403" in msg
                    or "invalid_api_key" in msg_lower
                    or "invalid api key" in msg_lower
                    or "unauthorized" in msg_lower
                    or "authentication" in msg_lower
                    or "api_key" in msg_lower and "invalid" in msg_lower
                )
                if is_auth:
                    raise ModelAuthError(
                        provider="fireworks",
                        model_name=self.model_name,
                        message=msg,
                    ) from exc

                is_not_found = (
                    "NOT_FOUND" in msg
                    or "not found" in msg_lower
                    or "model not found" in msg_lower
                    or "inaccessible" in msg_lower
                    or "not deployed" in msg_lower
                )
                if is_not_found:
                    raise ModelNotFoundError(
                        provider="fireworks",
                        model_name=self.model_name,
                        message=f"Fireworks model '{self.model_name}' not found or not deployed. "
                                f"Check available models with: "
                                f"from effgen.models.fireworks_models import available_models; "
                                f"print(available_models())",
                    ) from exc

                is_rate = (
                    "429" in msg
                    or "rate_limit" in msg_lower
                    or "rate limit" in msg_lower
                    or "exceeded your rate limit" in msg_lower
                )
                is_server = "500" in msg or "503" in msg or "internal" in msg_lower
                is_timeout = "timeout" in msg_lower

                if is_rate or is_server or is_timeout:
                    if _attempt >= self.max_retries:
                        logger.error(
                            "Fireworks API error after %d retries: %s", _attempt, exc
                        )
                        raise provider_runtime_error(
                            "fireworks", self.model_name, "generate", exc,
                            message=f"Fireworks API failed for {self.model_name} after "
                                    f"{self.max_retries} retries",
                        ) from exc
                    delay = min(60.0, 2.0 * (2 ** (_attempt - 1)) + random.uniform(0, 0.5))
                    logger.warning(
                        "Fireworks transient error on attempt %d/%d — retrying in %.1fs: %s",
                        _attempt, self.max_retries, delay, exc,
                    )
                    time.sleep(delay)
                    continue

                logger.error("Fireworks API call failed: %s", exc)
                raise provider_runtime_error("fireworks", self.model_name, "generate", exc, message="Fireworks generation failed") from exc
        else:
            assert _last_exc is not None
            raise provider_runtime_error(
                "fireworks", self.model_name, "generate", _last_exc,
                message=f"Fireworks generation failed after {self.max_retries} retries",
            ) from _last_exc

        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        finish_reason = normalize_finish_reason(choice.finish_reason)

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        tool_calls: list[dict[str, Any]] = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = tc.function.arguments
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                except (json.JSONDecodeError, AttributeError):
                    arguments = {}
                tool_calls.append({
                    "id": getattr(tc, "id", ""),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": tc.function.name,
                        "arguments": arguments,
                    },
                })

        cost = 0.0
        if self._enable_cost_tracking:
            cost = CostTracker.get().record(
                provider="fireworks",
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        logger.info(
            "Fireworks generated %d tokens (prompt=%d, completion=%d, cost=$%.6f)",
            total_tokens, prompt_tokens, completion_tokens, cost,
        )
        _set_span_attr(ModelAttrs.PROVIDER, "fireworks")
        _set_span_attr(ModelAttrs.NAME, self.model_name)
        _set_span_attr(ModelAttrs.INPUT_TOKENS, prompt_tokens)
        _set_span_attr(ModelAttrs.OUTPUT_TOKENS, completion_tokens)
        _set_span_attr(ModelAttrs.COST_USD, float(cost))
        _set_span_attr(ModelAttrs.OUTCOME, "ok")

        return GenerationResult(
            text=text,
            tokens_used=completion_tokens,
            finish_reason=finish_reason,
            model_name=self.model_name,
            metadata={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "provider": "fireworks",
                "cost_usd": cost,
                "tool_calls": tool_calls,
            },
        )

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]] | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate with native tool calling (OpenAI function-calling format).

        Args:
            prompt: User prompt text (ignored when *messages* is provided).
            tools: List of OpenAI-format tool dicts or effGen BaseTool objects.
            messages: Optional full conversation history (overrides *prompt*).
            config: Optional generation config.

        Returns:
            GenerationResult whose ``metadata["tool_calls"]`` contains parsed calls.
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("FireworksAdapter not loaded. Call load() first.")
        if config is None:
            config = GenerationConfig()
        return self._do_generate(prompt, config, tools=tools, messages=messages, **kwargs)

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a response token-by-token from the Fireworks AI API.

        Yields:
            str: Successive text chunks from the model.
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("FireworksAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        messages = [{"role": "user", "content": prompt}]
        request_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }

        if config.temperature is not None and config.temperature != 0.7:
            request_params["temperature"] = config.temperature
        if config.top_p is not None and config.top_p != 0.9:
            request_params["top_p"] = config.top_p
        if config.max_tokens is not None:
            request_params["max_tokens"] = config.max_tokens
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        if config.seed is not None:
            request_params["seed"] = config.seed
        if config.presence_penalty:
            request_params["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty:
            request_params["frequency_penalty"] = config.frequency_penalty

        request_params.update(kwargs)

        self._last_stream_tool_calls: list[dict[str, Any]] = []
        self._last_stream_finish_reason: str | None = None

        try:
            with timed_call("fireworks", self.model_name) as _stream_timer:
                _first_token = True
                stream = self._client.chat.completions.create(**request_params)

                prompt_tokens = 0
                completion_tokens = 0
                tool_calls_buf: dict[int, dict[str, Any]] = {}

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta and delta.content:
                        if _first_token:
                            _stream_timer.mark_first_token()
                            _first_token = False
                        yield delta.content

                    if delta and getattr(delta, "tool_calls", None):
                        for tc in delta.tool_calls:
                            idx = tc.index if getattr(tc, "index", None) is not None else 0
                            buf = tool_calls_buf.setdefault(
                                idx, {"id": "", "type": "function",
                                      "function": {"name": "", "arguments": ""}}
                            )
                            if getattr(tc, "id", None):
                                buf["id"] = tc.id
                            fn = getattr(tc, "function", None)
                            if fn is not None:
                                if getattr(fn, "name", None):
                                    buf["function"]["name"] = fn.name
                                if getattr(fn, "arguments", None):
                                    buf["function"]["arguments"] += fn.arguments

                    if choice.finish_reason:
                        self._last_stream_finish_reason = choice.finish_reason

                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        usage = chunk.usage
                        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

                finalized: list[dict[str, Any]] = []
                for _idx, buf in sorted(tool_calls_buf.items()):
                    raw_args = buf["function"]["arguments"]
                    try:
                        parsed_args = json.loads(raw_args) if raw_args else {}
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {}
                    finalized.append({
                        "id": buf["id"],
                        "type": buf["type"],
                        "function": {
                            "name": buf["function"]["name"],
                            "arguments": parsed_args,
                        },
                    })
                self._last_stream_tool_calls = finalized

            if self._enable_cost_tracking and (prompt_tokens or completion_tokens):
                CostTracker.get().record(
                    provider="fireworks",
                    model=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

        except Exception as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            if (
                "401" in msg
                or "403" in msg
                or "invalid_api_key" in msg_lower
                or "invalid api key" in msg_lower
                or "unauthorized" in msg_lower
                or "authentication" in msg_lower
            ):
                raise ModelAuthError(
                    provider="fireworks",
                    model_name=self.model_name,
                    message=msg,
                ) from exc
            if (
                "NOT_FOUND" in msg
                or "not found" in msg_lower
                or "not deployed" in msg_lower
            ):
                raise RuntimeError(
                    f"Fireworks model '{self.model_name}' not found or not deployed. "
                    f"Use available_models() to list valid IDs."
                ) from exc
            logger.error("Fireworks streaming failed: %s", exc)
            raise provider_runtime_error("fireworks", self.model_name, "stream", exc, message="Fireworks streaming failed") from exc

    # ------------------------------------------------------------------
    # Token counting / context length
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> TokenCount:
        """Estimate token count via tiktoken (cl100k_base)."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            count = len(enc.encode(text))
        except Exception:
            count = max(1, len(text) // 4)
        return TokenCount(count=count, model_name=self.model_name)

    def get_context_length(self) -> int:
        """Return the context window size for the loaded model."""
        return int(FIREWORKS_MODELS.get(self.model_name, {}).get("context", 131_072))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def rate_limit_status(self) -> dict[str, Any]:
        """Return current rate-limit window status from the coordinator."""
        if self._rate_limiter is None:
            return {"enabled": False}
        return {"enabled": True, "status": str(self._rate_limiter)}

    @property
    def supports_native_tools(self) -> bool:
        """True if this model supports OpenAI-format function calling."""
        return bool(FIREWORKS_MODELS.get(self.model_name, {}).get("supports_native_tools", False))

    def supports_tool_calling(self) -> bool:
        """Return True if the loaded model supports native tool-calling."""
        return bool(FIREWORKS_MODELS.get(self.model_name, {}).get("supports_native_tools", False))

    def supports_function_calling(self) -> bool:
        """Alias for :meth:`supports_tool_calling`."""
        return self.supports_tool_calling()

    @property
    def supports_streaming(self) -> bool:
        """True if this model supports streaming responses."""
        return bool(FIREWORKS_MODELS.get(self.model_name, {}).get("supports_streaming", True))

    def pricing(self) -> dict[str, float]:
        """Return per-request pricing info for the loaded model."""
        info = FIREWORKS_MODELS.get(self.model_name, {})
        return {
            "input_per_1m_usd": float(info.get("pricing_per_1m_input", 0)),
            "output_per_1m_usd": float(info.get("pricing_per_1m_output", 0)),
        }


# ---------------------------------------------------------------------------
# Self-register with the ProviderRegistry on first import (idempotent)
# ---------------------------------------------------------------------------
def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.fireworks_models import FIREWORKS_MODELS
        from effgen.models.registry import ProviderRegistry
        ProviderRegistry.register(
            "fireworks",
            FireworksAdapter,
            FIREWORKS_MODELS,
            env_keys=["FIREWORKS_API_KEY"],
            capabilities={Capability.chat, Capability.streaming, Capability.tools, Capability.json_schema},
            # $1 free credits on signup; pay-per-token after. Provider default = cheapest chat model.
            # qwen3-1p7b: $0.10/$0.10; llama-v3p2-1b-instruct: $0.10/$0.10; llama-v3p3-70b: $0.90/$0.90 per 1M.
            # Pricing verified: https://fireworks.ai/pricing (2026-05-11)
            pricing={"input_per_1m": 0.10, "output_per_1m": 0.10, "free_tier": False},
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
