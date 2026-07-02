"""
Groq Cloud SDK adapter for effGen.

Supports all Groq chat models with built-in rate-limit coordination,
real streaming via SSE, native function-calling on supported models, and
per-request cost tracking via CostTracker.

Groq uses an OpenAI-compatible API shape, so the implementation mirrors
the CerebrasAdapter closely with Groq-specific rate limits.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from effgen.models._adapter_utils import normalize_finish_reason, provider_runtime_error
from effgen.models._cost import CostTracker
from effgen.models._multimodal import require_vision_support
from effgen.models._rate_limit import RateLimitCoordinator
from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    TokenCount,
)
from effgen.models.errors import ModelAuthError, ModelNotFoundError
from effgen.models.groq_models import (
    GROQ_DEFAULT_MODEL,
    GROQ_MODELS,
    chat_models,
)
from effgen.models.latency_tracker import timed_call
from effgen.observability.spans import ModelAttrs
from effgen.observability.tracing import set_span_attribute as _set_span_attr
from effgen.utils.async_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from effgen.models._rate_limit_store import SQLiteRateLimitStore

logger = logging.getLogger(__name__)

_GROQ_MODEL_TYPE_VALUE = "groq"


def _redact_groq_org(message: str) -> str:
    """Remove the caller's organization id from a Groq error body before it is
    surfaced (it is an account identifier, not useful for debugging)."""
    return re.sub(r"organization `org_[^`]+`", "organization `***`", message)


def _is_request_too_large(message: str, message_lower: str) -> bool:
    """True when a Groq error is a 413 payload-too-large (a single oversized
    request), not a 429 rate limit. Groq returns 413 with a ``rate_limit_exceeded``
    code for a request over the per-minute token limit, so status + wording are
    checked rather than the misleading code."""
    return (
        "413" in message
        or "request too large" in message_lower
        or "reduce your message size" in message_lower
    )


def _parse_failed_generation_tool_call(message: str) -> dict[str, Any] | None:
    """Extract a tool call from Groq's ``tool_use_failed`` failed_generation text."""
    match = re.search(
        r"<function=([A-Za-z_]\w*)\s*>?\s*(.*?)</function>",
        message,
        re.DOTALL,
    )
    if not match:
        return None

    name = match.group(1)
    raw_args = match.group(2).strip()
    if raw_args.startswith("(") and raw_args.endswith(")"):
        raw_args = raw_args[1:-1].strip()

    try:
        arguments = json.loads(raw_args) if raw_args else {}
    except (json.JSONDecodeError, TypeError):
        arguments = {"__raw_input__": raw_args}

    if not isinstance(arguments, dict):
        arguments = {"__raw_input__": raw_args}

    return {
        "id": "",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


class _GroqModelType:
    """Sentinel so ModelType enum doesn't need patching."""
    value = _GROQ_MODEL_TYPE_VALUE


class GroqAdapter(BaseModel):
    """
    Adapter for Groq Cloud inference API.

    Wraps the ``groq`` SDK with the standard effGen BaseModel interface.
    Groq mirrors the OpenAI API shape. Supports:

    - Synchronous and async generation
    - Real token-by-token streaming (``generate_stream``)
    - Native function-calling on supported models (``generate_with_tools``)
    - Per-request cost tracking via :class:`~effgen.models._cost.CostTracker`
    - Per-model rate-limit coordination (RPM, RPD, TPM, TPD)

    Args:
        model_name: Groq model ID. Must be a key in
            :data:`~effgen.models.groq_models.GROQ_MODELS`.
            Defaults to ``"llama-3.1-8b-instant"``.
        api_key: Groq API key. If omitted, reads ``GROQ_API_KEY``
            from the environment.
        max_retries: Maximum number of SDK retry attempts.
        timeout: Per-request timeout in seconds.
        enable_rate_limiting: Wire built-in
            :class:`~effgen.models._rate_limit.RateLimitCoordinator`.
        enable_cost_tracking: Record token usage in the global
            :class:`~effgen.models._cost.CostTracker`.

    Example::

        from effgen.models.groq_adapter import GroqAdapter

        adapter = GroqAdapter("llama-3.3-70b-versatile")
        adapter.load()

        result = adapter.generate("What is the capital of France?")
        print(result.text)

        for chunk in adapter.generate_stream("Count from 1 to 5."):
            print(chunk, end="", flush=True)

        adapter.unload()
    """

    def __init__(
        self,
        model_name: str = GROQ_DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
        rate_limit_storage: "SQLiteRateLimitStore | None" = None,
        **kwargs: Any,
    ) -> None:
        if model_name not in GROQ_MODELS:
            from effgen.models._catalog import suggest_for_missing

            raise ModelNotFoundError(
                provider="groq",
                model_name=model_name,
                message=f"Unknown Groq model '{model_name}'."
                        + suggest_for_missing("groq", model_name),
            )

        info = GROQ_MODELS[model_name]
        if info.get("modality") not in ("chat", None):
            raise ValueError(
                f"Groq model '{model_name}' is a {info.get('modality')} model "
                f"and cannot be used via chat completions. "
                f"Chat models: {chat_models()}"
            )

        super().__init__(
            model_name=model_name,
            model_type=_GroqModelType(),  # type: ignore[arg-type]
            context_length=info.get("context", 131_072),
        )
        self._api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._extra_kwargs = kwargs
        self._client: Any = None
        self._enable_cost_tracking = enable_cost_tracking

        self._rate_limiter: RateLimitCoordinator | None = None
        if enable_rate_limiting:
            tpd = info.get("tpd") or 0
            rpd = info.get("rpd") or 0
            self._rate_limiter = RateLimitCoordinator(
                provider="groq",
                model=model_name,
                rpm=info.get("rpm", 30),
                rph=info.get("rpm", 30) * 60,
                rpd=rpd if rpd else 100_000,
                tpm=info.get("tpm", 6_000),
                tph=info.get("tpm", 6_000) * 60,
                tpd=tpd if tpd else 10_000_000,
                storage=rate_limit_storage,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the Groq SDK client.

        Raises:
            RuntimeError: If ``groq`` is not installed.
            ValueError: If no API key is available.
        """
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "groq SDK is not installed. "
                "Install with: pip install 'effgen[groq]'"
            ) from exc

        if not (self._api_key or os.getenv("GROQ_API_KEY")):
            raise ValueError(
                "Groq API key not found. Set the GROQ_API_KEY "
                "environment variable or pass api_key= to GroqAdapter."
            )

        self._client = Groq(
            api_key=self._api_key or os.getenv("GROQ_API_KEY"),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._is_loaded = True

        info = GROQ_MODELS.get(self.model_name, {})
        self._metadata = {
            "model_name": self.model_name,
            "context_length": self.get_context_length(),
            "provider": "groq",
            "family": info.get("family", ""),
            "modality": info.get("modality", "chat"),
            "supports_native_tools": info.get("supports_native_tools", False),
            "supports_streaming": info.get("supports_streaming", True),
            "rpm": info.get("rpm"),
            "rpd": info.get("rpd"),
            "tpm": info.get("tpm"),
            "tpd": info.get("tpd"),
            "notes": info.get("notes", ""),
        }
        logger.info("GroqAdapter loaded for model '%s'", self.model_name)

    def unload(self) -> None:
        """Release SDK client resources."""
        self._client = None
        self._is_loaded = False
        logger.info("GroqAdapter unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response synchronously via the Groq API.

        Args:
            prompt: User prompt text.
            config: Optional generation config.
            **kwargs: Forwarded to ``chat.completions.create``.

        Returns:
            GenerationResult with text and token usage.
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("GroqAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="groq",
            model_name=self.model_name,
            supports_vision=GROQ_MODELS.get(self.model_name, {}).get("supports_vision", False),
            hint="Use 'meta-llama/llama-4-scout-17b-16e-instruct' for Groq vision.",
        )

        try:
            prompt_text = prompt if isinstance(prompt, str) else str(getattr(prompt, "text", prompt))
            est_tokens = self.count_tokens(prompt_text).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            run_coroutine_sync(self._rate_limiter.acquire(est_tokens))

        with timed_call("groq", self.model_name):
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
            raise RuntimeError("GroqAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="groq",
            model_name=self.model_name,
            supports_vision=GROQ_MODELS.get(self.model_name, {}).get("supports_vision", False),
            hint="Use 'meta-llama/llama-4-scout-17b-16e-instruct' for Groq vision.",
        )

        try:
            prompt_text = prompt if isinstance(prompt, str) else str(getattr(prompt, "text", prompt))
            est_tokens = self.count_tokens(prompt_text).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(est_tokens)

        result = self._do_generate(prompt, config, **kwargs)

        if self._rate_limiter is not None:
            actual = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(actual)

        return result

    @staticmethod
    def _message_to_groq(message: Any) -> dict[str, Any]:
        """Convert an effGen Message to a Groq/OpenAI-compatible dict."""
        import base64

        from effgen.core.messages import ImagePart, TextPart, VideoPart
        from effgen.multimodal.image_pre import prepare as _preprocess_image

        role = message.role.value
        content_parts: list[dict[str, Any]] = []

        for part in message.content:
            if isinstance(part, TextPart):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                processed = _preprocess_image(part, "groq", "")
                b64 = base64.b64encode(processed.image).decode()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{processed.mime};base64,{b64}"},
                })
            elif isinstance(part, VideoPart):
                for frame in part.frames:
                    b64 = base64.b64encode(frame).decode()
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.mime};base64,{b64}"},
                    })

        if len(content_parts) == 1 and content_parts[0].get("type") == "text":
            return {"role": role, "content": content_parts[0]["text"]}
        return {"role": role, "content": content_parts}

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
            # Handle effGen Message objects
            try:
                from effgen.core.messages import Message
                if isinstance(prompt, Message):
                    messages = [self._message_to_groq(prompt)]
                elif isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
                    messages = [self._message_to_groq(m) for m in prompt]
                else:
                    messages = [{"role": "user", "content": prompt}]
            except ImportError:
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

        info = GROQ_MODELS.get(self.model_name, {})
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

        _MAX_RETRIES = 6
        _last_exc: Exception | None = None
        for _attempt in range(1, _MAX_RETRIES + 1):
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
                )
                if is_auth:
                    raise ModelAuthError(
                        provider="groq",
                        model_name=self.model_name,
                        message=msg,
                    ) from exc

                # A 413 payload-too-large is a permanent property of this
                # request, not a transient rate limit — fail fast with a
                # fix-oriented hint instead of routing it through retry/failover.
                if _is_request_too_large(msg, msg_lower):
                    from effgen.models.errors import InvalidRequestError as _IRE
                    raise _IRE(
                        provider="groq",
                        model_name=self.model_name,
                        message=(
                            f"request too large for {self.model_name}: "
                            f"{_redact_groq_org(msg)} — reduce the request "
                            "(fewer/smaller tools or shorter input) or use a "
                            "larger-context model."
                        ),
                    ) from exc

                is_rate = "429" in msg or "rate_limit" in msg_lower or "rate limit" in msg_lower
                is_server = "500" in msg or "503" in msg or "internal" in msg_lower
                is_timeout = "timeout" in msg_lower

                if is_rate:
                    # Raise RateLimitExceeded so the router can failover to another provider
                    from effgen.models._rate_limit import RateLimitExceeded as _RLE
                    raise _RLE(
                        f"Groq rate limit hit for {self.model_name}: {msg}"
                    ) from exc

                if is_server or is_timeout:
                    if _attempt >= _MAX_RETRIES:
                        logger.error("Groq API error after %d retries: %s", _attempt, exc)
                        from effgen.models.errors import ProviderTransientError as _PTE
                        raise _PTE(
                            provider="groq",
                            model_name=self.model_name,
                            status_code=500 if is_server else 0,
                            message=f"Groq API failed after {_MAX_RETRIES} retries: {exc}",
                        ) from exc
                    delay = min(60.0, 2.0 * (2 ** (_attempt - 1)) + random.uniform(0, 0.5))
                    logger.warning(
                        "Groq transient error on attempt %d/%d — retrying in %.1fs: %s",
                        _attempt, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                    continue

                failed_tool_call = _parse_failed_generation_tool_call(msg)
                if tools and "tool_use_failed" in msg_lower and failed_tool_call is not None:
                    logger.warning(
                        "Groq returned tool_use_failed but included a parseable tool call; "
                        "using failed_generation as structured tool call."
                    )
                    return GenerationResult(
                        text="",
                        tokens_used=0,
                        finish_reason="tool_calls",
                        model_name=self.model_name,
                        metadata={
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "provider": "groq",
                            "cost_usd": 0.0,
                            "tool_calls": [failed_tool_call],
                            "provider_error": "tool_use_failed",
                        },
                    )

                logger.error("Groq API call failed: %s", exc)
                raise provider_runtime_error("groq", self.model_name, "generate", exc, message="Groq generation failed") from exc
        else:
            assert _last_exc is not None
            raise provider_runtime_error(
                "groq", self.model_name, "generate", _last_exc,
                message=f"Groq generation failed after {_MAX_RETRIES} retries",
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
                provider="groq",
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        logger.info(
            "Groq generated %d tokens (prompt=%d, completion=%d, cost=$%.6f)",
            total_tokens, prompt_tokens, completion_tokens, cost,
        )
        # Emit span attributes on the current active span
        _set_span_attr(ModelAttrs.PROVIDER, "groq")
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
                "provider": "groq",
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
            raise RuntimeError("GroqAdapter not loaded. Call load() first.")
        if config is None:
            config = GenerationConfig()
        return self._do_generate(prompt, config, tools=tools, messages=messages, **kwargs)

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a response token-by-token from the Groq API.

        Yields:
            str: Successive text chunks from the model.
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("GroqAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        require_vision_support(
            prompt,
            provider="groq",
            model_name=self.model_name,
            supports_vision=GROQ_MODELS.get(self.model_name, {}).get("supports_vision", False),
            hint="Use 'meta-llama/llama-4-scout-17b-16e-instruct' for Groq vision.",
        )

        try:
            from effgen.core.messages import Message

            if isinstance(prompt, Message):
                messages = [self._message_to_groq(prompt)]
            elif isinstance(prompt, list) and prompt and isinstance(prompt[0], Message):
                messages = [self._message_to_groq(m) for m in prompt]
            else:
                messages = [{"role": "user", "content": prompt}]
        except ImportError:
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

        request_params.update(kwargs)

        self._last_stream_tool_calls: list[dict[str, Any]] = []
        self._last_stream_finish_reason: str | None = None

        try:
            with timed_call("groq", self.model_name) as _stream_timer:
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
                    provider="groq",
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
                    provider="groq",
                    model_name=self.model_name,
                    message=msg,
                ) from exc
            if _is_request_too_large(msg, msg_lower):
                from effgen.models.errors import InvalidRequestError as _IRE
                raise _IRE(
                    provider="groq",
                    model_name=self.model_name,
                    message=(
                        f"request too large for {self.model_name}: "
                        f"{_redact_groq_org(msg)} — reduce the request "
                        "(fewer/smaller tools or shorter input) or use a "
                        "larger-context model."
                    ),
                ) from exc
            is_rate = "429" in msg or "rate_limit" in msg_lower or "rate limit" in msg_lower
            if is_rate:
                from effgen.models._rate_limit import RateLimitExceeded as _RLE
                raise _RLE(f"Groq rate limit hit for {self.model_name}: {msg}") from exc
            is_server = "500" in msg or "503" in msg or "internal" in msg_lower
            if is_server:
                from effgen.models.errors import ProviderTransientError as _PTE
                raise _PTE(provider="groq", model_name=self.model_name, status_code=500, message=msg) from exc
            logger.error("Groq streaming failed: %s", exc)
            raise provider_runtime_error("groq", self.model_name, "stream", exc, message="Groq streaming failed") from exc

    # ------------------------------------------------------------------
    # Token counting / context length
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> TokenCount:
        """Estimate token count via tiktoken (cl100k_base, same as OpenAI)."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            count = len(enc.encode(text))
        except Exception:
            count = max(1, len(text) // 4)
        return TokenCount(count=count, model_name=self.model_name)

    def get_context_length(self) -> int:
        """Return the context window size for the loaded model."""
        return GROQ_MODELS.get(self.model_name, {}).get("context", 131_072)

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
        return GROQ_MODELS.get(self.model_name, {}).get("supports_native_tools", False)

    def supports_tool_calling(self) -> bool:
        """Return True if the loaded model supports native tool-calling."""
        return GROQ_MODELS.get(self.model_name, {}).get("supports_native_tools", False)

    def supports_function_calling(self) -> bool:
        """Alias for :meth:`supports_tool_calling`."""
        return self.supports_tool_calling()

    @property
    def supports_streaming(self) -> bool:
        """True if this model supports streaming responses."""
        return GROQ_MODELS.get(self.model_name, {}).get("supports_streaming", True)


# ---------------------------------------------------------------------------
# Self-register with the ProviderRegistry on first import (idempotent)
# ---------------------------------------------------------------------------
def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.groq_models import GROQ_MODELS
        from effgen.models.registry import ProviderRegistry
        ProviderRegistry.register(
            "groq",
            GroqAdapter,
            GROQ_MODELS,
            env_keys=["GROQ_API_KEY"],
            capabilities={Capability.chat, Capability.streaming, Capability.tools, Capability.json_schema, Capability.vision},
            # Free developer tier routes as zero out-of-pocket cost while quota remains.
            # Per-model paid list prices are retained in GROQ_MODELS for tie-break metadata.
            # llama-3.1-8b-instant: $0.05/$0.08; llama-3.3-70b: $0.59/$0.79 per 1M tokens.
            # Pricing verified: https://groq.com/pricing (2026-05-11)
            pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
