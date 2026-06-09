"""
Replicate adapter for effGen.

Replicate's API is async-first: predictions are created, then polled until
complete.  This adapter hides that run-then-poll pattern behind the standard
effGen generate() / generate_stream() surface.

Key behaviours
--------------
- generate()         — creates a prediction, polls with exponential backoff,
                       returns a GenerationResult once status == "succeeded".
- generate_stream()  — uses replicate.stream() (SSE) for models that support it;
                       falls back to polling + output_iterator() otherwise.
- ModelTimeoutError  — raised (not RuntimeError) when polling exceeds timeout.
- compute_seconds    — prediction metrics.predict_time is surfaced in
                       ModelResponse.metadata["compute_seconds"] so callers can
                       track Replicate's per-second billing.
- Native tools       — IBM Granite and similar models that accept a `tools` +
                       `messages` input schema get full native function-calling;
                       others fall back to ReAct.
- Dynamic drift      — Registry is checked at adapter init; unknown models emit
                       a warning directing the user to refresh_models().

Replicate billing
-----------------
Replicate charges per second of GPU compute (predict_time from metrics), not
per token.  The adapter records cost via CostTracker using:
    cost = predict_time * cost_per_second_usd (from registry)

Hardware pricing (typical, 2026-04-29):
  T4 GPU    — ~$0.000225/s
  L40S GPU  — ~$0.000575/s
  A100 GPU  — ~$0.001400/s
  H100 GPU  — ~$0.001974/s

Some hosted models (Anthropic/OpenAI on Replicate) are billed per-token instead;
cost_per_second_usd is 0 for those — per-token pricing from the registry is
recorded instead.
"""

from __future__ import annotations

import json
import logging
import os
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
from effgen.models.errors import (
    BudgetExceededError,
    ModelAuthError,
    ModelNotFoundError,
    ModelTimeoutError,
)
from effgen.models.latency_tracker import timed_call
from effgen.models.replicate_models import (
    REGISTRY_FETCH_DATE,
    REPLICATE_DEFAULT_MODEL,
    REPLICATE_MODELS,
)
from effgen.observability.spans import ModelAttrs
from effgen.observability.tracing import set_span_attribute as _set_span_attr

if TYPE_CHECKING:
    from effgen.models._rate_limit_store import SQLiteRateLimitStore

logger = logging.getLogger(__name__)

_REPLICATE_MODEL_TYPE_VALUE = "replicate"

_TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}
_POLL_INITIAL_DELAY = 1.0   # seconds
_POLL_MAX_DELAY = 30.0      # seconds
_POLL_BACKOFF_FACTOR = 1.5


class _ReplicateModelType:
    """Sentinel so ModelType enum doesn't need patching."""
    value = _REPLICATE_MODEL_TYPE_VALUE


class ReplicateAdapter(BaseModel):
    """
    Adapter for Replicate's inference API.

    Replicate runs models as async predictions (create → poll → result).
    This adapter wraps that pattern so callers use the same generate() /
    generate_stream() interface as every other effGen backend.

    Args:
        model_name: Replicate model ID in ``owner/name`` format, e.g.
            ``"meta/meta-llama-3-8b-instruct"``.  Defaults to
            ``"meta/meta-llama-3-8b-instruct"``.
        api_token: Replicate API token.  Falls back to ``REPLICATE_API_TOKEN``
            env var.
        timeout: Seconds to wait for a prediction to complete before raising
            :class:`~effgen.models.errors.ModelTimeoutError`.  Default 300s.
        poll_interval: Starting poll interval in seconds (grows with backoff).
        max_retries: Retries on transient HTTP errors (not on prediction failure).
        enable_rate_limiting: Wire built-in
            :class:`~effgen.models._rate_limit.RateLimitCoordinator`.
        enable_cost_tracking: Record compute cost in the global
            :class:`~effgen.models._cost.CostTracker`.
        warn_unknown_model: Emit a warning when the model is not in the
            bundled registry.

    Example::

        from effgen.models.replicate_adapter import ReplicateAdapter

        adapter = ReplicateAdapter("meta/meta-llama-3-8b-instruct")
        adapter.load()

        result = adapter.generate("What is the capital of France?")
        print(result.text)
        print("Compute seconds:", result.metadata.get("compute_seconds"))

        for chunk in adapter.generate_stream("Count from 1 to 5."):
            print(chunk, end="", flush=True)

        adapter.unload()
    """

    def __init__(
        self,
        model_name: str = REPLICATE_DEFAULT_MODEL,
        api_token: str | None = None,
        timeout: float = 300.0,
        poll_interval: float = _POLL_INITIAL_DELAY,
        max_retries: int = 4,
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
        warn_unknown_model: bool = True,
        rate_limit_storage: "SQLiteRateLimitStore | None" = None,
        **kwargs: Any,
    ) -> None:
        info = REPLICATE_MODELS.get(model_name)
        if info is None and warn_unknown_model:
            logger.warning(
                "ReplicateAdapter: model '%s' is not in the bundled registry "
                "(registry date: %s).  The model may be new — call "
                "replicate_models.refresh_models() to check for drift.  "
                "Proceeding with default parameters.",
                model_name,
                REGISTRY_FETCH_DATE,
            )
            info = {}

        super().__init__(
            model_name=model_name,
            model_type=_ReplicateModelType(),  # type: ignore[arg-type]
            context_length=(info or {}).get("context", 8_192),
        )

        self._api_token = api_token
        self.timeout = timeout
        self._initial_poll_interval = poll_interval
        self.max_retries = max_retries
        self._extra_kwargs = kwargs
        self._client: Any = None
        self._enable_cost_tracking = enable_cost_tracking
        self._info: dict[str, Any] = info or {}

        self._rate_limiter: RateLimitCoordinator | None = None
        if enable_rate_limiting:
            self._rate_limiter = RateLimitCoordinator(
                provider="replicate",
                model=model_name,
                rpm=60,
                rph=600,
                rpd=10_000,
                tpm=1_000_000,
                tph=10_000_000,
                tpd=100_000_000,
                storage=rate_limit_storage,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the Replicate SDK client.

        Raises:
            RuntimeError: If ``replicate`` is not installed.
            ValueError: If no API token is available.
        """
        try:
            import replicate as _replicate
        except ImportError as exc:
            raise RuntimeError(
                "replicate SDK is not installed.  "
                "Install with: pip install 'effgen[replicate]'"
            ) from exc

        if not (self._api_token or os.getenv("REPLICATE_API_TOKEN")):
            raise ValueError(
                "Replicate API token not found.  Set the REPLICATE_API_TOKEN "
                "environment variable or pass api_token= to ReplicateAdapter."
            )

        self._client = _replicate.Client(
            api_token=self._api_token or os.getenv("REPLICATE_API_TOKEN")
        )
        self._is_loaded = True

        info = self._info
        self._metadata = {
            "model_name": self.model_name,
            "context_length": self.get_context_length(),
            "provider": "replicate",
            "family": info.get("family", ""),
            "organization": info.get("organization", ""),
            "display_name": info.get("display_name", ""),
            "input_schema": info.get("input_schema", "prompt_only"),
            "supports_native_tools": info.get("supports_native_tools", False),
            "supports_streaming": info.get("supports_streaming", True),
            "cost_per_second_usd": info.get("cost_per_second_usd", 0.0),
            "pricing_per_1m_input": info.get("pricing_per_1m_input", 0.0),
            "pricing_per_1m_output": info.get("pricing_per_1m_output", 0.0),
            "registry_fetch_date": REGISTRY_FETCH_DATE,
        }
        logger.info("ReplicateAdapter loaded for model '%s'", self.model_name)

    def unload(self) -> None:
        """Release SDK client resources."""
        self._client = None
        self._is_loaded = False
        logger.info("ReplicateAdapter unloaded")

    # ------------------------------------------------------------------
    # Token counting (approximate — Replicate has no tokenizer endpoint)
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> TokenCount:
        """Approximate token count (4 chars ≈ 1 token)."""
        count = max(1, len(text) // 4)
        return TokenCount(count=count, model_name=self.model_name)

    def get_context_length(self) -> int:
        return self._info.get("context", 8_192)

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    def _build_input(
        self,
        prompt: str,
        config: GenerationConfig,
        system_prompt: str = "You are a helpful assistant.",
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the Replicate input dict from prompt + config."""
        info = self._info
        schema = info.get("input_schema", "prompt_only")
        max_tok = config.max_tokens or 512

        if schema == "messages" and messages is not None:
            # Models that accept an OpenAI-style messages array
            inp: dict[str, Any] = {
                "messages": messages,
                "max_tokens": max_tok,
            }
            sys_key = info.get("system_prompt_key")
            if sys_key and sys_key != "system":
                inp[sys_key] = system_prompt
            if config.temperature is not None:
                inp["temperature"] = config.temperature
            if config.top_p is not None:
                inp["top_p"] = config.top_p
            if config.stop_sequences:
                inp["stop"] = config.stop_sequences
            if config.seed is not None:
                inp["seed"] = config.seed
            if tools and info.get("supports_native_tools"):
                inp["tools"] = tools
                inp["tool_choice"] = "auto"
            return inp

        if schema == "messages":
            # Build messages from prompt string
            msgs: list[dict[str, Any]] = []
            sys_key = info.get("system_prompt_key")
            if sys_key == "system":
                # Some models take system as a top-level key
                pass
            msgs.append({"role": "user", "content": prompt})
            inp = {
                "messages": msgs,
                "max_tokens": max_tok,
            }
            if sys_key and sys_key != "system":
                inp[sys_key] = system_prompt
            elif sys_key == "system":
                inp["system"] = system_prompt
            if config.temperature is not None:
                inp["temperature"] = config.temperature
            if config.top_p is not None:
                inp["top_p"] = config.top_p
            if config.stop_sequences:
                inp["stop"] = config.stop_sequences
            if config.seed is not None:
                inp["seed"] = config.seed
            if tools and info.get("supports_native_tools"):
                inp["tools"] = tools
                inp["tool_choice"] = "auto"
            return inp

        # Default: prompt_only schema (most community models)
        tmpl = info.get("prompt_template")
        if tmpl:
            formatted = tmpl.format(system_prompt=system_prompt, prompt=prompt)
            inp = {"prompt": formatted, "max_tokens": max_tok}
        else:
            inp = {"prompt": prompt, "max_tokens": max_tok}

        if config.temperature is not None:
            inp["temperature"] = config.temperature
        if config.top_p is not None:
            inp["top_p"] = config.top_p
        if config.stop_sequences:
            inp["stop_sequences"] = ",".join(config.stop_sequences)
        if config.seed is not None:
            inp["seed"] = config.seed

        return inp

    # ------------------------------------------------------------------
    # Polling abstraction
    # ------------------------------------------------------------------

    def _poll_prediction(self, prediction: Any) -> Any:
        """
        Poll a Replicate prediction until terminal, applying exponential
        backoff.  Raises ModelTimeoutError if timeout is exceeded.

        Args:
            prediction: A replicate.Prediction object (already created).

        Returns:
            The completed Prediction object.

        Raises:
            ModelTimeoutError: If polling exceeds self.timeout seconds.
            RuntimeError: If the prediction fails or is cancelled.
        """
        deadline = time.monotonic() + self.timeout
        delay = self._initial_poll_interval

        while prediction.status not in _TERMINAL_STATUSES:
            if time.monotonic() >= deadline:
                # Try to cancel the hung prediction gracefully
                try:
                    prediction.cancel()
                except Exception:
                    logger.debug("Failed to cancel hung Replicate prediction", exc_info=True)
                raise ModelTimeoutError(
                    provider="replicate",
                    model_name=self.model_name,
                    timeout_seconds=self.timeout,
                    prediction_id=getattr(prediction, "id", ""),
                )

            time.sleep(delay)
            delay = min(delay * _POLL_BACKOFF_FACTOR, _POLL_MAX_DELAY)

            try:
                prediction.reload()
            except Exception as exc:
                logger.debug("Replicate poll reload error (will retry): %s", exc)

        return prediction

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response synchronously via Replicate.

        Creates a prediction, polls with exponential backoff until complete,
        and returns the assembled text.

        Args:
            prompt: User prompt.
            config: Optional generation config.
            **kwargs: Forwarded to the Replicate prediction input.

        Returns:
            GenerationResult with text, usage, and metadata including
            ``compute_seconds`` (predict_time from Replicate metrics).

        Raises:
            RuntimeError: If not loaded or prediction fails.
            ModelTimeoutError: If prediction exceeds timeout.
            ModelAuthError: On authentication failure.
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("ReplicateAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        if self._rate_limiter is not None:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    logger.debug("Skipping blocking rate-limit acquire (nested event loop)")
                else:
                    loop.run_until_complete(self._rate_limiter.acquire(500))
            except RuntimeError:
                import asyncio
                asyncio.run(self._rate_limiter.acquire(500))

        tools = kwargs.pop("tools", None)
        messages = kwargs.pop("messages", None)
        system_prompt = kwargs.pop("system_prompt", "You are a helpful assistant.")
        inp = self._build_input(prompt, config, system_prompt, tools, messages)
        inp.update(kwargs)

        with timed_call("replicate", self.model_name):
            result = self._do_generate(inp)

        if self._rate_limiter is not None:
            actual = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(actual)

        return result

    def _do_generate(self, inp: dict[str, Any]) -> GenerationResult:
        """Internal: create prediction, poll, assemble GenerationResult."""
        from replicate.exceptions import ReplicateError

        prediction = None
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                prediction = self._client.predictions.create(
                    model=self.model_name,
                    input=inp,
                )
                break
            except ReplicateError as exc:
                last_exc = exc
                if exc.status == 401:
                    raise ModelAuthError(
                        provider="replicate",
                        model_name=self.model_name,
                        message=str(exc),
                    ) from exc
                if exc.status == 404:
                    raise ModelNotFoundError(
                        provider="replicate",
                        model_name=self.model_name,
                        message=str(exc),
                    ) from exc
                if exc.status == 402:
                    raise RuntimeError(
                        "Replicate account has insufficient credits.  "
                        "Add billing at https://replicate.com/account/billing#billing"
                    ) from exc
                if exc.status in (429, 500, 503) and attempt < self.max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "Replicate API error (attempt %d/%d, status %s): %s — "
                        "retrying in %ds",
                        attempt, self.max_retries, exc.status, exc.detail, wait,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Replicate API error for model '{self.model_name}': {exc}"
                ) from exc
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise provider_runtime_error(
                    "replicate", self.model_name, "generate", exc,
                    message=f"Replicate request failed for '{self.model_name}'",
                ) from exc

        if prediction is None:
            raise RuntimeError(
                f"Replicate prediction could not be created for '{self.model_name}' "
                f"after {self.max_retries} attempts: {last_exc}"
            )

        # Poll until complete
        prediction = self._poll_prediction(prediction)

        if prediction.status == "failed":
            raise RuntimeError(
                f"Replicate prediction failed for '{self.model_name}': "
                f"{prediction.error}"
            )
        if prediction.status == "canceled":
            raise RuntimeError(
                f"Replicate prediction was cancelled for '{self.model_name}'"
            )

        # Assemble output text
        output = prediction.output or []
        if isinstance(output, list):
            text = "".join(str(t) for t in output)
        else:
            text = str(output)

        # Tool call extraction (Granite / OpenAI-compatible on Replicate)
        tool_calls = self._extract_tool_calls(text, output)

        # Metrics
        metrics = prediction.metrics or {}
        predict_time: float = metrics.get("predict_time", 0.0)
        input_tokens: int = int(metrics.get("input_token_count", 0))
        output_tokens: int = int(metrics.get("output_token_count", 0))
        total_tokens = input_tokens + output_tokens

        # Cost tracking
        cost_usd = 0.0
        info = self._info
        cost_per_sec = info.get("cost_per_second_usd", 0.0)
        if cost_per_sec and predict_time:
            cost_usd = predict_time * cost_per_sec
        elif info.get("pricing_per_1m_input") and input_tokens:
            cost_usd = (
                input_tokens * info["pricing_per_1m_input"] / 1_000_000
                + output_tokens * info.get("pricing_per_1m_output", 0.0) / 1_000_000
            )

        if self._enable_cost_tracking and cost_usd:
            try:
                CostTracker.get().record(
                    provider="replicate",
                    model=self.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
            except BudgetExceededError:
                raise
            except Exception:
                logger.debug("CostTracker recording failed for Replicate", exc_info=True)

        metadata = {
            "provider": "replicate",
            "model_name": self.model_name,
            "prediction_id": prediction.id,
            "compute_seconds": predict_time,
            "total_time": metrics.get("total_time", 0.0),
            "time_to_first_token": metrics.get("time_to_first_token", 0.0),
            "tokens_per_second": metrics.get("tokens_per_second", 0.0),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # Canonical aliases (OpenAI-style) so downstream token/cost
            # accounting reads the same keys across every provider.
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "cost_usd": cost_usd,
            "cost_per_second_usd": cost_per_sec,
            "tool_calls": tool_calls,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }

        _set_span_attr(ModelAttrs.PROVIDER, "replicate")
        _set_span_attr(ModelAttrs.NAME, self.model_name)
        _set_span_attr(ModelAttrs.INPUT_TOKENS, input_tokens)
        _set_span_attr(ModelAttrs.OUTPUT_TOKENS, output_tokens)
        _set_span_attr(ModelAttrs.COST_USD, float(cost_usd or 0.0))
        _set_span_attr(ModelAttrs.OUTCOME, "ok")

        return GenerationResult(
            text=text,
            tokens_used=total_tokens,
            finish_reason=normalize_finish_reason("stop"),
            model_name=self.model_name,
            metadata=metadata,
        )

    def _extract_tool_calls(
        self, text: str, raw_output: Any
    ) -> list[dict[str, Any]]:
        """
        Extract tool calls from native tool-capable models.

        IBM Granite and similar models return structured JSON for tool calls
        when the messages+tools input schema is used.  The output may be a
        list with a dict element containing a ``tool_calls`` key, or the text
        may be a JSON string.
        """
        if not self._info.get("supports_native_tools"):
            return []

        # Try to parse a structured list output (Granite format)
        if isinstance(raw_output, list):
            for item in raw_output:
                if isinstance(item, dict) and "tool_calls" in item:
                    tcs = item["tool_calls"]
                    if isinstance(tcs, list):
                        return tcs

        # Try to parse text as JSON
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    return parsed["tool_calls"]
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "tool_calls" in item:
                            return item["tool_calls"]
            except (json.JSONDecodeError, ValueError):
                pass

        return []

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream tokens from Replicate via SSE.

        For models that support streaming (``supports_streaming=True`` in the
        registry), this uses Replicate's SSE stream endpoint which delivers
        token-by-token output in real time.  For non-streaming models, it falls
        back to the polling-based output_iterator.

        Args:
            prompt: User prompt.
            config: Optional generation config.
            **kwargs: Forwarded to the Replicate prediction input.

        Yields:
            str: Token/chunk strings as they arrive.

        Raises:
            RuntimeError: If not loaded or prediction fails.
            ModelTimeoutError: If timeout exceeded before first token.
            ModelAuthError: On authentication failure.
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("ReplicateAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        tools = kwargs.pop("tools", None)
        messages = kwargs.pop("messages", None)
        system_prompt = kwargs.pop("system_prompt", "You are a helpful assistant.")
        inp = self._build_input(prompt, config, system_prompt, tools, messages)
        inp.update(kwargs)

        supports_streaming = self._info.get("supports_streaming", True)

        with timed_call("replicate", self.model_name) as _stream_timer:
            _first_token = True
            src = self._stream_sse(inp) if supports_streaming else self._stream_poll(inp)
            for token in src:
                if _first_token:
                    _stream_timer.mark_first_token()
                    _first_token = False
                yield token

    def _stream_sse(self, inp: dict[str, Any]) -> Iterator[str]:
        """Stream via Replicate SSE (real token-by-token)."""
        from replicate.exceptions import ReplicateError

        try:
            for event in self._client.stream(self.model_name, input=inp):
                # event is a ServerSentEvent; its string representation is the token
                yield str(event)
        except ReplicateError as exc:
            if exc.status == 401:
                raise ModelAuthError(
                    provider="replicate",
                    model_name=self.model_name,
                    message=str(exc),
                ) from exc
            if exc.status == 404:
                raise ModelNotFoundError(
                    provider="replicate",
                    model_name=self.model_name,
                    message=str(exc),
                ) from exc
            if exc.status == 402:
                raise RuntimeError(
                    "Replicate account has insufficient credits.  "
                    "Add billing at https://replicate.com/account/billing#billing"
                ) from exc
            raise RuntimeError(
                f"Replicate streaming error for '{self.model_name}': {exc}"
            ) from exc

    def _stream_poll(self, inp: dict[str, Any]) -> Iterator[str]:
        """Stream via polling output_iterator (non-SSE models)."""
        from replicate.exceptions import ReplicateError

        try:
            prediction = self._client.predictions.create(
                model=self.model_name,
                input=inp,
            )
        except ReplicateError as exc:
            if exc.status == 401:
                raise ModelAuthError(
                    provider="replicate",
                    model_name=self.model_name,
                    message=str(exc),
                ) from exc
            if exc.status == 404:
                raise ModelNotFoundError(
                    provider="replicate",
                    model_name=self.model_name,
                    message=str(exc),
                ) from exc
            raise RuntimeError(
                f"Replicate prediction creation failed for '{self.model_name}': {exc}"
            ) from exc

        deadline = time.monotonic() + self.timeout
        delay = self._initial_poll_interval

        previous_output: list[Any] = []
        while prediction.status not in _TERMINAL_STATUSES:
            if time.monotonic() >= deadline:
                try:
                    prediction.cancel()
                except Exception:
                    logger.debug("Failed to cancel hung Replicate prediction", exc_info=True)
                raise ModelTimeoutError(
                    provider="replicate",
                    model_name=self.model_name,
                    timeout_seconds=self.timeout,
                    prediction_id=getattr(prediction, "id", ""),
                )

            time.sleep(delay)
            delay = min(delay * _POLL_BACKOFF_FACTOR, _POLL_MAX_DELAY)

            try:
                prediction.reload()
            except Exception as exc:
                logger.debug("Replicate stream poll error: %s", exc)
                continue

            output = prediction.output or []
            new_tokens = output[len(previous_output):]
            for token in new_tokens:
                yield str(token)
            previous_output = list(output)

        if prediction.status == "failed":
            raise RuntimeError(
                f"Replicate prediction failed for '{self.model_name}': "
                f"{prediction.error}"
            )

        # Flush any remaining tokens
        output = prediction.output or []
        new_tokens = output[len(previous_output):]
        for token in new_tokens:
            yield str(token)

    # ------------------------------------------------------------------
    # Async generate
    # ------------------------------------------------------------------

    async def async_generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Async version of generate() — runs blocking poll in thread pool."""
        import asyncio
        import functools

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self.generate, prompt, config, **kwargs),
        )

    # ------------------------------------------------------------------
    # Tool-call generate (native tools on supported models)
    # ------------------------------------------------------------------

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[Any],
        config: GenerationConfig | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate with native tool-calling on supported models.

        For models that advertise ``supports_native_tools=True`` in the registry
        (e.g. IBM Granite, Claude/GPT via Replicate hosted), the tools list is
        passed directly to the model's input schema.

        For unsupported models, this raises ``NotImplementedError`` — use an
        Agent with strategy="react" instead.

        Args:
            prompt: User prompt / task description.
            tools: List of tool dicts (OpenAI function-calling schema) or effGen
                Tool objects.
            config: Optional generation config.
            messages: Full conversation history; overrides prompt if provided.
            **kwargs: Forwarded to the prediction input.

        Returns:
            GenerationResult with ``tool_calls`` populated.
        """
        if not self._info.get("supports_native_tools"):
            raise NotImplementedError(
                f"Model '{self.model_name}' does not support native tool calling.  "
                f"Use an Agent with strategy='react' instead, or switch to "
                f"a tool-capable model like 'ibm-granite/granite-3.3-8b-instruct'."
            )

        # Normalise tools to OpenAI function-calling schema
        openai_tools: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                openai_tools.append(t if "type" in t else {"type": "function", "function": t})
            else:
                # effGen Tool object
                try:
                    schema = t.metadata.to_json_schema()
                    openai_tools.append({"type": "function", "function": schema})
                except AttributeError:
                    openai_tools.append({"type": "function", "function": str(t)})

        return self.generate(
            prompt,
            config,
            tools=openai_tools,
            messages=messages,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def supports_tool_calling(self) -> bool:
        return bool(self._info.get("supports_native_tools", False))

    def supports_streaming(self) -> bool:
        return bool(self._info.get("supports_streaming", True))

    def get_cost_per_second(self) -> float:
        return float(self._info.get("cost_per_second_usd", 0.0))


# ---------------------------------------------------------------------------
# Self-register with the ProviderRegistry on first import (idempotent)
# ---------------------------------------------------------------------------
def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.registry import ProviderRegistry
        from effgen.models.replicate_models import REPLICATE_MODELS
        ProviderRegistry.register(
            "replicate",
            ReplicateAdapter,
            REPLICATE_MODELS,
            env_keys=["REPLICATE_API_TOKEN"],
            capabilities={Capability.chat, Capability.streaming, Capability.vision},
            # No free tier; pay-per-token for LLM models. Provider default = cheapest hosted LLM.
            # deepseek-r1: $3.75/$10.00; claude-3.7-sonnet: $3.00/$15.00 per 1M tokens.
            # Pricing verified: https://replicate.com/pricing (2026-05-11)
            pricing={"input_per_1m": 0.8, "output_per_1m": 4.0, "free_tier": False},
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
