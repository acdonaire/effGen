"""
Google Gemini API adapter.

Uses the `google-genai` SDK (google.genai). Supports Gemini Pro, Flash,
Flash-Lite, and Gemma models with:
- Native function calling / tool use
- Thinking budget (extended reasoning for supporting models)
- Multimodal inputs
- Streaming
- Cost tracking
- Rate-limit-aware retry with exponential back-off
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator
from typing import Any

from effgen.models.base import (
    FunctionCallingModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)
from effgen.models.gemini_files import FileRef
from effgen.models.gemini_models import (
    GEMINI_MODEL_ALIASES,
    GEMINI_MODELS,
)

logger = logging.getLogger(__name__)


class GeminiAdapter(FunctionCallingModel):
    """
    Adapter for Google Gemini API models.

    Wraps the `google.genai` SDK and exposes a unified effGen interface.

    Features:
    - Gemini 2.x / 3.x and Gemma 3 / 4 family support
    - Native function calling (OpenAI-format tools auto-converted)
    - thinking_budget / include_thoughts for reasoning models
    - Streaming via generate_content_stream
    - Per-call cost tracking
    - Rate-limit-aware retry (honors Retry-After from 429s)

    Attributes:
        model_name: Gemini model ID (e.g. 'gemini-3.1-flash-lite-preview')
        api_key: Google API key (reads GOOGLE_API_KEY from env if not given)
        safety_settings: Content safety filter settings
    """

    COST_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
        "gemini-1.5-pro": (3.5, 10.5),
        "gemini-1.5-flash": (0.35, 1.05),
        "gemini-pro": (0.5, 1.5),
        "gemini-pro-vision": (0.5, 1.5),
        "gemini-ultra": (10.0, 30.0),
        # Gemini 2.x / 3.x — free tier, cost effectively $0 on free quota
        "gemini-2.0-flash": (0.1, 0.4),
        "gemini-2.5-flash": (0.15, 0.6),
        "gemini-2.5-pro": (1.25, 5.0),
        "gemini-3.1-flash-lite": (0.01, 0.04),
    }

    CONTEXT_LENGTHS: dict[str, int] = {
        "gemini-1.5-pro": 1_000_000,
        "gemini-1.5-flash": 1_000_000,
        "gemini-pro": 32_760,
        "gemini-pro-vision": 16_384,
        "gemini-ultra": 32_760,
        **{m: info["context"] for m, info in GEMINI_MODELS.items()},
        **{
            alias: GEMINI_MODELS[c]["context"]
            for alias, c in GEMINI_MODEL_ALIASES.items()
            if c in GEMINI_MODELS
        },
    }

    MAX_RATE_LIMIT_RETRIES = 7

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        import os
        super().__init__(
            model_name=model_name,
            model_type=ModelType.GEMINI,
            context_length=self.CONTEXT_LENGTHS.get(model_name, 32_760),
        )
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not provided. Set GOOGLE_API_KEY environment "
                "variable or pass api_key parameter."
            )
        self.safety_settings = safety_settings
        self.additional_kwargs = kwargs
        self.client: Any = None
        self.total_cost = 0.0
        self.total_tokens = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialize the google.genai client."""
        try:
            import google.genai as genai  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is not installed. "
                "Install it with: pip install google-genai"
            ) from exc

        try:
            logger.info("Initializing Gemini client for model '%s'…", self.model_name)
            self.client = genai.Client(api_key=self.api_key)
            self._is_loaded = True
            self._metadata = {
                "model_name": self.model_name,
                "context_length": self.get_context_length(),
                "supports_functions": True,
                "supports_streaming": True,
                "supports_multimodal": True,
            }
            logger.info("Gemini client initialized for '%s'", self.model_name)
        except Exception as exc:
            logger.error("Failed to initialize Gemini client: %s", exc)
            raise RuntimeError(f"Gemini initialization failed: {exc}") from exc

    def unload(self) -> None:
        """Release the Gemini client."""
        if self.client is not None:
            logger.info(
                "Closing Gemini client. Total cost: $%.4f, Total tokens: %d",
                self.total_cost,
                self.total_tokens,
            )
            self.client = None
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_supports_thinking(self) -> bool:
        """Return True if the registered model entry has supports_thinking=True."""
        canonical = GEMINI_MODEL_ALIASES.get(self.model_name, self.model_name)
        info = GEMINI_MODELS.get(canonical) or GEMINI_MODELS.get(self.model_name)
        if info is None:
            return False
        return bool(info.get("supports_thinking", False))

    def _model_supports_grounding(self) -> bool:
        """Return True if the registered model entry has supports_grounding=True."""
        canonical = GEMINI_MODEL_ALIASES.get(self.model_name, self.model_name)
        info = GEMINI_MODELS.get(canonical) or GEMINI_MODELS.get(self.model_name)
        if info is None:
            return False
        return bool(info.get("supports_grounding", False))

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost_entry: tuple[float, float] | None = None
        for model_id, costs in self.COST_PER_1M_TOKENS.items():
            if model_id in self.model_name:
                cost_entry = costs
                break
        if cost_entry is None:
            cost_entry = (0.5, 1.5)
        return (prompt_tokens / 1_000_000) * cost_entry[0] + (
            completion_tokens / 1_000_000
        ) * cost_entry[1]

    def _build_config(self, config: GenerationConfig | None) -> Any:
        """Build a google.genai GenerateContentConfig from an effGen GenerationConfig."""
        from google.genai import types  # type: ignore[import]

        if config is None:
            config = GenerationConfig()

        kwargs: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_tokens or 2048,
        }
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences

        # Thinking config — only for models that support it.
        if config.thinking_budget is not None:
            if self._model_supports_thinking():
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=config.thinking_budget,
                    include_thoughts=config.include_thoughts,
                )
            else:
                logger.debug(
                    "Model '%s' does not support thinking_budget; ignoring.",
                    self.model_name,
                )

        # Grounding — attach Google Search tool when requested and supported.
        if config.grounding:
            if self._model_supports_grounding():
                kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            else:
                logger.debug(
                    "Model '%s' does not support grounding; ignoring grounding=True.",
                    self.model_name,
                )

        if self.safety_settings:
            kwargs["safety_settings"] = self.safety_settings

        return types.GenerateContentConfig(**kwargs)

    # Gemini's Schema accepts a strict subset of JSON Schema. Strip extras
    # before sending so the SDK does not raise "Unknown field for Schema".
    _GEMINI_SCHEMA_FIELDS = {
        "type", "description", "enum", "items", "properties",
        "required", "format", "nullable",
    }

    @classmethod
    def _sanitize_schema(cls, schema: Any) -> Any:
        if isinstance(schema, dict):
            cleaned: dict[str, Any] = {}
            for k, v in schema.items():
                if k not in cls._GEMINI_SCHEMA_FIELDS:
                    continue
                if k == "properties" and isinstance(v, dict):
                    cleaned[k] = {
                        name: cls._sanitize_schema(s) for name, s in v.items()
                    }
                elif k == "items":
                    cleaned[k] = cls._sanitize_schema(v)
                elif k == "required":
                    cleaned[k] = list(v) if isinstance(v, (list, tuple)) else v
                else:
                    cleaned[k] = v
            return cleaned
        if isinstance(schema, list):
            return [cls._sanitize_schema(v) for v in schema]
        return schema

    @classmethod
    def _convert_tools_to_genai(cls, tools: list) -> list:
        """Convert effGen / OpenAI-style tool specs to google.genai Tool objects.

        Handles:
        - ``GeminiNativeTool`` subclasses (GoogleSearch, UrlContext, CodeExecution)
        - ``google.genai.types.Tool`` passthrough
        - OpenAI-format dicts with ``type="function"``
        """
        from google.genai import types  # type: ignore[import]

        function_declarations: list[Any] = []
        passthrough: list[Any] = []

        for t in tools:
            # Gemini first-party native tools
            try:
                from effgen.tools.builtin.gemini_native import GeminiNativeTool
                if isinstance(t, GeminiNativeTool):
                    passthrough.append(t.to_gemini_tool())
                    continue
            except ImportError:
                pass

            if isinstance(t, types.Tool):
                passthrough.append(t)
                continue
            if isinstance(t, dict):
                func = t["function"] if t.get("type") == "function" and "function" in t else t
                params = func.get("parameters") or {"type": "object", "properties": {}}
                params = cls._sanitize_schema(params)
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=params,
                    )
                )

        result = list(passthrough)
        if function_declarations:
            result.append(types.Tool(function_declarations=function_declarations))
        return result

    @staticmethod
    def _suggested_retry_delay(exc: Exception) -> float | None:
        for attr in ("retry_delay", "_details", "details"):
            obj = getattr(exc, attr, None)
            if obj is None:
                continue
            try:
                if hasattr(obj, "seconds"):
                    return float(obj.seconds) + float(getattr(obj, "nanos", 0)) / 1e9
                items = obj() if callable(obj) else obj
                for item in items or []:
                    rd = getattr(item, "retry_delay", None)
                    if rd is not None and hasattr(rd, "seconds"):
                        return float(rd.seconds) + float(getattr(rd, "nanos", 0)) / 1e9
            except Exception:
                continue
        msg = str(exc)
        if "Please retry in" in msg:
            try:
                tail = msg.split("Please retry in", 1)[1]
                num = ""
                for ch in tail.strip():
                    if ch.isdigit() or ch == ".":
                        num += ch
                    else:
                        break
                return float(num) if num else None
            except Exception:
                return None
        return None

    def _generate_with_retry(
        self,
        *,
        contents: Any,
        gen_config: Any,
        stream: bool = False,
        tools: list | None = None,
    ) -> Any:
        """Call client.models.generate_content[_stream] with rate-limit retries."""
        if tools:
            gen_config = _inject_tools(gen_config, tools)

        last_exc: Exception | None = None
        for attempt in range(1, self.MAX_RATE_LIMIT_RETRIES + 1):
            try:
                if stream:
                    return self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=contents,
                        config=gen_config,
                    )
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=gen_config,
                )
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                exc_cls = type(exc).__name__
                is_quota = "429" in exc_str or "quota" in exc_str.lower() or "ResourceExhausted" in exc_cls
                is_transient = (
                    any(kw in exc_cls for kw in ("ServiceUnavailable", "DeadlineExceeded", "InternalServerError", "ServerError"))
                    or "503" in exc_str
                    or "500" in exc_str
                    or "UNAVAILABLE" in exc_str
                )
                if not (is_quota or is_transient):
                    raise
                if attempt >= self.MAX_RATE_LIMIT_RETRIES:
                    break
                delay = self._suggested_retry_delay(exc) if is_quota else None
                if delay is None:
                    delay = min(60.0, (2 ** attempt) + random.uniform(0, 0.5))
                else:
                    delay = max(delay + 0.5, 1.0)
                logger.warning(
                    "Gemini transient error on attempt %d/%d: %s — sleeping %.1fs",
                    attempt, self.MAX_RATE_LIMIT_RETRIES, type(exc).__name__, delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _prepare_content(
        self, prompt: str | list[str | dict[str, Any]]
    ) -> str | list:
        if isinstance(prompt, str):
            return prompt
        parts: list[Any] = []
        for item in prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "image" in item:
                import PIL.Image  # type: ignore[import]
                parts.append(PIL.Image.open(item["image"]))
            else:
                parts.append(item)
        return parts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str | list[str | dict[str, Any]],
        config: GenerationConfig | None = None,
        *,
        files: list[FileRef] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate text (or a tool call) from *prompt*.

        Args:
            prompt: Text prompt or multimodal content list.
            config: Generation configuration.
            files: Optional list of :class:`~effgen.models.gemini_files.FileRef`
                objects returned by :func:`~effgen.models.gemini_files.upload_file`.
                Each file is prepended to the request content so the model can
                read it before answering *prompt*.
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")
        if isinstance(prompt, str):
            self.validate_prompt(prompt)

        gen_config = self._build_config(config)
        content = self._prepare_content(prompt)

        # Prepend uploaded files to the content so the model sees them.
        if files:
            from google.genai import types as _gt  # type: ignore[import]
            file_parts: list[Any] = []
            for fref in files:
                sdk_obj = fref.raw
                if sdk_obj is not None:
                    file_parts.append(_gt.Part.from_uri(
                        file_uri=fref.uri,
                        mime_type=fref.mime_type,
                    ))
                else:
                    file_parts.append(_gt.Part.from_uri(
                        file_uri=fref.uri,
                        mime_type=fref.mime_type,
                    ))
            if isinstance(content, list):
                content = file_parts + content
            else:
                content = file_parts + [content]

        raw_tools: list | None = None
        if kwargs.get("tools"):
            raw_tools = self._convert_tools_to_genai(kwargs.pop("tools"))

        try:
            response = self._generate_with_retry(
                contents=content,
                gen_config=gen_config,
                tools=raw_tools,
            )

            generated_text = ""
            tool_calls: list[dict[str, Any]] = []
            thinking_text = ""

            if hasattr(response, "candidates") and response.candidates:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "thought", False):
                        thinking_text += getattr(part, "text", "") or ""
                    elif getattr(part, "text", None):
                        generated_text += part.text
                    # Parallel function calls
                    fc = getattr(part, "function_call", None)
                    if fc and getattr(fc, "name", None):
                        args: dict[str, Any] = {}
                        if hasattr(fc, "args"):
                            try:
                                args = dict(fc.args)
                            except Exception:
                                args = {k: fc.args[k] for k in fc.args}
                        tool_calls.append({"name": fc.name, "arguments": args})
                    # Code execution result — surface output in generated text
                    cer = getattr(part, "code_execution_result", None)
                    if cer is not None:
                        output = getattr(cer, "output", None)
                        if isinstance(output, str) and output:
                            generated_text += output
            else:
                try:
                    generated_text = response.text or ""
                except Exception:
                    pass

            # Emit ALL tool calls as <tool_call> tokens for the agent dispatcher.
            # Parallel function calls: Gemini may return multiple functionCall
            # parts in one response turn.  Append each as a separate tag.
            if tool_calls and "<tool_call>" not in generated_text:
                import json as _json
                tc_blocks = "".join(
                    f"\n<tool_call>{_json.dumps({'name': tc['name'], 'arguments': tc['arguments']})}</tool_call>"
                    for tc in tool_calls
                )
                generated_text = (generated_text + tc_blocks).strip()

            # Token usage
            try:
                um = response.usage_metadata
                prompt_tokens = um.prompt_token_count or 0
                completion_tokens = um.candidates_token_count or 0
                total_tokens = um.total_token_count or (prompt_tokens + completion_tokens)
                thoughts_tokens = getattr(um, "thoughts_token_count", None) or 0
            except AttributeError:
                prompt_tokens = self.count_tokens(
                    prompt if isinstance(prompt, str) else str(prompt)
                ).count
                completion_tokens = self.count_tokens(generated_text).count
                total_tokens = prompt_tokens + completion_tokens
                thoughts_tokens = 0

            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            self.total_cost += cost
            self.total_tokens += total_tokens

            finish_reason = "stop"
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)

            # Extract grounding chunks (URLs + snippets) from the response.
            grounding_chunks: list[dict[str, Any]] = []
            try:
                gmd = None
                if hasattr(response, "candidates") and response.candidates:
                    cand = response.candidates[0]
                    gmd = getattr(cand, "grounding_metadata", None)
                if gmd is None:
                    gmd = getattr(response, "grounding_metadata", None)
                if gmd is not None:
                    raw_chunks = getattr(gmd, "grounding_chunks", None) or []
                    for chunk in raw_chunks:
                        web = getattr(chunk, "web", None)
                        if web is not None:
                            grounding_chunks.append({
                                "url": getattr(web, "uri", None),
                                "title": getattr(web, "title", None),
                            })
                        else:
                            grounding_chunks.append({"raw": str(chunk)})
            except Exception as _ge:
                logger.debug("Could not extract grounding_chunks: %s", _ge)

            metadata: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "thoughts_token_count": thoughts_tokens,
                "cost": cost,
                "total_cost": self.total_cost,
                "safety_ratings": getattr(response, "safety_ratings", None),
                "tool_calls": tool_calls,
                "grounding_chunks": grounding_chunks,
            }
            if thinking_text:
                metadata["thinking"] = thinking_text

            return GenerationResult(
                text=generated_text,
                tokens_used=completion_tokens,
                finish_reason=finish_reason,
                model_name=self.model_name,
                metadata=metadata,
            )

        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise RuntimeError(f"Generation failed: {exc}") from exc

    def generate_stream(
        self,
        prompt: str | list[str | dict[str, Any]],
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream generated text chunks."""
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")
        if isinstance(prompt, str):
            self.validate_prompt(prompt)

        gen_config = self._build_config(config)
        content = self._prepare_content(prompt)

        raw_tools: list | None = None
        if kwargs.get("tools"):
            raw_tools = self._convert_tools_to_genai(kwargs.pop("tools"))

        try:
            response = self._generate_with_retry(
                contents=content,
                gen_config=gen_config,
                stream=True,
                tools=raw_tools,
            )
            for chunk in response:
                try:
                    if chunk.text:
                        yield chunk.text
                except Exception:
                    pass
        except Exception as exc:
            logger.error("Gemini streaming failed: %s", exc)
            raise RuntimeError(f"Streaming generation failed: {exc}") from exc

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate with explicit tool list (convenience wrapper over generate)."""
        return self.generate(prompt, config=config, tools=tools, **kwargs)

    # ------------------------------------------------------------------
    # Capabilities / token counting
    # ------------------------------------------------------------------

    def supports_function_calling(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    def count_tokens(self, text: str) -> TokenCount:
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")
        try:
            response = self.client.models.count_tokens(
                model=self.model_name, contents=text
            )
            return TokenCount(count=response.total_tokens, model_name=self.model_name)
        except Exception as exc:
            logger.warning("Token counting failed: %s. Using approximation.", exc)
            return TokenCount(count=len(text) // 4, model_name=self.model_name)

    def get_context_length(self) -> int:
        return self._context_length

    def get_total_cost(self) -> float:
        return self.total_cost

    def get_total_tokens(self) -> int:
        return self.total_tokens

    def reset_usage_stats(self) -> None:
        self.total_cost = 0.0
        self.total_tokens = 0
        logger.info("Usage statistics reset")

    # Keep the old _create_generation_config name as a shim so any external
    # code that called it directly still works (it's not a public API but
    # let's be safe).
    def _create_generation_config(self, config: GenerationConfig | None = None) -> Any:
        return self._build_config(config)


# ---------------------------------------------------------------------------
# Module-level helper (avoids circular import if ever needed)
# ---------------------------------------------------------------------------

def _inject_tools(gen_config: Any, tools: list) -> Any:
    """Return a copy of *gen_config* with tools injected.

    When the request mixes server-side built-in tools (google_search,
    url_context, code_execution) with user function declarations, the
    Gemini API requires ``tool_config.include_server_side_tool_invocations=True``
    or it returns a 400.
    """
    from google.genai import types  # type: ignore[import]

    d = gen_config.model_dump(exclude_none=True) if hasattr(gen_config, "model_dump") else {}
    d["tools"] = tools

    has_native = False
    has_function_decls = False
    for t in tools:
        if isinstance(t, types.Tool):
            if any(getattr(t, attr, None) is not None for attr in (
                "google_search", "url_context", "code_execution",
                "google_search_retrieval", "retrieval", "computer_use",
            )):
                has_native = True
            if getattr(t, "function_declarations", None):
                has_function_decls = True

    if has_native and has_function_decls:
        existing = d.get("tool_config")
        if isinstance(existing, dict):
            existing.setdefault("include_server_side_tool_invocations", True)
            d["tool_config"] = existing
        else:
            d["tool_config"] = types.ToolConfig(include_server_side_tool_invocations=True)

    return types.GenerateContentConfig(**d)
