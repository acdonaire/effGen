"""
Anthropic Claude API adapter.

Supports:
- Claude 4.x (Opus 4.7, Sonnet 4.6, Haiku 4.5) and Claude 3.x legacy
- Extended thinking via GenerationConfig.thinking
- redacted_thinking block preservation for multi-turn correctness
- Tool use API with parallel tool calls
- Prompt caching via cache_control markers
- Streaming responses
- Cost tracking
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

from effgen.models.anthropic_models import (
    get_context_length,
    get_cost_per_million,
    get_model_info,
    supports_thinking,
)
from effgen.models.base import (
    FunctionCallingModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)

logger = logging.getLogger(__name__)

# Anthropic content block types that must be preserved verbatim on re-submit
_PRESERVE_BLOCK_TYPES = {"thinking", "redacted_thinking", "tool_use"}


def _block_to_dict(block: Any) -> dict:
    """Convert an Anthropic SDK content block to a plain dict for serialization."""
    if isinstance(block, dict):
        return block
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "thinking":
        return {"type": "thinking", "thinking": block.thinking}
    if btype == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    if btype == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    # Fallback: attempt attribute-based serialization
    d = {"type": btype}
    for attr in ("text", "thinking", "data", "id", "name", "input"):
        val = getattr(block, attr, None)
        if val is not None:
            d[attr] = val
    return d


class AnthropicAdapter(FunctionCallingModel):
    """
    Adapter for Anthropic Claude API models.

    Supports Claude 4.x (Opus 4.7, Sonnet 4.6, Haiku 4.5) and legacy Claude 3.x.
    Extended thinking, redacted_thinking preservation, tool use, prompt caching,
    and streaming are all supported.

    Attributes:
        model_name: Anthropic model identifier (e.g., 'claude-opus-4-7')
        api_key: Anthropic API key (reads from ANTHROPIC_API_KEY env var if not provided)
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        model_name: str = "claude-opus-4-7",
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_type=ModelType.ANTHROPIC,
            context_length=get_context_length(model_name),
        )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY "
                "environment variable or pass api_key parameter."
            )

        self.max_retries = max_retries
        self.timeout = timeout
        self.additional_kwargs = kwargs

        self.client = None
        self.total_cost = 0.0
        self.total_tokens = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def load(self) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "Anthropic package is not installed. Install it with: pip install anthropic"
            ) from e

        try:
            logger.info(f"Initializing Anthropic client for model '{self.model_name}'...")

            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            client_kwargs.update(self.additional_kwargs)
            self.client = Anthropic(**client_kwargs)
            self._is_loaded = True

            info = get_model_info(self.model_name)
            self._metadata = {
                "model_name": self.model_name,
                "context_length": self.get_context_length(),
                "supports_tools": info.get("supports_native_tools", True),
                "supports_streaming": True,
                "supports_vision": info.get("supports_vision", False),
                "supports_thinking": info.get("supports_thinking", False),
                "supports_prompt_caching": info.get("supports_prompt_caching", False),
            }

            logger.info(f"Anthropic client initialized for '{self.model_name}'")

        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise RuntimeError(f"Anthropic initialization failed: {e}") from e

    def unload(self) -> None:
        if self.client is not None:
            logger.info(
                f"Closing Anthropic client. Total cost: ${self.total_cost:.4f}, "
                f"Total tokens: {self.total_tokens}"
            )
            self.client.close()
            self.client = None
        self._is_loaded = False

    # ── Request building ───────────────────────────────────────────────────

    @staticmethod
    def _build_content(prompt: str | list) -> str | list[dict]:
        """Build Anthropic content from a prompt (text or multimodal list)."""
        if isinstance(prompt, str):
            return prompt

        content: list[dict] = []
        for item in prompt:
            if isinstance(item, str):
                content.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                if "type" in item:
                    content.append(item)
                elif "image_url" in item:
                    content.append({
                        "type": "image",
                        "source": {"type": "url", "url": item["image_url"]},
                    })
                else:
                    content.append({"type": "text", "text": str(item)})
            else:
                content.append({"type": "text", "text": str(item)})
        return content

    def _build_request(
        self,
        prompt: str | list,
        config: GenerationConfig,
        system_prompt: str | None,
        tools: list[dict] | None,
        extra_kwargs: dict,
    ) -> dict:
        """Assemble the full request dict for messages.create."""
        request: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": config.max_tokens or 4096,
            "messages": [{"role": "user", "content": self._build_content(prompt)}],
        }

        # Anthropic does not support top_k or top_p at API level for all models;
        # only include them when non-default to avoid unexpected 400s on newer models.
        if config.temperature != 0.7:
            request["temperature"] = config.temperature
        if config.top_p != 0.9:
            request["top_p"] = config.top_p
        # top_k is supported by some Anthropic models but not all; omit by default.

        if system_prompt:
            request["system"] = system_prompt

        if config.stop_sequences:
            request["stop_sequences"] = config.stop_sequences

        # Extended thinking
        if config.thinking is not None:
            if supports_thinking(self.model_name):
                request["thinking"] = config.thinking
                # When thinking is enabled temperature must be 1 (Anthropic requirement)
                request["temperature"] = 1.0
            else:
                logger.debug(
                    f"Model '{self.model_name}' does not support thinking; "
                    "GenerationConfig.thinking ignored."
                )

        if tools:
            request["tools"] = tools

        request.update(extra_kwargs)
        return request

    # ── Response parsing ───────────────────────────────────────────────────

    def _parse_response(self, response: Any) -> tuple[str, list, list, list]:
        """
        Parse an Anthropic messages response.

        Returns:
            (generated_text, thinking_blocks, redacted_thinking_blocks, raw_content_blocks)

        raw_content_blocks contains ALL blocks as plain dicts and must be preserved
        verbatim when building the next assistant message in multi-turn conversations.
        Stripping redacted_thinking blocks causes a 400 on the following API call.
        """
        generated_text = ""
        thinking_blocks: list[str] = []
        redacted_thinking_blocks: list[dict] = []
        raw_content_blocks: list[dict] = []

        for block in response.content:
            d = _block_to_dict(block)
            raw_content_blocks.append(d)

            btype = d.get("type")
            if btype == "text":
                generated_text += d.get("text", "")
            elif btype == "thinking":
                thinking_blocks.append(d.get("thinking", ""))
            elif btype == "redacted_thinking":
                # Must be preserved on re-submit — do not strip
                redacted_thinking_blocks.append(d)

        return generated_text, thinking_blocks, redacted_thinking_blocks, raw_content_blocks

    def _parse_usage(self, response: Any) -> tuple[int, int]:
        return response.usage.input_tokens, response.usage.output_tokens

    # ── Cost ──────────────────────────────────────────────────────────────

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost_pm, output_cost_pm = get_cost_per_million(self.model_name)
        return (prompt_tokens / 1_000_000) * input_cost_pm + (completion_tokens / 1_000_000) * output_cost_pm

    def _record_usage(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = self._calculate_cost(prompt_tokens, completion_tokens)
        self.total_cost += cost
        self.total_tokens += prompt_tokens + completion_tokens
        return cost

    # ── Generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str | list,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate text from a prompt.

        When GenerationConfig.thinking is set (e.g., {"type": "enabled",
        "budget_tokens": 8000}), extended thinking is requested and the
        thinking trace is surfaced in result.metadata["thinking"].

        For multi-turn conversations, pass the assistant's raw_content_blocks
        (from result.metadata["raw_content_blocks"]) as the assistant message
        content in subsequent calls to preserve redacted_thinking blocks.
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        self.validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        try:
            request = self._build_request(prompt, config, system_prompt, None, kwargs)
            response = self.client.messages.create(**request)

            text, thinking, redacted, raw_blocks = self._parse_response(response)
            prompt_tokens, completion_tokens = self._parse_usage(response)
            cost = self._record_usage(prompt_tokens, completion_tokens)

            logger.info(
                f"Generated {completion_tokens} tokens. "
                f"Cost: ${cost:.4f}. Total cost: ${self.total_cost:.4f}"
            )

            metadata: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": cost,
                "total_cost": self.total_cost,
                # Preserve ALL content blocks for multi-turn re-submission.
                # Include this list verbatim as the assistant message content
                # on the next turn — stripping redacted_thinking causes 400.
                "raw_content_blocks": raw_blocks,
            }

            if thinking:
                metadata["thinking"] = thinking
            if redacted:
                metadata["redacted_thinking"] = redacted

            return GenerationResult(
                text=text,
                tokens_used=completion_tokens,
                finish_reason=response.stop_reason,
                model_name=self.model_name,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e

    # ── Generate stream ───────────────────────────────────────────────────

    def generate_stream(
        self,
        prompt: str | list,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Generate text with streaming output.

        Yields text chunks. Thinking deltas are consumed internally and not
        yielded; use generate() if you need the thinking trace.
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        self.validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        try:
            request = self._build_request(prompt, config, system_prompt, None, kwargs)
            with self.client.messages.stream(**request) as stream:
                yield from stream.text_stream

        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}") from e

    # ── Generate with tools ──────────────────────────────────────────────

    def generate_with_tools(
        self,
        prompt: str | list,
        tools: list[dict[str, Any]],
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate text with tool use support.

        Tool use results are surfaced in result.metadata["tool_uses"].
        Multiple parallel tool_use blocks are supported.
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        self.validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        try:
            request = self._build_request(prompt, config, system_prompt, tools, kwargs)
            response = self.client.messages.create(**request)

            text, thinking, redacted, raw_blocks = self._parse_response(response)
            prompt_tokens, completion_tokens = self._parse_usage(response)
            cost = self._record_usage(prompt_tokens, completion_tokens)

            tool_uses = [
                {"id": b["id"], "name": b["name"], "input": b["input"]}
                for b in raw_blocks
                if b.get("type") == "tool_use"
            ]

            metadata: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": cost,
                "total_cost": self.total_cost,
                "tool_uses": tool_uses,
                "raw_content_blocks": raw_blocks,
            }

            if thinking:
                metadata["thinking"] = thinking
            if redacted:
                metadata["redacted_thinking"] = redacted

            return GenerationResult(
                text=text,
                tokens_used=completion_tokens,
                finish_reason=response.stop_reason,
                model_name=self.model_name,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Anthropic API call with tools failed: {e}")
            raise RuntimeError(f"Generation with tools failed: {e}") from e

    # ── Multi-turn helpers ────────────────────────────────────────────────

    @staticmethod
    def build_assistant_message(result: GenerationResult) -> dict:
        """
        Build an assistant message dict from a GenerationResult for multi-turn use.

        Uses raw_content_blocks (which includes redacted_thinking) from metadata.
        If raw_content_blocks is absent (e.g., from an older result), falls back
        to a plain text block.

        Example::

            history = [{"role": "user", "content": "Hello"}]
            result = adapter.generate_with_history(history)
            history.append(adapter.build_assistant_message(result))
            history.append({"role": "user", "content": "Follow-up question"})
        """
        raw_blocks = (result.metadata or {}).get("raw_content_blocks")
        if raw_blocks:
            return {"role": "assistant", "content": raw_blocks}
        return {"role": "assistant", "content": result.text}

    def generate_with_history(
        self,
        messages: list[dict],
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate a response given a full conversation history.

        messages: list of {"role": "user"|"assistant", "content": str|list}
        When an assistant message was produced with thinking enabled, pass
        its content as the raw_content_blocks list (from build_assistant_message)
        to preserve redacted_thinking blocks and avoid 400 errors.
        """
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        if config is None:
            config = GenerationConfig()

        try:
            request: dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": config.max_tokens or 4096,
                "messages": messages,
            }

            if config.temperature != 0.7:
                request["temperature"] = config.temperature
            if config.top_p != 0.9:
                request["top_p"] = config.top_p
            if system_prompt:
                request["system"] = system_prompt
            if config.stop_sequences:
                request["stop_sequences"] = config.stop_sequences

            if config.thinking is not None:
                if supports_thinking(self.model_name):
                    request["thinking"] = config.thinking
                    request["temperature"] = 1.0

            request.update(kwargs)
            response = self.client.messages.create(**request)

            text, thinking, redacted, raw_blocks = self._parse_response(response)
            prompt_tokens, completion_tokens = self._parse_usage(response)
            cost = self._record_usage(prompt_tokens, completion_tokens)

            metadata: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": cost,
                "total_cost": self.total_cost,
                "raw_content_blocks": raw_blocks,
            }
            if thinking:
                metadata["thinking"] = thinking
            if redacted:
                metadata["redacted_thinking"] = redacted

            return GenerationResult(
                text=text,
                tokens_used=completion_tokens,
                finish_reason=response.stop_reason,
                model_name=self.model_name,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Anthropic multi-turn call failed: {e}")
            raise RuntimeError(f"Multi-turn generation failed: {e}") from e

    # ── Capabilities ──────────────────────────────────────────────────────

    def supports_function_calling(self) -> bool:
        return get_model_info(self.model_name).get("supports_native_tools", True)

    def supports_tool_calling(self) -> bool:
        return self.supports_function_calling()

    # ── Token counting ────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> TokenCount:
        if not self._is_loaded:
            raise RuntimeError("Client not initialized. Call load() first.")

        try:
            response = self.client.messages.count_tokens(
                model=self.model_name,
                messages=[{"role": "user", "content": text}],
            )
            return TokenCount(count=response.input_tokens, model_name=self.model_name)
        except Exception as e:
            logger.warning(f"Token counting API failed: {e}. Using approximation.")
            return TokenCount(count=len(text) // 4, model_name=self.model_name)

    # ── Context / usage ───────────────────────────────────────────────────

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
