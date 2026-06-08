"""
Tool calling strategies for the effGen framework.

This module provides a strategy abstraction for how agents invoke tools:
- ReActStrategy: Parse "Action:" / "Action Input:" from free-text (legacy default)
- NativeFunctionCallingStrategy: Use model's native tool/function calling
- HybridStrategy: Try native first, fall back to ReAct on parse failure

The agent selects the appropriate strategy based on model capabilities
and the ``tool_calling_mode`` setting in AgentConfig.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from effgen.observability import get_logger as _get_obs_logger

logger = logging.getLogger(__name__)
_obs_log = _get_obs_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolCallResult:
    """Unified result from parsing a model response for tool calls.

    Attributes:
        tool_name: Name of the tool to invoke (None if final answer).
        arguments: Parsed arguments dict for the tool.
        raw_text: The original model response text.
        thought: Extracted reasoning / chain-of-thought text.
        final_answer: If the model produced a final answer instead of a tool call.
        is_tool_call: True when a valid tool call was extracted.
    """
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    thought: str | None = None
    final_answer: str | None = None
    is_tool_call: bool = False


@dataclass
class ToolDefinition:
    """JSON Schema tool definition for native function calling.

    Mirrors the OpenAI tools format used by most model providers.
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI-style tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic-style tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


# ---------------------------------------------------------------------------
# Helper: convert BaseTool metadata to ToolDefinition
# ---------------------------------------------------------------------------

def tools_to_definitions(tools: list) -> list[ToolDefinition]:
    """Convert a list of BaseTool instances to ToolDefinition objects.

    Args:
        tools: List of BaseTool instances.

    Returns:
        List of ToolDefinition with JSON Schema parameters.
    """
    definitions: list[ToolDefinition] = []
    for tool in tools:
        meta = tool.metadata
        schema = meta.to_json_schema()
        definitions.append(ToolDefinition(
            name=schema["name"],
            description=schema["description"],
            parameters=schema["parameters"],
        ))
    return definitions


# ---------------------------------------------------------------------------
# Abstract strategy
# ---------------------------------------------------------------------------

class ToolCallingStrategy(ABC):
    """Abstract base class for tool calling strategies."""

    @abstractmethod
    def parse_response(self, text: str, tools: dict[str, Any] | None = None) -> ToolCallResult:
        """Parse a model response and extract tool call or final answer.

        Args:
            text: Raw model response text.
            tools: Dict mapping tool name -> tool object (for validation).

        Returns:
            ToolCallResult with extracted information.
        """

    @abstractmethod
    def format_tools_for_prompt(self, tools: list) -> Any:
        """Prepare tool information for the model.

        For ReAct this returns a text description; for native calling it
        returns JSON Schema definitions.

        Args:
            tools: List of BaseTool instances.

        Returns:
            Formatted tool information (str or list[dict]).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""


# ---------------------------------------------------------------------------
# ReAct Strategy — extracted from Agent._parse_react_response()
# ---------------------------------------------------------------------------

class ReActStrategy(ToolCallingStrategy):
    """Parse tool calls from ReAct-formatted free text.

    This strategy is the legacy default. It extracts ``Thought:``,
    ``Action:``, ``Action Input:``, and ``Final Answer:`` fields from
    the model's textual output using regex patterns.
    """

    @property
    def name(self) -> str:
        return "react"

    # -- Shared JSON-cleaning helpers (also used by Agent) -----------------

    @staticmethod
    def clean_json_input(raw: str) -> str:
        """Clean malformed JSON commonly produced by SLMs.

        Handles:
        - Markdown-wrapped JSON (```json ... ```)
        - Trailing commas  ({"key": "val",})
        - Unquoted keys    ({expression: "2+2"})
        """
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r'^```(?:json|JSON)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)
            text = text.strip()

        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        # Quote unquoted keys
        text = re.sub(
            r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:',
            r' "\1":',
            text,
        )

        return text

    # -- Core parsing ------------------------------------------------------

    def parse_response(self, text: str, tools: dict[str, Any] | None = None) -> ToolCallResult:
        """Parse ReAct formatted response.

        This is extracted from ``Agent._parse_react_response()`` with the
        same logic and patterns, returning a ``ToolCallResult`` instead of
        a plain dict.
        """
        result = ToolCallResult(raw_text=text)

        if not text or not isinstance(text, str):
            logger.warning(f"Invalid response text for parsing: {type(text)}")
            return result

        try:
            # --- Final answer (highest priority) ---
            final_patterns = [
                r"Final Answer:\s*(.+)",
                r"^Answer:\s*(.+)",
                r"^The answer is:\s*(.+)",
            ]
            for pattern in final_patterns:
                try:
                    final_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    if final_match:
                        answer = final_match.group(1).strip()
                        answer = re.split(
                            r'\n(?:Question|Thought|Action|Observation|Human):',
                            answer, maxsplit=1,
                        )[0].strip()
                        # Strip trailing hallucinated follow-up questions
                        trailing = re.search(r'([.!?])[\s]*[A-Z][^.!?]*\?', answer)
                        if trailing:
                            answer = answer[:trailing.start() + 1].strip()
                        result.final_answer = answer
                        logger.debug(f"Extracted final answer: {answer[:100]}...")
                        return result
                except Exception as e:
                    logger.warning(f"Error matching final answer pattern '{pattern}': {e}")
                    continue

            # --- Thought ---
            thought_patterns = [
                r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer|Question):|$)",
                r"Thought:\s*(.+?)(?:\n\n|\n[A-Z]|$)",
            ]
            for pattern in thought_patterns:
                try:
                    thought_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if thought_match:
                        result.thought = thought_match.group(1).strip()
                        break
                except Exception as e:
                    logger.warning(f"Error matching thought pattern '{pattern}': {e}")
                    continue

            # --- Action ---
            action_patterns = [
                r"Action:\s*([^\n]+)",
                r"Tool:\s*([^\n]+)",
                r"Use tool:\s*([^\n]+)",
            ]
            for pattern in action_patterns:
                try:
                    action_match = re.search(pattern, text, re.IGNORECASE)
                    if action_match:
                        action = action_match.group(1).strip()
                        action = action.replace('"', '').replace("'", "")

                        # "Action: Final Answer" → treat as final answer
                        if action.lower() in ["final answer", "finalanswer", "answer"]:
                            logger.debug(f"Action '{action}' detected as Final Answer indicator")
                            same_line = re.search(
                                r"Action:\s*Final\s*Answer[: \t]+([^\n]+)",
                                text, re.IGNORECASE,
                            )
                            if same_line:
                                answer_text = same_line.group(1).strip()
                                if answer_text:
                                    result.final_answer = answer_text
                                    return result

                            ai_match = re.search(
                                r"Action\s*Input:\s*(.+?)(?:\n|$)",
                                text, re.IGNORECASE,
                            )
                            if ai_match:
                                answer_text = ai_match.group(1).strip()
                                if answer_text and not answer_text.startswith(("{", "[")):
                                    result.final_answer = answer_text
                                    return result
                            break

                        # Handle function-call format: tool_name(args)
                        func_call_match = re.match(r'^(\w+)\s*\((.+)\)$', action, re.DOTALL)
                        if func_call_match:
                            tool_name = func_call_match.group(1).strip()
                            embedded_args = func_call_match.group(2).strip().strip('"\'')
                            result.tool_name = tool_name
                            result.is_tool_call = True
                            # Parse embedded args as JSON or plain text
                            try:
                                result.arguments = json.loads(self.clean_json_input(embedded_args))
                                if not isinstance(result.arguments, dict):
                                    result.arguments = {}
                            except (json.JSONDecodeError, TypeError):
                                pass
                            # Store raw for agent to handle via _map_input_to_parameters
                            result.raw_text = text
                        else:
                            result.tool_name = action
                            result.is_tool_call = True
                        break
                except Exception as e:
                    logger.warning(f"Error matching action pattern '{pattern}': {e}")
                    continue

            # --- Action Input (skip if already set from function-call style) ---
            if result.is_tool_call and not result.arguments:
                input_patterns = [
                    r"Action Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question|Final Answer):|$)",
                    r"Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
                    r"Parameters?:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
                ]
                for pattern in input_patterns:
                    try:
                        input_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                        if input_match:
                            action_input = input_match.group(1).strip()
                            action_input = re.split(r'\nObservation:', action_input, maxsplit=1)[0].strip()
                            # Try to parse as JSON
                            try:
                                cleaned = self.clean_json_input(action_input)
                                parsed = json.loads(cleaned)
                                if isinstance(parsed, dict):
                                    result.arguments = parsed
                                else:
                                    # Store raw text in a special key for agent to handle
                                    result.arguments = {"__raw_input__": action_input}
                            except (json.JSONDecodeError, TypeError):
                                result.arguments = {"__raw_input__": action_input}
                            break
                    except Exception as e:
                        logger.warning(f"Error matching action input pattern '{pattern}': {e}")
                        continue

        except Exception as e:
            logger.error(f"Critical error in ReAct parse_response: {e}", exc_info=True)

        return result

    def format_tools_for_prompt(self, tools: list) -> str:
        """Return a text description of tools (used in ReAct prompt).

        The actual formatting is delegated to ToolPromptGenerator, so
        this just returns a simple fallback description.
        """
        lines = []
        for tool in tools:
            meta = tool.metadata
            params = ", ".join(
                f"{p.name}: {p.type.value}" for p in meta.parameters
            )
            lines.append(f"- {meta.name}: {meta.description} ({params})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Native Function Calling Strategy
# ---------------------------------------------------------------------------

class NativeFunctionCallingStrategy(ToolCallingStrategy):
    """Use model's native tool/function calling via chat templates or API.

    This strategy converts BaseTool metadata to JSON Schema function
    definitions and parses structured tool calls from model responses.
    Works with:
    - Transformers models that support ``tools`` param in chat templates
    - vLLM with tool call parser
    - API models (OpenAI, Anthropic, Gemini) that return structured tool calls
    """

    @property
    def name(self) -> str:
        return "native"

    def parse_response(self, text: str, tools: dict[str, Any] | None = None) -> ToolCallResult:
        """Parse tool calls from native function calling response.

        Handles multiple response formats:
        1. Structured tool_calls in metadata (API models)
        2. Qwen-style: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        3. Llama-style: <|python_tag|>function_name(...)
        4. Mistral-style: [TOOL_CALLS][{"name": ..., "arguments": ...}]
        5. Generic JSON function call format

        Args:
            text: Raw model response (may contain structured markers).
            tools: Dict mapping tool name -> tool object for validation.

        Returns:
            ToolCallResult
        """
        result = ToolCallResult(raw_text=text)

        if not text or not isinstance(text, str):
            return result

        # --- Try Qwen format: <tool_call>...</tool_call> ---
        qwen_match = re.search(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            text, re.DOTALL,
        )
        if qwen_match:
            try:
                call_data = json.loads(qwen_match.group(1))
                tool_name = call_data.get("name") or call_data.get("function")
                arguments = call_data.get("arguments") or call_data.get("parameters") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if tool_name:
                    result.tool_name = tool_name
                    result.arguments = arguments
                    result.is_tool_call = True
                    logger.debug(f"Parsed Qwen-style tool call: {tool_name}")
                    return result
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse Qwen tool call JSON: {e}")

        # --- Try Llama/Hermes format: <|python_tag|> or <function= ---
        llama_match = re.search(
            r'(?:<\|python_tag\|>|<function=)(\w+)\s*[(\{](.+?)[)\}]',
            text, re.DOTALL,
        )
        if llama_match:
            tool_name = llama_match.group(1).strip()
            args_text = llama_match.group(2).strip()
            try:
                # Try JSON parse
                if not args_text.startswith("{"):
                    args_text = "{" + args_text
                if not args_text.endswith("}"):
                    args_text = args_text + "}"
                arguments = json.loads(args_text)
                result.tool_name = tool_name
                result.arguments = arguments if isinstance(arguments, dict) else {}
                result.is_tool_call = True
                logger.debug(f"Parsed Llama-style tool call: {tool_name}")
                return result
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse Llama tool call: {e}")

        # --- Try Mistral format: [TOOL_CALLS][...] ---
        mistral_match = re.search(
            r'\[TOOL_CALLS\]\s*\[(.+?)\]',
            text, re.DOTALL,
        )
        if mistral_match:
            try:
                calls = json.loads("[" + mistral_match.group(1) + "]")
                if calls and isinstance(calls, list):
                    call = calls[0]  # Take first tool call
                    tool_name = call.get("name") or call.get("function")
                    arguments = call.get("arguments") or call.get("parameters") or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if tool_name:
                        result.tool_name = tool_name
                        result.arguments = arguments
                        result.is_tool_call = True
                        logger.debug(f"Parsed Mistral-style tool call: {tool_name}")
                        return result
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse Mistral tool call: {e}")

        # --- Try generic JSON function call ---
        # Look for {"name": "tool", "arguments": {...}} pattern
        json_match = re.search(
            r'\{\s*"(?:name|function)"\s*:\s*"(\w+)".*?"(?:arguments|parameters)"\s*:\s*(\{.*?\})',
            text, re.DOTALL,
        )
        if json_match:
            try:
                tool_name = json_match.group(1)
                arguments = json.loads(json_match.group(2))
                result.tool_name = tool_name
                result.arguments = arguments if isinstance(arguments, dict) else {}
                result.is_tool_call = True
                logger.debug(f"Parsed generic JSON tool call: {tool_name}")
                return result
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Failed to parse generic JSON tool call: {e}")

        # --- No tool call found — check for plain text answer ---
        # If the response doesn't contain any tool call markers, treat as final answer
        text_stripped = text.strip()
        if text_stripped and not any(marker in text for marker in [
            "<tool_call>", "<|python_tag|>", "<function=", "[TOOL_CALLS]",
            "Thought:", "Action:", "Tool:", '"name"', '"function"',
        ]):
            result.final_answer = text_stripped
            logger.debug("No tool call markers found, treating as final answer")

        return result

    def format_tools_for_prompt(self, tools: list) -> list[dict[str, Any]]:
        """Convert tools to JSON Schema definitions for native calling.

        Returns:
            List of OpenAI-format tool definitions.
        """
        definitions = tools_to_definitions(tools)
        return [d.to_openai_format() for d in definitions]


# ---------------------------------------------------------------------------
# Hybrid Strategy
# ---------------------------------------------------------------------------

class HybridStrategy(ToolCallingStrategy):
    """Try native function calling first, fall back to ReAct on parse failure."""

    def __init__(self):
        self._native = NativeFunctionCallingStrategy()
        self._react = ReActStrategy()

    @property
    def name(self) -> str:
        return "hybrid"

    def parse_response(self, text: str, tools: dict[str, Any] | None = None) -> ToolCallResult:
        """Try native parsing first, then ReAct."""
        result = self._native.parse_response(text, tools)
        if result.is_tool_call or result.final_answer:
            logger.debug("Hybrid strategy: native parsing succeeded")
            return result

        logger.debug("Hybrid strategy: native parsing failed, trying ReAct")
        return self._react.parse_response(text, tools)

    def format_tools_for_prompt(self, tools: list) -> Any:
        """Use native format (JSON Schema definitions)."""
        return self._native.format_tools_for_prompt(tools)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_strategy(
    mode: str = "auto",
    model: Any | None = None,
) -> ToolCallingStrategy:
    """Create the appropriate tool calling strategy.

    Args:
        mode: One of "auto", "native", "react", "hybrid".
        model: The model instance (checked for supports_tool_calling).

    Returns:
        ToolCallingStrategy instance.
    """
    if mode == "react":
        return ReActStrategy()
    elif mode == "native":
        return NativeFunctionCallingStrategy()
    elif mode == "hybrid":
        return HybridStrategy()
    elif mode == "auto":
        # Auto-detect based on model capabilities
        if model is not None and hasattr(model, 'supports_tool_calling'):
            try:
                if model.supports_tool_calling():
                    logger.info("Auto-detected native tool calling support, using hybrid strategy")
                    return HybridStrategy()
            except Exception:
                logger.debug("Native tool-calling capability check failed; using default strategy", exc_info=True)
        logger.debug("Using ReAct strategy (model does not support native tool calling)")
        return ReActStrategy()
    else:
        logger.warning(f"Unknown tool_calling_mode '{mode}', defaulting to ReAct")
        return ReActStrategy()
