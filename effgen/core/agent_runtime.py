"""Runtime helpers for :class:`effgen.core.agent.Agent`.

Pure/utility helpers extracted from ``agent.py`` for maintainability — answer
sanitization, task/preview coercion, JSON cleanup, the sync-over-async bridge,
and small provider/number utilities. This module imports nothing from
``agent.py`` so it stays a dependency-free leaf. Behaviour is identical to the
original in-class definitions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from ..observability import get_logger as _get_obs_logger
from ..utils.async_bridge import run_coroutine_sync
from ..utils.structured_logging import (
    get_structured_logger,
)

if TYPE_CHECKING:
    from .messages import Message

logger = logging.getLogger(__name__)
_slog = get_structured_logger(__name__)
# Canonical structured observability logger — emits redacted JSON lines with OTel context
_obs_log = _get_obs_logger(__name__)

# Mirrors AgentConfig.system_prompt's default (effgen/core/agent.py) — kept as
# a literal here rather than imported to preserve this module's dependency-free
# leaf status.
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
IMAGE_GROUNDING_GUIDANCE = (
    "Describe images objectively — state only what is visible, and say so "
    "explicitly when a requested detail is not shown."
)

# Guardrail-chain signatures (sorted guardrail names) already warned about for
# missing TOOL_OUTPUT injection screening — a heads-up fires once per distinct
# configuration, not on every agent built with the same preset.
_tool_output_injection_gap_warned: set[tuple[str, ...]] = set()


_PROVIDER_BY_CLASS_PREFIX: dict[str, str] = {
    "OpenAI": "openai",
    "Cerebras": "cerebras",
    "Gemini": "google",
    "Anthropic": "anthropic",
    "Groq": "groq",
    "Together": "together",
    "Fireworks": "fireworks",
    "HFInference": "hf_inference",
    "Replicate": "replicate",
    "MLXVLM": "mlx_vlm",
}


def _infer_provider_from_model(model: Any, model_name: str | None = None) -> str:
    """
    Best-effort provider inference from a model instance.

    Order of preference:
    1. ``model.provider`` / ``model.provider_name`` attribute if present.
    2. Class-name prefix mapping (``OpenAIAdapter`` → ``openai``).
    3. Heuristic mapping from the model identifier string.
    4. ``"unknown"``.
    """
    if model is None:
        return "unknown"
    for attr in ("provider", "provider_name"):
        val = getattr(model, attr, None)
        if isinstance(val, str) and val:
            return val
    cls_name = type(model).__name__
    for prefix, provider in _PROVIDER_BY_CLASS_PREFIX.items():
        if cls_name.startswith(prefix):
            return provider
    m = (model_name or "").lower()
    if m.startswith(("gpt-", "o1", "o3", "o4", "text-")):
        return "openai"
    if m.startswith(("gemini", "models/gemini")):
        return "google"
    if m.startswith(("claude", "anthropic")):
        return "anthropic"
    if "llama" in m or "qwen" in m or m.startswith("cerebras"):
        return "cerebras"
    if m.startswith(("mixtral", "mistral")):
        return "groq"
    return "unknown"


def _safe_int_or_none(value: Any) -> int | None:
    """Convert numeric dashboard metadata values without raising."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float_or_none(value: Any) -> float | None:
    """Convert numeric dashboard metadata values without raising."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# Loop-bookkeeping nudges the ReAct scaffolding injects into the scratchpad to
# steer a stalling model toward a final answer. They are defined here, in ONE
# place, and imported by the injection sites (see ``agent_react``) so the set of
# strings a loop can append can never drift out of sync with the set of strings
# stripped from the user-facing answer below. Adding a new nudge => add a
# constant here and reference it at the injection site; it is then stripped for
# free. The injection sites prepend "\n" (and sometimes "Observation: "), which
# the strip patterns tolerate.
NUDGE_CONTINUE = "[Tool results computed above. Continue or provide Final Answer:]"
NUDGE_HAVE_ANSWER = "[You have the answer from the tool. Please respond with 'Final Answer:' now.]"
NUDGE_HAVE_RESULTS = (
    "[You already have results from this tool. If you have "
    "enough information, respond now with 'Final Answer:'.]"
)
NUDGE_ALREADY_COMPUTED = (
    "You already computed this. Please provide your final response "
    "using 'Final Answer:' now."
)
NUDGE_NO_TOOLS = "No tools available. Please provide your answer directly using 'Final Answer:'."
NUDGE_NOT_USABLE = (
    "That was not a usable answer. Call the tool correctly or give a "
    "plain Final Answer."
)

# Literal loop-bookkeeping strings to strip — every injectable nudge above, plus a
# couple of defensive bracketed/sub-phrase variants. These must never reach a
# user-facing answer. An adjacent newline is consumed when present so removing a
# mid-line marker doesn't leave an orphan line.
_SCAFFOLD_LITERALS: tuple[str, ...] = (
    NUDGE_CONTINUE,
    NUDGE_HAVE_ANSWER,
    NUDGE_HAVE_RESULTS,
    f"[{NUDGE_ALREADY_COMPUTED}]",
    NUDGE_ALREADY_COMPUTED,
    NUDGE_NO_TOOLS,
    NUDGE_NOT_USABLE,
    "Please provide your final response using 'Final Answer:' now.",
)
_SCAFFOLD_LITERAL_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(r"[ \t]*\n?[ \t]*" + re.escape(lit)) for lit in _SCAFFOLD_LITERALS
)

# A line-anchored "Final Answer:" / "Answer:" label (allows quote/list prefixes).
_ANSWER_LABEL_RE = re.compile(
    r"(?:^|\n)[ \t>*\-]*(?:final[ \t]*answer|answer)[ \t]*[:\-][ \t]*",
    re.IGNORECASE,
)
# Trailing ReAct bleed: an Observation/Thought/Question/Action section the model
# appended after its real answer.
_TRAILING_BLEED_RE = re.compile(
    r"\n[ \t>*\-]*(?:observation|thought|question|action(?:[ \t]+input)?)[ \t]*:.*\Z",
    re.IGNORECASE | re.DOTALL,
)
# Tool-echo fragment like "[calculator({'expression': '15*15'})] → 225". The
# scaffolding always emits the Unicode arrow (see the f-strings in the ReAct/
# native loops); we deliberately do NOT match an ASCII "->" here so legitimate
# prose like "see [1] -> next" is never corrupted.
_TOOL_ECHO_RE = re.compile(r"\[[^\[\]\n]*\][ \t]*→[ \t]*")

# A brace-delimited JSON object allowing up to two levels of nesting, so a tool
# call with structured args ("{\"args\": {\"x\": 1}}") is matched whole rather
# than leaving a dangling "}" behind.
_JSON_OBJ = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
# Model tool-call syntax that some families (e.g. Llama 3.1) emit as plain text
# when a native tool call isn't routed: "<function=calc>{...}</function>",
# "<tool_call>{...}</tool_call>", and stray special tokens. These are pure
# scaffolding and must never surface in an answer.
_TOOLCALL_CONSTRUCT_RE = re.compile(
    r"<?\s*(?:function\s*=\s*[\w.\-]+|tool_call)\s*>?\s*" + _JSON_OBJ
    + r"\s*(?:</\s*(?:function|tool_call)\s*>)?",
    re.IGNORECASE | re.DOTALL,
)
_TOOLCALL_TAG_RE = re.compile(
    r"</?\s*(?:function(?:\s*=\s*[\w.\-]+)?|tool_call)\s*>"
    r"|<\|[^>|]*\|>",
    re.IGNORECASE,
)
# A leaked tool call with no wrapping tag at all — just the bare tool name
# immediately followed by its JSON arguments, e.g.
# 'order_lookup {"order_id": "ORD-1001"} \nThe order status is "shipped"'.
# Anchored to the very start of the text (never mid-sentence) so an ordinary
# answer that happens to contain "word {...}" is never touched; a tool name
# is always a lowercase snake_case identifier by effGen convention.
_LEADING_TOOLCALL_ECHO_RE = re.compile(
    r"^[a-z_][a-z0-9_]{1,63}[ \t]+" + _JSON_OBJ + r"[ \t\n]*"
)


def sanitize_final_answer(text: str | None) -> str | None:
    """Strip internal ReAct/tool scaffolding from a user-facing answer.

    Conservative and idempotent: removes the loop-bookkeeping strings the agent
    injects, leading ``Final Answer:``/``Answer:`` labels (returning the text
    after the *last* such label, so ``"Canberra\\nFinal Answer: Canberra"`` →
    ``"Canberra"``), trailing ``Observation:``/``Thought:``/``Question:`` bleed,
    ``[tool(args)] →`` echo prefixes, tagged tool-call syntax
    (``<function=x>{...}</function>``, ``<tool_call>{...}</tool_call>``), and a
    leading untagged tool-call echo (``tool_name {"arg": "val"}`` at the very
    start of the text, before the real answer) — without reflowing markdown
    tables, code blocks, or legitimate pipe-separated content. ``None``/non-``str``
    input is returned unchanged.
    """
    if not text or not isinstance(text, str):
        return text
    s = text
    # 1. Remove literal loop-bookkeeping strings (with any adjacent newline).
    for pat in _SCAFFOLD_LITERAL_RES:
        s = pat.sub("", s)
    # 2. Remove tool-echo prefixes, keeping the result after the arrow.
    s = _TOOL_ECHO_RE.sub("", s)
    # 2b. Remove leaked model tool-call syntax (whole constructs, then stray tags).
    s = _TOOLCALL_CONSTRUCT_RE.sub("", s)
    s = _TOOLCALL_TAG_RE.sub("", s)
    # 2c. Remove a leading bare "tool_name {json}" echo with no wrapping tag.
    s = _LEADING_TOOLCALL_ECHO_RE.sub("", s)
    # 3. Drop trailing Observation/Thought/Question/Action bleed.
    s = _TRAILING_BLEED_RE.sub("", s)
    # 4. If a line-anchored answer label is present, the real answer is what
    #    follows the LAST such label (when that tail is non-empty). A dangling
    #    label with nothing after it (e.g. native web-search replies sometimes
    #    end with a bare "Final Answer:") is scaffolding — drop the label and
    #    keep the content before it.
    labels = list(_ANSWER_LABEL_RE.finditer(s))
    if labels:
        tail = s[labels[-1].end():].strip()
        if tail:
            s = tail
        else:
            head = s[:labels[-1].start()].strip()
            if head:
                s = head
    # 5. Tidy separators left by removed scaffolding, without disturbing
    #    multi-line content (tables/code): collapse empty "| |" fragments and
    #    runs of spaces, then strip a dangling leading/trailing bare pipe.
    s = re.sub(r"\|[ \t]*\|", "|", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = s.strip()
    # Strip a dangling leading/trailing bare pipe only for single-line content,
    # so markdown tables (multi-line, pipe-delimited) are left intact.
    if "\n" not in s:
        s = re.sub(r"^\|[ \t]*", "", s)
        s = re.sub(r"[ \t]*\|$", "", s)
    return s.strip()


class AgentRuntimeMixin:
    """Pure/stateless helper methods mixed into :class:`Agent`."""

    @staticmethod
    def _resolve_guardrails(guardrails: Any):
        """Resolve guardrails config to a GuardrailChain or None."""
        if guardrails is None:
            return None
        # Already a GuardrailChain
        from ..guardrails.base import GuardrailChain
        if isinstance(guardrails, GuardrailChain):
            return guardrails
        # Preset name string
        if isinstance(guardrails, str):
            from ..guardrails.presets import get_guardrail_preset
            return get_guardrail_preset(guardrails)
        return None

    @staticmethod
    def _warn_tool_output_injection_gap(guardrail_chain: Any, has_tools: bool) -> None:
        """Warn once when a tool-attached agent's guardrails skip TOOL_OUTPUT
        injection screening.

        An indirect prompt injection carried in a tool's return value (a
        scraped page, a ticket body, a retrieved document) reaches the model
        unscreened unless the chain's ``PromptInjectionGuardrail`` explicitly
        covers ``GuardrailPosition.TOOL_OUTPUT``. Logged once per distinct
        guardrail configuration at the point a tool-bearing agent is built, so
        the gap is visible at the point of choice rather than only in a
        preset's docstring.
        """
        if not has_tools or guardrail_chain is None:
            return
        from ..guardrails.base import GuardrailPosition
        from ..guardrails.injection import PromptInjectionGuardrail

        injection_guardrails = [
            g for g in guardrail_chain.guardrails
            if isinstance(g, PromptInjectionGuardrail) and g.enabled
        ]
        if not injection_guardrails:
            return
        if any(GuardrailPosition.TOOL_OUTPUT in g.positions for g in injection_guardrails):
            return

        sig = tuple(sorted(g.name for g in guardrail_chain.guardrails))
        if sig in _tool_output_injection_gap_warned:
            return
        _tool_output_injection_gap_warned.add(sig)
        logger.warning(
            "This agent has tools attached, but its guardrail configuration "
            "does not screen a tool's return value for prompt injection — an "
            "instruction embedded in a scraped page, a ticket body, or a "
            "retrieved document reaches the model unscreened. The 'strict' "
            "and 'phi' guardrail presets screen tool output too; pass "
            "PromptInjectionGuardrail(positions=[GuardrailPosition.INPUT, "
            "GuardrailPosition.TOOL_OUTPUT]) directly for a custom chain."
        )

    def _humanize_observation(self, obs: str) -> str:
        """
        Render a retrieval/search tool observation as readable passage text.

        When a knowledge-base tool's raw result (a ``{'results': [...]}`` dict)
        ends up being used as a fallback answer — e.g. a model that loops on the
        retrieval tool instead of synthesizing — dumping the Python dict repr is
        unreadable. Extract and join the passage ``content`` instead so the
        degraded answer is at least the retrieved evidence (citations are still
        attached separately to the response).
        """
        s = obs.strip()
        brace = s.find("{")
        if brace == -1 or "results" not in s:
            return obs
        # Extract a single balanced-brace object starting at the first '{' so
        # trailing scaffolding (e.g. an appended "[Tool results computed…]" nudge
        # or a following "Thought:") doesn't defeat the parse.
        depth = 0
        end = -1
        for i in range(brace, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = s[brace:end] if end != -1 else s[brace:]
        data: Any = None
        try:
            import ast
            data = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                return obs
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            passages: list[str] = []
            for item in data["results"]:
                if isinstance(item, dict):
                    content = item.get("content") or item.get("text") or item.get("snippet")
                    if content:
                        passages.append(str(content).strip())
            if passages:
                return "\n\n".join(passages)
        return obs

    @staticmethod
    def _run_coroutine_sync(coro, timeout: float = 120.0):
        """
        Run an async coroutine from synchronous code.

        Thin wrapper over the shared :func:`effgen.utils.run_coroutine_sync`
        so every sync-to-async bridge in the codebase behaves identically:
        drive the coroutine directly when no loop is running, otherwise run it
        on a worker thread and block — never skip it.

        Args:
            coro: The coroutine to run.
            timeout: Maximum seconds to wait (default 120s).
        """
        return run_coroutine_sync(coro, timeout=timeout)

    @staticmethod
    def _clean_json_input(raw: str) -> str:
        """
        Clean malformed JSON commonly produced by SLMs before parsing.

        Handles:
        - Markdown-wrapped JSON (```json ... ```)
        - Trailing commas  ({"key": "val",})
        - Unquoted keys    ({expression: "2+2"})

        Returns the cleaned string (still needs json.loads).
        """
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            # Remove opening fence (with optional language tag) and closing fence
            text = re.sub(r'^```(?:json|JSON)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)
            text = text.strip()

        # Remove trailing commas before } or ]
        # e.g. {"key": "val",} -> {"key": "val"}
        text = re.sub(r',\s*([}\]])', r'\1', text)

        # Quote unquoted keys:  {expression: "2+2"} -> {"expression": "2+2"}
        # Match word characters at the start of a key position (after { or ,)
        # but only if they aren't already quoted
        text = re.sub(
            r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:',
            r' "\1":',
            text
        )

        return text

    @staticmethod
    def _sanitize_tool_input(tool_input: str, max_length: int = 10000) -> str:
        """
        Sanitize tool input by stripping control characters and limiting length.

        Args:
            tool_input: Raw input string.
            max_length: Maximum allowed input length.

        Returns:
            Sanitized input string.
        """
        if not tool_input:
            return tool_input
        # Strip control characters except newline and tab
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', tool_input)
        # Limit length
        if len(sanitized) > max_length:
            logger.warning(f"Tool input truncated from {len(sanitized)} to {max_length} chars")
            sanitized = sanitized[:max_length]
        return sanitized

    @staticmethod
    def _extract_task_preview(task: Any, limit: int = 200) -> str:
        """Return a safe, short string preview of a ``run()`` task argument.

        ``task`` may be a ``str``, a ``Message``, or a ``list[ContentPart]`` —
        never assume it is subscriptable.
        """
        try:
            if isinstance(task, str):
                return task[:limit]
            from effgen.core.messages import Message, TextPart

            if isinstance(task, Message):
                return task.text[:limit]
            if isinstance(task, list | tuple):
                texts = [p.text for p in task if isinstance(p, TextPart)]
                if texts:
                    return " ".join(texts)[:limit]
                return f"<{len(task)} content part(s)>"
        except Exception:
            logger.debug("Failed to build task preview", exc_info=True)
        return str(task)[:limit]

    @staticmethod
    def _coerce_task_input(
        task: Any, inputs: Any = None,
    ) -> tuple[str, list[Any] | None]:
        """Normalise a ``run()``/``stream()`` task into ``(text, inputs)``.

        Accepts ``str``, a multimodal ``Message``, or a ``list[ContentPart]``.
        Text parts become the task string; image/audio/video parts are routed to
        the multimodal ``inputs`` path (merged with any explicit ``inputs=``).
        Any other type raises a clear ``TypeError``.
        """
        from effgen.core.messages import (
            _CONTENT_PART_TYPES,
            Message,
            TextPart,
        )

        # Fast path: the overwhelmingly common case.
        if isinstance(task, str):
            return task, inputs

        if isinstance(task, Message):
            parts: list[Any] = list(task.content)
        elif isinstance(task, list | tuple):
            parts = list(task)
        else:
            raise TypeError(
                "Agent.run(task=...) accepts a str, a Message, or a "
                "list[ContentPart]; got "
                f"{type(task).__name__}. For images/audio/video pass text plus "
                "inputs=[image_from(...)], e.g. "
                'agent.run("describe this", inputs=[image_from(path)]).'
            )

        texts: list[str] = []
        media: list[Any] = []
        for index, part in enumerate(parts):
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, TextPart):
                texts.append(part.text)
            elif isinstance(part, _CONTENT_PART_TYPES):
                media.append(part)
            else:
                raise TypeError(
                    f"Agent.run(task=...) content item {index} must be a "
                    "ContentPart (image_from/audio_from/video_from) or str; got "
                    f"{type(part).__name__}."
                )

        text_task = "\n".join(t for t in texts if t)

        merged: list[Any] = list(media)
        if inputs is not None:
            merged.extend(inputs if isinstance(inputs, list | tuple) else [inputs])

        return text_task, (merged or None)

    def _build_multimodal_prompt(self, task: str, inputs: Any) -> list[Any]:
        """Build structured Messages for adapter-native multimodal input."""
        from effgen.core.messages import ContentPart, Message, Role, TextPart

        if isinstance(inputs, tuple):
            inputs = list(inputs)
        elif not isinstance(inputs, list):
            inputs = [inputs]

        from pathlib import Path as _Path

        content: list[ContentPart] = [TextPart(text=task)]
        for index, part in enumerate(inputs):
            # Convenience: a bare path or URL string (or Path) is auto-wrapped
            # by extension into the matching image/audio/video part, so
            # inputs=["photo.png"] works without importing image_from().
            if isinstance(part, str | _Path):
                from effgen.core.multimodal import _document_extension, part_from
                from effgen.errors import InvalidMultimodalContent
                try:
                    part = part_from(part)
                except InvalidMultimodalContent as exc:
                    # A recognized document type already carries a complete hint
                    # pointing at RAG ingestion / the pdf/excel tools; appending
                    # the image_from import line would only send the user back to
                    # the media wrappers that do not fit a document.
                    if _document_extension(str(part)):
                        raise TypeError(
                            f"Agent.run(inputs=[...]) item {index}: {exc}"
                        ) from exc
                    raise TypeError(
                        f"Agent.run(inputs=[...]) item {index}: {exc}. "
                        "Import the helper with: from effgen import image_from"
                    ) from exc
            elif not hasattr(part, "type"):
                raise TypeError(
                    "Agent.run(inputs=[...]) expects effGen multimodal parts "
                    f"(image_from/audio_from/video_from); item {index} is {type(part).__name__}. "
                    "Import them with: from effgen import image_from, audio_from, video_from"
                )
            content.append(part)

        messages: list[Message] = []
        if self.config.system_prompt:
            system_text = self.config.system_prompt
            # The unconfigured default persona carries no grounding guidance
            # for images, unlike the multimodal preset's system prompt. Add
            # the same "describe only what's visible" discipline here so a
            # plain Agent(inputs=[image_from(...)]) call gets it too, without
            # touching a caller's own custom system_prompt.
            if system_text == DEFAULT_SYSTEM_PROMPT and any(
                getattr(part, "type", None) == "image" for part in content
            ):
                system_text = f"{system_text} {IMAGE_GROUNDING_GUIDANCE}"
            messages.append(Message(role=Role.SYSTEM, content=[TextPart(text=system_text)]))
        messages.append(Message(role=Role.USER, content=content))
        return messages

    def _persona_prefix(self) -> str:
        """Return the user's custom persona as a prompt prefix, or ``""``.

        When the user set a custom ``system_prompt`` (anything other than the
        default assistant prompt), the no-tool direct path and the native/hybrid
        tool path prepend it to the user turn so the persona actually steers the
        model — matching the ReAct-text and Gemini-native paths, which already
        embed it. Adapters that take a string prompt have no separate system
        slot, so prepending is the universal, family-agnostic way to deliver it.
        Returns an empty string for the default persona so default agents are
        byte-for-byte unchanged.
        """
        persona = getattr(self, "_custom_persona", None)
        return f"{persona}\n\n" if persona else ""

    def _direct_prompt(self, task: str, conversation_history: str = "") -> str:
        """Build the user prompt for the no-tool direct/streaming paths.

        Default agents keep the familiar ``"Answer this question directly and
        concisely: … Answer:"`` framing byte-for-byte. When the user set a custom
        persona, that persona *is* the response contract, so the framework's own
        "answer directly / Answer:" boilerplate is dropped — it competes with the
        persona (it literally commands "Answer:", which fights a Socratic "never
        give the answer" tutor) and on the smallest models it can override the
        persona entirely. The persona then leads, followed by any conversation
        history and the raw task — exactly the shape that steers reliably across
        cloud and local families.
        """
        persona = getattr(self, "_custom_persona", None)
        if persona:
            if conversation_history:
                return f"{persona}\n\n{conversation_history}\n\n{task}"
            return f"{persona}\n\n{task}"
        if conversation_history:
            return (
                f"{conversation_history}\n\n"
                f"Based on the conversation above, answer this question directly "
                f"and concisely:\n\n{task}\n\nAnswer:"
            )
        return f"Answer this question directly and concisely:\n\n{task}\n\nAnswer:"

    @staticmethod
    def _prompt_to_task_hint(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt[:500]
        try:
            from effgen.core.messages import Message

            if isinstance(prompt, Message):
                return prompt.text[:500]
            if isinstance(prompt, list):
                text = "\n".join(p.text for p in prompt if isinstance(p, Message))
                if text:
                    return text[:500]
        except Exception:
            logger.debug("Failed to extract task hint from prompt", exc_info=True)
        return str(prompt)[:500]
