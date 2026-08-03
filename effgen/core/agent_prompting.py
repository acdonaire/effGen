"""Prompt and context assembly for :class:`effgen.core.agent.Agent`.

Builds the pieces of text a request is assembled from: the system prompt, the
tool descriptions whose verbosity follows the model's size, the recent
conversation history, and the Anthropic prompt-caching wrappers for the system
prompt and the tool specs. Mixed into :class:`Agent`; this module imports
nothing from ``agent.py``.
"""

from __future__ import annotations

from ..memory.short_term import MessageRole


class AgentPromptingMixin:
    """System prompt, tool descriptions, and conversation-history formatting."""

    # Default ReAct prompt template
    REACT_PROMPT_TEMPLATE = """You are a helpful AI assistant that can reason step-by-step and use tools.
{conversation_history}
Available tools:
{tools_description}

IMPORTANT: If there is previous conversation context above, use that information to answer questions about past interactions.

Use the following format:

Question: the input question or task
Thought: think step-by-step about what to do next
Action: the tool to use (or "Final Answer" when ready to respond)
Action Input: the input for the tool
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the complete response to the original question

IMPORTANT: You do NOT have to use a tool for every question.
If you can answer directly from your knowledge or from the conversation history above, skip the Action step entirely:
Thought: I can answer this directly without any tools.
Final Answer: [your answer here]
Only use tools when you NEED external computation, data, or system access.

Example (no tool needed):
Question: Tell me a joke about programming.
Thought: This is a creative request. I can answer directly without tools.
Final Answer: Why do programmers prefer dark mode? Because light attracts bugs!

Begin!

Question: {task}
{scratchpad}"""

    def _get_anthropic_system(self) -> str | list | None:
        """
        Return the system prompt for Anthropic requests.

        When ``AgentConfig.cache_system_prompt=True`` and the model is an
        ``AnthropicAdapter``, the system prompt is returned as a list of
        content blocks with ``cache_control`` on the last block so that it is
        cached across sequential requests.
        """
        system = self.config.system_prompt if self.config.stable_system_prompt else None
        if system is None:
            return None
        try:
            from ..models.anthropic_adapter import AnthropicAdapter
            from ..models.anthropic_cache import apply_cache_to_system
        except ImportError:
            return system
        if isinstance(self.model, AnthropicAdapter) and self.config.cache_system_prompt:
            return apply_cache_to_system(system)
        return system

    def _get_anthropic_tools(self, tools: list[dict]) -> list[dict]:
        """
        Apply ``cache_control`` to the last tool spec when appropriate.

        Only active when ``AgentConfig.cache_tools=True`` and the model is an
        ``AnthropicAdapter``.
        """
        if not tools:
            return tools
        try:
            from ..models.anthropic_adapter import AnthropicAdapter
            from ..models.anthropic_cache import apply_cache_to_last_tool
        except ImportError:
            return tools
        if isinstance(self.model, AnthropicAdapter) and self.config.cache_tools:
            return apply_cache_to_last_tool(tools)
        return tools

    def _build_system_prompt(self) -> str:
        """Build a dynamic system prompt based on agent configuration and tools."""
        return self._system_prompt_builder.build(
            tools=self.config.tools,
            agent_name=self.name,
            base_system_prompt=None,  # Will generate default role
            enable_fallback=self._enable_fallback,
            verbose=self._verbose_tools,
        )

    def _auto_detect_verbose(self) -> bool:
        """Auto-detect whether to use verbose tool descriptions based on model size."""
        name = (self.model_name or "").lower()
        # Check for known small models (< 3B) -> full verbose with examples
        # Check for medium models (3B-7B) -> verbose without examples
        # Check for large models (> 7B) or API models -> compact
        for indicator in ["0.5b", "1b", "1.5b", "2b"]:
            if indicator in name:
                return True
        for indicator in ["3b", "4b", "5b", "7b"]:
            if indicator in name:
                return True
        # API models
        for indicator in ["gpt", "claude", "gemini"]:
            if indicator in name:
                return False
        # Default: verbose (safe for SLMs)
        return True

    def _get_tools_description(self, verbose: bool | None = None) -> str:
        """
        Get formatted description of available tools.

        Args:
            verbose: Override verbosity. If None, uses self._verbose_tools.

        Returns:
            Formatted tools description string.
        """
        if not self.tools:
            return "No tools available."

        use_verbose = verbose if verbose is not None else self._verbose_tools
        return self._tool_prompt_generator.generate_tools_section(verbose=use_verbose)

    def _format_conversation_history(self, max_turns: int = 25) -> str:
        """
        Format conversation history for inclusion in prompt.

        Uses ShortTermMemory to retrieve recent messages, including
        summaries of older messages when available.

        Args:
            max_turns: Maximum number of previous turns (user+assistant pairs)

        Returns:
            Formatted conversation history string
        """
        # Include summaries of older messages first
        summaries = self.short_term_memory.summaries
        messages = self.short_term_memory.get_recent_messages(n=max_turns * 2)
        if not messages and not summaries:
            return ""

        history = "\n\n=== Previous Conversation Context ===\n"

        # Add summaries if they exist (these cover older, summarized turns)
        if summaries:
            for summary in summaries:
                history += f"[Earlier context summary: {summary.summary}]\n\n"

        turn_num = 0
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == MessageRole.USER:
                turn_num += 1
                history += f"[Turn {turn_num}]\n"
                history += f"User: {msg.content}\n"
                # Check if next message is assistant
                if i + 1 < len(messages) and messages[i + 1].role == MessageRole.ASSISTANT:
                    # Truncate long assistant responses to save tokens
                    assistant_content = messages[i + 1].content
                    if len(assistant_content) > 300:
                        assistant_content = assistant_content[:300] + "..."
                    history += f"Assistant: {assistant_content}\n\n"
                    i += 2
                    continue
                else:
                    history += "\n"
            i += 1

        history += "=== End of Previous Context ===\n"
        return history if turn_num > 0 or summaries else ""
