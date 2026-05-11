"""Unit tests for tool-calling strategy edge cases."""

from __future__ import annotations

from effgen.core.tool_calling import HybridStrategy


def test_hybrid_does_not_treat_thought_only_text_as_final_answer():
    strategy = HybridStrategy()
    result = strategy.parse_response(
        "Thought: The previous calculation is correct and we don't need to perform it again."
    )

    assert result.final_answer is None
    assert result.is_tool_call is False
    assert result.thought
