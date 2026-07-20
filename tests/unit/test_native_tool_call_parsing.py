"""Parsing of native tool-call formats emitted by local model families.

The fixtures are raw generations captured from local Transformers models. Llama
3.2 emits a bare ``{"name": ..., "parameters": {...}}`` object — optionally
behind ``<|python_tag|>`` — and fills unused object parameters with the string
``"{}"``, which truncates a non-greedy brace regex and made a valid call parse
as no call at all.
"""

from __future__ import annotations

import pytest

from effgen.core.tool_calling import NativeFunctionCallingStrategy

# Captured verbatim from meta-llama/Llama-3.2-3B-Instruct on a retrieval task.
LLAMA_32_BARE = (
    '{"name": "retrieval", "parameters": {"query": "refund window and support '
    'hours", "top_k": 1, "score_threshold": 0.0, "diversity": 0.0, '
    '"filter_metadata": "{}"}}<|eom_id|>'
)
LLAMA_32_PYTHON_TAG = "<|python_tag|>" + LLAMA_32_BARE


@pytest.fixture
def strategy():
    return NativeFunctionCallingStrategy()


class TestJsonToolCallParsing:
    @pytest.mark.parametrize("text", [LLAMA_32_BARE, LLAMA_32_PYTHON_TAG])
    def test_brace_inside_a_string_argument_does_not_truncate_the_call(
        self, strategy, text
    ):
        result = strategy.parse_response(text)
        assert result.is_tool_call
        assert result.tool_name == "retrieval"
        assert result.arguments["query"] == "refund window and support hours"
        assert result.arguments["top_k"] == 1
        # The brace-bearing value survives to the tool, which coerces it.
        assert result.arguments["filter_metadata"] == "{}"

    def test_arguments_key_is_accepted_alongside_parameters(self, strategy):
        result = strategy.parse_response(
            '{"name": "calculator", "arguments": {"expression": "6*7"}}'
        )
        assert (result.tool_name, result.arguments) == (
            "calculator",
            {"expression": "6*7"},
        )

    def test_nested_function_object_with_stringified_arguments(self, strategy):
        result = strategy.parse_response(
            '{"function": {"name": "calculator", "arguments": '
            '"{\\"expression\\": \\"6*7\\"}"}}'
        )
        assert result.tool_name == "calculator"
        assert result.arguments == {"expression": "6*7"}

    def test_call_with_no_arguments_parses_with_empty_arguments(self, strategy):
        result = strategy.parse_response('{"name": "get_time"}')
        assert result.is_tool_call
        assert result.tool_name == "get_time"
        assert result.arguments == {}

    @pytest.mark.parametrize(
        "text",
        [
            "<tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"6*7\"}}</tool_call>",
            '[TOOL_CALLS][{"name": "calculator", "arguments": {"expression": "6*7"}}]',
        ],
    )
    def test_other_family_formats_still_parse(self, strategy, text):
        result = strategy.parse_response(text)
        assert (result.tool_name, result.arguments) == (
            "calculator",
            {"expression": "6*7"},
        )

    def test_prose_containing_braces_is_a_final_answer_not_a_tool_call(self, strategy):
        result = strategy.parse_response("Use {} to denote the empty set.")
        assert not result.is_tool_call
        assert result.final_answer == "Use {} to denote the empty set."

    def test_plain_prose_is_a_final_answer(self, strategy):
        result = strategy.parse_response("The capital is Paris.")
        assert not result.is_tool_call
        assert result.final_answer == "The capital is Paris."

    def test_json_that_is_not_a_tool_call_is_not_read_as_one(self, strategy):
        result = strategy.parse_response('{"temperature": 21, "unit": "C"}')
        assert not result.is_tool_call
        assert result.tool_name is None


class TestJsonAnswersAreNotCalls:
    """A JSON answer that happens to carry a "name" field is not a tool call."""

    TOOLS = {"calculator": object(), "retrieval": object()}

    @pytest.mark.parametrize(
        "text",
        [
            '{"name": "Acme Corp", "revenue": 5200000, "ceo": "R. Diaz"}',
            'Here is the record:\n{"name": "Ada Lovelace", "born": 1815}',
        ],
    )
    def test_data_alongside_the_name_is_an_answer(self, strategy, text):
        result = strategy.parse_response(text, tools=self.TOOLS)
        assert not result.is_tool_call
        assert result.final_answer == text.strip()

    def test_a_bare_name_that_is_a_known_tool_is_still_a_call(self, strategy):
        result = strategy.parse_response('{"name": "retrieval"}', tools=self.TOOLS)
        assert (result.is_tool_call, result.tool_name) == (True, "retrieval")

    def test_a_truncated_call_is_neither_a_call_nor_an_answer(self, strategy):
        """Raw call syntax cut short by the token budget is never the answer."""
        result = strategy.parse_response(
            '{"name": "retrieval", "parameters": {"query": "refund', tools=self.TOOLS
        )
        assert not result.is_tool_call
        assert result.final_answer is None


class TestFunctionTagFormat:
    """``<function=NAME>{...}</function>``, the shape the parser documents."""

    def test_function_tag_call_is_parsed(self, strategy):
        result = strategy.parse_response(
            '<function=get_weather>{"city": "SF"}</function>'
        )
        assert (result.is_tool_call, result.tool_name) == (True, "get_weather")
        assert result.arguments == {"city": "SF"}

    def test_brace_inside_a_string_argument_survives(self, strategy):
        result = strategy.parse_response(
            '<function=retrieval>{"query": "refund", "filter_metadata": "{}"}</function>'
        )
        assert result.arguments == {"query": "refund", "filter_metadata": "{}"}

    def test_parenthesised_python_tag_call_still_parses(self, strategy):
        result = strategy.parse_response(
            '<|python_tag|>calculator({"expression": "6*7"})'
        )
        assert (result.tool_name, result.arguments) == (
            "calculator",
            {"expression": "6*7"},
        )
