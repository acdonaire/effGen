"""Unit tests for ReAct loop efficiency guardrails.

Covers the loop-control logic that keeps small models from burning iterations
on a question they have already answered:

- A simple-arithmetic tool result is returned directly after a single call
  (``answer_source="direct_calculator_result"``), including power/root phrasings
  ("15 squared", "cube of 3", "square root of 144").
- Result-based short-circuit: when a tool reproduces a result it already
  returned — with a *different* input, so the exact-input loop guard does not
  fire — the loop stops and returns that confident answer
  (``answer_source="repeated_tool_result"``).
- A clean ``Final Answer:`` after one tool call stops immediately.
- The guardrails never change a correct answer.

These use an in-process scripted model (no network) to drive the loop
deterministically — not a mock of live API behavior, which the project forbids.
"""

from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.builtin.calculator import Calculator


class _ScriptedModel(BaseModel):
    """Returns a fixed sequence of ReAct responses, one per generate() call.

    The last scripted response repeats if the loop asks for more, so a test
    that fails to short-circuit will not hang — it will exhaust iterations.
    """

    def __init__(self, responses: list[str]):
        super().__init__(model_name="scripted-model", model_type=ModelType.OPENAI)
        self._responses = responses
        self.calls = 0

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def generate(self, prompt, config=None, **kwargs):
        idx = min(self.calls, len(self._responses) - 1)
        text = self._responses[idx]
        self.calls += 1
        return GenerationResult(
            text=text, tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield self.generate(prompt).text

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:  # pragma: no cover
        return 4096

    def unload(self) -> None:  # pragma: no cover
        pass

    def generate_batch(self, prompts, config=None, **kwargs):  # pragma: no cover
        return [self.generate(p, config=config) for p in prompts]

    def generate_with_tools(self, prompt, tools, config=None, **kwargs):  # pragma: no cover
        return self.generate(prompt, config=config)

    def supports_function_calling(self) -> bool:
        return False

    def supports_tool_calling(self) -> bool:
        # Force the text ReAct loop (the path the audit exercised).
        return False


def _calc_action(expr: str) -> str:
    return (
        f'Thought: I will compute this.\n'
        f'Action: calculator\n'
        f'Action Input: {{"operation": "calculate", "expression": "{expr}"}}'
    )


def _make_agent(model: _ScriptedModel, max_iterations: int = 10) -> Agent:
    cfg = AgentConfig(
        name="loop-test",
        model=model,
        tools=[Calculator()],
        max_iterations=max_iterations,
        tool_calling_mode="react",
    )
    return Agent(config=cfg)


@pytest.mark.parametrize(
    "task,expr",
    [
        ("Compute 15 squared", "15^2"),
        ("What is the cube of 3?", "3^3"),
        ("What is the square root of 144?", "sqrt(144)"),
        ("What is 23 times 19?", "23 * 19"),
    ],
)
def test_simple_arithmetic_returns_after_one_call(task, expr):
    """Power/root/arithmetic phrasings short-circuit after one tool call."""
    model = _ScriptedModel([_calc_action(expr), _calc_action(expr)])
    agent = _make_agent(model)
    resp = agent.run(task)
    assert resp.success is True
    assert resp.tool_calls == 1, f"expected 1 tool call, got {resp.tool_calls}"
    assert resp.metadata.get("answer_source") == "direct_calculator_result"


def test_repeated_result_short_circuits_with_different_input():
    """Same result via different inputs stops the loop (no exact-loop match).

    'explain' disables the direct-calculator fast path, so only the
    result-based short-circuit can stop this at two calls.
    """
    model = _ScriptedModel([
        _calc_action("15^2"),    # -> 225
        _calc_action("15 * 15"),  # different input, same 225 -> short-circuit
        _calc_action("225 * 1"),  # would be a 3rd call if dedup failed
    ])
    agent = _make_agent(model)
    resp = agent.run("Explain step by step and compute 15 squared")
    assert resp.success is True
    assert resp.tool_calls == 2, f"expected 2 tool calls, got {resp.tool_calls}"
    assert resp.metadata.get("answer_source") == "repeated_tool_result"
    assert "225" in (resp.output or "")


def test_clean_final_answer_after_one_call_stops():
    """A confident Final Answer after one tool call ends the loop immediately."""
    model = _ScriptedModel([
        _calc_action("7 * 6"),
        "Thought: I now know the answer.\nFinal Answer: 42",
    ])
    agent = _make_agent(model)
    resp = agent.run("Explain how to multiply 7 by 6")
    assert resp.success is True
    assert resp.tool_calls == 1
    assert "42" in (resp.output or "")


def test_guardrails_preserve_correct_answer():
    """The efficiency guardrails must not corrupt the tool's correct result."""
    model = _ScriptedModel([_calc_action("12 * 12")])
    agent = _make_agent(model)
    resp = agent.run("Compute 12 multiplied by 12")
    assert resp.success is True
    assert "144" in (resp.output or "")
    assert resp.tool_calls == 1
