"""Unit tests for ReAct loop efficiency guardrails.

Covers the loop-control logic that keeps small models from burning iterations
on a question they have already answered:

- A simple-arithmetic tool result is returned directly after a single call
  (``answer_source="direct_calculator_result"``), including power/root phrasings
  ("15 squared", "cube of 3", "square root of 144").
- Result-based short-circuit: when a tool reproduces a result it already
  returned — with a *different* input, so the exact-input loop guard does not
  fire — the loop stops the run. The model never wrote an answer, so the run
  reports ``outcome="stopped"`` and carries the results it had under
  ``partial``.
- A clean ``Final Answer:`` after one tool call stops immediately.
- The guardrails never change a correct answer.
- A repeated action with no usable partial answer (every attempt failed or was
  denied) stops re-offering tools so the model must respond in prose, instead
  of retrying the same call until ``max_iterations`` is exhausted.

These use an in-process scripted model (no network) to drive the loop
deterministically — not a mock of live API behavior, which the project forbids.
"""

from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
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
        raise_on_error=False,  # these assert the failure response, not the raise
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
    assert resp.outcome == "stopped"
    assert resp.stop_reason == "repeated_tool_result"
    assert resp.tool_calls == 2, f"expected 2 tool calls, got {resp.tool_calls}"
    # The result the tool computed is progress, not the answer the caller asked
    # for — so it travels under ``partial``, not in ``output``.
    assert "225" in resp.partial.text
    assert "225" not in (resp.output or "")


class _FixedRetrievalTool(BaseTool):
    """A retrieval-category tool that returns the same passages every call."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="retrieval",
            description="Search the knowledge base.",
            category=ToolCategory.INFORMATION_RETRIEVAL,
            parameters=[ParameterSpec(
                name="query", type=ParameterType.STRING,
                description="query", required=True,
            )],
        ))

    async def _execute(self, query: str = "", **kwargs):
        return "Passage: the VPN hostname is vpn.example.com on port 443."


def _retrieval_action(query: str) -> str:
    return (
        "Thought: I will search the knowledge base.\n"
        "Action: retrieval\n"
        f'Action Input: {{"query": "{query}"}}'
    )


def test_repeated_retrieval_dump_is_flagged_partial():
    """A retrieval tool whose raw passages are returned via the repeated-result
    fallback is flagged partial — a passage dump is not a synthesized answer."""
    model = _ScriptedModel([
        _retrieval_action("vpn hostname"),   # retrieves passages
        _retrieval_action("vpn address"),     # different input, same result -> dump
        _retrieval_action("vpn"),
    ])
    cfg = AgentConfig(
        raise_on_error=False,  # these assert the failure response, not the raise
        name="rag-loop-test", model=model, tools=[_FixedRetrievalTool()],
        max_iterations=6, tool_calling_mode="react",
    )
    resp = Agent(config=cfg).run("Explain and find the VPN hostname")
    assert resp.metadata.get("answer_source") == "repeated_tool_result"
    assert resp.metadata.get("partial") is True


def test_loop_detected_fallback_is_reported_as_a_stopped_run():
    """A repeated action ends the run without an answer, and says so.

    What the tool returned is progress the caller can read, so it travels under
    ``partial`` while ``output`` states what stopped the run.
    """
    model = _ScriptedModel([
        _calc_action("15^2"),   # -> 225, recorded
        _calc_action("15^2"),   # exact repeat -> loop detected
    ])
    agent = _make_agent(model)
    resp = agent.run("Explain step by step and compute 15 squared")
    assert resp.success is False
    assert resp.outcome == "stopped"
    assert resp.stop_reason == "loop_detected"
    assert resp.metadata.get("partial") is True
    assert resp.metadata.get("partial_output") == resp.partial.text


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


# ---------------------------------------------------------------------------
# Repeated action with no usable partial answer (e.g. every attempt of a
# tool call was denied) — the loop must stop re-offering tools rather than
# retrying the same denied call until max_iterations is exhausted.
# ---------------------------------------------------------------------------


class _ScriptedNativeToolModel(BaseModel):
    """A native/hybrid-tool-calling model whose response depends on whether
    ``tools`` were still offered on this call — records that per call so a
    test can assert tools stopped being offered after a dead-end loop."""

    def __init__(self):
        super().__init__(model_name="scripted-native", model_type=ModelType.OPENAI)
        self.calls = 0
        self.saw_tools_on_call: list[bool] = []

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def unload(self) -> None:  # pragma: no cover - trivial
        pass

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:  # pragma: no cover
        return 4096

    def generate_batch(self, prompts, config=None, **kwargs):  # pragma: no cover
        return [self.generate(p, config=config, **kwargs) for p in prompts]

    def generate_with_tools(self, prompt, tools, config=None, **kwargs):  # pragma: no cover
        return self.generate(prompt, config=config, tools=tools, **kwargs)

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield self.generate(prompt, config=config, **kwargs).text

    def supports_function_calling(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    def generate(self, prompt, config=None, **kwargs):
        self.calls += 1
        offered = bool(kwargs.get("tools"))
        self.saw_tools_on_call.append(offered)
        if offered:
            return GenerationResult(
                text="", tokens_used=5, finish_reason="tool_calls",
                model_name=self.model_name,
                metadata={"tool_calls": [{
                    "id": "", "type": "function",
                    "function": {
                        "name": "issue_refund",
                        "arguments": '{"order_id": "ORD-1001"}',
                    },
                }]},
            )
        return GenerationResult(
            text="Final Answer: The refund for ORD-1001 was denied.",
            tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )


def test_repeated_denied_tool_call_stops_offering_tools_and_answers():
    """A tool call that always fails/is denied leaves no partial answer to
    extract; the second identical attempt trips loop detection, which must
    stop re-offering the tool so the model answers in prose on the very next
    call instead of retrying until max_iterations."""
    from effgen import tool

    @tool
    def issue_refund(order_id: str) -> str:
        """Refund an order."""
        return "Error executing tool 'issue_refund': execution denied by human approval (denied)"

    model = _ScriptedNativeToolModel()
    cfg = AgentConfig(
        raise_on_error=False,  # these assert the failure response, not the raise
        name="denied-loop-test",
        model=model,
        tools=[issue_refund],
        max_iterations=10,
    )
    agent = Agent(config=cfg)
    resp = agent.run("Refund ORD-1001")

    assert resp.success is True
    assert "denied" in (resp.output or "").lower()
    # Tool executed exactly once — the second identical attempt is caught by
    # loop detection before a second execution.
    assert resp.tool_calls == 1
    # Tools were offered on the first two calls, then withheld once the loop
    # was detected with no partial answer to fall back on — the run finishes
    # well short of max_iterations instead of exhausting the budget.
    assert model.saw_tools_on_call == [True, True, False]
    assert model.calls == 3


# ---------------------------------------------------------------------------
# A tool that declares its output to be retrieved context
#
# The classifier reads a tool's category, which is right for the tools that
# ship. A tool whose category says otherwise — a file tool narrowed to reading,
# whose output is source material — declares it on the instance instead, and the
# loop must then give the model a tool-free turn rather than handing the file
# back as the answer.
# ---------------------------------------------------------------------------

FILE_BODY = "def add(a, b):\n    return a + b\n"


class _FileReadTool(BaseTool):
    """A file-reading tool, in the category the shipped file tool carries."""

    def __init__(self, *, context_retrieval: bool = False) -> None:
        super().__init__(metadata=ToolMetadata(
            name="file_operations",
            description="Read files.",
            category=ToolCategory.FILE_OPERATIONS,
            parameters=[ParameterSpec(
                name="path", type=ParameterType.STRING,
                description="path", required=True,
            )],
        ))
        if context_retrieval:
            self.is_context_retrieval = True

    async def _execute(self, path: str = "", **kwargs):
        return FILE_BODY


def _read_action(path: str) -> str:
    return (
        "Thought: I will read the file.\n"
        "Action: file_operations\n"
        f'Action Input: {{"path": "{path}"}}'
    )


_REVIEW = "Final Answer: divide() has no guard against a zero divisor."


def _review_agent(*, context_retrieval: bool) -> Agent:
    model = _ScriptedModel([_read_action("calc.py"), _read_action("calc.py"), _REVIEW])
    return Agent(config=AgentConfig(
        raise_on_error=False,  # these assert the failure response, not the raise
        name="review-loop-test", model=model,
        tools=[_FileReadTool(context_retrieval=context_retrieval)],
        max_iterations=6, tool_calling_mode="react",
    ))


def test_a_repeated_read_keeps_the_file_as_progress_not_as_the_answer():
    """A plain file tool that repeats leaves the run stopped, holding the file.

    The model never reviewed the file, so the file's contents are not the review
    the caller asked for: they are what the run had reached.
    """
    resp = _review_agent(context_retrieval=False).run(
        "Review calc.py and report any correctness risk."
    )
    assert resp.outcome == "stopped"
    assert resp.stop_reason == "loop_detected"
    assert resp.metadata.get("partial") is True
    assert " ".join(resp.partial.text.split()) == " ".join(FILE_BODY.split())
    assert resp.output != resp.partial.text


def test_a_tool_that_declares_retrieved_context_gets_the_synthesis_turn():
    """The same script, with the tool declaring what its output is."""
    resp = _review_agent(context_retrieval=True).run(
        "Review calc.py and report any correctness risk."
    )
    assert resp.metadata.get("answer_source") in (None, "")
    assert not resp.metadata.get("partial")
    assert "zero divisor" in (resp.output or "")


def test_the_declaration_is_read_off_the_tool_instance():
    """A positive declaration only ever adds; it never reclassifies a tool down."""
    from effgen.tools.base_tool import BaseTool as _BaseTool

    assert _BaseTool.is_context_retrieval is False
    agent = _review_agent(context_retrieval=False)
    assert agent._is_context_retrieval_tool("file_operations") is False
    assert agent._is_context_retrieval_tool("retrieval") is True
    flagged = _review_agent(context_retrieval=True)
    assert flagged._is_context_retrieval_tool("file_operations") is True


def test_an_explicit_none_iteration_cap_falls_back_to_the_configured_one():
    """``max_iterations=None`` is what an unset optional flag forwards.

    It has to mean "use the configured cap", not reach the loop's comparison
    as ``None`` and raise a bare ``TypeError`` from inside the run.
    """
    model = _ScriptedModel([
        _calc_action("2 + 2"),
        "Thought: I have it.\nFinal Answer: 4",
    ])
    agent = _make_agent(model, max_iterations=6)

    response = agent.run("What is 2 + 2?", max_iterations=None)

    assert response.success
    assert "4" in (response.output or "")


def test_an_explicit_iteration_cap_of_zero_is_honored():
    """Zero means zero — the fallback must not read it as "unset"."""
    model = _ScriptedModel(["Thought: thinking.\nFinal Answer: 4"])
    agent = _make_agent(model, max_iterations=6)

    response = agent.run("What is 2 + 2?", max_iterations=0)

    assert model.calls == 0
    assert not response.success


# ---------------------------------------------------------------------------
# A recovery that could not obtain an answer says so
# ---------------------------------------------------------------------------
#
# The two loop fallbacks used to return the tool observation itself as the
# answer with success=True. For a retrieval tool that is a passage dump
# presented as if the model had written it — the same thing the iteration cap
# stopped doing — and a caller keyed on .success could not tell the difference.
# reason also read "final_answer" although no model wrote one.


def _retrieval_loop_agent(*, script=None):
    """An agent whose retrieval tool keeps returning the same passage.

    The model asks for it, is nudged to synthesize with the tools withdrawn,
    and asks again — the shape the report measured on the smallest local
    models and on groq's 8B.
    """
    model = _ScriptedModel(script or [_read_action("calc.py")])
    return Agent(config=AgentConfig(
        raise_on_error=False,  # these assert the failure response, not the raise
        name="retrieval-loop-test", model=model,
        tools=[_FileReadTool(context_retrieval=True)],
        max_iterations=4, tool_calling_mode="react",
    ))


class TestUnsynthesizedRetrievalIsNotASuccess:
    def test_it_reports_a_failure_carrying_the_observation(self):
        resp = _retrieval_loop_agent().run("Review calc.py and report any risk.")

        assert resp.success is False, "a passage is not an answer"
        assert resp.metadata.get("partial") is True
        # The observation travels beside the result, not as the result.
        assert FILE_BODY.split()[0] in " ".join(
            (resp.metadata.get("partial_output") or "").split()
        )
        assert resp.output != resp.metadata.get("partial_output")

    def test_the_outcome_is_typed_and_names_the_repeated_tool(self):
        resp = _retrieval_loop_agent().run("Review calc.py and report any risk.")

        error = resp.metadata.get("error") or {}
        assert error.get("type") == "UnsynthesizedToolResult"
        assert error.get("repeated_tool") == "file_operations"
        assert error.get("retryable") is False
        assert "file_operations" in (resp.output or "")

    def test_reason_says_what_ended_the_run_not_final_answer(self):
        resp = _retrieval_loop_agent().run("Review calc.py and report any risk.")

        assert resp.metadata.get("reason") in (
            "repeated_tool_result", "loop_detected",
        ), resp.metadata.get("reason")
        assert resp.metadata.get("reason") != "final_answer"

    def test_a_computed_result_that_repeats_is_reported_the_same_way(self):
        """A computing tool that repeats reports the stop, like a retrieval one.

        A calculator returning the same number twice is a confident
        *computation*; it is not an answer to the question, because the model
        never wrote one. Both categories therefore report the same shape and
        differ only in the wording of the statement.
        """
        model = _ScriptedModel([_calc_action("15^2"), _calc_action("15^2")])
        resp = _make_agent(model).run("Explain step by step and compute 15 squared")

        assert resp.success is False
        assert resp.outcome == "stopped"
        assert resp.stop_reason == "loop_detected"
        assert resp.metadata.get("partial") is True
        assert "225" in resp.partial.text
        assert "calculator" in (resp.output or "")
