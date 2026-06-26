"""Unit tests for ``sanitize_final_answer`` — scaffolding-leak elimination.

The fixtures here are the real messy strings captured from small models (groq
``llama-3.1-8b``, OpenAI native web-search) plus the literal loop-bookkeeping
markers the ReAct scaffolding injects. The sanitizer must remove every internal
marker while leaving legitimate answer content (including markdown tables and
code) untouched.
"""

import pytest

from effgen.core.agent import sanitize_final_answer as sanitize

# --- Real leaked strings from the audit ------------------------------------

@pytest.mark.parametrize(
    "leaked, expected",
    [
        # OpenAI native-tool output: "Canberra\nFinal Answer: Canberra"
        ("Canberra\nFinal Answer: Canberra", "Canberra"),
        # Groq llama-3.1-8b multi-tool join with the bookkeeping marker embedded.
        (
            "225 | 225\n[Tool results computed above. Continue or provide "
            "Final Answer:] | 450 | 225 | 675 | 0",
            "225 | 225 | 450 | 225 | 675 | 0",
        ),
        # Plain "Final Answer:" label.
        ("Final Answer: The capital is Paris.", "The capital is Paris."),
        ("Answer: 42", "42"),
        # Reasoning then a final-answer label — keep only the answer.
        ("Let me think. 15*15=225.\nFinal Answer: 225", "225"),
        # Trailing observation/thought bleed.
        ("Paris\nObservation: tool returned Paris", "Paris"),
        ("The result is 42.\nThought: I should double check", "The result is 42."),
        # Tool-echo fragment.
        ("[calculator({'expression': '15*15'})] → 225", "225"),
        # List-prefixed final-answer label.
        ("- Final Answer: done", "done"),
        # OpenAI native web-search replies sometimes end with a bare, dangling
        # "Final Answer:" label and nothing after it — strip the label, keep the
        # real answer that precedes it.
        (
            "The outage happened on July 19, 2024. ([blogs.microsoft.com](https://x))\n\nFinal Answer:",
            "The outage happened on July 19, 2024. ([blogs.microsoft.com](https://x))",
        ),
        ("Paris.\nFinal Answer: ", "Paris."),
        # The "you already have results from this tool" nudge the loop injects
        # when a tool has run twice — a small model (groq llama-3.1-8b) echoed it
        # straight into a pipe-joined answer. Must be stripped like the others.
        (
            "79.96 | 86.3568 [You already have results from this tool. If you "
            "have enough information, respond now with 'Final Answer:'.] | 85.49",
            "79.96 | 86.3568 | 85.49",
        ),
        # The "you already computed this" nudge injected on a repeated action.
        (
            "42 You already computed this. Please provide your final response "
            "using 'Final Answer:' now.",
            "42",
        ),
    ],
)
def test_strips_scaffolding(leaked, expected):
    assert sanitize(leaked) == expected


# --- Leaked model tool-call syntax (Llama-style) -----------------------------

@pytest.mark.parametrize(
    "leaked, expected",
    [
        # Groq llama-3.1-8b emitted its tool call as text instead of routing it.
        (
            'function=calculator>{"expression": "15*90", "operation": '
            '"calculate", "precision": 0}</function>',
            "",
        ),
        ('<function=calc>{"a": 1}</function>', ""),
        # Nested-object args must be matched whole, not leave a dangling "}".
        ('<function=calc>{"a": {"b": 1}}</function>', ""),
        (
            '<function=calculator>{"args": {"expression": "1+1"}, "op": "x"}'
            "</function>",
            "",
        ),
        ("<tool_call>{\"name\": \"calc\"}</tool_call>", ""),
        # Real answer with a trailing stray tag — keep the answer, drop the tag.
        ("The total is 1350 </function>", "The total is 1350"),
        # Special tokens.
        ("Paris <|eot_id|>", "Paris"),
    ],
)
def test_strips_tool_call_syntax(leaked, expected):
    assert sanitize(leaked) == expected


# --- Idempotency --------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Canberra\nFinal Answer: Canberra",
        "225 | 225\n[Tool results computed above. Continue or provide Final Answer:] | 0",
        "Final Answer: 42",
        "Just a plain answer.",
    ],
)
def test_idempotent(text):
    once = sanitize(text)
    assert sanitize(once) == once


# --- Must NOT corrupt legitimate content -------------------------------------

def test_markdown_table_preserved():
    table = "| Name | Age |\n|------|-----|\n| Bob | 30 |"
    assert sanitize(table) == table


def test_code_block_with_pipe_preserved():
    code = "```python\nresult = a | b\n```"
    assert sanitize(code) == code


def test_plain_answer_unchanged():
    assert sanitize("The capital of France is Paris.") == "The capital of France is Paris."


def test_answer_mentioning_observation_word_unchanged():
    # "observation" as a normal word (not a line-anchored ReAct label) stays.
    text = "My observation is that the sky is blue."
    assert sanitize(text) == text


def test_ascii_arrow_in_prose_preserved():
    # The tool-echo scaffolding always uses the Unicode arrow; legitimate prose
    # with an ASCII "->" (e.g. a footnote reference) must not be mangled.
    text = "Step A [1] -> Step B"
    assert sanitize(text) == text


# --- Edge cases ---------------------------------------------------------------

def test_none_passthrough():
    assert sanitize(None) is None


def test_empty_passthrough():
    assert sanitize("") == ""


def test_non_string_passthrough():
    assert sanitize(123) == 123  # type: ignore[arg-type]


def test_only_label_returns_label_stripped_text():
    # If stripping a label would empty the string, keep the (stripped) content.
    # "Final Answer:" with nothing after → empty tail, label kept as-is content.
    assert sanitize("Final Answer:") == "Final Answer:"


# --- Drift guard: injection sites and the strip-list share one source --------

def test_every_injected_nudge_is_stripped():
    """Each loop nudge the ReAct scaffolding can inject must be sanitized away.

    The nudge strings are defined once in ``agent_runtime`` and referenced by
    the injection sites in ``agent_react``; this guards against a future nudge
    being added at an injection site without being added to the strip-list (the
    bug that let "[You already have results from this tool…]" reach a user).
    """
    import effgen.core.agent_runtime as rt

    nudges = [v for k, v in vars(rt).items() if k.startswith("NUDGE_")]
    assert nudges, "expected NUDGE_* constants to exist"
    for nudge in nudges:
        # Embedded mid-answer (the way a small model echoes it) must vanish.
        leaked = f"42 {nudge} done"
        assert nudge not in sanitize(leaked), f"nudge not stripped: {nudge!r}"


def test_injection_sites_use_the_shared_nudges():
    """The agent_react injection sites must reference the shared NUDGE_* values.

    Reading the source keeps the two files honest: every literal a loop appends
    should be one of the named constants (so it is also on the strip-list).
    """
    import inspect

    import effgen.core.agent_react as ar
    import effgen.core.agent_runtime as rt

    src = inspect.getsource(ar)
    for name in ("NUDGE_CONTINUE", "NUDGE_HAVE_ANSWER", "NUDGE_HAVE_RESULTS",
                 "NUDGE_ALREADY_COMPUTED", "NUDGE_NO_TOOLS", "NUDGE_NOT_USABLE"):
        assert getattr(rt, name, None), f"missing nudge constant {name}"
        assert name in src, f"agent_react no longer references {name}"
