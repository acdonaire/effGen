"""
Canonical parity task for effGen backend interchangeability tests.

Task: "What is (17 * 23) + sqrt(144)?"
Expected answer: 403  (= 391 + 12)

All backends must produce an output that contains "403" using the Calculator tool.
"""

from __future__ import annotations

CANONICAL_QUESTION = "What is (17 * 23) + sqrt(144)?"
CANONICAL_ANSWER = "403"

# Calculator tool schema in OpenAI function-calling format
CALCULATOR_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the numeric result. "
            "Supports +, -, *, /, **, sqrt(), and standard Python math."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression, e.g. '17 * 23' or 'math.sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
}


def run_canonical_task(adapter, strategy: str = "react") -> dict:
    """Run the canonical task against *adapter* and return a result dict.

    Args:
        adapter: A loaded effGen BaseModel adapter instance.
        strategy: "react" or "native".  For "react" the agent loop handles
                  tool parsing; for "native" we use the adapter's
                  ``generate_with_tools`` method directly.

    Returns:
        dict with keys:
            answer_correct  — bool
            answer_text     — the final answer string
            tool_calls      — int, number of tool invocations recorded
            latency_ms      — end-to-end wall time in milliseconds
            error           — str or None
    """
    import time

    from effgen import Agent
    from effgen.core.agent import AgentConfig
    from effgen.tools.builtin import Calculator

    t0 = time.monotonic()
    error = None
    answer_text = ""
    tool_calls_count = 0

    try:
        agent = Agent(
            config=AgentConfig(
                name="parity_agent",
                model=adapter,
                tools=[Calculator()],
                max_iterations=8,
                temperature=0.0,
                enable_memory=False,
                enable_sub_agents=False,
                tool_calling_mode=strategy if strategy != "react" else "react",
            )
        )
        result = agent.run(CANONICAL_QUESTION)
        answer_text = result.output or ""
        tool_calls_count = result.tool_calls
        # Propagate agent-level errors (e.g., Replicate billing) from metadata
        if not result.success and result.output:
            # Check if output contains a known non-answer (billing message etc.)
            if "billing" in result.output.lower() or "insufficient credits" in result.output.lower():
                error = f"BillingError: {result.output[:200]}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.monotonic() - t0) * 1000
    answer_correct = CANONICAL_ANSWER in answer_text

    return {
        "answer_correct": answer_correct,
        "answer_text": answer_text,
        "tool_calls": tool_calls_count,
        "latency_ms": latency_ms,
        "error": error,
    }
