"""Loop policy for a tool-calling turn loop.

An agent that drives a model's tool calling has to decide more than "did the
model ask for a tool": whether the same call is repeating, whether a tool has
reproduced a result it already returned, when to stop offering tools so the
model must write prose, and whether a turn wrote its call out as text instead of
making it. :class:`NativeToolLoop` holds that policy and the state it needs, so
the blocking loop in :mod:`effgen.core.agent_react` and the streaming loop in
:mod:`effgen.core.agent_stream_native` reach the same decisions from the same
code rather than from two copies of it.

The class is state plus predicates. It never calls a model, never dispatches a
tool and never builds a prompt — the caller does all of that and tells the loop
what happened. This module imports nothing from ``agent.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..tools.base_tool import ToolCategory
from .agent_runtime import (
    NUDGE_HAVE_ANSWER,
    NUDGE_HAVE_RESULTS,
    written_call_only,
)

#: Prefix :meth:`Agent._execute_tool` puts on a failed dispatch. A failed call is
#: not evidence of a repeated result, so the result-based short circuit skips it.
TOOL_ERROR_PREFIX = "Error executing tool"

#: How many calls to one tool count as a loop when the inputs keep changing.
#: Small models re-format the same call rather than repeating it byte for byte,
#: so the exact-pair check alone never fires for them.
FUZZY_LOOP_THRESHOLD = 5

#: The same threshold for a tool whose job is to chew through data, where
#: several calls in a row are the normal shape of the work.
FUZZY_LOOP_THRESHOLD_DATA = 7

#: Batch tool runs allowed before the loop stops offering tools. A model that
#: answers a multi-call turn with another multi-call turn is not converging.
MAX_BATCH_TOOL_RUNS = 2


@dataclass
class LoopCheck:
    """What :meth:`NativeToolLoop.check_action` found about one proposed call."""

    #: ``(action, normalized_input)``, the key the exact-repeat check uses.
    pair: tuple[str, str]
    #: How many times this tool was already called in this run.
    action_call_count: int
    #: The same tool with the same input has already been dispatched.
    is_exact_loop: bool
    #: The same tool has been called enough times to read as a loop.
    is_fuzzy_loop: bool

    @property
    def is_loop(self) -> bool:
        """True when either repeat check fired."""
        return self.is_exact_loop or self.is_fuzzy_loop

    @property
    def loop_type(self) -> str:
        """A short label for the log line, or ``""`` when nothing fired."""
        if self.is_exact_loop:
            return "exact"
        if self.is_fuzzy_loop:
            return f"fuzzy ({self.action_call_count + 1} calls)"
        return ""


@dataclass
class NativeToolLoop:
    """Per-run state and the decisions a tool-calling loop makes from it.

    Args:
        tools: The tools the agent holds, by name — the same mapping the loop
            dispatches against.
        nudge_cap: The configured iteration cap, used to decide when a turn is
            close enough to the limit to ask for an answer outright.
    """

    tools: dict[str, Any]
    nudge_cap: int = 10

    #: ``(action, normalized_input)`` for every call dispatched so far.
    previous_actions: list[tuple[str, str]] = field(default_factory=list)
    #: ``(action, normalized_result)`` for every call that returned without error.
    previous_results: list[tuple[str, str]] = field(default_factory=list)
    #: Multi-call turns dispatched as one batch.
    batch_tool_runs: int = 0
    #: Set once a repeat left no usable partial answer: stop offering tools so
    #: the model has to write the answer from what it already has.
    force_text_answer: bool = False
    #: The tool whose call a turn wrote out as text instead of making.
    written_call: str | None = None
    #: How many turns did that.
    written_call_turns: int = 0
    #: Tools this run actually dispatched.
    executed_tools: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Offering tools
    # ------------------------------------------------------------------
    def tools_suppressed(self) -> bool:
        """True when this run should stop passing tool definitions to the model.

        Either the model has spent its allowance of multi-call turns, or a
        repeat was detected with nothing usable to fall back on. In both cases
        re-offering the same tools reproduces the same turn.
        """
        return self.batch_tool_runs >= MAX_BATCH_TOOL_RUNS or self.force_text_answer

    def note_batch_run(self) -> None:
        """Record that a turn dispatched several calls at once."""
        self.batch_tool_runs += 1

    # ------------------------------------------------------------------
    # Repeat detection
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_input(action_input: str) -> str:
        """Return *action_input* in a form two equivalent calls share.

        JSON arguments are re-serialized with sorted keys so the same call
        written with its keys in a different order compares equal; anything that
        is not JSON is compared as trimmed text.
        """
        normalized = (action_input or "").strip()
        try:
            return json.dumps(json.loads(normalized), sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            return normalized

    def fuzzy_threshold(self, action: str) -> int:
        """How many calls to *action* read as a loop when the inputs differ."""
        tool = self.tools.get(action)
        category = getattr(getattr(tool, "metadata", None), "category", None)
        if category == ToolCategory.DATA_PROCESSING:
            return FUZZY_LOOP_THRESHOLD_DATA
        return FUZZY_LOOP_THRESHOLD

    def check_action(self, action: str, action_input: str) -> LoopCheck:
        """Report whether dispatching *action* now would repeat earlier work.

        Reads state only; :meth:`record_action` is what remembers the call.
        """
        pair = (action, self.normalize_input(action_input))
        known = action in self.tools
        action_call_count = sum(1 for a, _ in self.previous_actions if a == action)
        exact_count = sum(1 for seen in self.previous_actions if seen == pair)
        return LoopCheck(
            pair=pair,
            action_call_count=action_call_count,
            is_exact_loop=exact_count >= 1 and known,
            is_fuzzy_loop=action_call_count >= self.fuzzy_threshold(action) and known,
        )

    def record_action(self, check: LoopCheck) -> None:
        """Remember the call *check* describes as dispatched."""
        self.previous_actions.append(check.pair)

    def record_execution(self, action: str) -> None:
        """Remember that *action* actually ran in this turn."""
        self.executed_tools.add(action)

    # ------------------------------------------------------------------
    # Result repeats
    # ------------------------------------------------------------------
    @staticmethod
    def _result_key(action: str, tool_result: str) -> tuple[str, str]:
        return (action, " ".join(tool_result.split())[:500])

    def result_is_repeat(self, action: str, tool_result: str) -> bool:
        """True when *action* has already returned this result in this run.

        A tool that reproduces its own output means the answer is settled: the
        model is re-deriving something it already has. A failed dispatch is
        never a repeat.
        """
        if tool_result.startswith(TOOL_ERROR_PREFIX):
            return False
        return self._result_key(action, tool_result) in self.previous_results

    def record_result(self, action: str, tool_result: str) -> None:
        """Remember what *action* returned, unless the dispatch failed."""
        if tool_result.startswith(TOOL_ERROR_PREFIX):
            return
        self.previous_results.append(self._result_key(action, tool_result))

    # ------------------------------------------------------------------
    # Nudges
    # ------------------------------------------------------------------
    def post_tool_nudge(
        self, iteration: int, action_call_count: int, tool_result: str
    ) -> str | None:
        """Return the line to append after a tool ran, or ``None`` for silence.

        Near the cap the loop asks for the answer outright; a tool on its second
        run has almost certainly produced what the answer needs, so the model is
        pointed at the scratchpad before it drifts into re-planning.

        Args:
            iteration: The turn number that just ran, counted from one.
            action_call_count: How many times this tool had already been called
                before this turn.
            tool_result: What the dispatch returned, so a failed one earns no
                "you already have the answer" nudge.

        Returns:
            The line to append to the scratchpad, or ``None``.
        """
        if iteration >= self.nudge_cap - 2:
            return NUDGE_HAVE_ANSWER
        if action_call_count >= 1 and not tool_result.startswith(TOOL_ERROR_PREFIX):
            return NUDGE_HAVE_RESULTS
        return None

    # ------------------------------------------------------------------
    # Calls written out instead of made
    # ------------------------------------------------------------------
    def is_unmade_call(self, written: str, text: str) -> bool:
        """True when a written-out call block means the work never happened.

        Either the named tool never ran in this run, or the text is nothing but
        the call — a recap beside a real answer is neither.
        """
        return written not in self.executed_tools or written_call_only(text, self.tools)

    def note_written_call(self, written: str) -> bool:
        """Record a turn that wrote its call out; True once it is time to report.

        One such turn earns a nudge. A second means the model is not going to
        make the call, so the caller reports the cause instead of billing the
        rest of the iteration budget for the same outcome.
        """
        self.written_call = self.written_call or written
        self.written_call_turns += 1
        return self.written_call_turns > 1

    def tool_ran(self, name: str | None) -> bool:
        """True when *name* was dispatched at some point in this run."""
        return bool(name) and name in self.executed_tools
