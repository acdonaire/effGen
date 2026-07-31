"""ReAct loop, response parsing, and native-tool internals for :class:`Agent`.

Holds the ReAct reasoning loop, its response parsers, sub-agent delegation, and
the provider-native tool-calling run paths. Tool dispatch lives in
:class:`~effgen.core.agent_tool_execution.AgentToolExecutionMixin` and citation
assembly in :class:`~effgen.core.agent_citations.AgentCitationsMixin`;
:class:`AgentReActMixin` inherits both so every method resolves on
:class:`Agent` exactly as before.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from ..models._adapter_utils import default_max_output_tokens
from ..models.base import GenerationConfig
from ..observability import get_logger as _get_obs_logger
from ..observability.spans import ModelAttrs, ToolAttrs
from ..observability.tracing import (
    start_agent_iteration,
    start_model_call,
    start_tool_call,
)
from ..tools.base_tool import ToolCategory
from ..utils.prometheus_metrics import metrics as prom_metrics
from ..utils.structured_logging import (
    get_structured_logger,
)
from .agent_citations import AgentCitationsMixin
from .agent_tool_execution import AgentToolExecutionMixin
from .execution_tracker import EventType, ExecutionEvent
from .router import RoutingDecision, RoutingStrategy
from .tool_calling import (
    ToolCallResult,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
_slog = get_structured_logger(__name__)
# Canonical structured observability logger — emits redacted JSON lines with OTel context
_obs_log = _get_obs_logger(__name__)

from .agent import AgentMode, AgentResponse  # noqa: E402
from .agent_runtime import (  # noqa: E402
    _SCAFFOLD_LITERAL_RES,
    _TOOL_ECHO_RE,
    CONTEXT_ANSWER_INSTRUCTION,
    CONTINUE_INSTRUCTION,
    NUDGE_ALREADY_COMPUTED,
    NUDGE_CONTINUE,
    NUDGE_HAVE_ANSWER,
    NUDGE_HAVE_RESULTS,
    NUDGE_NO_TOOLS,
    NUDGE_NOT_USABLE,
    _infer_provider_from_model,
    find_written_tool_call,
    sanitize_final_answer,
    written_call_only,
)

#: Models that complete a tool loop through the provider's tool-calling API,
#: named in the hint a written-out call block produces. Kept short and stable;
#: ``effgen models list`` marks every model that advertises tool calling.
_TOOL_CALLING_EXAMPLES = (
    "openai:gpt-5-nano, gemini:gemini-3.1-flash-lite or groq:llama-3.3-70b-versatile"
)


class AgentReActMixin(AgentToolExecutionMixin, AgentCitationsMixin):
    """ReAct-loop, tool-execution and native-tool methods for :class:`Agent`."""

    def _run_single_agent(self,
                         task: str,
                         context: dict[str, Any],
                         **kwargs) -> AgentResponse:
        """
        Execute task using single agent with ReAct loop or direct inference.

        Args:
            task: Task description
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # Extract debug flags (set by run())
        debug = kwargs.pop("_debug", False)
        run_id = kwargs.pop("_run_id", "")
        # Pop custom kwargs so they don't leak to the model layer.
        # We re-read them locally below before they would propagate further.
        _ckpt_interval_arg = kwargs.pop("checkpoint_interval", 0) or 0
        _ckpt_dir_arg = kwargs.pop("checkpoint_dir", None)
        _resume_scratchpad_arg = kwargs.pop("_resume_scratchpad", None)

        # Structured multimodal inputs must reach adapters as Message parts.
        # The ReAct prompt is text-only, so use direct inference for these calls
        # even when the preset includes tools.
        if kwargs.get("inputs") is not None:
            return self._run_direct_inference(task, context, **kwargs)

        # If no tools available, use direct inference instead of ReAct
        if not self.tools:
            return self._run_direct_inference(task, context, **kwargs)

        # If any native OpenAI tools are present and the model supports it,
        # route through the Responses API directly (not the ReAct loop).
        if self._has_native_tools():
            return self._run_with_native_tools(task, context, **kwargs)

        # If any Gemini native tools are present, route through the Gemini
        # native-tool path which passes tool objects directly to the adapter.
        if self._has_gemini_native_tools():
            return self._run_with_gemini_native_tools(task, context, **kwargs)

        iterations = 0
        tool_calls = 0
        tokens_used = 0
        scratchpad = ""
        max_iterations = kwargs.get("max_iterations", self.config.max_iterations)

        # Debug trace collector
        debug_trace = None
        if debug:
            from ..debug.inspector import DebugTrace
            debug_trace = DebugTrace(
                task=task, agent_name=self.name, run_id=run_id,
            )

        # Format conversation history
        conversation_history = self._format_conversation_history()

        # ReAct loop
        previous_actions: list[tuple[str, str]] = []  # Track (action, input) pairs for loop detection
        previous_results: list[tuple[str, str]] = []  # Track (action, result) pairs to stop on a confident, repeated answer
        _batch_tool_runs = 0  # Count of batch native-tool runs; cap at 2 to prevent infinite loops
        # Set once a repeated action loops with no usable partial answer (every
        # attempt failed/was denied) — stops re-offering tools so the model
        # must synthesize a prose answer instead of retrying forever.
        _force_text_answer = False
        # The tool whose call the model wrote out as text instead of making,
        # and how many turns did it — one nudge is allowed before the run is
        # reported as a turn that ran no tool.
        _written_call: str | None = None
        _written_call_turns = 0
        # Tools this run actually dispatched. An answer that recaps a call the
        # run really made is an answer, not a call the model only described.
        _executed_tools: set[str] = set()

        def _is_unmade_call(written: str, text: str) -> bool:
            """True when a written-out call block means the work never happened.

            Either the named tool never ran in this run, or the text is nothing
            but the call — a recap beside a real answer is neither.
            """
            return written not in _executed_tools or written_call_only(text, self.tools)

        # Optional periodic checkpointing
        _ckpt_interval = _ckpt_interval_arg
        _ckpt_dir = _ckpt_dir_arg
        _ckpt_mgr = None
        if _ckpt_interval and _ckpt_dir:
            try:
                from .checkpoint import CheckpointManager as _CM
                _ckpt_mgr = _CM(_ckpt_dir)
            except Exception as _e:
                logger.warning("Failed to init CheckpointManager: %s", _e)
        # Allow resuming with a seeded scratchpad
        if _resume_scratchpad_arg:
            scratchpad = _resume_scratchpad_arg
        while iterations < max_iterations:
            iterations += 1
            iter_start = time.time()
            if _ckpt_mgr is not None and iterations > 1 and (iterations - 1) % _ckpt_interval == 0:
                try:
                    from .checkpoint import CheckpointManager as _CM2
                    cp = _CM2.snapshot_agent(
                        self,
                        task=task,
                        iteration=iterations,
                        scratchpad=scratchpad,
                        tool_calls=tool_calls,
                        tokens_used=tokens_used,
                        metadata={"interval": _ckpt_interval},
                    )
                    self._last_checkpoint_id = _ckpt_mgr.save(cp)
                except Exception as _e:
                    logger.warning("Periodic checkpoint failed: %s", _e)

            # Determine if we should use native tool calling prompt format
            use_native_prompt = (
                self._tool_calling_strategy.name in ("native", "hybrid")
                and self.model is not None
                and hasattr(self.model, 'supports_tool_calling')
                and self.model.supports_tool_calling()
            )

            # Build prompt
            gen_kwargs = dict(kwargs)
            # After 2 multi-tool batches, or once a loop with no usable partial
            # answer was detected, stop passing tools to force synthesis.
            if _batch_tool_runs >= 2 or _force_text_answer:
                use_native_prompt = False
            if use_native_prompt and not self.config.system_prompt_template:
                # Native/hybrid mode: use a simple user message and pass
                # tool definitions via the chat template's tools parameter.
                # The model will produce native tool call tokens (e.g.
                # <tool_call> for Qwen, [TOOL_CALLS] for Mistral).
                if scratchpad:
                    prompt = (
                        f"{task}\n\n"
                        f"Previous steps:\n{scratchpad}\n\n"
                        f"{self._continuation_instruction(previous_actions)}"
                    )
                else:
                    prompt = task
                # Carry prior conversation turns into the native tool-calling
                # prompt. Without this the model only sees the latest task and
                # forgets earlier turns, so a multi-turn *session* loses its
                # context the moment any tool is attached (the ReAct/template
                # branches below already inject this history).
                if conversation_history:
                    prompt = f"{conversation_history}\n\n{prompt}"
                # Steer the model with the user's custom persona. The native/
                # hybrid path sends a bare user message (the chat template owns
                # the system slot for tools), so prepend the persona — otherwise
                # a custom persona is dropped the moment a tool is attached, even
                # though the ReAct-text and Gemini-native paths honor it.
                prompt = f"{self._persona_prefix()}{prompt}"
                # Pass tool definitions for the chat template
                tool_defs = self._tool_calling_strategy.format_tools_for_prompt(
                    list(self.tools.values())
                )
                if isinstance(tool_defs, list):
                    gen_kwargs["tools"] = tool_defs
            elif self.config.system_prompt_template:
                # User-provided custom template
                tools_description = self._get_tools_description()
                prompt = self.config.system_prompt_template.format(
                    tools_description=tools_description,
                    conversation_history=conversation_history,
                    task=task,
                    scratchpad=scratchpad
                )
            else:
                # ReAct mode: use enhanced ToolPromptGenerator
                prompt = self._tool_prompt_generator.generate_react_prompt(
                    task=task,
                    scratchpad=scratchpad,
                    conversation_history=conversation_history,
                    system_prompt=self.config.system_prompt,
                    verbose=self._verbose_tools,
                    closing_instruction=self._context_answer_instruction(previous_actions),
                )

            # Debug: log first iteration prompt to see if history is included
            if iterations == 1 and conversation_history:
                logger.info(f"[Memory] Including conversation history ({len(self.short_term_memory.messages)} messages)")

            # Track reasoning step
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.REASONING_STEP,
                agent_id=self.name,
                message=f"Iteration {iterations}: Reasoning...",
                data={"iteration": iterations}
            ))

            # Generate response inside tracing span
            with start_agent_iteration(preset=self.name, iteration=iterations):
                model_name = getattr(self, "model_name", None) or "unknown"
                provider = _infer_provider_from_model(self.model, model_name)
                with start_model_call(provider=provider, model=model_name) as _mspan:
                    response = self._generate(prompt, **gen_kwargs)
                    # Annotate span with token counts from response
                    _meta = response.get("metadata") or {}
                    _in_tok = _meta.get("prompt_tokens", 0) or 0
                    _out_tok = response.get("tokens_used", 0) or 0
                    _cached = _meta.get("cached_input_tokens", 0) or 0
                    try:
                        _mspan.set_attribute(ModelAttrs.INPUT_TOKENS, int(_in_tok))
                        _mspan.set_attribute(ModelAttrs.OUTPUT_TOKENS, int(_out_tok))
                        if _cached:
                            _mspan.set_attribute(ModelAttrs.CACHED_TOKENS, int(_cached))
                        _mspan.set_attribute(ModelAttrs.OUTCOME, "ok" if response.get("finish_reason") != "error" else "error")
                    except Exception:
                        logger.debug("Failed to set model span attributes", exc_info=True)
                iter_tokens = response.get("tokens_used", 0)
                tokens_used += iter_tokens

            _slog.iteration_event(iterations, "generate", tokens=iter_tokens)
            _obs_log.event("agent.iteration.generate", iteration=iterations, tokens=iter_tokens, model=getattr(self, "model_name", "unknown"))

            if response.get("finish_reason") == "error":
                return self._generation_failure_response(
                    response,
                    iterations=iterations,
                    tool_calls=tool_calls,
                    tokens=tokens_used,
                    debug_trace=debug_trace,
                )

            # Debug: Log the raw response
            logger.info(f"[Iteration {iterations}] Raw model output: {response['text'][:300]}...")
            logger.debug(f"[Iteration {iterations}] Full model output: {response['text']}")

            # Parse response using strategy. If the adapter returned a native
            # tool call (empty text + structured tool_calls in metadata), use
            # it directly — no text parsing needed.
            native_tool_calls = response.get("tool_calls") or []

            # Execute ALL native tool calls in one batch (OpenAI/Cerebras can
            # return multiple tool_calls in a single response).
            if len(native_tool_calls) > 1 and self.tools:
                batch_observations: list[str] = []
                for _tc in native_tool_calls:
                    _fn = _tc.get("function", _tc)
                    _tname = _fn.get("name", "")
                    _targs = _fn.get("arguments", {})
                    if isinstance(_targs, str):
                        try:
                            _targs = json.loads(_targs)
                        except (json.JSONDecodeError, TypeError):
                            _targs = {"__raw_input__": _targs}
                    if _tname in self.tools:
                        with start_tool_call(tool_name=_tname, tool_input=str(_targs)[:500]) as _btspan:
                            _obs = self._execute_tool(_tname, json.dumps(_targs))
                            try:
                                _btspan.set_attribute(ToolAttrs.STATUS, "ok")
                            except Exception:
                                logger.debug("Failed to set tool span status", exc_info=True)
                        tool_calls += 1
                        _executed_tools.add(_tname)
                        batch_observations.append(f"[{_tname}({_targs})] → {_obs}")
                        scratchpad += f"\nAction: {_tname}\nAction Input: {json.dumps(_targs)}\nObservation: {_obs}"
                    else:
                        batch_observations.append(f"[{_tname}] → Tool not found")
                # After batch execution, nudge model to synthesize a final answer.
                scratchpad += f"\n{NUDGE_CONTINUE}"
                _batch_tool_runs += 1
                parsed = {"thought": "", "action": None, "action_input": None, "final_answer": None}
                cur_observation = "\n".join(batch_observations)
                logger.info(f"[Batch native tool calls] {len(native_tool_calls)} calls executed (batch run #{_batch_tool_runs})")
            elif native_tool_calls:
                strategy_result = self._parse_native_tool_calls(native_tool_calls)
                # Convert to legacy dict format for compatibility with rest of loop
                parsed = self._tool_call_result_to_dict(strategy_result)
            else:
                strategy_result = self._tool_calling_strategy.parse_response(
                    response["text"], tools=self.tools,
                )
                # Convert to legacy dict format for compatibility with rest of loop
                parsed = self._tool_call_result_to_dict(strategy_result)

            # Debug: Log what was parsed
            logger.info(f"[Iteration {iterations}] Parsed - Action: {parsed.get('action')}, Input: {parsed.get('action_input')}, Final: {parsed.get('final_answer')}")

            # Add to scratchpad
            scratchpad += f"\nThought: {parsed.get('thought', '')}"

            # Capture debug iteration data
            cur_observation = None  # filled later if tool runs

            def _build_response(
                output: str,
                success: bool = True,
                _tokens_used: int = tokens_used,
                _iterations: int = iterations,
                _tool_calls: int = tool_calls,
                _iter_start: float = iter_start,
                **extra_meta: Any,
            ) -> AgentResponse:
                """Helper to build response and attach debug trace."""
                if success:
                    raw_answer = output
                    output = sanitize_final_answer(output) or output
                    # An answer that writes out a call for a tool this agent
                    # holds means the tool never ran: the turn describes work
                    # that did not happen, so it is reported as a failure.
                    # Sanitizing a tagged call can leave its arguments behind as
                    # a bare JSON fragment, so the text as the model wrote it is
                    # scanned as well as the cleaned answer.
                    written = find_written_tool_call(
                        output, self.tools
                    ) or find_written_tool_call(raw_answer, self.tools)
                    if written and _is_unmade_call(written, raw_answer):
                        return self._written_tool_call_response(
                            written,
                            output,
                            iterations=_iterations,
                            tool_calls=_tool_calls,
                            tokens_used=_tokens_used,
                            tool_ran=written in _executed_tools,
                            debug_trace=debug_trace,
                        )
                meta: dict[str, Any] = {
                    "reason": "final_answer",
                    "tool_calling_strategy": self._tool_calling_strategy.name,
                }
                meta.update(extra_meta)
                if debug_trace is not None:
                    debug_trace.total_tokens = _tokens_used
                    debug_trace.total_latency = time.time() - (_iter_start - (_iterations - 1) * 0.001)
                    debug_trace.final_answer = output if success else None
                    debug_trace.success = success
                    meta["debug_trace"] = debug_trace
                return AgentResponse(
                    output=output,
                    success=success,
                    mode=AgentMode.SINGLE,
                    iterations=_iterations,
                    tool_calls=_tool_calls,
                    tokens_used=_tokens_used,
                    metadata=meta,
                )

            # Check for final answer
            final_answer = parsed.get("final_answer")
            if final_answer and tool_calls > 0 and final_answer.strip().lower() in {
                "none",
                "null",
                "n/a",
                "na",
            }:
                partial = self._extract_partial_answer(scratchpad)
                if partial:
                    logger.info(
                        "Ignoring null-like final answer after tool execution; "
                        "returning latest observation"
                    )
                    return _build_response(partial, answer_source="null_final_from_model", partial=True)

            # A "final answer" that is purely leaked tool-call syntax /
            # scaffolding (sanitizes to nothing) is not a real answer — keep
            # looping so the tool actually runs or a partial is extracted. When
            # what leaked is a call for a tool this agent holds, the model is
            # writing the call instead of making it: nudge once, then report it
            # rather than billing the rest of the iteration budget for the same
            # outcome.
            if final_answer and not (sanitize_final_answer(final_answer) or "").strip():
                written = find_written_tool_call(final_answer, self.tools)
                if written and _is_unmade_call(written, final_answer):
                    _written_call = _written_call or written
                    _written_call_turns += 1
                    if _written_call_turns > 1:
                        return self._written_tool_call_response(
                            _written_call,
                            final_answer,
                            iterations=iterations,
                            tool_calls=tool_calls,
                            tokens_used=tokens_used,
                            tool_ran=_written_call in _executed_tools,
                            debug_trace=debug_trace,
                        )
                logger.info(
                    "Discarding scaffolding-only final answer; continuing loop"
                )
                scratchpad += f"\nObservation: {NUDGE_NOT_USABLE}"
                final_answer = None

            if final_answer:
                # Record final debug iteration
                if debug_trace is not None:
                    from ..debug.inspector import DebugIteration
                    debug_trace.iterations.append(DebugIteration(
                        iteration=iterations,
                        raw_prompt=prompt[:2000],
                        raw_response=response["text"][:2000],
                        thought=parsed.get("thought", ""),
                        final_answer=final_answer,
                        tokens_used=iter_tokens,
                        latency=time.time() - iter_start,
                        scratchpad_snapshot=scratchpad,
                    ))
                return _build_response(final_answer)

            # Check if model is stating an answer without "Final Answer:" keyword
            # This happens when model provides result after tool execution
            if tool_calls > 0 and not parsed.get("action"):
                # No action and we've used tools - model might be stating the answer
                response_text = response["text"].strip()
                # Check for answer-like patterns
                if any(phrase in response_text.lower() for phrase in ["the answer is", "the result is", "the sum is", "equals", "="]):
                    logger.info("Detected answer statement without 'Final Answer:' keyword")
                    if debug_trace is not None:
                        from ..debug.inspector import DebugIteration
                        debug_trace.iterations.append(DebugIteration(
                            iteration=iterations,
                            raw_prompt=prompt[:2000],
                            raw_response=response_text[:2000],
                            thought=parsed.get("thought", ""),
                            final_answer=response_text,
                            tokens_used=iter_tokens,
                            latency=time.time() - iter_start,
                            scratchpad_snapshot=scratchpad,
                        ))
                    return _build_response(response_text)

            # Execute action if present
            if parsed.get("action") and parsed.get("action_input"):
                action = parsed["action"]
                action_input = parsed["action_input"]

                # Loop detection: check if we've seen this exact (action, input) before
                # Also detect fuzzy loops: same tool called 3+ times with different inputs
                # (SLMs like Llama produce slightly different formatting each time)
                # Normalize action_input for comparison
                normalized_input = action_input.strip()
                try:
                    parsed_json = json.loads(normalized_input)
                    normalized_input = json.dumps(parsed_json, sort_keys=True)
                except (json.JSONDecodeError, TypeError):
                    pass
                current_pair = (action, normalized_input)
                action_call_count = sum(1 for a, _ in previous_actions if a == action)
                exact_loop_count = sum(1 for pair in previous_actions if pair == current_pair)
                is_exact_loop = exact_loop_count >= 1 and action in self.tools
                fuzzy_threshold = 5
                if action in self.tools:
                    tool = self.tools[action]
                    if hasattr(tool, 'metadata') and hasattr(tool.metadata, 'category'):
                        if tool.metadata.category == ToolCategory.DATA_PROCESSING:
                            fuzzy_threshold = 7
                is_fuzzy_loop = action_call_count >= fuzzy_threshold and action in self.tools
                if is_exact_loop or is_fuzzy_loop:
                    loop_type = "exact" if is_exact_loop else f"fuzzy ({action_call_count + 1} calls)"
                    logger.info(
                        f"[Loop detected] Repeated action '{action}' ({loop_type}) — "
                        f"breaking loop and returning last observation"
                    )
                    # Extract the last successful observation from scratchpad
                    partial = self._extract_partial_answer(scratchpad)
                    # A retrieval/search tool's observation is source material,
                    # so returning it here hands back a passage dump in place of
                    # an answer. The model already has what it retrieved: stop
                    # offering tools and give it one turn to write the answer
                    # from the scratchpad before falling back to the passages.
                    if (
                        partial
                        and self._is_context_retrieval_tool(action)
                        and not _force_text_answer
                    ):
                        _force_text_answer = True
                        scratchpad += (
                            f"\nAction: {action}"
                            f"\nAction Input: {action_input}"
                            f"\nObservation: {NUDGE_HAVE_RESULTS}"
                        )
                        continue
                    if partial:
                        return _build_response(
                            partial,
                            answer_source="loop_detected",
                            repeated_action=action,
                            partial=True,
                        )
                    # No partial answer to fall back on — every attempt of this
                    # action failed or was denied, so simply nudging and
                    # re-offering the same tool just repeats the loop until
                    # max_iterations (the model keeps retrying the tool it was
                    # just told is already computed). Stop offering tools for
                    # the rest of this run so the model must respond in prose.
                    _force_text_answer = True
                    scratchpad += (
                        f"\nAction: {action}"
                        f"\nAction Input: {action_input}"
                        f"\nObservation: {NUDGE_ALREADY_COMPUTED}"
                    )
                    continue

                previous_actions.append(current_pair)

                # Check if tool is available (handle no-tool mode without raising)
                if not self.tools or action not in self.tools:
                    # No tools available - model is hallucinating tools
                    # Guide it to provide direct answer
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {NUDGE_NO_TOOLS}"
                else:
                    # Execute tool inside tracing span
                    tool_start = time.time()
                    with start_tool_call(tool_name=action, tool_input=str(action_input)) as _tspan:
                        tool_result = self._execute_tool(action, action_input)
                        try:
                            _tspan.set_attribute(ToolAttrs.STATUS, "ok")
                        except Exception:
                            logger.debug("Failed to set tool span status", exc_info=True)
                    tool_elapsed = time.time() - tool_start
                    tool_calls += 1
                    _executed_tools.add(action)
                    cur_observation = tool_result

                    # Metrics for tool execution
                    tool_labels = {"tool_name": action, "agent_name": self.name}
                    prom_metrics.tool_calls.inc(labels=tool_labels)
                    prom_metrics.tool_execution_time.observe(tool_elapsed, labels=tool_labels)
                    _slog.tool_event(action, "executed", latency=tool_elapsed)
                    _obs_log.tool_event("executed", tool=action, latency_ms=round(tool_elapsed * 1000, 1))

                    # Add observation to scratchpad
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {tool_result}"

                    # Log the observation for debugging
                    logger.info(f"Tool result added to scratchpad: {tool_result[:100]}...")

                    if self._should_return_direct_calculator_result(task, action, action_input):
                        logger.info(
                            "Returning direct calculator result for simple arithmetic task"
                        )
                        return _build_response(
                            tool_result,
                            _tool_calls=tool_calls,
                            answer_source="direct_calculator_result",
                        )

                    # Result-based short-circuit: small models often re-derive an
                    # answer they already have (e.g. "15^2" then "15*15", both
                    # 225) with slightly different inputs, so the exact-input loop
                    # guard never fires. If a tool reproduces a result it already
                    # returned, the answer is confident — stop and return it
                    # instead of burning iterations on redundant re-planning.
                    if not tool_result.startswith("Error executing tool"):
                        normalized_result = " ".join(tool_result.split())[:500]
                        result_key = (action, normalized_result)
                        if result_key in previous_results:
                            # A retrieval/search tool's output is context, not a
                            # synthesized answer, so returning it verbatim hands
                            # back a passage dump. The observation is already in
                            # the scratchpad: stop offering tools and give the
                            # model one turn to write the answer from it. A
                            # compute tool (e.g. calculator) reproducing its
                            # result is a confident answer and is returned as-is.
                            if (
                                self._is_context_retrieval_tool(action)
                                and not _force_text_answer
                            ):
                                logger.info(
                                    "[Loop efficiency] Retrieval tool '%s' repeated a "
                                    "result; asking for a synthesized answer",
                                    action,
                                )
                                _force_text_answer = True
                                scratchpad += f"\n{NUDGE_HAVE_RESULTS}"
                                continue
                            logger.info(
                                "[Loop efficiency] Tool '%s' reproduced an identical "
                                "result; returning it as the final answer",
                                action,
                            )
                            extra = (
                                {"partial": True}
                                if self._is_context_retrieval_tool(action)
                                else {}
                            )
                            return _build_response(
                                tool_result,
                                _tool_calls=tool_calls,
                                answer_source="repeated_tool_result",
                                **extra,
                            )
                        previous_results.append(result_key)

                    # Nudge model to answer when iterations are running low
                    if iterations >= self.config.max_iterations - 2:
                        scratchpad += f"\n{NUDGE_HAVE_ANSWER}"
                    elif action_call_count >= 1 and not tool_result.startswith("Error executing tool"):
                        # This tool has now run at least twice. The needed data is
                        # almost certainly in the scratchpad — nudge toward a final
                        # answer before the loop drifts into repeated re-planning.
                        scratchpad += f"\n{NUDGE_HAVE_RESULTS}"

            else:
                # A turn that produced neither an action nor an answer, but did
                # write out a call for a tool this agent holds, is the same
                # failure the answer path reports: the model is writing the call
                # instead of making it. Say so once, and on a second such turn
                # report the cause rather than grinding to the iteration cap
                # and reporting only that the cap was reached.
                written = find_written_tool_call(response["text"], self.tools)
                if written and _is_unmade_call(written, response["text"]):
                    _written_call = _written_call or written
                    _written_call_turns += 1
                    if _written_call_turns > 1:
                        return self._written_tool_call_response(
                            _written_call,
                            response["text"],
                            iterations=iterations,
                            tool_calls=tool_calls,
                            tokens_used=tokens_used,
                            tool_ran=_written_call in _executed_tools,
                            debug_trace=debug_trace,
                        )
                    scratchpad += f"\nObservation: {NUDGE_NOT_USABLE}"
                # No action specified, prompt to continue
                scratchpad += "\nAction: (continue reasoning)"

            # Record debug iteration
            if debug_trace is not None:
                from ..debug.inspector import DebugIteration
                debug_trace.iterations.append(DebugIteration(
                    iteration=iterations,
                    raw_prompt=prompt[:2000],
                    raw_response=response["text"][:2000],
                    thought=parsed.get("thought", ""),
                    action=parsed.get("action"),
                    action_input=parsed.get("action_input"),
                    observation=cur_observation,
                    tokens_used=iter_tokens,
                    latency=time.time() - iter_start,
                    scratchpad_snapshot=scratchpad,
                ))

        # Max iterations reached. When every turn wrote its tool call out as
        # text and nothing ran, the cap is a symptom: report the cause instead.
        partial_answer = self._extract_partial_answer(scratchpad)
        if _written_call and not partial_answer:
            return self._written_tool_call_response(
                _written_call,
                "",
                iterations=iterations,
                tool_calls=tool_calls,
                tokens_used=tokens_used,
                tool_ran=_written_call in _executed_tools,
                debug_trace=debug_trace,
            )
        # The run stopped without a final answer. Whatever the scratchpad holds
        # is a tool observation or a half-finished thought — source material, not
        # something the model wrote as its answer — so it is reported as progress
        # under ``partial_output`` and the outcome itself states what happened
        # and what to do about it.
        if partial_answer:
            partial_answer = sanitize_final_answer(partial_answer) or partial_answer
        detail = self._iteration_cap_detail(max_iterations, partial_answer)
        meta: dict[str, Any] = {
            "reason": (
                "max_iterations_partial" if partial_answer else "max_iterations_exhausted"
            ),
            "error": detail,
            "tool_calling_strategy": self._tool_calling_strategy.name,
        }
        if partial_answer:
            meta["partial"] = True
            meta["partial_output"] = partial_answer
            logger.info(
                "Max iterations reached; reporting the cap with the recovered "
                "progress under partial_output"
            )
        else:
            logger.info("Max iterations reached with no recoverable progress")
        if debug_trace is not None:
            debug_trace.total_tokens = tokens_used
            debug_trace.final_answer = None
            debug_trace.success = False
            meta["debug_trace"] = debug_trace
        return AgentResponse(
            output=detail["message"],
            success=False,
            mode=AgentMode.SINGLE,
            iterations=iterations,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            metadata=meta,
        )

    def _model_advertises_tool_calling(self) -> bool:
        """True when the loaded model reports native tool-calling support."""
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "supports_tool_calling"):
            return False
        try:
            return bool(model.supports_tool_calling())
        except Exception:  # noqa: BLE001 - a capability probe never breaks a run
            logger.debug("Tool-calling capability check failed", exc_info=True)
            return False

    def _written_tool_call_detail(
        self, tool_name: str, answer: str, *, tool_ran: bool = False,
    ) -> dict[str, Any]:
        """Return the typed error for an answer that writes out a tool call.

        The remediation depends on which tool-calling path ran: a model that was
        sent the tool definitions natively and still answered with the call as
        text needs replacing, while a model that advertises native tool calling
        but ran the ReAct text protocol only needs to be asked for the native
        path. *tool_ran* says whether the named tool was dispatched earlier in
        the run, which decides what the answer failed to do.
        """
        strategy = self._tool_calling_strategy.name
        model_id = (
            getattr(self.model, "model_name", None) or self.model_name or "the model"
        )
        advertises = self._model_advertises_tool_calling()
        if strategy in ("native", "hybrid") and advertises:
            remedy = (
                f"'{model_id}' was sent the tool definitions through the "
                "provider's tool-calling API and answered with the call as text "
                f"anyway. Run the task on a model that calls tools — "
                f"{_TOOL_CALLING_EXAMPLES} — or on a larger local model."
            )
        elif advertises:
            remedy = (
                f"This run used the ReAct text protocol, but '{model_id}' "
                "advertises native tool calling: build the agent with "
                "AgentConfig(tool_calling_mode='native') so the tool definitions "
                "reach the provider's tool-calling API."
            )
        else:
            remedy = (
                f"'{model_id}' does not advertise native tool calling. Run the "
                f"task on a model that does — {_TOOL_CALLING_EXAMPLES}."
            )
        if tool_ran:
            message = (
                f"The model returned a '{tool_name}' tool call as its answer "
                "instead of an answer, so the run has no result to report and "
                "the call as written was not carried out. "
            ) + remedy
        else:
            message = (
                f"The model wrote a '{tool_name}' tool call into its answer "
                f"instead of calling the tool, so {tool_name} never ran and "
                f"nothing the answer describes was carried out. "
            ) + remedy
        preview = " ".join((answer or "").split())[:300]
        return {
            "type": "WrittenToolCall",
            "category": "written_tool_call",
            "provider": self._model_provider(self.model),
            "model": model_id,
            "tool": tool_name,
            "tool_calling_strategy": strategy,
            "answer_preview": preview,
            "message": message,
            "retryable": False,
        }

    def _written_tool_call_response(
        self,
        tool_name: str,
        answer: str,
        *,
        iterations: int,
        tool_calls: int,
        tokens_used: int,
        tool_ran: bool = False,
        debug_trace: Any = None,
    ) -> AgentResponse:
        """Report a turn whose answer only describes the tool call it should have made."""
        detail = self._written_tool_call_detail(tool_name, answer, tool_ran=tool_ran)
        logger.warning("Tool call was written as text, not made: %s", detail["message"])
        meta: dict[str, Any] = {
            "reason": "written_tool_call",
            "error": detail,
            "tool_calling_strategy": detail["tool_calling_strategy"],
        }
        if debug_trace is not None:
            debug_trace.total_tokens = tokens_used
            debug_trace.final_answer = None
            debug_trace.success = False
            meta["debug_trace"] = debug_trace
        return AgentResponse(
            output=detail["message"],
            success=False,
            mode=AgentMode.SINGLE,
            iterations=iterations,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            metadata=meta,
        )

    def _iteration_cap_detail(self, cap: int, progress: str | None) -> dict[str, Any]:
        """Return the typed outcome for a run that stopped at its iteration cap.

        The loop ran out of iterations before the model wrote a final answer, so
        the run has no answer to report. What the scratchpad holds at that point
        is tool output and reasoning: returning it as the result presents a
        retrieved passage as if the model had written it. The outcome therefore
        states what happened and what to do, and the recovered text travels
        beside it as ``metadata["partial_output"]``.
        """
        model_id = (
            getattr(self.model, "model_name", None) or self.model_name or "the model"
        )
        step = "iteration" if cap == 1 else "iterations"
        message = (
            f"Stopped after {cap} {step} without a final answer: '{model_id}' "
            "was still taking tool steps when the limit was reached."
        )
        if progress:
            message += (
                " What it had reached by then is reported as partial progress "
                "— tool output and reasoning, not an answer."
            )
        message += (
            f" Raise max_iterations above {cap} to give the run more steps, or "
            "run the task on a model that needs fewer."
        )
        return {
            "type": "MaxIterationsReached",
            "category": "max_iterations",
            "provider": self._model_provider(self.model),
            "model": model_id,
            "max_iterations": cap,
            "message": message,
            "retryable": False,
        }

    def _is_context_retrieval_tool(self, action: str) -> bool:
        """True when ``action`` is a knowledge-base/search tool whose output is
        retrieved context rather than a computed answer.

        Used to flag a fallback that returns such a tool's raw observation as
        partial, so a passage dump is not presented as a synthesized answer, and
        to pick the continuation instruction in :meth:`_continuation_instruction`.
        """
        tool = self.tools.get(action)
        category = getattr(getattr(tool, "metadata", None), "category", None)
        if category is ToolCategory.INFORMATION_RETRIEVAL:
            return True
        return action in {"retrieval", "web_search", "search", "knowledge_base"}

    def _context_answer_instruction(
        self, previous_actions: list[tuple[str, str]]
    ) -> str:
        """Return the answer-shaping line when the latest observation is
        retrieved context, or ``""`` for every other tool.

        A tool prompt ends with the last tool's observation, so whatever follows
        it is the final thing the model reads before answering. After a
        retrieval/search tool that observation is a block of source passages, and
        a generic close leaves the strongest recent signal a wall of text that
        reads like a finished answer: the smallest models return it verbatim,
        dropping the citation markers and the question's scope along the way.
        This line states what to do with the passages instead. Returning ``""``
        for every other tool keeps those prompts byte-for-byte unchanged.
        """
        if previous_actions and self._is_context_retrieval_tool(previous_actions[-1][0]):
            return CONTEXT_ANSWER_INSTRUCTION
        return ""

    def _continuation_instruction(
        self, previous_actions: list[tuple[str, str]]
    ) -> str:
        """Return the line that closes the native/hybrid prompt after a tool ran."""
        return self._context_answer_instruction(previous_actions) or CONTINUE_INSTRUCTION

    def _extract_partial_answer(self, scratchpad: str) -> str | None:
        """
        Extract the best partial answer from the scratchpad when max iterations is reached.

        Looks for patterns like "I now know the answer", recent observations with
        answer-like content, or the last substantive thought.

        Args:
            scratchpad: The accumulated scratchpad text.

        Returns:
            A partial answer string, or None if nothing useful found.
        """
        if not scratchpad:
            return None

        # Remove injected loop-bookkeeping markers before extraction so they are
        # never captured into an observation/thought and leaked as the answer.
        for pat in _SCAFFOLD_LITERAL_RES:
            scratchpad = pat.sub("", scratchpad)

        # Pattern 1: "I now know" type thoughts
        know_match = re.search(
            r"Thought:\s*I (?:now )?know[^.]*\.\s*(.+?)(?=\nThought:|\nAction:|\Z)",
            scratchpad, re.IGNORECASE | re.DOTALL
        )
        if know_match:
            return know_match.group(1).strip()

        # Pattern 2: Observations with clear result values
        observations = re.findall(r"Observation:\s*(.+?)(?=\nThought:|\nAction:|\Z)", scratchpad, re.DOTALL)
        if observations:
            # If multiple observations, combine non-error ones for multi-tool tasks.
            # Strip tool-echo prefixes ("[tool(args)] → result") so only the
            # results are joined, not the scaffolding.
            valid_obs = [
                self._humanize_observation(_TOOL_ECHO_RE.sub("", o.strip()))
                for o in observations
                if o.strip() and not o.strip().lower().startswith("error")
            ]
            # Drop any segment that is pure ReAct scaffolding (a stray
            # "Thought:/Action:" that slipped past the boundary regex).
            valid_obs = [
                o for o in valid_obs
                if o and not re.match(r"^(thought|action|observation|final answer)\b", o.strip(), re.IGNORECASE)
            ]
            # Deduplicate while preserving order — a model that loops on the same
            # retrieval/search tool produces the same passage repeatedly, and
            # joining the duplicates yields a messy answer.
            seen_obs: set[str] = set()
            deduped_obs: list[str] = []
            for o in valid_obs:
                if o and o not in seen_obs:
                    seen_obs.add(o)
                    deduped_obs.append(o)
            valid_obs = deduped_obs
            if len(valid_obs) > 1:
                return sanitize_final_answer(" | ".join(valid_obs))
            elif valid_obs:
                return sanitize_final_answer(valid_obs[-1])

        # Pattern 2b: Look for day names or numeric results in any observation
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for obs in reversed(observations) if observations else []:
            obs_lower = obs.strip().lower()
            for day in day_names:
                if day in obs_lower:
                    return obs.strip()

        # Pattern 3: Last substantive thought
        thoughts = re.findall(r"Thought:\s*(.+?)(?=\nAction:|\nObservation:|\Z)", scratchpad, re.DOTALL)
        if thoughts:
            last_thought = thoughts[-1].strip()
            if len(last_thought) > 20:
                return last_thought

        return None

    def _should_return_direct_calculator_result(
        self,
        task: str,
        action: str,
        action_input: str,
    ) -> bool:
        """Return calculator output directly for simple arithmetic questions."""
        if action != "calculator" or not action_input:
            return False

        task_lower = task.lower()
        if any(
            marker in task_lower
            for marker in (
                "explain",
                "show your work",
                "step by step",
                "steps",
                "why",
                "reason",
            )
        ):
            return False

        try:
            parsed_input = json.loads(action_input)
        except (json.JSONDecodeError, TypeError):
            parsed_input = {"expression": action_input}
        if not isinstance(parsed_input, dict):
            return False

        operation = str(parsed_input.get("operation", "calculate")).lower()
        if operation != "calculate":
            return False

        expression = str(parsed_input.get("expression", "")).strip()
        if not expression:
            return False

        task_numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", task)
        expr_numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", expression)
        if not task_numbers or not all(num in expr_numbers for num in task_numbers):
            return False

        arithmetic_markers = (
            "+",
            "-",
            "*",
            "/",
            "^",
            "%",
            " plus ",
            " minus ",
            " times ",
            " multiplied ",
            " divide ",
            " divided ",
            " square root ",
            " sqrt",
            " squared",
            " cube",
            " cubed",
            " to the power",
            " power of ",
            " modulo ",
            " factorial",
        )
        return any(marker in f" {task_lower} " for marker in arithmetic_markers)

    def _run_with_sub_agents(self,
                            task: str,
                            routing_decision: RoutingDecision,
                            context: dict[str, Any],
                            **kwargs) -> AgentResponse:
        """
        Execute task using sub-agents based on routing decision.

        Args:
            task: Task description
            routing_decision: Router's decision
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        if self._current_depth >= self.config.max_sub_agent_depth:
            logger.warning(f"Sub-agent depth limit reached ({self.config.max_sub_agent_depth})")
            return self._run_single_agent(task, context, **kwargs)

        self._current_depth += 1

        try:
            # Track decomposition
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_DECOMPOSITION,
                agent_id=self.name,
                message=f"Decomposed into {routing_decision.num_sub_agents} subtasks using {routing_decision.strategy.value}",
                data={
                    "strategy": routing_decision.strategy.value,
                    "num_subtasks": routing_decision.num_sub_agents,
                    "specializations": routing_decision.specializations
                }
            ))

            # Execute based on strategy
            strategy = routing_decision.strategy
            subtasks = routing_decision.decomposition

            if strategy == RoutingStrategy.PARALLEL_SUB_AGENTS:
                # Execute in parallel (use helper to handle existing event loops)
                results = self._run_coroutine_sync(
                    self.sub_agent_manager.execute_parallel(subtasks)
                )
            elif strategy == RoutingStrategy.SEQUENTIAL_SUB_AGENTS:
                # Execute sequentially
                results = self.sub_agent_manager.execute_sequential(subtasks)
            elif strategy == RoutingStrategy.HYBRID:
                # Execute with hybrid approach
                results = self.sub_agent_manager.execute_hybrid(subtasks)
            else:
                # Default to sequential
                results = self.sub_agent_manager.execute_sequential(subtasks)

            # Synthesize results
            synthesis = self.sub_agent_manager.synthesize_results(
                results,
                task,
                strategy
            )

            # Calculate totals
            total_tokens = synthesis["metrics"]["total_tokens_used"]
            total_tool_calls = synthesis["metrics"]["total_tool_calls"]

            return AgentResponse(
                output=sanitize_final_answer(synthesis["final_output"]) or synthesis["final_output"],
                success=synthesis["successful"] > 0,
                mode=AgentMode.SUB_AGENTS,
                iterations=len(subtasks),
                tool_calls=total_tool_calls,
                tokens_used=total_tokens,
                routing_decision=routing_decision,
                metadata={
                    "synthesis": synthesis,
                    "failed_subtasks": synthesis["failed"]
                }
            )
        finally:
            self._current_depth -= 1

    def _parse_react_response(self, text: str) -> dict[str, Any]:
        """
        Parse ReAct formatted response with robust error handling.

        Args:
            text: Response text

        Returns:
            Dictionary with parsed components

        Notes:
            This parser handles various formats and edge cases:
            - Case-insensitive matching
            - Multiple thought/action patterns
            - Malformed responses
            - Missing fields
        """
        parsed = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None
        }

        if not text or not isinstance(text, str):
            logger.warning(f"Invalid response text for parsing: {type(text)}")
            return parsed

        try:
            # Check for final answer first (highest priority)
            # NOTE: "Answer:" must be at the start of a line to avoid greedy
            # mid-text matches (e.g. "The answer is 42" should NOT match here).
            final_patterns = [
                r"Final Answer:\s*(.+)",
                r"^Answer:\s*(.+)",
                r"^The answer is:\s*(.+)"
            ]

            for pattern in final_patterns:
                try:
                    final_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    if final_match:
                        answer = final_match.group(1).strip()
                        # Stop at next section marker, observation, or human turn
                        answer = re.split(r'\n(?:Question|Thought|Action|Observation|Human):', answer, maxsplit=1)[0].strip()
                        # Strip trailing unrelated content (e.g. Phi-4 generating
                        # follow-up questions after the answer like "...is 42.What year...")
                        # Match sentence boundary (.!?) followed by a new sentence
                        # that itself contains a question mark — likely a hallucinated follow-up.
                        trailing = re.search(r'([.!?])[\s]*[A-Z][^.!?]*\?', answer)
                        if trailing:
                            answer = answer[:trailing.start() + 1].strip()
                        parsed["final_answer"] = answer
                        logger.debug(f"Extracted final answer: {answer[:100]}...")
                        return parsed
                except Exception as e:
                    logger.warning(f"Error matching final answer pattern '{pattern}': {e}")
                    continue

            # Extract thought
            thought_patterns = [
                r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer|Question):|$)",
                r"Thought:\s*(.+?)(?:\n\n|\n[A-Z]|$)"
            ]

            for pattern in thought_patterns:
                try:
                    thought_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        parsed["thought"] = thought
                        logger.debug(f"Extracted thought: {thought[:100]}...")
                        break
                except Exception as e:
                    logger.warning(f"Error matching thought pattern '{pattern}': {e}")
                    continue

            # Extract action
            action_patterns = [
                r"Action:\s*([^\n]+)",
                r"Tool:\s*([^\n]+)",
                r"Use tool:\s*([^\n]+)"
            ]

            for pattern in action_patterns:
                try:
                    action_match = re.search(pattern, text, re.IGNORECASE)
                    if action_match:
                        action = action_match.group(1).strip()
                        # Clean up common artifacts
                        action = action.replace('"', '').replace("'", "")

                        # Check if action is actually "Final Answer" - treat it as final answer, not tool
                        if action.lower() in ["final answer", "finalanswer", "answer"]:
                            logger.debug(f"Action '{action}' detected as Final Answer indicator")
                            # When model writes "Action: Final Answer", the answer may be:
                            # 1. On the same line: "Action: Final Answer: The answer is 42"
                            # 2. On the Action Input line: "Action Input: The answer is 42"
                            # But NOT if Action Input is JSON (model repeating tool input)

                            # Try same-line first — [: \t]+ excludes newlines
                            same_line = re.search(
                                r"Action:\s*Final\s*Answer[: \t]+([^\n]+)", text, re.IGNORECASE,
                            )
                            if same_line:
                                answer_text = same_line.group(1).strip()
                                if answer_text:
                                    parsed["final_answer"] = answer_text
                                    logger.debug("Extracted final answer from Action line")
                                    return parsed

                            # Try Action Input line (only if it's natural language, not JSON)
                            ai_match = re.search(
                                r"Action\s*Input:\s*(.+?)(?:\n|$)",
                                text, re.IGNORECASE,
                            )
                            if ai_match:
                                answer_text = ai_match.group(1).strip()
                                if answer_text and not answer_text.startswith(("{", "[")):
                                    parsed["final_answer"] = answer_text
                                    logger.debug("Extracted final answer from Action Input line")
                                    return parsed

                            # If we get here, the model wrote "Action: Final Answer"
                            # but didn't provide a proper answer text. Don't extract
                            # anything — let the loop continue for another iteration.
                            logger.debug("Action: Final Answer detected but no answer text found")
                            break

                        # Handle function-call format: tool_name(args) or tool_name("args")
                        # Extract just the tool name and put args into action_input
                        func_call_match = re.match(r'^(\w+)\s*\((.+)\)$', action, re.DOTALL)
                        if func_call_match:
                            tool_name = func_call_match.group(1).strip()
                            embedded_args = func_call_match.group(2).strip()
                            # Remove surrounding quotes if present
                            embedded_args = embedded_args.strip('"\'')
                            parsed["action"] = tool_name
                            # Only set action_input if not already set
                            if "action_input" not in parsed or not parsed["action_input"]:
                                parsed["action_input"] = embedded_args
                            logger.debug(f"Extracted function-call style: action={tool_name}, input={embedded_args[:100]}...")
                        else:
                            parsed["action"] = action
                            logger.debug(f"Extracted action: {action}")
                        break
                except Exception as e:
                    logger.warning(f"Error matching action pattern '{pattern}': {e}")
                    continue

            # Extract action input (only if not already set from function-call style)
            # Skip if we already have embedded args from tool_name(args) format
            if "action_input" not in parsed or not parsed.get("action_input"):
                input_patterns = [
                    r"Action Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question|Final Answer):|$)",
                    r"Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
                    r"Parameters?:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)"
                ]

                for pattern in input_patterns:
                    try:
                        input_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                        if input_match:
                            action_input = input_match.group(1).strip()
                            # Remove trailing observation text if present
                            action_input = re.split(r'\nObservation:', action_input, maxsplit=1)[0].strip()
                            parsed["action_input"] = action_input
                            logger.debug(f"Extracted action input: {action_input[:100]}...")
                            break
                    except Exception as e:
                        logger.warning(f"Error matching action input pattern '{pattern}': {e}")
                        continue

        except Exception as e:
            logger.error(f"Critical error in parse_react_response: {e}", exc_info=True)
            # Return partial parse results even if there was an error

        return parsed

    @staticmethod
    def _parse_native_tool_calls(native_tool_calls: list[dict[str, Any]]) -> ToolCallResult:
        """Convert an adapter's reported tool_calls into a ToolCallResult.

        Adapters report::

            [{"id": "...", "type": "function",
              "function": {"name": "...", "arguments": "json-string"}}]

        Flat keys and an already-parsed ``arguments`` are still accepted, so a
        list that reached here from somewhere other than an adapter — a hosted
        model's own output passed through verbatim, say — is read the same way.
        """
        result = ToolCallResult(raw_text="")
        if not native_tool_calls:
            return result
        tc = native_tool_calls[0]
        fn = tc.get("function", tc)
        tool_name = fn.get("name")
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {"__raw_input__": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        if tool_name:
            result.tool_name = tool_name
            result.arguments = arguments
            result.is_tool_call = True
        return result

    @staticmethod
    def _tool_call_result_to_dict(result: ToolCallResult) -> dict[str, Any]:
        """Convert a ToolCallResult to the legacy dict format used by the ReAct loop.

        This bridges the new strategy-based parsing with the existing loop
        logic that expects ``{'thought', 'action', 'action_input', 'final_answer'}``.
        """
        parsed: dict[str, Any] = {
            "thought": result.thought,
            "action": None,
            "action_input": None,
            "final_answer": result.final_answer,
        }
        if result.is_tool_call and result.tool_name:
            parsed["action"] = result.tool_name
            # Convert arguments dict back to the string form the loop expects
            if result.arguments:
                raw = result.arguments.get("__raw_input__")
                if raw:
                    parsed["action_input"] = raw
                else:
                    parsed["action_input"] = json.dumps(result.arguments)
            else:
                parsed["action_input"] = "{}"
        return parsed

    def _has_native_tools(self) -> bool:
        """Return True if any OpenAI native tool is in the tools list and the model supports Responses API."""
        try:
            from ..models.openai_adapter import OpenAIAdapter
            from ..tools.builtin.openai_native import OpenAINativeTool
        except ImportError:
            return False
        has_native = any(isinstance(t, OpenAINativeTool) for t in self.tools.values())
        is_openai = isinstance(self.model, OpenAIAdapter)
        return has_native and is_openai

    def _run_with_native_tools(self, task: str, context: dict[str, Any], **kwargs) -> AgentResponse:
        """Execute a task using OpenAI native tools via the Responses API.

        Separates native tools (web_search, code_interpreter, file_search) from
        local effGen tools.  Native tools are passed directly to the Responses API
        spec; local tools are serialised as function-call specs alongside them.
        """
        from ..models.openai_adapter import OpenAIAdapter
        from ..tools.builtin.openai_native import OpenAINativeTool

        native_specs: list[dict] = []
        function_specs: list[dict] = []

        for tool in self.tools.values():
            if isinstance(tool, OpenAINativeTool):
                native_specs.append(tool.to_openai_tool_spec())
            else:
                # Regular effGen tool — include as function call spec
                schema = tool.metadata.to_json_schema()
                function_specs.append({
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema["description"],
                        "parameters": schema["parameters"],
                    },
                })

        adapter: OpenAIAdapter = self.model  # already validated in _has_native_tools

        system_prompt = self.config.system_prompt if self.config.stable_system_prompt else None

        # The native-tool call honours the same per-run settings every other
        # path does; a reasoning model in particular needs the budget the
        # caller asked for, or it can spend the default on reasoning alone.
        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get(
                "max_tokens",
                self.config.max_tokens or default_max_output_tokens(self.model, base=2048),
            ),
            top_p=kwargs.get("top_p", self.config.top_p),
            seed=kwargs.get("seed", self.config.seed),
        )

        try:
            result = adapter.generate_with_native_tools(
                prompt=task,
                native_tool_specs=native_specs,
                function_tool_specs=function_specs if function_specs else None,
                system_prompt=system_prompt,
                config=gen_config,
            )
        except Exception as e:
            logger.debug("Native tool generation failed", exc_info=True)
            detail = self._build_error_detail(e, self.model)
            return AgentResponse(
                output=f"Generation failed: {detail['message']}",
                success=False,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=0,
                tokens_used=0,
                metadata={"reason": "generation_failed", "error": detail},
            )

        # Handle any function tool calls that came back (local effGen tools)
        native_results = result.metadata.get("native_tool_results", [])
        # "function_call" is what the Responses API names a call to a local
        # tool; "function" is the name every other adapter reports. Built-in
        # server-side tools (web search, file search) use their own types and
        # are not dispatched here.
        local_calls = [
            r for r in native_results
            if r.get("type") in ("function_call", "function")
        ]

        tool_calls_made = len(native_results)

        # Execute local tool calls and stitch their results into a follow-up
        executed_tools: set[str] = set()
        if local_calls and function_specs:
            observations: list[str] = []
            for call in local_calls:
                fn = call.get("function", call)
                fn_name = call.get("name") or fn.get("name", "")
                fn_args_raw = call.get("arguments", fn.get("arguments", "{}"))
                try:
                    fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                if fn_name in self.tools:
                    obs = self._execute_tool(fn_name, json.dumps(fn_args))
                    tool_calls_made += 1
                    executed_tools.add(fn_name)
                    observations.append(f"[{fn_name}({fn_args})] → {obs}")

            if observations and not result.text:
                # Re-prompt with the observations to get a final answer
                obs_text = "\n".join(observations)
                followup = f"Tool results:\n{obs_text}\n\nBased on these results, answer the user's question: {task}"
                try:
                    followup_result = adapter.generate(followup)
                    return AgentResponse(
                        output=sanitize_final_answer(followup_result.text) or followup_result.text,
                        success=True,
                        mode=AgentMode.SINGLE,
                        iterations=2,
                        tool_calls=tool_calls_made,
                        tokens_used=result.tokens_used + followup_result.tokens_used,
                        metadata=result.metadata,
                    )
                except Exception:
                    logger.debug("Native tool follow-up assembly failed; using prior result", exc_info=True)

        if not result.text and (result.metadata or {}).get("reasoning_only"):
            return self._reasoning_only_native_response(result, tool_calls_made)

        answer = sanitize_final_answer(result.text) or ""
        # Sanitizing a tagged call can leave its arguments behind as a bare JSON
        # fragment, so the text as the model wrote it is scanned as well.
        written = None
        if result.text:
            written = find_written_tool_call(
                answer, self.tools
            ) or find_written_tool_call(result.text, self.tools)
        if written and (
            written not in executed_tools or written_call_only(result.text, self.tools)
        ):
            return self._written_tool_call_response(
                written,
                result.text,
                iterations=1,
                tool_calls=tool_calls_made,
                tokens_used=result.tokens_used,
                tool_ran=written in executed_tools,
            )
        meta = dict(result.metadata or {})
        meta.setdefault("tool_calling_strategy", "openai_native")
        return AgentResponse(
            output=answer or "(no output from native tools call)",
            success=bool(result.text),
            mode=AgentMode.SINGLE,
            iterations=1,
            tool_calls=tool_calls_made,
            tokens_used=result.tokens_used,
            metadata=meta,
        )

    def _reasoning_only_native_response(
        self, result: Any, tool_calls_made: int,
    ) -> AgentResponse:
        """Failure response for a native-tool turn that produced only reasoning.

        The adapter already worked out why the answer is empty and named the cap
        and the reasoning budget; report that instead of stating only that there
        was no output.
        """
        meta = result.metadata or {}
        detail = {
            "type": "ReasoningOnlyResponse",
            "category": "reasoning_only",
            "provider": self._model_provider(self.model),
            "model": getattr(self.model, "model_name", None) or self.model_name or "unknown",
            "message": meta.get("empty_response_reason") or (
                "The model produced only internal reasoning and no answer."
            ),
            "reasoning_tokens": meta.get("reasoning_tokens", 0),
            "retryable": False,
        }
        return AgentResponse(
            output=f"Generation failed: {detail['message']}",
            success=False,
            mode=AgentMode.SINGLE,
            iterations=1,
            tool_calls=tool_calls_made,
            tokens_used=result.tokens_used,
            metadata={"reason": "generation_failed", "error": detail},
        )

    def _has_gemini_native_tools(self) -> bool:
        """Return True if any Gemini native tool is in the tools list and model is Gemini."""
        try:
            from ..models.gemini_adapter import GeminiAdapter
            from ..tools.builtin.gemini_native import GeminiNativeTool
        except ImportError:
            return False
        has_native = any(isinstance(t, GeminiNativeTool) for t in self.tools.values())
        is_gemini = isinstance(self.model, GeminiAdapter)
        return has_native and is_gemini

    def _run_with_gemini_native_tools(self, task: str, context: dict[str, Any], **kwargs) -> AgentResponse:
        """Execute a task using Gemini server-side native tools.

        Passes ``GeminiNativeTool`` objects (google_search, url_context, code_execution)
        directly to the Gemini adapter alongside any local effGen tools serialised
        as function declarations.  The adapter routes them correctly to the SDK.
        """
        from ..tools.builtin.gemini_native import GeminiNativeTool

        # Separate native vs. local tools
        native_tools: list = []
        function_tool_specs: list[dict] = []
        for tool in self.tools.values():
            if isinstance(tool, GeminiNativeTool):
                native_tools.append(tool)
            else:
                schema = tool.metadata.to_json_schema()
                function_tool_specs.append({
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema["description"],
                        "parameters": schema["parameters"],
                    },
                })

        # Combine: native tool objects first, then regular function specs
        all_tools = native_tools + function_tool_specs
        mixed = bool(native_tools and function_tool_specs)

        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get(
                "max_tokens",
                self.config.max_tokens or default_max_output_tokens(self.model, base=2048),
            ),
            top_p=kwargs.get("top_p", self.config.top_p),
            seed=kwargs.get("seed", self.config.seed),
        )

        system_note = ""
        if self.config.system_prompt:
            system_note = f"{self.config.system_prompt}\n\n"

        prompt = f"{system_note}{task}"

        try:
            result = self.model.generate(prompt, config=gen_config, tools=all_tools)
        except Exception as exc:
            logger.debug("Gemini native tool generation failed", exc_info=True)
            detail = self._build_error_detail(exc, self.model)
            # Most Gemini models reject combining a built-in tool (google_search,
            # url_context, code_execution) with custom function tools in one
            # request ("cannot be combined" / "context circulation is not
            # enabled"). Translate that raw 400 into an actionable remediation
            # instead of a bare provider error.
            low = detail.get("message", "").lower()
            if mixed and ("cannot be combined" in low or "circulation" in low
                          or "function calling" in low):
                native_names = ", ".join(t.name for t in native_tools)
                detail["message"] = (
                    f"Gemini model '{self.model_name}' cannot use its built-in tool(s) "
                    f"({native_names}) together with custom function tools in a single "
                    "request. Build the agent with EITHER the native tool(s) OR your "
                    "custom tools — not both — or use a model that supports tool-call "
                    "context circulation."
                )
                detail["category"] = "invalid_request"
                detail["retryable"] = False
            return AgentResponse(
                output=f"Generation failed: {detail['message']}",
                success=False,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=0,
                tokens_used=0,
                metadata={"reason": "generation_failed", "error": detail},
            )

        tool_calls_made = len(result.metadata.get("tool_calls") or [])

        # Execute any local effGen function calls that came back
        local_tc = result.metadata.get("tool_calls") or []
        observations: list[str] = []
        executed_tools: set[str] = set()
        for tc in local_tc:
            # Adapters report a call as {"function": {"name", "arguments"}};
            # an adapter that also carries top-level keys is read the same way.
            fn = tc.get("function", tc)
            fn_name = tc.get("name") or fn.get("name", "")
            fn_args = tc.get("arguments", fn.get("arguments", {}))
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except (json.JSONDecodeError, TypeError):
                    fn_args = {"__raw_input__": fn_args}
            if fn_name in self.tools and not isinstance(self.tools[fn_name], GeminiNativeTool):
                obs = self._execute_tool(fn_name, json.dumps(fn_args) if isinstance(fn_args, dict) else fn_args)
                executed_tools.add(fn_name)
                observations.append(f"[{fn_name}({fn_args})] → {obs}")

        # The adapter also encodes each call as a <tool_call> block in the
        # generated text, so a turn that did nothing but call tools is not
        # empty — it is empty once those blocks are stripped. Testing the raw
        # text would skip the follow-up and report the answer as missing while
        # the tool results sit unused.
        answer = sanitize_final_answer(result.text) or ""

        if observations and not answer.strip():
            obs_text = "\n".join(observations)
            followup = f"Tool results:\n{obs_text}\n\nBased on these results, answer: {task}"
            try:
                followup_result = self.model.generate(followup, config=gen_config)
                followup_answer = (
                    sanitize_final_answer(followup_result.text) or followup_result.text
                )
                # A follow-up that came back empty answers nothing; fall through
                # rather than report an empty output as a successful turn.
                if followup_answer and followup_answer.strip():
                    followup_meta = dict(result.metadata or {})
                    followup_meta.setdefault("tool_calling_strategy", "gemini_native")
                    return AgentResponse(
                        output=followup_answer,
                        success=True,
                        mode=AgentMode.SINGLE,
                        iterations=2,
                        tool_calls=tool_calls_made,
                        tokens_used=result.tokens_used + followup_result.tokens_used,
                        metadata=followup_meta,
                    )
            except Exception:
                logger.debug("Native tool follow-up assembly failed; using prior result", exc_info=True)

        if not result.text and (result.metadata or {}).get("reasoning_only"):
            return self._reasoning_only_native_response(result, tool_calls_made)

        # Sanitizing a tagged call can leave its arguments behind as a bare JSON
        # fragment, so the text as the model wrote it is scanned as well.
        written = None
        if result.text:
            written = find_written_tool_call(
                answer, self.tools
            ) or find_written_tool_call(result.text, self.tools)
        if written and (
            written not in executed_tools or written_call_only(result.text, self.tools)
        ):
            return self._written_tool_call_response(
                written,
                result.text,
                iterations=1,
                tool_calls=tool_calls_made,
                tokens_used=result.tokens_used,
                tool_ran=written in executed_tools,
            )
        meta = dict(result.metadata or {})
        meta.setdefault("tool_calling_strategy", "gemini_native")
        return AgentResponse(
            output=answer or "(no output from Gemini native tools call)",
            # A turn whose whole text was a tool-call block leaves nothing to
            # report once it is stripped; that is a failed turn, not a success
            # carrying a placeholder.
            success=bool(answer),
            mode=AgentMode.SINGLE,
            iterations=1,
            tool_calls=tool_calls_made,
            tokens_used=result.tokens_used,
            metadata=meta,
        )
