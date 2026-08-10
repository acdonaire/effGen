"""The ReAct reasoning loop for :class:`Agent`.

Holds the loop itself — the scratchpad, the instructions and nudges it appends,
the stop it reports when a turn writes a call out instead of making it or when
the iteration cap is reached — and sub-agent delegation. The surrounding
concerns live beside it and are inherited by :class:`AgentReActMixin`, so every
method resolves on :class:`Agent` as before: reading a turn in
:class:`~effgen.core.agent_react_parsing.AgentReActParsingMixin`, the
provider-native run paths in
:class:`~effgen.core.agent_native_tools.AgentNativeToolsMixin`, tool dispatch in
:class:`~effgen.core.agent_tool_execution.AgentToolExecutionMixin` and citation
assembly in :class:`~effgen.core.agent_citations.AgentCitationsMixin`.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

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
from .agent_native_tools import AgentNativeToolsMixin
from .agent_react_parsing import AgentReActParsingMixin
from .agent_tool_execution import AgentToolExecutionMixin
from .agent_tool_loop import NativeToolLoop
from .execution_tracker import EventType, ExecutionEvent
from .router import RoutingDecision, RoutingStrategy

logger = logging.getLogger(__name__)
_slog = get_structured_logger(__name__)
# Canonical structured observability logger — emits redacted JSON lines with OTel context
_obs_log = _get_obs_logger(__name__)

from .agent import AgentMode, AgentResponse  # noqa: E402
from .agent_runtime import (  # noqa: E402
    CONTEXT_ANSWER_INSTRUCTION,
    CONTINUE_INSTRUCTION,
    NUDGE_ALREADY_COMPUTED,
    NUDGE_CONTINUE,
    NUDGE_HAVE_RESULTS,
    NUDGE_NO_TOOLS,
    NUDGE_NOT_USABLE,
    TEMPLATE_TOOL_USE_INSTRUCTION,
    _infer_provider_from_model,
    find_written_tool_call,
    sanitize_final_answer,
    unknown_tool_observation,
)

#: Models that complete a tool loop through the provider's tool-calling API,
#: named in the hint a written-out call block produces. Kept short and stable;
#: ``effgen models list`` marks every model that advertises tool calling.
_TOOL_CALLING_EXAMPLES = (
    "openai:gpt-5-nano, gemini:gemini-3.1-flash-lite or groq:llama-3.3-70b-versatile"
)


class AgentReActMixin(
    AgentReActParsingMixin,
    AgentNativeToolsMixin,
    AgentToolExecutionMixin,
    AgentCitationsMixin,
):
    """The ReAct loop, and the surrounding tool-calling surface it inherits."""

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
        # An explicit ``max_iterations=None`` — what an optional flag forwards when
        # the user did not set it — must fall back to the configured cap rather
        # than reach the loop comparison as None.
        _requested_iterations = kwargs.get("max_iterations")
        max_iterations: int = (
            self.config.max_iterations
            if _requested_iterations is None
            else int(_requested_iterations)
        )

        # Debug trace collector
        debug_trace = None
        if debug:
            from ..debug.inspector import DebugTrace
            debug_trace = DebugTrace(
                task=task, agent_name=self.name, run_id=run_id,
            )

        # Format conversation history
        conversation_history = self._format_conversation_history()

        # ReAct loop. The repeat guards — which calls have been dispatched, which
        # results have already come back, when to stop offering tools and when a
        # written-out call has been seen once too often — live in the loop policy
        # the streaming loop shares, so both reach the same decisions.
        guards = NativeToolLoop(self.tools, nudge_cap=self.config.max_iterations)

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
            if guards.tools_suppressed():
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
                        f"{self._continuation_instruction(guards.previous_actions)}"
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
                # A chat template hands the model the tool definitions and says
                # nothing about using them, so state the expectation once, on the
                # opening turn. Continuation turns already carry their own
                # instruction, and provider-side tool calling is left alone: it
                # decides for itself, and the extra line only pushes it into
                # calls it was right to skip.
                if not scratchpad and self.tools and self._model_tool_call_support() == "template":
                    prompt = f"{prompt}\n\n{TEMPLATE_TOOL_USE_INSTRUCTION}"
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
                    closing_instruction=self._context_answer_instruction(guards.previous_actions),
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
                        guards.record_execution(_tname)
                        batch_observations.append(f"[{_tname}({_targs})] → {_obs}")
                        scratchpad += f"\nAction: {_tname}\nAction Input: {json.dumps(_targs)}\nObservation: {_obs}"
                    else:
                        batch_observations.append(f"[{_tname}] → Tool not found")
                # After batch execution, nudge model to synthesize a final answer.
                scratchpad += f"\n{NUDGE_CONTINUE}"
                guards.note_batch_run()
                parsed = {"thought": "", "action": None, "action_input": None, "final_answer": None}
                cur_observation = "\n".join(batch_observations)
                logger.info(f"[Batch native tool calls] {len(native_tool_calls)} calls executed (batch run #{guards.batch_tool_runs})")
            elif native_tool_calls:
                strategy_result = self._parse_native_tool_calls(native_tool_calls)
                # Convert to legacy dict format for compatibility with rest of loop
                parsed = self._tool_call_result_to_dict(strategy_result)
            else:
                parse_strategy = self._text_parse_strategy(use_native_prompt)
                strategy_result = parse_strategy.parse_response(
                    response["text"], tools=self.tools,
                )
                # Convert to legacy dict format for compatibility with rest of loop
                parsed = self._tool_call_result_to_dict(strategy_result)

            # Debug: Log what was parsed
            logger.info(f"[Iteration {iterations}] Parsed - Action: {parsed.get('action')}, Input: {parsed.get('action_input')}, Final: {parsed.get('final_answer')}")

            # Add to scratchpad. A turn that made a native tool call reports no
            # thought, and the scratchpad is prompt text the model reads back —
            # so an absent thought is an empty line, never the word "None".
            scratchpad += f"\nThought: {parsed.get('thought') or ''}"

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
                    if written and guards.is_unmade_call(written, raw_answer):
                        return self._written_tool_call_response(
                            written,
                            output,
                            iterations=_iterations,
                            tool_calls=_tool_calls,
                            tokens_used=_tokens_used,
                            tool_ran=guards.tool_ran(written),
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
                if written and guards.is_unmade_call(written, final_answer):
                    if guards.note_written_call(written):
                        return self._written_tool_call_response(
                            guards.written_call,
                            final_answer,
                            iterations=iterations,
                            tool_calls=tool_calls,
                            tokens_used=tokens_used,
                            tool_ran=guards.tool_ran(guards.written_call),
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

                # Repeat detection: the same call again, or the same tool
                # enough times with drifting inputs that it reads as a loop.
                check = guards.check_action(action, action_input)
                action_call_count = check.action_call_count
                if check.is_loop:
                    logger.info(
                        f"[Loop detected] Repeated action '{action}' ({check.loop_type}) — "
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
                        and not guards.force_text_answer
                    ):
                        guards.force_text_answer = True
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
                    guards.force_text_answer = True
                    scratchpad += (
                        f"\nAction: {action}"
                        f"\nAction Input: {action_input}"
                        f"\nObservation: {NUDGE_ALREADY_COMPUTED}"
                    )
                    continue

                guards.record_action(check)

                # Check if tool is available (handle no-tool mode without raising)
                if not self.tools or action not in self.tools:
                    # The action names no tool the agent holds. With tools
                    # attached, say which ones are callable — telling a model
                    # that owns a calculator there are "no tools available"
                    # sends it off to do the work itself. With no tools at all,
                    # answering directly is the only option left.
                    observation = (
                        unknown_tool_observation(action, list(self.tools))
                        if self.tools
                        else NUDGE_NO_TOOLS
                    )
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {observation}"
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
                    guards.record_execution(action)
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
                    if guards.result_is_repeat(action, tool_result):
                        # A retrieval/search tool's output is context, not a
                        # synthesized answer, so returning it verbatim hands
                        # back a passage dump. The observation is already in
                        # the scratchpad: stop offering tools and give the
                        # model one turn to write the answer from it. A
                        # compute tool (e.g. calculator) reproducing its
                        # result is a confident answer and is returned as-is.
                        if (
                            self._is_context_retrieval_tool(action)
                            and not guards.force_text_answer
                        ):
                            logger.info(
                                "[Loop efficiency] Retrieval tool '%s' repeated a "
                                "result; asking for a synthesized answer",
                                action,
                            )
                            guards.force_text_answer = True
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
                    guards.record_result(action, tool_result)

                    nudge = guards.post_tool_nudge(
                        iterations, action_call_count, tool_result
                    )
                    if nudge:
                        scratchpad += f"\n{nudge}"

            else:
                # A turn that produced neither an action nor an answer, but did
                # write out a call for a tool this agent holds, is the same
                # failure the answer path reports: the model is writing the call
                # instead of making it. Say so once, and on a second such turn
                # report the cause rather than grinding to the iteration cap
                # and reporting only that the cap was reached.
                written = find_written_tool_call(response["text"], self.tools)
                if written and guards.is_unmade_call(written, response["text"]):
                    if guards.note_written_call(written):
                        return self._written_tool_call_response(
                            guards.written_call,
                            response["text"],
                            iterations=iterations,
                            tool_calls=tool_calls,
                            tokens_used=tokens_used,
                            tool_ran=guards.tool_ran(guards.written_call),
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
        if guards.written_call and not partial_answer:
            return self._written_tool_call_response(
                guards.written_call,
                "",
                iterations=iterations,
                tool_calls=tool_calls,
                tokens_used=tokens_used,
                tool_ran=guards.tool_ran(guards.written_call),
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

    def _written_tool_call_detail(
        self, tool_name: str, answer: str, *, tool_ran: bool = False,
    ) -> dict[str, Any]:
        """Return the typed error for an answer that writes out a tool call.

        The remediation depends on which tool-calling path ran: a model that was
        sent the tool definitions natively and still answered with the call as
        text needs replacing, while a model that advertises native tool calling
        but ran the ReAct text protocol only needs to be asked for the native
        path. It also names how the definitions reached the model — a provider's
        tool-calling API or a local chat template — so the advice matches what
        actually happened. *tool_ran* says whether the named tool was dispatched
        earlier in the run, which decides what the answer failed to do.
        """
        strategy = self._tool_calling_strategy.name
        model_id = (
            getattr(self.model, "model_name", None) or self.model_name or "the model"
        )
        advertises = self._model_advertises_tool_calling()
        if strategy in ("native", "hybrid") and advertises:
            delivery = (
                "rendered into the prompt by its chat template"
                if self._model_tool_call_support() == "template"
                else "sent through the provider's tool-calling API"
            )
            remedy = (
                f"'{model_id}' had the tool definitions {delivery} and answered "
                "with the call as text anyway. Run the task on a model that "
                f"calls tools — {_TOOL_CALLING_EXAMPLES} — or on a larger local "
                "model."
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

        A tool may declare it directly with ``is_context_retrieval = True``,
        which is how a tool whose category says otherwise — a file tool narrowed
        to reading, whose output is source material — opts in. The category and
        name checks below are unchanged, so every other agent classifies exactly
        as before.
        """
        tool = self.tools.get(action)
        if getattr(tool, "is_context_retrieval", False):
            return True
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
