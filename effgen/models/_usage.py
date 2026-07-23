"""Shared usage-accounting plumbing for provider adapters.

Cloud adapters repeat the same bookkeeping around every call: read token
usage off the SDK response (including cached prompt tokens), record the call
in the process-global :class:`~effgen.models._cost.CostTracker`, and build the
per-call ``GenerationResult.metadata`` block. These helpers keep that plumbing
in one place so every adapter reports the same shapes.

Internal module — no public API surface.
"""

from __future__ import annotations

import logging
from typing import Any

from effgen.models.errors import BudgetExceededError


def extract_openai_usage(usage: Any) -> tuple[int, int, int, int]:
    """Return ``(prompt, completion, total, cached)`` token counts from *usage*.

    *usage* is an OpenAI-style chat-completions usage object. Cached prompt
    tokens live under ``usage.prompt_tokens_details.cached_tokens`` and default
    to 0 when the details block is absent or ``None``.
    """
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    cached_tokens = 0
    if getattr(usage, "prompt_tokens_details", None):
        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
    return prompt_tokens, completion_tokens, total_tokens, cached_tokens


def usage_metadata(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cached_tokens: int,
    cost: float,
    total_cost: float,
) -> dict[str, Any]:
    """Build the canonical per-call token/cost metadata block.

    ``cost_usd`` is this call's cost. ``total_cost`` is a different number,
    not an alias: the cumulative cost across every call made on the adapter
    instance so far (including this one).
    """
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_tokens,
        "cost_usd": cost,
        "total_cost": total_cost,
    }


def tool_calls_from_message(message: Any) -> list[dict[str, Any]]:
    """Convert OpenAI-style ``message.tool_calls`` into a list of plain dicts."""
    tool_calls: list[dict[str, Any]] = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
    return tool_calls


def record_tracker_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    log: logging.Logger,
) -> float | None:
    """Record one call in the process-global :class:`CostTracker`.

    Returns the catalog-priced USD cost (so it matches the model catalog and
    ``effgen cost``), or ``None`` when tracking is unavailable.
    :class:`BudgetExceededError` propagates so budget limits stop the run; any
    other tracker failure is confined to a debug log — accounting must never
    break a successful generation.
    """
    try:
        from effgen.models._cost import CostTracker

        return CostTracker.get().record(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except BudgetExceededError:
        raise
    except Exception:
        log.debug("CostTracker recording failed for %s", provider, exc_info=True)
        return None


__all__ = [
    "extract_openai_usage",
    "usage_metadata",
    "tool_calls_from_message",
    "record_tracker_cost",
]
