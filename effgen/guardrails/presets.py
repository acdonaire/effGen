"""
Guardrail presets for effGen.

Provides pre-configured guardrail chains for common use cases.
"""

from __future__ import annotations

from .base import GuardrailChain
from .content import LengthGuardrail, PIIGuardrail, ToxicityGuardrail
from .injection import PromptInjectionGuardrail
from .tool_safety import ToolInputGuardrail, ToolOutputGuardrail, ToolPermissionGuardrail


def strict_guardrails(
    max_length: int = 50_000,
    tool_deny: list[str] | None = None,
) -> GuardrailChain:
    """All guardrails enabled, high sensitivity.

    Includes: length, injection (high), toxicity, PII (block),
    topic (none by default), tool input validation, tool output
    sanitization with PII stripping, and tool permissions.
    """
    chain = GuardrailChain([
        LengthGuardrail(max_length=max_length),
        PromptInjectionGuardrail(sensitivity="high"),
        ToxicityGuardrail(),
        PIIGuardrail(action="block"),
        ToolInputGuardrail(),
        ToolOutputGuardrail(strip_pii=True, max_output_length=max_length),
    ])
    if tool_deny:
        chain.add(ToolPermissionGuardrail(deny=tool_deny))
    return chain


def standard_guardrails(
    max_length: int = 100_000,
) -> GuardrailChain:
    """PII + injection + tool safety, medium sensitivity.

    A balanced preset suitable for most production deployments.
    """
    return GuardrailChain([
        LengthGuardrail(max_length=max_length),
        PromptInjectionGuardrail(sensitivity="medium"),
        PIIGuardrail(action="block"),
        ToolInputGuardrail(),
        ToolOutputGuardrail(max_output_length=max_length),
    ])


def minimal_guardrails(
    max_length: int = 200_000,
) -> GuardrailChain:
    """Basic length + injection only.

    Lightweight preset for low-risk applications.
    """
    return GuardrailChain([
        LengthGuardrail(max_length=max_length),
        PromptInjectionGuardrail(sensitivity="low"),
    ])


def no_guardrails() -> GuardrailChain:
    """No guardrails — empty chain. For development/testing only."""
    return GuardrailChain([])


# Named preset constants for convenience
STRICT = "strict"
STANDARD = "standard"
MINIMAL = "minimal"
NONE = "none"

_PRESET_FACTORIES = {
    STRICT: strict_guardrails,
    STANDARD: standard_guardrails,
    MINIMAL: minimal_guardrails,
    NONE: no_guardrails,
}

# Friendly aliases for the canonical preset names. ``"default"`` is the name most
# users reach for first, so it maps to the balanced "standard" preset.
_PRESET_ALIASES = {
    "default": STANDARD,
    "balanced": STANDARD,
    "off": NONE,
    "disabled": NONE,
}


def get_guardrail_preset(name: str, **kwargs) -> GuardrailChain:
    """Get a guardrail chain by preset name.

    Args:
        name: One of "strict", "standard", "minimal", "none". The aliases
            "default"/"balanced" map to "standard" and "off"/"disabled" map
            to "none".
        **kwargs: Passed to the preset factory function.

    Returns:
        Configured GuardrailChain.
    """
    key = name.lower()
    key = _PRESET_ALIASES.get(key, key)
    factory = _PRESET_FACTORIES.get(key)
    if factory is None:
        available = ", ".join(_PRESET_FACTORIES.keys())
        raise ValueError(f"Unknown guardrail preset: {name!r}. Available: {available}")
    return factory(**kwargs)
