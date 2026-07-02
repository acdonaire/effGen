"""
effGen Guardrails Module

Provides input/output validation, content filtering, prompt injection
detection, PII protection, and tool safety guardrails.

All guardrails work offline with no external APIs or ML models.
"""

from .base import Guardrail, GuardrailChain, GuardrailPosition, GuardrailResult
from .content import LengthGuardrail, PIIGuardrail, TopicGuardrail, ToxicityGuardrail
from .injection import PromptInjectionGuardrail
from .presets import (
    MINIMAL,
    NONE,
    PHI,
    STANDARD,
    STRICT,
    get_guardrail_preset,
    minimal_guardrails,
    no_guardrails,
    phi_guardrails,
    standard_guardrails,
    strict_guardrails,
)
from .tool_safety import ToolInputGuardrail, ToolOutputGuardrail, ToolPermissionGuardrail

__all__ = [
    # Base
    "Guardrail",
    "GuardrailChain",
    "GuardrailPosition",
    "GuardrailResult",
    # Content
    "ToxicityGuardrail",
    "PIIGuardrail",
    "LengthGuardrail",
    "TopicGuardrail",
    # Injection
    "PromptInjectionGuardrail",
    # Tool Safety
    "ToolInputGuardrail",
    "ToolOutputGuardrail",
    "ToolPermissionGuardrail",
    # Presets
    "STRICT",
    "STANDARD",
    "PHI",
    "MINIMAL",
    "NONE",
    "get_guardrail_preset",
    "strict_guardrails",
    "standard_guardrails",
    "phi_guardrails",
    "minimal_guardrails",
    "no_guardrails",
]
