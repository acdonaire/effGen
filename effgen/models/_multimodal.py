"""Shared helpers for adapter-side multimodal capability checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from effgen.core.messages import Message
from effgen.errors import CapabilityNotSupportedError
from effgen.models.capabilities import Capability


def has_image_input(prompt: Any) -> bool:
    """Return True when *prompt* contains at least one ImagePart."""
    if isinstance(prompt, Message):
        return prompt.has_image
    if isinstance(prompt, list):
        return any(isinstance(item, Message) and item.has_image for item in prompt)
    return False


def require_vision_support(
    prompt: Any,
    *,
    provider: str,
    model_name: str,
    supports_vision: bool | Callable[[str], bool],
    hint: str = "",
) -> None:
    """Raise CapabilityNotSupportedError if image input targets a non-vision model."""
    if not has_image_input(prompt):
        return

    supported = supports_vision(model_name) if callable(supports_vision) else supports_vision
    if supported:
        return

    detail = hint or f"Select a vision-capable {provider} model for image inputs."
    raise CapabilityNotSupportedError(
        Capability.vision,
        provider=provider,
        hint=f"Model '{model_name}' does not support vision. {detail}",
    )
