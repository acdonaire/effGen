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


def has_audio_input(prompt: Any) -> bool:
    """Return True when *prompt* contains at least one AudioPart."""
    if isinstance(prompt, Message):
        return prompt.has_audio
    if isinstance(prompt, list):
        return any(isinstance(item, Message) and item.has_audio for item in prompt)
    return False


def has_video_input(prompt: Any) -> bool:
    """Return True when *prompt* contains at least one VideoPart."""
    if isinstance(prompt, Message):
        return prompt.has_video
    if isinstance(prompt, list):
        return any(isinstance(item, Message) and item.has_video for item in prompt)
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


def require_audio_support(
    prompt: Any,
    *,
    provider: str,
    model_name: str,
    supports_audio: bool | Callable[[str], bool],
    hint: str = "",
) -> None:
    """Raise CapabilityNotSupportedError if audio input targets a non-audio model."""
    if not has_audio_input(prompt):
        return

    supported = supports_audio(model_name) if callable(supports_audio) else supports_audio
    if supported:
        return

    detail = hint or f"Select an audio-capable {provider} model for audio inputs."
    raise CapabilityNotSupportedError(
        Capability.audio_input,
        provider=provider,
        hint=f"Provider '{provider}' / model '{model_name}' does not support audio input. {detail}",
    )


def require_video_support(
    prompt: Any,
    *,
    provider: str,
    model_name: str,
    supports_video: bool | Callable[[str], bool],
    hint: str = "",
) -> None:
    """Raise CapabilityNotSupportedError if video input targets a non-video model.

    Note: most providers handle VideoPart via frame-sampling fallback (vision
    capability required).  Only call this when the provider truly cannot handle
    video in any form (e.g. no vision support at all).
    """
    if not has_video_input(prompt):
        return

    supported = supports_video(model_name) if callable(supports_video) else supports_video
    if supported:
        return

    detail = hint or f"Select a video-capable {provider} model or enable vision for frame-sampling fallback."
    raise CapabilityNotSupportedError(
        Capability.video_input,
        provider=provider,
        hint=f"Provider '{provider}' / model '{model_name}' does not support video input. {detail}",
    )
