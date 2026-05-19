"""
Top-level error types for the effGen framework.

These are raised at runtime when a required system dependency or backend
is unavailable, so callers can catch them specifically and display helpful
messages rather than cryptic import errors.
"""

from __future__ import annotations


class EffGenError(Exception):
    """Base class for all effGen-specific errors."""


class MissingSystemDependency(EffGenError):
    """Raised when a required system-level binary is not found on PATH.

    Attributes:
        dependency: Name of the missing binary (e.g. "tesseract").
        install_instructions: Per-OS install guidance.
    """

    def __init__(self, dependency: str, install_instructions: str = "") -> None:
        self.dependency = dependency
        self.install_instructions = install_instructions
        msg = f"System dependency '{dependency}' is not installed or not on PATH."
        if install_instructions:
            msg += f"\n\n{install_instructions}"
        super().__init__(msg)


class OCRBackendUnavailable(EffGenError):
    """Raised when no OCR backend (Tesseract or OCR.space) is available.

    Attributes:
        tried_backends: List of backend names that were checked.
    """

    _DEFAULT_INSTALL = (
        "Install Tesseract:\n"
        "  Ubuntu/Debian : sudo apt install tesseract-ocr\n"
        "  macOS         : brew install tesseract\n"
        "  Windows       : choco install tesseract\n"
        "  conda         : conda install -c conda-forge tesseract\n\n"
        "Or set OCR_SPACE_API_KEY to use the OCR.space cloud fallback."
    )

    def __init__(
        self,
        tried_backends: list[str] | None = None,
        extra: str = "",
    ) -> None:
        self.tried_backends = tried_backends or []
        msg = "No OCR backend is available."
        if self.tried_backends:
            msg += f" Tried: {', '.join(self.tried_backends)}."
        msg += f"\n\n{self._DEFAULT_INSTALL}"
        if extra:
            msg += f"\n\n{extra}"
        super().__init__(msg)


class NoVisionProviderAvailable(EffGenError):
    """Raised when ImageCaptionTool cannot find a vision-capable provider.

    Attributes:
        tried_providers: List of provider names that were checked.
    """

    _DEFAULT_HINT = (
        "Configure a vision-capable provider:\n"
        "  OpenAI  : set OPENAI_API_KEY  (gpt-4o supports vision)\n"
        "  Google  : set GOOGLE_API_KEY  (gemini-2.5-flash-lite supports vision)\n"
        "  Replicate: set REPLICATE_API_TOKEN\n\n"
        "Then pass the provider when creating ImageCaptionTool or use "
        "the model router with Capability.vision."
    )

    def __init__(
        self,
        tried_providers: list[str] | None = None,
        extra: str = "",
    ) -> None:
        self.tried_providers = tried_providers or []
        msg = "No vision-capable provider is configured."
        if self.tried_providers:
            msg += f" Tried: {', '.join(self.tried_providers)}."
        msg += f"\n\n{self._DEFAULT_HINT}"
        if extra:
            msg += f"\n\n{extra}"
        super().__init__(msg)


class AudioBackendUnavailable(EffGenError):
    """Raised when no audio transcription backend is available.

    Attributes:
        tried_backends: List of backend names that were checked.
    """

    _DEFAULT_INSTALL = (
        "Install local transcription support:\n"
        "  pip install 'effgen[audio]'\n"
        "  Ubuntu/Debian : sudo apt install ffmpeg\n"
        "  macOS         : brew install ffmpeg\n"
        "  Windows       : choco install ffmpeg\n"
        "  conda         : conda install -c conda-forge ffmpeg\n\n"
        "Or set HF_TOKEN to use the HuggingFace Inference fallback."
    )

    def __init__(
        self,
        tried_backends: list[str] | None = None,
        extra: str = "",
    ) -> None:
        self.tried_backends = tried_backends or []
        msg = "No audio transcription backend is available."
        if self.tried_backends:
            msg += f" Tried: {', '.join(self.tried_backends)}."
        msg += f"\n\n{self._DEFAULT_INSTALL}"
        if extra:
            msg += f"\n\n{extra}"
        super().__init__(msg)
