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


class CorruptDocumentError(EffGenError):
    """Raised when a document file is malformed or cannot be parsed.

    Attributes:
        doc_type: The document type that failed (e.g. 'pdf', 'docx', 'xlsx').
        detail: Additional detail from the underlying parser.
    """

    def __init__(self, doc_type: str, detail: str = "") -> None:
        self.doc_type = doc_type
        self.detail = detail
        msg = f"Cannot parse {doc_type.upper()} document - file may be corrupt or invalid."
        if detail:
            msg += f"\nDetail: {detail}"
        super().__init__(msg)


class MissingCredentialsError(EffGenError):
    """Raised when required credentials (env vars) are absent for a tool.

    Attributes:
        tool_name: The tool that needs credentials.
        missing_vars: List of env var names that are missing.
    """

    def __init__(self, tool_name: str, missing_vars: list[str], hint: str = "") -> None:
        self.tool_name = tool_name
        self.missing_vars = missing_vars
        msg = (
            f"{tool_name} requires credentials that are not configured: "
            f"{', '.join(missing_vars)}."
        )
        if hint:
            msg += f"\n\n{hint}"
        super().__init__(msg)


class CapabilityNotSupportedError(EffGenError):
    """Raised when an adapter cannot handle a requested capability (e.g. vision, audio).

    Attributes:
        capability: The Capability enum value that is not supported.
        provider: Name of the provider/adapter that raised this.
        hint: Optional guidance on which provider to use instead.
    """

    def __init__(self, capability: object, provider: str = "", hint: str = "") -> None:
        capability_name = getattr(capability, "value", str(capability))
        self.capability = capability_name
        self.provider = provider
        msg = f"Capability '{capability_name}' is not supported"
        if provider:
            msg += f" by provider '{provider}'"
        msg += "."
        if hint:
            msg += f"\n\n{hint}"
        super().__init__(msg)


class InvalidMultimodalContent(EffGenError):
    """Raised when multimodal content (image, audio, video) fails validation.

    Attributes:
        part_type: The part type that failed (e.g. 'image', 'audio', 'video_frames').
        reason: Description of the validation failure.
    """

    def __init__(self, part_type: str, reason: str = "") -> None:
        self.part_type = part_type
        self.reason = reason
        msg = f"Invalid {part_type} content"
        if reason:
            msg += f": {reason}"
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
