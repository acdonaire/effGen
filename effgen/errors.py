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
