"""
Secret redaction for effGen structured logging.

All log attributes are passed through a ``Redactor`` before being serialised
to JSON.  The redactor applies a set of regex patterns that match common API
keys, bearer tokens, and webhook URLs, replacing the matched text with a
labelled placeholder like ``<REDACTED:openai_key>``.

Built-in patterns
-----------------
- ``openai_key``    — ``sk-[a-zA-Z0-9]{20,}``
- ``anthropic_key`` — ``sk-ant-[a-zA-Z0-9]{20,}``
- ``cerebras_key``  — ``csk-[a-zA-Z0-9]{20,}``
- ``google_key``    — ``AIza[0-9A-Za-z_-]{35}``
- ``hf_key``        — ``hf_[a-zA-Z0-9]{20,}``
- ``groq_key``      — ``gsk_[a-zA-Z0-9]{20,}``
- ``bearer_token``  — ``Bearer [^\\s]+``
- ``slack_webhook`` — Slack/Discord webhook URLs (host kept, path replaced)
- ``discord_webhook`` — Discord webhook path

User-extensible via ``Redactor.add_pattern(name, regex)``.

Usage
-----
    from effgen.observability.redact import get_redactor, Redactor

    r = get_redactor()
    r.scrub("my key is sk-abc123xyz9876543210")
    # → "my key is <REDACTED:openai_key>"

    # Redact a dict (recursive, values only — keys are never secrets)
    clean = r.scrub_dict({"Authorization": "Bearer sk-abc123xyz9876543210"})
    # → {"Authorization": "<REDACTED:bearer_token>"}
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

# Ordered so that more-specific patterns are applied before less-specific ones
# (e.g. anthropic_key before openai_key because both start with "sk-")
_BUILTIN_PATTERNS: list[tuple[str, str]] = [
    # More-specific first
    ("anthropic_key", r"sk-ant-[a-zA-Z0-9_\-]{20,}"),
    ("cerebras_key",  r"csk-[a-zA-Z0-9_\-]{20,}"),
    ("google_key",    r"AIza[0-9A-Za-z_\-]{35}"),
    ("hf_key",        r"hf_[a-zA-Z0-9]{20,}"),
    ("groq_key",      r"gsk_[a-zA-Z0-9_\-]{20,}"),
    # openai key AFTER more-specific sk-ant- / csk- patterns
    ("openai_key",    r"sk-[a-zA-Z0-9_\-]{20,}"),
    # Bearer token — matches "Bearer <anything non-whitespace>"
    ("bearer_token",  r"Bearer [^\s]{6,}"),
    # Slack incoming webhook: keep host, mask path
    ("slack_webhook", r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+"),
    # Discord webhook: keep host, mask path
    ("discord_webhook", r"https://discord(?:app)?\.com/api/webhooks/[0-9/A-Za-z_\-]+"),
]


class Redactor:
    """
    Apply regex-based secret redaction to strings and nested data structures.

    Thread-safe: compiled pattern list is built once at construction; later
    ``add_pattern`` calls append to the list atomically enough for logging
    use (GIL-protected list append in CPython).
    """

    def __init__(self) -> None:
        # List of (name, compiled_regex) pairs — order matters (first match wins
        # when the same span is covered by multiple patterns, though that is rare)
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for name, pattern in _BUILTIN_PATTERNS:
            self._patterns.append((name, re.compile(pattern)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_pattern(self, name: str, regex: str) -> None:
        """
        Register a custom redaction pattern.

        Args:
            name:  Label used in the placeholder, e.g. ``"my_secret"``.
            regex: Python regex string.  The entire match is replaced by
                   ``<REDACTED:{name}>``.

        Example::

            r = get_redactor()
            r.add_pattern("custom_token", r"tok-[A-Za-z0-9]{32}")
        """
        self._patterns.append((name, re.compile(regex)))

    def scrub(self, text: str) -> str:
        """
        Redact all secrets in *text* and return the sanitised string.

        Args:
            text: Arbitrary string that may contain secrets.

        Returns:
            Copy of *text* with every matched secret replaced by a labelled
            placeholder (e.g. ``<REDACTED:openai_key>``).
        """
        for name, pattern in self._patterns:
            text = pattern.sub(f"<REDACTED:{name}>", text)
        return text

    def scrub_value(self, value: Any) -> Any:
        """
        Redact a single log-field value.

        - ``str`` → scrub and return new string
        - ``dict`` → recursively scrub values (not keys)
        - ``list`` / ``tuple`` → recursively scrub elements
        - Anything else → return unchanged (cannot contain raw secret tokens)
        """
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, dict):
            return self.scrub_dict(value)
        if isinstance(value, list | tuple):
            scrubbed = [self.scrub_value(v) for v in value]
            return type(value)(scrubbed)
        return value

    def scrub_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Return a new dict with all string values redacted.

        Only values are scrubbed; keys are never treated as secrets.

        Args:
            data: Dictionary to sanitise (may be nested).

        Returns:
            New dictionary with all string values scrubbed.
        """
        return {k: self.scrub_value(v) for k, v in data.items()}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def pattern_names(self) -> list[str]:
        """Return the names of all registered redaction patterns."""
        return [name for name, _ in self._patterns]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_redactor: Redactor | None = None


def get_redactor() -> Redactor:
    """
    Return the process-wide default :class:`Redactor` instance.

    The instance is created lazily on first call.  The same instance is
    returned on subsequent calls — add custom patterns to it once at
    application startup.

    Returns:
        The default :class:`Redactor`.
    """
    global _default_redactor
    if _default_redactor is None:
        _default_redactor = Redactor()
    return _default_redactor


__all__ = [
    "Redactor",
    "get_redactor",
]
