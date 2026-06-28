"""
Content guardrails for effGen.

Provides regex/keyword-based content validation guardrails that work
entirely offline with no external APIs or ML models.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Guardrail, GuardrailPosition, GuardrailResult


class ToxicityGuardrail(Guardrail):
    """Detect toxic, hateful, or abusive content using keyword matching.

    Uses a curated list of slurs, hate speech indicators, and threat
    patterns. No external API or ML model required.
    """

    # Patterns indicating threats or incitement to violence
    _THREAT_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\b(?:i(?:'ll| will| am going to)\s+)?kill\s+(?:you|them|him|her|everyone)\b", re.I),
        re.compile(r"\b(?:i(?:'ll| will| am going to)\s+)?murder\s+(?:you|them|him|her)\b", re.I),
        re.compile(r"\bdie\s+in\s+a\s+fire\b", re.I),
        re.compile(r"\b(?:death|bomb)\s+threat\b", re.I),
        re.compile(r"\bshoot\s+up\b", re.I),
    ]

    # Slurs and hate speech keywords (deliberately small, high-precision set)
    _HATE_KEYWORDS: set[str] = {
        "kike", "spic", "wetback", "chink", "gook",
        "faggot", "tranny", "dyke",
        "nigger", "nigga", "coon",
        "raghead", "towelhead", "sandnigger",
    }

    def __init__(
        self,
        extra_blocked_words: list[str] | None = None,
        positions: list[GuardrailPosition] | None = None,
        enabled: bool = True,
    ):
        super().__init__(
            name="ToxicityGuardrail",
            positions=positions or [GuardrailPosition.INPUT, GuardrailPosition.OUTPUT],
            enabled=enabled,
        )
        self._blocked_words = self._HATE_KEYWORDS.copy()
        if extra_blocked_words:
            self._blocked_words.update(w.lower() for w in extra_blocked_words)

    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        lower = content.lower()

        # Check threat patterns
        for pattern in self._THREAT_PATTERNS:
            if pattern.search(content):
                return GuardrailResult(
                    passed=False,
                    reason="Toxicity guardrail: content contains threatening language",
                )

        # Check hate keywords (word-boundary matching)
        for word in self._blocked_words:
            if re.search(rf"\b{re.escape(word)}\b", lower):
                return GuardrailResult(
                    passed=False,
                    reason="Toxicity guardrail: content contains hateful language",
                )

        return GuardrailResult(passed=True)


class PIIGuardrail(Guardrail):
    """Detect personally identifiable information using regex patterns.

    Covers: US SSN, email addresses, phone numbers (US and international),
    credit card numbers (with Luhn validation), IP addresses, and — when
    ``detect_secrets`` is enabled (default) — common API keys / cloud
    credentials (AWS access keys, ``sk-``/``gsk_`` style API keys, Google API
    keys, GitHub/Slack tokens, bearer tokens, and PEM private-key headers), so a
    leaked credential is treated as sensitive and not just classic PII.
    """

    # Common credential / API-key shapes. High-precision prefixes keep these from
    # firing on ordinary prose. Each pattern matches the whole token so redaction
    # removes the full secret.
    _SECRET_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[0-9A-Z]{16}\b"),  # AWS access key id
        re.compile(r"\b(?:sk|gsk|rk|pk)-[A-Za-z0-9]{20,}\b"),                   # OpenAI/Groq-style API keys
        re.compile(r"\b(?:gsk|sk)_[A-Za-z0-9]{20,}\b"),                         # underscore-style keys
        re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),                              # Google API key
        re.compile(r"\bghp_[A-Za-z0-9]{36,}\b"),                                # GitHub personal token
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),                        # GitHub fine-grained token
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),                        # Slack token
        re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b"),                       # bearer token
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),                   # PEM private key
    ]

    # US Social Security Number: XXX-XX-XXXX
    _SSN_PATTERN = re.compile(
        r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
    )

    # Email address
    _EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    )

    # US phone: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx, +1xxxxxxxxxx
    _PHONE_US_PATTERN = re.compile(
        r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"
    )

    # International phone: +XX XXXXXXXXXX (at least 7 digits after country code)
    _PHONE_INTL_PATTERN = re.compile(
        r"\+\d{1,3}[\s.-]?\d{4,14}\b"
    )

    # Credit card: 13-19 digits, possibly separated by spaces or dashes
    _CC_PATTERN = re.compile(
        r"\b(?:\d[ -]*?){13,19}\b"
    )

    # IPv4 address. The leading ``(?<![\w.])`` guard keeps the dotted-quad from
    # starting inside a longer token. The trailing ``(?!\w)(?!\.\d)`` guard rejects
    # a 5th octet (``1.2.3.4.5``) and a trailing alphanumeric, but — unlike the old
    # ``(?![\w.])`` — still matches an address that ends a sentence (``10.2.3.4.``),
    # so an internal IP at end-of-sentence is no longer silently let through.
    _IPV4_PATTERN = re.compile(
        r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\w)(?!\.\d)"
    )

    # Words that, when they immediately precede a dotted-quad, mark it as a version
    # or build number rather than an IP address (avoids over-redacting release notes).
    _VERSION_CUE_BEFORE = re.compile(
        r"(?i)\b(?:v|ver|version|versions|release|released|rev|revision|build|"
        r"patch|update|firmware|driver|sdk|api|schema|spec|protocol)\b\W*$"
    )
    # Version cues that follow a dotted-quad (e.g. "1.2.3.4 release").
    _VERSION_CUE_AFTER = re.compile(
        r"(?i)^\W*(?:release|released|rc\d|beta|alpha|stable|lts|build|patch)\b"
    )

    @classmethod
    def _is_version_string(cls, text: str, start: int, end: int) -> bool:
        """Return True if the dotted-quad at ``text[start:end]`` reads as a version."""
        before = text[:start]
        after = text[end:]
        return bool(
            cls._VERSION_CUE_BEFORE.search(before)
            or cls._VERSION_CUE_AFTER.search(after)
        )

    def __init__(
        self,
        detect_ssn: bool = True,
        detect_email: bool = True,
        detect_phone: bool = True,
        detect_credit_card: bool = True,
        detect_ip: bool = True,
        detect_secrets: bool = True,
        action: str = "block",  # "block" or "redact"
        positions: list[GuardrailPosition] | None = None,
        enabled: bool = True,
    ):
        super().__init__(
            name="PIIGuardrail",
            positions=positions or [
                GuardrailPosition.INPUT,
                GuardrailPosition.OUTPUT,
                GuardrailPosition.TOOL_OUTPUT,
            ],
            enabled=enabled,
        )
        self.detect_ssn = detect_ssn
        self.detect_email = detect_email
        self.detect_phone = detect_phone
        self.detect_credit_card = detect_credit_card
        self.detect_ip = detect_ip
        self.detect_secrets = detect_secrets
        self.action = action

    @staticmethod
    def _luhn_check(number_str: str) -> bool:
        """Validate a credit card number using the Luhn algorithm."""
        digits = [int(d) for d in number_str if d.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False
        checksum = 0
        reverse = digits[::-1]
        for i, d in enumerate(reverse):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        return checksum % 10 == 0

    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        detections: list[str] = []
        redacted = content

        if self.detect_ssn and self._SSN_PATTERN.search(content):
            detections.append("SSN")
            redacted = self._SSN_PATTERN.sub("[SSN REDACTED]", redacted)

        if self.detect_email and self._EMAIL_PATTERN.search(content):
            detections.append("email")
            redacted = self._EMAIL_PATTERN.sub("[EMAIL REDACTED]", redacted)

        if self.detect_phone:
            if self._PHONE_US_PATTERN.search(content):
                detections.append("phone")
                redacted = self._PHONE_US_PATTERN.sub("[PHONE REDACTED]", redacted)
            elif self._PHONE_INTL_PATTERN.search(content):
                detections.append("phone")
                redacted = self._PHONE_INTL_PATTERN.sub("[PHONE REDACTED]", redacted)

        if self.detect_credit_card:
            for match in self._CC_PATTERN.finditer(content):
                digits_only = re.sub(r"[ -]", "", match.group())
                if self._luhn_check(digits_only):
                    detections.append("credit_card")
                    redacted = redacted.replace(match.group(), "[CC REDACTED]")
                    break  # one detection is enough

        if self.detect_ip:
            found_ip = False

            def _ip_repl(match: re.Match[str]) -> str:
                nonlocal found_ip
                if self._is_version_string(match.string, match.start(), match.end()):
                    return match.group()  # leave version/build numbers untouched
                found_ip = True
                return "[IP REDACTED]"

            redacted = self._IPV4_PATTERN.sub(_ip_repl, redacted)
            if found_ip:
                detections.append("IP_address")

        if self.detect_secrets:
            found_secret = False
            for pattern in self._SECRET_PATTERNS:
                if pattern.search(redacted):
                    found_secret = True
                    redacted = pattern.sub("[SECRET REDACTED]", redacted)
            if found_secret:
                detections.append("secret")

        if not detections:
            return GuardrailResult(passed=True)

        reason = f"PII guardrail: detected {', '.join(detections)}"

        if self.action == "redact":
            return GuardrailResult(
                passed=True,
                reason=reason,
                modified_content=redacted,
                metadata={"pii_types": detections},
            )

        return GuardrailResult(
            passed=False,
            reason=reason,
            metadata={"pii_types": detections},
        )


class LengthGuardrail(Guardrail):
    """Enforce maximum and minimum length on content."""

    def __init__(
        self,
        max_length: int = 100_000,
        min_length: int = 0,
        positions: list[GuardrailPosition] | None = None,
        enabled: bool = True,
    ):
        super().__init__(
            name="LengthGuardrail",
            positions=positions or [GuardrailPosition.INPUT, GuardrailPosition.OUTPUT],
            enabled=enabled,
        )
        self.max_length = max_length
        self.min_length = min_length

    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        length = len(content)

        if length > self.max_length:
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Length guardrail: content length {length} exceeds "
                    f"maximum {self.max_length}"
                ),
                metadata={"length": length, "max_length": self.max_length},
            )

        if length < self.min_length:
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Length guardrail: content length {length} below "
                    f"minimum {self.min_length}"
                ),
                metadata={"length": length, "min_length": self.min_length},
            )

        return GuardrailResult(passed=True)


class TopicGuardrail(Guardrail):
    """Filter content based on allowed/blocked topic keyword lists.

    Either ``allowed_topics`` or ``blocked_topics`` can be set (not both).
    Topics are matched by case-insensitive keyword search.
    """

    def __init__(
        self,
        allowed_topics: list[str] | None = None,
        blocked_topics: list[str] | None = None,
        positions: list[GuardrailPosition] | None = None,
        enabled: bool = True,
    ):
        super().__init__(
            name="TopicGuardrail",
            positions=positions or [GuardrailPosition.INPUT],
            enabled=enabled,
        )
        self.allowed_topics = [t.lower() for t in (allowed_topics or [])]
        self.blocked_topics = [t.lower() for t in (blocked_topics or [])]

    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        lower = content.lower()

        # Check blocked topics
        for topic in self.blocked_topics:
            if topic in lower:
                return GuardrailResult(
                    passed=False,
                    reason=f"Topic guardrail: blocked topic '{topic}' detected",
                    metadata={"blocked_topic": topic},
                )

        # Check allowed topics (if set, content must match at least one)
        if self.allowed_topics:
            if not any(topic in lower for topic in self.allowed_topics):
                return GuardrailResult(
                    passed=False,
                    reason="Topic guardrail: content does not match any allowed topic",
                    metadata={"allowed_topics": self.allowed_topics},
                )

        return GuardrailResult(passed=True)
