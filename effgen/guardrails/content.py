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

    Covers, out of the box:

    - **US SSN** — dashed (``123-45-6789``), space-grouped (``123 45 6789``),
      and, when preceded by an ``SSN``/``Social Security`` label, undelimited
      (``123456789``).
    - **Email**, **phone** (US and international), **credit card** (Luhn-checked),
      and **IPv4** addresses.
    - When ``detect_secrets`` is enabled (default): common API keys / cloud
      credentials (AWS access keys, ``sk-``/``gsk_`` style API keys, Google API
      keys, GitHub/Slack tokens, bearer tokens, and PEM private-key headers), so a
      leaked credential is treated as sensitive and not just classic PII.
    - When ``detect_labeled`` is enabled (default): **label-anchored fields**
      common in semi-structured records — ``Name:``/``Patient``, ``DOB:``,
      ``MRN:``/``Medical Record #``, ``Address:``, and member/beneficiary/policy
      IDs (``Member ID``, ``Insurance ID``, ``Policy Number``). The label is kept
      and only the value is replaced (e.g. ``MRN: [MRN REDACTED]``).

    **Coverage limits.** Label-anchored detection relies on a preceding label; a
    free-text personal name with no label (``Seen by Maria Gonzalez``) is not
    detected by regex alone. For records that use site-specific identifiers, pass
    ``custom_patterns`` (regexes) and/or ``custom_terms`` (literal strings) to
    extend coverage. For a privacy-critical path, set ``strict=True``: in redact
    mode the check then reports ``passed=False`` whenever any PII is detected, so
    the pipeline fails closed instead of forwarding text on the assumption that
    regex redaction was complete.

    Args:
        detect_ssn / detect_email / detect_phone / detect_credit_card /
            detect_ip / detect_secrets: Toggle each built-in detector.
        detect_labeled: Toggle the label-anchored clinical/record fields
            (name, DOB, MRN, address, member/policy ID). Default True.
        custom_patterns: Extra regexes to redact. Each item is a regex ``str``,
            a compiled ``re.Pattern``, or a ``(pattern, label)`` pair; ``label``
            names the detection and its placeholder (``[LABEL REDACTED]``).
        custom_terms: Literal strings to redact (case-insensitive, whole-word).
            Each item is a ``str`` or a ``(term, label)`` pair.
        action: ``"block"`` (default) or ``"redact"``.
        strict: When True and ``action="redact"``, report ``passed=False`` on any
            detection so a privacy-critical path fails closed rather than
            forwarding redacted-but-not-guaranteed-clean text. The redacted text
            is still provided in ``modified_content`` for inspection.
    """

    # Common credential / API-key shapes. High-precision prefixes keep these from
    # firing on ordinary prose. Each pattern matches the whole token so redaction
    # removes the full secret.
    _SECRET_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[0-9A-Z]{16}\b"),  # AWS access key id
        # AWS secret access key: exactly 40 base64-alphabet chars, no context
        # keyword required (mirrors the format-only patterns below). The
        # mixed-case + digit lookaheads keep it from firing on a long run of
        # plain lowercase prose.
        re.compile(
            r"(?<![A-Za-z0-9/+=])(?=[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+=]))"
            r"(?=[A-Za-z0-9/+]*[A-Z])(?=[A-Za-z0-9/+]*[a-z])(?=[A-Za-z0-9/+]*[0-9])"
            r"[A-Za-z0-9/+]{40}"
        ),
        re.compile(r"\b(?:sk|gsk|rk|pk)-[A-Za-z0-9]{20,}\b"),                   # OpenAI/Groq-style API keys
        re.compile(r"\b(?:gsk|sk)_[A-Za-z0-9]{20,}\b"),                         # underscore-style keys
        re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),                              # Google API key
        re.compile(r"\bghp_[A-Za-z0-9]{36,}\b"),                                # GitHub personal token
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),                        # GitHub fine-grained token
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),                        # Slack token
        re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b"),                       # bearer token
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),                   # PEM private key
    ]

    # US Social Security Number, dashed or space-grouped: XXX-XX-XXXX / XXX XX XXXX.
    # The 3-2-4 grouping is distinctive enough to redact without a label cue.
    _SSN_PATTERN = re.compile(
        r"\b(?!000|666|9\d{2})\d{3}[-\s](?!00)\d{2}[-\s](?!0000)\d{4}\b"
    )
    # SSN written with no delimiter (XXXXXXXXX) is ambiguous with other 9-digit
    # ids, so it is only redacted when an ``SSN``/``Social Security`` label
    # precedes it. The label is preserved; only the digits are replaced.
    # Separator between a field label and its value. Tolerates a trailing period
    # from an abbreviation (``D.O.B.``), a ``#`` before the colon (``Record #:``),
    # a dash (``DOB - ...``), and surrounding whitespace. Values never begin with
    # these characters, so it does not eat into a value.
    _LSEP = r"\s*[:#.\-]{0,3}\s*"
    # A labeled personal name: a capitalized first token, then up to ``n`` more
    # tokens that may be an initial (``Q.``) or a capitalized word including
    # apostrophe/hyphen surnames (``O'Brien``, ``Smith-Jones``). The first token
    # keeps the ``[A-Z][a-z]`` shape so all-caps headers (``PATIENT NOTE``) and
    # trailing lowercase prose are not captured.
    # The full-word alternative precedes the single-initial one so a surname is
    # captured whole ("Gonzalez"), not truncated to its first letter. An initial
    # must carry a period or be followed by a non-letter, so a following acronym
    # label ("MRN") is not mistaken for a middle initial.
    _NAME_VALUE = (
        r"[A-Z][a-z][A-Za-z'.\-]*"
        r"(?:[ \t]+(?:[A-Z][a-z'][A-Za-z'.\-]*|[A-Z](?:\.|(?![A-Za-z])))){{0,{n}}}"
    )

    _SSN_CUED_PATTERN = re.compile(
        r"(?P<label>(?i:\bSSN\b|\bSocial\s+Security(?:\s+(?:No|Number|#))?\b)" + _LSEP + r")"
        r"(?P<value>(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4})\b"
    )

    # Label-anchored fields common in semi-structured records (clinical notes,
    # intake forms). Each keeps its label and replaces only the value. Ordered so
    # that the more specific fields resolve before the line-spanning address.
    _LABELED_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
        (
            "name", "[NAME REDACTED]",
            re.compile(
                r"(?P<label>(?i:\b(?:patient(?:\s+name)?|name)\b)[ \t]*[:#.]{0,2}[ \t]+)"
                r"(?P<value>" + _NAME_VALUE.format(n=3) + r")"
            ),
        ),
        (
            "name", "[NAME REDACTED]",
            re.compile(
                r"(?P<label>(?i:\b(?:Dr|Mr|Mrs|Ms)\b)\.?[ \t]+)"
                r"(?P<value>" + _NAME_VALUE.format(n=2) + r")"
            ),
        ),
        (
            "DOB", "[DOB REDACTED]",
            re.compile(
                r"(?P<label>(?i:\b(?:DOB|D\.O\.B\.?|date\s+of\s+birth|birth\s*date)\b)" + _LSEP + r")"
                r"(?P<value>\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})"
            ),
        ),
        (
            "MRN", "[MRN REDACTED]",
            re.compile(
                r"(?P<label>(?i:\b(?:MRN|medical\s+record(?:\s+(?:no|number|#))?|"
                r"record\s+(?:no|number|#))\b)" + _LSEP + r")"
                r"(?P<value>[A-Za-z0-9][A-Za-z0-9\-]{3,})"
            ),
        ),
        (
            "member_id", "[ID REDACTED]",
            re.compile(
                r"(?P<label>(?i:\b(?:insurance\s+(?:member\s+)?id|member\s+id|"
                r"beneficiary\s+id|subscriber\s+id|policy\s+(?:no|number|#|id)|"
                r"group\s+(?:no|number|#)|health\s+plan\s+id)\b)" + _LSEP + r")"
                r"(?P<value>[A-Za-z0-9][A-Za-z0-9\-]{3,})"
            ),
        ),
        # Street address up to a ZIP code (preferred: precise end so a following
        # sentence or instruction on the same line is not swallowed).
        (
            "address", "[ADDRESS REDACTED]",
            re.compile(
                r"(?P<label>(?i:\b(?:mailing\s+address|home\s+address|street\s+address|"
                r"address|addr)\b)" + _LSEP + r")"
                r"(?P<value>\d+[^\n]*?\b\d{5}(?:-\d{4})?\b)"
            ),
        ),
        # Fallback for an address with no ZIP: stop at a sentence terminator or
        # line break rather than running to end of line.
        (
            "address", "[ADDRESS REDACTED]",
            re.compile(
                r"(?P<label>(?i:\b(?:mailing\s+address|home\s+address|street\s+address|"
                r"address|addr)\b)" + _LSEP + r")"
                r"(?P<value>\d+[^\n.;]*)"
            ),
        ),
    ]

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
        detect_labeled: bool = True,
        custom_patterns: list[str | re.Pattern[str] | tuple[str, str]] | None = None,
        custom_terms: list[str | tuple[str, str]] | None = None,
        action: str = "block",  # "block" or "redact"
        strict: bool = False,
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
        self.detect_labeled = detect_labeled
        self.action = action
        self.strict = strict
        self._custom_patterns = self._normalize_custom_patterns(custom_patterns)
        self._custom_terms = self._normalize_custom_terms(custom_terms)

    @staticmethod
    def _normalize_custom_patterns(
        patterns: list[str | re.Pattern[str] | tuple[str, str]] | None,
    ) -> list[tuple[str, str, re.Pattern[str]]]:
        """Return a list of ``(type, placeholder, compiled)`` from user input.

        Each item may be a regex string, a compiled pattern, or a
        ``(pattern, label)`` pair. ``label`` names the detection and the
        placeholder; without one, the detection is typed ``"custom"``.
        """
        out: list[tuple[str, str, re.Pattern[str]]] = []
        for item in patterns or []:
            label = "custom"
            if isinstance(item, tuple):
                pat, label = item[0], item[1]
            else:
                pat = item
            compiled = pat if isinstance(pat, re.Pattern) else re.compile(pat)
            placeholder = f"[{label.upper()} REDACTED]"
            out.append((label, placeholder, compiled))
        return out

    @staticmethod
    def _normalize_custom_terms(
        terms: list[str | tuple[str, str]] | None,
    ) -> list[tuple[str, str, re.Pattern[str]]]:
        """Return ``(type, placeholder, compiled)`` for literal terms.

        Terms match case-insensitively on whole-word boundaries.
        """
        out: list[tuple[str, str, re.Pattern[str]]] = []
        for item in terms or []:
            label = "custom"
            if isinstance(item, tuple):
                term, label = item[0], item[1]
            else:
                term = item
            if not term:
                continue
            compiled = re.compile(rf"\b{re.escape(term)}\b", re.I)
            placeholder = f"[{label.upper()} REDACTED]"
            out.append((label, placeholder, compiled))
        return out

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
        counts: dict[str, int] = {}
        redacted = content

        def _record(kind: str, n: int = 1) -> None:
            if n > 0:
                counts[kind] = counts.get(kind, 0) + n

        if self.detect_ssn:
            redacted, n = self._SSN_PATTERN.subn("[SSN REDACTED]", redacted)
            _record("SSN", n)
            # Undelimited SSN only when an SSN/Social-Security label precedes it.
            redacted, n = self._SSN_CUED_PATTERN.subn(
                lambda m: m.group("label") + "[SSN REDACTED]", redacted
            )
            _record("SSN", n)

        if self.detect_email:
            redacted, n = self._EMAIL_PATTERN.subn("[EMAIL REDACTED]", redacted)
            _record("email", n)

        if self.detect_phone:
            redacted, n = self._PHONE_US_PATTERN.subn("[PHONE REDACTED]", redacted)
            _record("phone", n)
            redacted, n = self._PHONE_INTL_PATTERN.subn("[PHONE REDACTED]", redacted)
            _record("phone", n)

        if self.detect_credit_card:
            for match in self._CC_PATTERN.finditer(content):
                digits_only = re.sub(r"[ -]", "", match.group())
                if self._luhn_check(digits_only):
                    _record("credit_card")
                    redacted = redacted.replace(match.group(), "[CC REDACTED]")
                    break  # one detection is enough

        if self.detect_ip:
            found_ip = 0

            def _ip_repl(match: re.Match[str]) -> str:
                nonlocal found_ip
                if self._is_version_string(match.string, match.start(), match.end()):
                    return match.group()  # leave version/build numbers untouched
                found_ip += 1
                return "[IP REDACTED]"

            redacted = self._IPV4_PATTERN.sub(_ip_repl, redacted)
            _record("IP_address", found_ip)

        if self.detect_secrets:
            for pattern in self._SECRET_PATTERNS:
                redacted, n = pattern.subn("[SECRET REDACTED]", redacted)
                _record("secret", n)

        if self.detect_labeled:
            for kind, placeholder, pattern in self._LABELED_PATTERNS:
                redacted, n = pattern.subn(
                    lambda m, ph=placeholder: m.group("label") + ph, redacted
                )
                _record(kind, n)

        for kind, placeholder, pattern in self._custom_patterns:
            redacted, n = pattern.subn(placeholder, redacted)
            _record(kind, n)

        for kind, placeholder, pattern in self._custom_terms:
            redacted, n = pattern.subn(placeholder, redacted)
            _record(kind, n)

        if not counts:
            return GuardrailResult(passed=True)

        detections = list(counts.keys())
        reason = f"PII guardrail: detected {', '.join(detections)}"
        metadata = {"pii_types": detections, "pii_counts": counts}

        if self.action == "redact":
            # In strict mode a privacy-critical path fails closed: any detection
            # reports passed=False so redacted-but-not-guaranteed-clean text is
            # not forwarded on the assumption regex redaction was complete. The
            # redacted text is still returned for inspection.
            if self.strict:
                return GuardrailResult(
                    passed=False,
                    reason=reason + " (strict mode: blocking)",
                    modified_content=redacted,
                    metadata=metadata,
                )
            return GuardrailResult(
                passed=True,
                reason=reason,
                modified_content=redacted,
                metadata=metadata,
            )

        return GuardrailResult(
            passed=False,
            reason=reason,
            metadata=metadata,
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
