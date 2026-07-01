"""
Prompt injection detection guardrail for effGen.

Detects common prompt injection patterns including instruction override,
role-play injection, system prompt extraction, and delimiter injection.
All detection is regex/keyword-based — no external APIs or ML models.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Guardrail, GuardrailPosition, GuardrailResult


class PromptInjectionGuardrail(Guardrail):
    """Detect common prompt injection attempts at configurable sensitivity levels.

    This is a fast, offline, regex-based detector — **best-effort
    defense-in-depth, not a security boundary.** It catches the well-known
    textbook phrasings (instruction-override, identity/role-play hijacks, system
    prompt extraction / prompt-leak, and role-delimiter spoofing) but a
    determined attacker can paraphrase around any regex. Treat a pass as "no
    obvious injection seen", not as "input is safe"; pair it with least-privilege
    tools and output guardrails rather than relying on it as a gate.

    Sensitivity levels:
        - ``low``: Only obvious, high-confidence instruction/identity overrides.
        - ``medium`` (default): Adds system-prompt extraction / prompt-leak,
          role-play jailbreaks, role-delimiter spoofing, and ``System:`` headers.
        - ``high``: Adds the most aggressive markdown/divider patterns.

    Tuned to avoid false positives on normal questions like
    "What is a system prompt?" or "Ignore the noise and focus on the data."
    """

    # ---- HIGH-CONFIDENCE patterns (all levels) ----

    # Direct instruction override: "Ignore previous instructions", etc.
    # Requires an imperative verb plus either an instruction-noun or an explicit
    # "the above/preceding" reference, so plain text like "ignore the noise"
    # does not match.
    _INSTRUCTION_OVERRIDE: list[re.Pattern[str]] = [
        # verb + (optional qualifier words) + instruction-noun
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass|skip|do\s+not\s+follow)"
            r"(?:\s+(?:all|any|the|your|my|these|those|this|previous|prior|above|"
            r"earlier|original|initial|preceding|foregoing|aforementioned)){0,5}"
            r"\s+(?:instructions?|prompts?|rules?|guidelines?|directives?|"
            r"constraints?|context|messages?|commands?)",
            re.I,
        ),
        # verb + "the above / preceding / prior" (the object itself is the
        # instructions being overridden)
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+|everything\s+)?(?:the\s+|my\s+|your\s+)?"
            r"(?:above|preceding|foregoing|aforementioned)\b",
            re.I,
        ),
        # "forget everything you were told / I said / above"
        re.compile(
            r"\b(?:forget|ignore|disregard|erase)\s+(?:everything|anything|all)\s+"
            r"(?:(?:above|before|previously|earlier)\b|"
            r"(?:you(?:'?ve)?\s+(?:were\s+|have\s+been\s+|been\s+)?"
            r"(?:told|instructed|given|taught))|"
            r"(?:i\s+(?:said|told|wrote|gave)))",
            re.I,
        ),
    ]

    # Explicit new identity assignment: "You are now ...", "Act as ...",
    # "Pretend you have no restrictions".
    _IDENTITY_OVERRIDE: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|"
            r"you\s+(?:must|should|will)\s+now\s+(?:act|behave|respond)\s+as)\b",
            re.I,
        ),
        # "pretend (to be | you are/have/can/...)" anywhere — a hijack regardless
        # of position.
        re.compile(
            r"\bpretend\s+(?:that\s+)?(?:to\s+be|you(?:'?re|\s+(?:are|have|can|"
            r"do|don'?t|no\s+longer)))\b",
            re.I,
        ),
    ]

    # ---- MEDIUM-CONFIDENCE patterns (medium + high) ----

    # System prompt extraction / prompt-leak attempts
    _SYSTEM_PROMPT_EXTRACTION: list[re.Pattern[str]] = [
        re.compile(
            r"\b(?:(?:show|reveal|print|display|output|repeat|echo|tell|give|what\s+(?:is|are))"
            r"(?:\s+me)?\s+(?:your|the)\s+(?:system\s+(?:prompt|message|instructions?)|"
            r"initial\s+(?:prompt|instructions?)|hidden\s+(?:prompt|instructions?)|"
            r"(?:original|full|complete|entire)\s+(?:prompt|instructions?)))",
            re.I,
        ),
        # Prompt-leak: "repeat / print the text above ..."
        re.compile(
            r"\b(?:repeat|print|output|echo|show|reveal|say|spell\s+out)\s+(?:back\s+)?"
            r"(?:me\s+)?(?:the\s+|all\s+(?:the\s+)?|everything\s+)?"
            r"(?:text|words?|content|message|prompt|instructions?|everything)\s+"
            r"(?:above|before|preceding|that\s+(?:came|comes|appeared?)\s+(?:before|above))",
            re.I,
        ),
    ]

    # Role-play injection: "You are DAN", jailbreak patterns
    _ROLEPLAY_INJECTION: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:you\s+are\s+(?:DAN|STAN|DUDE|AIM|KEVIN|BARD|Sydney)\b|"
            r"enable\s+(?:developer|DAN|jailbreak)\s+mode|"
            r"(?:enter|switch\s+to|activate)\s+(?:unrestricted|unfiltered|uncensored|DAN)\s+mode)",
            re.I,
        ),
    ]

    # Role-delimiter spoofing: faking system/assistant/user message boundaries
    # so the model treats injected text as a privileged turn.
    _ROLE_DELIMITER_SPOOF: list[re.Pattern[str]] = [
        re.compile(r"</?\s*(?:system|assistant|user|human)\s*>", re.I),
        re.compile(r"<\|\s*(?:system|im_start|im_end|endoftext|assistant|user)\b[^>]*\|?>", re.I),
        re.compile(r"(?:^|\n)\s*\[/?INST\]", re.I),
        re.compile(r"(?:^|\n)\s*<</?SYS>>", re.I),
    ]

    # Direct "System:" / "### New system prompt:" header injection
    _SYSTEM_HEADER_INJECTION: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:#{1,4}\s*)?(?:new\s+|updated\s+|revised\s+)?"
            r"system\s+(?:prompt|message|instructions?)\s*:",
            re.I,
        ),
        # Bare ALL-CAPS "SYSTEM:" header (case-sensitive to avoid benign prose
        # like "the operating system: linux").
        re.compile(r"(?:^|\n)\s*SYSTEM\s*:\s*\S"),
    ]

    # Plaintext role-turn spoofing: a line that opens with a conversational
    # role label ("System:"/"Assistant:"/"Human:"/"User:"/"AI:") and content,
    # faking a turn boundary without any special delimiter token so the model
    # treats what follows as if it came from a privileged party or an earlier
    # (compliant) reply. Anchored to the start of a line so it does not fire
    # on the label appearing mid-sentence (e.g. "the operating system: linux").
    _ROLE_LABEL_SPOOF: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:#{1,4}\s*)?(?:system|assistant|human|user|ai)\s*:\s*\S",
            re.I,
        ),
    ]

    # ---- HIGH-SENSITIVITY patterns (high only) ----

    # Markdown-style divider + system prompt injection.
    _DELIMITER_INJECTION: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:---+\s*\n\s*(?:System|New\s+System)\s*(?:Prompt|Instructions?)\s*:)",
            re.I,
        ),
        re.compile(
            r"(?:^|\n)\s*###\s*(?:System|Human|Assistant|User)\s*(?::|###)",
            re.I,
        ),
    ]

    def __init__(
        self,
        sensitivity: str = "medium",
        positions: list[GuardrailPosition] | None = None,
        enabled: bool = True,
    ):
        super().__init__(
            name="PromptInjectionGuardrail",
            positions=positions or [GuardrailPosition.INPUT],
            enabled=enabled,
        )
        if sensitivity not in ("low", "medium", "high"):
            raise ValueError(f"Invalid sensitivity: {sensitivity!r}. Must be 'low', 'medium', or 'high'.")
        self.sensitivity = sensitivity

    def _get_patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        """Return (label, pattern) pairs for the current sensitivity level."""
        patterns: list[tuple[str, re.Pattern[str]]] = []

        # Low: always included
        for p in self._INSTRUCTION_OVERRIDE:
            patterns.append(("instruction_override", p))
        for p in self._IDENTITY_OVERRIDE:
            patterns.append(("identity_override", p))

        if self.sensitivity in ("medium", "high"):
            for p in self._SYSTEM_PROMPT_EXTRACTION:
                patterns.append(("system_prompt_extraction", p))
            for p in self._ROLEPLAY_INJECTION:
                patterns.append(("roleplay_injection", p))
            for p in self._ROLE_DELIMITER_SPOOF:
                patterns.append(("role_delimiter_spoof", p))
            for p in self._SYSTEM_HEADER_INJECTION:
                patterns.append(("system_header_injection", p))
            for p in self._ROLE_LABEL_SPOOF:
                patterns.append(("role_label_spoof", p))

        if self.sensitivity == "high":
            for p in self._DELIMITER_INJECTION:
                patterns.append(("delimiter_injection", p))

        return patterns

    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        for label, pattern in self._get_patterns():
            match = pattern.search(content)
            if match:
                return GuardrailResult(
                    passed=False,
                    reason=f"Prompt injection guardrail: {label} detected",
                    metadata={
                        "pattern_type": label,
                        "matched_text": match.group()[:100],
                        "sensitivity": self.sensitivity,
                    },
                )

        return GuardrailResult(passed=True)
