"""Telling "the provider would not serve this" apart from "effGen is wrong".

A live test asserts something about effGen's behaviour, and it can only do that
if the provider ran the call. When a free tier is spent, a queue is full, or the
provider is having an outage, the run reports the *account's* state or the
*provider's* state — neither is a defect in the tree, and reading it as one
makes a suite that cannot be green for reasons nobody can fix.

So a failure whose text carries an unambiguous provider-capacity or
provider-outage statement is converted to a skip, the same way an absent
optional dependency is (see ``optional_deps``).

**The rule is deliberately narrow, and the exclusion matters more than the
inclusions.** A bare connection error is NOT here. "Connection error" is what
effGen reports when it cannot reach the endpoint at all, and the commonest cause
of that is local: a wrong or blank ``OPENAI_BASE_URL``, a proxy, a firewall. On
14 Aug 2026 exactly that misconfiguration turned 45 live cells red, and every
one of them said "Transient provider error — retry shortly; check the provider
status page". Converting that shape to a skip would have hidden the defect
instead of surfacing it. A provider that is genuinely down says so with a status
code and a sentence; that is what this matches.
"""

from __future__ import annotations

import re
from typing import Any

#: Statements that mean "your account has no capacity right now". Each is a
#: quota, a rate, or a billing state — recoverable by waiting or by paying, and
#: unaffected by any change to effGen.
_EXHAUSTED = (
    "[rate_limited]",
    "error code: 429",
    "resource_exhausted",
    "rate_limit_exceeded",
    "quota exceeded",
    "too_many_requests",
    "queue_exceeded",
    "tokens per minute",
    "requests per minute",
    "insufficient credit",
    "insufficient_quota",
    "error code: 402",
)

#: Statements that mean "the provider is not serving right now" — its own
#: infrastructure, reported with a server-side status code or its own words.
_PROVIDER_DOWN = (
    "error code: 500",
    "error code: 502",
    "error code: 503",
    "error code: 504",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "server had an error",
    "overloaded_error",
    "is currently overloaded",
    "model_not_ready",
)

#: A provider has to be named for the text to be about a provider call at all.
_PROVIDERS = (
    "openai", "anthropic", "gemini", "google", "groq", "cerebras", "together",
    "fireworks", "replicate", "huggingface", "hugging face", "mistral", "cohere",
)


def provider_unavailable_reason(text: str) -> str | None:
    """Return why the provider could not serve the call, or None.

    Args:
        text: The rendered failure text of a test.

    Returns:
        A short reason naming what was matched, or None when the failure is not
        an unambiguous provider-capacity or provider-outage statement.
    """
    if not text:
        return None
    low = text.lower()
    if not any(name in low for name in _PROVIDERS):
        return None

    for needle in _EXHAUSTED:
        if needle in low:
            return f"the provider has no capacity for this call ({needle.strip('[]')})"
    for needle in _PROVIDER_DOWN:
        if needle in low:
            return f"the provider is not serving right now ({needle})"
    return None


def first_provider_sentence(text: str, limit: int = 240) -> str:
    """Return the provider's own sentence from *text*, for the skip reason."""
    for line in text.splitlines():
        stripped = line.lstrip("E ").strip()
        if re.search(r"error code: \d{3}|\[rate_limited\]|RESOURCE_EXHAUSTED", stripped, re.I):
            return stripped[:limit]
    return text.strip().splitlines()[-1][:limit] if text.strip() else ""


def assert_cli_succeeded(proc: Any, what: str = "the command") -> None:
    """Assert a CLI subprocess exited 0, keeping both of its streams.

    The rule above classifies a failure by reading its text, and across a
    subprocess boundary the provider's sentence only reaches the parent through
    what the child printed. A CLI that reports per-item outcomes prints them on
    **stdout** — with ``--json``, the refusal is a field in that document — while
    the process-level summary goes to stderr. So a message built from stderr
    alone drops exactly the sentence the rule needs, and a throttled run is
    reported as a defect in the tree.

    Both streams therefore go into the message. Use this instead of
    ``assert proc.returncode == 0, proc.stderr`` wherever the subprocess calls a
    provider.
    """
    if proc.returncode == 0:
        return
    raise AssertionError(
        f"{what} exited {proc.returncode}\n"
        f"--- stderr ---\n{proc.stderr}\n"
        f"--- stdout ---\n{proc.stdout}"
    )
