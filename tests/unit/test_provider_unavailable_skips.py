"""A provider that would not serve the call is a skip; anything else fails.

The rule and the reasoning live in ``tests/_harness/provider_unavailable.py``;
the hook that applies it is in ``tests/conftest.py``. A hook that turns failures
into skips hides real breakage the moment it is too broad, so the negative cases
below matter more than the positive ones — particularly the connection-error
case, which is the shape a local misconfiguration takes.
"""
from __future__ import annotations

import pytest

from tests._harness.provider_unavailable import (
    first_provider_sentence,
    provider_unavailable_reason,
)


@pytest.mark.parametrize(
    "text",
    [
        # A spent free-tier daily quota.
        (
            "RuntimeError: Gemini generation failed [rate_limited]: 429 RESOURCE_EXHAUSTED. "
            "Quota exceeded for metric: generate_content_free_tier_requests, limit: 500"
        ),
        # A full inference queue.
        (
            "AssertionError: Live eval failed: model error: Cerebras rate limit hit for "
            "gpt-oss-120b: Error code: 429 - {'code': 'queue_exceeded'}"
        ),
        # A per-minute token allowance, which Groq reports as 413.
        (
            "groq: Error code: 413 - Request too large ... on tokens per minute (TPM): "
            "Limit 6000, Requested 8212"
        ),
        # An account with no credit.
        "replicate: Error code: 402 - insufficient credit for this prediction",
        # The provider's own infrastructure.
        "openai: Error code: 503 - Service Unavailable",
        "anthropic: overloaded_error: the model is currently overloaded",
    ],
)
def test_a_provider_that_cannot_serve_is_recognised(text):
    assert provider_unavailable_reason(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        # THE case this rule must never absorb. A blank OPENAI_BASE_URL sent 45
        # live cells here on 14 Aug 2026; the wording blames the provider and
        # the cause was local. Converting it to a skip hides the defect.
        (
            "RuntimeError: OpenAI generation failed [will_retry]: Connection error.. "
            "Transient provider error — retry shortly; check the provider status page."
        ),
        "httpx.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol.",
        # A real assertion about the answer.
        "AssertionError: expected 'red' in response; got: 'blue'",
        # effGen built a bad request: a defect, and a 400 is not capacity.
        "openai: Error code: 400 - unknown parameter 'reasoning_effort'",
        # A rejected credential is a configuration problem, not exhaustion.
        "openai: Error code: 401 - Incorrect API key provided",
        # No provider is named, so this is not about a provider call at all.
        "AssertionError: rate limit bucket should refill after 1s",
    ],
)
def test_everything_else_still_fails(text):
    assert provider_unavailable_reason(text) is None


def test_the_reason_quotes_the_provider_s_own_sentence():
    text = (
        "tests/x.py:12: in test_thing\n"
        "E   RuntimeError: Gemini generation failed [rate_limited]: Error code: 429 - quota\n"
    )
    assert "429" in first_provider_sentence(text)


def test_an_empty_failure_is_not_a_provider_problem():
    assert provider_unavailable_reason("") is None
    assert first_provider_sentence("") == ""
