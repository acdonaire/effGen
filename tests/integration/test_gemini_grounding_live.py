"""Integration tests for Gemini grounding (live API calls, skipped without key)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
_project_env = Path(__file__).parent.parent.parent / ".env"
if _project_env.exists():
    load_dotenv(_project_env, override=False)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
pytestmark = pytest.mark.skipif(
    not GOOGLE_API_KEY,
    reason="GOOGLE_API_KEY not in ~/.effgen/.env",
)

# gemini-2.5-flash is required for Google Search grounding;
# gemini-3.1-flash-lite-preview hits quota on the grounding endpoint.
MODEL = "gemini-2.5-flash"


def test_grounding_returns_real_urls():
    from effgen.models.base import GenerationConfig
    from effgen.models.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
    with adapter:
        result = adapter.generate(
            "What's a major news headline from this week?",
            config=GenerationConfig(grounding=True, max_tokens=512),
        )

    assert result.text, "Expected non-empty response text"
    chunks = result.metadata.get("grounding_chunks", [])
    assert len(chunks) > 0, (
        f"Expected grounding_chunks to be non-empty; got: {chunks}\n"
        f"Full metadata: {result.metadata}"
    )
    urls = [c.get("url") for c in chunks if c.get("url")]
    assert len(urls) > 0, f"No URLs found in grounding_chunks: {chunks}"
    for url in urls:
        assert url.startswith("http"), f"Expected http URL, got: {url}"


def test_grounding_false_no_grounding_chunks():
    """Without grounding, metadata.grounding_chunks is empty."""
    from effgen.models.base import GenerationConfig
    from effgen.models.gemini_adapter import GeminiAdapter

    adapter = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
    with adapter:
        result = adapter.generate(
            "Name the capital of France.",
            config=GenerationConfig(grounding=False, max_tokens=64),
        )

    chunks = result.metadata.get("grounding_chunks", [])
    assert chunks == [], f"Expected no grounding chunks without grounding=True, got: {chunks}"
