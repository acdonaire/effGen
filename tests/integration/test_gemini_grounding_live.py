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

# gemini-2.5-flash supports Google Search grounding.
# Fall back to gemini-2.5-flash-lite if 2.5-flash hits daily quota.
_GROUNDING_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


def _grounding_generate(prompt: str, grounding: bool, max_tokens: int = 512):
    """Try grounding-capable models in order; skip on sustained quota exhaustion."""
    from effgen.models.base import GenerationConfig
    from effgen.models.gemini_adapter import GeminiAdapter

    last_exc: Exception | None = None
    for model_id in _GROUNDING_MODELS:
        try:
            adapter = GeminiAdapter(model_name=model_id, api_key=GOOGLE_API_KEY)
            with adapter:
                return adapter.generate(
                    prompt,
                    config=GenerationConfig(grounding=grounding, max_tokens=max_tokens),
                )
        except RuntimeError as exc:
            if "quota" in str(exc).lower() or "429" in str(exc):
                last_exc = exc
                continue
            raise
    pytest.skip(f"All grounding models exhausted daily quota: {last_exc}")


def test_grounding_returns_real_urls():
    result = _grounding_generate(
        "What's a major news headline from this week?", grounding=True
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
    result = _grounding_generate(
        "Name the capital of France.", grounding=False, max_tokens=64
    )

    chunks = result.metadata.get("grounding_chunks", [])
    assert chunks == [], f"Expected no grounding chunks without grounding=True, got: {chunks}"
