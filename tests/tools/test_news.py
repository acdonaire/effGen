"""Tests for NewsTool.

Live tests hit real RSS feeds. Network failures are treated as skips.
"""

from __future__ import annotations

import asyncio
import os
import socket

import pytest

from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin import NewsTool


def _has_network(host: str = "feeds.bbci.co.uk", port: int = 443, timeout: float = 4.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


NETWORK = _has_network()
needs_net = pytest.mark.skipif(not NETWORK, reason="no network")


def _run(coro):
    return asyncio.run(coro)


_TRANSIENT_MARKERS = (
    "HTTP 429",
    "HTTP 5",
    "Too Many Requests",
    "timed out",
    "Timeout",
    "Network error",
    "Connection",
)


def _ok(result: ToolResult) -> dict:
    assert isinstance(result, ToolResult)
    if not result.success and result.error and any(m in result.error for m in _TRANSIENT_MARKERS):
        pytest.skip(f"transient network error: {result.error}")
    assert result.success, f"tool failed: {result.error}"
    out = result.output
    inner_err = out.get("error") if isinstance(out, dict) else None
    if isinstance(out, dict) and out.get("success") is False and inner_err and any(m in str(inner_err) for m in _TRANSIENT_MARKERS):
        pytest.skip(f"transient feed error: {inner_err}")
    return out


# ---------------------------------------------------------------------------
# Metadata / unit tests
# ---------------------------------------------------------------------------

def test_news_metadata():
    tool = NewsTool()
    assert tool.metadata.name == "news"
    assert "news" in tool.metadata.tags


def test_news_unknown_operation_fails():
    r = _run(NewsTool().execute(operation="nope"))
    assert not r.success


def test_news_search_requires_query():
    r = _run(NewsTool().execute(operation="search"))
    assert not r.success


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

@needs_net
def test_news_top_headlines_general():
    out = _ok(_run(NewsTool().execute(operation="top_headlines", max_results=10)))
    assert out["success"] is True
    assert out["backend"] in ("rss", "newsapi")
    assert isinstance(out["articles"], list)
    assert out["count"] >= 10
    assert out["data"]["articles"] == out["articles"]
    first = out["articles"][0]
    assert "title" in first
    assert "link" in first
    assert "source" in first


@needs_net
def test_news_top_headlines_technology():
    out = _ok(_run(NewsTool().execute(operation="top_headlines", category="technology", max_results=5)))
    assert out["success"] is True
    assert out["category"] == "technology"
    assert isinstance(out["articles"], list)


@needs_net
def test_news_top_headlines_science():
    out = _ok(_run(NewsTool().execute(operation="top_headlines", category="science", max_results=5)))
    assert out["success"] is True
    assert out["category"] == "science"


@needs_net
def test_news_search_rss_fallback():
    """Without NEWS_API_KEY, should fall back to RSS search."""
    import os
    saved = os.environ.pop("NEWS_API_KEY", None)
    try:
        out = _ok(_run(NewsTool().execute(operation="search", query="technology", max_results=10)))
        assert out["success"] is True
        assert out["backend"] == "rss"
        assert out["query"] == "technology"
    finally:
        if saved:
            os.environ["NEWS_API_KEY"] = saved


@needs_net
def test_news_max_results_respected():
    out = _ok(_run(NewsTool().execute(operation="top_headlines", max_results=3)))
    assert len(out["articles"]) <= 3


@needs_net
def test_newsapi_backend_when_key_is_available():
    if not os.environ.get("NEWS_API_KEY"):
        pytest.skip("NEWS_API_KEY not set")
    out = _ok(_run(NewsTool().execute(operation="top_headlines", max_results=5)))
    assert out["backend"] == "newsapi"
    assert out["count"] >= 1
