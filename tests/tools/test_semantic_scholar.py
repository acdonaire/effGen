"""Tests for SemanticScholarTool.

All tests hit the real Semantic Scholar Graph API. Because the public,
unauthenticated endpoint is heavily rate-limited (100 req / 5 min globally),
transient 429 responses are treated as skips rather than failures.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin import SemanticScholarTool


def _has_network(host: str = "api.semanticscholar.org", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


NETWORK = _has_network()
needs_net = pytest.mark.skipif(not NETWORK, reason="no network")


def _run(coro):
    return asyncio.run(coro)


def _ok(result: ToolResult):
    assert isinstance(result, ToolResult)
    assert result.success, f"tool failed: {result.error}"
    return result.output


def test_semantic_scholar_metadata():
    assert SemanticScholarTool().metadata.name == "semantic_scholar"


def test_semantic_scholar_search_requires_query():
    r = _run(SemanticScholarTool().execute(operation="search"))
    assert not r.success


def test_semantic_scholar_paper_requires_id():
    r = _run(SemanticScholarTool().execute(operation="paper"))
    assert not r.success


@needs_net
def test_semantic_scholar_search_returns_papers():
    out = _ok(_run(SemanticScholarTool().execute(
        query="attention is all you need", max_results=3,
    )))
    assert out["source"] == "semantic_scholar"
    if out["count"] == 0:
        pytest.skip("S2 returned 0 results — transient")
    top_titles = [paper.get("title") or "" for paper in out["results"][:3]]
    top_authors = [author for paper in out["results"][:3] for author in paper.get("authors", [])]
    assert any("attention is all you need" in title.lower() for title in top_titles)
    assert any("Vaswani" in author for author in top_authors)


@needs_net
def test_semantic_scholar_paper_by_arxiv_id():
    out = _ok(_run(SemanticScholarTool().execute(
        operation="paper", paper_id="ARXIV:1706.03762",
    )))
    paper = out["result"]
    assert paper
    assert "Attention" in (paper.get("title") or "")


@needs_net
def test_semantic_scholar_references():
    out = _ok(_run(SemanticScholarTool().execute(
        operation="references", paper_id="ARXIV:1706.03762", max_results=3,
    )))
    assert out["source"] == "semantic_scholar"
    # References list might be empty under rate limiting; just assert shape.
    assert isinstance(out["results"], list)
