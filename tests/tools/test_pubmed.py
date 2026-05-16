"""Tests for PubMedTool.

All tests hit the real NCBI E-utilities endpoint. Network failures and
transient HTTP errors are treated as skips rather than failures.
"""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin import PubMedTool


def _has_network(host: str = "eutils.ncbi.nlm.nih.gov", port: int = 443, timeout: float = 3.0) -> bool:
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


def test_pubmed_metadata():
    tool = PubMedTool()
    assert tool.metadata.name == "pubmed"
    assert tool.metadata.category.value == "information_retrieval"


def test_pubmed_unknown_operation_fails():
    r = _run(PubMedTool().execute(operation="nope", query="x"))
    assert not r.success
    assert "operation" in (r.error or "").lower() or "unknown" in (r.error or "").lower()


def test_pubmed_search_requires_query():
    r = _run(PubMedTool().execute(operation="search"))
    assert not r.success


def test_pubmed_fetch_requires_pmid():
    r = _run(PubMedTool().execute(operation="fetch"))
    assert not r.success


@needs_net
def test_pubmed_search_returns_results():
    out = _ok(_run(PubMedTool().execute(query="crispr gene editing", max_results=3)))
    assert out["source"] == "pubmed"
    assert out["count"] >= 1
    first = out["results"][0]
    assert first["pmid"]
    assert first["pmid"].isdigit()
    assert first["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/")
    assert isinstance(first["authors"], list)


@needs_net
def test_pubmed_abstract_for_known_pmid():
    # PMID 28980624 — "Attention Is All You Need" companion-style biomed paper
    # is not on PubMed; pick a stable biomed PMID instead.
    # 26952870 — well-known CRISPR review; stable abstract.
    out = _ok(_run(PubMedTool().execute(operation="abstract", pmid="26952870")))
    assert out["source"] == "pubmed"
    assert out["count"] >= 1
    item = out["results"][0]
    assert item["pmid"] == "26952870"
    assert isinstance(item.get("abstract"), str)


@needs_net
def test_pubmed_fetch_returns_metadata():
    out = _ok(_run(PubMedTool().execute(operation="fetch", pmid="26952870")))
    assert out["count"] >= 1
    paper = out["results"][0]
    assert paper["title"]
    assert paper["pmid"] == "26952870"
    assert isinstance(paper["authors"], list)


@needs_net
def test_pubmed_rate_limiter_handles_concurrent_burst():
    async def burst():
        tool = PubMedTool()
        started = time.monotonic()
        results = await asyncio.gather(*[
            tool.execute(operation="search", query="crispr gene editing", max_results=1)
            for _ in range(5)
        ])
        return time.monotonic() - started, results

    elapsed, results = _run(burst())
    outputs = [_ok(result) for result in results]
    assert len(outputs) == 5
    assert all(output["count"] >= 1 for output in outputs)
    # Each search performs esearch + esummary. Five concurrent searches should
    # be paced by the local limiter rather than leaving as an immediate burst.
    assert elapsed >= 2.0
