"""Tests for PubMedTool.

The operation tests hit the real NCBI E-utilities endpoint; a network failure or
a transient upstream condition (a 429, a body cut short) is a skip rather than a
failure, because it says nothing about this code. The retry contract those
conditions trigger cannot be requested from NCBI on demand, so it is pinned at
the end of the file against a stand-in transport.
"""

from __future__ import annotations

import asyncio
import socket
import time
from http.client import IncompleteRead
from urllib.error import HTTPError

import pytest

from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin import PubMedTool
from effgen.tools.builtin import pubmed as pubmed_module


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


#: Upstream conditions the tool reports after exhausting its own backoff. NCBI
#: sheds load with 429s and with bodies cut short mid-read, and neither says
#: anything about this code, so they skip instead of failing the run.
_UPSTREAM_SIGNS = (
    "429",
    "truncated response",
    "network error",
    "dropped",
    "timed out",
    "after 4 retries",
)


def _ok(result: ToolResult):
    assert isinstance(result, ToolResult)
    if not result.success:
        error = result.error or ""
        if any(sign in error.lower() for sign in _UPSTREAM_SIGNS):
            pytest.skip(f"PubMed upstream unavailable: {error}")
        pytest.fail(f"tool failed: {error}")
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


# -- transient upstream conditions retry rather than failing the call ---------
#
# NCBI cannot be asked for a truncated body on demand, so the retry contract is
# pinned against a stand-in transport; the operations above prove the live path.

class _FakeResponse:
    """A response whose ``read`` either returns a body or breaks mid-body."""

    def __init__(self, body: bytes | None) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def read(self) -> bytes:
        if self._body is None:
            raise IncompleteRead(b"")
        return self._body


_SEARCH_BODY = b'{"esearchresult": {"idlist": [], "count": "0"}}'


def _transport(bodies: list[bytes | None], seen: list[str]):
    def opener(url, headers=None, timeout=20, allowed_hosts=None):
        seen.append(url)
        return _FakeResponse(bodies[min(len(seen) - 1, len(bodies) - 1)])

    return opener


def test_truncated_body_is_retried(monkeypatch):
    """A body cut short mid-read is transient: back off and ask again."""
    seen: list[str] = []
    monkeypatch.setattr(
        pubmed_module, "safe_urlopen", _transport([None, None, _SEARCH_BODY], seen),
    )
    result = _run(PubMedTool().execute(operation="search", query="crispr", max_results=1))
    assert result.success, result.error
    assert len(seen) == 3, "two truncated reads, then the answer"


def test_truncated_body_that_never_recovers_names_the_cause(monkeypatch):
    """Once the retries are spent the tool reports a network failure, not a repr."""
    seen: list[str] = []
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([None], seen))
    result = _run(PubMedTool().execute(operation="search", query="crispr", max_results=1))
    assert result.success is False
    assert "truncated response" in (result.error or "").lower()
    assert "retries" in (result.error or "")
    assert result.metadata["error_type"] == "ConnectionError"
    assert len(seen) == 5, "the initial call plus four retries"


def test_a_gateway_error_is_retried_too(monkeypatch):
    """A 503 from the gateway is the same shedding behavior as a 429."""
    seen: list[str] = []

    def opener(url, headers=None, timeout=20, allowed_hosts=None):
        seen.append(url)
        if len(seen) == 1:
            raise HTTPError(url, 503, "Service Unavailable", {}, None)
        return _FakeResponse(_SEARCH_BODY)

    monkeypatch.setattr(pubmed_module, "safe_urlopen", opener)
    result = _run(PubMedTool().execute(operation="search", query="crispr", max_results=1))
    assert result.success, result.error
    assert len(seen) == 2


def test_a_client_error_is_not_retried(monkeypatch):
    """A 400 is the request's fault, so it is reported at once."""
    seen: list[str] = []

    def opener(url, headers=None, timeout=20, allowed_hosts=None):
        seen.append(url)
        raise HTTPError(url, 400, "Bad Request", {}, None)

    monkeypatch.setattr(pubmed_module, "safe_urlopen", opener)
    result = _run(PubMedTool().execute(operation="search", query="crispr", max_results=1))
    assert result.success is False
    assert "400" in (result.error or "")
    assert len(seen) == 1, "a client error is not worth a retry"
