"""Unit tests for streaming correctness.

Covers:
- No-tool ``Agent.stream()`` uses the direct path (no ReAct scaffold, no
  "[Max iterations reached]"); tokens stream incrementally.
- Mid-stream provider errors are *raised* at the iterator boundary, not
  buffered into a normal chunk that looks like model output.
- The final assembled answer is sanitized before it is
  handed to ``on_answer`` and stored in short-term memory.
- The ReAct/tool path also fails honestly (raises) on a streaming error.
- OpenAI-compatible streaming emits a final usage chunk when the client opts
  in via ``stream_options.include_usage``, and omits it otherwise.

These use an in-process fake model (no network) — not a mock of live API
behaviour, which the project forbids.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount


class _StreamModel(BaseModel):
    """In-process model that streams ``tokens`` and may raise after ``raise_after``."""

    def __init__(self, tokens: list[str], *, raise_after: int | None = None,
                 exc: Exception | None = None):
        super().__init__(model_name="fake-stream", model_type=ModelType.OPENAI)
        self._tokens = tokens
        self._raise_after = raise_after
        self._exc = exc or RuntimeError("provider stream blew up")
        self.stream_calls = 0

    def load(self) -> None:
        pass

    def generate(self, prompt, config=None, **kwargs):
        return GenerationResult(
            text="".join(self._tokens), tokens_used=len(self._tokens),
            finish_reason="stop", model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        self.stream_calls += 1
        for i, tok in enumerate(self._tokens):
            if self._raise_after is not None and i == self._raise_after:
                raise self._exc
            yield tok

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 4096

    def unload(self) -> None:
        pass

    def generate_batch(self, prompts, config=None, **kwargs):
        return [self.generate(p, config) for p in prompts]


def _agent(model, tools=None):
    return Agent(config=AgentConfig(name="stream-test", model=model, tools=tools or []))


# --------------------------------------------------------------------------- #
# No-tool direct streaming
# --------------------------------------------------------------------------- #

def test_no_tool_stream_is_direct_no_scaffold():
    """A no-tool agent streams the answer directly — no ReAct bookkeeping leaks."""
    model = _StreamModel(["1", " 2", " 3", " 4", " 5"])
    out = list(_agent(model).stream("Count from 1 to 5"))
    text = "".join(out)
    assert text == "1 2 3 4 5"
    assert "[Max iterations reached]" not in text
    assert "Thought:" not in text and "Action:" not in text
    assert model.stream_calls == 1  # one direct pass, not a ReAct loop


def test_no_tool_stream_increments_token_by_token():
    """Tokens are yielded one at a time (true incrementality)."""
    model = _StreamModel(["a", "b", "c", "d"])
    chunks = list(_agent(model).stream("hi"))
    assert chunks == ["a", "b", "c", "d"]


def test_final_answer_sanitized_and_stored():
    """Assembled answer is sanitized before on_answer + memory."""
    model = _StreamModel(["Final Answer: ", "Canberra"])
    captured: list[str] = []
    agent = _agent(model)
    list(agent.stream("Capital of Australia?", on_answer=captured.append))
    assert captured == ["Canberra"]  # leading "Final Answer:" label stripped
    msgs = agent.short_term_memory.get_recent_messages()
    assert any(
        getattr(m.role, "value", m.role) == "assistant" and m.content == "Canberra"
        for m in msgs
    )


# --------------------------------------------------------------------------- #
# Honest mid-stream errors
# --------------------------------------------------------------------------- #

def test_mid_stream_error_raises_not_buffered_direct():
    """A provider error mid-stream is raised, never yielded as a fake chunk."""
    model = _StreamModel(["ok ", "still ok "], raise_after=1,
                         exc=RuntimeError("boom"))
    received: list[str] = []
    with pytest.raises(RuntimeError, match="boom"):
        for tok in _agent(model).stream("anything"):
            received.append(tok)
    # The good tokens before the failure streamed; the error did NOT become a chunk.
    assert received == ["ok "]
    assert not any("boom" in c or "[Error" in c for c in received)


def test_immediate_stream_error_raises_with_no_chunks():
    model = _StreamModel(["x"], raise_after=0, exc=ValueError("nope"))
    received: list[str] = []
    with pytest.raises(ValueError, match="nope"):
        for tok in _agent(model).stream("anything"):
            received.append(tok)
    assert received == []


def test_react_path_stream_error_raises():
    """The tool/ReAct streaming path also fails honestly (raises)."""
    from effgen.tools.builtin.calculator import Calculator

    model = _StreamModel(["Thought:"], raise_after=0, exc=RuntimeError("react boom"))
    received: list[str] = []
    with pytest.raises(RuntimeError, match="react boom"):
        for tok in _agent(model, tools=[Calculator()]).stream("15 squared?"):
            received.append(tok)
    assert not any("[Error" in c for c in received)


# --------------------------------------------------------------------------- #
# OpenAI-compatible streaming usage
# --------------------------------------------------------------------------- #

def _compat_client():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from effgen.api.openai_compat import create_openai_router

    def runner(prompt, *, model, tools, stream, **kw):
        if stream:
            return iter(["Hello", " world", "!"])
        return "Hello world!"

    app = FastAPI()
    app.include_router(create_openai_router(runner))
    return TestClient(app)


def _usage_chunks(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            payload = json.loads(line[len("data: "):])
            if "usage" in payload:
                out.append(payload)
    return out


def test_compat_stream_emits_usage_when_requested():
    client = _compat_client()
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert resp.status_code == 200
    chunks = _usage_chunks(resp.text)
    assert len(chunks) == 1
    usage = chunks[0]["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    # OpenAI's usage-only chunk has empty choices.
    assert chunks[0]["choices"] == []


def test_compat_stream_omits_usage_by_default():
    client = _compat_client()
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert resp.status_code == 200
    assert _usage_chunks(resp.text) == []
    assert "[DONE]" in resp.text
