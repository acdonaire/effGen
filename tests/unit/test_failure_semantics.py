"""Unit tests for failure semantics & error taxonomy.

Covers:
- classify_provider_error() mapping (effGen typed errors + raw SDK shapes).
- run() never returns success=True with empty output; the no-tool direct path
  and the ReAct tool path return the identical failure shape.
- Retry classification: non-retryable errors (auth/not_found) fail fast; only
  retryable/rate-limited errors are retried.
- AgentConfig.raise_on_error raises a typed error.
- metadata["reason"] taxonomy (final_answer / generation_failed).
- AgentConfig.require_model defaults to True and provider/raise_on_error exist.
- VLLMEngine import-error distinction (not-installed vs ABI/CUDA failure).

These use an in-process fake model (no network) — not a mock of live API
behavior, which the project forbids.
"""

from __future__ import annotations

import dataclasses

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.models.errors import (
    InvalidRequestError,
    ModelAuthError,
    ModelNotFoundError,
    ModelRefusalError,
    ModelTimeoutError,
    ProviderTransientError,
    classify_provider_error,
)


class _FakeModel(BaseModel):
    """Minimal in-process model: either returns fixed text or raises `exc`."""

    def __init__(self, *, text: str = "", exc: Exception | None = None, provider: str = "fake"):
        super().__init__(model_name="fake-model", model_type=ModelType.OPENAI)
        self._text = text
        self._exc = exc
        self._provider = provider
        self.calls = 0

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def generate(self, prompt, config=None, **kwargs):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return GenerationResult(
            text=self._text, tokens_used=3, finish_reason="stop", model_name=self.model_name, metadata={}
        )

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield self._text

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:  # pragma: no cover
        return 4096

    def unload(self) -> None:  # pragma: no cover
        pass

    def generate_batch(self, prompts, config=None, **kwargs):  # pragma: no cover
        return [self.generate(p, config=config) for p in prompts]

    def generate_with_tools(self, prompt, tools, config=None, **kwargs):  # pragma: no cover
        return self.generate(prompt, config=config)

    def supports_function_calling(self) -> bool:
        return False

    def supports_tool_calling(self) -> bool:
        return False


def _agent(model: _FakeModel, *, tools=None, **cfg_kwargs) -> Agent:
    return Agent(
        AgentConfig(
            name="t",
            model=model,
            tools=tools or [],
            enable_sub_agents=False,
            enable_memory=False,
            **cfg_kwargs,
        )
    )


# ---------------------------------------------------------------------------
# classify_provider_error
# ---------------------------------------------------------------------------


def test_classify_typed_errors():
    assert classify_provider_error(ModelAuthError("openai")).category == "auth"
    assert classify_provider_error(ModelAuthError("openai")).should_retry is False
    assert classify_provider_error(ModelNotFoundError("cerebras", "x")).category == "not_found"
    assert classify_provider_error(ModelNotFoundError("cerebras")).should_retry is False
    assert classify_provider_error(ModelRefusalError("no")).category == "refusal"
    assert classify_provider_error(InvalidRequestError("groq")).category == "invalid_request"
    assert classify_provider_error(ModelTimeoutError("replicate")).should_retry is True
    assert classify_provider_error(ProviderTransientError("groq", status_code=503)).should_retry is True


def test_classify_by_status_code():
    class _E(Exception):
        def __init__(self, status):
            self.status_code = status

    assert classify_provider_error(_E(401)).category == "auth"
    assert classify_provider_error(_E(404)).category == "not_found"
    assert classify_provider_error(_E(429)).category == "rate_limited"
    assert classify_provider_error(_E(429)).should_retry is True
    assert classify_provider_error(_E(400)).category == "invalid_request"
    assert classify_provider_error(_E(503)).should_retry is True


def test_classify_by_name_and_message():
    assert classify_provider_error(type("RateLimitError", (Exception,), {})()).category == "rate_limited"
    assert classify_provider_error(type("AuthenticationError", (Exception,), {})()).category == "auth"
    assert classify_provider_error(Exception("Invalid API key provided")).category == "auth"
    assert classify_provider_error(Exception("The model does not exist")).category == "not_found"
    # Unknown errors default to retryable so a transient blip is not hard-failed.
    assert classify_provider_error(Exception("something weird")).should_retry is True


# ---------------------------------------------------------------------------
# run() failure semantics
# ---------------------------------------------------------------------------


def test_direct_path_never_empty_success():
    a = _agent(_FakeModel(exc=ModelNotFoundError("cerebras", "x")))
    r = a.run("hi")
    assert r.success is False
    assert r.metadata["reason"] == "generation_failed"
    assert isinstance(r.metadata["error"], dict)
    assert r.metadata["error"]["category"] == "not_found"
    assert r.output  # non-empty, human-readable


def test_tool_path_same_shape_as_direct_path():
    from effgen.tools.builtin.calculator import Calculator

    err = ModelNotFoundError("cerebras", "x")
    direct = _agent(_FakeModel(exc=err)).run("hi")
    tooled = _agent(_FakeModel(exc=err), tools=[Calculator()]).run("hi")
    assert direct.success is tooled.success is False
    assert direct.metadata["reason"] == tooled.metadata["reason"] == "generation_failed"
    assert direct.metadata["error"]["category"] == tooled.metadata["error"]["category"]
    assert set(direct.metadata["error"]) >= {"type", "provider", "model", "message", "category"}


def test_no_retry_storm_on_auth():
    m = _FakeModel(exc=ModelAuthError("openai"))
    r = _agent(m).run("hi")
    assert r.success is False
    assert m.calls == 1  # exactly one call — no 3x retry storm


def test_retries_only_retryable(monkeypatch):
    import effgen.core.agent as agent_mod

    monkeypatch.setattr(agent_mod.time, "sleep", lambda *_: None)
    m = _FakeModel(exc=ProviderTransientError("groq", status_code=503))
    r = _agent(m).run("hi")
    assert r.success is False
    assert m.calls == 3  # max_retries reached for a retryable error


def test_success_reason_final_answer():
    r = _agent(_FakeModel(text="Canberra")).run("capital of Australia?")
    assert r.success is True
    assert r.output.strip() == "Canberra"
    assert r.metadata["reason"] == "final_answer"


def test_raise_on_error_raises_typed():
    a = _agent(_FakeModel(exc=ModelNotFoundError("cerebras", "x")), raise_on_error=True)
    with pytest.raises(ModelNotFoundError):
        a.run("hi")

    a2 = _agent(_FakeModel(exc=ModelAuthError("openai")), raise_on_error=True)
    with pytest.raises(ModelAuthError):
        a2.run("hi")


def test_error_message_is_redacted():
    leaked = "boom sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd failed"
    r = _agent(_FakeModel(exc=ProviderTransientError("openai", message=leaked))).run("hi")
    # monkeypatch-free: the real redactor should scrub the key shape
    assert "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in r.metadata["error"]["message"]


# ---------------------------------------------------------------------------
# AgentConfig contract
# ---------------------------------------------------------------------------


def test_agentconfig_defaults_and_fields():
    fields = {f.name: f for f in dataclasses.fields(AgentConfig)}
    assert "provider" in fields
    assert "raise_on_error" in fields
    assert fields["require_model"].default is True
    assert fields["raise_on_error"].default is False
    assert fields["provider"].default is None


def test_require_model_true_fails_fast_on_bad_string_model():
    with pytest.raises(RuntimeError):
        Agent(
            AgentConfig(
                name="t",
                model="totally-made-up-model-9000",
                enable_sub_agents=False,
                enable_memory=False,
            )
        )


# ---------------------------------------------------------------------------
# Engine import-error distinction
# ---------------------------------------------------------------------------


def test_vllm_not_installed_vs_abi_error(monkeypatch):
    import importlib.util
    import sys

    from effgen.models.vllm_engine import VLLMEngine

    eng = VLLMEngine(model_name="x", tensor_parallel_size=0)

    # Force `from vllm import ...` to raise ImportError even though vllm may be
    # installed in the test env (setting the module to None does this).
    monkeypatch.setitem(sys.modules, "vllm", None)

    # Case 1: package genuinely absent -> "not installed"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="not installed"):
        eng.load()

    # Case 2: package present but import failed (ABI/CUDA) -> ABI message
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    with pytest.raises(RuntimeError, match="ABI"):
        eng.load()
