"""Output-token budgeting, truncation reporting, per-run cost, and parsed output.

These cover the behaviours a token-heavy task on a reasoning model needs:

* reasoning families (gpt-5*, o-series) get a larger default output budget so a
  small cap is not entirely consumed by hidden reasoning, leaving an empty —
  but billed — result;
* an empty result with ``finish_reason="length"`` is treated as deterministic
  truncation (grow the budget once, or fail with an actionable message) instead
  of a misleading, retried "empty response";
* every run surfaces ``cost_usd`` + token counts on the response metadata;
* ``output_schema=PydanticModel`` populates ``metadata["parsed"]`` just like
  ``output_model=`` does.

The model is a small in-process fake so the *agent's* control flow is exercised
deterministically — no live behaviour is mocked.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel as PydanticBaseModel

from effgen.core.agent import Agent, AgentConfig
from effgen.core.structured_output import is_pydantic_model_class
from effgen.models._adapter_utils import (
    default_max_output_tokens,
    needs_reasoning_headroom,
)
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount


class _BudgetModel(BaseModel):
    """Fake model whose output depends on the ``max_tokens`` it is handed.

    Returns empty text with ``finish_reason="length"`` whenever the budget is
    below ``min_tokens_for_text`` (mimicking a reasoning model that spends the
    whole budget before emitting a visible token); otherwise returns ``text``.
    Records every budget it saw so tests can assert escalation behaviour.
    """

    def __init__(
        self,
        *,
        name: str = "fake-model",
        min_tokens_for_text: int | None = None,
        text: str = "OK",
        metadata: dict | None = None,
    ):
        super().__init__(model_name=name, model_type=ModelType.OPENAI)
        self._min = min_tokens_for_text
        self._text = text
        self._meta = metadata or {}
        self.seen_max_tokens: list[int | None] = []

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def generate(self, prompt, config=None, **kwargs):
        mt = config.max_tokens if config else None
        self.seen_max_tokens.append(mt)
        if self._min is not None and (mt is None or mt < self._min):
            return GenerationResult(
                text="", tokens_used=mt or 0, finish_reason="length",
                model_name=self.model_name, metadata=dict(self._meta),
            )
        return GenerationResult(
            text=self._text, tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata=dict(self._meta),
        )

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield self._text

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:  # pragma: no cover
        return 200_000

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


def _agent(model: _BudgetModel, **cfg) -> Agent:
    return Agent(
        AgentConfig(
            raise_on_error=False,  # these assert the failure response, not the raise
            name="t", model=model, tools=[],
            enable_sub_agents=False, enable_memory=False, **cfg,
        )
    )


# ---------------------------------------------------------------------------
# reasoning-headroom detection
# ---------------------------------------------------------------------------


def test_needs_reasoning_headroom_by_name():
    assert needs_reasoning_headroom(_BudgetModel(name="gpt-5-nano")) is True
    assert needs_reasoning_headroom(_BudgetModel(name="gpt-5-mini")) is True
    assert needs_reasoning_headroom(_BudgetModel(name="o3-mini")) is True
    assert needs_reasoning_headroom(_BudgetModel(name="gpt-4o-mini")) is False
    assert needs_reasoning_headroom(_BudgetModel(name="llama-3.1-8b-instant")) is False
    # A "provider:" prefix is tolerated.
    assert needs_reasoning_headroom(_BudgetModel(name="openai:gpt-5")) is True


def test_needs_reasoning_headroom_by_flag():
    m = _BudgetModel(name="some-model")
    m._is_reasoning_model = True
    assert needs_reasoning_headroom(m) is True


def test_default_max_output_tokens():
    assert default_max_output_tokens(_BudgetModel(name="gpt-4o-mini")) == 1024
    assert default_max_output_tokens(_BudgetModel(name="gpt-5-nano")) == 4096
    assert default_max_output_tokens(_BudgetModel(name="gpt-4o-mini"), base=2048) == 2048
    assert default_max_output_tokens(_BudgetModel(name="gpt-5-nano"), base=2048) == 4096


# ---------------------------------------------------------------------------
# reasoning default budget + truncation reporting
# ---------------------------------------------------------------------------


def test_reasoning_model_gets_larger_default_budget():
    # A reasoning model needs >1024 tokens to emit text; with the old fixed
    # default of 1024 it would have returned empty. The larger default works.
    m = _BudgetModel(name="gpt-5-nano", min_tokens_for_text=2000, text="hello")
    r = _agent(m).run("token-heavy task")
    assert r.success is True
    assert m.seen_max_tokens[0] == 4096  # not the old 1024


def test_truncation_escalates_then_succeeds():
    # First call at the 4096 default truncates; the budget grows to 8192 and the
    # next call succeeds. No temperature-retry storm.
    m = _BudgetModel(name="gpt-5-nano", min_tokens_for_text=5000, text="recovered")
    r = _agent(m).run("very token-heavy task")
    assert r.success is True
    assert m.seen_max_tokens == [4096, 8192]


def test_truncation_fails_with_actionable_message_not_empty_response():
    # The budget can never satisfy this model: the run fails fast with a
    # truncation error (not the misleading "empty response after retries"),
    # and it is not retried at the same budget.
    m = _BudgetModel(name="gpt-5-nano", min_tokens_for_text=10**9)
    r = _agent(m).run("impossible task")
    assert r.success is False
    err = r.metadata.get("error") or {}
    assert err.get("category") == "truncation"
    assert err.get("retryable") is False
    assert "max_tokens" in r.output
    assert "empty response after retries" not in r.output
    # Escalated 4096 -> 8192 then gave up; never spun a 3x same-budget retry.
    assert m.seen_max_tokens == [4096, 8192]


def test_user_pinned_max_tokens_is_respected_and_not_escalated():
    # When the caller pins max_tokens, honour it: no escalation, and a clear
    # truncation message rather than silently growing the budget.
    m = _BudgetModel(name="gpt-5-nano", min_tokens_for_text=10**9)
    r = _agent(m).run("task", max_tokens=64)
    assert r.success is False
    assert (r.metadata.get("error") or {}).get("category") == "truncation"
    assert m.seen_max_tokens == [64]


# ---------------------------------------------------------------------------
# per-run cost + token usage on the response metadata
# ---------------------------------------------------------------------------


def test_cost_and_tokens_on_response_metadata():
    meta = {"cost_usd": 0.0012, "prompt_tokens": 30, "completion_tokens": 70,
            "total_tokens": 100}
    m = _BudgetModel(name="gpt-4o-mini", text="answer", metadata=meta)
    r = _agent(m).run("hi")
    assert r.success is True
    assert r.metadata.get("cost_usd") == pytest.approx(0.0012)
    assert r.metadata.get("prompt_tokens") == 30
    assert r.metadata.get("completion_tokens") == 70
    assert r.metadata.get("total_tokens") == 100


def test_no_cost_key_when_provider_reports_none():
    # A local model with no cost data must not fabricate a $0 cost — the key is
    # simply absent, though token counts still surface when present.
    m = _BudgetModel(name="local-model", text="answer",
                     metadata={"completion_tokens": 12})
    r = _agent(m).run("hi")
    assert "cost_usd" not in r.metadata
    assert r.metadata.get("completion_tokens") == 12


# ---------------------------------------------------------------------------
# per-run latency folded onto the response metadata (like cost/tokens)
# ---------------------------------------------------------------------------


def test_latency_on_response_metadata():
    # Every run surfaces latency on metadata so callers can budget per call
    # without wrapping each run in their own timer — even a local model with no
    # cost data still reports latency.
    m = _BudgetModel(name="local-model", text="answer",
                     metadata={"completion_tokens": 5})
    r = _agent(m).run("hi")
    assert r.metadata.get("latency_ms") is not None
    assert r.metadata.get("duration_s") is not None
    # latency_ms is the millisecond mirror of duration_s / execution_time.
    assert r.metadata["latency_ms"] == pytest.approx(r.metadata["duration_s"] * 1000.0, rel=0.01)
    assert r.metadata["duration_s"] == pytest.approx(r.execution_time, rel=0.05, abs=0.01)


# ---------------------------------------------------------------------------
# output_schema=PydanticModel populates metadata["parsed"]
# ---------------------------------------------------------------------------


class _Figures(PydanticBaseModel):
    ticker: str
    revenue_musd: float


def test_is_pydantic_model_class():
    assert is_pydantic_model_class(_Figures) is True
    assert is_pydantic_model_class({"type": "object"}) is False
    assert is_pydantic_model_class(_Figures(ticker="X", revenue_musd=1.0)) is False
    assert is_pydantic_model_class(None) is False


def test_output_schema_pydantic_populates_parsed():
    m = _BudgetModel(name="gpt-4o-mini",
                     text='{"ticker": "ACME", "revenue_musd": 1284.6}')
    r = _agent(m).run("extract figures", output_schema=_Figures)
    parsed = r.metadata.get("parsed")
    assert isinstance(parsed, _Figures)
    assert parsed.ticker == "ACME"
    assert parsed.revenue_musd == pytest.approx(1284.6)


# ---------------------------------------------------------------------------
# str(GenerationResult) is the text, mirroring AgentResponse
# ---------------------------------------------------------------------------


def test_generation_result_str_is_text():
    g = GenerationResult(text="The answer is 42.", tokens_used=5,
                         finish_reason="stop", model_name="m")
    assert str(g) == "The answer is 42."
    # print() / f-strings show the answer, not the dataclass repr...
    assert f"{g}" == "The answer is 42."
    # ...but repr() still carries every field for debugging.
    assert "GenerationResult(" in repr(g)
    assert "tokens_used=5" in repr(g)


# ---------------------------------------------------------------------------
# AgentConfig turns a stray engine= into an actionable error
# ---------------------------------------------------------------------------


def test_agentconfig_rejects_model_load_kwargs_with_hint():
    with pytest.raises(TypeError) as ei:
        AgentConfig(name="x", model="Qwen/Qwen2.5-1.5B-Instruct", engine="transformers")
    msg = str(ei.value)
    assert "engine" in msg
    assert "load_model" in msg
    assert "create_agent" in msg


def test_agentconfig_normal_construction_still_works():
    cfg = AgentConfig(name="ok", model="gpt-5-nano", require_model=False,
                      temperature=0.3)
    assert cfg.name == "ok"
    assert cfg.temperature == pytest.approx(0.3)
    # dataclasses.replace round-trips through the wrapped __init__
    import dataclasses
    assert dataclasses.replace(cfg, temperature=0.9).temperature == pytest.approx(0.9)
