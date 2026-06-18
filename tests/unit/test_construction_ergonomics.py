"""Construction/run input ergonomics — the obvious call works or fails clearly.

Covers three natural first-user calls that previously hit cryptic failures:

* ``run(output_schema=PydanticClass)`` returned a buried
  ``success=False, "Object of type ModelMetaclass is not JSON serializable"``
  instead of converting the class (it ships ``pydantic_model_to_schema``).
* ``Agent("groq:model")`` (a bare model string) raised a deferred
  ``AttributeError: 'str' object has no attribute 'name'`` from inside run().
* ``create_agent("minimal", model=..., engine="transformers")`` raised
  ``TypeError: AgentConfig.__init__() got an unexpected keyword argument 'engine'``.

These are offline, lightweight checks of the input validation/normalization
paths — no live provider calls. Live proof lives in
``tests/integration/test_construction_ergonomics_live.py``.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from effgen import Agent, create_agent
from effgen.core.agent import AgentConfig
from effgen.core.structured_output import normalize_output_schema


class _Capital(BaseModel):
    country: str
    capital: str


# ── output_schema accepts a JSON-Schema dict or a Pydantic class ──────────────

class TestOutputSchemaNormalization:
    def test_none_passthrough(self):
        assert normalize_output_schema(None) is None

    def test_dict_passthrough_identity(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert normalize_output_schema(schema) is schema

    def test_pydantic_class_converts_to_schema(self):
        out = normalize_output_schema(_Capital)
        assert isinstance(out, dict)
        assert out["type"] == "object"
        assert set(out["properties"]) == {"country", "capital"}
        # title fields are stripped so SLMs don't echo them back as data
        assert "title" not in out
        assert all("title" not in p for p in out["properties"].values())

    @pytest.mark.parametrize("bad", [12345, "a string", ["list"], 3.14, object()])
    def test_bad_type_raises_clear_typeerror(self, bad):
        with pytest.raises(TypeError) as exc:
            normalize_output_schema(bad)
        msg = str(exc.value)
        assert "output_schema" in msg
        assert "dict" in msg and "Pydantic" in msg

    def test_run_with_pydantic_class_does_not_crash_in_serialization(self):
        # require_model=False keeps construction cheap; the schema is normalized
        # before any model call, so a bad value errors immediately.
        agent = Agent(AgentConfig(name="t", model="x", require_model=False))
        with pytest.raises(TypeError):
            agent.run("hi", output_schema=12345)


# ── Agent(...) constructor validates its config ───────────────────────────────

class TestAgentConstructorGuard:
    @pytest.mark.parametrize("bad", ["groq:llama-3.1-8b-instant", None, 123, ["x"]])
    def test_non_agentconfig_raises_typeerror(self, bad):
        with pytest.raises(TypeError) as exc:
            Agent(bad)
        msg = str(exc.value)
        assert "AgentConfig" in msg
        assert "create_agent" in msg

    def test_valid_config_still_builds(self):
        agent = Agent(AgentConfig(name="t", model="x", require_model=False))
        assert agent.name == "t"

    @pytest.mark.parametrize("bad_model", [12345, ["x"], _Capital])
    def test_unsupported_model_type_fails_fast(self, bad_model):
        # AgentConfig.model that is neither a str id nor a loaded instance must
        # fail at construction, not silently build a model-less agent.
        with pytest.raises(TypeError) as exc:
            Agent(AgentConfig(name="t", model=bad_model))
        assert "AgentConfig.model" in str(exc.value)


# ── create_agent routes load_model kwargs / errors clearly ────────────────────

class TestCreateAgentKwargs:
    def test_engine_kwarg_with_loaded_instance_raises(self):
        # `engine` only applies to a model-id string; with a pre-loaded instance
        # it must raise a clear error rather than a cryptic AgentConfig TypeError.
        class _FakeModel:  # stand-in loaded model (not a str)
            model_name = "fake"

        with pytest.raises(TypeError) as exc:
            create_agent("minimal", model=_FakeModel(), engine="transformers")
        assert "engine" in str(exc.value)

    def test_unknown_kwarg_lists_accepted_options(self):
        with pytest.raises(TypeError) as exc:
            create_agent("minimal", "x", not_a_real_field=1, require_model=False)
        msg = str(exc.value)
        assert "not_a_real_field" in msg
        # the actionable message points at both load-model options and config fields
        assert "engine" in msg
        assert "temperature" in msg or "max_iterations" in msg

    def test_real_config_field_passthrough_still_works(self):
        agent = create_agent("minimal", "x", require_model=False, max_context_length=2048)
        assert agent.config.max_context_length == 2048
