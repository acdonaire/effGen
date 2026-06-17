"""Live proof for construction/run input ergonomics (VF5/VF7).

* VF5: ``run(output_schema=PydanticClass)`` produces valid JSON matching the
  model on a real provider (skipped without GROQ_API_KEY).
* VF7: ``create_agent(..., engine="transformers")`` actually loads a local model
  via the transformers engine (gpu marker).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)


def _has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


class Capital(BaseModel):
    country: str
    capital: str


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_groq(), reason="SKIPPED: GROQ_API_KEY not set")
def test_output_schema_pydantic_class_live_groq():
    from effgen import create_agent
    from effgen.models._rate_limit import RateLimitExceeded

    agent = create_agent("minimal", "groq:llama-3.1-8b-instant")
    try:
        result = agent.run("What is the capital of France?", output_schema=Capital)
    except RateLimitExceeded as exc:
        pytest.skip(f"Groq transient rate limit/quota: {exc}")
    finally:
        pass

    assert result.success, f"expected success, got: {result.output!r}"
    parsed = json.loads(result.output)  # must be valid JSON matching the schema
    assert parsed["capital"].lower() == "paris"
    assert result.metadata.get("structured_output_attempts", 0) >= 1
    agent.close()


@pytest.mark.gpu
def test_create_agent_engine_passthrough_loads_local_model():
    from effgen import create_agent
    from effgen.models.transformers_engine import TransformersEngine

    agent = create_agent(
        "minimal",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        engine="transformers",
        temperature=0.0,
    )
    try:
        # engine= must have routed through load_model -> a TransformersEngine.
        assert isinstance(agent.model, TransformersEngine)
        result = agent.run("Reply with exactly: OK", max_tokens=16)
        assert result.success
    finally:
        agent.close()
