"""The ``effgen batch`` CLI resolves structured-output flags and reads an
existing JSONL output for --resume.

``--schema`` loads a JSON Schema file, ``--output-model`` imports a Pydantic
model, and a bad path/spec yields a one-line actionable error. ``--resume``
reads the indices already present in the output file so a rerun skips them.
"""

from __future__ import annotations

import argparse
import json

import pytest

from effgen.cli._main import _batch_structured_kwargs, _read_done_indices


def _args(**kw):
    ns = argparse.Namespace(schema_path=None, output_model=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_schema_flag_loads_json_schema(tmp_path):
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(schema))
    kwargs = _batch_structured_kwargs(_args(schema_path=str(p)))
    assert kwargs == {"output_schema": schema}


def test_schema_flag_missing_file_errors():
    with pytest.raises(ValueError, match="Schema file not found"):
        _batch_structured_kwargs(_args(schema_path="/no/such/schema.json"))


def test_output_model_imports_pydantic_class(tmp_path, monkeypatch):
    mod = tmp_path / "mymodels.py"
    mod.write_text(
        "from pydantic import BaseModel\n"
        "class Ticket(BaseModel):\n"
        "    product: str\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    kwargs = _batch_structured_kwargs(_args(output_model="mymodels:Ticket"))
    assert "output_model" in kwargs
    assert kwargs["output_model"].__name__ == "Ticket"


def test_output_model_bad_spec_errors():
    with pytest.raises(ValueError, match="module:ClassName"):
        _batch_structured_kwargs(_args(output_model="NoColonHere"))


def test_schema_and_output_model_are_mutually_exclusive():
    with pytest.raises(ValueError, match="only one of"):
        _batch_structured_kwargs(_args(schema_path="a.json", output_model="m:C"))


def test_read_done_indices(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps({"index": 0, "success": True, "output": "a"}) + "\n"
        + json.dumps({"index": 2, "success": False, "output": ""}) + "\n"
        + "not-json-should-be-ignored\n"
    )
    done = _read_done_indices(out)
    assert set(done) == {0, 2}
    assert done[0]["output"] == "a"


def test_read_done_indices_missing_file(tmp_path):
    assert _read_done_indices(tmp_path / "nope.jsonl") == {}
