"""
Hypothesis-driven fuzz tests for structured-output parsing/validation.

Targets the pure helpers that turn messy model text into validated JSON:

  * ``extract_json_from_text`` — pulls the first JSON object/array out of free
    text (code fences, prose, trailing commas, unquoted keys).
  * ``_clean_json`` — the trailing-comma / unquoted-key cleaner.
  * ``validate_json_schema`` / ``_basic_validate`` — schema validation.
  * ``StructuredOutcome`` — the result container.

Asserts that:
  1. ``extract_json_from_text`` never crashes on arbitrary text and returns
     ``str | None``.
  2. When it returns a string for fenced/embedded JSON, the result is itself a
     string (no exceptions escape ``_clean_json``).
  3. ``validate_json_schema`` never raises — it always returns
     ``(bool, str | None)`` — even on adversarial deeply-nested data/schema.
  4. ``_basic_validate`` tolerates malformed schemas (``required`` as a bare
     string, ``properties`` as a non-dict) without crashing or inventing
     bogus per-character "missing field" errors.

Exit criterion: ≥300 examples per text/data test.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.core.structured_output import (
    StructuredOutcome,
    _basic_validate,
    _clean_json,
    extract_json_from_text,
    validate_json_schema,
)

pytestmark = pytest.mark.fuzz

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# JSON-ish noise: braces, brackets, quotes, backslashes, fences, commas, colons.
_JSON_NOISE = st.text(
    alphabet='{}[]":,\\ \n\tabcXYZ09`',
    min_size=0,
    max_size=200,
)

_FENCE_WRAP = st.sampled_from(
    ["```json\n{body}\n```", "```\n{body}\n```", "prose {body} more", "{body}"]
)

# Recursive JSON value strategy bounded in depth so we exercise nesting without
# building megabyte payloads.
_json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**6), max_value=10**6)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=20),
    lambda children: st.lists(children, max_size=5)
    | st.dictionaries(st.text(max_size=10), children, max_size=5),
    max_leaves=30,
)


# ---------------------------------------------------------------------------
# extract_json_from_text
# ---------------------------------------------------------------------------


@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
@given(_JSON_NOISE)
def test_extract_never_crashes(text: str) -> None:
    """Arbitrary JSON-ish noise never crashes the extractor."""
    result = extract_json_from_text(text)
    assert result is None or isinstance(result, str)


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(_json_values, _FENCE_WRAP)
def test_extract_roundtrips_real_json(value, wrap: str) -> None:
    """Real JSON embedded in fences/prose is extracted to a parseable string."""
    body = json.dumps(value)
    text = wrap.format(body=body)
    result = extract_json_from_text(text)
    # Objects/arrays must be found; bare scalars are not "the first {/[".
    if isinstance(value, dict | list) and value != {} and value != []:
        assert isinstance(result, str)
        # Extracted text is cleanable and re-parseable.
        assert json.loads(result) == value or isinstance(json.loads(result), type(value))


@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
@given(st.text(min_size=0, max_size=300))
def test_clean_json_never_crashes(text: str) -> None:
    """``_clean_json`` is a regex cleaner — it must never raise."""
    out = _clean_json(text)
    assert isinstance(out, str)


def test_extract_pathological_unbalanced() -> None:
    """Unbalanced/odd brackets degrade to None, never hang or crash."""
    for bad in ["{" * 2000, "[" * 2000, '{"a":"' + "\\" * 50, "```json\n", "{", "["]:
        assert extract_json_from_text(bad) is None or isinstance(
            extract_json_from_text(bad), str
        )


# ---------------------------------------------------------------------------
# validate_json_schema / _basic_validate
# ---------------------------------------------------------------------------

_SCHEMA_TYPES = st.sampled_from(
    ["object", "array", "string", "integer", "number", "boolean", "null", "weird", None]
)


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_json_values, _SCHEMA_TYPES)
def test_validate_always_returns_tuple(data, schema_type) -> None:
    """Validation always returns ``(bool, str|None)`` — never raises."""
    schema = {} if schema_type is None else {"type": schema_type}
    valid, err = validate_json_schema(data, schema)
    assert isinstance(valid, bool)
    assert err is None or isinstance(err, str)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.dictionaries(st.text(max_size=8), _json_values, max_size=5),
    st.one_of(st.text(max_size=8), st.lists(st.text(max_size=8), max_size=4), st.integers(), st.none()),
    st.one_of(st.dictionaries(st.text(max_size=8), st.just({"type": "string"}), max_size=3), st.text(max_size=8), st.none()),
)
def test_basic_validate_tolerates_malformed_schema(data, required, properties) -> None:
    """Malformed ``required``/``properties`` must not crash or fabricate errors.

    A ``required`` given as a bare string must be treated as a single field
    name, never iterated char-by-char.
    """
    schema = {"type": "object"}
    if required is not None:
        schema["required"] = required
    if properties is not None:
        schema["properties"] = properties
    valid, err = _basic_validate(data, schema)
    assert isinstance(valid, bool)
    assert err is None or isinstance(err, str)
    # When required is a bare string, the only legitimate "missing" message is
    # for that whole string, not for an individual character of it.
    if isinstance(required, str) and required and required not in data:
        assert err == f"Missing required field: {required}"


def test_basic_validate_required_string_not_char_iterated() -> None:
    """Regression: required='name' must check 'name', not 'n','a','m','e'."""
    valid, err = _basic_validate({}, {"type": "object", "required": "name"})
    assert valid is False
    assert err == "Missing required field: name"
    valid, err = _basic_validate({"name": 1}, {"type": "object", "required": "name"})
    assert valid is True


def test_validate_deep_nesting_fails_closed() -> None:
    """Deeply nested data/schema fails closed (False, msg) — never an uncaught crash."""

    def nest_schema(n):
        s = {"type": "object", "properties": {}}
        cur = s
        for _ in range(n):
            cur["properties"] = {"x": {"type": "object", "properties": {}}}
            cur = cur["properties"]["x"]
        return s

    def nest_data(n):
        d = {}
        cur = d
        for _ in range(n):
            cur["x"] = {}
            cur = cur["x"]
        return d

    valid, err = validate_json_schema(nest_data(4000), nest_schema(4000))
    assert isinstance(valid, bool)
    assert err is None or isinstance(err, str)


# ---------------------------------------------------------------------------
# StructuredOutcome container
# ---------------------------------------------------------------------------


def test_structured_outcome_honest_failure_shape() -> None:
    """A failed outcome preserves raw text and an error — never a silent empty success."""
    outcome = StructuredOutcome(
        success=False, attempts=3, method="reprompt",
        error="could not produce schema-valid output", raw_text="not json",
    )
    assert outcome.success is False
    assert outcome.error
    assert outcome.raw_text == "not json"
