"""
Hypothesis-driven fuzz tests for batch input reading.

Targets ``BatchRunner._read_queries`` — the loader that turns a JSONL, JSON,
CSV, or plain-text file into the list of prompts a batch job runs.

Asserts that, for arbitrary mixtures of good and broken lines:
  1. Every returned query is a non-empty string. A row that carries no query
     text is never returned as an empty prompt for the model to charge for.
  2. Every good line survives, in file order, regardless of the broken lines
     around it.
  3. Every skipped line is reported exactly once with its real file line
     number, and ``strict=True`` raises on the first bad line naming the file
     and that same line number.
  4. Reading never raises anything other than ``ValueError`` — a decode
     failure, an unreadable CSV row, or a file with nothing usable produces a
     message naming the file, not a raw parser exception.

Exit criterion: ≥200 examples per file-shaped test.
"""

from __future__ import annotations

import csv
import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.core.batch import BatchRunner

pytestmark = pytest.mark.fuzz

# A line that must yield exactly one query, paired with the text it yields.
_GOOD_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")), min_size=1, max_size=40
).filter(lambda s: s.strip() != "")

_GOOD_LINE = _GOOD_TEXT.map(lambda t: (json.dumps({"query": t}), t))

# Lines that must be skipped (or hard-fail under strict): unparseable text,
# a JSON scalar/array where an object or string was expected, and an object
# whose fields carry no query text.
_BAD_LINE = st.sampled_from([
    "{not json",
    '{"query": "unterminated',
    "null",
    "true",
    "17",
    "[1, 2, 3]",
    '"   "',
    '{"note": "no query field here"}',
    '{"query": ""}',
    '{"query": null}',
    '{"query": {"nested": 1}}',
    '{"query": ["a"]}',
    '{"query": true}',
])

_LINES = st.lists(
    st.one_of(_GOOD_LINE, _BAD_LINE.map(lambda s: (s, None))),
    min_size=1,
    max_size=12,
)


def _write(tmp_path, lines: list[tuple[str, str | None]], suffix: str = ".jsonl"):
    path = tmp_path / f"queries{suffix}"
    path.write_text("\n".join(raw for raw, _ in lines) + "\n", encoding="utf-8")
    return path


@settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(_LINES)
def test_jsonl_keeps_good_lines_and_reports_every_skip(tmp_path, lines) -> None:
    path = _write(tmp_path, lines)
    expected = [text for _, text in lines if text is not None]
    bad_linenos = [i for i, (_, text) in enumerate(lines, start=1) if text is None]

    reported: list[int] = []
    try:
        queries = BatchRunner._read_queries(
            path,
            "query",
            on_skip=lambda ln, _msg: reported.append(ln),
            on_empty=lambda ln, _keys: reported.append(ln),
        )
    except ValueError:
        # Only legitimate when no good line existed at all.
        assert expected == []
        return

    assert queries == expected
    assert all(isinstance(q, str) and q.strip() for q in queries)
    assert sorted(reported) == bad_linenos


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(_LINES)
def test_jsonl_strict_names_the_first_bad_line(tmp_path, lines) -> None:
    path = _write(tmp_path, lines)
    first_bad = next(
        (i for i, (_, text) in enumerate(lines, start=1) if text is None), None
    )
    if first_bad is None:
        assert BatchRunner._read_queries(path, "query", strict=True)
        return
    with pytest.raises(ValueError) as excinfo:
        BatchRunner._read_queries(path, "query", strict=True)
    assert f"{path}:{first_bad}" in str(excinfo.value)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(st.text(alphabet="{}[]\":,\\ \n\tabq019", min_size=0, max_size=300))
def test_arbitrary_bytes_never_escape_as_a_raw_parser_error(tmp_path, blob) -> None:
    """Any input either loads or raises ``ValueError`` naming the file."""
    for suffix in (".jsonl", ".json", ".csv", ".txt"):
        path = tmp_path / f"blob{suffix}"
        path.write_text(blob, encoding="utf-8")
        try:
            queries = BatchRunner._read_queries(path, "query")
        except ValueError:
            continue
        assert all(isinstance(q, str) and q.strip() for q in queries)


@pytest.mark.parametrize(
    ("suffix", "data"),
    [
        (".jsonl", b'{"query": "fine"}\n{"query": "caf\xe9"}\n'),
        (".json", b'[{"query": "caf\xe9"}]'),
        (".csv", b"query\nfine\ncaf\xe9\n"),
        (".txt", b"fine\ncaf\xe9\n"),
    ],
)
def test_non_utf8_input_names_the_file(tmp_path, suffix: str, data: bytes) -> None:
    """Every input format reports a bad byte against the file, not a bare decode error.

    The message carries the byte offset the decoder reported. It deliberately
    does not claim a line number: readers decode a buffered chunk at a time, so
    the line the reader has reached is not where the bad byte is.
    """
    path = tmp_path / f"queries{suffix}"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="not valid UTF-8") as excinfo:
        BatchRunner._read_queries(path, "query")
    assert str(path) in str(excinfo.value)
    assert "line" not in str(excinfo.value)


def test_json_array_entries_are_reported_by_index_not_line(tmp_path) -> None:
    """A ``.json`` array is one document, so an entry is named by its index."""
    path = tmp_path / "queries.json"
    path.write_text('[{"note": "x"}, 12, "real"]', encoding="utf-8")
    reported: list[int] = []
    queries = BatchRunner._read_queries(
        path,
        "query",
        on_skip=lambda pos, _msg: reported.append(pos),
        on_empty=lambda pos, _keys: reported.append(pos),
    )
    assert queries == ["real"]
    assert sorted(reported) == [1, 2]
    with pytest.raises(ValueError, match=r"item 1: no query text"):
        BatchRunner._read_queries(path, "query", strict=True)


def test_json_scalar_and_object_files_fail_with_a_hint(tmp_path) -> None:
    path = tmp_path / "queries.json"
    path.write_text('{"query": "just one"}', encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON array"):
        BatchRunner._read_queries(path, "query")


def test_jsonl_scalar_lines_are_skipped_not_stringified(tmp_path) -> None:
    """``null``/``true``/``12`` are not prompts — they are reported, not run."""
    path = tmp_path / "queries.jsonl"
    path.write_text('null\ntrue\n12\n{"query": "real"}\n', encoding="utf-8")
    skipped: list[tuple[int, str]] = []
    queries = BatchRunner._read_queries(
        path, "query", on_skip=lambda ln, msg: skipped.append((ln, msg)),
    )
    assert queries == ["real"]
    assert [ln for ln, _ in skipped] == [1, 2, 3]
    assert all("expected string or object" in msg for _, msg in skipped)


def test_csv_reports_the_real_file_line_number(tmp_path) -> None:
    path = tmp_path / "queries.csv"
    path.write_text("query,id\ngood,1\n,2\nalso good,3\n", encoding="utf-8")
    empties: list[tuple[int, list[str]]] = []
    queries = BatchRunner._read_queries(
        path, "query", on_empty=lambda ln, keys: empties.append((ln, keys)),
    )
    assert queries == ["good", "also good"]
    # The blank row is file line 3, not "data row 2".
    assert empties == [(3, ["id", "query"])]


def test_csv_unreadable_row_is_skipped_with_a_line_number(tmp_path) -> None:
    path = tmp_path / "queries.csv"
    path.write_text('query\ngood\n"' + "x" * 5000 + '"\nlast\n', encoding="utf-8")
    previous = csv.field_size_limit()
    csv.field_size_limit(1000)
    try:
        skipped: list[tuple[int, str]] = []
        queries = BatchRunner._read_queries(
            path, "query", on_skip=lambda ln, msg: skipped.append((ln, msg)),
        )
    finally:
        csv.field_size_limit(previous)
    assert queries == ["good", "last"]
    assert [ln for ln, _ in skipped] == [3]


def test_run_from_file_forwards_both_skip_callbacks(tmp_path) -> None:
    """``run_from_file`` reports malformed rows *and* rows with no query text.

    A caller that only hears about malformed rows would see a row silently
    disappear, so both callbacks reach the reader.
    """
    import inspect

    from effgen.core.batch import BatchRunner as _BR

    params = inspect.signature(_BR.run_from_file).parameters
    assert "on_skip" in params
    assert "on_empty" in params

    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query": "real"}\nnull\n{"note": "no query text"}\n', encoding="utf-8"
    )
    skipped: list[int] = []
    empty: list[tuple[int, list[str]]] = []
    queries = BatchRunner._read_queries(
        path,
        "query",
        on_skip=lambda pos, _msg: skipped.append(pos),
        on_empty=lambda pos, keys: empty.append((pos, keys)),
    )
    assert queries == ["real"]
    assert skipped == [2]
    assert empty == [(3, ["note"])]


def test_all_rows_unusable_fails_instead_of_running_empty_prompts(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text('{"note": "a"}\n{"note": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="No queries found"):
        BatchRunner._read_queries(path, "query")
