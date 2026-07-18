"""Tests for the normalized, drift-aware model catalog (``effgen.models._catalog``).

These are offline tests: they exercise the record normalization, the snapshot
store round-trip, the diff/drift machinery, and the aggregator (lookup,
nearest_alternatives, staleness) without any network I/O.
"""

from __future__ import annotations

import datetime

import pytest

from effgen.models import _catalog as C
from effgen.models._catalog import ModelRecord

# ---------------------------------------------------------------------------
# ModelRecord
# ---------------------------------------------------------------------------


def test_record_round_trip_dict():
    rec = ModelRecord(
        id="gpt-5-nano",
        provider="openai",
        context_window=1_047_576,
        max_output=32_768,
        price_in_per_1m=0.05,
        price_out_per_1m=0.40,
        supports_tools=True,
    )
    again = ModelRecord.from_dict(rec.to_dict())
    assert again == rec
    # display_name defaults to id
    assert again.display_name == "gpt-5-nano"


def test_from_dict_ignores_unknown_keys():
    rec = ModelRecord.from_dict(
        {"id": "x", "provider": "p", "totally_unknown": 1, "context_window": 5}
    )
    assert rec.id == "x" and rec.context_window == 5


def test_is_priced_flag():
    assert ModelRecord("a", "p", price_in_per_1m=1.0).is_priced
    assert ModelRecord("a", "p", free_tier=True).is_priced
    assert ModelRecord("a", "p", price_note="$/sec").is_priced
    assert not ModelRecord("a", "p").is_priced


# ---------------------------------------------------------------------------
# Normalization across heterogeneous provider schemas
# ---------------------------------------------------------------------------


def test_normalize_openai_pricing_keys():
    r = C.normalize_record(
        "openai",
        "gpt-5-nano",
        {
            "context": 1000,
            "max_output": 100,
            "input_price_per_1m": 0.05,
            "output_price_per_1m": 0.40,
            "supports_native_tools": True,
            "family": "chat",
        },
    )
    assert r.price_in_per_1m == 0.05
    assert r.price_out_per_1m == 0.40
    assert r.context_window == 1000
    assert r.supports_tools is True


def test_normalize_groq_pricing_keys():
    r = C.normalize_record(
        "groq",
        "llama-3.1-8b-instant",
        {"context": 131072, "pricing_per_1m_input": 0.05, "pricing_per_1m_output": 0.08},
    )
    assert r.price_in_per_1m == 0.05 and r.price_out_per_1m == 0.08


def test_normalize_cerebras_free_no_price():
    r = C.normalize_record(
        "cerebras",
        "zai-glm-4.7",
        {"context": 65536, "free_tier": True, "supports_native_tools": True},
    )
    assert r.free_tier is True
    assert r.price_in_per_1m is None
    assert r.is_priced  # free counts as priced


def test_normalize_replicate_per_second_note():
    r = C.normalize_record(
        "replicate",
        "meta/meta-llama-3-8b-instruct",
        {"context": 8192, "cost_per_second_usd": 0.000225},
    )
    assert r.price_in_per_1m is None
    assert "sec" in r.price_note
    assert r.is_priced


def test_normalize_anthropic_pulls_side_table_pricing():
    # Real model id present in ANTHROPIC_MODEL_COSTS.
    from effgen.models.anthropic_models import ANTHROPIC_MODELS

    mid = next(iter(ANTHROPIC_MODELS))
    r = C.normalize_record("anthropic", mid, ANTHROPIC_MODELS[mid])
    assert r.price_in_per_1m is not None
    assert r.price_out_per_1m is not None


def test_normalize_deprecation_from_active_flag():
    r = C.normalize_record("groq", "old", {"context": 1, "active": False})
    assert r.deprecated is True
    r2 = C.normalize_record("groq", "new", {"context": 1, "active": True})
    assert r2.deprecated is False


# ---------------------------------------------------------------------------
# build_records over the real in-package catalogs
# ---------------------------------------------------------------------------


def test_all_providers_build_non_empty():
    for prov in C.known_providers():
        recs = C.build_records(prov)
        assert recs, f"{prov} produced no records"
        assert all(r.provider == prov for r in recs)
        assert all(r.id for r in recs)


def test_build_records_unknown_provider():
    with pytest.raises(KeyError):
        C.build_records("not-a-provider")


# ---------------------------------------------------------------------------
# Snapshot store round-trip
# ---------------------------------------------------------------------------


def test_snapshot_round_trip(tmp_path):
    recs = C.build_records("openai")
    p = tmp_path / "openai.json"
    C.save_snapshot("openai", recs, verified_on="2026-06-08", path=p)
    loaded = C.load_snapshot_records("openai", path=p)
    assert [r.to_dict() for r in recs] == [r.to_dict() for r in loaded]
    meta = C.load_snapshot("openai", path=p)
    assert meta["count"] == len(recs)
    assert meta["verified_on"] == "2026-06-08"


def test_snapshot_age_days(tmp_path):
    p = tmp_path / "openai.json"
    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
    C.save_snapshot("openai", C.build_records("openai"), verified_on=week_ago, path=p)
    assert C.snapshot_age_days("openai", path=p) == 7


def test_missing_snapshot_returns_none(tmp_path):
    assert C.load_snapshot("openai", path=tmp_path / "nope.json") is None
    assert C.snapshot_age_days("openai", path=tmp_path / "nope.json") is None
    assert C.load_snapshot_records("openai", path=tmp_path / "nope.json") == []


def test_bundled_snapshots_exist_and_match_catalog():
    # The shipped snapshots must agree with the live in-package catalogs.
    for prov in C.known_providers():
        assert C.snapshot_path(prov).exists(), f"missing bundled snapshot for {prov}"
        diff = C.check_drift_against_snapshot(prov)
        assert not diff["added"], f"{prov} bundled snapshot missing: {diff['added'][:3]}"
        assert not diff["removed"], f"{prov} bundled snapshot extra: {diff['removed'][:3]}"
        assert not diff["changed"], f"{prov} bundled snapshot drifted: {list(diff['changed'])[:3]}"


# ---------------------------------------------------------------------------
# Diff / drift
# ---------------------------------------------------------------------------


def test_diff_added_removed_changed():
    old = [
        ModelRecord("a", "p", context_window=100, price_in_per_1m=1.0),
        ModelRecord("b", "p", context_window=200),
    ]
    new = [
        ModelRecord("a", "p", context_window=128, price_in_per_1m=2.0),  # changed
        ModelRecord("c", "p", context_window=300),  # added
    ]
    d = C.diff_records(old, new)
    assert d["added"] == ["c"]
    assert d["removed"] == ["b"]
    assert "a" in d["changed"]
    assert d["changed"]["a"]["context_window"] == [100, 128]
    assert d["changed"]["a"]["price_in_per_1m"] == [1.0, 2.0]
    assert d["price_changed"] == ["a"]


def test_diff_identical_is_empty():
    recs = C.build_records("gemini")
    d = C.diff_records(recs, recs)
    assert not d["added"] and not d["removed"] and not d["changed"]


# ---------------------------------------------------------------------------
# Aggregator: list_models / lookup
# ---------------------------------------------------------------------------


def test_list_models_all_spans_providers():
    all_recs = C.list_models()
    provs = {r.provider for r in all_recs}
    assert provs == set(C.known_providers())
    # at least as many as the largest single provider
    assert len(all_recs) > len(C.list_models("openai"))


def test_lookup_bare_and_prefixed():
    r = C.lookup("gpt-5-nano")
    assert r is not None and r.provider == "openai"
    r2 = C.lookup("cerebras:gpt-oss-120b")
    assert r2 is not None and r2.provider == "cerebras"
    # explicit provider arg
    r3 = C.lookup("gpt-oss-120b", provider="cerebras")
    assert r3 is not None and r3.provider == "cerebras"


def test_lookup_unknown_returns_none():
    assert C.lookup("definitely-not-a-real-model-xyz") is None


def test_variants_lists_every_provider_of_a_shared_id():
    provs = C.providers_for("Qwen/Qwen2.5-7B-Instruct")
    if len(provs) < 2:
        import pytest
        pytest.skip("catalog no longer serves this id on multiple providers")
    recs = C.variants("Qwen/Qwen2.5-7B-Instruct")
    assert {r.provider for r in recs} == set(provs)
    assert all(r.id == "Qwen/Qwen2.5-7B-Instruct" for r in recs)


def test_variants_single_provider_returns_one():
    recs = C.variants("gpt-5-nano")
    assert [r.provider for r in recs] == ["openai"]


def test_variants_honors_provider_prefix():
    recs = C.variants("openai:gpt-5-nano")
    assert [r.provider for r in recs] == ["openai"]


# ---------------------------------------------------------------------------
# nearest_alternatives
# ---------------------------------------------------------------------------


def test_nearest_suggests_same_family_first():
    alts = C.nearest_alternatives("gpt-5.5-nano", provider="openai", n=3)
    assert alts, "expected suggestions"
    # The closest suggestions should be the other nano chat models.
    assert any("nano" in r.id for r in alts[:2])
    # Never suggest the (missing) query itself.
    assert all(r.id != "gpt-5.5-nano" for r in alts)


def test_nearest_pushes_deprecated_to_back():
    alts = C.nearest_alternatives("llama3.1-8b", provider="cerebras", n=3)
    assert alts
    # The first suggestion must be a live (non-deprecated) model.
    assert alts[0].deprecated is False


def test_nearest_global_scope():
    alts = C.nearest_alternatives("Qwen/Qwen2.5-7B-Instruct", n=4)
    assert alts
    # Should surface Qwen-family models from some provider.
    assert any("qwen" in r.id.lower() for r in alts)


def test_nearest_empty_for_unknown_scope():
    with pytest.raises(KeyError):
        C.nearest_alternatives("anything", provider="bogus")


# ---------------------------------------------------------------------------
# Staleness + once-per-process warning
# ---------------------------------------------------------------------------


def test_stale_providers_flags_old_and_missing(monkeypatch):
    # Force every provider to look ancient.
    monkeypatch.setattr(C, "snapshot_age_days", lambda prov, **kw: 9999)
    assert set(C.stale_providers(max_age_days=120)) == set(C.known_providers())
    # Missing snapshot (None) also counts as stale.
    monkeypatch.setattr(C, "snapshot_age_days", lambda prov, **kw: None)
    assert "openai" in C.stale_providers()


def test_warn_if_stale_once_per_process(monkeypatch):
    C.reset_stale_warnings()
    monkeypatch.setattr(C, "snapshot_age_days", lambda prov, **kw: 9999)
    seen = []
    assert C.warn_if_stale("openai", warn_fn=seen.append) is True
    assert C.warn_if_stale("openai", warn_fn=seen.append) is False  # silenced
    assert len(seen) == 1
    assert "refresh" in seen[0]


def test_warn_if_stale_quiet_when_fresh(monkeypatch):
    C.reset_stale_warnings()
    monkeypatch.setattr(C, "snapshot_age_days", lambda prov, **kw: 1)
    seen = []
    assert C.warn_if_stale("openai", warn_fn=seen.append) is False
    assert seen == []


def test_warn_if_stale_warns_when_missing(monkeypatch):
    C.reset_stale_warnings()
    monkeypatch.setattr(C, "snapshot_age_days", lambda prov, **kw: None)
    seen = []
    assert C.warn_if_stale("groq", warn_fn=seen.append) is True
    assert "groq" in seen[0]
