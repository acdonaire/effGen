"""Normalized, refreshable, drift-aware model catalog for effGen.

Historically each provider shipped its own model dict with its own field names
(``OPENAI_MODELS`` uses ``input_price_per_1m``; ``GROQ_MODELS`` uses
``pricing_per_1m_input``; Cerebras has no per-token price at all; Replicate
prices per *second*; HF nests a provider list).  Callers that wanted a uniform
view — the cost tracker, the router, the CLI ``models`` command — had to special
-case every provider.  There was also no single place that knew *when* a catalog
was last verified, so a provider could silently 404 a "known" model for weeks.

This module gives the rest of effGen one shape to depend on:

* :class:`ModelRecord` — a normalized per-model record with the same field names
  for every provider (id, provider, context window, max output, input/output
  price per 1M tokens, tool/vision/audio/video support, free-tier flag, rate
  limits, family, deprecation, a ``price_source`` provenance tag and a
  ``verified_on`` date).
* A small **snapshot store** under ``models/_data/<provider>.json`` with
  ``{verified_on, count, models:[...]}`` plus :func:`load_snapshot`,
  :func:`save_snapshot`, :func:`snapshot_age_days`.
* An **aggregator** over every provider's in-package catalog:
  :func:`list_models`, :func:`lookup`, :func:`nearest_alternatives`,
  :func:`stale_providers`, and a once-per-process :func:`warn_if_stale`.

The aggregator reads the live in-package ``*_MODELS`` dicts (the same data the
adapters use) so it can never disagree with what actually gets called; the
snapshot store is the persisted layer that the refresh path (and drift
detection) build on.  This module performs **no** network I/O — refreshing
against a live provider endpoint is layered on top per provider.
"""

from __future__ import annotations

import datetime
import difflib
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "_data"

# Default freshness horizon: a bundled snapshot older than this is "stale".
DEFAULT_MAX_AGE_DAYS = 120

# Tag describing where a record's numbers came from, for honesty in the UI.
PRICE_SOURCE_BUNDLED = "bundled-catalog"
PRICE_SOURCE_LIVE = "live-api"
PRICE_SOURCE_TABLE = "maintained-table"


# ---------------------------------------------------------------------------
# The normalized record
# ---------------------------------------------------------------------------


@dataclass
class ModelRecord:
    """One model, normalized to identical field names across every provider.

    Prices are USD per 1,000,000 tokens; ``None`` means "not published".  A
    record is considered *priced* when at least one of the per-token prices is
    set, or when :attr:`free_tier` is True (genuinely free), or when
    :attr:`price_note` explains a non-token pricing model (e.g. Replicate's
    per-second billing).
    """

    id: str
    provider: str
    display_name: str = ""
    family: str = ""
    context_window: int = 0
    max_output: int = 0
    price_in_per_1m: float | None = None
    price_out_per_1m: float | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    free_tier: bool = False
    rpm: int | None = None
    tpm: int | None = None
    rpd: int | None = None
    deprecated: bool = False
    price_source: str = PRICE_SOURCE_BUNDLED
    verified_on: str | None = None
    price_note: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.id

    @property
    def is_priced(self) -> bool:
        """True when this record carries usable cost information."""
        return (
            self.price_in_per_1m is not None
            or self.price_out_per_1m is not None
            or self.free_tier
            or bool(self.price_note)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-friendly dict (stable key order)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRecord":
        """Rebuild a record from :meth:`to_dict` output, ignoring extra keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Provider source table — the in-package catalogs are the source of truth.
# ---------------------------------------------------------------------------
#
# Each entry: provider -> (module_path, models_var, default_var | None)
# Loaded lazily so importing this module stays cheap and avoids import cycles.

_PROVIDER_SOURCES: dict[str, tuple[str, str, str | None]] = {
    "openai": ("effgen.models.openai_models", "OPENAI_MODELS", "OPENAI_DEFAULT_MODEL"),
    "anthropic": ("effgen.models.anthropic_models", "ANTHROPIC_MODELS", None),
    "gemini": ("effgen.models.gemini_models", "GEMINI_MODELS", "GEMINI_DEFAULT_MODEL"),
    "cerebras": ("effgen.models.cerebras_models", "CEREBRAS_MODELS", "CEREBRAS_DEFAULT_MODEL"),
    "groq": ("effgen.models.groq_models", "GROQ_MODELS", "GROQ_DEFAULT_MODEL"),
    "together": ("effgen.models.together_models", "TOGETHER_MODELS", "TOGETHER_DEFAULT_MODEL"),
    "fireworks": ("effgen.models.fireworks_models", "FIREWORKS_MODELS", "FIREWORKS_DEFAULT_MODEL"),
    "replicate": ("effgen.models.replicate_models", "REPLICATE_MODELS", "REPLICATE_DEFAULT_MODEL"),
    "hf": ("effgen.models.hf_inference_models", "HF_MODELS", "HF_DEFAULT_MODEL"),
}


def known_providers() -> list[str]:
    """Return the providers the catalog aggregator knows how to read."""
    return list(_PROVIDER_SOURCES)


def _load_models_dict(provider: str) -> dict[str, dict[str, Any]]:
    import importlib

    mod_path, var, _ = _PROVIDER_SOURCES[provider]
    module = importlib.import_module(mod_path)
    return dict(getattr(module, var, {}))


def default_model(provider: str) -> str | None:
    """Return the configured default model id for *provider*, if any."""
    import importlib

    src = _PROVIDER_SOURCES.get(provider)
    if not src:
        return None
    mod_path, _, default_var = src
    if not default_var:
        return None
    module = importlib.import_module(mod_path)
    return getattr(module, default_var, None)


# ---------------------------------------------------------------------------
# Normalization — map each provider's native dict onto a ModelRecord.
# ---------------------------------------------------------------------------


def _first(raw: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in raw and raw[k] is not None:
            return raw[k]
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_price(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_record(
    provider: str,
    model_id: str,
    raw: dict[str, Any],
    *,
    verified_on: str | None = None,
    price_source: str = PRICE_SOURCE_BUNDLED,
) -> ModelRecord:
    """Map one provider-native model dict onto a :class:`ModelRecord`.

    Handles the field-name divergence between providers (``input_price_per_1m``
    vs ``pricing_per_1m_input``), the providers that price by the second
    (Replicate) or not at all (Cerebras free tier), and the per-provider
    vision/audio/video flags.
    """
    raw = raw or {}

    price_in = _as_price(_first(raw, "input_price_per_1m", "pricing_per_1m_input"))
    price_out = _as_price(_first(raw, "output_price_per_1m", "pricing_per_1m_output"))
    price_note = ""

    # Anthropic keeps pricing in a side table rather than the model dict.
    if provider == "anthropic" and price_in is None and price_out is None:
        try:
            from effgen.models.anthropic_models import get_cost_per_million

            pin, pout = get_cost_per_million(model_id)
            price_in, price_out = float(pin), float(pout)
        except Exception:  # pragma: no cover - defensive only
            pass

    # Replicate bills per GPU-second, not per token.
    cps = raw.get("cost_per_second_usd")
    if cps is not None and price_in is None and price_out is None:
        price_note = f"${cps}/sec (GPU-time billing, not per-token)"

    free_tier = bool(raw.get("free_tier", False))

    # deprecated: explicit flag wins; an explicit active=False also counts.
    deprecated = bool(raw.get("deprecated"))
    if not deprecated and raw.get("active") is False:
        deprecated = True
    if not deprecated and raw.get("serverless") is False and provider == "together":
        # serverless=False is *not* deprecation; leave as-is. (kept for clarity)
        pass

    # Vision: explicit flag, else an OpenAI heuristic for that provider.
    supports_vision = bool(raw.get("supports_vision", False))
    if not supports_vision and provider == "openai":
        try:
            from effgen.models.openai_models import supports_vision as _ov

            supports_vision = bool(_ov(model_id))
        except Exception:  # pragma: no cover
            supports_vision = False

    notes = str(raw.get("notes") or "")

    return ModelRecord(
        id=model_id,
        provider=provider,
        display_name=str(_first(raw, "display_name") or model_id),
        family=str(raw.get("family") or ""),
        context_window=_as_int(_first(raw, "context", "context_window")) or 0,
        max_output=_as_int(raw.get("max_output")) or 0,
        price_in_per_1m=price_in,
        price_out_per_1m=price_out,
        supports_tools=bool(raw.get("supports_native_tools", False)),
        supports_vision=supports_vision,
        supports_audio=bool(raw.get("supports_audio", False)),
        supports_video=bool(raw.get("supports_video", False)),
        free_tier=free_tier,
        rpm=_as_int(raw.get("rpm")),
        tpm=_as_int(raw.get("tpm")),
        rpd=_as_int(raw.get("rpd")),
        deprecated=deprecated,
        price_source=price_source,
        verified_on=verified_on,
        price_note=price_note,
        notes=notes,
    )


def build_records(provider: str, *, verified_on: str | None = None) -> list[ModelRecord]:
    """Build normalized records for *provider* from its in-package catalog."""
    if provider not in _PROVIDER_SOURCES:
        raise KeyError(
            f"Unknown provider {provider!r}. Known: {known_providers()}"
        )
    models = _load_models_dict(provider)
    return [
        normalize_record(provider, mid, raw, verified_on=verified_on)
        for mid, raw in models.items()
    ]


# ---------------------------------------------------------------------------
# Snapshot store: models/_data/<provider>.json
# ---------------------------------------------------------------------------


def snapshot_path(provider: str) -> Path:
    """Return the bundled snapshot path for *provider*."""
    return _DATA_DIR / f"{provider}.json"


def save_snapshot(
    provider: str,
    records: Iterable[ModelRecord],
    *,
    verified_on: str | None = None,
    source: str = PRICE_SOURCE_BUNDLED,
    path: Path | None = None,
) -> Path:
    """Persist *records* for *provider* as ``{verified_on, count, models:[...]}``.

    Returns the path written.  ``verified_on`` defaults to today (UTC).
    """
    recs = list(records)
    when = verified_on or datetime.date.today().isoformat()
    payload = {
        "provider": provider,
        "verified_on": when,
        "source": source,
        "count": len(recs),
        "models": [r.to_dict() for r in recs],
    }
    out = path or snapshot_path(provider)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    return out


def load_snapshot(provider: str, *, path: Path | None = None) -> dict[str, Any] | None:
    """Load the persisted snapshot for *provider*, or None if absent/invalid."""
    src = path or snapshot_path(provider)
    if not src.exists():
        return None
    try:
        with src.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read model snapshot %s: %s", src, exc)
        return None
    return data


def load_snapshot_records(provider: str, *, path: Path | None = None) -> list[ModelRecord]:
    """Load a snapshot and return it as :class:`ModelRecord` objects."""
    data = load_snapshot(provider, path=path)
    if not data:
        return []
    return [ModelRecord.from_dict(m) for m in data.get("models", [])]


def snapshot_age_days(provider: str, *, path: Path | None = None) -> int | None:
    """Return how many days old *provider*'s snapshot is, or None if unknown."""
    data = load_snapshot(provider, path=path)
    if not data:
        return None
    when = data.get("verified_on")
    if not when:
        return None
    try:
        d = datetime.date.fromisoformat(str(when)[:10])
    except ValueError:
        return None
    return (datetime.date.today() - d).days


def snapshot_meta(provider: str) -> dict[str, Any]:
    """Return ``{verified_on, count, source, age_days}`` for *provider*."""
    data = load_snapshot(provider) or {}
    return {
        "provider": provider,
        "verified_on": data.get("verified_on"),
        "count": data.get("count"),
        "source": data.get("source"),
        "age_days": snapshot_age_days(provider),
    }


# ---------------------------------------------------------------------------
# Diffing — the core of drift detection (no network I/O here).
# ---------------------------------------------------------------------------

# Fields whose change is worth flagging as drift (ignore provenance/free-text).
_DRIFT_FIELDS = (
    "context_window",
    "max_output",
    "price_in_per_1m",
    "price_out_per_1m",
    "supports_tools",
    "supports_vision",
    "supports_audio",
    "supports_video",
    "free_tier",
    "deprecated",
)


def diff_records(
    old: Iterable[ModelRecord],
    new: Iterable[ModelRecord],
) -> dict[str, Any]:
    """Compare two record sets and report added/removed/changed models.

    ``changed`` maps a model id to ``{field: [old, new]}`` for every drift-worthy
    field that differs.  ``price_changed`` is the subset whose pricing moved.
    """
    old_map = {r.id: r for r in old}
    new_map = {r.id: r for r in new}
    old_ids = set(old_map)
    new_ids = set(new_map)

    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)

    changed: dict[str, dict[str, list[Any]]] = {}
    price_changed: list[str] = []
    for mid in sorted(old_ids & new_ids):
        o, n = old_map[mid], new_map[mid]
        deltas: dict[str, list[Any]] = {}
        for f in _DRIFT_FIELDS:
            ov, nv = getattr(o, f), getattr(n, f)
            if ov != nv:
                deltas[f] = [ov, nv]
        if deltas:
            changed[mid] = deltas
            if "price_in_per_1m" in deltas or "price_out_per_1m" in deltas:
                price_changed.append(mid)

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "price_changed": price_changed,
        "old_count": len(old_ids),
        "new_count": len(new_ids),
    }


def check_drift_against_snapshot(provider: str) -> dict[str, Any]:
    """Diff the live in-package catalog for *provider* against its snapshot.

    Useful as an offline self-check that the bundled snapshot matches the
    catalog the adapters actually use.  Returns an empty diff if no snapshot
    exists yet.
    """
    live = build_records(provider)
    snap = load_snapshot_records(provider)
    return diff_records(snap, live)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def list_models(provider: str | None = None) -> list[ModelRecord]:
    """Return normalized records for one provider, or all providers if None.

    Reads the in-package catalogs (the source of truth the adapters use), so
    the result always matches what would actually be called.
    """
    if provider is not None:
        return build_records(provider)
    out: list[ModelRecord] = []
    for prov in known_providers():
        try:
            out.extend(build_records(prov))
        except Exception as exc:  # pragma: no cover - one bad catalog shouldn't kill the list
            logger.debug("Could not build catalog for %s: %s", prov, exc, exc_info=True)
    return out


def _split_prefix(model_id: str) -> tuple[str | None, str]:
    """Split ``provider:model`` into (provider, model). Provider must be known."""
    if ":" in model_id:
        head, tail = model_id.split(":", 1)
        if head in _PROVIDER_SOURCES:
            return head, tail
    return None, model_id


def lookup(model_id: str, provider: str | None = None) -> ModelRecord | None:
    """Resolve *model_id* to a single :class:`ModelRecord`, or None.

    Accepts an optional ``provider:`` prefix on *model_id* or an explicit
    *provider*.  When a bare id is exposed by exactly one provider it resolves
    unambiguously; when several providers expose it and none is specified, the
    first match (in :func:`known_providers` order) is returned.
    """
    pref_provider, bare = _split_prefix(model_id)
    provider = provider or pref_provider

    providers = [provider] if provider else known_providers()
    for prov in providers:
        if prov not in _PROVIDER_SOURCES:
            continue
        models = _load_models_dict(prov)
        if bare in models:
            return normalize_record(prov, bare, models[bare])
        # also accept the fully-qualified id stored as-is
        if model_id in models:
            return normalize_record(prov, model_id, models[model_id])
    return None


def nearest_alternatives(
    model_id: str,
    provider: str | None = None,
    n: int = 5,
) -> list[ModelRecord]:
    """Suggest up to *n* live models closest to *model_id*.

    Generalizes the per-provider "did you mean" helpers: prefers models in the
    same family, then the same provider, then fuzzy id similarity; deprecated
    models are pushed to the back.  When *provider* is given the search is
    scoped to that provider, otherwise it spans every known catalog.
    """
    pref_provider, bare = _split_prefix(model_id)
    provider = provider or pref_provider

    candidates = list_models(provider)
    # Drop the exact model if present; never suggest the thing they asked for.
    candidates = [r for r in candidates if r.id != bare and r.id != model_id]
    if not candidates:
        return []

    # Infer the target family from the requested id (even if it isn't known).
    target_family = _guess_family(bare)

    close = set(difflib.get_close_matches(bare, [r.id for r in candidates], n=n * 4, cutoff=0.3))
    # also match on the trailing path segment (org/model -> model)
    bare_tail = bare.rsplit("/", 1)[-1].lower()

    def score(r: ModelRecord) -> tuple:
        same_family = bool(target_family and r.family and target_family == r.family.lower())
        tail = r.id.rsplit("/", 1)[-1].lower()
        ratio = difflib.SequenceMatcher(None, bare_tail, tail).ratio()
        in_close = r.id in close
        return (
            0 if not r.deprecated else 1,       # live before deprecated
            0 if same_family else 1,            # same family first
            0 if in_close else 1,               # textual near-matches
            -ratio,                             # higher similarity first
        )

    candidates.sort(key=score)
    return candidates[:n]


def _guess_family(model_id: str) -> str:
    """Best-effort family guess for an arbitrary id (mirrors hf's heuristic)."""
    lower = model_id.lower()
    for tag in (
        "gpt-5.4", "gpt-5", "gpt-4.1", "gpt-4o", "gpt-4", "gpt-3.5", "gpt-oss",
        "o4-mini", "o3", "o1",
        "claude-opus", "claude-sonnet", "claude-haiku", "opus", "sonnet", "haiku",
        "gemini", "flash-lite", "flash", "pro",
        "qwen3", "qwen2.5", "qwen2", "qwen",
        "llama-3.3", "llama-3.1", "llama-3", "llama",
        "mixtral", "mistral", "gemma-3", "gemma-2", "gemma",
        "deepseek-v3", "deepseek-r1", "deepseek", "phi-4", "phi-3", "phi",
        "kimi", "command", "zephyr", "minimax", "glm", "zai-glm",
    ):
        if tag in lower:
            return tag
    return ""


# ---------------------------------------------------------------------------
# Staleness + once-per-process warning
# ---------------------------------------------------------------------------


def stale_providers(max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> list[str]:
    """Return providers whose snapshot is older than *max_age_days* (or missing)."""
    out: list[str] = []
    for prov in known_providers():
        age = snapshot_age_days(prov)
        if age is None or age > max_age_days:
            out.append(prov)
    return out


# Guard so a stale warning fires at most once per provider per process.
_WARNED: set[str] = set()


def warn_if_stale(
    provider: str,
    *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    warn_fn: Callable[[str], None] | None = None,
) -> bool:
    """Warn once per process if *provider*'s snapshot is stale or missing.

    Returns True if a warning was emitted on this call.  Non-spammy: subsequent
    calls for the same provider are silent for the life of the process.  Pass
    *warn_fn* to route the message somewhere other than the module logger.
    """
    if provider in _WARNED:
        return False
    age = snapshot_age_days(provider)
    if age is not None and age <= max_age_days:
        return False

    _WARNED.add(provider)
    if age is None:
        msg = (
            f"No verified model snapshot for '{provider}'. "
            f"Run `effgen models refresh --provider {provider}` to fetch the live catalog."
        )
    else:
        msg = (
            f"Model catalog for '{provider}' was last verified {age} days ago and may be "
            f"outdated. Run `effgen models refresh --provider {provider}` to update it."
        )
    if warn_fn is not None:
        warn_fn(msg)
    else:
        logger.warning(msg)
    return True


def reset_stale_warnings() -> None:
    """Clear the once-per-process warning guard (test/CLI-refresh helper)."""
    _WARNED.clear()


__all__ = [
    "ModelRecord",
    "PRICE_SOURCE_BUNDLED",
    "PRICE_SOURCE_LIVE",
    "PRICE_SOURCE_TABLE",
    "DEFAULT_MAX_AGE_DAYS",
    "known_providers",
    "default_model",
    "normalize_record",
    "build_records",
    "snapshot_path",
    "save_snapshot",
    "load_snapshot",
    "load_snapshot_records",
    "snapshot_age_days",
    "snapshot_meta",
    "diff_records",
    "check_drift_against_snapshot",
    "list_models",
    "lookup",
    "nearest_alternatives",
    "stale_providers",
    "warn_if_stale",
    "reset_stale_warnings",
]
