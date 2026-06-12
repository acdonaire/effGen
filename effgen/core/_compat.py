"""Backward-compatibility helpers for loading serialized data across versions.

Saved artifacts (sessions, agent state, memory snapshots, configs) may have been
written by an older or newer effGen build whose field set differs slightly from the
running one. Naively splatting such a dict into a constructor raises a cryptic
``TypeError: __init__() got an unexpected keyword argument 'x'`` and makes the file
unloadable. The helpers here load forgivingly instead:

* unknown fields (renamed/removed since the file was written) are dropped, with a
  single, quiet :class:`DeprecationWarning` that points at the migration path;
* missing optional fields fall back to their defaults;
* a genuinely missing *required* field raises a clear, named error.

This keeps v0.2.x files loadable on newer builds (and vice-versa) without silently
losing data.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

# Remember which (target, unknown-keys) combinations we have already warned about so a
# loop loading many records emits at most one warning per distinct drift, not one per row.
_warned_drift: set[tuple[str, tuple[str, ...]]] = set()


def _accepted_param_names(target: Callable[..., Any]) -> tuple[set[str], bool]:
    """Return (accepted keyword names, accepts **kwargs) for ``target``'s signature."""
    sig = inspect.signature(target)
    accepts_var_kw = False
    names: set[str] = set()
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_kw = True
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        names.add(name)
    return names, accepts_var_kw


def coerce_kwargs(
    target: Callable[..., Any],
    data: dict[str, Any],
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Filter ``data`` to the keyword arguments ``target`` actually accepts.

    Unrecognized keys are dropped and reported once via :class:`DeprecationWarning`.
    If ``target`` accepts ``**kwargs`` the dict is returned unchanged.
    """
    label = label or getattr(target, "__name__", str(target))
    accepted, accepts_var_kw = _accepted_param_names(target)
    if accepts_var_kw:
        return dict(data)

    known = {k: v for k, v in data.items() if k in accepted}
    unknown = sorted(k for k in data if k not in accepted)
    if unknown:
        drift_key = (label, tuple(unknown))
        if drift_key not in _warned_drift:
            _warned_drift.add(drift_key)
            warnings.warn(
                f"{label}: ignoring unrecognized field(s) {unknown} while loading saved "
                f"data written by a different effGen version. They were dropped; re-save "
                f"the file to migrate it to the current format.",
                DeprecationWarning,
                stacklevel=3,
            )
    return known


def load_from_dict(
    target: Callable[..., T],
    data: dict[str, Any],
    *,
    label: str | None = None,
) -> T:
    """Construct ``target`` from a possibly-drifted dict, forgiving extra keys.

    Drops unknown keys (one warning), lets optional fields default, and turns a
    missing *required* field into a clear ``ValueError`` naming the field rather than
    a bare ``TypeError``.
    """
    label = label or getattr(target, "__name__", str(target))
    kwargs = coerce_kwargs(target, data, label=label)
    try:
        return target(**kwargs)
    except TypeError as exc:  # missing required positional/keyword argument
        msg = str(exc)
        if "required" in msg or "missing" in msg or "argument" in msg:
            raise ValueError(
                f"{label}: cannot load saved data — it is missing a required field "
                f"({msg}). The file may be from an incompatible version or corrupt."
            ) from exc
        raise
