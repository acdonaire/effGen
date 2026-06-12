"""The live public surface must stay a superset of the v0.2.x golden snapshot.

These tests fail loudly if a name, command, tool, endpoint, signature parameter, or
method that v0.2.x users relied on disappears. Adding new names is always fine — the
checks are one-directional (superset), so the snapshot only needs updating when a
removal/rename is *deliberate*.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SNAPSHOT = json.loads(
    (Path(__file__).parent / "public_surface_v0_2.json").read_text()
)


def _missing(expected, actual):
    return sorted(set(expected) - set(actual))


def test_toplevel_all_is_superset():
    import effgen

    missing = _missing(SNAPSHOT["toplevel_all"], effgen.__all__)
    assert not missing, f"effgen.__all__ dropped public names: {missing}"


def test_toplevel_names_importable():
    import effgen

    broken = []
    for name in SNAPSHOT["toplevel_all"]:
        try:
            getattr(effgen, name)
        except Exception as exc:  # pragma: no cover - failure path
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not broken, "Top-level names present in __all__ but not resolvable:\n" + "\n".join(broken)


def test_tools_module_public_superset():
    etools = importlib.import_module("effgen.tools")
    live = [n for n in dir(etools) if not n.startswith("_")]
    missing = _missing(SNAPSHOT["tools_module_public"], live)
    assert not missing, f"effgen.tools dropped public names: {missing}"


def test_providers_superset():
    import effgen

    missing = _missing(SNAPSHOT["providers"], effgen.list_providers())
    assert not missing, f"Provider(s) removed: {missing}"


def test_presets_superset():
    import effgen

    missing = _missing(SNAPSHOT["presets"], effgen.list_presets())
    assert not missing, f"Preset(s) removed: {missing}"


def test_cli_commands_superset():
    from effgen.cli._main import create_parser

    parser = create_parser()
    live = []
    for action in parser._actions:
        if getattr(action, "dest", None) == "command" and getattr(action, "choices", None):
            live = list(action.choices.keys())
    missing = _missing(SNAPSHOT["cli_commands"], live)
    assert not missing, f"CLI command(s) removed: {missing}"


def test_tool_names_superset():
    from effgen.tools import get_registry

    live = [t.name if hasattr(t, "name") else t for t in get_registry().list_tools()]
    missing = _missing(SNAPSHOT["tool_names"], live)
    assert not missing, f"Built-in tool(s) removed/renamed: {missing}"


def test_server_endpoints_superset():
    from effgen.server.app import create_app

    app = create_app()
    live = {r.path for r in app.routes if hasattr(r, "path")}
    missing = _missing(SNAPSHOT["server_endpoints"], live)
    assert not missing, f"Server endpoint(s) removed: {missing}"


def test_documented_signatures_keep_params():
    import effgen

    targets = {
        "Agent.run": effgen.Agent.run,
        "AgentConfig.__init__": effgen.AgentConfig.__init__,
        "ConfigLoader.load_config": effgen.ConfigLoader.load_config,
        "ConfigLoader.load": effgen.ConfigLoader.load,
        "TestSuite.__init__": effgen.TestSuite.__init__,
        "TestCase.__init__": effgen.TestCase.__init__,
    }
    problems = []
    for name, expected_params in SNAPSHOT["signatures"].items():
        obj = targets[name]
        live_params = set(inspect.signature(obj).parameters) - {"self"}
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(obj).parameters.values()
        )
        missing = sorted(set(expected_params) - live_params)
        if missing and not accepts_var_kw:
            problems.append(f"{name} no longer accepts: {missing}")
    assert not problems, "Documented parameters removed:\n" + "\n".join(problems)


def test_required_methods_present():
    import effgen

    classes = {
        "ConfigLoader": effgen.ConfigLoader,
        "ShortTermMemory": effgen.ShortTermMemory,
        "Session": importlib.import_module("effgen.core.session").Session,
        "AgentState": effgen.AgentState,
    }
    problems = []
    for cls_name, methods in SNAPSHOT["required_methods"].items():
        cls = classes[cls_name]
        for m in methods:
            if not hasattr(cls, m):
                problems.append(f"{cls_name}.{m}")
    assert not problems, f"Required methods missing: {problems}"


def test_rag_public_names_present():
    rag = importlib.import_module("effgen.rag")
    missing = [n for n in SNAPSHOT["rag_public"] if not hasattr(rag, n)]
    assert not missing, f"rag public names missing: {missing}"
