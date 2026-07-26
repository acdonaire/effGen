"""Every module and public definition in ``effgen`` carries a docstring.

Walks the package source with ``ast`` and asserts that every module, and every
public class/function/method (name not starting with ``_``), has a docstring
whose first line stands alone as a sentence. Conditionally defined objects
(inside ``if``/``try``/``with`` blocks, e.g. optional-dependency fallbacks) are
included; definitions nested inside functions are local helpers and are
skipped.
"""

from __future__ import annotations

import ast
from pathlib import Path

import effgen

PACKAGE_ROOT = Path(effgen.__file__).parent

# A summary line reads as a sentence when it ends in one of these.
_TERMINATORS = (".", "!", "?", ":")


def _summary_finding(doc: str, where: str) -> str | None:
    """Return a style finding for *doc*'s summary line, or ``None`` if it reads well."""
    lines = doc.strip().splitlines()
    if not lines:
        return f"{where}: empty docstring"
    first = lines[0].strip()
    if first.endswith(_TERMINATORS):
        return None
    if len(lines) > 1 and lines[1].strip():
        return f"{where}: summary wraps onto line 2: {first[:70]!r}"
    return f"{where}: summary has no terminal punctuation: {first[:70]!r}"


def _walk(
    body: list[ast.stmt],
    qualname: list[str],
    rel: str,
    missing: list[str],
    style: list[str],
) -> None:
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                name = ".".join(qualname + [node.name])
                doc = ast.get_docstring(node)
                if not doc:
                    missing.append(f"{rel}:{node.lineno}: {name}")
                else:
                    finding = _summary_finding(doc, f"{rel}:{node.lineno}: {name}")
                    if finding:
                        style.append(finding)
            # Definitions nested inside functions are local helpers, not API.
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                name = ".".join(qualname + [node.name])
                doc = ast.get_docstring(node)
                if not doc:
                    missing.append(f"{rel}:{node.lineno}: {name}")
                else:
                    finding = _summary_finding(doc, f"{rel}:{node.lineno}: {name}")
                    if finding:
                        style.append(finding)
                _walk(node.body, qualname + [node.name], rel, missing, style)
        elif isinstance(node, ast.If):
            _walk(node.body, qualname, rel, missing, style)
            _walk(node.orelse, qualname, rel, missing, style)
        elif isinstance(node, ast.Try):
            for sub in (node.body, node.orelse, node.finalbody, *[h.body for h in node.handlers]):
                _walk(sub, qualname, rel, missing, style)
        elif isinstance(node, ast.With):
            _walk(node.body, qualname, rel, missing, style)


def test_every_module_and_public_definition_is_documented() -> None:
    missing_modules: list[str] = []
    missing: list[str] = []
    style: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = str(path.relative_to(PACKAGE_ROOT.parent))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            missing_modules.append(rel)
        _walk(tree.body, [], rel, missing, style)
    assert not missing_modules, (
        f"{len(missing_modules)} module(s) without a docstring:\n" + "\n".join(missing_modules)
    )
    assert not missing, (
        f"{len(missing)} public definition(s) without a docstring:\n" + "\n".join(missing)
    )
    assert not style, f"{len(style)} summary line(s) to reword:\n" + "\n".join(style)


def test_gate_detects_a_missing_and_a_wrapped_docstring() -> None:
    """The walker itself flags an undocumented def and a two-line summary (self-test)."""
    src = (
        "class Thing:\n"
        '    """Documented."""\n'
        "    def compute(self):\n"
        "        return 1\n"
        "def helper():\n"
        '    """A summary that keeps\n'
        '    going onto a second line."""\n'
    )
    missing: list[str] = []
    style: list[str] = []
    _walk(ast.parse(src).body, [], "sample.py", missing, style)
    assert len(missing) == 1 and "Thing.compute" in missing[0]
    assert len(style) == 1 and "wraps onto line 2" in style[0]
