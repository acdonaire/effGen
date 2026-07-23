"""Every public function signature in ``effgen`` carries complete type hints.

Walks the package source with ``ast`` and asserts that each public function or
method (name not starting with ``_``, plus the context-manager and container
dunders) annotates every parameter and its return type. Conditionally defined
functions (inside ``if``/``try`` blocks, e.g. optional-dependency fallbacks)
are included; functions nested inside other functions are not part of the
public surface and are skipped.
"""

from __future__ import annotations

import ast
from pathlib import Path

import effgen

PACKAGE_ROOT = Path(effgen.__file__).parent

# Dunders that form part of a class's public contract.
PUBLIC_DUNDERS = {
    "__init__",
    "__call__",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    "__iter__",
    "__next__",
    "__anext__",
    "__aiter__",
    "__getitem__",
    "__setitem__",
    "__len__",
    "__contains__",
}


def _is_public(name: str) -> bool:
    return name in PUBLIC_DUNDERS or not name.startswith("_")


def _missing_hints(node: ast.FunctionDef | ast.AsyncFunctionDef, in_class: bool) -> list[str]:
    """Return a list of unannotated parameter names (plus ``return``)."""
    args = node.args
    params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    decorators = {d.id for d in node.decorator_list if isinstance(d, ast.Name)}
    if in_class and "staticmethod" not in decorators and params and params[0].arg in ("self", "cls"):
        params = params[1:]
    missing = [p.arg for p in params if p.annotation is None]
    if args.vararg is not None and args.vararg.annotation is None:
        missing.append("*" + args.vararg.arg)
    if args.kwarg is not None and args.kwarg.annotation is None:
        missing.append("**" + args.kwarg.arg)
    if node.returns is None:
        missing.append("return")
    return missing


def _walk(body: list[ast.stmt], qualname: list[str], in_class: bool, rel: str, findings: list[str]) -> None:
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if _is_public(node.name):
                missing = _missing_hints(node, in_class)
                if missing:
                    name = ".".join(qualname + [node.name])
                    findings.append(f"{rel}:{node.lineno}: {name} missing {', '.join(missing)}")
            # Functions nested inside functions are local helpers, not API.
        elif isinstance(node, ast.ClassDef):
            if _is_public(node.name):
                _walk(node.body, qualname + [node.name], True, rel, findings)
        elif isinstance(node, ast.If):
            _walk(node.body, qualname, in_class, rel, findings)
            _walk(node.orelse, qualname, in_class, rel, findings)
        elif isinstance(node, ast.Try):
            for sub in (node.body, node.orelse, node.finalbody, *[h.body for h in node.handlers]):
                _walk(sub, qualname, in_class, rel, findings)
        elif isinstance(node, ast.With):
            _walk(node.body, qualname, in_class, rel, findings)


def test_public_signatures_fully_annotated() -> None:
    findings: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = str(path.relative_to(PACKAGE_ROOT.parent))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _walk(tree.body, [], False, rel, findings)
    assert not findings, (
        f"{len(findings)} public signature(s) missing type hints:\n" + "\n".join(findings)
    )


def test_gate_detects_missing_hints() -> None:
    """The walker itself flags an unannotated public def (self-test)."""
    src = (
        "class Thing:\n"
        "    def compute(self, x):\n"
        "        return x\n"
        "def helper(a, *b, **c):\n"
        "    pass\n"
    )
    findings: list[str] = []
    _walk(ast.parse(src).body, [], False, "sample.py", findings)
    assert len(findings) == 2
    assert "Thing.compute missing x, return" in findings[0]
    assert "helper missing a, *b, **c, return" in findings[1]
