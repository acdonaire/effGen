"""Code execution tool result reflects the program's own outcome.

The tool result mirrors the exit code: a process that exits non-zero (uncaught
exception, ``sys.exit(n)``, syntax error, non-zero shell command, or a runtime
abort) is reported as a failure, while a clean exit is a success. JavaScript runs
under the default memory limit because the tool raises the limit to a floor the
runtime needs to start.

These route through the real subprocess sandbox (no Docker required); the
JavaScript cases skip when ``node`` is not installed.
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

from effgen.security.sandbox import reset_sandbox_cache
from effgen.tools.builtin.code_executor import CodeExecutor

_HAS_NODE = shutil.which("node") is not None


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _subprocess_backend(monkeypatch):
    monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "subprocess")
    reset_sandbox_cache()
    yield
    reset_sandbox_cache()


def _exec(**kwargs):
    ex = CodeExecutor()
    _run(ex.initialize())
    return _run(ex.execute(**kwargs))


def test_clean_python_exit_is_success():
    r = _exec(code="print('hi')", language="python")
    assert r.success is True
    assert r.output["exit_code"] == 0
    assert "hi" in r.output["stdout"]
    assert r.error is None


@pytest.mark.parametrize(
    "code,language",
    [
        ("raise ValueError('boom')", "python"),
        ("import sys; sys.exit(2)", "python"),
        ("def (:", "python"),  # syntax error
        ("exit 3", "bash"),
    ],
)
def test_nonzero_exit_is_failure(code, language):
    r = _exec(code=code, language=language)
    assert r.success is False, "a non-zero exit must be reported as a failure"
    assert r.output["exit_code"] != 0
    # The full payload is preserved so a caller/model still sees the detail.
    assert "stdout" in r.output and "stderr" in r.output
    assert r.error  # an actionable message is attached


def test_failure_message_names_the_exception():
    r = _exec(code="raise ValueError('boom')", language="python")
    assert r.success is False
    assert "ValueError: boom" in r.error


def test_clean_bash_exit_is_success():
    r = _exec(code="echo hello", language="bash")
    assert r.success is True
    assert r.output["exit_code"] == 0
    assert "hello" in r.output["stdout"]


@pytest.mark.skipif(not _HAS_NODE, reason="node not installed")
def test_javascript_runs_under_default_memory_limit():
    # The default 256m is below node/V8's startup reservation; the tool raises it
    # to a floor so JavaScript runs out of the box.
    r = _exec(code="console.log('js', 6 * 7)", language="javascript")
    assert r.success is True, r.error
    assert r.output["exit_code"] == 0
    assert "js 42" in r.output["stdout"]


@pytest.mark.skipif(not _HAS_NODE, reason="node not installed")
def test_javascript_runtime_error_is_failure():
    r = _exec(code="throw new Error('nope')", language="javascript")
    assert r.success is False
    assert r.output["exit_code"] != 0
    assert "nope" in r.error


def test_memory_floor_only_raises_below_the_minimum():
    ex = CodeExecutor()
    # JavaScript has a floor; a request below it is raised, an equal/higher
    # request is left untouched, and languages without a floor are unchanged.
    assert ex._resolve_memory_limit("javascript", "256m") == "1g"
    assert ex._resolve_memory_limit("javascript", "2g") == "2g"
    assert ex._resolve_memory_limit("python", "256m") == "256m"
