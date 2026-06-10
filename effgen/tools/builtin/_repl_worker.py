"""Out-of-process worker that executes Python REPL code under hard limits.

This module is launched as a **standalone subprocess** by
:class:`effgen.tools.builtin.python_repl.PythonREPL`. It deliberately imports
nothing from ``effgen`` so start-up stays cheap and the worker has a minimal,
auditable surface.

Protocol
--------
* The parent writes one JSON request per line to the worker's **stdin**.
* The worker writes one JSON response per line to the descriptor named by the
  ``EFFGEN_REPL_RESP_FD`` environment variable (an inherited pipe). Standard
  out/err are redirected to ``/dev/null`` while user code runs so stray writes
  can never corrupt the protocol channel.

Each request is ``{"code", "restricted_mode", "allowed_imports",
"return_variables", "reset", "max_output"}``. The persistent namespace lives in
this process, so variables survive across requests — exactly like an in-process
REPL — but a runaway request (``while True: pass``) can be killed by the parent
sending ``SIGKILL`` to the worker's process group with no lasting harm.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

_RESP_FD = int(os.environ.get("EFFGEN_REPL_RESP_FD", "3"))

# Builtins withheld in restricted mode (mirrors PythonREPL.RESTRICTED_BUILTINS).
_RESTRICTED_BUILTINS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
}


def _install_parent_death_signal() -> None:
    """Ask the kernel to SIGKILL this worker if its parent dies (Linux only).

    Without this, a worker busy-executing user code (e.g. ``while True: pass``)
    never returns to read stdin, so an abrupt parent death would leave it
    running as an orphan. ``PR_SET_PDEATHSIG`` closes that gap; it is set here
    (post-exec) because the flag is cleared across ``execve``.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        import signal

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        # Guard the race where the parent already exited before prctl ran.
        if os.getppid() == 1:
            os._exit(0)
    except Exception:
        pass


def _send(obj: dict) -> None:
    """Write a single JSON response line on the protocol descriptor."""
    data = (json.dumps(obj) + "\n").encode("utf-8", errors="replace")
    os.write(_RESP_FD, data)


def _restricted_builtins(allowed_imports: set[str]) -> dict:
    """Build a restricted ``__builtins__`` mapping for user code."""
    safe = {
        name: getattr(builtins, name)
        for name in dir(builtins)
        if name not in _RESTRICTED_BUILTINS
    }
    original_import = builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        base_module = name.split(".")[0]
        if base_module not in allowed_imports:
            raise ImportError(f"Import of '{name}' is not allowed")
        return original_import(name, globals, locals, fromlist, level)

    safe["__import__"] = safe_import
    return safe


def _serialize_result(value):
    """Return a JSON-safe representation of an execution result.

    Plain JSON-serialisable values pass through unchanged (so ``2 + 2`` stays an
    ``int``); everything else degrades to ``repr`` so the protocol channel never
    breaks on an exotic object.
    """
    if value is None:
        return None
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        try:
            return repr(value)
        except Exception:  # pragma: no cover - repr of a hostile object
            return "<unrepresentable result>"


def _run_one(namespace: dict, req: dict) -> dict:
    code = req.get("code", "")
    restricted = req.get("restricted_mode", True)
    allowed_imports = set(req.get("allowed_imports", ()))
    return_variables = req.get("return_variables", False)
    max_output = int(req.get("max_output", 102400))

    if req.get("reset"):
        namespace.clear()

    if restricted:
        namespace["__builtins__"] = _restricted_builtins(allowed_imports)

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = None
    error = None

    # Redirect the OS-level stdout/stderr to /dev/null during execution so a
    # stray ``os.write(1, ...)`` cannot corrupt the JSON protocol on fd 3.
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                tree = ast.parse(code, mode="eval")
                compiled = compile(tree, "<repl>", mode="eval")
                result = eval(compiled, namespace, namespace)
            except SyntaxError:
                tree = ast.parse(code, mode="exec")
                compiled = compile(tree, "<repl>", mode="exec")
                exec(compiled, namespace, namespace)
                # Re-evaluate a trailing bare expression (but never a Call, which
                # would run side effects twice) to surface its value.
                if (
                    tree.body
                    and isinstance(tree.body[-1], ast.Expr)
                    and not isinstance(tree.body[-1].value, ast.Call)
                ):
                    last_expr = tree.body[-1].value
                    result = eval(
                        compile(ast.Expression(body=last_expr), "<repl>", mode="eval"),
                        namespace,
                        namespace,
                    )
    except Exception as e:  # noqa: BLE001 - user code may raise anything
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)

    stdout_value = stdout_capture.getvalue()
    stderr_value = stderr_capture.getvalue()
    if len(stdout_value) > max_output:
        stdout_value = stdout_value[:max_output] + "\n... (output truncated)"
    if len(stderr_value) > max_output:
        stderr_value = stderr_value[:max_output] + "\n... (output truncated)"

    response = {
        "success": error is None,
        "result": _serialize_result(result),
        "stdout": stdout_value,
        "stderr": stderr_value,
        "error": error,
    }

    if return_variables:
        variables = {}
        for k, v in namespace.items():
            if k.startswith("_") or k == "__builtins__":
                continue
            try:
                variables[k] = repr(v)
            except Exception:  # pragma: no cover
                variables[k] = "<unrepresentable>"
        response["variables"] = variables

    return response


def main() -> int:
    _install_parent_death_signal()
    namespace: dict = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _send({"success": False, "result": None, "stdout": "", "stderr": "",
                   "error": "worker: malformed request"})
            continue
        try:
            _send(_run_one(namespace, req))
        except Exception as e:  # pragma: no cover - defensive
            _send({"success": False, "result": None, "stdout": "", "stderr": "",
                   "error": f"worker fatal: {type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
