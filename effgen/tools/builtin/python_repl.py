"""
Python REPL tool with persistent sessions.

This module provides a Python REPL (Read-Eval-Print Loop) tool with
persistent session state, safe evaluation, and comprehensive error handling.

User code never runs in the host process. Each session is backed by a
dedicated **worker subprocess** (:mod:`effgen.tools.builtin._repl_worker`)
started in its own process group. The advertised timeout is enforced by the
parent with a wall-clock deadline and a hard ``SIGKILL`` of the worker's
process group, so a runaway such as ``while True: pass`` is killed on time
instead of pinning a CPU. Memory and output caps are likewise enforced from
outside the executed code.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import select
import subprocess
import sys
import threading
import time
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)

_WORKER_PATH = os.path.join(os.path.dirname(__file__), "_repl_worker.py")


class _Worker:
    """Handle to a single persistent REPL worker subprocess."""

    __slots__ = ("proc", "resp_fd", "lock")

    def __init__(self, proc: subprocess.Popen, resp_fd: int):
        self.proc = proc
        self.resp_fd = resp_fd
        self.lock = threading.Lock()

    def kill(self) -> None:
        """Hard-kill the worker's whole process group and release its fds."""
        proc = self.proc
        try:
            if proc.poll() is None:
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal_SIGKILL())
                    except (ProcessLookupError, PermissionError, OSError):
                        proc.kill()
                else:  # pragma: no cover - non-POSIX
                    proc.kill()
        except Exception:  # pragma: no cover - defensive
            pass
        for closer in (
            lambda: proc.stdin and proc.stdin.close(),
            lambda: os.close(self.resp_fd),
        ):
            try:
                closer()
            except Exception:  # pragma: no cover
                pass
        try:
            proc.wait(timeout=5)
        except Exception:  # pragma: no cover
            pass


def signal_SIGKILL() -> int:
    import signal

    return signal.SIGKILL


class PythonREPL(BaseTool):
    """
    Python REPL tool with persistent session state.

    Features:
    - Persistent variable state across executions
    - Safe evaluation with restricted builtins
    - Import management
    - Comprehensive error handling
    - Output capture (stdout/stderr)
    - Expression vs statement handling
    - Session management (reset, save, restore)

    Security:
    - User code runs in an isolated worker subprocess (its own process group)
    - Wall-clock timeout enforced by the parent with a hard process-group kill
    - Address-space (memory) limit applied to the worker via RLIMIT_AS
    - stdout/stderr output capped outside the executed code
    - Restricted builtins (no file operations, eval, exec by default)
    - Configurable allowed imports
    """

    # Standardized resource limits
    DEFAULT_TIMEOUT = 30          # seconds (matches ToolMetadata.timeout_seconds)
    DEFAULT_MAX_OUTPUT = 102400   # 100 KB
    DEFAULT_MAX_MEMORY_MB = 1024  # per-worker address-space cap (RLIMIT_AS)
    # Hard ceiling on bytes read back from a worker for a single request.
    _MAX_RESPONSE_BYTES = 4 * 1024 * 1024

    # Dangerous builtins to restrict
    RESTRICTED_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
    }

    # Default allowed imports
    DEFAULT_ALLOWED_IMPORTS = {
        "math",
        "random",
        "datetime",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "operator",
        "statistics",
        "decimal",
        "fractions",
    }

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_output: int = DEFAULT_MAX_OUTPUT,
        max_memory_mb: int | None = DEFAULT_MAX_MEMORY_MB,
    ):
        """Initialize the Python REPL.

        Args:
            timeout: Wall-clock seconds before a runaway worker is killed.
            max_output: Maximum captured stdout/stderr bytes per execution.
            max_memory_mb: Per-worker address-space cap in MiB (``None`` disables
                the RLIMIT_AS limit; the timeout kill still applies).
        """
        super().__init__(
            metadata=ToolMetadata(
                name="python_repl",
                description="Execute Python code in a persistent REPL session with safe evaluation",
                category=ToolCategory.CODE_EXECUTION,
                parameters=[
                    ParameterSpec(
                        name="code",
                        type=ParameterType.STRING,
                        description="Python code to execute",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="session_id",
                        type=ParameterType.STRING,
                        description="Session identifier for persistent state (default: 'default')",
                        required=False,
                        default="default",
                    ),
                    ParameterSpec(
                        name="reset_session",
                        type=ParameterType.BOOLEAN,
                        description="Whether to reset the session before execution",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="return_variables",
                        type=ParameterType.BOOLEAN,
                        description="Whether to return all session variables",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="restricted_mode",
                        type=ParameterType.BOOLEAN,
                        description="Whether to use restricted builtins (safer but limited)",
                        required=False,
                        default=True,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "result": {"type": "any"},
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                        "error": {"type": "string"},
                        "variables": {"type": "object"},
                    },
                },
                timeout_seconds=30,
                tags=["python", "repl", "code", "execution"],
                examples=[
                    {
                        "code": "x = 10\ny = 20\nx + y",
                        "session_id": "default",
                        "output": {
                            "result": 30,
                            "stdout": "",
                            "error": None,
                        },
                    },
                    {
                        "code": "import math\nmath.sqrt(16)",
                        "output": {
                            "result": 4.0,
                            "stdout": "",
                        },
                    },
                ],
            )
        )
        self._timeout = int(timeout)
        self._max_output = int(max_output)
        self._max_memory_bytes = (
            int(max_memory_mb) * 1024 * 1024 if max_memory_mb else None
        )
        self._allowed_imports = self.DEFAULT_ALLOWED_IMPORTS.copy()
        # session_id -> _Worker (live worker subprocess holding the namespace)
        self._workers: dict[str, _Worker] = {}
        self._workers_lock = threading.Lock()

    async def initialize(self) -> None:
        """Initialize the REPL (workers are spawned lazily on first use)."""
        await super().initialize()

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def _spawn_worker(self) -> _Worker:
        """Start a fresh worker subprocess in its own process group."""
        resp_r, resp_w = os.pipe()
        env = os.environ.copy()
        env["EFFGEN_REPL_RESP_FD"] = str(resp_w)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        max_mem = self._max_memory_bytes

        def _preexec() -> None:  # pragma: no cover - runs in the child
            # New session => new process group, so the parent can kill the
            # whole tree (including grandchildren) with a single signal.
            os.setsid()
            if max_mem:
                try:
                    import resource

                    resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))
                except Exception:
                    pass

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "pass_fds": (resp_w,),
            "env": env,
        }
        if hasattr(os, "setsid"):
            popen_kwargs["preexec_fn"] = _preexec
        try:
            proc = subprocess.Popen([sys.executable, _WORKER_PATH], **popen_kwargs)
        finally:
            os.close(resp_w)  # the child owns its copy now
        return _Worker(proc, resp_r)

    def _get_worker(self, session_id: str) -> _Worker:
        """Return the live worker for a session, spawning one if needed."""
        with self._workers_lock:
            worker = self._workers.get(session_id)
            if worker is None or worker.proc.poll() is not None:
                if worker is not None:
                    worker.kill()
                worker = self._spawn_worker()
                self._workers[session_id] = worker
            return worker

    def _kill_worker(self, session_id: str) -> None:
        with self._workers_lock:
            worker = self._workers.pop(session_id, None)
        if worker is not None:
            worker.kill()

    def _request(
        self,
        session_id: str,
        payload: dict[str, Any],
        _attempt: int = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send one request to a session worker and read its response.

        Enforces the wall-clock timeout: if the worker does not answer within
        ``timeout`` (or ``self._timeout`` when no per-call override is given)
        seconds it is hard-killed (process group) and a typed timeout failure is
        returned. Runs blocking I/O — call via a thread.

        If a *reused* worker has exited between requests before producing any
        output, the request is transparently retried once on a fresh worker —
        this is safe (no output was emitted, so nothing double-runs) and fixes a
        rare race where a worker died after the previous request.
        """
        worker = self._get_worker(session_id)
        line = (json.dumps(payload) + "\n").encode("utf-8")
        with worker.lock:
            try:
                worker.proc.stdin.write(line)
                worker.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                # Worker died between requests — respawn once and retry.
                self._kill_worker(session_id)
                worker = self._get_worker(session_id)
                try:
                    worker.proc.stdin.write(line)
                    worker.proc.stdin.flush()
                except Exception as e:
                    return self._failure(f"REPL worker unavailable: {e}")

            effective_timeout = timeout if (timeout is not None and timeout > 0) else self._timeout
            deadline = time.monotonic() + effective_timeout
            buf = b""
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_worker(session_id)
                    return self._failure(
                        f"Execution timed out after {effective_timeout:g}s "
                        "(worker process killed)",
                        timeout=True,
                    )
                try:
                    ready, _, _ = select.select([worker.resp_fd], [], [], remaining)
                except OSError:
                    self._kill_worker(session_id)
                    return self._failure("REPL worker channel closed")
                if not ready:
                    continue
                try:
                    chunk = os.read(worker.resp_fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    self._kill_worker(session_id)
                    # Worker died before answering. If it produced no partial
                    # output and we haven't retried yet, respawn and retry once
                    # (safe: nothing was emitted for this request).
                    if buf == b"" and _attempt == 0:
                        return self._request(
                            session_id, payload, _attempt=1, timeout=timeout
                        )
                    return self._failure("REPL worker exited unexpectedly")
                buf += chunk
                if len(buf) > self._MAX_RESPONSE_BYTES:
                    self._kill_worker(session_id)
                    return self._failure("REPL produced too much output")
                nl = buf.find(b"\n")
                if nl != -1:
                    try:
                        return json.loads(buf[:nl].decode("utf-8", "replace"))
                    except Exception as e:
                        self._kill_worker(session_id)
                        return self._failure(f"REPL protocol error: {e}")

    @staticmethod
    def _failure(message: str, timeout: bool = False) -> dict[str, Any]:
        result = {
            "success": False,
            "result": None,
            "stdout": "",
            "stderr": "",
            "error": message,
        }
        if timeout:
            result["timeout"] = True
        return result

    def _is_import_allowed(self, module_name: str) -> bool:
        """Check if an import is allowed."""
        # Get base module name (before any dots)
        base_module = module_name.split(".")[0]
        return base_module in self._allowed_imports

    def _check_imports(self, code: str) -> str | None:
        """
        Check if code contains only allowed imports.

        Returns:
            Optional[str]: Error message if disallowed import found, None otherwise
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._is_import_allowed(alias.name):
                            return f"Import of '{alias.name}' is not allowed"
                elif isinstance(node, ast.ImportFrom):
                    if node.module and not self._is_import_allowed(node.module):
                        return f"Import from '{node.module}' is not allowed"
        except SyntaxError:
            # Let the actual execution handle syntax errors
            pass

        return None

    def _check_security(self, code: str) -> str | None:
        """
        Check for sandbox-escape patterns beyond import restrictions.

        Blocks:
        - Direct ``__import__`` calls
        - ``importlib`` usage
        - ``__builtins__`` dict manipulation
        - ``getattr``/``setattr`` targeting blocked modules or dunder attrs
        - String-concatenation bypasses  (e.g. ``eval("__" + "import__")``)

        Returns:
            Error message if dangerous pattern found, None otherwise.
        """
        # Fast string-level checks (catches dynamic construction too)
        blocked_strings = [
            "__import__",
            "importlib",
            "__builtins__",
            "__subclasses__",
            "__bases__",
            "__mro__",
            "__globals__",
            "__code__",
        ]
        for pattern in blocked_strings:
            if pattern in code:
                return f"Access to '{pattern}' is blocked in restricted mode"

        # AST-level checks
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None  # let execution handle it

        for node in ast.walk(tree):
            # Block getattr/setattr calls that could reach blocked objects
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in ("getattr", "setattr", "delattr"):
                    # Check if second arg (attr name) is a blocked dunder
                    if len(node.args) >= 2:
                        attr_arg = node.args[1]
                        if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
                            if attr_arg.value.startswith("__") and attr_arg.value.endswith("__"):
                                return (
                                    f"getattr/setattr with dunder attribute "
                                    f"'{attr_arg.value}' is blocked in restricted mode"
                                )

            # Block attribute access to __builtins__, __globals__ etc.
            if isinstance(node, ast.Attribute):
                if node.attr in ("__builtins__", "__globals__", "__subclasses__",
                                 "__bases__", "__mro__", "__code__"):
                    return f"Access to '{node.attr}' is blocked in restricted mode"

        return None

    async def _execute(
        self,
        code: str,
        session_id: str = "default",
        reset_session: bool = False,
        return_variables: bool = False,
        restricted_mode: bool = True,
        timeout: float | int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute Python code in a REPL session.

        Args:
            code: Python code to execute
            session_id: Session identifier for state persistence
            reset_session: Whether to reset session state before execution
            return_variables: Whether to include all variables in response
            restricted_mode: Whether to use restricted builtins
            timeout: Per-call wall-clock cap (seconds) overriding the instance
                default for this execution. A runaway block is hard-killed at
                this deadline; values <= 0 or ``None`` fall back to the default.

        Returns:
            Dict containing result, stdout, stderr, error, and optionally variables
        """
        # Check imports and security in restricted mode (cheap parent-side
        # gate that rejects obvious escapes before any subprocess work).
        if restricted_mode:
            import_error = self._check_imports(code)
            if import_error:
                resp = self._failure(import_error)
                if return_variables:
                    resp["variables"] = {}
                return resp
            security_error = self._check_security(code)
            if security_error:
                resp = self._failure(security_error)
                if return_variables:
                    resp["variables"] = {}
                return resp

        payload = {
            "code": code,
            "restricted_mode": restricted_mode,
            "allowed_imports": sorted(self._allowed_imports),
            "return_variables": return_variables,
            "reset": reset_session,
            "max_output": self._max_output,
        }

        # Resolve the effective wall-clock cap: an explicit positive per-call
        # ``timeout`` overrides the instance default for this execution only.
        effective_timeout: float | None = None
        if timeout is not None:
            try:
                t = float(timeout)
                if t > 0:
                    effective_timeout = t
            except (TypeError, ValueError):
                effective_timeout = None

        # The worker runs user code; do the blocking request off the event loop
        # so a slow (or killed-on-timeout) worker never stalls the caller.
        response = await asyncio.to_thread(
            self._request, session_id, payload, timeout=effective_timeout
        )

        if response.get("error"):
            logger.debug("REPL execution error: %s", response["error"])
        if return_variables and "variables" not in response:
            response["variables"] = {}
        return response

    def add_allowed_import(self, module_name: str) -> None:
        """
        Add a module to the allowed imports list.

        Args:
            module_name: Name of the module to allow
        """
        self._allowed_imports.add(module_name)

    def remove_allowed_import(self, module_name: str) -> None:
        """
        Remove a module from the allowed imports list.

        Args:
            module_name: Name of the module to disallow
        """
        self._allowed_imports.discard(module_name)

    def reset_session(self, session_id: str) -> None:
        """
        Reset a session to clean state by terminating its worker.

        The next execution for this session spawns a fresh worker with an empty
        namespace.

        Args:
            session_id: Session to reset
        """
        self._kill_worker(session_id)

    def get_session_variables(self, session_id: str) -> dict[str, str]:
        """
        Get all variables in a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict mapping variable names to their string representations
        """
        with self._workers_lock:
            live = session_id in self._workers
        if not live:
            return {}
        response = self._request(
            session_id,
            {
                "code": "",
                "restricted_mode": False,
                "allowed_imports": sorted(self._allowed_imports),
                "return_variables": True,
                "reset": False,
                "max_output": self._max_output,
            },
        )
        return response.get("variables") or {}

    async def cleanup(self) -> None:
        """Terminate every worker subprocess and clear state."""
        with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.kill()
        await super().cleanup()
