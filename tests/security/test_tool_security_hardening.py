"""Security regression tests for the built-in execution/IO tools.

Covers, with real subprocesses and real sockets (no mocks of the behaviour
under test):

* PythonREPL — a runaway (``while True``) is hard-killed at the wall-clock
  timeout; the worker runs out-of-process; persistence still works.
* URLFetch — private/loopback/link-local/metadata targets are blocked by
  default, redirects are re-validated, and the block is opt-out only.
* Retrieval — indexes persist as JSON; a malicious pickle is refused (no code
  execution) unless explicitly allowed.
* BashTool — a timeout kills the whole process group (no orphaned children);
  argv execution is preferred over the shell when no shell features are used.
* FileOperations — path traversal, symlink escape and out-of-root writes are
  rejected.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import pickle
import subprocess
import tempfile
import time

import pytest

from effgen.tools.builtin.bash_tool import BashTool
from effgen.tools.builtin.file_ops import FileOperations
from effgen.tools.builtin.python_repl import PythonREPL
from effgen.tools.builtin.retrieval import Document, Retrieval
from effgen.tools.builtin.url_fetch import URLFetchTool, _is_blocked_ip


# --------------------------------------------------------------------------- #
# PythonREPL                                                                   #
# --------------------------------------------------------------------------- #
class TestPythonREPLSandbox:
    async def test_infinite_loop_killed_at_timeout(self):
        repl = PythonREPL(timeout=2)
        await repl.initialize()
        try:
            t0 = time.time()
            res = await repl.execute(code="while True:\n    pass")
            elapsed = time.time() - t0
            assert res.success is False
            assert res.output.get("timeout") is True
            # Killed promptly at the deadline, not hanging indefinitely.
            assert elapsed < 6, f"timeout not enforced (took {elapsed:.1f}s)"
        finally:
            await repl.cleanup()

    async def test_basic_exec_and_persistence(self):
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            a = await repl.execute(code="x = 21\ny = 21\nx + y")
            assert a.success and a.output["result"] == 42
            # State persists across calls in the same session worker.
            b = await repl.execute(code="x + y + 1")
            assert b.success and b.output["result"] == 43
            c = await repl.execute(code="print('hello')")
            assert c.output["stdout"].strip() == "hello"
        finally:
            await repl.cleanup()

    async def test_runs_out_of_process(self):
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            res = await repl.execute(
                code="import os\nos.getpid()", restricted_mode=False
            )
            assert res.success
            assert res.output["result"] != os.getpid()
        finally:
            await repl.cleanup()

    async def test_import_gate_blocks_disallowed(self):
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            res = await repl.execute(code="import os")
            assert res.success is False
            assert "not allowed" in str(res.output["error"]).lower()
        finally:
            await repl.cleanup()

    async def test_no_orphan_workers_after_cleanup(self):
        repl = PythonREPL(timeout=5)
        await repl.initialize()
        await repl.execute(code="1 + 1")
        await repl.execute(code="1 + 1", session_id="other")
        # The exact worker PIDs this instance owns.
        pids = [w.proc.pid for w in repl._workers.values()]
        assert len(pids) == 2
        await repl.cleanup()
        await asyncio.sleep(0.3)

        def _alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:  # pragma: no cover
                return True

        survivors = [pid for pid in pids if _alive(pid)]
        assert survivors == [], f"orphan REPL workers after cleanup: {survivors}"


# --------------------------------------------------------------------------- #
# URLFetch / SSRF                                                              #
# --------------------------------------------------------------------------- #
class TestURLFetchSSRF:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/",
            "http://localhost/",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/",
            "http://192.168.1.1/",
            "http://0.0.0.0/",
            "http://[::1]/",
        ],
    )
    async def test_internal_targets_blocked_by_default(self, url):
        tool = URLFetchTool(timeout=5)
        res = await tool.execute(url=url)
        assert res.success is False
        assert "ssrf" in str(res.error).lower() or "resolve" in str(res.error).lower()

    async def test_allow_private_opt_in_attempts_connection(self):
        # With the override the SSRF guard must NOT short-circuit; the request is
        # attempted (and fails on connection refused at a closed port).
        tool = URLFetchTool(timeout=3, allow_private=True)
        res = await tool.execute(url="http://127.0.0.1:9/")
        assert "SSRF" not in str(res.error)

    async def test_redirect_to_internal_is_revalidated(self):
        """A public first hop must not be able to bounce us to an internal IP."""
        tool = URLFetchTool(timeout=5)

        class _Resp:
            def __init__(self, status, location=None, text=""):
                self.status_code = status
                self.is_redirect = status in (301, 302, 303, 307, 308)
                self.headers = {"Location": location} if location else {}
                self.text = text

            def raise_for_status(self):
                pass

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, **kw):
                return _Resp(302, location="http://169.254.169.254/latest/meta-data/")

        class _FakeRequests:
            @staticmethod
            def Session():
                return _Session()

        with pytest.raises(ValueError):
            # First hop (example.com) resolves public, redirect points internal.
            tool._fetch_with_requests(_FakeRequests, "https://example.com/")

    def test_is_blocked_ip_classification(self):
        assert _is_blocked_ip(ipaddress.ip_address("127.0.0.1"))
        assert _is_blocked_ip(ipaddress.ip_address("169.254.169.254"))
        assert _is_blocked_ip(ipaddress.ip_address("10.1.2.3"))
        assert _is_blocked_ip(ipaddress.ip_address("::1"))
        assert _is_blocked_ip(ipaddress.ip_address("::ffff:127.0.0.1"))
        assert not _is_blocked_ip(ipaddress.ip_address("8.8.8.8"))


# --------------------------------------------------------------------------- #
# Retrieval pickle gating                                                      #
# --------------------------------------------------------------------------- #
class TestRetrievalPickle:
    async def test_json_roundtrip(self, tmp_path):
        import numpy as np

        r = Retrieval()
        await r.initialize()
        r.documents["d1"] = Document(
            id="d1", content="hello world", metadata={"src": "x"},
            embedding=np.array([0.1, 0.2, 0.3]),
        )
        r.doc_ids = ["d1"]
        path = str(tmp_path / "index.json")
        r.save_index(path)
        # Stored as JSON text, not a pickle.
        assert open(path, encoding="utf-8").read(1) == "{"

        r2 = Retrieval(index_path=path)
        assert list(r2.documents) == ["d1"]
        assert r2.documents["d1"].embedding.tolist() == pytest.approx([0.1, 0.2, 0.3])

    def test_malicious_pickle_refused_no_rce(self, tmp_path):
        flag = tmp_path / "pwned"

        class Evil:
            def __reduce__(self):
                return (os.system, (f"touch {flag}",))

        pkl = tmp_path / "evil.pkl"
        with open(pkl, "wb") as f:
            pickle.dump({"documents": {}, "doc_ids": [], "x": Evil()}, f)

        with pytest.raises(ValueError) as exc:
            Retrieval(index_path=str(pkl))
        assert "pickle" in str(exc.value).lower()
        assert not flag.exists(), "pickle executed code — RCE not prevented"

    def test_allow_pickle_opt_in_loads(self, tmp_path):
        data = {"documents": {}, "doc_ids": [], "chunk_size": 500, "chunk_overlap": 100}
        pkl = tmp_path / "legacy.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(data, f)
        r = Retrieval(index_path=str(pkl), allow_pickle=True)
        assert r.doc_ids == []


# --------------------------------------------------------------------------- #
# BashTool                                                                     #
# --------------------------------------------------------------------------- #
class TestBashSecurity:
    async def test_timeout_kills_process_group(self):
        bash = BashTool(timeout=2)
        t0 = time.time()
        res = await bash.execute(command="sleep 120 & exec sleep 120")
        assert res.success is False
        assert (time.time() - t0) < 6
        await asyncio.sleep(0.5)
        out = subprocess.run(
            ["pgrep", "-af", "sleep 120"], capture_output=True, text=True
        ).stdout
        leftovers = [ln for ln in out.splitlines() if "sleep 120" in ln and "pgrep" not in ln]
        assert leftovers == [], f"orphaned children after timeout: {leftovers}"

    async def test_argv_execution_when_no_shell_features(self):
        bash = BashTool(timeout=10)
        # Quotes are handled by argv parsing; no shell needed.
        res = await bash.execute(command="echo 'literal | not a pipe'")
        assert res.success
        assert res.output["stdout"].strip() == "literal | not a pipe"
        assert bash._needs_shell("echo hi") is False
        assert bash._needs_shell("ls *.py") is True
        assert bash._needs_shell("a | b") is True

    async def test_basic_command(self):
        bash = BashTool(timeout=10)
        res = await bash.execute(command="echo hello")
        assert res.success and res.output["stdout"].strip() == "hello"


# --------------------------------------------------------------------------- #
# FileOperations path safety                                                   #
# --------------------------------------------------------------------------- #
class TestFileOpsPathSafety:
    async def _tool(self):
        d = tempfile.mkdtemp(prefix="p17_fo_")
        with open(os.path.join(d, "ok.txt"), "w") as f:
            f.write("inside")
        return d, FileOperations(allowed_directories=[d])

    async def test_read_inside_allowed(self):
        d, tool = await self._tool()
        res = await tool.execute(operation="read", path=os.path.join(d, "ok.txt"))
        assert res.success and res.output["data"] == "inside"

    async def test_traversal_blocked(self):
        d, tool = await self._tool()
        res = await tool.execute(
            operation="read", path=os.path.join(d, "../../../../etc/passwd")
        )
        assert res.success is False

    async def test_absolute_outside_blocked(self):
        d, tool = await self._tool()
        res = await tool.execute(operation="read", path="/etc/passwd")
        assert res.success is False

    async def test_symlink_escape_blocked(self):
        d, tool = await self._tool()
        link = os.path.join(d, "evil.txt")
        os.symlink("/etc/passwd", link)
        res = await tool.execute(operation="read", path=link)
        assert res.success is False

    async def test_write_outside_blocked(self):
        d, tool = await self._tool()
        res = await tool.execute(
            operation="write", path="/tmp/p17_should_not_exist.txt", content="x"
        )
        assert res.success is False
        assert not os.path.exists("/tmp/p17_should_not_exist.txt")
