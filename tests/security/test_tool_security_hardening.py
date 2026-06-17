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
from pathlib import Path

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
        # Use a process-unique sleep duration (a large float, valid for `sleep`)
        # as a sentinel so the system-wide pgrep matches ONLY this test's own
        # children — never a co-tenant's plain `sleep 120` on a shared host.
        # The bash kill itself is correct; this keeps the TEST hermetic.
        marker = f"7777.{os.getpid() % 1000:03d}{int(time.time() * 1000) % 1000:03d}"
        bash = BashTool(timeout=2)
        t0 = time.time()
        res = await bash.execute(command=f"sleep {marker} & exec sleep {marker}")
        assert res.success is False
        assert (time.time() - t0) < 6
        await asyncio.sleep(0.5)
        out = subprocess.run(
            ["pgrep", "-af", f"sleep {marker}"], capture_output=True, text=True
        ).stdout
        leftovers = [ln for ln in out.splitlines() if marker in ln and "pgrep" not in ln]
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


# --------------------------------------------------------------------------- #
# PromptChain condition evaluation (no eval of model output)           #
# --------------------------------------------------------------------------- #
class TestChainConditionSafety:
    def _state(self):
        from effgen.prompts.chain_manager import ChainState

        return ChainState()

    def test_legit_conditions_still_work(self):
        st = self._state()
        st.set_variable("quality_score", 0.85)
        st.set_variable("result_type", "success")
        st.iteration_count = 1
        assert st.evaluate_condition("quality_score >= 0.8") is True
        assert st.evaluate_condition("result_type == 'success'") is True
        assert st.evaluate_condition("iteration_count < 3") is True
        assert st.evaluate_condition("{quality_score} >= 0.8") is True
        assert st.evaluate_condition(
            "quality_score >= 0.8 and result_type == 'success'"
        ) is True
        assert st.evaluate_condition("result_type in ['success', 'ok']") is True

    def test_model_output_is_data_not_code(self):
        # A prior step result of "1+1" must compare as the string, never compute.
        st = self._state()
        st.set_variable("step_result", "1+1")
        assert st.evaluate_condition("{step_result} == 2") is False
        assert st.evaluate_condition("step_result == '1+1'") is True

    @pytest.mark.parametrize(
        "payload",
        [
            "().__class__.__name__ == 'tuple'",
            "len(().__class__.__base__.__subclasses__()) > 0",
            "__import__('os').system('echo pwned') == 0",
            "''.__class__.__mro__[1].__subclasses__()",
            "(lambda: 1)() == 1",
        ],
    )
    def test_adversarial_payloads_inert(self, payload):
        st = self._state()
        # Either rejected (False) — never raises, never executes.
        assert st.evaluate_condition(payload) is False

    def test_injected_payload_does_not_execute(self, tmp_path):
        st = self._state()
        flag = tmp_path / "chain_pwned"
        st.set_variable(
            "evil", f"__import__('os').system('touch {flag}')"
        )
        # Interpolated into the condition the way a vulnerable eval() would see it.
        st.evaluate_condition("{evil} == 0")
        st.evaluate_condition("evil == 0")
        assert not flag.exists(), "chain condition executed code — injection not prevented"


# --------------------------------------------------------------------------- #
# shared SSRF guard used by every URL-taking tool                       #
# --------------------------------------------------------------------------- #
class TestSharedSSRFGuard:
    INTERNAL = [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://[::1]/",
    ]

    def test_net_helpers_classify(self):
        from effgen.tools.builtin import _net

        assert _net.is_blocked_ip(__import__("ipaddress").ip_address("127.0.0.1"))
        assert _net.is_blocked_ip(__import__("ipaddress").ip_address("169.254.169.254"))
        assert not _net.is_blocked_ip(__import__("ipaddress").ip_address("8.8.8.8"))
        for url in self.INTERNAL:
            with pytest.raises(_net.BlockedURLError):
                _net.check_url_safe(url)
        # opt-out lets it through the guard (no exception)
        _net.check_url_safe("http://127.0.0.1/", allow_private=True)

    def test_host_pin_rejects_other_hosts(self):
        from effgen.tools.builtin import _net

        with pytest.raises(_net.BlockedURLError):
            _net.check_url_safe(
                "https://evil.example.com/", allowed_hosts={"api.example.org"}
            )
        # wildcard suffix match
        _net.check_url_safe(
            "https://en.wikipedia.org/w/api.php", allowed_hosts={"*.wikipedia.org"}
        )

    async def test_rss_blocks_internal(self):
        from effgen.tools.builtin.rss import RSSFeedTool

        res = await RSSFeedTool().execute(operation="latest", url="http://127.0.0.1/x", n=1)
        assert res.success is False

    async def test_news_newsapi_blocks_internal(self):
        # NewsAPI host pin: an internal URL never reaches the network.
        from effgen.tools.builtin import news
        from effgen.tools.builtin._net import BlockedURLError

        with pytest.raises(BlockedURLError):
            news._newsapi_request("http://127.0.0.1:9/v2/top-headlines")

    async def test_devops_http_blocks_internal(self):
        from effgen.tools.builtin.devops import HTTPTool

        res = await HTTPTool().execute(url="http://169.254.169.254/latest/meta-data/")
        assert res.success is False

    async def test_webhook_override_blocks_internal(self):
        from effgen.tools.builtin.slack_webhook import SlackWebhookTool

        res = await SlackWebhookTool().execute(
            text="hi", webhook_url="http://127.0.0.1:9/services/x"
        )
        assert res.success is False

    def test_url_fetch_uses_shared_guard(self):
        # url_fetch must re-export the shared classifier (single source of truth).
        from effgen.tools.builtin import _net, url_fetch

        assert url_fetch._is_blocked_ip is _net.is_blocked_ip


# --------------------------------------------------------------------------- #
# file-path tools may not read sensitive locations; tighten on demand    #
# --------------------------------------------------------------------------- #
class TestFilePathConfinement:
    # Sensitive targets an attacker actually wants (refused by default).
    SENSITIVE = ["/etc/passwd", str(Path.home() / ".ssh" / "id_rsa")]

    async def test_image_info_blocks_sensitive(self):
        from effgen.tools.builtin.image_info import ImageInfoTool

        for target in self.SENSITIVE:
            res = await ImageInfoTool().execute(operation="info", image_path=target)
            assert res.success is False
            assert "protected" in str(res.error).lower() or "refus" in str(res.error).lower()

    async def test_pdf_blocks_etc_passwd(self):
        from effgen.tools.builtin.pdf import PDFTool

        res = await PDFTool().execute(operation="text", path="/etc/passwd")
        assert res.success is False
        assert "protected" in str(res.error).lower() or "refus" in str(res.error).lower()

    async def test_data_analysis_blocks_etc(self):
        from effgen.tools.builtin.data_analysis import DataFrameTool

        res = await DataFrameTool().execute(file_path="/etc/passwd", operation="head")
        assert res.success is False

    async def test_ocr_blocks_ssh_key(self):
        from effgen.tools.builtin.ocr import OCRTool

        res = await OCRTool().execute(
            operation="extract", image_path="/etc/shadow"
        )
        assert res.success is False
        assert "protected" in str(res.error).lower() or "refus" in str(res.error).lower()

    async def test_ordinary_file_allowed_by_default(self, tmp_path):
        # A normal (non-sensitive) file is readable by default — the tools stay
        # usable; confinement only fails on a *format* error here, never on the
        # path policy.
        from effgen.tools.builtin.data_analysis import DataFrameTool

        good = tmp_path / "data.csv"
        good.write_text("a,b\n1,2\n3,4\n")
        res = await DataFrameTool().execute(file_path=str(good), operation="head", n=1)
        assert res.success is True

    async def test_allowed_directories_strict_allowlist(self, tmp_path):
        # When configured, only the granted roots are readable; everything else
        # (even an ordinary temp file elsewhere) is refused.
        from effgen.tools.builtin.data_analysis import DataFrameTool

        good = tmp_path / "data.csv"
        good.write_text("a,b\n1,2\n")
        tool = DataFrameTool(allowed_directories=[str(tmp_path)])
        assert (await tool.execute(file_path=str(good), operation="head", n=1)).success

        other = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        other.write("a,b\n1,2\n")
        other.close()
        try:
            res = await tool.execute(file_path=other.name, operation="head")
            assert res.success is False
            assert "outside" in str(res.error).lower()
        finally:
            os.unlink(other.name)

    def test_confine_path_helper(self, tmp_path):
        from effgen.tools.builtin._fs import (
            PathNotAllowedError,
            confine_path,
            normalize_allowed_dirs,
        )

        # default (deny-list) posture: ordinary file ok, sensitive refused
        inside = tmp_path / "ok.txt"
        inside.write_text("x")
        assert confine_path(str(inside), None) == inside.resolve()
        with pytest.raises(PathNotAllowedError):
            confine_path("/etc/passwd", None)
        # a symlink that resolves into a denied tree is refused
        link = tmp_path / "escape"
        link.symlink_to("/etc/passwd")
        with pytest.raises(PathNotAllowedError):
            confine_path(str(link), None)

        # strict allow-list posture
        roots = normalize_allowed_dirs([str(tmp_path)])
        assert confine_path(str(inside), roots) == inside.resolve()
        with pytest.raises(PathNotAllowedError):
            confine_path("/etc/passwd", roots)
        with pytest.raises(PathNotAllowedError):
            confine_path(str(link), roots)
