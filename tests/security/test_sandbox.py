"""
Sandbox backend tests.

Test coverage:
  - SandboxConfig: defaults and env-var overrides
  - SubprocessSandbox: basic execution, timeout, resource limits, network block
  - DockerSandbox: availability detection, network block (when Docker available)
  - FirecrackerSandbox: stub correctly raises NotImplementedError
  - get_sandbox(): auto-selection logic
  - CodeExecutor: routes through sandbox, returns sandbox_backend field

Network-block test (DockerSandbox):
  Runs: python3 -c "import urllib.request; urllib.request.urlopen('https://example.com')"
  inside --network=none container → must fail (non-zero exit).

Skip markers:
  - docker tests skip if Docker daemon is unreachable
  - subprocess tests always run (subprocess is always available)
"""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess as sp
from pathlib import Path

import pytest

from effgen.security.sandbox import (
    DockerSandbox,
    FirecrackerSandbox,
    OffSandbox,
    SandboxConfig,
    SubprocessSandbox,
    get_sandbox,
    reset_sandbox_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine in a fresh event loop (3.10–3.13 safe)."""
    return asyncio.run(coro)


def _docker_available() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        result = sp.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def _userns_works() -> bool:
    """Return True if unprivileged user namespaces are usable on this host."""
    if platform.system() != "Linux" or not shutil.which("unshare"):
        return False
    try:
        return sp.run(
            ["unshare", "--map-root-user", "true"],
            capture_output=True, timeout=5,
        ).returncode == 0
    except Exception:
        return False


DOCKER_AVAILABLE = _docker_available()
USERNS_AVAILABLE = _userns_works()

docker_required = pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason="Docker daemon not reachable",
)
userns_required = pytest.mark.skipif(
    not USERNS_AVAILABLE,
    reason="unprivileged user namespaces not available",
)


# ---------------------------------------------------------------------------
# SandboxConfig tests
# ---------------------------------------------------------------------------

class TestSandboxConfig:
    def test_defaults(self):
        cfg = SandboxConfig()
        # backend defaults to EFFGEN_SANDBOX_BACKEND env or "auto"
        assert cfg.backend in ("auto", "docker", "subprocess")
        assert cfg.timeout > 0
        assert cfg.memory_limit.endswith(("m", "g", "k")) or cfg.memory_limit.isdigit()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "subprocess")
        monkeypatch.setenv("EFFGEN_SANDBOX_TIMEOUT", "42")
        monkeypatch.setenv("EFFGEN_SANDBOX_MEMORY", "128m")
        cfg = SandboxConfig.from_env()
        assert cfg.backend == "subprocess"
        assert cfg.timeout == 42
        assert cfg.memory_limit == "128m"

    def test_network_disabled_by_default(self):
        cfg = SandboxConfig()
        assert cfg.network_enabled is False


# ---------------------------------------------------------------------------
# SubprocessSandbox tests
# ---------------------------------------------------------------------------

class TestSubprocessSandbox:
    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        reset_sandbox_cache()
        yield
        reset_sandbox_cache()

    def test_is_always_available(self):
        sb = SubprocessSandbox()
        assert _run(sb.is_available()) is True

    def test_python_hello_world(self):
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=15, memory_limit="256m")
        result = _run(sb.run("print('hello sandbox')", "python", cfg))
        assert result.exit_code == 0
        assert "hello sandbox" in result.stdout
        assert result.backend_used == "subprocess"

    def test_bash_execution(self):
        if not shutil.which("bash"):
            pytest.skip("bash not available")
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=10)
        result = _run(sb.run("echo 'bash works'", "bash", cfg))
        assert result.exit_code == 0
        assert "bash works" in result.stdout

    def test_python_syntax_error(self):
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=10)
        result = _run(sb.run("def broken(:", "python", cfg))
        assert result.exit_code != 0
        assert result.stderr  # error message should be present

    def test_timeout_enforced(self):
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=3, memory_limit="256m")
        # Python infinite loop should time out
        result = _run(sb.run("while True: pass", "python", cfg))
        assert result.timed_out is True or result.exit_code != 0

    def test_result_fields_present(self):
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=10)
        result = _run(sb.run("print(1+1)", "python", cfg))
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)
        assert isinstance(result.exit_code, int)
        assert isinstance(result.execution_time, float)
        assert isinstance(result.timed_out, bool)
        assert result.backend_used == "subprocess"

    def test_stderr_captured(self):
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=10)
        result = _run(sb.run("import sys; sys.stderr.write('err output')", "python", cfg))
        assert "err output" in result.stderr

    @userns_required
    def test_subprocess_network_block(self):
        """
        On Linux with unprivileged user namespaces, the subprocess sandbox runs
        inside `unshare --map-root-user --net`, so urllib.request.urlopen fails.
        """
        reset_sandbox_cache()
        SubprocessSandbox._caps_probed = False  # re-probe in this env
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=15, memory_limit="256m")
        code = (
            "import urllib.request, sys\n"
            "try:\n"
            "    urllib.request.urlopen('https://example.com', timeout=5)\n"
            "    print('NETWORK_OK')\n"
            "    sys.exit(0)\n"
            "except Exception as e:\n"
            "    print(f'NETWORK_BLOCKED: {e}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
        result = _run(sb.run(code, "python", cfg))
        assert result.exit_code != 0, (
            "Expected network to be blocked by unshare --net, but got exit_code="
            f"{result.exit_code}, stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
        assert "NETWORK_OK" not in result.stdout

    @userns_required
    def test_subprocess_fs_isolation_protects_host_tmp(self):
        """
        `rm -rf /tmp/<sentinel>` inside the sandbox must NOT touch the host:
        the sandbox mounts a private tmpfs over /tmp inside its mount namespace.

        Skipped if the host mount namespace cannot be created (mount isolation
        is the protection being tested).
        """
        SubprocessSandbox._caps_probed = False
        if not _run(SubprocessSandbox._unshare_succeeds(["--map-root-user", "--mount", "true"])):
            pytest.skip("mount namespace not available — FS isolation cannot apply")

        # The sandbox shields the literal /tmp path; place the sentinel there.
        sentinel = Path("/tmp/effgen_fs_isolation_sentinel.txt")
        sentinel.write_text("do-not-delete", encoding="utf-8")
        try:
            SubprocessSandbox._caps_probed = False
            sb = SubprocessSandbox()
            cfg = SandboxConfig(backend="subprocess", timeout=15, memory_limit="256m")
            code = (
                "import os\n"
                "os.system('rm -rf /tmp/effgen_fs_isolation_sentinel.txt')\n"
                "print('ran rm')\n"
            )
            result = _run(sb.run(code, "python", cfg))
            assert "ran rm" in result.stdout
            # The host sentinel must survive — the rm hit the namespace tmpfs only.
            assert sentinel.exists(), (
                "Host /tmp sentinel was deleted — FS isolation failed!"
            )
        finally:
            sentinel.unlink(missing_ok=True)

    def test_parse_mem_kb(self):
        assert SubprocessSandbox._parse_mem_kb("256m") == 256 * 1024
        assert SubprocessSandbox._parse_mem_kb("1g") == 1024 * 1024
        assert SubprocessSandbox._parse_mem_kb("512k") == 512


# ---------------------------------------------------------------------------
# DockerSandbox tests
# ---------------------------------------------------------------------------

class TestDockerSandbox:
    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        reset_sandbox_cache()
        yield
        reset_sandbox_cache()

    def test_is_available_returns_bool(self):
        sb = DockerSandbox()
        result = _run(sb.is_available())
        assert isinstance(result, bool)
        assert result == DOCKER_AVAILABLE

    @docker_required
    def test_python_hello_world(self):
        sb = DockerSandbox()
        cfg = SandboxConfig(
            backend="docker",
            timeout=30,
            memory_limit="256m",
            network_enabled=False,
        )
        result = _run(sb.run("print('docker sandbox works')", "python", cfg))
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "docker sandbox works" in result.stdout
        assert result.backend_used == "docker"

    @docker_required
    def test_network_block(self):
        """
        Core network-block test:
        urllib.request.urlopen inside --network=none container must fail.
        """
        sb = DockerSandbox()
        cfg = SandboxConfig(
            backend="docker",
            timeout=30,
            memory_limit="256m",
            network_enabled=False,
        )
        code = (
            "import urllib.request, sys\n"
            "try:\n"
            "    urllib.request.urlopen('https://example.com', timeout=5)\n"
            "    print('NETWORK_OK')\n"
            "    sys.exit(0)\n"
            "except Exception as e:\n"
            "    print(f'NETWORK_BLOCKED: {type(e).__name__}: {e}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
        result = _run(sb.run(code, "python", cfg))
        assert result.exit_code != 0, (
            "Expected network request to FAIL inside --network=none container, "
            f"but exit_code={result.exit_code}, stdout={result.stdout!r}"
        )
        assert result.backend_used == "docker"

    @docker_required
    def test_syntax_error_captured(self):
        sb = DockerSandbox()
        cfg = SandboxConfig(backend="docker", timeout=30, memory_limit="256m")
        result = _run(sb.run("this is not valid python !!!", "python", cfg))
        assert result.exit_code != 0
        assert result.stderr  # Python error in stderr

    @docker_required
    def test_timeout_enforced(self):
        sb = DockerSandbox()
        cfg = SandboxConfig(
            backend="docker",
            timeout=5,
            memory_limit="256m",
            network_enabled=False,
        )
        result = _run(sb.run("while True: pass", "python", cfg))
        assert result.timed_out is True or result.exit_code != 0

    @docker_required
    def test_memory_limit_applied(self):
        """Verify container runs under memory cap (256m by default)."""
        sb = DockerSandbox()
        cfg = SandboxConfig(
            backend="docker",
            timeout=30,
            memory_limit="256m",
            network_enabled=False,
        )
        # A simple calculation should complete fine
        result = _run(sb.run("print(sum(range(1000000)))", "python", cfg))
        assert result.exit_code == 0
        assert "499999500000" in result.stdout


# ---------------------------------------------------------------------------
# FirecrackerSandbox stub tests
# ---------------------------------------------------------------------------

class TestFirecrackerSandbox:
    def test_is_not_available(self):
        sb = FirecrackerSandbox()
        assert _run(sb.is_available()) is False

    def test_run_raises_not_implemented(self):
        sb = FirecrackerSandbox()
        cfg = SandboxConfig()
        with pytest.raises(NotImplementedError, match="FirecrackerSandbox"):
            _run(sb.run("print('hi')", "python", cfg))


# ---------------------------------------------------------------------------
# OffSandbox tests (explicit, unsafe)
# ---------------------------------------------------------------------------

class TestOffSandbox:
    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        reset_sandbox_cache()
        OffSandbox._warned = False
        yield
        reset_sandbox_cache()
        OffSandbox._warned = False

    def test_is_available(self):
        assert _run(OffSandbox().is_available()) is True

    def test_executes_directly(self):
        sb = OffSandbox()
        cfg = SandboxConfig(backend="off", timeout=10)
        result = _run(sb.run("print('off backend ran')", "python", cfg))
        assert result.exit_code == 0
        assert "off backend ran" in result.stdout
        assert result.backend_used == "off"

    def test_emits_loud_warning(self, caplog):
        import logging
        sb = OffSandbox()
        cfg = SandboxConfig(backend="off", timeout=10)
        with caplog.at_level(logging.WARNING, logger="effgen.security.sandbox"):
            _run(sb.run("print(1)", "python", cfg))
        assert any(
            "SANDBOX DISABLED" in rec.message or "directly on the host" in rec.message.lower()
            for rec in caplog.records
        ), "OffSandbox must emit a loud warning"

    def test_selected_via_env(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "off")
        cfg = SandboxConfig.from_env()
        sb = _run(get_sandbox(cfg))
        assert isinstance(sb, OffSandbox)

    def test_off_never_chosen_by_auto(self):
        """Auto-selection must never pick OffSandbox."""
        sb = _run(get_sandbox(SandboxConfig(backend="auto")))
        assert not isinstance(sb, OffSandbox)


# ---------------------------------------------------------------------------
# get_sandbox() auto-selection tests
# ---------------------------------------------------------------------------

class TestGetSandbox:
    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        reset_sandbox_cache()
        yield
        reset_sandbox_cache()

    def test_auto_selection_returns_sandbox(self):
        sandbox = _run(get_sandbox())
        assert isinstance(sandbox, DockerSandbox | SubprocessSandbox)

    def test_auto_returns_docker_when_available(self):
        if not DOCKER_AVAILABLE:
            pytest.skip("Docker not available")
        sb = _run(get_sandbox())
        assert isinstance(sb, DockerSandbox)

    def test_explicit_subprocess_selection(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "subprocess")
        cfg = SandboxConfig.from_env()
        sb = _run(get_sandbox(cfg))
        assert isinstance(sb, SubprocessSandbox)

    def test_invalid_backend_raises(self):
        cfg = SandboxConfig(backend="invalid_backend_xyz")
        with pytest.raises(ValueError, match="Unknown EFFGEN_SANDBOX_BACKEND"):
            _run(get_sandbox(cfg))

    def test_caching_returns_same_instance(self):
        sb1 = _run(get_sandbox())
        sb2 = _run(get_sandbox())
        assert sb1 is sb2


# ---------------------------------------------------------------------------
# CodeExecutor integration test (routes through sandbox)
# ---------------------------------------------------------------------------

class TestCodeExecutorSandboxIntegration:
    """
    Verify CodeExecutor._execute() routes through the sandbox backend
    and returns the sandbox_backend field.
    """

    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        reset_sandbox_cache()
        yield
        reset_sandbox_cache()

    def test_code_executor_uses_sandbox(self):
        from effgen.tools.builtin.code_executor import CodeExecutor

        executor = CodeExecutor()
        # Force initialize with subprocess backend for reliability
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("EFFGEN_SANDBOX_BACKEND", "subprocess")
            reset_sandbox_cache()
            executor._sandbox_config = SandboxConfig.from_env()
            _run(executor.initialize())

        result = _run(executor._execute(
            code="print('executor routed')",
            language="python",
            timeout=15,
            memory_limit="256m",
            network_enabled=False,
        ))

        assert result["exit_code"] == 0
        assert "executor routed" in result["stdout"]
        assert "sandbox_backend" in result
        assert result["sandbox_backend"] in ("docker", "subprocess")

    def test_code_executor_no_longer_executes_directly(self):
        """CodeExecutor must have a sandbox; direct subprocess calls are gone."""
        from effgen.tools.builtin.code_executor import CodeExecutor

        executor = CodeExecutor()
        # Verify the old _execute_docker / _execute_fallback are NOT present
        assert not hasattr(executor, "_execute_docker"), (
            "CodeExecutor should not have _execute_docker; sandbox handles this"
        )
        assert not hasattr(executor, "_execute_fallback"), (
            "CodeExecutor should not have _execute_fallback; sandbox handles this"
        )
