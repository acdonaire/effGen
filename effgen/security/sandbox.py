"""
effgen.security.sandbox — Sandbox backends for CodeExecutor.

Threat model: attacker-controlled prompt → agent emits arbitrary Python/shell →
execution must be isolated from the host.

Backends (in order of preference):
  DockerSandbox     — Docker container with --read-only --network=none --cap-drop=ALL
                      --pids-limit=100 --memory=256m. Default when Docker is reachable.
  SubprocessSandbox — fork/exec with ulimit + unshare(1) on Linux. Best-effort.
                      Used when Docker is unavailable; emits a loud warning.
  FirecrackerSandbox — Stub only; not implemented in this release.

Selection:
  Auto (default): DockerSandbox if Docker daemon is reachable; SubprocessSandbox + warning otherwise.
  Override via env var EFFGEN_SANDBOX_BACKEND=docker|subprocess|firecracker|off
  Timeout via EFFGEN_SANDBOX_TIMEOUT=<seconds>  (default 10)

  EFFGEN_SANDBOX_BACKEND=off disables sandboxing entirely. This is UNSAFE: code
  runs directly on the host with the privileges of the effGen process. A loud
  warning is emitted whenever the off backend is used. Intended only for trusted
  code in controlled environments.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SandboxConfig:
    """Environment-driven sandbox configuration."""

    backend: str = field(
        default_factory=lambda: os.environ.get("EFFGEN_SANDBOX_BACKEND", "auto")
    )
    timeout: int = field(
        default_factory=lambda: int(os.environ.get("EFFGEN_SANDBOX_TIMEOUT", "10"))
    )
    memory_limit: str = field(
        default_factory=lambda: os.environ.get("EFFGEN_SANDBOX_MEMORY", "256m")
    )
    # Docker-specific
    docker_image: str = field(
        default_factory=lambda: os.environ.get(
            "EFFGEN_SANDBOX_IMAGE", "effgen/sandbox:python-3.11"
        )
    )
    # Fallback image if custom image not available
    docker_fallback_image: str = "python:3.11-slim"
    network_enabled: bool = False

    @classmethod
    def from_env(cls) -> "SandboxConfig":
        """Create config from environment variables."""
        return cls()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Result of a sandboxed code execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    timed_out: bool = False
    backend_used: str = ""
    error: str | None = None


def _killed_at_deadline(exit_code: int | None, elapsed: float, timeout: float) -> bool:
    """True when a process was killed by a signal at/after its time budget.

    The subprocess backends enforce the wall-clock timeout two ways: an outer
    ``asyncio.wait_for`` (raises ``TimeoutError``) and an inner CPU-time
    ``ulimit -t`` that ``SIGXCPU``/``SIGKILL``s a busy loop *before* the outer
    guard fires. A negative exit code (signal kill) that lands at or after the
    requested deadline is therefore a timeout, not a clean exit — report it
    explicitly instead of as a silent success. (An OOM from ``ulimit -v`` surfaces
    as a Python ``MemoryError`` with a positive exit code, so it is not matched
    here.)
    """
    if exit_code is None or exit_code >= 0 or timeout <= 0:
        return False
    return elapsed >= timeout


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SandboxBase(ABC):
    """Abstract sandbox backend interface."""

    name: str = "base"

    @abstractmethod
    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> SandboxResult:
        """Execute *code* in the sandbox and return a SandboxResult."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if this backend can be used in the current environment."""

    # ------------------------------------------------------------------
    # Helpers shared by all backends
    # ------------------------------------------------------------------

    _CODE_FILENAMES: dict[str, str] = {
        "python": "script.py",
        "javascript": "script.js",
        "bash": "script.sh",
        "sh": "script.sh",
    }

    _EXEC_COMMANDS: dict[str, list[str]] = {
        "python": ["python3", "script.py"],
        "javascript": ["node", "script.js"],
        "bash": ["bash", "script.sh"],
        "sh": ["sh", "script.sh"],
    }

    def _code_filename(self, language: str) -> str:
        return self._CODE_FILENAMES.get(language, "script.txt")

    def _exec_command(self, language: str) -> list[str]:
        return self._EXEC_COMMANDS.get(language, ["bash", "script.sh"])


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------

_DOCKER_IMAGES: dict[str, str] = {
    "python": "python:3.11-slim",
    "javascript": "node:20-slim",
    "bash": "bash:5",
    "sh": "bash:5",
}


class DockerSandbox(SandboxBase):
    """
    Run code inside a Docker container with strict isolation:

    - ``--read-only``         container FS is read-only (tmpfs for /tmp)
    - ``--network=none``      no outbound network connectivity
    - ``--cap-drop=ALL``      all Linux capabilities dropped
    - ``--no-new-privileges`` prevent privilege escalation via setuid/setgid
    - ``--pids-limit=100``    cap process/thread count
    - ``--memory=<limit>``    memory cap (default 256m)
    - ``--rm``                auto-remove container on exit

    The sandbox image is configurable via ``EFFGEN_SANDBOX_IMAGE``; falls back
    to the stock ``python:3.11-slim`` (or language-appropriate image) if the
    custom image is not available.
    """

    name = "docker"

    async def is_available(self) -> bool:
        """Return True if Docker daemon is reachable."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except (FileNotFoundError, PermissionError, OSError):
            return False

    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> SandboxResult:
        start = time.monotonic()
        tmp_dir = Path(tempfile.mkdtemp(prefix="effgen_sandbox_"))
        try:
            # Write code to temp dir
            fname = self._code_filename(language)
            script_path = tmp_dir / fname
            script_path.write_text(code, encoding="utf-8")

            # The container process may run under a *different* uid than this
            # process — e.g. a Docker daemon with userns-remap, or an image whose
            # default USER is non-root. ``mkdtemp`` makes the dir 0700 and the
            # file 0600 (owner-only), so that uid cannot traverse the dir or read
            # the bind-mounted script and Python aborts with "[Errno 13]
            # Permission denied". Make the ephemeral, read-only-mounted code dir
            # and file world-readable/traversable so any container uid can read
            # them. Safe: the dir is a per-run temp holding only the user's own
            # code, is mounted read-only, and is removed in the finally below.
            os.chmod(tmp_dir, 0o755)
            os.chmod(script_path, 0o644)

            # Decide image
            image = await self._resolve_image(language, config)

            # Build docker run command
            cmd = self._build_docker_cmd(
                tmp_dir=tmp_dir,
                image=image,
                language=language,
                config=config,
            )

            logger.debug("DockerSandbox cmd: %s", " ".join(cmd))

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=config.timeout + 5
                )
                exit_code = proc.returncode
                timed_out = False
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                stdout_b, stderr_b = b"", b"Execution timed out"
                exit_code = -1
                timed_out = True

            return SandboxResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=exit_code,
                execution_time=time.monotonic() - start,
                timed_out=timed_out,
                backend_used="docker",
            )

        except Exception as exc:
            logger.error("DockerSandbox error: %s", exc)
            return SandboxResult(
                exit_code=-1,
                execution_time=time.monotonic() - start,
                backend_used="docker",
                error=str(exc),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_image(self, language: str, config: SandboxConfig) -> str:
        """
        Pick the container image.

        The custom sandbox image (``effgen/sandbox:python-3.11``) only ships a
        Python runtime, so it is used solely for ``python``. Other languages
        always use their language-appropriate stock image.
        """
        if language == "python":
            custom = config.docker_image
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker", "image", "inspect", custom,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
                if proc.returncode == 0:
                    return custom
            except Exception:
                pass
        # Fall back to language-appropriate stock image
        return _DOCKER_IMAGES.get(language, "python:3.11-slim")

    def _build_docker_cmd(
        self,
        tmp_dir: Path,
        image: str,
        language: str,
        config: SandboxConfig,
    ) -> list[str]:
        """Assemble the ``docker run`` command."""
        exec_cmd = self._exec_command(language)

        cmd: list[str] = [
            "docker", "run",
            "--rm",
            # Filesystem isolation
            "--read-only",
            "--tmpfs", "/tmp:size=64m,noexec",
            # Mount code (read-only)
            "-v", f"{tmp_dir}:/workspace:ro",
            "-w", "/workspace",
            # Security
            "--network", "none" if not config.network_enabled else "bridge",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            # Resource limits
            "--pids-limit", "100",
            "--memory", config.memory_limit,
            "--memory-swap", config.memory_limit,  # disable swap
            "--cpus", "1",
        ]

        cmd.append(image)
        cmd.extend(exec_cmd)
        return cmd


# ---------------------------------------------------------------------------
# SubprocessSandbox
# ---------------------------------------------------------------------------

class SubprocessSandbox(SandboxBase):
    """
    Best-effort isolation via subprocess with resource limits.

    On Linux, execution is wrapped in an *unprivileged* user namespace via
    ``unshare --map-root-user`` combined with:

    - ``--net``   a fresh network namespace with no interfaces → outbound
                  network is blocked (DNS + connect fail).
    - ``--mount`` a private mount namespace in which a fresh ``tmpfs`` is
                  mounted over ``/tmp``, so writes/deletes under ``/tmp``
                  (e.g. ``rm -rf /tmp/evil``) never reach the host.
    - ``ulimit``  address-space, CPU-time and process-count caps.

    Code is fed to the interpreter over **stdin** (``python3 -`` / ``bash -s``),
    so no script file is written to the host filesystem.

    Capability probing happens once per process. Each isolation primitive
    degrades independently: if user namespaces are unavailable the sandbox
    falls back to ``ulimit``-only mode and emits a warning describing exactly
    which protections are NOT in effect.

    **Caveats / limitations:**
    - Filesystem isolation only shields ``/tmp`` and namespace-private mounts;
      the rest of the host FS is still *readable* (not writable as root inside
      the user namespace, but readable). Use DockerSandbox for full isolation.
    - Requires unprivileged user namespaces to be enabled
      (``kernel.unprivileged_userns_clone=1`` / ``user.max_user_namespaces>0``).
    - Memory limit is advisory (via ``ulimit -v``), not hard-enforced like cgroups.
    - Should only be used when Docker is unavailable.

    A loud warning is emitted at startup when this backend is selected as the
    default.
    """

    name = "subprocess"

    # Executables and the flag that makes each read program text from stdin.
    _LANG_STDIN_CMD: dict[str, list[str]] = {
        "python": ["python3", "-"],
        "javascript": ["node", "-"],
        "bash": ["bash", "-s"],
        "sh": ["sh", "-s"],
    }

    # Probed once per process: does ``unshare`` support each isolation mode?
    _caps_probed: bool = False
    _userns_ok: bool = False
    _mountns_ok: bool = False

    async def is_available(self) -> bool:
        """Always available (subprocess is a Python built-in)."""
        return True

    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> SandboxResult:
        start = time.monotonic()
        try:
            cmd, net_isolated, fs_isolated = await self._build_cmd(language, config)
            logger.debug(
                "SubprocessSandbox cmd: %s (net=%s, fs=%s)",
                " ".join(cmd), net_isolated, fs_isolated,
            )

            env = self._build_env()

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=code.encode("utf-8")),
                    timeout=config.timeout + 2,
                )
                exit_code = proc.returncode
                elapsed = time.monotonic() - start
                # The CPU-time ulimit can SIGKILL a busy loop before the outer
                # wait_for fires; surface that as a timeout, not a clean exit.
                timed_out = _killed_at_deadline(exit_code, elapsed, config.timeout)
                if timed_out and not stderr_b:
                    stderr_b = b"Execution timed out"
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                stdout_b, stderr_b = b"", b"Execution timed out"
                exit_code = -1
                timed_out = True

            return SandboxResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=exit_code if exit_code is not None else -1,
                execution_time=time.monotonic() - start,
                timed_out=timed_out,
                backend_used="subprocess",
            )
        except Exception as exc:
            logger.error("SubprocessSandbox error: %s", exc)
            return SandboxResult(
                exit_code=-1,
                execution_time=time.monotonic() - start,
                backend_used="subprocess",
                error=str(exc),
            )

    # ------------------------------------------------------------------

    async def _build_cmd(
        self, language: str, config: SandboxConfig
    ) -> tuple[list[str], bool, bool]:
        """
        Assemble the execution command.

        Returns:
            (cmd_list, net_isolated, fs_isolated)
        """
        stdin_cmd = self._LANG_STDIN_CMD.get(language, ["python3", "-"])
        interpreter = " ".join(stdin_cmd)
        mem_kb = self._parse_mem_kb(config.memory_limit)
        ulimit_prefix = (
            f"ulimit -v {mem_kb} -t {config.timeout} -u 256 2>/dev/null; "
        )

        if platform.system() != "Linux":
            return ["bash", "-c", f"{ulimit_prefix}exec {interpreter}"], False, False

        await self._probe_caps()
        unshare_bin = shutil.which("unshare")

        if unshare_bin and self._userns_ok:
            # Build the inner shell: mount a private tmpfs over /tmp when the
            # mount namespace is available (we are root inside the user ns),
            # then apply ulimits and exec the interpreter reading from stdin.
            want_net = not config.network_enabled
            inner = ulimit_prefix
            if self._mountns_ok:
                # /tmp gets a fresh tmpfs so host /tmp is untouched.
                inner += "mount -t tmpfs none /tmp 2>/dev/null; "
            inner += f"exec {interpreter}"

            unshare_flags = [unshare_bin, "--map-root-user"]
            if self._mountns_ok:
                unshare_flags.append("--mount")
            if want_net:
                unshare_flags.append("--net")
            cmd = unshare_flags + ["bash", "-c", inner]
            return cmd, want_net, self._mountns_ok

        # Degraded: no user namespaces. ulimit-only, no net/fs isolation.
        logger.warning(
            "SubprocessSandbox: unprivileged user namespaces unavailable; "
            "running in ulimit-only mode. Network is NOT isolated and "
            "filesystem writes are NOT confined. Install Docker or enable "
            "kernel.unprivileged_userns_clone for stronger isolation."
        )
        return ["bash", "-c", f"{ulimit_prefix}exec {interpreter}"], False, False

    @classmethod
    async def _probe_caps(cls) -> None:
        """Probe (once) whether unprivileged user/mount namespaces work."""
        if cls._caps_probed:
            return
        cls._userns_ok = await cls._unshare_succeeds(["--map-root-user", "true"])
        cls._mountns_ok = await cls._unshare_succeeds(
            ["--map-root-user", "--mount", "true"]
        )
        cls._caps_probed = True

    @staticmethod
    async def _unshare_succeeds(args: list[str]) -> bool:
        """Return True if ``unshare <args>`` exits 0 in this environment."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "unshare", *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _parse_mem_kb(mem_str: str) -> int:
        """Convert Docker-style memory string (e.g. '256m', '1g') to KB."""
        mem_str = mem_str.strip().lower()
        if mem_str.endswith("g"):
            return int(float(mem_str[:-1]) * 1024 * 1024)
        if mem_str.endswith("m"):
            return int(float(mem_str[:-1]) * 1024)
        if mem_str.endswith("k"):
            return int(float(mem_str[:-1]))
        return int(mem_str)  # assume bytes, convert to KB approximately

    @staticmethod
    def _build_env() -> dict[str, str]:
        """Return a minimal environment stripping sensitive vars."""
        safe_keys = {
            "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE",
            "TERM", "USER", "LOGNAME", "SHELL",
            "TZ", "TMPDIR", "TEMP", "TMP",
            "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
        }
        env = {k: v for k, v in os.environ.items() if k in safe_keys}
        # Harden Python
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env


# ---------------------------------------------------------------------------
# FirecrackerSandbox (stub)
# ---------------------------------------------------------------------------

class FirecrackerSandbox(SandboxBase):
    """
    Stub interface for Firecracker MicroVM isolation.

    **Not implemented in this release.**

    Firecracker provides VM-level isolation with sub-second boot times,
    making it suitable for multi-tenant code execution with stronger guarantees
    than Docker containers. Implementation requires:
    - Firecracker binary + KVM access on the host
    - A minimal guest rootfs (e.g., Alpine-based initrd)
    - jailer for additional isolation
    - vsock or REST API for code injection + result retrieval

    This stub exists to define the interface and allow future implementation
    without breaking the sandbox dispatch machinery.
    """

    name = "firecracker"

    async def is_available(self) -> bool:
        """Not yet implemented — always returns False."""
        return False

    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> SandboxResult:
        raise NotImplementedError(
            "FirecrackerSandbox is not yet implemented in this release. "
            "Use 'docker' or 'subprocess' backend instead."
        )


# ---------------------------------------------------------------------------
# OffSandbox (explicit, unsafe)
# ---------------------------------------------------------------------------

class OffSandbox(SandboxBase):
    """
    NO sandbox. Executes code directly on the host with the privileges of the
    effGen process.

    **This is unsafe and must only be enabled explicitly** via
    ``EFFGEN_SANDBOX_BACKEND=off``. It is never selected by auto-resolution.
    A loud warning is emitted on every execution.

    Code is fed to the interpreter over stdin; ``ulimit`` caps are still applied
    on Linux as a courtesy, but provide no real isolation.
    """

    name = "off"

    _warned: bool = False

    async def is_available(self) -> bool:
        """Always available — but unsafe."""
        return True

    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> SandboxResult:
        if not OffSandbox._warned:
            logger.warning(
                "\n"
                "┌─────────────────────────────────────────────────────────────┐\n"
                "│  effGen SANDBOX DISABLED (EFFGEN_SANDBOX_BACKEND=off)         │\n"
                "│                                                              │\n"
                "│  Code is executing DIRECTLY ON THE HOST with no isolation.   │\n"
                "│  Network, filesystem, and resources are NOT restricted.      │\n"
                "│  Only use this for fully trusted code in controlled envs.    │\n"
                "└─────────────────────────────────────────────────────────────┘"
            )
            OffSandbox._warned = True

        start = time.monotonic()
        stdin_cmd = SubprocessSandbox._LANG_STDIN_CMD.get(language, ["python3", "-"])
        interpreter = " ".join(stdin_cmd)
        if platform.system() == "Linux":
            mem_kb = SubprocessSandbox._parse_mem_kb(config.memory_limit)
            cmd = [
                "bash", "-c",
                f"ulimit -v {mem_kb} -t {config.timeout} -u 256 2>/dev/null; "
                f"exec {interpreter}",
            ]
        else:
            cmd = ["bash", "-c", f"exec {interpreter}"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=code.encode("utf-8")),
                    timeout=config.timeout + 2,
                )
                exit_code = proc.returncode
                elapsed = time.monotonic() - start
                timed_out = _killed_at_deadline(exit_code, elapsed, config.timeout)
                if timed_out and not stderr_b:
                    stderr_b = b"Execution timed out"
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout_b, stderr_b = b"", b"Execution timed out"
                exit_code = -1
                timed_out = True

            return SandboxResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=exit_code if exit_code is not None else -1,
                execution_time=time.monotonic() - start,
                timed_out=timed_out,
                backend_used="off",
            )
        except Exception as exc:
            logger.error("OffSandbox error: %s", exc)
            return SandboxResult(
                exit_code=-1,
                execution_time=time.monotonic() - start,
                backend_used="off",
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Sandbox factory / dispatch
# ---------------------------------------------------------------------------

_BACKEND_MAP: dict[str, type[SandboxBase]] = {
    "docker": DockerSandbox,
    "subprocess": SubprocessSandbox,
    "firecracker": FirecrackerSandbox,
    "off": OffSandbox,
}

# Cache the resolved backend so availability checks run once per process
_resolved_backend: SandboxBase | None = None


async def get_sandbox(config: SandboxConfig | None = None) -> SandboxBase:
    """
    Return the best available sandbox backend.

    Resolution order when ``config.backend == "auto"`` (the default):

    1. DockerSandbox — preferred; strong isolation.
    2. SubprocessSandbox — fallback; emits ``WARNING`` on first use.

    The resolved backend is cached for the lifetime of the process.
    """
    global _resolved_backend

    if config is None:
        config = SandboxConfig.from_env()

    if _resolved_backend is not None and config.backend == "auto":
        return _resolved_backend

    backend_name = config.backend.lower()

    if backend_name != "auto":
        cls = _BACKEND_MAP.get(backend_name)
        if cls is None:
            raise ValueError(
                f"Unknown EFFGEN_SANDBOX_BACKEND={backend_name!r}. "
                f"Valid choices: {sorted(_BACKEND_MAP)}"
            )
        backend = cls()
        if not await backend.is_available():
            raise RuntimeError(
                f"Sandbox backend {backend_name!r} is not available in this environment."
            )
        _resolved_backend = backend
        return backend

    # Auto-selection
    docker = DockerSandbox()
    if await docker.is_available():
        logger.info("Sandbox: using DockerSandbox (--network=none --cap-drop=ALL).")
        _resolved_backend = docker
        return docker

    # Fallback
    logger.warning(
        "\n"
        "┌─────────────────────────────────────────────────────────────┐\n"
        "│  effGen SANDBOX WARNING                                      │\n"
        "│                                                              │\n"
        "│  Docker is not available. Code execution will use            │\n"
        "│  SubprocessSandbox, which provides PARTIAL isolation only.   │\n"
        "│                                                              │\n"
        "│  Limitations:                                                │\n"
        "│  • No filesystem isolation (code can read host FS)           │\n"
        "│  • Network isolation via unshare (may require privileges)    │\n"
        "│  • Memory limit is advisory, not hard-enforced               │\n"
        "│                                                              │\n"
        "│  To enable full isolation, install Docker and ensure the     │\n"
        "│  daemon is running and accessible by the current user.       │\n"
        "│  Set EFFGEN_SANDBOX_BACKEND=subprocess to silence this.      │\n"
        "└─────────────────────────────────────────────────────────────┘\n"
    )
    sub = SubprocessSandbox()
    _resolved_backend = sub
    return sub


def reset_sandbox_cache() -> None:
    """Reset the cached sandbox backend (useful for testing)."""
    global _resolved_backend
    _resolved_backend = None


__all__ = [
    "SandboxConfig",
    "SandboxResult",
    "SandboxBase",
    "DockerSandbox",
    "SubprocessSandbox",
    "FirecrackerSandbox",
    "OffSandbox",
    "get_sandbox",
    "reset_sandbox_cache",
]
