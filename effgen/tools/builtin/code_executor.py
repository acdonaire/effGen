"""
Secure code execution sandbox tool.

This module provides a sandboxed code execution environment supporting
multiple languages (Python, JavaScript, Bash) with Docker isolation,
resource limits, and comprehensive security measures.

Sandbox dispatch:
  - DockerSandbox (default when Docker daemon is available):
      --read-only --network=none --cap-drop=ALL --pids-limit=100 --memory=256m
  - SubprocessSandbox (fallback): ulimit + unshare --net on Linux
  - Backend configurable via EFFGEN_SANDBOX_BACKEND=docker|subprocess
  - Timeout configurable via EFFGEN_SANDBOX_TIMEOUT=<seconds>
"""

from __future__ import annotations

import logging
from typing import Any

from ...security.sandbox import (
    SandboxBase,
    SandboxConfig,
    get_sandbox,
)
from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


class CodeExecutionError(Exception):
    """Raised when code execution fails."""
    pass


class CodeExecutor(BaseTool):
    """
    Secure sandboxed code execution tool.

    Features:
    - Multi-language support (Python, JavaScript, Bash)
    - Docker-based isolation (primary)
    - Subprocess fallback with ulimit + unshare
    - Resource limits (CPU, memory, time)
    - Network isolation (default: off)
    - Output capture (stdout, stderr, return values)
    - Comprehensive error handling

    Security:
    - Isolated execution via DockerSandbox when Docker is available
    - Configurable resource limits
    - Network restricted by default
    - Timeout mechanisms
    - Automatic cleanup

    Configuration (env vars):
    - EFFGEN_SANDBOX_BACKEND: docker | subprocess | auto (default: auto)
    - EFFGEN_SANDBOX_TIMEOUT: seconds (default: 10)
    - EFFGEN_SANDBOX_MEMORY: memory limit (default: 256m)
    - EFFGEN_SANDBOX_IMAGE: custom Docker image (default: effgen/sandbox:python-3.11)
    """

    SUPPORTED_LANGUAGES = ["python", "javascript", "bash", "sh"]

    # Default resource limits
    DEFAULT_TIMEOUT = 30           # seconds (tool-level; sandbox timeout is shorter)
    DEFAULT_MEMORY_LIMIT = "256m"  # 256 MB
    DEFAULT_MAX_OUTPUT = 102400    # 100 KB

    # Docker images for different languages (used by DockerSandbox fallback)
    DOCKER_IMAGES = {
        "python": "python:3.11-slim",
        "javascript": "node:20-slim",
        "bash": "bash:5",
        "sh": "bash:5",
    }

    def __init__(self):
        """Initialize the code executor."""
        super().__init__(
            metadata=ToolMetadata(
                name="code_executor",
                description=(
                    "Execute code in a secure sandboxed environment with support "
                    "for Python, JavaScript, and Bash. Uses Docker isolation when "
                    "available; falls back to subprocess with ulimit restrictions."
                ),
                category=ToolCategory.CODE_EXECUTION,
                parameters=[
                    ParameterSpec(
                        name="code",
                        type=ParameterType.STRING,
                        description="The code to execute",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="language",
                        type=ParameterType.STRING,
                        description="Programming language to use",
                        required=True,
                        enum=["python", "javascript", "bash", "sh"],
                    ),
                    ParameterSpec(
                        name="timeout",
                        type=ParameterType.INTEGER,
                        description="Execution timeout in seconds",
                        required=False,
                        default=30,
                        min_value=1,
                        max_value=300,
                    ),
                    ParameterSpec(
                        name="memory_limit",
                        type=ParameterType.STRING,
                        description="Memory limit (e.g., '512m', '1g')",
                        required=False,
                        default="256m",
                    ),
                    ParameterSpec(
                        name="network_enabled",
                        type=ParameterType.BOOLEAN,
                        description="Whether to allow network access (default: False)",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="files",
                        type=ParameterType.OBJECT,
                        description="Additional files to make available (filename: content)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="env_vars",
                        type=ParameterType.OBJECT,
                        description="Environment variables to set (not supported in Docker mode)",
                        required=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                        "exit_code": {"type": "integer"},
                        "execution_time": {"type": "number"},
                        "timed_out": {"type": "boolean"},
                        "sandbox_backend": {"type": "string"},
                    },
                },
                timeout_seconds=300,
                tags=["code", "execution", "sandbox", "docker", "security"],
                examples=[
                    {
                        "code": "print('Hello, World!')",
                        "language": "python",
                        "output": {
                            "stdout": "Hello, World!\n",
                            "exit_code": 0,
                            "sandbox_backend": "docker",
                        },
                    },
                    {
                        "code": "console.log('Hello from Node.js')",
                        "language": "javascript",
                        "output": {
                            "stdout": "Hello from Node.js\n",
                            "exit_code": 0,
                            "sandbox_backend": "docker",
                        },
                    },
                ],
            )
        )
        self._sandbox: SandboxBase | None = None
        self._sandbox_config: SandboxConfig = SandboxConfig.from_env()

    async def initialize(self) -> None:
        """Initialize the executor and resolve the sandbox backend."""
        await super().initialize()
        try:
            self._sandbox = await get_sandbox(self._sandbox_config)
            logger.info(
                "CodeExecutor initialized with sandbox backend: %s",
                self._sandbox.name,
            )
        except Exception as exc:
            logger.error("Failed to initialize sandbox: %s", exc)
            # Hard fail — we must not execute unsandboxed code silently
            raise RuntimeError(
                f"CodeExecutor could not initialize a sandbox backend: {exc}"
            ) from exc

    async def _execute(
        self,
        code: str,
        language: str,
        timeout: int = DEFAULT_TIMEOUT,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        network_enabled: bool = False,
        files: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute code in the configured sandbox.

        Args:
            code: Source code to execute.
            language: One of "python", "javascript", "bash", "sh".
            timeout: Wall-clock timeout in seconds (overrides sandbox default).
            memory_limit: Memory cap (e.g. "256m").
            network_enabled: Allow outbound network (default False).
            files: Extra files to write alongside the code (DockerSandbox: ignored).
            env_vars: Extra env vars (SubprocessSandbox only).

        Returns:
            Dict with stdout, stderr, exit_code, execution_time, timed_out,
            sandbox_backend.
        """
        if language not in self.SUPPORTED_LANGUAGES:
            raise CodeExecutionError(
                f"Unsupported language: {language!r}. "
                f"Supported: {self.SUPPORTED_LANGUAGES}"
            )

        # Lazily initialize if not yet done (e.g. tool used without explicit init)
        if self._sandbox is None:
            await self.initialize()

        # Build a per-call config (override from call params)
        call_config = SandboxConfig(
            backend=self._sandbox_config.backend,
            timeout=timeout,
            memory_limit=memory_limit,
            docker_image=self._sandbox_config.docker_image,
            docker_fallback_image=self._sandbox_config.docker_fallback_image,
            network_enabled=network_enabled,
        )

        try:
            result = await self._sandbox.run(
                code=code,
                language=language,
                config=call_config,
            )
        except Exception as exc:
            logger.error("Sandbox execution error: %s", exc)
            raise CodeExecutionError(f"Execution failed: {exc}") from exc

        if result.error:
            raise CodeExecutionError(f"Sandbox error: {result.error}")

        out: dict[str, Any] = {
            "stdout": result.stdout[: self.DEFAULT_MAX_OUTPUT],
            "stderr": result.stderr[: self.DEFAULT_MAX_OUTPUT],
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
            "timed_out": result.timed_out,
            "sandbox_backend": result.backend_used,
        }
        # A timed-out run is a failure, not a silent success: reflect it in the
        # honest tool envelope so the outer ToolResult.success agrees.
        if result.timed_out:
            out["success"] = False
            out["error"] = (
                f"Execution timed out after {timeout}s "
                f"(sandbox '{result.backend_used}' killed the process)"
            )
        return out
