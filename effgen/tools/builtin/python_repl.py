"""
Python REPL tool with persistent sessions.

This module provides a Python REPL (Read-Eval-Print Loop) tool with
persistent session state, safe evaluation, and comprehensive error handling.
"""

from __future__ import annotations

import ast
import builtins
import io
import logging
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


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
    - Restricted builtins (no file operations, eval, exec by default)
    - Configurable allowed imports
    - Timeout protection (inherited from base)
    - Memory limits (via base class)
    """

    # Standardized resource limits
    DEFAULT_TIMEOUT = 30          # seconds (matches ToolMetadata.timeout_seconds)
    DEFAULT_MAX_OUTPUT = 102400   # 100 KB

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

    def __init__(self):
        """Initialize the Python REPL."""
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
        self._sessions: dict[str, dict[str, Any]] = {}
        self._allowed_imports = self.DEFAULT_ALLOWED_IMPORTS.copy()

    async def initialize(self) -> None:
        """Initialize the REPL."""
        await super().initialize()
        # Create default session
        self._create_session("default")

    def _create_session(self, session_id: str) -> None:
        """Create a new session with clean namespace.

        Note: We use the same dict for both globals and locals to ensure
        that functions defined in exec() can access themselves for recursion.
        """
        namespace = {}
        self._sessions[session_id] = {
            "globals": namespace,
            "locals": namespace,  # Same dict for proper function recursion
        }

    def _get_session(self, session_id: str) -> dict[str, Any]:
        """Get or create a session."""
        if session_id not in self._sessions:
            self._create_session(session_id)
        return self._sessions[session_id]

    def _get_restricted_builtins(self) -> dict[str, Any]:
        """Get restricted builtins dictionary."""
        safe_builtins = {}
        for name in dir(builtins):
            if name not in self.RESTRICTED_BUILTINS:
                safe_builtins[name] = getattr(builtins, name)

        # Add a safe __import__ that only allows whitelisted modules
        original_import = builtins.__import__

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Safe import that only allows whitelisted modules."""
            if not self._is_import_allowed(name):
                raise ImportError(f"Import of '{name}' is not allowed")
            return original_import(name, globals, locals, fromlist, level)

        safe_builtins['__import__'] = safe_import
        return safe_builtins

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

        Returns:
            Dict containing result, stdout, stderr, error, and optionally variables
        """
        # Reset session if requested
        if reset_session:
            self._create_session(session_id)

        # Get session
        session = self._get_session(session_id)

        # Check imports and security in restricted mode
        if restricted_mode:
            import_error = self._check_imports(code)
            if import_error:
                return {
                    "success": False,
                    "result": None,
                    "stdout": "",
                    "stderr": "",
                    "error": import_error,
                    "variables": {} if return_variables else None,
                }
            security_error = self._check_security(code)
            if security_error:
                return {
                    "success": False,
                    "result": None,
                    "stdout": "",
                    "stderr": "",
                    "error": security_error,
                    "variables": {} if return_variables else None,
                }

        # Set up restricted builtins if needed
        if restricted_mode:
            session["globals"]["__builtins__"] = self._get_restricted_builtins()

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = None
        error = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to parse as expression first
                try:
                    tree = ast.parse(code, mode="eval")
                    compiled = compile(tree, "<repl>", mode="eval")
                    result = eval(
                        compiled,
                        session["globals"],
                        session["locals"],
                    )
                except SyntaxError:
                    # Not an expression, try as statements
                    tree = ast.parse(code, mode="exec")
                    compiled = compile(tree, "<repl>", mode="exec")
                    exec(
                        compiled,
                        session["globals"],
                        session["locals"],
                    )

                    # Check if last statement was a bare expression (not a
                    # function call).  Re-evaluating Call nodes like print()
                    # would execute the call a second time, producing
                    # duplicate output — BUG-011.
                    if (
                        tree.body
                        and isinstance(tree.body[-1], ast.Expr)
                        and not isinstance(tree.body[-1].value, ast.Call)
                    ):
                        last_expr = tree.body[-1].value
                        result = eval(
                            compile(
                                ast.Expression(body=last_expr),
                                "<repl>",
                                mode="eval",
                            ),
                            session["globals"],
                            session["locals"],
                        )

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.debug(f"REPL execution error: {error}")

        # Get output (truncate to max_output_size)
        stdout_value = stdout_capture.getvalue()
        stderr_value = stderr_capture.getvalue()
        if len(stdout_value) > self.DEFAULT_MAX_OUTPUT:
            stdout_value = stdout_value[:self.DEFAULT_MAX_OUTPUT] + "\n... (output truncated)"
        if len(stderr_value) > self.DEFAULT_MAX_OUTPUT:
            stderr_value = stderr_value[:self.DEFAULT_MAX_OUTPUT] + "\n... (output truncated)"

        # Prepare response. ``success`` mirrors the actual outcome so the tool
        # envelope never claims success while carrying an execution error.
        response = {
            "success": error is None,
            "result": result,
            "stdout": stdout_value,
            "stderr": stderr_value,
            "error": error,
        }

        # Include variables if requested
        if return_variables:
            # Filter out private variables and builtins
            variables = {
                k: repr(v)
                for k, v in {**session["globals"], **session["locals"]}.items()
                if not k.startswith("_") and k != "__builtins__"
            }
            response["variables"] = variables

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
        Reset a session to clean state.

        Args:
            session_id: Session to reset
        """
        if session_id in self._sessions:
            self._create_session(session_id)

    def get_session_variables(self, session_id: str) -> dict[str, str]:
        """
        Get all variables in a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict mapping variable names to their string representations
        """
        if session_id not in self._sessions:
            return {}

        session = self._sessions[session_id]
        return {
            k: repr(v)
            for k, v in {**session["globals"], **session["locals"]}.items()
            if not k.startswith("_") and k != "__builtins__"
        }

    async def cleanup(self) -> None:
        """Clean up all sessions."""
        self._sessions.clear()
        await super().cleanup()
