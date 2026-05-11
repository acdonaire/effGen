"""
Base tool interface for the effGen framework.

This module provides the abstract base class that all tools must inherit from,
ensuring consistent interfaces for tool metadata, parameter validation, and execution.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ToolCategory(Enum):
    """Categories for organizing tools."""
    INFORMATION_RETRIEVAL = "information_retrieval"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATIONS = "file_operations"
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    DATA_PROCESSING = "data_processing"
    SYSTEM = "system"
    EXTERNAL_API = "external_api"


class ParameterType(Enum):
    """Supported parameter types for tool inputs."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


@dataclass
class ParameterSpec:
    """Specification for a tool parameter."""
    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Any = None
    enum: list[Any] | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    items_type: ParameterType | None = None  # For arrays

    def validate(self, value: Any) -> tuple[bool, str | None]:
        """
        Validate a value against this parameter specification.

        Args:
            value: The value to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check required
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' is required"
            return True, None

        # Type validation
        type_checks = {
            ParameterType.STRING: lambda v: isinstance(v, str),
            ParameterType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            ParameterType.FLOAT: lambda v: isinstance(v, int | float) and not isinstance(v, bool),
            ParameterType.BOOLEAN: lambda v: isinstance(v, bool),
            ParameterType.ARRAY: lambda v: isinstance(v, list | tuple),
            ParameterType.OBJECT: lambda v: isinstance(v, dict),
            ParameterType.ANY: lambda v: True,
        }

        if self.type in type_checks and not type_checks[self.type](value):
            return False, f"Parameter '{self.name}' must be of type {self.type.value}"

        # Enum validation
        if self.enum is not None and value not in self.enum:
            return False, f"Parameter '{self.name}' must be one of {self.enum}"

        # Numeric range validation
        if self.type in (ParameterType.INTEGER, ParameterType.FLOAT):
            if self.min_value is not None and value < self.min_value:
                return False, f"Parameter '{self.name}' must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Parameter '{self.name}' must be <= {self.max_value}"

        # String length validation
        if self.type == ParameterType.STRING:
            if self.min_length is not None and len(value) < self.min_length:
                return False, f"Parameter '{self.name}' must have length >= {self.min_length}"
            if self.max_length is not None and len(value) > self.max_length:
                return False, f"Parameter '{self.name}' must have length <= {self.max_length}"

        # Array items validation
        if self.type == ParameterType.ARRAY and self.items_type:
            type_check = type_checks.get(self.items_type)
            if type_check:
                for i, item in enumerate(value):
                    if not type_check(item):
                        return False, f"Parameter '{self.name}[{i}]' must be of type {self.items_type.value}"

        return True, None


@dataclass
class ToolMetadata:
    """Metadata describing a tool's capabilities and requirements."""
    name: str
    description: str
    category: ToolCategory
    parameters: list[ParameterSpec] = field(default_factory=list)
    returns: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    author: str | None = None
    requires_auth: bool = False
    requires_api_key: bool = False
    cost_estimate: str = "low"  # low, medium, high
    timeout_seconds: int = 30
    tags: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    requires_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                }
                for p in self.parameters
            ],
            "returns": self.returns,
            "version": self.version,
            "author": self.author,
            "requires_auth": self.requires_auth,
            "requires_api_key": self.requires_api_key,
            "cost_estimate": self.cost_estimate,
            "timeout_seconds": self.timeout_seconds,
            "tags": self.tags,
            "examples": self.examples,
        }

    def to_json_schema(self) -> dict[str, Any]:
        """
        Convert metadata to JSON Schema format for LLM function calling.

        Returns:
            Dict: JSON Schema representation
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type.value,
                "description": param.description,
            }

            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            if param.min_value is not None:
                prop["minimum"] = param.min_value
            if param.max_value is not None:
                prop["maximum"] = param.max_value
            if param.min_length is not None:
                prop["minLength"] = param.min_length
            if param.max_length is not None:
                prop["maxLength"] = param.max_length
            if param.items_type:
                prop["items"] = {"type": param.items_type.value}

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
            }
        }

        if required:
            schema["parameters"]["required"] = required

        return schema


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    error: str | None = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BaseTool(ABC):
    """
    Abstract base class for all tools in the effGen framework.

    All tools must inherit from this class and implement the required methods.
    This ensures a consistent interface across all tools for metadata, validation,
    and execution.

    Example:
        class MyTool(BaseTool):
            def __init__(self):
                super().__init__(
                    metadata=ToolMetadata(
                        name="my_tool",
                        description="Does something useful",
                        category=ToolCategory.COMPUTATION,
                        parameters=[
                            ParameterSpec(
                                name="input",
                                type=ParameterType.STRING,
                                description="Input value",
                                required=True
                            )
                        ]
                    )
                )

            async def _execute(self, input: str, **kwargs) -> Any:
                # Tool implementation
                return f"Processed: {input}"
    """

    def __init__(self, metadata: ToolMetadata):
        """
        Initialize the tool with metadata.

        Args:
            metadata: Tool metadata specification
        """
        self._metadata = metadata
        self._initialized = False
        self._dependencies: list[str] = []

    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return self._metadata

    @property
    def name(self) -> str:
        """Get tool name."""
        return self._metadata.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self._metadata.description

    @property
    def category(self) -> ToolCategory:
        """Get tool category."""
        return self._metadata.category

    @property
    def dependencies(self) -> list[str]:
        """Get tool dependencies (names of other required tools)."""
        return self._dependencies

    def validate_parameters(self, **kwargs) -> tuple[bool, str | None]:
        """
        Validate input parameters against the tool's parameter specifications.

        Args:
            **kwargs: Parameters to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check for unknown parameters — warn and strip rather than reject,
        # since SLMs often hallucinate extra parameters
        known_params = {p.name for p in self._metadata.parameters}
        unknown = set(kwargs.keys()) - known_params
        if unknown:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning(
                f"Tool '{self._metadata.name}': ignoring unknown parameters {unknown}"
            )
            for k in unknown:
                del kwargs[k]

        # Validate each parameter
        for param_spec in self._metadata.parameters:
            value = kwargs.get(param_spec.name)
            is_valid, error = param_spec.validate(value)
            if not is_valid:
                return False, error

        return True, None

    async def initialize(self) -> None:
        """
        Initialize the tool (load resources, connect to services, etc.).
        Called once before first use. Override in subclasses if needed.
        """
        self._initialized = True

    async def cleanup(self) -> None:
        """
        Clean up tool resources (close connections, release memory, etc.).
        Called when tool is no longer needed. Override in subclasses if needed.
        """
        self._initialized = False

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """
        Execute the tool's main functionality.

        This method must be implemented by all tool subclasses.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Any: Tool execution result

        Raises:
            Exception: Any errors during execution
        """
        pass

    def _coerce_parameters(self, kwargs: dict) -> dict:
        """Coerce LLM-supplied parameter values to their declared types.

        LLMs occasionally send integers as strings (e.g. ``"precision": "0"``).
        This method silently converts them so downstream validation succeeds.
        Only applies to INTEGER and FLOAT spec types; leaves everything else
        untouched.
        """
        coerced = dict(kwargs)
        param_map = {p.name: p for p in self._metadata.parameters}
        for name, value in list(coerced.items()):
            spec = param_map.get(name)
            if spec is None or value is None:
                continue
            if spec.type == ParameterType.INTEGER and isinstance(value, str):
                try:
                    coerced[name] = int(value)
                except (ValueError, TypeError):
                    pass
            elif spec.type == ParameterType.FLOAT and isinstance(value, str):
                try:
                    coerced[name] = float(value)
                except (ValueError, TypeError):
                    pass
            elif spec.type == ParameterType.BOOLEAN and isinstance(value, str):
                coerced[name] = value.lower() in ("true", "1", "yes")
        return coerced

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with parameter validation and error handling.

        This is the main entry point for tool execution. It handles:
        - Parameter validation
        - Tool initialization (if needed)
        - Execution timing
        - Error handling
        - Result formatting

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult: Execution result with metadata
        """
        start_time = time.time()

        try:
            # Coerce string-typed numerics from LLM responses before validation
            kwargs = self._coerce_parameters(kwargs)

            # Validate parameters
            is_valid, error = self.validate_parameters(**kwargs)
            if not is_valid:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Parameter validation failed: {error}",
                    execution_time=time.time() - start_time
                )

            # Initialize if needed
            if not self._initialized:
                await self.initialize()

            # Execute the tool
            output = await self._execute(**kwargs)

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                output=output,
                execution_time=execution_time,
                metadata={
                    "tool_name": self.name,
                    "tool_version": self._metadata.version,
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool execution failed: {str(e)}",
                execution_time=execution_time,
                metadata={
                    "tool_name": self.name,
                    "error_type": type(e).__name__,
                }
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', category='{self.category.value}')>"

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
