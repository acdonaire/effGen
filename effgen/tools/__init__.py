"""
Tools module for the effGen framework.

This module provides the tool integration system including base classes,
registry, built-in tools, and protocol implementations.
"""

# Import protocol submodules
from . import protocols
from .base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
    ToolResult,
)
from .registry import (
    ToolDependencyError,
    ToolRegistrationError,
    ToolRegistry,
    async_reset_registry,
    get_registry,
    reset_registry,
)

__all__ = [
    # Base classes
    "BaseTool",
    "ToolMetadata",
    "ToolCategory",
    "ToolResult",
    "ParameterSpec",
    "ParameterType",
    # Registry
    "ToolRegistry",
    "ToolDependencyError",
    "ToolRegistrationError",
    "get_registry",
    "reset_registry",
    "async_reset_registry",
    # Protocols
    "protocols",
]
