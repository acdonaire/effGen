"""``EffGenMCPServer(config=EffGenMCPServerConfig(...))`` must actually work.

``EffGenMCPServerConfig`` is exported public surface whose fields mirror the
server's constructor kwargs, so ``config=`` is the natural call a contributor
tries first. The constructor must accept it (mapping its fields), not just the
individual keyword arguments.
"""

from __future__ import annotations

import asyncio

import pytest

mcp = pytest.importorskip("mcp")

from effgen.tools.protocols.mcp_official import (  # noqa: E402
    EffGenMCPServer,
    EffGenMCPServerConfig,
)
from effgen.tools.registry import ToolRegistry  # noqa: E402


def test_server_accepts_config_object():
    cfg = EffGenMCPServerConfig(
        name="temp-tools",
        version="2.0.0",
        allowed_tools=["calculator"],
    )
    server = EffGenMCPServer(config=cfg)
    assert server.name == "temp-tools"
    assert server.version == "2.0.0"
    assert server.allowed_tools == ["calculator"]


def test_server_direct_kwargs_still_work():
    """Backward compatible: the pre-existing direct-kwargs form is unaffected."""
    server = EffGenMCPServer(name="direct", version="3.0.0")
    assert server.name == "direct"
    assert server.version == "3.0.0"


def test_server_config_drives_real_tool_exposure():
    registry = ToolRegistry()
    cfg = EffGenMCPServerConfig(
        name="temp-tools",
        tools_registry=registry,
        allowed_tools=["calculator"],
    )
    server = EffGenMCPServer(config=cfg)
    asyncio.run(server.initialize())
    try:
        assert server._exposed_tools == ["calculator"]
    finally:
        asyncio.run(server.cleanup())
