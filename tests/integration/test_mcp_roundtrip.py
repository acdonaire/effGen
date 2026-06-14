"""Integration: round-trip the effGen MCP server with the OFFICIAL mcp client.

These tests spawn ``EffGenMCPServer`` (stdio + HTTP) in a subprocess and drive it
with the official ``mcp`` Python SDK — the real interop path. They require the
``mcp`` package (``pip install "mcp[cli]"``) but no API keys.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

mcp = pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from effgen.tools.protocols.mcp_official import (  # noqa: E402
    EffGenMCPClient,
    MCPServerConfig,
)

pytestmark = [pytest.mark.integration]


_SERVER_TEMPLATE = textwrap.dedent(
    """
    import asyncio, os
    os.environ.setdefault("EFFGEN_SANDBOX_BACKEND", "subprocess")
    from effgen.tools.protocols.mcp_official import create_server
    server = create_server({kwargs})
    asyncio.run(server.run_stdio())
    """
)


def _server_params(**kwargs):
    code = _SERVER_TEMPLATE.format(
        kwargs=", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    )
    return StdioServerParameters(command=sys.executable, args=["-c", code])


async def test_official_client_stdio_round_trip():
    """initialize -> tools/list -> tools/call calculator + datetime."""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert "calculator" in names
            assert "datetime" in names

            # calculator schema is a valid object schema with the required arg.
            calc = next(t for t in tools.tools if t.name == "calculator")
            assert calc.inputSchema["type"] == "object"
            assert "expression" in calc.inputSchema["properties"]
            assert "expression" in calc.inputSchema.get("required", [])

            res = await session.call_tool("calculator", {"expression": "15*15"})
            assert res.isError is False
            payload = json.loads(res.content[0].text)
            assert payload["success"] is True
            assert payload["output"]["result"] == 225

            res = await session.call_tool("datetime", {"operation": "now"})
            assert res.isError is False


async def test_official_client_invalid_args_is_mcp_error():
    """Missing a required argument -> proper MCP error, not a server crash."""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("calculator", {})  # missing 'expression'
            assert res.isError is True


async def test_unsafe_tools_hidden_by_default():
    """Fail-closed: no shell/REPL exposed unless explicitly enabled."""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert "bash" not in names
            assert "python_repl" not in names


async def test_allowlist_opt_in_exposes_named_tool():
    params = _server_params(allowed_tools=["calculator", "bash"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert names == {"calculator", "bash"}


async def test_effgen_client_stdio_round_trip():
    """The effGen client wrapper round-trips against its own server."""
    code = _SERVER_TEMPLATE.format(kwargs="")
    cfg = MCPServerConfig(
        name="effgen", transport="stdio", command=sys.executable, args=["-c", code]
    )
    async with EffGenMCPClient(cfg) as client:
        names = {t.name for t in client.get_tools()}
        assert "calculator" in names
        res = await client.call_tool("calculator", {"expression": "2**10"})
        payload = json.loads(res.content[0].text)
        assert payload["output"]["result"] == 1024
