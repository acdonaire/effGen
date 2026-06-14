"""
Protocol integrations for effGen.

This package provides implementations for various agent communication protocols:

- **MCP (Model Context Protocol)** — Anthropic's protocol for context and tool sharing.
  Round-tripped against the official MCP client (stdio + HTTP); production-ready.

  - ``mcp_official``: built on the official MCP SDK (FastMCP). **Recommended** —
    use it whenever ``pip install "mcp[cli]"`` is available.
  - ``mcp``: a self-contained implementation with no external SDK dependency.
    Use it only when the official ``mcp`` package cannot be installed.

- **A2A (Agent-to-Agent)** — Google's agent-communication protocol.
  *Experimental*: effGen ships the client side (``A2AClient``, ``AgentCard``,
  auth handlers) and the wire-protocol/task model; there is no bundled A2A
  server. See ``effgen.tools.protocols.a2a``.

- **ACP (Agent Communication Protocol)** — IBM's agent-interoperability protocol.
  *Experimental*: client + server smoke-tested locally; not yet validated against
  the external BeeAI ecosystem. See ``effgen.tools.protocols.acp``.

HTTP transports bind to 127.0.0.1 by default and warn when exposed to the
network. Apply authentication before binding to a public address.
"""

from . import a2a, acp, mcp, mcp_official

__all__ = [
    "mcp",
    "mcp_official",
    "a2a",
    "acp",
]
