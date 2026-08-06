import React from 'react';
import { Link } from 'react-router-dom';
import { Globe } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';

export default function Protocols() {
  const protocolDiagram = `
flowchart LR
    subgraph Agent["effGen"]
        A[Agent]
    end

    subgraph Protocols["Protocol Layer"]
        MCP[MCP Client]
        A2A[A2A Protocol]
        ACP[ACP Protocol]
    end

    subgraph External["External Systems"]
        MCP_S[MCP Servers]
        OTHER[Other Agents]
        SVC[Services]
    end

    A --> MCP
    A --> A2A
    A --> ACP
    MCP --> MCP_S
    A2A --> OTHER
    ACP --> SVC
`;

  return (
    <DocPage
      title="Communication Protocols"
      subtitle="effGen supports MCP (Model Context Protocol), A2A (Agent-to-Agent), and ACP (Agent Communication Protocol) for interoperability."
      icon={<Globe size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Advanced', path: '/multi-agent' },
        { label: 'Protocols' },
      ]}
    >
      <h2>Overview</h2>
      <p>
        effGen implements multiple communication protocols for seamless integration
        with external tools, services, and other agent systems.
      </p>

      <MermaidDiagram chart={protocolDiagram} title="Protocol Architecture" />

      <FeatureList
        features={[
          { icon: '🔌', title: 'MCP (Model Context Protocol)', description: 'Anthropic\'s protocol for connecting to external tools and data sources' },
          { icon: '🤖', title: 'A2A (Agent-to-Agent)', description: 'Google\'s protocol for agent interoperability' },
          { icon: '📡', title: 'ACP (Agent Communication)', description: 'Standard protocol for agent-service communication' },
        ]}
      />

      <h2>MCP Integration</h2>
      <p>
        The Model Context Protocol (MCP) enables agents to connect with external tools and data sources.
        effGen ships both a custom MCP client (with auto-reconnection, MCP→effGen tool bridge, resource→context bridge, and health monitoring)
        and an official MCP SDK integration (<code>effgen.tools.protocols.mcp_official</code>). v0.2.0 also fixed a
        correlation-ID-based pending request tracker and SSE exponential-backoff reconnection (max 5 retries).
      </p>

      <InfoBox type="success" title="v0.3.0 — verified MCP round-trip">
        <p>
          v0.3.0 hardens the MCP server so it <strong>round-trips with the official{' '}
          <code>mcp</code> client</strong> over both stdio and HTTP, behind a fail-closed tool
          allowlist (unknown tools are rejected rather than silently exposed). This is part of the
          broader stabilization release — see <a href="/docs/releases">Release Notes</a>.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.1 — no MCP deadlock, plugin auto-discovery">
        <p>
          v0.3.1 makes protocol integration safer to wire in. A sync <code>Agent.run()</code> handed
          a tool whose async resource is bound to the calling event loop (e.g. an MCP stdio session){' '}
          <strong>no longer hangs forever</strong> — it runs on a daemon thread bounded by the
          timeout and raises a clear <code>TimeoutError</code> pointing at{' '}
          <code>await agent.run_async(...)</code>. <strong>Installed tool plugins auto-discover</strong>:
          a package published with an <code>effgen.plugins</code> entry point (what{' '}
          <code>effgen create-plugin</code> scaffolds) has its tools folded into the registry on
          first use (set <code>EFFGEN_DISABLE_PLUGINS=1</code> to opt out). The official MCP server
          gained a package entry point, so <code>python -m effgen.tools.protocols.mcp_official</code>{' '}
          starts without the runpy double-import warning.
        </p>
      </InfoBox>

      <h3>MCP Client</h3>
      <CodeBlock
        code={`from effgen.tools.protocols.mcp import MCPClient, MCPServerConfig
from effgen.tools.protocols.mcp.protocol import TransportType

# Configure and connect to an HTTP MCP server
config = MCPServerConfig(
    name="my-mcp-server",
    transport=TransportType.HTTP,    # STDIO | HTTP | SSE | WEBSOCKET
    url="http://localhost:3000",
    timeout=30,
)
client = MCPClient(config=config, max_reconnect_attempts=5)

# Connect — discovers tools and resources from the server
await client.connect()
for tool in client.get_tools():
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")

# Execute a tool
result = await client.call_tool(
    tool_name="web_search",
    arguments={"query": "quantum computing"},
)
print(result)`}
        language="python"
        filename="mcp_client.py"
      />

      <h3>MCP Server</h3>
      <p>
        Expose effGen tools as MCP tools using a <code>ToolRegistry</code>. The server runs over
        STDIO (compatible with Claude Desktop) by default.
      </p>
      <CodeBlock
        code={`import asyncio
from effgen.tools import ToolRegistry
from effgen.tools.builtin import Calculator, WebSearch
from effgen.tools.protocols.mcp import MCPServer

# Build a registry of effGen tools to expose
registry = ToolRegistry()
registry.register_tool(Calculator)
registry.register_tool(WebSearch)

# Create MCP server
server = MCPServer(
    name="effgen-tools",
    version="1.0.0",
    tools_registry=registry,
    enable_sampling=False,
)

# Run over STDIO (Claude Desktop compatible)
asyncio.run(server.run_stdio())`}
        language="python"
        filename="mcp_server.py"
      />

      <h3>Using MCP Tools in Agents</h3>
      <CodeBlock
        code={`import asyncio
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.protocols.mcp import MCPClient, MCPServerConfig
from effgen.tools.protocols.mcp.protocol import TransportType

model = load_model("Qwen/Qwen2.5-7B-Instruct")

# Connect to MCP server and bridge tools
config = MCPServerConfig(
    name="remote-mcp",
    transport=TransportType.HTTP,
    url="http://localhost:3000",
)
mcp_client = MCPClient(config=config)

async def setup():
    await mcp_client.connect()
    # Each discovered MCP tool is wrapped as an effGen BaseTool
    return mcp_client.get_effgen_tools()

mcp_tools = asyncio.run(setup())

# Create agent with MCP tools
agent_config = AgentConfig(
    name="mcp_agent",
    model=model,
    tools=mcp_tools,        # MCP tools work like native tools
)
agent = Agent(config=agent_config)
result = agent.run("Search for latest AI news")`}
        language="python"
        filename="mcp_agent.py"
      />

      <h2>A2A Protocol</h2>
      <p>
        The Agent-to-Agent (A2A) protocol enables communication between different agent systems.
        Agents advertise themselves with an <code>AgentCard</code> (name, version, capabilities,
        endpoint), connect via <code>A2AClient</code>, and exchange tasks.
      </p>

      <CodeBlock
        code={`from effgen.tools.protocols.a2a import (
    A2AClient, AgentCard, Capability, EndpointConfig,
    A2AMessage, MessagePart, MessagePartType,
)

# Build an AgentCard for the remote agent we want to talk to
remote_card = AgentCard(
    name="remote-research-agent",
    description="Research-specialised remote agent",
    version="1.0.0",
    capabilities=[
        Capability(name="research", description="Web research"),
        Capability(name="analysis", description="Data analysis"),
    ],
    endpoint=EndpointConfig(url="http://other-agent:8000"),
)

client = A2AClient(agent_card=remote_card)
await client.connect()

# Build the instruction message and create a remote task
instruction = A2AMessage(parts=[
    MessagePart(type=MessagePartType.TEXT, content="Analyse data.csv"),
])
task = await client.create_task(
    instruction=instruction,
    capability="analysis",
    context={"file": "data.csv"},
)
print(f"Task state: {task.state}")

# Or wait for completion in one call
result = await client.execute_task(
    instruction=instruction,
    capability="analysis",
    timeout=120.0,
)`}
        language="python"
        filename="a2a_protocol.py"
      />

      <h2>ACP Protocol</h2>
      <p>
        The Agent Communication Protocol provides standardized messaging for agent-service interactions
        with full JSON Schema validation for all messages:
      </p>

      <CodeBlock
        code={`from effgen.tools.protocols.acp import ACPClient

# Create ACP client (manifest is auto-fetched on connect)
client = ACPClient(agent_url="http://ai-service:8080")
await client.connect()

# Synchronous capability execution — capability must exist in the manifest
response = await client.execute_sync(
    capability="analysis",
    input_data={"depth": "detailed", "data": {...}},
    context={"requester": "effgen"},
)
print(response.status, response.output)

# Or kick off an async task (poll/stream as it runs)
task_info = await client.execute_async(
    capability="analysis",
    input_data={...},
)`}
        language="python"
        filename="acp_protocol.py"
      />

      <h2>WebSocket Streaming Endpoint</h2>
      <p>
        effGen provides a WebSocket endpoint for real-time streaming from the API server:
      </p>

      <CodeBlock
        code={`# Start the API server
# effgen serve --port 8000

# Python client
import websockets

async def stream_response():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send("Explain quantum computing")
        async for chunk in ws:
            print(chunk, end="", flush=True)`}
        language="python"
        filename="websocket_streaming.py"
      />

      <h2>Protocol Comparison</h2>

      <ApiTable
        headers={['Protocol', 'Best For', 'Features']}
        rows={[
          ['MCP', 'Tool integration', 'Standard tool interface, server/client model'],
          ['A2A', 'Agent coordination', 'Agent discovery, task delegation, status updates'],
          ['ACP', 'Service communication', 'Structured messaging, streaming, versioning'],
        ]}
      />

      <h2>Security Considerations</h2>

      <InfoBox type="warning" title="Security Best Practices">
        <ul>
          <li><strong>Authentication:</strong> Always use authentication tokens for protocol connections</li>
          <li><strong>TLS:</strong> Use HTTPS/TLS for all network communications</li>
          <li><strong>Input validation:</strong> Validate all incoming messages before processing</li>
          <li><strong>Rate limiting:</strong> Implement rate limits to prevent abuse</li>
          <li><strong>Sandboxing:</strong> Run external tool calls in sandboxed environments</li>
        </ul>
      </InfoBox>

      <CodeBlock
        code={`from effgen.tools.protocols.mcp import MCPClient, MCPServerConfig
from effgen.tools.protocols.mcp.protocol import TransportType

# Secure MCP connection (HTTPS for HTTP/SSE; auth handled at HTTP layer)
config = MCPServerConfig(
    name="secure-mcp",
    transport=TransportType.HTTP,
    url="https://secure-server:3000",
    timeout=30,
)
client = MCPClient(config=config)`}
        language="python"
        filename="secure_protocols.py"
      />

      <InfoBox type="success" title="Next Steps">
        <p>
          Learn about <Link to="/execution">Code Execution</Link> and sandboxing,
          or explore <Link to="/configuration">Configuration</Link> for deployment settings.
        </p>
      </InfoBox>
    </DocPage>
  );
}
