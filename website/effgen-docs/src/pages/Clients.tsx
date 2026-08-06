import React from 'react';
import { Link } from 'react-router-dom';
import { Code } from 'lucide-react';
import DocPage, { InfoBox, ApiTable } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Clients() {
  return (
    <DocPage
      title="Clients &amp; SDKs"
      subtitle="First-party Python (sync + async) and TypeScript clients for the effGen API server — with retries, streaming, and typed exceptions."
      icon={<Code size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Deployment', path: '/api-server' },
        { label: 'Clients' },
      ]}
    >
      <h2>Python Client</h2>
      <p>
        <code>effgen.client.EffGenClient</code> exposes both sync (<code>requests</code>) and
        async (<code>httpx</code>) surfaces, exponential-backoff retries, and seven typed
        exception classes.
      </p>

      <CodeBlock
        code={`from effgen.client import EffGenClient

client = EffGenClient(
    base_url="http://localhost:8000",
    api_key="sk-...",
    timeout=30.0,
    max_retries=3,
)

# Sync
resp = client.chat("What is 2+2?", tools=["calculator"])
print(resp.content)

# Sync streaming
for chunk in client.chat_stream_sync("Tell me a story"):
    print(chunk, end="")

# Embeddings — embed() takes a list and returns a list of vectors
vecs = client.embed(["Hello world"])
print(len(vecs), len(vecs[0]))      # 1, 384 (depending on model)

# Health
print(client.health())`}
        language="python"
        filename="sync_client.py"
      />

      <h3>Async</h3>
      <CodeBlock
        code={`import asyncio
from effgen.client import EffGenClient

client = EffGenClient(base_url="http://localhost:8000", api_key="sk-...")

async def main():
    resp = await client.achat("What is 2+2?", tools=["calculator"])
    print(resp.content)

    async for chunk in client.chat_stream("Tell me a story"):
        print(chunk, end="")

asyncio.run(main())`}
        language="python"
        filename="async_client.py"
      />

      <h3>Typed Exceptions</h3>
      <ApiTable
        headers={['Exception', 'When']}
        rows={[
          [<code>EffGenClientError</code>, 'Base class for all client errors'],
          [<code>EffGenAPIError</code>, 'Server returned a non-2xx response'],
          [<code>EffGenAuthError</code>, '401 / 403 from the server'],
          [<code>EffGenRateLimitError</code>, '429 — server payload available on .payload'],
          [<code>EffGenServerError</code>, '5xx — retried automatically up to max_retries'],
          [<code>EffGenConnectionError</code>, 'Network / DNS / TLS failure'],
          [<code>EffGenTimeoutError</code>, 'Request exceeded the configured timeout'],
        ]}
      />

      <CodeBlock
        code={`from effgen.client import EffGenClient, EffGenRateLimitError, EffGenAuthError

try:
    client.chat("hi")
except EffGenRateLimitError as e:
    # 429 — server response body is on e.payload, status code on e.status_code
    print(e.status_code, e.payload)
except EffGenAuthError:
    print("Invalid API key")`}
        language="python"
        filename="error_handling.py"
      />

      <h2>TypeScript / JavaScript Client</h2>
      <p>
        <code>clients/typescript/</code> in the effGen repo ships a fetch-based client that
        works in Node 18+, Deno, Bun, and modern browsers.
      </p>

      <CodeBlock
        code={`import { EffGenClient } from "effgen-client";

const client = new EffGenClient({
  baseUrl: "http://localhost:8000",
  apiKey: "sk-...",
  timeoutMs: 30_000,
});

// Sync-style (Promise)
const resp = await client.chat("What is 2+2?", { tools: ["calculator"] });
console.log(resp.content);

// Streaming
for await (const chunk of client.chatStream("Tell me a story")) {
  process.stdout.write(chunk);
}

// Embeddings + health — embed() takes an array of strings
const vecs = await client.embed(["Hello world"]);
const status = await client.health();   // { status, details, ok }
console.log(status.ok);`}
        language="typescript"
        filename="client.ts"
      />

      <h2>Using the Official OpenAI SDK</h2>
      <p>
        The effGen API server is OpenAI-compatible — you can point any OpenAI SDK at it:
      </p>
      <CodeBlock
        code={`# Python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-demo")

# Node.js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "sk-demo" });`}
        language="python"
        filename="openai_interop.py"
      />

      <InfoBox type="success" title="Which client should I use?">
        <p>
          Use <strong>effGen's own client</strong> if you want typed exceptions, streaming
          helpers, and transparent retries. Use the <strong>OpenAI SDK</strong> if you want
          to migrate existing OpenAI code without changes — just swap the base URL.
        </p>
      </InfoBox>

      <h2>See Also</h2>
      <p>
        <Link to="/api-server">API Server v2</Link> · <Link to="/quickstart">Quick Start</Link>
      </p>
    </DocPage>
  );
}
