import React from 'react';
import { Link } from 'react-router-dom';
import { Server } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function APIServer() {
  return (
    <DocPage
      title="API Server v2"
      subtitle="OpenAI-compatible production gateway with streaming, priority queue, auto-scaling agent pool, multi-tenant API keys, and embeddings."
      icon={<Server size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Deployment', path: '/api-server' },
        { label: 'API Server' },
      ]}
    >
      <h2>Running the Server</h2>
      <CodeBlock
        code={`# Start the gateway
effgen serve --host 0.0.0.0 --port 8000

# With a pre-loaded default model
effgen serve --model Qwen/Qwen2.5-3B-Instruct --quantization 4bit`}
        language="bash"
        filename="terminal"
      />

      <h2>Endpoints</h2>
      <ApiTable
        headers={['Endpoint', 'Compatibility', 'Features']}
        rows={[
          [<code>POST /v1/chat/completions</code>, 'OpenAI Chat', 'tools, stream=true (SSE), model aliases'],
          [<code>POST /v1/completions</code>, 'OpenAI Completions', 'stream=true (SSE), model aliases'],
          [<code>POST /v1/embeddings</code>, 'OpenAI Embeddings', 'SentenceTransformers + TFIDF fallback, LRU + SQLite cache'],
          [<code>GET  /health</code>, 'effGen', 'Liveness + readiness (public, no auth)'],
          [<code>GET  /metrics</code>, 'effGen', 'Prometheus histograms + counters (v0.2.9); requires auth by default in v0.3.0 unless explicitly opened'],
          [<code>GET  /slo</code>, 'effGen', 'SLO burn-rate tracking (v0.2.9)'],
          [<code>GET  /dashboard</code>, 'effGen', 'Live local dashboard SPA + /dashboard/data.json + SSE /dashboard/spans (v0.2.10); requires auth by default in v0.3.0'],
        ]}
      />
      <InfoBox type="success" title="v0.2.9 / v0.2.10 — auth, observability & dashboard">
        <p>
          As of <strong>v0.2.10</strong> the server validates <strong>Bearer JWTs</strong> via
          OIDC on every non-public endpoint, enforces <strong>RBAC</strong> with daily cost caps,
          and writes a per-request <strong>audit log</strong> — see{' '}
          <a href="/docs/security">Security</a>. <strong>v0.2.9</strong> added the Prometheus{' '}
          <code>/metrics</code> and <code>/slo</code> endpoints — see{' '}
          <a href="/docs/observability">Observability</a>. The v0.2.10 live{' '}
          <a href="/docs/dx">dashboard</a> is served at <code>/dashboard</code>. For
          containerised and serverless deploys (Docker, Helm, Lambda, Cloudflare) see{' '}
          <a href="/docs/deployment">Deployment</a>.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.0 — the server fails closed">
        <p>
          v0.3.0 closes the gaps around auth and error reporting. With no configured issuer / JWKS
          outside dev mode, the server now <strong>rejects all bearer tokens</strong> — a forged
          HS256 JWT can no longer reach <code>/whoami</code> or <code>/v1/chat/completions</code>,
          and <code>/v1/*</code> return <code>401</code> without credentials. Production CORS no
          longer combines a wildcard origin with credentials; <code>/metrics</code> and the
          dashboard require auth unless explicitly opened; the <code>viewer</code> role can no
          longer run tools and unknown roles are rejected (strict mode). Budget enforcement{' '}
          <strong>reserves then reconciles</strong> so failed calls are not charged, request bodies
          are size-limited before buffering, and upstream / provider-auth / missing-key failures map
          to <strong>502/503</strong> — <code>401</code> is reserved for genuine client-auth
          failures. The reported server version is sourced from <code>effgen.__version__</code>.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.1 — an honest OpenAI-compatible surface">
        <p>
          v0.3.1 seals two silent quality downgrades. A client-defined function tool the server
          does not host is no longer dropped silently — it is rejected with a clear{' '}
          <code>400</code> (<code>unknown_tool</code>) naming it; built-in tools still resolve and
          run server-side. <code>/v1/embeddings</code> strips a <code>provider:</code> prefix so{' '}
          <code>openai:text-embedding-3-small</code> reaches the real neural model, and when that
          backend can&apos;t load it reflects the lexical fallback to the caller (or fails closed
          with <code>503</code> under <code>EFFGEN_EMBEDDINGS_STRICT=1</code>) instead of quietly
          serving near-zero hash vectors. Auth / validation / rate-limit / budget errors now share
          the same <code>{'{"error":{message,type,code}}'}</code> envelope as model errors, an
          empty <code>messages</code> array returns <code>400</code>, per-call <code>cost_usd</code>{' '}
          is surfaced in the <code>effgen</code> response extension, and <code>effgen serve --help</code>{' '}
          documents the operational env knobs and adds <code>--rate-limit</code>.
        </p>
      </InfoBox>

      <h3>Model Aliases</h3>
      <p>
        Callers can use OpenAI-style model names. The server resolves them to local SLMs:
      </p>
      <ApiTable
        headers={['Alias', 'Resolves to']}
        rows={[
          [<code>gpt-4</code>, 'Qwen/Qwen2.5-7B-Instruct'],
          [<code>gpt-3.5-turbo</code>, 'Qwen/Qwen2.5-3B-Instruct'],
          [<code>text-embedding-ada-002</code>, 'sentence-transformers/all-MiniLM-L6-v2'],
        ]}
      />

      <h2>Calling the Server</h2>

      <h3>With the effGen Python Client</h3>
      <CodeBlock
        code={`from effgen.client import EffGenClient

client = EffGenClient(base_url="http://localhost:8000", api_key="sk-demo")

# Sync chat
resp = client.chat("What is 2+2?", tools=["calculator"])
print(resp.content)

# Sync streaming
for chunk in client.chat_stream_sync("Tell me a story"):
    print(chunk, end="")

# Async streaming
import asyncio
async def main():
    async for chunk in client.chat_stream("Tell me a story"):
        print(chunk, end="")
asyncio.run(main())

# Embeddings — embed() takes a list and returns a list of vectors
vecs = client.embed(["Hello world"])
print(len(vecs), len(vecs[0]))       # 1, 384 (depending on model)

# Health
print(client.health())`}
        language="python"
        filename="python_client.py"
      />

      <h3>With the Official OpenAI SDK</h3>
      <CodeBlock
        code={`from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-demo")

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",      # → Qwen2.5-3B
    messages=[{"role": "user", "content": "What is 2+2?"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")`}
        language="python"
        filename="openai_sdk.py"
      />

      <h3>With the TypeScript / JavaScript Client</h3>
      <CodeBlock
        code={`// clients/typescript/
import { EffGenClient } from "effgen-client";

const client = new EffGenClient({ baseUrl: "http://localhost:8000", apiKey: "sk-demo" });

const resp = await client.chat("What is 2+2?");
console.log(resp.content);

// Streaming
for await (const chunk of client.chatStream("Tell me a story")) {
  process.stdout.write(chunk);
}`}
        language="typescript"
        filename="client.ts"
      />

      <h2>Under the Hood</h2>

      <h3>Request Queue</h3>
      <p>
        <code>RequestQueue</code> is a priority queue with fair scheduling, deadlines, and
        backpressure. When full, it raises <code>QueueFullError</code> rather than silently
        dropping requests.
      </p>

      <h3>Agent Pool</h3>
      <p>
        <code>AgentPool</code> manages pre-warmed agents with min / max size, idle TTL, health
        checking, and acquire / release semantics. Sits behind the request handler so each
        request gets an already-initialised agent.
      </p>

      <h3>Multi-Tenancy</h3>
      <p>
        <code>TenantManager</code> enforces rate limits, allowed model lists, and tool
        permissions per tenant. <code>APIKey</code> management uses hashed storage with
        constant-time resolution (safe against timing attacks).
      </p>

      <CodeBlock
        code={`from effgen.api import TenantManager

tm = TenantManager()
tenant = tm.create_tenant(
    name="acme",
    tenant_id="acme",                        # optional — auto-generated when omitted
    allowed_models=["Qwen/Qwen2.5-3B-Instruct"],
    allowed_tools=["calculator", "web_search"],
    rate_limit_per_min=60,
)

# create_api_key returns (record, raw_key). The raw key is shown ONCE.
record, raw_key = tm.create_api_key(tenant_id=tenant.id)
print(raw_key)  # → "eg-..."`}
        language="python"
        filename="tenancy.py"
      />

      <h3>Production Middleware</h3>
      <FeatureList
        features={[
          { icon: '🆔', title: 'Request IDs', description: 'Every request gets an X-Request-ID for log correlation.' },
          { icon: '🌐', title: 'CORS', description: 'Configurable CORS origins.' },
          { icon: '📦', title: 'GZip', description: 'Response compression.' },
          { icon: '🛑', title: 'Graceful shutdown', description: 'Drains the queue and closes pooled agents cleanly.' },
        ]}
      />

      <h2>Embeddings</h2>
      <p>
        <code>/v1/embeddings</code> is backed by <code>EmbeddingEngine</code> with two backends:
        <code> SentenceTransformerEmbedder</code> (when installed) and <code>TFIDFEmbedder</code>
        as a zero-dep fallback. Results are cached via <code>LRUCache</code> and optionally a
        durable <code>SQLiteCache</code>.
      </p>

      <CodeBlock
        code={`from effgen.api.embeddings import EmbeddingEngine

engine = EmbeddingEngine(lru_size=2048, persistent_cache=True)
# embed() takes a list of texts and returns a list of vectors.
# When sentence-transformers is installed, the configured embedder is used;
# otherwise the engine falls back to the built-in TFIDFEmbedder automatically.
vecs = engine.embed(["hello world", "another sentence"])
print(len(vecs), len(vecs[0]))`}
        language="python"
        filename="embeddings_engine.py"
      />

      <InfoBox type="success" title="Guardrails at the boundary">
        <p>
          Guardrails configured on the default <code>AgentPool</code> agent run on every
          request — inbound prompts go through <code>INPUT</code> guardrails and outbound
          responses through <code>OUTPUT</code> guardrails. See
          {' '}<Link to="/guardrails">Guardrails</Link>.
        </p>
      </InfoBox>

      <h2>See Also</h2>
      <p>
        <Link to="/clients">Clients &amp; SDKs</Link> · <Link to="/guardrails">Guardrails</Link> ·
        {' '}<Link to="/workflows">Workflows</Link>
      </p>
    </DocPage>
  );
}
