import React from 'react';
import { Cloud } from 'lucide-react';
import DocPage, { ApiTable, InfoBox, QuickLinks } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Deployment() {
  return (
    <DocPage
      title="Deployment"
      subtitle="effGen ships four production deployment targets (added in v0.2.10, hardened in v0.3.0): a multi-stage Docker image, a Helm chart with HPA / PDB / NetworkPolicy, an AWS Lambda handler via the Mangum adapter with a SAM template, and a Cloudflare Worker edge proxy for auth, rate-limiting, and CORS. Every target is validated in CI (Dockerfile structure, helm lint + kubeconform, cfn-lint, wrangler dry-run). As of v0.3.0 the server fails closed — a missing OIDC config rejects all bearer tokens rather than accepting them unverified."
      icon={<Cloud size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Deployment' },
      ]}
    >
      <InfoBox type="success" title="New in v0.2.10">
        <p>
          Four production deployment paths ship in this release under <code>deploy/</code>:
          Docker, Kubernetes/Helm, AWS Lambda, and a Cloudflare Worker edge proxy. They pair
          with the new <a href="/docs/security">Security</a> (OIDC auth, RBAC, sandbox) and{' '}
          <a href="/docs/dx">Developer Experience</a> features.
        </p>
      </InfoBox>

      <h2>Docker</h2>
      <p>
        A production-grade multi-stage Dockerfile (<code>deploy/docker/Dockerfile</code>) builds
        on <code>python:3.11-slim</code> with a non-root user (uid 1000), a read-only filesystem,
        and a <code>HEALTHCHECK</code> on <code>/health</code>.
      </p>
      <CodeBlock
        code={`# Build (server extras = auth + Prometheus metrics; FastAPI/uvicorn are core)
docker build -f deploy/docker/Dockerfile --build-arg EXTRAS=server -t effgen:0.3.1 .

# Dev run (auth disabled — local testing only)
docker run --rm -p 8080:8080 -e EFFGEN_DEV_MODE=1 effgen:0.3.1
curl http://localhost:8080/health   # {"status":"ok"}

# Production run with OIDC auth
docker run --rm -p 8080:8080 \\
  -e EFFGEN_DEV_MODE=0 \\
  -e EFFGEN_OIDC_ISSUER=https://your-issuer.example.com \\
  -e EFFGEN_OIDC_CLIENT_ID=your-client-id \\
  -e EFFGEN_OIDC_JWKS_URI=https://your-issuer.example.com/.well-known/jwks.json \\
  effgen:0.3.1`}
        language="bash"
        filename="terminal"
      />

      <h2>Kubernetes / Helm</h2>
      <p>
        The Helm chart (<code>deploy/k8s/helm/effgen/</code>) renders a full production
        topology: Deployment, Service, Ingress, ConfigMap, Secret, ServiceAccount,
        NetworkPolicy, PodDisruptionBudget, HorizontalPodAutoscaler, and a PersistentVolumeClaim.
        All rendered manifests are validated against the Kubernetes 1.29 strict schema with{' '}
        <code>kubeconform</code>.
      </p>
      <ApiTable
        headers={['Setting', 'Default']}
        rows={[
          ['Replicas', '2'],
          ['Resource requests', 'cpu: 100m / memory: 256Mi'],
          ['HPA', 'min 2, max 10 — scales on CPU + effgen_model_call_latency_seconds'],
        ]}
      />
      <CodeBlock
        code={`helm install effgen deploy/k8s/helm/effgen/ -f deploy/k8s/helm/effgen/values.yaml`}
        language="bash"
        filename="terminal"
      />

      <h2>AWS Lambda</h2>
      <p>
        <code>deploy/aws_lambda/handler.py</code> wraps the FastAPI app with the Mangum adapter
        (<code>lifespan="off"</code>) and preloads the <code>ProviderRegistry</code> at module
        level for cold-start optimisation (first call &lt; 3 s, warm call &lt; 100 ms). A
        per-invocation timeout budget returns 504 on overrun. The SAM template
        (<code>sam-template.yaml</code>) provisions an HTTP API, the Lambda function, a
        CloudWatch Log Group, and a SecretsManager reference for API keys.
      </p>
      <CodeBlock
        code={`cd deploy/aws_lambda
sam build
sam deploy --guided`}
        language="bash"
        filename="terminal"
      />

      <h2>Cloudflare Worker (Edge Proxy)</h2>
      <p>
        <code>deploy/cloudflare/worker.js</code> is a thin edge proxy that adds CORS headers,
        Bearer JWT auth, fixed-window rate limiting (KV-backed), and security response headers,
        then forwards upstream with <code>duplex:"half"</code> for streaming bodies.{' '}
        <code>wrangler.toml</code> declares routes, the <code>RATE_LIMIT</code> KV namespace, and
        staging / production environments.
      </p>
      <CodeBlock
        code={`cd deploy/cloudflare
wrangler deploy                 # production
wrangler deploy --env staging   # staging`}
        language="bash"
        filename="terminal"
      />

      <h2>CI Validation</h2>
      <ApiTable
        headers={['Target', 'Validation']}
        rows={[
          ['Docker', 'Dockerfile structure checks (non-root, HEALTHCHECK, EXPOSE, multi-stage); /health 200 via uvicorn smoke'],
          ['Helm', 'helm lint clean; helm template valid; kubeconform strict K8s 1.29; HPA custom metric present'],
          ['AWS Lambda', '41 tests: v1 + v2 APIGateway events, 4xx/404, cold-start timing, 504 timeout, SAM struct + cfn-lint, live Cerebras call'],
          ['Cloudflare Worker', '30 tests: structural + stubbed unit + real-fetch round-trip (file:// → worker.js → live origin)'],
        ]}
      />

      <QuickLinks
        links={[
          { icon: 'S', title: 'Security', description: 'OIDC auth, RBAC, audit log, sandbox', path: '/security' },
          { icon: 'A', title: 'API Server', description: 'OpenAI-compatible server and endpoints', path: '/api-server' },
          { icon: 'O', title: 'Observability', description: 'Metrics + SLO endpoints scraped in production', path: '/observability' },
          { icon: 'R', title: 'Release Notes', description: 'v0.3.1 real-world usability & polish release', path: '/releases' },
        ]}
      />
    </DocPage>
  );
}
