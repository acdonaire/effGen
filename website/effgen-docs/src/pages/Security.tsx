import React from 'react';
import { Lock } from 'lucide-react';
import DocPage, { ApiTable, InfoBox, QuickLinks } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Security() {
  return (
    <DocPage
      title="Security"
      subtitle="effGen hardens the framework end-to-end: a sandboxed CodeExecutor (Docker by default, unprivileged-namespace subprocess fallback), OAuth2/OIDC auth with RBAC and a per-request audit log on the API server, secret scanning with gitleaks, a CycloneDX SBOM pipeline, pip-audit dependency auditing, and hash-verified supply-chain locks (all v0.2.10). v0.3.0 makes the server fail closed (forged JWTs rejected, locked-down CORS / metrics / dashboard, 502 not 401 on upstream failures) and hardens every built-in tool: one shared SSRF guard, an out-of-process PythonREPL timeout, path-confined file tools, and the removal of the unsafe pickle / eval paths. No breaking API changes — every security feature is additive."
      icon={<Lock size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Security' },
      ]}
    >
      <InfoBox type="success" title="New in v0.3.0 — fail-closed server &amp; hardened tools">
        <p>
          The <strong>stabilization &amp; hardening</strong> release makes the API server{' '}
          <strong>fail closed</strong> and sandboxes every built-in tool. There are no breaking API
          changes — these are defensive hardening of existing surfaces. See{' '}
          <a href="#fail-closed-server">Fail-Closed Server</a> and{' '}
          <a href="#hardened-tools">Hardened Built-in Tools</a> below.
        </p>
      </InfoBox>

      <InfoBox type="success" title="New in v0.2.10">
        <p>
          The <strong>Security, Edge &amp; Developer Experience</strong> release adds the{' '}
          <code>effgen.security</code> sandbox, server-side <code>auth</code> /{' '}
          <code>rbac</code> / <code>audit</code> modules, a gitleaks pre-commit + CI workflow, a
          CycloneDX SBOM, pip-audit CI, and startup hash verification. Deployment and DX features
          shipped alongside — see <a href="/docs/deployment">Deployment</a>.
        </p>
      </InfoBox>

      <h2 id="fail-closed-server">Fail-Closed Server (v0.3.0)</h2>
      <p>
        Outside dev mode with no configured issuer / JWKS, the server now{' '}
        <strong>rejects all bearer tokens</strong> — a forged HS256 JWT can no longer reach{' '}
        <code>/whoami</code> or <code>/v1/chat/completions</code>, and <code>/v1/*</code> return{' '}
        <code>401</code> without credentials. This closes the gap where a missing OIDC
        configuration previously meant tokens were accepted unverified.
      </p>
      <ApiTable
        headers={['Hardening', 'Behavior in v0.3.0']}
        rows={[
          ['Forged tokens', 'With no configured issuer/JWKS (outside dev mode) the server rejects every bearer token instead of accepting it unverified.'],
          ['CORS', 'Production CORS no longer combines a wildcard origin with credentials.'],
          ['Metrics & dashboard', 'Require auth unless explicitly opened; the viewer role can no longer run tools, and unknown roles are rejected (strict mode).'],
          ['Budget', 'Reserves then reconciles, so failed calls are not charged; request bodies are size-limited before buffering.'],
          ['Error mapping', 'Upstream / provider-auth / missing-key failures map to 502/503 (server-side); 401 is reserved for genuine client-auth failures.'],
          ['Version', 'The reported server version is sourced from effgen.__version__.'],
        ]}
      />

      <h2 id="hardened-tools">Hardened Built-in Tools (v0.3.0)</h2>
      <p>
        Every built-in tool that runs code, takes a URL, or touches the filesystem is now
        hardened by default.
      </p>
      <ApiTable
        headers={['Tool surface', 'Hardening']}
        rows={[
          [<code>PythonREPL</code>, 'Runs user code in a worker subprocess with a hard wall-clock timeout enforced from outside the executed code, process-group kill, and memory/output caps — a `while True: pass` now dies at its timeout, not ~30 s later.'],
          ['URL-taking tools', 'One shared SSRF guard (tools/builtin/_net.py): DNS resolution, per-IP classification, and re-validation on every redirect. Private / loopback / link-local / metadata addresses are blocked by default (opt-out for user-URL tools, host-pinned for fixed endpoints).'],
          ['File-path tools', 'Path confinement (tools/builtin/_fs.py) on every file-path tool — no traversal outside the allowed root.'],
          ['State / persistence', 'The un-gated pickle.load path was removed in favor of JSON-only state.'],
          ['Prompt-chain conditions', 'Evaluated with an AST-whitelist comparator instead of eval() over interpolated model output. A security-pattern gate test prevents regressions.'],
        ]}
      />
      <CodeBlock
        code={`from effgen.tools.builtin import PythonREPL

repl = PythonREPL()  # v0.3.0: out-of-process timeout, process-group kill, mem/output caps
result = repl.run("while True: pass")   # killed at the timeout, not ~30 s later
print(result.error)  # timeout

# URL tools share one SSRF guard — private / loopback / metadata addresses are blocked
# by default, and the guard re-validates on every HTTP redirect.`}
        language="python"
        filename="v030_hardened_tools.py"
      />

      <h2>Sandboxed Code Execution</h2>
      <p>
        When an agent receives an attacker-controlled prompt it may emit arbitrary
        Python/shell code and pass it to <code>CodeExecutor</code>. As of v0.2.10 that code
        runs inside a sandbox by default — <code>DockerSandbox</code> when a Docker daemon is
        available, otherwise a <code>SubprocessSandbox</code> using unprivileged user
        namespaces (no root, no <code>CAP_SYS_ADMIN</code>).
      </p>
      <ApiTable
        headers={['Backend', 'Isolation', 'Selection']}
        rows={[
          [<code>DockerSandbox</code>, '--read-only, --network=none, --cap-drop=ALL, --no-new-privileges, --pids-limit=100, --memory=256m, non-root user', 'Auto-selected when the Docker daemon is available'],
          [<code>SubprocessSandbox</code>, 'unshare --map-root-user --net --mount; private tmpfs over /tmp; network blocked; ulimit caps', 'Fallback when Docker is unreachable (loud warning)'],
          [<code>FirecrackerSandbox</code>, 'Stub — NotImplementedError with microVM install hints', 'Never auto-selected'],
          [<code>OffSandbox</code>, 'Runs on host with a loud startup warning', 'Only via EFFGEN_SANDBOX_BACKEND=off; never auto-selected'],
        ]}
      />
      <CodeBlock
        code={`import asyncio
from effgen.security.sandbox import get_sandbox, SandboxConfig

async def main():
    config = SandboxConfig(backend="subprocess", timeout=10)
    sandbox = await get_sandbox(config)
    result = await sandbox.run('print("hello")', "python", config)
    print(result.stdout)  # hello

asyncio.run(main())

# Or use EFFGEN_SANDBOX_BACKEND=docker for full Docker isolation`}
        language="python"
        filename="sandbox.py"
      />
      <InfoBox type="info" title="Sandbox configuration">
        <p>
          Env-driven: <code>EFFGEN_SANDBOX_BACKEND=docker|subprocess|off</code> and{' '}
          <code>EFFGEN_SANDBOX_TIMEOUT=10</code>. Tests assert the subprocess path blocks
          network (<code>urllib.request.urlopen</code> → <code>OSError</code>) and isolates the
          filesystem (<code>rm -rf /tmp/evil</code> leaves the host untouched). Docker paths
          skip cleanly when the daemon is unavailable.
        </p>
      </InfoBox>

      <h2>API Server Authentication (OIDC / JWT)</h2>
      <p>
        The API server validates <strong>Bearer JWTs</strong> on every non-public endpoint via{' '}
        <code>authlib</code>. Tokens are issued by any standard OIDC provider (Auth0, Keycloak,
        Google, Azure AD). As of v0.3.0 the only public endpoint by default is <code>/health</code>:{' '}
        <code>/metrics</code> and the <code>/dashboard</code> data endpoints
        (<code>/dashboard/data.json</code>, <code>/dashboard/spans</code>) <strong>require auth by
        default</strong> and are only opened with <code>EFFGEN_PUBLIC_METRICS=1</code> /{' '}
        <code>EFFGEN_PUBLIC_DASHBOARD=1</code> (or in dev mode). The dashboard SPA shell still loads
        so the page renders, but it shows no data without auth.
      </p>
      <ApiTable
        headers={['Environment variable', 'Description']}
        rows={[
          [<code>EFFGEN_OIDC_ISSUER</code>, 'Token issuer URL (required in prod)'],
          [<code>EFFGEN_OIDC_CLIENT_ID</code>, 'Expected aud claim in JWTs (required in prod)'],
          [<code>EFFGEN_OIDC_JWKS_URI</code>, 'JWKS endpoint; omit to use OIDC discovery'],
          [<code>EFFGEN_DEV_MODE</code>, 'Set to 1 to disable auth with a loud warning (never in prod)'],
          [<code>EFFGEN_PUBLIC_METRICS</code>, 'v0.3.0: set to 1 to make /metrics public (protected by default)'],
          [<code>EFFGEN_PUBLIC_DASHBOARD</code>, 'v0.3.0: set to 1 to make the /dashboard data endpoints public (protected by default)'],
          [<code>EFFGEN_METRICS_AUTH</code>, 'Legacy: set to 1 to force auth on /metrics (metrics already requires auth by default in v0.3.0)'],
        ]}
      />
      <CodeBlock
        code={`# Production — OIDC auth on
export EFFGEN_OIDC_ISSUER=https://your-tenant.auth0.com/
export EFFGEN_OIDC_CLIENT_ID=https://effgen.api
export EFFGEN_OIDC_JWKS_URI=https://your-tenant.auth0.com/.well-known/jwks.json
effgen serve --port 8000

# Dev mode (auth disabled — local development only, prints a loud warning)
EFFGEN_DEV_MODE=1 effgen serve --port 8000`}
        language="bash"
        filename="terminal"
      />

      <h2>Role-Based Access Control</h2>
      <p>
        Each authenticated request carries role names (from the JWT <code>roles</code> /{' '}
        <code>scope</code> claim). The server resolves an effective policy as the{' '}
        <strong>union</strong> of all granted roles — the most-permissive role wins. A principal
        with no recognized roles falls back to read-only (no tools) rather than failing open.
        The pure-ASGI <code>RBACBudgetMiddleware</code> enforces tool allow-lists (403) and the
        daily cost cap (429 <code>BudgetExceeded</code>).
      </p>
      <ApiTable
        headers={['Role', 'Allowed tools', 'Allowed models', 'Max cost/day']}
        rows={[
          [<code>admin</code>, 'all', 'all', 'unlimited'],
          [<code>researcher</code>, 'all', 'all', '$50'],
          [<code>viewer</code>, 'all', 'all', '$5'],
          [<code>reader</code>, 'none (deny_tools: true)', 'all', '$1'],
        ]}
      />
      <InfoBox type="info" title="Empty list means all">
        <p>
          <code>allowed_tools: []</code> and <code>allowed_models: []</code> mean "no
          restriction" (all). To express a read-only role with no tools, set{' '}
          <code>deny_tools: true</code> — that's how the built-in <code>reader</code> role
          works.
        </p>
      </InfoBox>

      <h2>Audit Log</h2>
      <p>
        Every request/response pair is appended to{' '}
        <code>~/.effgen/audit/&lt;date&gt;.jsonl</code> with fields{' '}
        <code>ts, principal, role, endpoint, request_summary, response_summary, outcome</code>.
        Content is redacted via the same <code>Redactor</code> used by the logging layer, so the
        audit log never contains secrets.
      </p>

      <h2>Supply-Chain Hardening</h2>
      <ApiTable
        headers={['Control', 'What it does']}
        rows={[
          ['Secret scanning', 'A tuned .gitleaks.toml + pre-commit hook block commits with secret-like strings; the secret-scan CI workflow scans both the working tree and full git history.'],
          ['SBOM', 'sbom.cdx.json (CycloneDX 1.5) lists every runtime dependency with name, version, and PURL; CI generates and schema-validates it on each push.'],
          ['Dependency audit', 'pip-audit CI fails on HIGH/CRITICAL advisories and runs clean across the documented extras (v0.3.0); pypdf was bumped to >=6.13.3 (GHSA-jm82-fx9c-mx94) and the known-malicious fastapi==0.136.3 is pinned away in the lockfiles.'],
          ['Hash verification', 'EFFGEN_VERIFY_HASHES=1 compares installed-wheel hashes against the lockfile on startup; logs hash_verification: ok or drift with the first drifted package.'],
          ['Reproducible installs', 'requirements-lock.txt and requirements-all-lock.txt are hash-verified locks for base and .[all] installs.'],
        ]}
      />
      <CodeBlock
        code={`# Reproducible, hash-verified install
pip install -e . -c requirements-lock.txt

# Verify installed wheel hashes against the lockfile on startup
EFFGEN_VERIFY_HASHES=1 effgen serve`}
        language="bash"
        filename="terminal"
      />

      <QuickLinks
        links={[
          { icon: 'D', title: 'Deployment', description: 'Docker, Kubernetes/Helm, AWS Lambda, Cloudflare Worker', path: '/deployment' },
          { icon: 'S', title: 'API Server', description: 'OpenAI-compatible server, endpoints, auth boundary', path: '/api-server' },
          { icon: 'G', title: 'Guardrails', description: 'Input/output guardrails and policy enforcement', path: '/guardrails' },
          { icon: 'R', title: 'Release Notes', description: 'v0.3.1 real-world usability & polish release', path: '/releases' },
        ]}
      />
    </DocPage>
  );
}
