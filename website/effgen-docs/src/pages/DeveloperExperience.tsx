import React from 'react';
import { Code2 } from 'lucide-react';
import DocPage, { ApiTable, InfoBox, QuickLinks } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function DeveloperExperience() {
  return (
    <DocPage
      title="Developer Experience"
      subtitle="effGen v0.2.10 adds three DX surfaces: a VSCode extension with prompt-template completion and inline run code lenses, three Jupyter magics for interactive chat / agents / metrics, and a live local dashboard served at /dashboard. v0.3.0 makes the CLI quiet and scriptable (--json everywhere, --provider on run / chat / debug, non-zero exit codes) and adds a live thinking UX, rotating tips, 'did you mean?' suggestions, rich Markdown rendering, and a polished effgen chat REPL."
      icon={<Code2 size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Developer Experience' },
      ]}
    >
      <InfoBox type="success" title="New in v0.3.0 — a quiet, scriptable, delightful CLI">
        <p>
          v0.3.0 overhauls the command line. It is <strong>quiet by default</strong> (clean tables,
          no INFO spam) and scriptable: <code>--json</code> on <code>doctor</code> /{' '}
          <code>models</code> / <code>tools</code> / <code>cost</code>, <code>--provider</code> on{' '}
          <code>run</code> / <code>chat</code> / <code>debug</code>, and non-zero exit codes on user
          errors. A new live UX adds TTY-aware "thinking" / working status, a live elapsed / tokens
          / cost meter that freezes into a final summary, graceful Ctrl-C (partial result +
          "Stopped."), rotating tips, "did you mean?" suggestions, a first-run welcome,{' '}
          <code>effgen quickstart</code>, rich Markdown / code rendering, and a polished{' '}
          <code>effgen chat</code> REPL with slash commands and per-turn cost. All animation honors{' '}
          <code>NO_COLOR</code>, <code>--no-animation</code>, <code>--quiet</code>, and CI output.
        </p>
      </InfoBox>

      <h2>Command-Line Interface (v0.3.0)</h2>
      <CodeBlock
        code={`# Quiet, scriptable output everywhere
effgen doctor --json
effgen models list --json
effgen tools list --json
effgen cost --json

# --provider on run / chat / debug
effgen run --provider groq "Summarize the theory of relativity in two sentences."
effgen chat --provider cerebras            # polished REPL: slash commands + per-turn cost
effgen debug --provider openai "What is sqrt(144)?"

# Catalog-backed model management and a one-command onboarding
effgen models refresh                      # diff the live API: added / removed / changed
effgen doctor --live --cheap               # which keys are actually usable, not just present
effgen quickstart                          # first-run welcome and guided setup`}
        language="bash"
        filename="terminal"
      />
      <p>
        User errors now exit non-zero so the CLI composes cleanly in scripts and CI, and{' '}
        <code>effgen tools info / test</code> discover tools and exit non-zero on failure.{' '}
        <code>print(result)</code> renders the answer, and a Jupyter <code>_repr_html_</code> card
        is shown in notebooks.
      </p>

      <InfoBox type="success" title="New in v0.3.1 — a real headless contract">
        <p>
          <code>effgen run --json</code> now emits the full result document (output, success,
          tool_calls, tokens, cost, trace, citations, metadata) to <strong>stdout</strong> for
          piping to <code>jq</code> — combine with <code>-q</code> for a pristine stdout while human
          output routes to stderr. The same <code>--json</code> is added to the CI quality-gate
          commands <code>eval</code>, <code>compare</code>, <code>workflow</code>, and{' '}
          <code>sessions list</code>. Sub-cent costs now show real digits (e.g.{' '}
          <code>$0.000049</code>) instead of <code>$0.0000</code>, and <code>chat</code> gains{' '}
          <code>--system-prompt</code> / <code>--persona</code> to stand up a custom assistant from
          the terminal.
        </p>
        <CodeBlock
          code={`effgen run --json -q "What is 25 * 17?" | jq .output   # pure-JSON stdout for CI
effgen compare --json --max-cases 3 --models "openai:gpt-5-nano,groq:llama-3.1-8b-instant"`}
          language="bash"
          filename="terminal"
        />
      </InfoBox>

      <InfoBox type="success" title="New in v0.2.10">
        <p>
          The VSCode extension, Jupyter magics, and local dashboard all ship in the{' '}
          <strong>Security, Edge &amp; Developer Experience</strong> release. The Jupyter extras add
          an <code>ipython</code> dependency under <code>effgen[jupyter]</code>. (As of v0.3.0 the{' '}
          dashboard requires auth by default — see below.)
        </p>
      </InfoBox>

      <h2>Jupyter Magics</h2>
      <p>
        Load the extension with <code>%load_ext effgen.jupyter</code> after{' '}
        <code>pip install "effgen[jupyter]"</code>. Three magics are registered:
      </p>
      <ApiTable
        headers={['Magic', 'Purpose']}
        rows={[
          [<code>%effgen_chat &lt;message&gt;</code>, 'One-shot chat; displays the formatted response inline.'],
          [<code>%%effgen_agent &lt;preset&gt;</code>, 'Cell body becomes the task; displays the final answer plus the tool trace.'],
          [<code>%effgen_metrics</code>, 'Snapshot of the current Prometheus counters inline.'],
        ]}
      />
      <CodeBlock
        code={`%load_ext effgen.jupyter

# One-shot chat
%effgen_chat What is 17 * 23?

# Run an agent over the cell body
%%effgen_agent general
Summarise the top HackerNews stories today.

# Inline metrics snapshot
%effgen_metrics`}
        language="python"
        filename="notebook.ipynb"
      />

      <h2>VSCode Extension</h2>
      <p>
        The TypeScript extension (<code>tools/vscode-effgen/</code>) adds editor support for the
        Prompt Library and the Jupyter magics.
      </p>
      <ApiTable
        headers={['Feature', 'Description']}
        rows={[
          ['Prompt-template completion', 'Triggers on LibraryPrompt(, effgen.prompts., or %effgen_ — offers every built-in template with snippet insertion.'],
          ['Run code lens', 'A ▶ Run with effGen button above any LibraryPrompt(, effgen_chat(, %%effgen_agent, or %effgen_chat line; sends to the configured server and prints to the effGen output channel.'],
          ['Hover docs', 'Hover a quoted template name to see its description, category, input schema, and usage snippet.'],
          ['Prompt registry viewer', 'effGen: Show Prompt Registry opens a webview listing all templates.'],
        ]}
      />
      <CodeBlock
        code={`cd tools/vscode-effgen
npm install
npm run compile          # TypeScript 5.3 strict, 0 errors
# Package: npm install -g @vscode/vsce && vsce package
# Then: Extensions → Install from VSIX… → vscode-effgen-0.3.1.vsix`}
        language="bash"
        filename="terminal"
      />

      <h2>Local Dashboard</h2>
      <p>
        A static SPA served by the API server at <code>/dashboard</code> gives a real-time view
        into a running deployment. (As of v0.3.0 it requires auth by default unless explicitly
        opened.) It polls{' '}
        <code>/dashboard/data.json</code> every 5 seconds and streams new spans over SSE from{' '}
        <code>/dashboard/spans</code>.
      </p>
      <ApiTable
        headers={['Panel', 'Shows']}
        rows={[
          ['Summary cards', 'Total requests, errors, average latency, estimated daily cost, total tokens.'],
          ['SLO burn rates', 'Progress bars for p99 latency, error rate, and availability against thresholds.'],
          ['Request latency chart', 'Rolling Chart.js line chart of average latency.'],
          ['Recent agent runs', 'Last 50 runs with model, token counts, cost, duration, success/error badge.'],
          ['Live span stream', 'Real-time SSE feed of trace spans with pause/clear controls.'],
          ['Prometheus metrics (raw)', 'Sortable table of all registered metric names and current values.'],
        ]}
      />
      <CodeBlock
        code={`# Dev mode — auth disabled
EFFGEN_DEV_MODE=1 uvicorn effgen.server.app:create_app --factory --host 0.0.0.0 --port 8080
# Open http://localhost:8080/dashboard

# Query the data endpoint directly
curl http://localhost:8080/dashboard/data.json | python -m json.tool`}
        language="bash"
        filename="terminal"
      />
      <InfoBox type="info" title="Dashboard auth (v0.3.0)">
        <p>
          As of v0.3.0 the <code>/dashboard</code> and <code>/dashboard/*</code> paths{' '}
          <strong>require auth by default</strong> (along with <code>/metrics</code>) unless you
          explicitly open them. In dev mode (<code>EFFGEN_DEV_MODE=1</code>) the dashboard loads
          without a token for local development. Don't expose it on a public interface in
          production without a network-level control in front.
        </p>
      </InfoBox>

      <QuickLinks
        links={[
          { icon: 'P', title: 'Prompts', description: 'The Prompt Library that powers VSCode completion', path: '/prompts' },
          { icon: 'O', title: 'Observability', description: 'Metrics, SLOs, and spans surfaced in the dashboard', path: '/observability' },
          { icon: 'D', title: 'Deployment', description: 'Docker, K8s, Lambda, Cloudflare', path: '/deployment' },
          { icon: 'R', title: 'Release Notes', description: 'v0.3.1 real-world usability & polish release', path: '/releases' },
        ]}
      />
    </DocPage>
  );
}
