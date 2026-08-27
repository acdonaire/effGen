import { Rocket } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  Callout,
  CodeBlock,
  DocPage,
  FeatureList,
  MermaidDiagram,
  QuickLinks,
  SeeAlso,
  Terminal,
} from '../components/docs';
import {
  commandCount,
  modelCount,
  presetCount,
  providerCount,
  providersWithCatalog,
  publicNameCount,
  pythonVersions,
  siteData,
  toolCount,
  version,
} from '../siteData';

const RUN_LOOP = `flowchart LR
    T["Task"] --> A["Agent"]
    A --> M["Model"]
    M -->|"answer"| R["AgentResponse"]
    M -->|"tool call"| X["Tool"]
    X -->|"ToolResult"| A
    A -.->|"history"| Mem["Memory / session"]
`;

export default function Introduction() {
  return (
    <DocPage
      subtitle={`What effGen is, what it ships, and what changed in ${version}.`}
      icon={<Rocket size={48} />}
    >
      <p>
        effGen is a Python framework for building agents on small language models — and on
        cloud models, and on anything you already serve yourself. An agent is a model, a set of
        tools and a loop that runs between them; effGen supplies the loop, {toolCount} tools,{' '}
        {presetCount} ready-made configurations, and one way of saying which model you mean
        whether it runs on your GPU, behind an API key, or on a server on your own network.
      </p>

      <h2>The shortest thing that works</h2>
      <p>
        Two lines make an agent, and a third runs it. The preset supplies the tools and the
        system prompt; the model id says where the generation happens.
      </p>

      <CodeBlock
        language="bash"
        code={`pip install effgen
export OPENAI_API_KEY=...`}
      />

      <CodeBlock
        filename="hello.py"
        code={`from effgen import create_agent

agent = create_agent("math", "openai:gpt-5-nano")
result = agent.run("What is 17% of 250?")

print(result)                  # printing the response prints the answer
print(result.success, result.tool_call_count)`}
      />

      <Terminal
        command="python hello.py"
        output={`42.5
True 1`}
        caption={`Run against effGen ${version}. The agent reached for its calculator once, which is the 1.`}
      />

      <Callout type="tip" title="No key yet?">
        <p>
          Swap the model id for a local one —{' '}
          <code>create_agent("math", "Qwen/Qwen2.5-1.5B-Instruct")</code> downloads the weights
          once and runs them on your own machine, with no account anywhere.{' '}
          <Link to="/local-models">Local models and engines</Link> covers the four engines that
          can run them.
        </p>
      </Callout>

      <h2>How a run is put together</h2>
      <p>
        Every path through effGen is the same shape. The agent sends the task and the tool
        schemas to the model; the model either answers or asks for a tool; a tool that is asked
        for returns a <code>ToolResult</code> that goes back into the conversation; the loop ends
        when the model answers or when <code>max_iterations</code> is reached. What comes back is
        an <code>AgentResponse</code> carrying the answer, the calls that were made, the sources,
        and what the run cost.
      </p>

      <MermaidDiagram
        chart={RUN_LOOP}
        title="One agent run"
        description="A task enters the agent, which calls the model. The model either returns an answer, which becomes the AgentResponse, or requests a tool, whose ToolResult returns to the agent for another pass. The agent's memory or session records the conversation."
      />

      <h2>What effGen gives you</h2>

      <FeatureList
        features={[
          {
            icon: '🧠',
            title: 'One name for any model',
            description: (
              <>
                {providerCount} provider adapters, {providersWithCatalog} of which ship a bundled
                catalog of {modelCount} models, plus the{' '}
                {siteData.models.local_engines.join(', ')} engines for weights on your own
                machine, plus any server that speaks the OpenAI protocol.
              </>
            ),
          },
          {
            icon: '🔧',
            title: `${toolCount} tools that already work`,
            description: (
              <>
                Search, documents, code execution, HTTP, mail, images, audio and more, across{' '}
                {Object.keys(siteData.tools.category_counts).length} categories — and a decorator
                that turns one of your own functions into another.
              </>
            ),
          },
          {
            icon: '🎛️',
            title: 'Control over the loop',
            description: (
              <>
                Middleware around every run, model call and tool call; a compaction strategy for
                what gets dropped when a conversation outgrows the window; sessions so one agent
                can serve many conversations; and checkpoints so a workflow that died can be
                resumed.
              </>
            ),
          },
          {
            icon: '📟',
            title: `A command line of ${commandCount} commands`,
            description: (
              <>
                Run a task, hold a conversation, edit a repository with{' '}
                <code>effgen code</code>, watch live traffic with <code>effgen top</code>, compare
                models, render an HTML report — each with <code>--json</code> for scripting.
              </>
            ),
          },
          {
            icon: '📊',
            title: 'Surfaces you can show someone',
            description: (
              <>
                A real-time dashboard, an in-browser playground, a model and pricing browser, and
                shareable HTML reports and run cards. They are served by effGen itself and fetch
                nothing from a third party.
              </>
            ),
          },
          {
            icon: '🛡️',
            title: 'The operational half',
            description: (
              <>
                An OpenAI-compatible server with auth, roles, audit and rate limits; Prometheus
                metrics, tracing, SLOs and alerting; guardrails, sandboxed execution and a spend
                cap.
              </>
            ),
          },
        ]}
      />

      <h2>What {version} changed</h2>
      <p>
        {version}, released on 14 August 2026, is the first stable release. It supports Python{' '}
        {pythonVersions.join(', ')} and exports {publicNameCount} public names, none of which was
        removed or renamed on the way here. The theme running through it is control over where a
        model runs and visibility into what a run did — a backend that never answered now raises
        instead of returning something that reads like an answer, and a failed run says so.
      </p>

      <p>
        <strong>Three changes are breaking</strong>, each with a one-line migration:
      </p>
      <ul>
        <li>
          <strong>Python 3.10 is no longer supported.</strong> The floor is 3.11.
        </li>
        <li>
          <strong>
            <code>AgentConfig.raise_on_error</code> now defaults to <code>True</code>.
          </strong>{' '}
          A failed run raises its typed error instead of returning a plausible-looking string.
        </li>
        <li>
          <strong>A backend that was never reached raises</strong>{' '}
          <code>BackendUnreachableError</code>, whatever that flag says.
        </li>
      </ul>
      <p>
        <Link to="/migration">Migrating to {version}</Link> carries all three with the code each
        one asks you to change, and the smaller change to four enums that is worth knowing before
        you upgrade.
      </p>

      <h2>Where to go next</h2>

      <QuickLinks
        links={[
          {
            icon: '⚡',
            title: 'Quick start',
            description: 'An agent that answers a question, from an empty shell to a result.',
            path: '/quickstart',
          },
          {
            icon: '📦',
            title: 'Installation',
            description: 'The extras matrix, GPU wheels and the Apple Silicon path.',
            path: '/installation',
          },
          {
            icon: '🤖',
            title: 'Agents',
            description: 'Every AgentConfig field and every AgentResponse field.',
            path: '/agents',
          },
          {
            icon: '🎯',
            title: 'Presets',
            description: `The ${presetCount} ready-made configurations and what each one turns on.`,
            path: '/presets',
          },
          {
            icon: '🔌',
            title: 'Any OpenAI-compatible server',
            description: 'Point an agent at vLLM, Ollama, TGI or a gateway with base_url.',
            path: '/openai-compatible',
          },
          {
            icon: '📖',
            title: 'API reference',
            description: `All ${publicNameCount} names the package exports.`,
            path: '/api-reference',
          },
        ]}
      />

      <SeeAlso paths={['/quickstart', '/installation', '/migration']} />
    </DocPage>
  );
}
