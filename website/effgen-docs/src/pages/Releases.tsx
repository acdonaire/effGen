import { Tag } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { publicNameCount, pythonVersions, siteData, version } from '../siteData';

/** The published release history, newest first. Dates are from CHANGELOG.md. */
const HISTORY = [
  ['0.3.2', '2026-07-05', 'Results integrity, traceable grounding, a consistent server contract, and batch that survives real data.'],
  ['0.3.1', '2026-06-29', 'Evidence on every result: populated sources and citations, cost and latency in metadata, and a system prompt that steers every path.'],
  ['0.3.0', '2026-06-19', 'Stabilisation and hardening. No new providers or subsystems — fail-closed behaviour, a drift-aware model catalog and real GPU support.'],
  ['0.2.10', '2026-05-27', 'Security and supply chain: secret scanning, dependency auditing, an SBOM pipeline and a sandboxed code executor.'],
  ['0.2.9', '2026-05-23', 'Observability: structured logs with secret redaction, OpenTelemetry tracing and Prometheus metrics.'],
  ['0.2.8', '2026-05-21', 'Multimodal input — image, audio and video as ordinary message content across six cloud providers.'],
  ['0.2.7', '2026-05-20', 'The prompt library: a domain-organised template catalog with an evaluation harness and a playground.'],
  ['0.2.6', '2026-05-19', 'Document, media and communication tools — OCR, transcription, image analysis, document parsing, geo and mail.'],
  ['0.2.5', '2026-05-18', 'Free, no-auth tools for academic research, news, RSS, YouTube, social media, translation and QR codes.'],
  ['0.2.4', '2026-05-14', 'The model router: three composable policies, provider failover with retry, and cross-process rate-limit coordination.'],
  ['0.2.3', '2026-05-04', 'The provider ecosystem grew to nine backends — Groq, Together, Fireworks, Replicate and HuggingFace Inference.'],
  ['0.2.2', '2026-04-28', 'Gemini: newer model families, a thinking budget, Google Search grounding, the Files API and three native tools.'],
  ['0.2.1', '2026-04-25', 'The Cerebras backend, and a modernised OpenAI adapter with the reasoning tier.'],
  ['0.2.0', '2026-04-09', 'Native tool calling, guardrails, multi-agent orchestration, RAG pipelines and evaluation.'],
  ['0.1.3', '2026-03-25', 'Sub-agent depth limiting, and guidance for answering without a tool.'],
  ['0.1.2', '2026-03-12', 'Ten example agents and cross-model prompt work.'],
  ['0.1.1', '2026-03-06', 'Licence and packaging fixes.'],
  ['0.1.0', '2026-03-01', 'Foundation hardening: dynamic tool prompts, model-specific formatting and tool fallback.'],
  ['0.0.2', '2026-02-03', 'The retrieval and agentic-search tools.'],
  ['0.0.1', '2026-01-31', 'The first release: the agent system, task management and agent state.'],
] as const;

export default function Releases() {
  return (
    <DocPage
      subtitle="What each release changed, newest first."
      icon={<Tag size={48} />}
    >
      <h2>{version} — 14 August 2026</h2>
      <p>
        The first stable release. The theme running through it is
        control over where a model runs and visibility into what a run did. You can point effGen at
        any server speaking the OpenAI protocol, read back which tool calls a run made, wrap the
        agent loop in middleware, hand a single agent many conversations, choose how history is
        compacted, and resume a workflow that died half way through. A backend that never answered
        raises instead of returning something that reads like an answer.
      </p>
      <p>
        Around that sits a terminal coding agent, a branded command line that works on any
        terminal, a real-time dashboard, an in-browser playground, shareable HTML reports and run
        cards, a cross-provider model and pricing browser, a terminal mission-control view, a live
        model battle, and a browsable run and session history.
      </p>
      <p>
        Underneath both is the least visible and largest part of the release: a pass over
        everything that used to report the wrong thing confidently. A run that failed now says so.
        An unpriced model reports no cost instead of a made-up one. A turn whose every action
        failed is not a success. A tool call written in a shape effGen could not read no longer
        ends a turn with nothing.
      </p>

      <Callout type="warning" title="Three changes are breaking">
        <p>
          The Python floor is 3.11; <code>AgentConfig.raise_on_error</code> defaults to{' '}
          <code>True</code>; and a backend that was never reached raises{' '}
          <code>BackendUnreachableError</code> whatever that flag says.{' '}
          <Link to="/migration">Migrating to {version}</Link> carries all three with the code each
          one asks you to change. The public surface grew to {publicNameCount} names and nothing
          was removed or renamed.
        </p>
      </Callout>

      <h3>Connecting to models</h3>
      <p>
        <strong>Point effGen at any OpenAI-compatible server.</strong> <code>base_url</code> reaches{' '}
        <code>load_model()</code> and <code>AgentConfig</code>, so effGen can drive a model you
        already serve — vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, LiteLLM, a gateway or a
        corporate proxy — instead of loading a second copy of the weights inside the agent process.
      </p>

      <CodeBlock
        code={`from effgen import load_model

model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
)`}
        caption={
          <>
            The endpoint also comes from <code>EFFGEN_BASE_URL</code>,{' '}
            <code>OPENAI_BASE_URL</code> or <code>OPENAI_API_BASE</code>, in that order. Calls
            report no price rather than a fabricated $0, and{' '}
            <code>list_served_models()</code> asks the endpoint what it has. See{' '}
            <Link to="/openai-compatible">Any OpenAI-compatible server</Link>.
          </>
        }
      />

      <p>
        <strong>A multi-turn tool loop you can write by hand, on any provider.</strong>{' '}
        <code>build_assistant_message()</code> and <code>build_tool_result_message()</code> are on{' '}
        every adapter and build each provider's own message shape, so one loop runs against all of
        them rather than only the first. <Link to="/tool-calling">Tool calling</Link> has it.
      </p>

      <p>
        <strong>Python 3.14 is supported</strong> — installed and run, not just resolved. The
        supported set is {pythonVersions.join(', ')}.
      </p>

      <h3>The agent surface</h3>

      <p>
        <strong>Middleware around the agent loop.</strong> Hooks at three points — the run, each
        model call, each tool call — each with a <em>before</em> and an <em>after</em>. A{' '}
        <em>before</em> hook can rewrite the request or short-circuit it entirely; an{' '}
        <em>after</em> hook can transform the result. <em>Before</em> hooks run in order and{' '}
        <em>after</em> hooks in reverse, so middleware nest.
      </p>

      <CodeBlock
        filename="budget.py"
        code={`from effgen import Agent, AgentConfig
from effgen.core.middleware import AgentMiddleware
from effgen.tools.builtin import Calculator

class ToolBudget(AgentMiddleware):
    def __init__(self, limit=1):
        self.limit, self.used = limit, 0

    def before_tool_call(self, ctx):
        if self.used >= self.limit:
            return "Skipped: this run has spent its tool budget."
        self.used += 1
        return None

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    tools=[Calculator()],
    tool_calling_mode="react",
    temperature=0.0,
    middleware=[ToolBudget(limit=1)],
))
r = agent.run("What is 4817 * 236, and then what is that plus 1000?")
print(r.tool_calls.names)
print(r.output.strip()[:90])`}
      />

      <Terminal
        command="python budget.py"
        output={`['calculator']
1136812`}
        caption={
          <>
            One call was allowed and one was made; the second was refused by the hook.{' '}
            <code>LoggingMiddleware</code> and <code>ToolApprovalMiddleware</code> ship, and{' '}
            <code>run(..., middleware=[...])</code> adds hooks for one call. See{' '}
            <Link to="/middleware">Middleware</Link>.
          </>
        }
      />

      <p>
        <strong>One agent, many conversations.</strong> <code>run(..., session=...)</code> builds
        the prompt from that conversation's history and appends the turn to it, restoring the
        agent's own session and memory afterwards — including when the run fails. A server handling
        many users no longer needs an agent object per user.
      </p>

      <CodeBlock
        filename="sessions.py"
        code={`from effgen import create_agent

agent = create_agent("minimal", "openai:gpt-5-nano")

agent.run("My dog is named Pixel.", session="user-123")
agent.run("My cat is named Mote.", session="user-456")

print(agent.run("What is my dog called?", session="user-123").text.strip())
print(agent.run("What is my cat called?", session="user-456").text.strip())`}
      />

      <Terminal command="python sessions.py" output={`Pixel
Mote.`} />

      <p>
        <strong>Pluggable context compaction.</strong> What gets dropped when a conversation
        outgrows the window is now a strategy: <code>SummarizeOldest</code> (the default, with
        behaviour unchanged), <code>DropOldest</code> (no model call, nothing invented),{' '}
        <code>KeepFirstAndLast</code> (the turns carrying the task survive verbatim) and{' '}
        <code>KeepToolResults</code> (the evidence stays, the reasoning is compacted). Choose one
        with <code>AgentConfig(compaction_strategy=DropOldest())</code>, or subclass{' '}
        <code>CompactionStrategy</code>. <code>AgentConfig(tokenizer=...)</code> measures the
        history in the units the window is measured in rather than characters divided by four. See{' '}
        <Link to="/compaction">Context compaction</Link>.
      </p>

      <p>
        <strong>A workflow that died part way through can be resumed.</strong>{' '}
        <code>WorkflowDAG.run()</code> takes a <code>checkpoint=</code> store and a{' '}
        <code>run_id=</code>; running the same line again after a crash continues where it stopped.
        Completed nodes are not re-run and their outputs flow downstream, failed nodes are retried,
        and a finished run replays its stored outputs without calling a model — so a retrying job
        runner cannot double-bill you. There is no separate resume call: an unknown run id starts
        from the beginning and a known one continues. See{' '}
        <Link to="/checkpointing">Checkpointing and resumable runs</Link>.
      </p>

      <p>
        <strong>
          <code>AgentResponse.tool_calls</code> reports the calls, not just how many.
        </strong>{' '}
        Each entry carries <code>name</code>, <code>arguments</code>, <code>result</code>,{' '}
        <code>duration</code>, <code>error</code> and the <code>iteration</code> it was made on,
        with <code>failed</code> and <code>by_name()</code> to narrow them. Iterating the field used
        to raise <code>TypeError: 'int' object is not iterable</code>. It still compares and casts
        as the count.
      </p>

      <h3>The coding agent</h3>
      <p>
        <strong>
          <code>effgen code</code> is a coding agent in the terminal.
        </strong>{' '}
        It reads your workspace, proposes edits as unified diffs, and writes nothing until you say
        so. <code>--undo</code> rolls the last change back from a journal bounded to{' '}
        {siteData.code.undo_journal_entries} entries.
      </p>

      <CodeBlock
        language="bash"
        code={`effgen code "add a --dry-run flag to the importer"
effgen code --review                      # one read-only pass
effgen code --session-id my-refactor      # continue where you left off`}
      />

      <p>
        It runs in one of four permission modes that gate every write, every shell command and every
        commit. Writes are confined to the workspace, and a hunk that no longer applies is reported
        rather than clobbering the file. An interactive session keeps one run record across turns
        and carries {siteData.code.slash_command_count} slash commands. It is repository-aware —
        branch, status and a layout inventory that honours <code>.gitignore</code> go into the
        prompt, and an <code>AGENTS.md</code> brief is read when present. Git actions run through an
        allow-list, so push, reset, checkout, clean, rebase and force are refused before a
        subprocess starts, including when the model tries to reach them through the shell. See{' '}
        <Link to="/cli/code">effgen code</Link>.
      </p>

      <h3>Surfaces you can show someone</h3>

      <ApiTable
        headers={['Surface', 'What it is']}
        rows={[
          [
            <Link to="/dashboard">A real-time dashboard</Link>,
            'Per-model and per-provider cost, latency percentiles that are real percentiles, an error breakdown, a run waterfall, a model catalog panel and a history panel. Every chart is drawn locally.',
          ],
          [
            <Link to="/playground">An in-browser playground</Link>,
            'Model and preset pickers, tool toggles, the run’s tool trace, and copy-as-curl, copy-as-CLI and copy-as-Python for the form you filled in.',
          ],
          [
            <Link to="/catalog">A model and pricing browser</Link>,
            'In the terminal and in the dashboard, with search, provider, capability, context and price filters, sorting and paging.',
          ],
          [
            <Link to="/cli/reports">HTML reports and run cards</Link>,
            <>
              <code>--report out.html</code> for compare, eval, cost and loadtest;{' '}
              <code>run --card</code> for a single run; and <code>effgen report</code> to render a
              saved result after the fact.
            </>,
          ],
          [
            <Link to="/cli/top">effgen top</Link>,
            'A terminal mission-control view over the telemetry you already collect — activity, traffic, per-model, spend and GPU panels, each stating the window and process it describes.',
          ],
          [
            <Link to="/compare">effgen battle</Link>,
            'Races several models on one prompt and reports the tally, the cost and an optional judge’s verdict separately from the measurements.',
          ],
          [
            'A live topology graph',
            'Multi-agent topology, terminal trace timelines, a workflow DAG diagram and a run waterfall.',
          ],
          [
            <Link to="/cli/appearance">Named terminal themes</Link>,
            <>
              {siteData.cli.themes.map((theme, i) => (
                <span key={theme}>
                  {i > 0 ? ', ' : ''}
                  <code>{theme}</code>
                </span>
              ))}
              , drawn from one shared palette the dashboard reads too.
            </>,
          ],
        ]}
        caption={`Every web surface is self-contained: no CDN, no external font and nothing fetched at view time — enforced by a test that inspects what a browser would fetch rather than by searching for a substring. Across the shipped files it finds ${siteData.web.external_references} external references.`}
      />

      <h3>History, projects and the command line</h3>
      <ul>
        <li>
          <strong>Durable run and session history.</strong> Every run is recorded with its model,
          provider, tokens, cost, status and task, keyed by the same run id its trace spans carry.
          Runs from the command line, a script and the server share one history and survive a
          restart — <Link to="/cli/history">Runs and sessions history</Link>.
        </li>
        <li>
          <strong>Project scaffolding.</strong>{' '}
          <code>effgen quickstart --init</code> writes a configuration, an{' '}
          <code>.env</code> template, a runnable example and a <code>.gitignore</code>, and puts a
          daily spend cap in force when none is configured —{' '}
          <Link to="/first-project">Your first project</Link>.
        </li>
        <li>
          <strong>Flags and output that behave the same everywhere.</strong>{' '}
          <code>--json</code> on every command that had no machine output, and{' '}
          <code>--json</code> stdout is now a single valid document on a pipe and on a terminal,
          with no spinner, table or warning mixed into it. <code>-o</code> picks its format from the
          extension. Thirteen short flags now mean the same thing across commands, and a bare group
          command prints its own help and exits 0.
        </li>
        <li>
          <strong>Your own prompt templates load beside the shipped ones.</strong>{' '}
          <code>EFFGEN_PROMPTS_DIR</code> names one or more directories, so a team's library sits
          next to the built-in one without a fork.
        </li>
      </ul>

      <h3>A documentation site</h3>
      <p>
        effGen has a project site and a documentation site, both static and both published from the
        framework repository. Every public definition documents its arguments and its result: the
        package was walked module by module, and a gate fails when a public definition is added
        without that.
      </p>

      <h3>What was fixed</h3>
      <p>
        The largest part of the release is a pass over reporting. In summary:
      </p>
      <ul>
        <li>
          <strong>Results that report what actually happened.</strong> A turn whose every action
          failed is not a success; an unpriced model reports no cost rather than <code>$0</code>;
          and a failed run carries the reason it stopped.
        </li>
        <li>
          <strong>Tool calling across providers.</strong> Call shapes effGen could not read no
          longer end a turn with nothing, and the reported shape is the same on every adapter.
        </li>
        <li>
          <strong>Errors that name the fix.</strong> A wrong model id suggests near matches, a
          missing key names the variable, and a CUDA mismatch names the torch build against the
          driver.
        </li>
        <li>
          <strong>A rate limit no longer multiplies.</strong> Three layers each retried a throttled
          call and multiplied rather than shared a budget.
        </li>
        <li>
          The server and the API, security, guardrails and sandboxing, local models and GPUs,
          documents, RAG and batch input, the built-in tools, the terminal and web surfaces, and
          installation and packaging each have their own section in the changelog.
        </li>
      </ul>

      <h2>Earlier releases</h2>

      <ApiTable
        headers={['Version', 'Date', 'What it was']}
        rows={HISTORY.map(([release, date, summary]) => [<code>{release}</code>, date, summary])}
        caption={
          <>
            Summarised from the framework's <code>CHANGELOG.md</code>, which carries every entry in
            full. Code samples on this page are from {version}; earlier releases' examples are in
            the changelog, and some of them describe APIs that have since gained better ones.
          </>
        }
      />

      <Callout type="note" title="Where the record lives">
        <p>
          <code>CHANGELOG.md</code> in the framework repository is the full record, and{' '}
          <code>NEWS.md</code> is the readable version of the current release. This page summarises
          both; where they disagree with anything here, they are right.
        </p>
      </Callout>

      <SeeAlso paths={['/migration', '/introduction', '/api-reference']} />
    </DocPage>
  );
}
