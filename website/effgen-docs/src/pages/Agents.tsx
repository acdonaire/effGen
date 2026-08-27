import { Bot } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  MermaidDiagram,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';
import type { Param } from '../components/docs';
import { siteData, version } from '../siteData';
import { siteHref } from '../siteLinks';

/**
 * What each `AgentConfig` field is for, in the words of the class's own
 * docstring.
 *
 * The name, the type and the default are **not** written here — they come from
 * `data/effgen.json`, which is generated from the installed dataclass. This map
 * supplies only the sentence, and `paramsFor` below reports a field that has
 * gained or lost a description rather than letting the table drift.
 */
const CONFIG_NOTES: Record<string, string> = {
  model: 'A loaded model instance, or an id string such as "openai:gpt-5-nano". The only field with no default.',
  name: 'Agent identifier. Defaults to the model id, or "agent" for a model instance.',
  tools: 'The tools this agent may call. Each one is a BaseTool instance.',
  system_prompt: 'System-level instructions, applied on every path including streaming and the native tool loop.',
  max_iterations: 'How many times the loop may go round on one task before it stops.',
  temperature: 'Generation temperature. A run() keyword of the same name overrides it for one call.',
  max_tokens: 'Output-token budget for every run. None lets the model pick a size-aware default.',
  top_p: 'Nucleus-sampling threshold.',
  top_k: 'Top-k sampling cutoff. Providers that do not support it ignore it.',
  seed: 'Sampling seed. With temperature=0 this reproduces a generation exactly on Gemini, Groq and the local engines; OpenAI documents its seed as best-effort rather than a guarantee.',
  presence_penalty: 'Penalises tokens already present anywhere in the text.',
  frequency_penalty: 'Penalises tokens in proportion to how often they already appeared — the anti-repetition knob for long text.',
  repetition_penalty: 'Multiplicative repeat penalty, used by the local and HuggingFace engines.',
  mode: 'Default execution mode for run(). SINGLE never decomposes on its own; AUTO lets the router decide per call.',
  enable_sub_agents: 'Whether the agent may spawn sub-agents for parts of a task.',
  enable_memory: 'Whether the memory subsystem is active.',
  enable_streaming: 'Whether tokens are streamed as they arrive.',
  max_context_length: 'Context window to plan against. None reads it from the model.',
  router_config: 'Settings for the sub-agent router.',
  sub_agent_config: 'Settings for the sub-agent manager.',
  model_config: 'Engine options passed through when the model is loaded from an id.',
  require_model: 'Whether a string model must load at construction. True means a typo or a missing key fails immediately instead of at the first run.',
  provider: 'Explicit provider for a bare model id — the same choice the "provider:model" prefix makes.',
  base_url: 'Endpoint for a server speaking the OpenAI protocol. Giving one loads the model through that server instead of in this process.',
  api_key: 'Credential for that endpoint. A local server that checks nothing needs none.',
  middleware: 'Hooks around the run, each model call and each tool call.',
  compaction_strategy: 'How the conversation is shortened as it approaches the window. Accepts a strategy, a class, or a name.',
  tokenizer: 'Anything with count_tokens(text) or encode(text), used to measure history in the units the window is measured in.',
  raise_on_error: 'Whether a failed run raises its typed error. True since 1.0.0.',
  system_prompt_template: 'A template for the assembled system prompt, replacing the built-in one.',
  verbose_tools: 'Whether tool descriptions are sent in full. None follows the model.',
  fallback_chain: 'A mapping from tool name to the tools tried when it fails.',
  enable_fallback: 'Whether those tool fallback chains apply.',
  max_sub_agent_depth: 'How deep sub-agents may nest.',
  tool_calling_mode: '"auto", "native", "react" or "hybrid" — how tools are offered to the model.',
  output_format: 'Default output format for every run: "json", "yaml", "csv" or None.',
  output_schema: 'Default JSON Schema every run must produce.',
  guardrails: 'A GuardrailChain, or the name of a guardrail preset.',
  memory_config: 'Memory settings — token and message caps, the long-term backend, and whether old context is summarised.',
  models: 'Additional models this agent may fall back to or route between.',
  speculative_execution: 'Run on two models and take the first that succeeds.',
  approval_callback: 'Called before a tool runs, to approve or refuse it.',
  approval_mode: '"never", "always", "first_time" or "dangerous_only".',
  approval_timeout: 'Seconds to wait for approval. 0 waits forever.',
  clarification_callback: 'Called when the agent needs the user to choose between options.',
  input_callback: 'Called when the agent needs a line of input from the user.',
  stable_system_prompt: 'Keep the system prompt at a fixed position so a provider can cache the prefix.',
  cache_system_prompt: "Mark the system prompt's last block for Anthropic prompt caching.",
  cache_tools: 'Mark the last tool spec for Anthropic prompt caching.',
};

const RESPONSE_NOTES: Record<string, string> = {
  output: 'The answer. `text` and `content` are read-only aliases, and str(response) is the same string.',
  success: 'Whether the run produced an answer. False only when raise_on_error is off.',
  mode: 'The mode the run actually used.',
  iterations: 'How many times the loop went round.',
  tool_calls: 'The calls the run made. Iterable, and still compares and casts as the count.',
  tokens_used: 'Total tokens across every model call in the run.',
  execution_time: 'Wall-clock seconds for the whole run.',
  execution_trace: 'One entry per step, for reconstructing what happened.',
  execution_tree: 'The same steps as a tree, when sub-agents were involved.',
  routing_decision: 'Which model was chosen and why, when routing was in play.',
  metadata: 'Cost, tokens, latency, partial_output, input_redaction and anything a subsystem attached.',
  citations: 'Citations built from what the run retrieved, never scraped from the prose.',
  sources: 'The deduplicated source URLs behind those citations.',
  task: 'The task the run was given.',
  model: 'The model id that answered.',
  provider: 'The provider that served it.',
  started_at: 'When the run started, as an ISO-8601 string.',
};

/**
 * Join the generated field list to the sentences above.
 *
 * A field the framework no longer has cannot appear, because the list is the
 * framework's. A field that has appeared and has no sentence yet is marked
 * rather than silently described as nothing.
 */
function paramsFor(
  fields: typeof siteData.api.agent_config,
  notes: Record<string, string>,
): Param[] {
  return fields.map((field) => ({
    name: field.name,
    type: field.type,
    default: field.required ? undefined : (field.default ?? '(empty)'),
    required: field.required,
    description: notes[field.name] ?? 'Not described on this page yet.',
  }));
}

const LOOP = `flowchart TD
    Start["agent.run(task)"] --> Build["Build the prompt:<br/>system prompt + history + task"]
    Build --> Call["Call the model"]
    Call --> Decide{"Answer or<br/>tool call?"}
    Decide -->|"answer"| Done["AgentResponse"]
    Decide -->|"tool call"| Run["Run the tool"]
    Run --> Obs["Append the ToolResult"]
    Obs --> Cap{"iterations<br/>&lt; max_iterations?"}
    Cap -->|"yes"| Call
    Cap -->|"no"| Stop["Stop and report why"]
`;

export default function Agents() {
  return (
    <DocPage
      subtitle="The Agent class, the config it takes and the response it returns."
      icon={<Bot size={48} />}
    >
      <p>
        An agent is a model, a set of tools and a loop between them.{' '}
        <code>Agent</code> holds the loop, <code>AgentConfig</code> holds every setting it obeys,
        and <code>run()</code> returns an <code>AgentResponse</code> carrying the answer and
        everything that is true about how it was reached.
      </p>

      <p className="doc-crosslink">
        This page is the reference: every field, every return value, every failure. For what an
        agent is for and one worked run end to end, see <a href={siteHref('/agents')}>the agents
        page</a> on the main site.
      </p>

      <h2>The shortest agent</h2>

      <CodeBlock
        filename="agent.py"
        code={`from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import Calculator

model = load_model("openai:gpt-5-nano")

agent = Agent(AgentConfig(
    model=model,
    tools=[Calculator()],
    system_prompt="You are a careful arithmetic assistant.",
    temperature=0.0,
    max_iterations=5,
))

response = agent.run("What is (17 * 23) + 12?")
print(response.output)`}
      />

      <Terminal
        command="python agent.py"
        output={`403

Explanation:
- 17 × 23 = 391
- 391 + 12 = 403`}
        caption={`Run against effGen ${version}.`}
      />

      <p>
        <code>model</code> is the only field with no default. Everything else has one, so a
        one-field config is legal: <code>Agent(AgentConfig(model="openai:gpt-5-nano"))</code> is a
        working agent with no tools. For a configuration someone has already worked out, start
        from a <Link to="/presets">preset</Link> instead.
      </p>

      <h2>The loop</h2>

      <MermaidDiagram
        chart={LOOP}
        title="What run() does"
        description="run() builds a prompt from the system prompt, the history and the task, and calls the model. If the model answers, the run returns an AgentResponse. If it asks for a tool, the tool runs, its result is appended, and the loop calls the model again until max_iterations is reached, at which point the run stops and reports why."
      />

      <p>
        How tools are offered to the model — as native function definitions, as text the model
        writes back in, or both — is <code>tool_calling_mode</code>, covered on{' '}
        <Link to="/tool-calling">Tool calling</Link>. Nothing else about the loop changes with it.
      </p>

      <h2>Constructing an agent</h2>

      <ParamTable
        nameLabel="Parameter"
        params={[
          {
            name: 'config',
            type: 'AgentConfig | None',
            default: 'None',
            description: 'The settings. Omitting it is only useful in subclasses that supply one.',
          },
          {
            name: 'session_id',
            type: 'str | None',
            default: 'None',
            description: (
              <>
                A stored conversation to load or create, so multi-turn context survives across
                processes. Per-call conversations use <code>run(session=...)</code> instead — see{' '}
                <Link to="/sessions">Sessions</Link>.
              </>
            ),
          },
        ]}
        caption="Agent(config=None, session_id=None)"
      />

      <h2>Running a task</h2>

      <ParamTable
        nameLabel="Parameter"
        params={[
          {
            name: 'task',
            type: 'str | Message | list[ContentPart]',
            required: true,
            description:
              'The task. A plain string, a multimodal Message, or a list of content parts — text is extracted and any image, audio or video parts go through the multimodal path.',
          },
          {
            name: 'mode',
            type: 'AgentMode | None',
            default: 'None',
            description:
              'Overrides config.mode for this call. AgentMode.AUTO lets the router decide from task complexity.',
          },
          {
            name: 'context',
            type: 'dict[str, Any] | None',
            default: 'None',
            description: 'Extra context for this call.',
          },
          {
            name: 'output_schema',
            type: 'dict | type[BaseModel] | None',
            default: 'None',
            description:
              'A JSON Schema dict or a Pydantic model class. The final output is then valid JSON matching it. Any other type raises TypeError.',
          },
          {
            name: 'output_model',
            type: 'type[BaseModel] | None',
            default: 'None',
            description: (
              <>
                A Pydantic model class. The output is validated and the parsed instance is stored
                in <code>response.metadata["parsed"]</code>. See{' '}
                <Link to="/generation">Generation controls</Link>.
              </>
            ),
          },
          {
            name: 'inputs',
            type: 'list[ContentPart] | None',
            default: 'None',
            description:
              'Multimodal parts made by image_from, audio_from or video_from. With these present the agent sends a structured Message.',
          },
          {
            name: 'session',
            type: 'Session | str',
            description:
              'A conversation, by object or by id. The prompt is built from that conversation and this turn is appended to it, so one agent can serve many conversations.',
          },
          {
            name: 'middleware',
            type: 'list[AgentMiddleware]',
            description: 'Hooks for this call only, appended to any on the config.',
          },
          {
            name: 'debug',
            type: 'bool',
            default: 'False',
            description: 'Attach a DebugTrace to the response.',
          },
        ]}
        caption={
          <>
            <code>
              Agent.run(task, mode=None, context=None, output_schema=None, output_model=None,
              inputs=None, **kwargs)
            </code>{' '}
            — <code>session</code>, <code>middleware</code> and <code>debug</code> arrive through{' '}
            <code>**kwargs</code>. Sampling keywords such as <code>temperature</code> and{' '}
            <code>max_tokens</code> do too, overriding the config for one call.
          </>
        }
      />

      <h2>AgentConfig</h2>
      <p>
        Every field, with the type and default the dataclass declares. Anything not passed keeps
        the default shown.
      </p>

      <ParamTable
        nameLabel="Field"
        params={paramsFor(siteData.api.agent_config, CONFIG_NOTES)}
        caption={
          <>
            Generated from the installed <code>AgentConfig</code> dataclass. A field listed as{' '}
            <code>(empty)</code> is built per instance — an empty list or dict.
          </>
        }
      />

      <Callout type="warning" title={`raise_on_error changed in ${version}`}>
        <p>
          It now defaults to <code>True</code>: a failed run raises its typed error rather than
          returning a response with <code>success=False</code> and a plausible-looking string in{' '}
          <code>output</code>. Set it to <code>False</code> to inspect the response yourself — and
          note that with the flag off, a failed run's <code>output</code> is effGen's report of
          what stopped it, while the model's own text is in{' '}
          <code>metadata["partial_output"]</code>. A backend that was never reached raises either
          way. <Link to="/migration">Migrating to {version}</Link> has the migration.
        </p>
      </Callout>

      <h2>AgentResponse</h2>
      <p>
        Not a string. <code>print(response)</code> prints the answer, and every field below says
        something about the run that produced it. It is imported from{' '}
        <code>effgen.core.agent</code> — it is not exported from the top-level package.
      </p>

      <ParamTable
        nameLabel="Field"
        params={paramsFor(siteData.api.agent_response, RESPONSE_NOTES)}
        caption={
          <>
            Generated from the installed <code>AgentResponse</code> dataclass. On top of these it
            carries <code>text</code> and <code>content</code> (aliases for <code>output</code>),{' '}
            <code>tool_call_count</code>, <code>to_dict()</code>, and <code>show()</code> /{' '}
            <code>trace()</code> for printing a run in a terminal.
          </>
        }
      />

      <CodeBlock
        filename="response.py"
        code={`from effgen import create_agent

agent = create_agent("math", "openai:gpt-5-nano")
r = agent.run("What is 6 * 7?")

print(r.text)                        # the answer; r.output and r.content are the same string
print(r.success, r.iterations)
print(r.tool_calls.total, r.tool_call_count)
print(r.model, r.provider)
print(r.metadata["cost_usd"], r.metadata["latency_ms"])`}
      />

      <Terminal
        command="python response.py"
        output={`42

Verification:
- Calculator result: 6 * 7 = 42
- Python result: print(6 * 7) -> 42

Conclusion: Indeed, 6 multiplied by 7 equals 42.
True 2
2 2
openai:gpt-5-nano openai
0.00065765 12403.7`}
        caption="Two tool calls over two loop iterations. r.model carries the id as it was given, prefix and all."
      />

      <h2>Reading the tool calls</h2>
      <p>
        <code>tool_calls</code> is a <code>ToolCallList</code>: iterate it for the calls
        themselves, or use it as the count. Each entry is a <code>ToolCall</code>.
      </p>

      <ApiTable
        headers={['Field', 'What it is']}
        rows={siteData.api.tool_call.map((field) => [
          <code>{field.name}</code>,
          <code>{field.type}</code>,
        ])}
        caption={
          <>
            The reading surface <code>ToolCallList</code> adds on top of a list:{' '}
            {siteData.api.tool_call_list.map((name, i) => (
              <span key={name}>
                {i > 0 ? ', ' : ''}
                <code>{name}</code>
              </span>
            ))}
            .
          </>
        }
      />

      <CodeBlock
        filename="calls.py"
        code={`from effgen import create_agent

agent = create_agent("math", "gemini:gemini-3.1-flash-lite")
r = agent.run("What is 4817 * 236?")

for call in r.tool_calls:
    print(call.iteration, call.name, call.arguments, "->", call.error or call.result)

print("names:", r.tool_calls.names)
print("failed:", r.tool_calls.failed.total)
print("calculator calls:", r.tool_calls.by_name("calculator").total)`}
      />

      <Terminal command="python calls.py" output={`1 calculator {"expression": "4817 * 236"} -> 1136812
names: ['calculator']
failed: 0
calculator calls: 1`} />

      <Callout type="note" title="How much of a call is recorded depends on the provider">
        <p>
          <code>iteration</code>, <code>arguments</code> and <code>result</code> are filled in by
          the adapter that made the call. The Gemini adapter records all three. The OpenAI adapter
          records the name and leaves the rest <code>None</code>, so the same loop against{' '}
          <code>openai:gpt-5-nano</code> prints <code>None calculator None -&gt; None</code>.{' '}
          <code>names</code>, <code>failed</code> and <code>by_name()</code> report the same thing
          on both.
        </p>
      </Callout>

      <Callout type="note" title={`tool_calls changed in ${version}`}>
        <p>
          It used to be an integer, and iterating it raised{' '}
          <code>TypeError: 'int' object is not iterable</code>. It still compares and casts as the
          count, so <code>tool_calls == 2</code> and <code>tool_calls &gt; 0</code> are unchanged,
          and <code>to_dict()</code> keeps the count under its original key while adding{' '}
          <code>tool_call_details</code>.
        </p>
      </Callout>

      <h2>When a run fails</h2>

      <ApiTable
        headers={['Error', 'When', 'What to do']}
        rows={[
          [
            <code>BackendUnreachableError</code>,
            'A refused connection, an unresolvable host or a missing route — the backend was never reached.',
            <>
              Raises whatever <code>raise_on_error</code> says, by design: there is no result to
              inspect. Catch it where you want to handle it.
            </>,
          ],
          [
            <code>ModelAuthError</code>,
            'The provider rejected the credential.',
            <>
              Check the key with <code>effgen doctor</code>.
            </>,
          ],
          [
            <code>ModelNotFoundError</code>,
            'The provider does not serve that model id.',
            <>
              <code>effgen models list --provider &lt;name&gt;</code>, then{' '}
              <code>effgen models refresh</code>.
            </>,
          ],
          [
            <code>RateLimitExceeded</code>,
            "The provider's limit was hit.",
            <>
              Retryable. A <Link to="/routing">router</Link> fails over to another provider on
              this.
            </>,
          ],
          [
            <>The iteration cap</>,
            <>
              The loop reached <code>max_iterations</code> without the model answering.
            </>,
            <>
              Raise the cap, simplify the task, or set <code>raise_on_error=False</code> and read{' '}
              <code>metadata["partial_output"]</code>.
            </>,
          ],
        ]}
      />

      <CodeBlock
        filename="inspect_failure.py"
        code={`from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    max_iterations=1,
    raise_on_error=False,          # 1.0.0 default is True
))
r = agent.run("Research the full history of the Byzantine Empire and cite ten sources.")

print(r.success)
print("reason:", r.metadata.get("reason"))
print("model's own text:", (r.metadata.get("partial_output") or "")[:60])`}
      />

      <Terminal command="python inspect_failure.py" output={`success: False
reason: max_iterations_partial
output: Stopped after 1 iteration without a final answer: 'gpt-5-nano' was still taking tool steps
the model's own text: 1136812`} />

      <SeeAlso paths={['/presets', '/configuration', '/tool-calling']} />
    </DocPage>
  );
}
