// What changed in each release, adapted from the framework's own CHANGELOG.md
// and NEWS.md. Nothing here is a summary written from memory: each group below
// carries the heading the changelog files it under, and each item is one of the
// entries under that heading.
//
// Counts and version numbers are not written here — the page reads them from
// data/effgen.json, so this file cannot disagree with the installed package.

export interface ChangeItem {
  title: string;
  body: string;
  /** A sample, run before it was written down, and what it printed. */
  code?: { source: string; language?: "python" | "bash"; output?: string };
}

export interface ChangeGroup {
  id: string;
  title: string;
  lede: string;
  accent: string;
  items: ChangeItem[];
}

export interface BreakingChange {
  title: string;
  why: string;
  /** The one line that migrates it, and what it printed when it was run. */
  migration: { source: string; language: "python" | "bash"; output?: string };
  note?: string;
}

/* ── 1.0.0 ───────────────────────────────────────────────────────────────── */

/** Released 14 August 2026. The tag commit is dated the 15th. */
export const RELEASE_DATE = "14 August 2026";

/** `git rev-list --count v0.3.2..v1.0.0` in the framework repository. */
export const COMMITS_SINCE_0_3_2 = 640;

export const breakingChanges: BreakingChange[] = [
  {
    title: "Python 3.10 is no longer supported",
    why:
      "The floor is 3.11. tomllib, asyncio.timeout, datetime.UTC and the TimeoutError unification are all standard library from 3.11, and the package carried a hand-written fallback for each.",
    migration: {
      source: "python --version   # 3.11, 3.12, 3.13 or 3.14",
      language: "bash",
      output: "Python 3.11.15",
    },
    note: "Nothing in the API changed.",
  },
  {
    title: "AgentConfig.raise_on_error defaults to True",
    why:
      "A failed run raises its typed error instead of returning an AgentResponse with success=False and a plausible-looking string in .output — which a caller reading .output without checking .success never noticed.",
    migration: {
      source: `from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="openai:gpt-5-nano", raise_on_error=False))
response = agent.run("Reply with the single word: ready")

print(response.success, "|", response.text)`,
      language: "python",
      output: "True | ready",
    },
    note:
      "raise_on_error=False is also the documented setting for batch evaluation: with the flag off, a failed run's output is effGen's report of what stopped it, and the model's own text is in metadata[\"partial_output\"].",
  },
  {
    title: "An unreachable backend raises whatever that flag says",
    why:
      "A refused connection, an unresolvable host or a missing route is classified separately from a server that answered badly, and raises BackendUnreachableError. A task that ran and failed is a result you can inspect; a backend that was never reached is not, and returning one is how a whole batch completes against nothing and still looks healthy.",
    migration: {
      source: `from effgen import Agent, AgentConfig
from effgen.models.errors import BackendUnreachableError

agent = Agent(AgentConfig(
    model="openai:gpt-5-nano",
    base_url="http://127.0.0.1:9/v1",
    api_key="not-used",
    raise_on_error=False,
))

try:
    agent.run("Anything.")
except BackendUnreachableError as error:
    print(type(error).__name__)`,
      language: "python",
      output: `BackendUnreachableError`,
    },
    note: "There is no opt-out, by design. Catch the error where you want to handle it.",
  },
];

export const oneZeroGroups: ChangeGroup[] = [
  {
    id: "models",
    title: "Connecting to models",
    lede: "Where the weights live stopped being effGen's decision.",
    accent: "#00e5ff",
    items: [
      {
        title: "Point effGen at any OpenAI-compatible server",
        body:
          "base_url reaches load_model() and AgentConfig, so effGen drives a model you already serve — vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, LiteLLM, a gateway or a corporate proxy — instead of loading a second copy of the weights inside the agent process. The endpoint also comes from EFFGEN_BASE_URL, OPENAI_BASE_URL or OPENAI_API_BASE, in that order.",
        code: {
          source: `from effgen.models import load_model

model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    provider="openai_compatible",
    base_url="http://127.0.0.1:8000/v1",
)`,
          language: "python",
        },
      },
      {
        title: "The server's ids are the server's",
        body:
          "No OpenAI catalog is consulted: the full sampling surface is offered, calls report no price rather than a fabricated $0, and list_served_models() asks the endpoint what it has. Pass context_length= when your server's window is not the assumed 32,768 tokens; effGen now warns when it is assuming, naming the value and the flag that sets the real one, instead of failing later at a size nobody chose.",
      },
      {
        title: "A tool loop you can write by hand, on any provider",
        body:
          "build_assistant_message() and build_tool_result_message() on BaseModel, and so on every adapter, build each provider's own message shape. A loop written once runs against OpenAI, Gemini, Anthropic, Groq, Together, Fireworks, Cerebras, Replicate and HF Inference instead of only the first.",
      },
      {
        title: "Python 3.14 is supported",
        body:
          "Installed and run, not just resolved. One caveat, because it changes the install line: plain pip install effgen[all] does not resolve on 3.14, because pip backtracks through the wide vLLM range into a release pinned to numba==0.61. On 3.14, install the extras through the shipped lock file first.",
        code: {
          source: `pip install -r requirements-all-py314-lock.txt
pip install --no-deps effgen`,
          language: "bash",
        },
      },
    ],
  },
  {
    id: "agents",
    title: "The agent surface",
    lede: "The extension points people arrive expecting, under the names they have elsewhere.",
    accent: "#00ff88",
    items: [
      {
        title: "Middleware around the agent loop",
        body:
          "Hooks at three points — the run, each model call, each tool call — each with a before and an after. A before hook can rewrite the request or short-circuit it entirely; an after hook can transform the result. Before hooks run in order and after hooks in reverse, so middleware nest. LoggingMiddleware and ToolApprovalMiddleware ship, and run(..., middleware=[...]) adds one for a single call.",
        code: {
          source: `from effgen import Agent, AgentConfig, LoggingMiddleware

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    middleware=[LoggingMiddleware()],
))
print(agent.run("Reply with one word: hello").text)`,
          language: "python",
          output: "Hello",
        },
      },
      {
        title: "One agent, many conversations",
        body:
          "run(..., session=...) builds the prompt from that conversation's history and appends the turn to it, restoring the agent's own session and memory afterwards, including when the run fails. A server handling many users no longer needs an agent object per user, nor history bookkeeping outside the framework.",
        code: {
          source: `agent.run("My dog is named Pixel.", session="user-123")
agent.run("My cat is named Mote.",  session="user-456")`,
          language: "python",
        },
      },
      {
        title: "Compaction is a strategy",
        body:
          "What gets dropped when a conversation outgrows the window is now yours to choose: SummarizeOldest (the default, unchanged), DropOldest (no model call, nothing invented), KeepFirstAndLast (the turns carrying the task survive verbatim) and KeepToolResults (the evidence stays, the reasoning is compacted) — or subclass CompactionStrategy. AgentConfig(tokenizer=...) measures the history in the units the window is measured in, rather than characters divided by four.",
      },
      {
        title: "A workflow that died part way through can be resumed",
        body:
          "WorkflowDAG.run() takes a checkpoint= store and a run_id=. Run the same line again after a crash and it continues where it stopped. Completed nodes are not re-run and their outputs flow downstream, failed nodes are retried, and a finished run replays its stored outputs without calling a model, so a retrying job runner cannot double-bill you. There is no separate resume call: an unknown run id starts from the beginning and a known one continues.",
        code: {
          source: `from effgen import FileCheckpointStore, WorkflowDAG, WorkflowNode

store = FileCheckpointStore()          # ~/.effgen/workflows by default
result = dag.run("Write the Q3 summary.", checkpoint=store, run_id="q3-summary")`,
          language: "python",
        },
      },
      {
        title: "AgentResponse.tool_calls reports the calls, not just how many",
        body:
          "Each entry is a ToolCall carrying name, arguments, result, duration, error and the iteration it was made on, with .failed and .by_name() to narrow them. Iterating the field used to raise TypeError: 'int' object is not iterable. It still compares and casts as the count, so tool_calls == 2 is unchanged, and .total says the number plainly. How much a call carries depends on the provider.",
        code: {
          source: `for call in result.tool_calls:
    print(call.name, call.arguments, "->", call.error or call.result)`,
          language: "python",
        },
      },
      {
        title: "load_env()",
        body:
          "Runs the same .env search the command line does, so a library script picks up the keys the CLI already finds. It honours EFFGEN_NO_DOTENV and never overwrites a value you exported.",
      },
    ],
  },
  {
    id: "code",
    title: "The coding agent",
    lede: "effgen code reads your workspace, proposes edits as unified diffs, and writes nothing until you say so.",
    accent: "#a78bfa",
    items: [
      {
        title: "Four permission modes, and an undo",
        body:
          "plan, ask, auto-edit and yes gate every write, every shell command and every commit. Writes are confined to the workspace, and a hunk that no longer applies is reported rather than clobbering the file. --undo rolls the last change back from a journal bounded to 100 entries.",
        code: {
          source: `effgen code "add a --dry-run flag to the importer"
effgen code --review                      # one read-only pass
effgen code --session-id my-refactor      # continue where you left off`,
          language: "bash",
        },
      },
      {
        title: "It knows the repository it is in",
        body:
          "Branch, status and a layout inventory that honours .gitignore go into the prompt, and an AGENTS.md brief is read when present. Git actions run through an allow-list, so push, reset, checkout, clean, rebase and force are refused before a subprocess starts — including when the model tries to reach them through the shell. A commit is confirmed like a write, uses the repository's own identity, and leaves your other staged work alone.",
      },
      {
        title: "A session that survives the process",
        body:
          "An interactive session keeps one run record across turns and carries a slash-command set — /plan, /diff, /apply, /reject, /undo, /run, /test, /context, /mode, /model, /cost, /trace, /git, /review, /compact, /save, /session and more. --session-id resumes it later. --review makes one read-only pass with a tool set that holds nothing that writes, runs or executes.",
      },
      {
        title: "Scriptable",
        body:
          "-p, --json and piped stdin run the single-shot path with byte-clean stdout. effgen doctor reports coding readiness — workspace, sandbox backend, git — and quickstart and tutorial include a coding step that writes and runs a real program.",
      },
    ],
  },
  {
    id: "surfaces",
    title: "Surfaces you can show someone",
    lede: "Everything here is self-contained: no CDN, no external font, nothing fetched at view time — enforced by a test that inspects what a browser would fetch.",
    accent: "#ffd700",
    items: [
      {
        title: "A real-time dashboard",
        body:
          "Per-model and per-provider cost, latency percentiles that are real percentiles, an error breakdown, a run waterfall, a model catalog panel and a history panel. Every chart is drawn locally.",
      },
      {
        title: "An in-browser playground",
        body:
          "On the existing chat endpoint, with model and preset pickers, tool toggles, the run's tool trace, and copy-as-curl, copy-as-CLI and copy-as-Python for the form you filled in.",
      },
      {
        title: "A cross-provider model and pricing browser",
        body:
          "In the terminal and in the dashboard: search, provider, capability, context and price filters, sorting and paging. models info shows every provider that serves a shared id.",
        code: {
          source: `effgen models browse --tools --min-context 100000 --sort price-in`,
          language: "bash",
        },
      },
      {
        title: "Shareable reports and run cards",
        body:
          "--report out.html for compare, eval, cost and loadtest, plus run --card and runs show <id> --card, and effgen report <result.json> to render a saved document after the fact. A generated report is inert: model output containing markup renders as text, and only http and https links keep an href.",
      },
      {
        title: "effgen top, and effgen battle",
        body:
          "top (alias monitor) is a terminal mission-control view over the telemetry you already collect — activity, traffic, per-model, spend and GPU panels, each stating the window and process it describes. battle races several models on one prompt side by side and reports the tally, the cost and an optional judge's verdict separately from the measurements.",
      },
      {
        title: "Graphs, palettes and themes",
        body:
          "A live multi-agent topology graph, terminal trace timelines, a workflow DAG diagram (workflow run --diagram) and a run waterfall. A command palette and keyboard-first navigation on both web surfaces, with a skip link, jump links, focus restoration and screen-reader announcements. Named terminal themes — default, high-contrast, monochrome, light — drawn from one palette the dashboard reads too.",
      },
    ],
  },
  {
    id: "history",
    title: "History, projects and the command line",
    lede: "A run is a record you can find again, and every command answers the same flags.",
    accent: "#f472b6",
    items: [
      {
        title: "Durable run and session history",
        body:
          "Every run is recorded with its model, provider, tokens, cost, status and task, keyed by the same run id its trace spans carry. Runs from the command line, a script and the server share one history and survive a restart, with search, status, model and date filters.",
        code: {
          source: `effgen runs list --status failed --since 7d
effgen sessions show <id>`,
          language: "bash",
        },
      },
      {
        title: "Project scaffolding",
        body:
          "effgen quickstart --init [DIR] writes effgen.yaml, a .env.example carrying one named variable per registered provider with no value invented, a runnable example.py and a .gitignore; puts a $1.00/day spend cap in force when none is configured; and prints the next three commands. effgen config init writes a document a run actually reads.",
      },
      {
        title: "Flags and output that behave the same everywhere",
        body:
          "--json on every command that had no machine output, and --json stdout is now a single valid document on a pipe and on a terminal, with no spinner, table or warning mixed in. -o picks its format from the extension. Thirteen short flags now mean the same thing across commands, and a bare group command prints its own help and exits 0 instead of reporting an unknown subcommand.",
      },
      {
        title: "Your own prompt templates load beside the shipped ones",
        body:
          "EFFGEN_PROMPTS_DIR names one or more directories; each *.py in them is imported and its templates registered under their own names, so a team's library sits next to the built-in one without a fork. prompts run now fails closed on an empty or truncated result rather than printing nothing and exiting 0, and reports the tokens, cost and latency of the call it made.",
      },
      {
        title: "Load testing through the server, not around it",
        body:
          "effgen loadtest --url drives a running effgen serve over HTTP, through auth, rate limiting and the middleware stack, instead of only driving an adapter directly.",
      },
    ],
  },
  {
    id: "truthful",
    title: "Results that report what actually happened",
    lede: "The largest and least visible part of the release: a pass over everything that used to report the wrong thing confidently.",
    accent: "#ff9500",
    items: [
      {
        title: "A turn that did nothing no longer reports success",
        body:
          "A coding turn whose every action failed, and a retrieval loop that produced no answer, are reported as partial outcomes with the recovered text under metadata[\"partial_output\"] and a typed reason for what stopped the run. A run stopped at the iteration cap reports the stop, not the last passage it retrieved, and carries that through every surface that shows it.",
      },
      {
        title: "A tool call the model wrote out instead of making is a failed turn",
        body:
          "Not an answer — including the shapes that used to slip through: a stray angle bracket, a missing separator, a query string with HTML entities, call syntax whose arguments were dropped, and a tag named after the tool itself.",
      },
      {
        title: "An unpriced model reports no cost, not a fabricated one",
        body:
          "A provider's placeholder rate made every id the bundled catalog had not seen read as priced, so a fine-tuned ft: id was billed at a made-up rate and the invented number reported as a published price. call_cost returns None for an unpriced model and 0.0 only for a genuine free tier, and every surface says \"no price\" instead of $0.",
      },
      {
        title: "Streamed runs report their cost and tokens",
        body:
          "On every provider, including Replicate and HF Inference, which recorded neither. model.total_tokens is correct on every adapter — six never assigned it at all — and a response that reports an all-zero usage block for a call it billed is estimated and flagged rather than recorded as free. Team and workflow totals now include the manager's own calls.",
      },
      {
        title: "A citation is a source the answer actually used",
        body:
          ".sources still carries every URL a search returned; .citations now carries the ones the answer references, and a PDF citation carries its page number.",
      },
      {
        title: "A model that could not run at all is reported as failed, not as scoring zero",
        body:
          "An evaluation where the key was missing or the provider refused every call used to print 0% beside the models that did run, which reads as a bad model rather than as a model that never answered.",
      },
    ],
  },
  {
    id: "toolcalls",
    title: "Tool calling across providers",
    lede: "Chat templates disagree about how a call is spelled. effGen now reads the shapes, not the model families.",
    accent: "#00c896",
    items: [
      {
        title: "A tool call written as XML tags is understood",
        body:
          "Many chat templates render JSON; others render nested tags. effGen read only the JSON spellings, so on a model whose template emits tags the turn parsed to nothing: no tool was called, and the run ended at the iteration cap. The reader is keyed on the shape rather than on a model family — five call tags, four argument tags, both <tag=NAME> and <tag name=\"NAME\"> — so any family whose template writes that shape can use tools.",
      },
      {
        title: "One call shape across every adapter",
        body:
          "generate_with_tools() takes config third on all ten adapters; it was messages on Groq, Together and Fireworks, so a positional call misrouted its argument and failed as a retryable error. Both spellings still work, told apart by type, so there is no migration.",
      },
      {
        title: "A rate limit no longer multiplies",
        body:
          "Three layers each retried a throttled call and multiplied rather than shared a budget: one client request became twelve upstream requests and held the caller 20.5 seconds at a stated 2-second delay. One layer now owns provider retry. The same measurement reads four requests and 6.7 seconds.",
      },
      {
        title: "A plain run() no longer fans out into sub-agents on its own",
        body:
          "AgentConfig.mode defaults to SINGLE. A task over roughly a hundred words used to become six billed calls, and a decomposed run could report a number the source text never contained. --mode auto opts back in, and a genuinely multi-part task still decomposes.",
      },
    ],
  },
  {
    id: "errors",
    title: "Errors that name the fix",
    lede: "Every message a reader sees is bounded, redacted and ends with what to do next.",
    accent: "#ff6b6b",
    items: [
      {
        title: "A connection failure names the endpoint the call was sent to",
        body:
          "Instead of pointing at the provider's status page, which is advice about the wrong machine when the server is yours. A URL with no scheme is refused, naming the environment variable it came from, and a blank endpoint variable no longer redirects every OpenAI call.",
      },
      {
        title: "A rate limit delivered as HTTP 413 is classified as one",
        body:
          "One provider reports a spent tokens-per-minute allowance that way. It used to be unknown, so there was no backoff and a throttle was reported as a permanent failure. A genuinely oversized body is still an invalid request.",
      },
      {
        title: "The submitted credential never reaches the caller",
        body:
          "effGen redacted its own message, but raise ... from exc kept the SDK exception, and a 401 body quotes the key. The whole __cause__/__context__ chain is now scrubbed, including per-SDK attributes and parsed JSON bodies, and a rendered traceback carries nothing. Quoted upstream text is bounded to 240 characters — one real provider body reached 42 kB.",
      },
      {
        title: "A malformed input names its file",
        body:
          "A workflow YAML that is not a workflow, a config file that is not a mapping, a drifted session or checkpoint, a damaged catalog snapshot and a batch row with no query text are each named with the file and the position, instead of raising from somewhere unrelated.",
      },
      {
        title: "A call can be bounded",
        body:
          "One adapter took no timeout at all and another's deadline governed polling only, so a peer that never answers held the call for 90 seconds. Both take timeout and max_retries now, and with_timeout() re-arms rather than firing once into an SDK retry loop that swallowed it — a 2-second bound now stops a call at 2.25 seconds instead of 22 to 50.",
      },
    ],
  },
  {
    id: "server",
    title: "The server, security and sandboxing",
    lede: "The OpenAI-compatible server, the guardrail presets and the execution sandbox.",
    accent: "#22d3ee",
    items: [
      {
        title: "One error envelope, and a loop that stays responsive",
        body:
          "The server answers every failure with the same envelope — unknown URLs, wrong methods, missing static assets, unhandled route errors, RBAC denials, the shutdown drain, websockets and the edge adapters. A non-streaming completion used to block the event loop, so /health timed out for the length of the call; it now runs off the loop, measured at 6 ms worst case during a 14.5 second completion.",
      },
      {
        title: "Content-free requests are refused before they are billed",
        body:
          "An empty or whitespace prompt, absent content and a non-positive max_tokens each return a 4xx before any upstream call. An absent provider key is a 503 on every provider, an upstream 429 passes its delay on as Retry-After, and a mid-stream failure emits a terminal error event rather than truncating the stream.",
      },
      {
        title: "Rate limiting is not defeated by a header",
        body:
          "X-Forwarded-For is trusted only when you enable it, and effgen serve no longer lets the ASGI server rewrite the client address behind that setting. Body size limits now cover /v1/embeddings, which used to accept an unbounded body.",
      },
      {
        title: "The sandbox masks credential stores and isolates the process table",
        body:
          "Executed code sees one process rather than the host's, and the credential directories beside it are masked. Both are reported on the result as credential_reads_masked and process_table_isolated. This is a deny-list over a known set of paths, not read confinement, and the documentation says so.",
      },
      {
        title: "The standard guardrail preset screens tool output for injection",
        body:
          "Not just input, so an instruction planted in a tool's return value no longer reaches the model under the default preset. standard also redacts personal data rather than blocking the message, so a customer quoting their own email address is answered instead of refused. strict still blocks.",
      },
    ],
  },
  {
    id: "local",
    title: "Local models, GPUs and long runs",
    lede: "What happens when the weights are on your own machine, and the run goes on for a while.",
    accent: "#a3e635",
    items: [
      {
        title: "A model that does not fit the GPU says so",
        body:
          "VRAM sizing reads free memory rather than total, the engine reconciles .device with where the parameters actually are, the run's metadata carries it, and require_gpu=True fails fast rather than falling back to CPU silently. Device memory comes back when a local model unloads — a 1.5B model used to keep 2.9 GB reserved.",
      },
      {
        title: "Automatic sharding across several GPUs no longer produces invalid output",
        body:
          "On a multi-GPU node, device_map=\"auto\" could place a model so that sampling read invalid logits and the run died in a CUDA assert. The engine now probes the logits after loading and pins the model to one device before sampling.",
      },
      {
        title: "Per-call sampling keywords are honoured on the local engines",
        body:
          "Including seed and stop_sequences, which one engine read off the config before it looked at the call. A local reasoning model is recognised from its own chat template, so it gets the larger budget instead of spending the base one on a hidden chain and returning nothing.",
      },
      {
        title: "A long conversation stops growing its own prompt",
        body:
          "Session summaries were unbounded and replayed into every prompt, so past a threshold each turn added another summary until every call was refused for exceeding the context window. Summaries are now folded within a token budget measured with the model's own tokenizer, and a 25,000-turn session stays flat.",
      },
      {
        title: "Long runs hold up under concurrency",
        body:
          "Tool discovery, registry replacement and rate-limit accounting are serialised; costs and tokens are folded under a lock, so a shared adapter's totals match the calls; two writers of one session no longer publish a blend. Batch rows no longer contaminate each other: eight concurrent rows report eight distinct costs, and the job total reconciles with the sum.",
      },
    ],
  },
  {
    id: "rag",
    title: "Documents, retrieval and batch input",
    lede: "What goes into a knowledge base, and what comes back out of it.",
    accent: "#818cf8",
    items: [
      {
        title: "The rag preset refuses to run without a knowledge base",
        body:
          "Instead of succeeding with zero documents. Retrieval also keeps distinct topics: the preset configures a wider top_k with MMR re-ranking, so a two-topic question returns both topics.",
      },
      {
        title: "Ingestion says what it skipped and why",
        body:
          "A corrupt file, an empty file, a file whose content duplicates an earlier one, an image and an unsupported extension each have their own reason, and DocumentIngester.last_summary reports what was indexed. PDFs carry page numbers.",
      },
      {
        title: "A weak model no longer answers with the passages",
        body:
          "The prompt now ends with an answer-shaping instruction after a retrieval tool, and both loop fallbacks give the model one tool-free turn to answer from what it has. Measured on the worst case: verbatim passage dumps went from 8 of 9 runs to 0 of 9, with citations on every run.",
      },
      {
        title: "Structured output extraction stopped corrupting valid JSON",
        body:
          "Repairs used to run inside string literals, so a value containing \", note:\" was rewritten into something unparseable. Measured over 8,000 adversarial examples: 36 mis-parses before, 0 after, with 1,427 inputs newly recovered.",
      },
      {
        title: "Batch input is read carefully",
        body:
          "A row keyed on prompt, input, question or text is recognised, a scalar or dict row does not become a prompt, CSV rows report the right line, a non-UTF-8 file names itself, and --strict fails the job on any unusable row. run --file reads source code and plain text, not only documents, and refuses binaries.",
      },
    ],
  },
  {
    id: "terminal",
    title: "The terminal, the tools and the install",
    lede: "The parts you meet before you write any code.",
    accent: "#fb923c",
    items: [
      {
        title: "Every command works on a terminal that cannot encode what effGen prints",
        body:
          "Twenty-two commands used to exit non-zero purely because of the console encoding. Text is folded to ASCII where it becomes bytes, and --json escapes rather than transliterates, so a French or Chinese answer survives a hard-ASCII console byte for byte.",
      },
      {
        title: "Piped output is clean, and a closed pipe is quiet",
        body:
          "No spinner, no placeholder, no chrome on stdout, one answer per input line under -q, and zero colour codes under NO_COLOR. effgen ... | head used to end in a BrokenPipeError traceback; the command now exits 141, the convention the shell expects.",
      },
      {
        title: "The whole command surface works without rich and without torch",
        body:
          "Twelve commands used to exit with a missing-module error, and a direct engine import reported the import system rather than naming PyTorch and how to install it.",
      },
      {
        title: "A tool that cannot do its job says so",
        body:
          "Translation with no language pair available, a knowledge-base search the API refused, a news fetch where every source was unreachable, and a web search whose unset filters were sent to the backend all reported success or the wrong error. Each now fails with the reason. A blocked request is not read as an empty result.",
      },
      {
        title: "Every documentation snippet runs",
        body:
          "The 57 tool gallery snippets were rewritten to the awaited keyword API, all 30 network snippets now check ToolResult.success before reading output, and the command-line pages were re-run command by command. ./install.sh no longer fails when run without a terminal, and the docker compose file binds to loopback.",
      },
    ],
  },
];

/** The 19 names 1.0.0 adds to the top-level package. */
export const newPublicNames = [
  "OpenAICompatibleAdapter",
  "BackendUnreachableError",
  "AgentMiddleware",
  "MiddlewareChain",
  "LoggingMiddleware",
  "ToolApprovalMiddleware",
  "ToolCall",
  "ToolCallList",
  "CompactionStrategy",
  "SummarizeOldest",
  "DropOldest",
  "KeepFirstAndLast",
  "KeepToolResults",
  "WorkflowCheckpoint",
  "CheckpointStore",
  "FileCheckpointStore",
  "InMemoryCheckpointStore",
  "SystemPromptLeakGuardrail",
  "load_env",
];

/* ── Earlier releases ────────────────────────────────────────────────────── */

export interface EarlierRelease {
  version: string;
  date: string;
  title: string;
  summary: string;
}

/** Every release before 1.0.0, with the date and headline CHANGELOG.md gives it. */
export const earlierReleases: EarlierRelease[] = [
  {
    version: "0.3.2",
    date: "5 July 2026",
    title: "Usability, robustness and polish",
    summary:
      "A point release driven by living with the framework: results integrity re-certified, grounding traceable, a consistent server contract, batch that survives real data, and observability an on-call rotation can use. No breaking API changes.",
  },
  {
    version: "0.3.1",
    date: "29 June 2026",
    title: "Real-world usability and polish",
    summary:
      "Grounded results carry the sources they were built from, reasoning models finish token-heavy work, a custom persona is honoured on every path, a multi-agent team reports the failure instead of a partial result, and a knowledge domain becomes a runnable agent in one call. No breaking API changes.",
  },
  {
    version: "0.3.0",
    date: "19 June 2026",
    title: "Stabilization and hardening",
    summary:
      "No new providers, tools or subsystems. Failures became loud and typed instead of silently succeeding, the model catalog updates itself, local GPUs work out of the box, the server fails closed, the built-in tools are sandboxed, and import effgen is effectively instant.",
  },
  {
    version: "0.2.10",
    date: "27 May 2026",
    title: "Security, edge and developer experience",
    summary:
      "Secret scanning, dependency auditing, an SBOM pipeline, supply-chain integrity verification, a sandboxed code executor, OIDC auth with RBAC and per-request audit logging, four deploy targets, and three developer-experience surfaces.",
  },
  {
    version: "0.2.9",
    date: "23 May 2026",
    title: "Observability and reliability",
    summary:
      "Structured JSON logs with secret redaction, OpenTelemetry tracing with configurable samplers, Prometheus histograms, SLO tracking, circuit breakers, bulkheads, a deterministic chaos harness and load testing.",
  },
  {
    version: "0.2.8",
    date: "21 May 2026",
    title: "Multimodal input",
    summary:
      "Image, audio and video become input types in their own right across six cloud providers plus local MLX-VLM, with a unified content schema, per-provider preprocessing and capability gating — no silent downcast when a model lacks vision or audio.",
  },
  {
    version: "0.2.7",
    date: "20 May 2026",
    title: "The prompt library",
    summary:
      "A curated, domain-organised catalog of reusable prompt templates, paired with a golden evaluation harness, a command-line surface and an interactive playground.",
  },
  {
    version: "0.2.6",
    date: "19 May 2026",
    title: "Documents, media and communication tools",
    summary:
      "Fourteen new built-in tools across OCR, audio transcription, image analysis, document parsing, geo and weather, and email and webhook communication — plus the media and notify presets.",
  },
  {
    version: "0.2.5",
    date: "18 May 2026",
    title: "Thirteen free, no-auth tools",
    summary:
      "Academic search, news and RSS, YouTube, social media, translation, language detection and QR codes, all wired into the research and general presets.",
  },
  {
    version: "0.2.4",
    date: "14 May 2026",
    title: "Model routing and cost tracking",
    summary:
      "A ModelRouter with three composable routing policies, transparent provider failover with retry logic, cross-process rate-limit coordination, and a persisted cost ledger behind effgen cost.",
  },
  {
    version: "0.2.3",
    date: "4 May 2026",
    title: "Nine inference backends",
    summary:
      "Groq, Together, Fireworks, Replicate and HF Inference join the provider set, each with streaming, native tool calling, cost tracking and rate-limit coordination, behind one provider interface.",
  },
  {
    version: "0.2.2",
    date: "28 April 2026",
    title: "The Gemini adapter, expanded",
    summary:
      "The current model families, thinking budgets, search grounding, the Files API and three provider-native tools.",
  },
  {
    version: "0.2.1",
    date: "25 April 2026",
    title: "Cerebras, and a modern OpenAI adapter",
    summary:
      "The Cerebras backend with streaming, native tool calling and cost tracking, alongside reasoning models, reasoning effort, prompt-cache reporting and structured outputs on OpenAI.",
  },
  {
    version: "0.2.0",
    date: "9 April 2026",
    title: "From toolkit to platform",
    summary:
      "Native tool calling, guardrails, multi-agent orchestration, RAG pipelines, evaluation and an API server.",
  },
  {
    version: "0.1.0",
    date: "1 March 2026",
    title: "Prompts built for small models",
    summary:
      "Dynamic system prompts carrying exact tool-usage examples, per-family prompt formatting, and tool fallback chains for when a tool fails.",
  },
  {
    version: "0.0.1",
    date: "31 January 2026",
    title: "The first release",
    summary:
      "The agent loop, task management, agent state and the ReAct pattern, built for models between one and seven billion parameters.",
  },
];
