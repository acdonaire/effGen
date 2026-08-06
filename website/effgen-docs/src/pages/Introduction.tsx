import React from 'react';
import { Link } from 'react-router-dom';
import { Book } from 'lucide-react';
import DocPage, { InfoBox, QuickLinks, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';

export default function Introduction() {
  const architectureDiagram = `
flowchart TB
    subgraph Agent["🤖 Agent Layer"]
        A[Agent Executor]
        AC[Agent Config]
        AR[Agent Response]
    end

    subgraph Tools["🔧 Tool System"]
        TR[Tool Registry]
        BT[Built-in Tools]
        CT[Custom Tools]
    end

    subgraph Memory["💾 Memory System"]
        STM[Short-Term Memory]
        LTM[Long-Term Memory]
    end

    subgraph Model["🧠 Model Layer"]
        ML[Model Loader]
        VL[vLLM Engine]
        TF[Transformers]
        API[API Providers]
    end

    subgraph Core["⚙️ Core Engine"]
        RL[ReAct Loop]
        TD[Task Decomposition]
        MAO[Multi-Agent Orchestrator]
    end

    A --> AC
    A --> RL
    RL --> Tools
    RL --> Memory
    RL --> Model

    TR --> BT
    TR --> CT

    ML --> VL
    ML --> TF
    ML --> API

    RL --> TD
    TD --> MAO
`;

  const quickStartCode = `from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Load a model (local engines or 9 cloud providers)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="transformers",
    quantization="4bit"
)

# Create an agent with tools
agent = Agent(config=AgentConfig(
    name="my_agent",
    model=model,
    tools=[Calculator()],
    system_prompt="You are a helpful assistant.",
    temperature=0.1,
    max_iterations=10
))

# Run a task
result = agent.run("What is 25 * 17 + sqrt(144)?")
print(result.output)  # "The result is 437"
print(f"Success: {result.success}")
print(f"Tools used: {result.tool_calls}")`;

  return (
    <DocPage
      title="Introduction"
      subtitle="Build powerful AI agents with Small Language Models. effGen is a production-ready framework designed specifically for SLMs."
      icon={<Book size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Introduction' },
      ]}
    >
      <InfoBox type="info" title="What is effGen?">
        <p>
          effGen is a comprehensive framework for building AI agents powered by Small Language Models (SLMs).
          Unlike other frameworks optimized for large models, effGen provides specialized optimizations
          for smaller, more efficient models while maintaining production-grade capabilities. v0.2.3+
          supports 14 inference backends (5 local engines + 9 cloud providers); v0.2.7 adds the
          Prompt Library (31 curated templates with a golden + live eval harness); v0.2.8 adds
          multimodal image / audio / video input; v0.2.9 adds a full observability &amp; reliability
          stack; and v0.2.10 adds a sandboxed CodeExecutor, OIDC auth + RBAC, production deploys
          (Docker / Helm / Lambda / Cloudflare), and developer-experience surfaces (VSCode, Jupyter,
          dashboard). <strong>v0.3.0</strong> is a stabilization &amp; hardening release: failures
          are now loud and typed (no more silent <code>success=True</code>), the model catalog
          updates itself, local GPUs work out of the box, the server fails closed, the built-in
          tools are sandboxed, and <code>import effgen</code> is effectively instant. Most recently,{' '}
          <strong>v0.3.1</strong> is a real-world usability &amp; polish release: grounded{' '}
          <code>response.sources</code> / <code>.citations</code>, reasoning models that finish
          token-heavy work, custom personas honored on every path, one-call domain agents, honest
          multi-agent teams and an honest OpenAI-compatible server, and <code>effgen run --json</code>.
          No breaking API changes throughout.
        </p>
      </InfoBox>

      <h2>Why effGen?</h2>
      <p>
        Traditional agent frameworks like LangChain and AutoGen are optimized for large language models with
        billions of parameters. effGen bridges the gap, enabling smaller models (1-7B parameters) to
        perform complex agentic tasks through:
      </p>

      <FeatureList
        features={[
          {
            icon: '⚡',
            title: 'vLLM-First Architecture',
            description: '5-10x faster inference with automatic multi-GPU tensor parallelism and PagedAttention',
          },
          {
            icon: '🧠',
            title: 'SLM-Optimized Prompts',
            description: 'Concise, structured prompts designed specifically for smaller context windows',
          },
          {
            icon: '🔀',
            title: 'Intelligent Task Decomposition',
            description: 'Automatically break down complex tasks into manageable subtasks for better accuracy',
          },
          {
            icon: '🔧',
            title: 'Universal Tool Integration',
            description: '66 built-in tools, native + ReAct + hybrid function calling, MCP/A2A/ACP',
          },
          {
            icon: '📚',
            title: 'Prompt Library',
            description: '35 curated, domain-organized templates across 8 domains (research / coding / data-SQL / legal / medical / creative / business / education) with golden + live evals, CLI, and playground',
          },
          {
            icon: '🖼️',
            title: 'Multimodal Input (v0.2.8)',
            description: 'Image / audio / video as first-class types across Gemini, OpenAI, Groq, Anthropic, Together, HF + local MLX-VLM, with a unified ContentPart schema and capability gating',
          },
          {
            icon: '📈',
            title: 'Observability & Reliability (v0.2.9)',
            description: 'Structured logging + secret redaction, Prometheus metrics, SLO burn-rate tracking, OTel tracing, timeouts / retries / circuit breakers / bulkheads, chaos + fuzz, load testing, alerting',
          },
          {
            icon: '🔒',
            title: 'Security, Edge & DX (v0.2.10)',
            description: 'Sandboxed CodeExecutor, OIDC auth + RBAC + audit log, Docker / Helm / Lambda / Cloudflare deploys, VSCode extension, Jupyter magics, and a live local dashboard',
          },
          {
            icon: '🛡️',
            title: 'Stabilization & Hardening (v0.3.0)',
            description: 'Fail-closed typed errors (no silent success=True), a self-updating drift-aware model catalog (effgen models refresh), real GPU support + greedy temperature=0, a fail-closed server, sandboxed built-in tools (shared SSRF guard, out-of-process PythonREPL timeout, path confinement), and a near-instant import effgen (~7.5s → ~20ms)',
          },
          {
            icon: '🤖',
            title: 'Multi-Agent Orchestration',
            description: 'DAG workflow engine, MessageBus pub/sub, SharedState, AgentPool + lifecycle',
          },
          {
            icon: '🔒',
            title: 'Production Infrastructure',
            description: 'Docker sandbox, OpenTelemetry tracing, Prometheus metrics, guardrails, API server v2, checkpointing',
          },
        ]}
      />

      <h2>Architecture Overview</h2>
      <p>
        effGen follows a modular architecture built around the ReAct (Reasoning + Acting) paradigm.
        The framework consists of several interconnected layers:
      </p>

      <MermaidDiagram chart={architectureDiagram} title="effGen Architecture" />

      <h2>Quick Example</h2>
      <p>
        Here's a complete example showing how to create an agent with tools:
      </p>

      <CodeBlock
        code={quickStartCode}
        language="python"
        filename="quick_example.py"
      />

      <h2>Core Components</h2>

      <h3>The ReAct Loop</h3>
      <p>
        effGen implements the ReAct (Reasoning + Acting) paradigm, where the agent iteratively:
      </p>
      <ol>
        <li><strong>Thinks</strong> - Reasons about the current state and what action to take</li>
        <li><strong>Acts</strong> - Executes a tool or provides a final answer</li>
        <li><strong>Observes</strong> - Receives and processes tool results</li>
        <li><strong>Repeats</strong> - Continues until the task is complete or max iterations reached</li>
      </ol>

      <h3>Tool System</h3>
      <p>
        Tools extend agent capabilities by connecting to external systems, APIs, and functions.
        effGen ships <strong>58+ built-in tools</strong> across many categories — all lazy-loaded, all overridable. v0.2.6 added 14 new tools (OCR, audio transcription, image analysis, document parsing — PDF/DOCX/Excel, geo/weather, email SMTP/IMAP, and Slack/Discord webhooks); v0.2.5 added 13 free / no-auth tools spanning academic research (PubMed, ArXiv, Semantic Scholar), news &amp; RSS, YouTube, social (Reddit, HN), translation, language detection, and QR codes. Core:
      </p>
      <ul>
        <li><strong>Calculator</strong> - Mathematical operations, unit conversions, statistics</li>
        <li><strong>PythonREPL</strong> - Interactive Python with state persistence</li>
        <li><strong>WebSearch</strong> - Search the web for information</li>
        <li><strong>CodeExecutor</strong> - Execute Python code in a secure sandbox</li>
        <li><strong>FileOperations</strong> - Read, write, and manipulate files</li>
        <li><strong>BashTool</strong> - Execute shell commands with security controls</li>
        <li><strong>Retrieval</strong> - RAG-based semantic search with BM25 hybrid</li>
        <li><strong>AgenticSearch</strong> - ripgrep-based search for precise queries</li>
        <li><strong>JSONTool</strong> - Parse, query, transform, and validate JSON</li>
        <li><strong>DateTimeTool</strong> - Current time, timezone conversion, date arithmetic</li>
        <li><strong>TextProcessingTool</strong> - Word count, regex operations, text comparison</li>
        <li><strong>URLFetchTool</strong> - Fetch and extract text from web pages</li>
        <li><strong>WeatherTool</strong> - Weather data via Open-Meteo API</li>
        <li><strong>WikipediaTool</strong> - Search and retrieve Wikipedia articles</li>
      </ul>
      <p>
        Plus 17 domain tools added in v0.2.0 — <strong>finance</strong> (StockPriceTool, CurrencyConverterTool, CryptoTool),
        <strong>data science</strong> (DataFrameTool, PlotTool, StatsTool),
        <strong>DevOps</strong> (GitTool, DockerTool, SystemInfoTool, HTTPTool),
        <strong>knowledge</strong> (ArxivTool, StackOverflowTool, GitHubTool, WolframAlphaTool),
        and <strong>communication</strong> (EmailDraftTool, SlackDraftTool, NotificationTool) — all draft-only where relevant.
        OpenAI/Gemini provider-native tools and Anthropic experimental adapter specs are documented separately in
        <Link to="/native-provider-tools"> Native Provider Tools</Link>. See the full local catalog in
        <Link to="/tools"> Tools</Link>.
      </p>

      <h3>Providers and Registry</h3>
      <p>
        v0.2.1 added Cerebras, v0.2.2 modernized Gemini and Anthropic, and v0.2.3 added Groq,
        Together AI, Fireworks, Replicate, and HuggingFace Inference. Use
        <code> provider:model_id</code> prefixes, <code>ProviderRegistry</code>, and
        <code> effgen doctor</code> in v0.2.3+ to keep cloud model routing explicit. See
        <Link to="/providers"> Providers</Link>.
      </p>

      <h3>Safety & Guardrails</h3>
      <p>
        The <code>effgen.guardrails</code> module provides offline, ML-free validation that runs before,
        during, and after agent execution: toxicity detection, PII redaction (SSN, email, phone, credit-card
        with Luhn, IP), prompt-injection detection (low/medium/high sensitivity), topic allow/deny, length
        limits, and tool input/output/permission controls. Wire them in via <code>AgentConfig.guardrails</code>,
        or pick a preset: <code>strict</code>, <code>standard</code>, <code>minimal</code>, <code>none</code>. See <Link to="/guardrails">Guardrails</Link>.
      </p>

      <h3>RAG Pipeline</h3>
      <p>
        <code>effgen.rag</code> is a full retrieval pipeline: <code>DocumentIngester</code> loads txt/md/json/jsonl/csv/html
        (plus optional pdf/docx/epub); semantic / code / table / hierarchical chunkers; <code>HybridSearchEngine</code>
        fusing dense + BM25 + keyword + metadata via Reciprocal Rank Fusion; CrossEncoder / LLM / rule-based
        rerankers; and <code>ContextBuilder</code> with token-budgeting and inline [N] citations. Citations flow into
        <code>AgentResponse.citations</code> and <code>.sources</code>. See <Link to="/rag">RAG</Link>.
      </p>

      <h3>Workflows, Evaluation &amp; API Server</h3>
      <p>
        <code>WorkflowDAG</code> validates, topo-sorts, and auto-parallelises multi-agent pipelines (<Link to="/workflows">Workflows</Link>).
        <code>effgen.eval</code> ships five test suites (Math, ToolUse, Reasoning, Safety, Conversation) with
        LLM-judge / semantic / regex scoring and <code>RegressionTracker</code> baselines (<Link to="/evaluation">Evaluation</Link>).
        The <Link to="/api-server">API Server v2</Link> serves an OpenAI-compatible gateway with priority queue,
        agent pool, multi-tenancy, and streaming.
      </p>

      <h3>Memory Systems</h3>
      <p>
        Dual memory architecture for maintaining context:
      </p>
      <ul>
        <li><strong>Short-Term Memory</strong> - Conversation history with automatic summarization</li>
        <li><strong>Long-Term Memory</strong> - Persistent storage with JSON/SQLite backends and semantic search</li>
      </ul>

      <h2>Get Started</h2>
      <QuickLinks
        links={[
          {
            icon: '⚙️',
            title: 'Installation',
            description: 'Set up effGen in 60 seconds',
            path: '/installation',
          },
          {
            icon: '🚀',
            title: 'Quick Start',
            description: 'Build your first agent',
            path: '/quickstart',
          },
          {
            icon: '💡',
            title: 'Examples',
            description: 'Real-world use cases',
            path: '/examples',
          },
          {
            icon: '📖',
            title: 'Guides',
            description: 'In-depth tutorials',
            path: '/guides',
          },
        ]}
      />

      <InfoBox type="success" title="Ready to start?">
        <p>
          Head to the <Link to="/installation">Installation guide</Link> to set up effGen,
          or jump directly to the <Link to="/quickstart">Quick Start</Link> tutorial to build your first agent.
        </p>
      </InfoBox>
    </DocPage>
  );
}
