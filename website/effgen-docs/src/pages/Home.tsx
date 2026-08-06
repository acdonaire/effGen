import React from 'react';
import { Link } from 'react-router-dom';
import {
  Zap, Bot, Wrench, Database, GitBranch,
  Shield, ArrowRight, BookOpen, Rocket, FileCode, FileText, Layers,
  Lock, Activity, Image
} from 'lucide-react';
import CodeBlock from '../components/CodeBlock';
import { usePyPIVersion } from '../hooks/usePyPIVersion';
import './Home.css';

export default function Home() {
  const { version: pypiVersion } = usePyPIVersion();
  const quickStartCode = `from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Load model with 4-bit quantization
model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

# Create agent configuration
config = AgentConfig(
    name="calculator_agent",
    model=model,
    tools=[Calculator()],
    system_prompt="You are a helpful assistant."
)

# Create agent
agent = Agent(config=config)

# Run a task
result = agent.run("What is 25 * 17 + sqrt(144)?")
print(result.output)  # "437"`;

  return (
    <div className="home">
      {/* Hero Section */}
      <section className="hero hero-docs">
        <div className="hero-bg">
          <div className="gradient-orb orb-1"></div>
          <div className="gradient-orb orb-2"></div>
          <div className="gradient-orb orb-3"></div>
          <div className="grid-pattern"></div>
          <div className="floating-shapes">
            <div className="shape shape-1"></div>
            <div className="shape shape-2"></div>
            <div className="shape shape-3"></div>
            <div className="shape shape-4"></div>
            <div className="shape shape-5"></div>
            <div className="shape shape-6"></div>
            <div className="shape shape-7"></div>
            <div className="shape shape-8"></div>
          </div>
          <div className="hero-rings">
            <div className="ring ring-1"></div>
            <div className="ring ring-2"></div>
            <div className="ring ring-3"></div>
            <div className="ring ring-4"></div>
          </div>
          <div className="hero-scan-line"></div>
        </div>

        <div className="hero-content">
          <div className="hero-badge">
            <BookOpen size={16} />
            <span>Documentation v{pypiVersion}</span>
          </div>

          <h1 className="hero-title">
            <span className="hero-title-line">Build AI Agents with <span className="gradient-text">effGen</span></span>
            <span className="hero-title-line">Small Models, Big Results</span>
          </h1>

          <p className="hero-subtitle">
            Everything you need to build, deploy, and scale AI agents with Small Language Models.
            Explore comprehensive guides, API references, and production-ready examples.
          </p>

          {/* Terminal Preview */}
          <div className="hero-terminal">
            <div className="terminal-header">
              <div className="terminal-dot red"></div>
              <div className="terminal-dot yellow"></div>
              <div className="terminal-dot green"></div>
              <span className="terminal-title">quickstart.py</span>
            </div>
            <div className="terminal-body">
              <div className="terminal-line">
                <span className="terminal-prompt">$</span>
                <span className="terminal-command">pip install effgen</span>
              </div>
              <div className="terminal-line">
                <span className="terminal-prompt">$</span>
                <span className="terminal-command">python -c "from effgen import Agent"</span>
              </div>
              <div className="terminal-line">
                <span className="terminal-output">effGen v{pypiVersion} loaded successfully</span>
              </div>
              <div className="terminal-line">
                <span className="terminal-prompt">$</span>
                <span className="terminal-command">python quickstart.py</span>
              </div>
              <div className="terminal-line">
                <span className="terminal-output">Agent ready. Result: 437</span>
                <span className="terminal-cursor"></span>
              </div>
            </div>
          </div>

          <div className="doc-nav-cards">
            <Link to="/quickstart" className="doc-nav-card">
              <div className="doc-nav-icon">
                <Rocket size={24} />
              </div>
              <div className="doc-nav-content">
                <h3>Quick Start</h3>
                <p>Get up and running in 5 minutes</p>
              </div>
              <ArrowRight size={18} className="doc-nav-arrow" />
            </Link>

            <Link to="/introduction" className="doc-nav-card">
              <div className="doc-nav-icon">
                <BookOpen size={24} />
              </div>
              <div className="doc-nav-content">
                <h3>Introduction</h3>
                <p>Learn the fundamentals</p>
              </div>
              <ArrowRight size={18} className="doc-nav-arrow" />
            </Link>

            <Link to="/api-reference" className="doc-nav-card">
              <div className="doc-nav-icon">
                <FileCode size={24} />
              </div>
              <div className="doc-nav-content">
                <h3>API Reference</h3>
                <p>Complete API documentation</p>
              </div>
              <ArrowRight size={18} className="doc-nav-arrow" />
            </Link>

            <Link to="/examples" className="doc-nav-card">
              <div className="doc-nav-icon">
                <Layers size={24} />
              </div>
              <div className="doc-nav-content">
                <h3>Examples</h3>
                <p>Real-world use cases</p>
              </div>
              <ArrowRight size={18} className="doc-nav-arrow" />
            </Link>
          </div>
        </div>
      </section>

      {/* What's New */}
      <section className="quick-start">
        <div className="section-header">
          <h2>What's New in v{pypiVersion}</h2>
          <p>The v0.2.1-v0.3.1 release train adds Cerebras, modern OpenAI/Gemini/Anthropic adapters, five new cloud providers, ProviderRegistry, a policy-based ModelRouter with transparent failover, persistent cost tracking, a <strong>66-tool</strong> built-in library, the <strong>Prompt Library</strong> (35 templates across 8 domains) in v0.2.7, <strong>multimodal</strong> image / audio / video input in v0.2.8, a full <strong>observability &amp; reliability</strong> stack in v0.2.9, <strong>security, edge &amp; developer-experience</strong> features in v0.2.10, the <strong>v0.3.0 stabilization &amp; hardening</strong> release, and — most recently — the <strong>v0.3.1 real-world usability &amp; polish</strong> release: grounded <code>response.sources</code> / <code>.citations</code>, reasoning models that finish token-heavy work, personas honored on every path, one-call domain agents, honest multi-agent teams and an honest OpenAI-compatible server, and <code>effgen run --json</code>.</p>
        </div>

        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon"><Shield size={28} /></div>
            <h3>v0.3.0: Stabilization &amp; Hardening</h3>
            <p>No new features — instead everything already in effGen becomes robust, fast, and secure. <Link to="/agents">Fail-closed typed errors</Link> (no silent <code>success=True</code>), a self-updating drift-aware <Link to="/models">model catalog</Link> (<code>effgen models refresh</code>), real GPU support (<code>temperature=0</code> greedy decoding, NVML allocator), a <Link to="/security">fail-closed server</Link> (forged JWTs rejected, 502 not 401), sandboxed built-in tools (shared SSRF guard, out-of-process PythonREPL timeout, path confinement), <code>import effgen</code> ~7.5s → ~20ms, and a quiet <Link to="/dx">scriptable CLI</Link>. No breaking changes.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Lock size={28} /></div>
            <h3>v0.2.10: Security, Edge &amp; DX</h3>
            <p>A sandboxed <Link to="/security">CodeExecutor</Link> (Docker by default, unprivileged-namespace subprocess fallback), OIDC/JWT auth with RBAC and a per-request audit log, gitleaks + CycloneDX SBOM + pip-audit supply-chain hardening, production <Link to="/deployment">deploys</Link> (Docker, Helm, AWS Lambda, Cloudflare Worker), and three <Link to="/dx">DX surfaces</Link>: a VSCode extension, Jupyter magics, and a live <code>/dashboard</code>.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Activity size={28} /></div>
            <h3>v0.2.9: Observability &amp; Reliability</h3>
            <p>Structured JSON logging with secret redaction, Prometheus histograms, <Link to="/observability">SLO burn-rate tracking</Link>, OTel tracing with explicit samplers, plus <Link to="/reliability">reliability primitives</Link>: timeouts, jittered retries, per-provider circuit breakers, and bulkheads — validated by a deterministic chaos harness, a Hypothesis fuzz suite, and the <code>effgen loadtest</code> CLI.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Image size={28} /></div>
            <h3>v0.2.8: Multimodal Input</h3>
            <p><Link to="/multimodal">Image, audio, and video</Link> as first-class input types across Gemini, OpenAI, Groq, Anthropic, Together, and HF — plus local MLX-VLM on Apple Silicon. A unified <code>ContentPart</code> Message schema, per-provider preprocessing, capability gating, a <code>multimodal</code> preset, the <code>MultimodalDescribeTool</code>, and 5 cookbook walkthroughs.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><FileText size={28} /></div>
            <h3>v0.2.7: Prompt Library · 31 Templates</h3>
            <p>Curated, domain-organized templates across <Link to="/prompts">research, coding, data/SQL, legal, medical, creative, and business</Link>. Every template is a Python callable that renders deterministically with a fixture and a golden test. CLI: <code>effgen prompts list / show / eval / render / run</code> + interactive <code>playground</code>. Live eval validates output via <code>sqlglot.parse()</code>, <code>ast.parse()</code>, regex, or JSON schema. Legal + medical templates render mandatory non-advice disclaimers verbatim.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Wrench size={28} /></div>
            <h3>v0.2.6: Docs · Media · Comms · 58+ Total</h3>
            <p>14 new tools across OCR (<Link to="/tools">OCRTool</Link>), audio (AudioTranscribeTool), image (ImageInfoTool, ImageCaptionTool), documents (PDFTool, DOCXTool, ExcelTool), geo (WeatherTool, GeocodeTool, MapsTool), and live comms (EmailSMTPTool, EmailIMAPTool, SlackWebhookTool, DiscordWebhookTool) — plus two new presets: <code>media</code> (audio + vision) and <code>notify</code> (email + Slack + Discord).</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Wrench size={28} /></div>
            <h3>v0.2.5: 13 New Free Tools · 44+ Total</h3>
            <p>Academic (<Link to="/tools">PubMed, ArXiv, SemanticScholar</Link>), news &amp; RSS (RSSFeed, News), YouTube (Transcript, Metadata), social (Reddit, HackerNews), translation (Translate via LibreTranslate + offline argostranslate, LanguageDetect for 55+ languages), and fully-local QR (Generate, Read). All wired into the <code>research</code> and <code>general</code> presets.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><GitBranch size={28} /></div>
            <h3>v0.2.4: Policy ModelRouter + Cost CLI</h3>
            <p>Opt-in <Link to="/models">PolicyBasedRouter</Link> with FirstAvailable / CostBased / LatencyBased policies, explainable RouterDecisions, and <code>route_and_execute</code> failover on rate-limits, 5xx, timeouts, and BudgetExceededError. New <code>effgen cost today / week / by-provider / set-budget</code> CLI backed by SQLiteCostStore and SQLiteRateLimitStore.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Zap size={28} /></div>
            <h3>v0.2.1: Cerebras + OpenAI</h3>
            <p>Cerebras becomes a first-class backend with four registered models, two reliably free-tier callable models, streaming, model-dependent native tools, rate-limit coordination, and cost tracking. OpenAI adds gpt-5/gpt-5.4-nano chat models, o-series reasoning_effort controls, cached token metadata, structured outputs v2, and native tools.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Shield size={28} /></div>
            <h3>v0.2.2: Gemini + Anthropic</h3>
            <p>Gemini gains thinking_budget, grounding, Files API, and native tools. Anthropic gains Claude 4.x registry coverage, extended thinking, prompt caching, streaming polish, and experimental computer-use tools.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Database size={28} /></div>
            <h3>v0.2.3: ProviderRegistry</h3>
            <p>Groq, Together AI, Fireworks, Replicate, and HuggingFace Inference join the cloud provider roster. <Link to="/providers">ProviderRegistry</Link> powers lookup, auth checks, ambiguous ID handling, and <code>effgen doctor</code>.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Wrench size={28} /></div>
            <h3>Native Provider Tools</h3>
            <p><Link to="/native-provider-tools">Native provider tools</Link> expose OpenAI server-side web search, code execution, and file search in v0.2.1; Gemini grounding, URL context, and Anthropic computer-use wrappers are v0.2.2+.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Layers size={28} /></div>
            <h3>Backward Compatible</h3>
            <p>All v0.2.1-v0.3.0 additions are opt-in or additive. Existing v0.2.0 agents, tools, guardrails, RAG, workflows, clients, and API server deployments keep working without modification. The multimodal schema, observability / reliability layer, and v0.2.10 sandbox / auth / deploy / DX features are additive; the older Jinja2 TemplateManager, PromptChain, and PromptOptimizer all continue to work alongside the Prompt Library. v0.3.0 has no breaking API changes — every ergonomic addition is an additive alias. (Two behavior notes: as of v0.2.10 the CodeExecutor sandboxes by default — set <code>EFFGEN_SANDBOX_BACKEND=off</code> to opt out; and as of v0.3.0 <code>Agent.run()</code> never reports <code>success=True</code> with empty output, so branch on <code>result.success</code> / <code>result.metadata["reason"]</code>.)</p>
          </div>
        </div>
      </section>

      {/* Quick Start */}
      <section className="quick-start">
        <div className="section-header">
          <h2>Quick Start</h2>
          <p>Create your first agent in 30 seconds</p>
        </div>

        <div className="quick-start-content">
          <div className="install-steps">
            <div className="step">
              <span className="step-number">1</span>
              <div className="step-content">
                <h3>Install</h3>
                <code>pip install effgen</code>
              </div>
            </div>
            <div className="step">
              <span className="step-number">2</span>
              <div className="step-content">
                <h3>Code</h3>
                <span>Create an agent with tools</span>
              </div>
            </div>
            <div className="step">
              <span className="step-number">3</span>
              <div className="step-content">
                <h3>Run</h3>
                <span>Execute tasks autonomously</span>
              </div>
            </div>
          </div>

          <CodeBlock code={quickStartCode} language="python" filename="quickstart.py" />
        </div>
      </section>

      {/* Features */}
      <section className="features">
        <div className="section-header">
          <h2>Why effGen?</h2>
          <p>Built for Small Language Models, designed for production</p>
        </div>

        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon"><Zap size={28} /></div>
            <h3>vLLM-First Architecture</h3>
            <p>5-10x faster inference with automatic multi-GPU support, PagedAttention, and continuous batching.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Bot size={28} /></div>
            <h3>SLM Optimization + Prompt Library</h3>
            <p>Concise prompts, structured outputs, and step-by-step reasoning designed for smaller models. v0.2.7 adds the new <Link to="/prompts">Prompt Library</Link>: 31 curated, domain-organized templates with a golden + live eval harness and an interactive playground.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Wrench size={28} /></div>
            <h3>Universal Tools</h3>
            <p>66 built-in tools (computation, code execution, web, academic, news/RSS, YouTube, social, translation, language detection, QR codes, OCR, audio transcription, image analysis, document parsing — PDF/DOCX/Excel, geo/weather, email, webhooks, data science, DevOps, finance) with native function-calling plus MCP, A2A, and ACP protocol support. Plugin system for custom tools — installed plugins auto-discover as of v0.3.1.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Database size={28} /></div>
            <h3>Memory Systems</h3>
            <p>Three-layer memory: short-term conversation, long-term persistent storage, and vector semantic search.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><GitBranch size={28} /></div>
            <h3>Multi-Agent Orchestration</h3>
            <p>DAG workflows with auto-parallel execution, MessageBus pub/sub, SharedState, and AgentRegistry lifecycle management.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><Shield size={28} /></div>
            <h3>Production Ready</h3>
            <p>Docker sandboxing, security validation, logging, monitoring, and configuration management.</p>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta">
        <div className="cta-shapes">
          <div className="cta-shape cta-shape-1"></div>
          <div className="cta-shape cta-shape-2"></div>
          <div className="cta-shape cta-shape-3"></div>
        </div>
        <div className="cta-content">
          <h2>Ready to build?</h2>
          <p>Start building powerful AI agents with effGen today.</p>
          <div className="cta-actions">
            <Link to="/quickstart" className="btn btn-primary btn-lg">
              Get Started
              <ArrowRight size={20} />
            </Link>
            <Link to="/examples" className="btn btn-secondary btn-lg">
              View Examples
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
