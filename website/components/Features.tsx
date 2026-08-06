"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiZap, FiBox, FiTool, FiCode, FiUsers, FiShield, FiLayers, FiDatabase, FiX, FiCpu, FiBookOpen, FiTarget, FiServer, FiGitBranch, FiCheckCircle, FiActivity, FiImage, FiBarChart2, FiLock } from "react-icons/fi";
import { useRef, useState, useCallback, useEffect } from "react";
import Container from "./Container";
import { highlightCode } from "./syntaxHighlight";

/* ── Animated visuals for large cards ── */

function TaskDecompAnimation() {
  return (
    <div className="relative w-full h-40 flex items-center justify-center">
      <svg viewBox="0 0 400 180" className="w-full h-full" fill="none">
        <defs>
          <filter id="glow-decomp">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* Main task box */}
        <motion.rect
          x="145" y="5" width="110" height="34" rx="6"
          stroke="#00ff88" strokeWidth="2" fill="#00ff8810"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        />
        <motion.text x="200" y="27" textAnchor="middle" fill="#00ff88" fontSize="11" fontFamily="monospace"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
        >Complex Task</motion.text>

        {/* Splitting lines */}
        {[{ x: 80 }, { x: 200 }, { x: 320 }].map((pos, i) => (
          <motion.line
            key={i}
            x1="200" y1="39" x2={pos.x} y2="75"
            stroke="#00ff8860" strokeWidth="1" strokeDasharray="4 3"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.5 + i * 0.15 }}
          />
        ))}

        {/* Sub-task boxes */}
        {[
          { x: 25, label: "Research", color: "#00e5ff" },
          { x: 145, label: "Analyze", color: "#a78bfa" },
          { x: 265, label: "Synthesize", color: "#ffd700" },
        ].map((st, i) => (
          <motion.g key={i}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 + i * 0.15 }}
          >
            <rect x={st.x} y="75" width="110" height="30" rx="5" stroke={st.color} strokeWidth="1" fill={`${st.color}10`} />
            <text x={st.x + 55} y="95" textAnchor="middle" fill={st.color} fontSize="11" fontFamily="monospace">{st.label}</text>
          </motion.g>
        ))}

        {/* Result lines flowing back up */}
        {[{ x: 80 }, { x: 200 }, { x: 320 }].map((pos, i) => (
          <motion.line
            key={`r-${i}`}
            x1={pos.x} y1="105" x2="200" y2="135"
            stroke="#00ff8840" strokeWidth="1" strokeDasharray="3 4"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 1.4 + i * 0.1 }}
          />
        ))}

        {/* Final result box */}
        <motion.rect
          x="145" y="130" width="110" height="30" rx="6"
          stroke="#00ff88" strokeWidth="1.5" fill="#00ff8820"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0, 1, 1, 0.8, 1] }}
          transition={{ delay: 1.8, duration: 0.6 }}
        />
        <motion.text x="200" y="150" textAnchor="middle" fill="#00ff88" fontSize="11" fontFamily="monospace" fontWeight="bold"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2 }}
        >Merged Result</motion.text>

        {/* Animated pulse traveling along lines */}
        <motion.circle
          r="3" fill="#00ff88"
          filter="url(#glow-decomp)"
          animate={{
            cx: [200, 80, 80, 200, 200, 320, 320, 200, 200],
            cy: [39, 75, 105, 135, 39, 75, 105, 135, 39],
            opacity: [1, 1, 1, 1, 1, 1, 1, 1, 0],
          }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", repeatDelay: 1 }}
        />
      </svg>
    </div>
  );
}

function OrchestrationAnimation() {
  const agents = [
    { label: "Research", angle: 0, color: "#00e5ff", desc: "Web & APIs" },
    { label: "Coding", angle: 120, color: "#a78bfa", desc: "Python & JS" },
    { label: "Analysis", angle: 240, color: "#ffd700", desc: "Data & Reports" },
  ];

  return (
    <div className="relative w-full h-44 flex items-center justify-center">
      <svg viewBox="0 0 340 220" className="w-full h-full" fill="none">
        {/* Center orchestrator */}
        <motion.circle
          cx="170" cy="110" r="28"
          stroke="#00ff88" strokeWidth="1.5" fill="#00ff8812"
          animate={{ r: [28, 30, 28] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <text x="170" y="107" textAnchor="middle" fill="#00ff88" fontSize="9" fontFamily="monospace">Orchestrator</text>
        <motion.circle
          cx="170" cy="118" r="2" fill="#00ff88"
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />

        {/* Orbit ring */}
        <motion.circle
          cx="170" cy="110" r="75"
          stroke="#00ff8820" strokeWidth="0.5" fill="none" strokeDasharray="4 4"
          animate={{ strokeDashoffset: [0, 20] }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />

        {/* Agent nodes */}
        {agents.map((agent, i) => {
          const rad = (agent.angle * Math.PI) / 180;
          const cx = 170 + Math.cos(rad) * 75;
          const cy = 110 + Math.sin(rad) * 75;
          return (
            <motion.g
              key={i}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 + i * 0.2 }}
            >
              {/* Connection line */}
              <motion.line
                x1="170" y1="110" x2={cx} y2={cy}
                stroke={`${agent.color}50`} strokeWidth="1.5" strokeDasharray="3 3"
                animate={{ strokeDashoffset: [0, -12] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
              />
              {/* Node */}
              <circle cx={cx} cy={cy} r="18" stroke={agent.color} strokeWidth="1" fill={`${agent.color}12`} />
              <text x={cx} y={cy + 1} textAnchor="middle" fill={agent.color} fontSize="9" fontFamily="monospace">{agent.label}</text>
              {/* Description text */}
              <text x={cx} y={cy + 12} textAnchor="middle" fill="#888888" fontSize="7" fontFamily="monospace">{agent.desc}</text>
            </motion.g>
          );
        })}

        {/* Data pulses between orchestrator and agents */}
        {agents.map((agent, i) => {
          const rad = (agent.angle * Math.PI) / 180;
          const cx = 170 + Math.cos(rad) * 75;
          const cy = 110 + Math.sin(rad) * 75;
          return (
            <motion.circle
              key={`pulse-${i}`}
              r="2" fill={agent.color}
              style={{ filter: `drop-shadow(0 0 3px ${agent.color})` }}
              animate={{
                cx: [170, cx, cx, 170],
                cy: [110, cy, cy, 110],
                opacity: [1, 1, 0, 0],
              }}
              transition={{ duration: 3, delay: i * 1, repeat: Infinity, ease: "easeInOut" }}
            />
          );
        })}
      </svg>
    </div>
  );
}

function PresetStripVisual() {
  const presets = [
    { name: "math", tools: 2, temp: 0.3, color: "#00ff88", emoji: "\u{1F9EE}" },
    { name: "research", tools: 15, temp: 0.5, color: "#00e5ff", emoji: "\u{1F52C}" },
    { name: "coding", tools: 4, temp: 0.4, color: "#a78bfa", emoji: "\u{1F4BB}" },
    { name: "general", tools: 32, temp: 0.7, color: "#ffd700", emoji: "\u{1F680}" },
    { name: "rag", tools: 1, temp: 0.3, color: "#00c896", emoji: "\u{1F4D6}" },
    { name: "media", tools: 2, temp: 0.3, color: "#f59e0b", emoji: "\u{1F39E}\uFE0F" },
    { name: "notify", tools: 4, temp: 0.3, color: "#ec4899", emoji: "\u{1F4E2}" },
    { name: "multimodal", tools: 7, temp: 0.3, color: "#f472b6", emoji: "\u{1F5BC}\uFE0F" },
    { name: "minimal", tools: 0, temp: 0.7, color: "#ff9500", emoji: "\u26A1" },
  ];

  return (
    <div className="flex gap-2 w-full overflow-x-auto py-1">
      {presets.map((p, i) => (
        <motion.div
          key={i}
          className="flex-shrink-0 flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 dark:bg-black/30 border border-gray-200 dark:border-gray-800"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.1 }}
          whileHover={{ y: -2, borderColor: `${p.color}50` }}
        >
          <span className="text-base">{p.emoji}</span>
          <div className="flex flex-col">
            <span className="text-[10px] font-bold uppercase" style={{ color: p.color }}>{p.name}</span>
            <div className="flex items-center gap-2">
              <span className="text-[9px] text-gray-500">{p.tools} tools</span>
              <div className="w-8 h-1 rounded-full bg-gray-200 dark:bg-gray-800 overflow-hidden">
                <div className="h-full rounded-full" style={{ width: `${p.temp * 100}%`, background: p.temp <= 0.3 ? "#00e5ff" : p.temp <= 0.5 ? "#00ff88" : p.temp <= 0.7 ? "#ffd700" : "#ff6b6b" }} />
              </div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function MemoryFlowVisual() {
  return (
    <div className="flex items-center gap-1 text-[10px] font-mono py-2">
      <motion.span className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-400 border border-violet-500/20"
        animate={{ opacity: [0.6, 1, 0.6] }} transition={{ duration: 2, repeat: Infinity }}
      >short-term</motion.span>
      <motion.span className="text-gray-500"
        animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1.5, repeat: Infinity }}
      >&rarr;</motion.span>
      <motion.span className="px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-400 border border-violet-500/30"
        animate={{ opacity: [0.6, 1, 0.6] }} transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
      >long-term</motion.span>
      <motion.span className="text-gray-500"
        animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1.5, repeat: Infinity, delay: 0.3 }}
      >&rarr;</motion.span>
      <motion.span className="px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-400 border border-violet-500/40"
        animate={{ opacity: [0.6, 1, 0.6] }} transition={{ duration: 2, repeat: Infinity, delay: 1 }}
      >vector</motion.span>
    </div>
  );
}

/* ── Feature data ── */

interface FeatureItem {
  icon: any;
  title: string;
  description: string;
  features: string[];
  accent: string;
  visual?: "decomp" | "orchestration" | "presets" | "memory";
  code?: string;
  expandedContent?: string;
}

const features: FeatureItem[] = [
  /* ── Row 1: Most novel / important features ── */
  {
    icon: FiZap,
    title: "Real-World Usability & Polish (v0.3.1)",
    description: "The v0.3.1 release seals the sharp edges real professionals hit first: grounded results carry the sources they were built from, reasoning models finish token-heavy work, custom personas are honored on every path, multi-agent teams fail honestly, and a knowledge domain becomes a runnable agent in one call. No breaking API changes.",
    features: ["response.sources / .citations filled from the URLs a run actually retrieved — never the model's prose", "gpt-5 / o-series reasoning models finish instead of empty, billed results; cost + tokens + latency on every result", "One-call domain agents: LegalDomain().to_agent(); custom personas honored on every path"],
    accent: "#00e5ff",
    code: 'agent = create_agent("research", "openai:gpt-5-nano")\nr = agent.run("What is the capital of France? Cite a source.")\nprint(r.sources)                 # [\'https://en.wikipedia.org/wiki/Paris\']\nprint(r.metadata["cost_usd"], r.metadata["latency_ms"])\n\nlegal = LegalDomain().to_agent("openai:gpt-5-nano")   # prompt + tools + guardrails',
    expandedContent: "v0.3.1 is a real-world usability & polish release with no new providers or subsystems and no breaking API changes — every addition is additive. response.sources and response.citations are populated from the URLs a run actually retrieved (web_search / url_fetch / news / wikipedia) plus provider-native grounding (OpenAI url_citation annotations surfaced as metadata['grounding_chunks']; Gemini search grounding) — never from the model's prose. Reasoning models (the gpt-5 family, o-series) no longer return empty, billed results on token-heavy tasks: they get a larger default output budget (4096) across every path and truncation is detected and retried once. Cost, tokens, and latency land on every result (cost_usd, prompt/completion/total tokens, latency_ms/duration_s), with a readable sub-cent formatter. A custom system_prompt now steers every response (it was silently dropped on the direct, streaming, and native-tool paths). Collaborative teams and workflow DAGs fail closed and route by name; the OpenAI-compatible server stops silently downgrading (a client tool it doesn't host is rejected with a clear 400 unknown_tool, and /v1/embeddings reflects its real backend instead of serving TF-IDF vectors under a neural model's name). A knowledge domain becomes a runnable agent in one call — LegalDomain().to_agent('gpt-5-nano') wires the domain's prompt, recommended tools, and guardrails; a RAG agent accepts a pre-built VectorMemoryStore as its knowledge_base. models status shows physical GPU memory, sync Agent.run() no longer hangs on an MCP stdio tool (clear TimeoutError), installed tool plugins auto-discover, and effgen run --json emits the full result document to stdout.",
  },
  {
    icon: FiCheckCircle,
    title: "Fail-Closed & Hardened (v0.3.0)",
    description: "The v0.3.0 stabilization release makes everything robust: failures are loud and typed instead of silently succeeding, the model catalog updates itself, the server fails closed, and the built-in tools are sandboxed.",
    features: ["Agent.run() never returns success=True with empty output", "Self-updating catalog: effgen models refresh + drift warnings", "Shared SSRF guard, sandboxed PythonREPL, path-confined file tools"],
    accent: "#00ff88",
    code: 'result = agent.run("What is 24344 * 334?")\nif not result.success:\n    print(result.metadata["error"]["category"])  # auth / not_found / rate_limited / transient / timeout / fatal\nelse:\n    print(result.output)',
    expandedContent: "v0.3.0 is a major stabilization & hardening release with no breaking API changes — every ergonomic addition is an additive alias. Agent.run() can no longer return success=True with an empty answer; the direct and tool paths return the same failure shape (success=False, a coarse metadata['reason'] stage label, and a typed redacted metadata['error'] dict), and classify_provider_error() populates metadata['error']['category'] with a stable taxonomy (auth / not_found / rate_limited / transient / timeout / fatal) so retries fire only when retrying could help. Every provider ships a priced, dated catalog snapshot; effgen models refresh diffs the live API and check_drift() warns once when stale. The API server fails closed (forged JWTs are rejected, CORS + the metrics dashboard are locked down, a missing upstream key returns 502 not 401). Built-in tools are sandboxed: PythonREPL enforces its timeout from outside the executed code, one shared SSRF guard protects every URL tool and re-validates on each redirect, file tools are path-confined, and the unsafe pickle / eval paths are gone.",
  },
  {
    icon: FiGitBranch,
    title: "Policy-Based ModelRouter",
    description: "Compose FirstAvailable, CostBased, and LatencyBased policies over the cloud providers with explainable RouterDecisions and transparent failover.",
    features: ["Cost, latency, and first-available policies", "Auto-failover on rate-limit, 5xx, timeout, budget", "RouterEvent subscribers · cost CLI"],
    accent: "#ff6b6b",
    code: 'from effgen import PolicyBasedRouter, RoutingContext, CostBasedPolicy\nfrom effgen.models.capabilities import Capability\nrouter = PolicyBasedRouter(policies=[CostBasedPolicy()])\ndecision = router.route(RoutingContext(\n    user_budget_usd=0.01,\n    required_capabilities={Capability.chat, Capability.tools},\n))',
    expandedContent: "v0.2.4 adds an opt-in PolicyBasedRouter that sits between your code and the 9 cloud providers. Describe the request with RoutingContext (prompt_tokens_estimate, user_budget_usd, latency_budget_ms, required_capabilities) and the router runs FirstAvailable, CostBased, or LatencyBased policies to pick a provider. Every RouterDecision lists eliminated candidates with reasons (no_key, rate_limited, cost_exceeds_budget, latency_exceeds_sla) and the winning policy. route_and_execute(context, fn) auto-retries RateLimitExceeded, ProviderTransientError, ModelTimeoutError, and BudgetExceededError, emitting a RouterEvent per failover hop. SQLiteRateLimitStore and SQLiteCostStore coordinate across worker processes via ~/.effgen/rate_limits.sqlite and ~/.effgen/costs.sqlite, and the new effgen cost CLI prints today/week/by-provider summaries and sets daily/monthly budgets.",
  },
  {
    icon: FiCpu,
    title: "14 Inference Backends",
    description: "Five local engines plus 9 cloud providers: OpenAI, Anthropic, Gemini, Cerebras, Groq, Together, Fireworks, Replicate, and HF Inference.",
    features: ["9 cloud providers with one Agent API", "Provider-prefixed model IDs", "Streaming + provider-supported tools"],
    accent: "#00ff88",
    code: 'model = load_model("groq:llama-3.3-70b-versatile")\n# or provider="groq"',
    expandedContent: "effGen v0.2.3 brings the backend surface to 14 total entry points: local Transformers, vLLM, MLX, MLX-VLM, and GGUF, plus OpenAI, Anthropic, Gemini, Cerebras, Groq, Together AI, Fireworks, Replicate, and HuggingFace Inference. All cloud providers use the same Agent API, v0.2.3 unifies ModelAuthError on bad credentials, and native tool support is tracked per provider and model.",
  },
  {
    icon: FiActivity,
    title: "ProviderRegistry + Doctor",
    description: "A unified provider registry lists providers, resolves models, catches ambiguous IDs, and checks API key readiness.",
    features: ["list_providers() / list_models()", "effgen doctor --json", "Unified ModelAuthError"],
    accent: "#00e5ff",
    code: 'from effgen.models.registry import lookup\nprovider, cls, info = lookup("groq:llama-3.1-8b-instant")',
    expandedContent: "ProviderRegistry is the shared source of truth for cloud adapters. Adapters self-register on import, model lookup understands provider:model_id prefixes, duplicate bare IDs raise AmbiguousModelError with the valid providers, and effgen doctor reports which provider keys are ready from your shell or project .env.",
  },
  {
    icon: FiBox,
    title: "Automatic Sub-Agent Routing",
    description: "Runs complex tasks through Agent sub-agent mode with routing, decomposition, and result synthesis built into the agent loop.",
    features: ["AgentMode.AUTO routing", "Built-in decomposition engine", "Parallel or sequential sub-agents"],
    accent: "#00ff88",
    code: 'from effgen import Agent, AgentConfig\nfrom effgen.core.agent import AgentMode\nagent.run("analyze this market", mode=AgentMode.AUTO)',
    visual: "decomp",
    expandedContent: "AgentConfig(enable_sub_agents=True) lets the Agent decide when to keep a task in single-agent mode or route it through specialized sub-agents. The router and decomposition engine score the task, choose a strategy, execute subtasks, and synthesize the final response.",
  },
  {
    icon: FiUsers,
    title: "Multi-Agent Orchestration",
    description: "Coordinate multiple specialized agents with lifecycle management and agent-to-agent communication.",
    features: ["Team patterns", "Shared state", "Message bus"],
    accent: "#ff6b6b",
    code: 'from effgen.core import MultiAgentOrchestrator\norch = MultiAgentOrchestrator()\nteam = orch.create_team("research", agents)\norch.assign_task("analyze data", team)',
    visual: "orchestration",
    expandedContent: "MultiAgentOrchestrator coordinates registered Agent instances through sequential, parallel, hierarchical, collaborative, competitive, and pipeline team patterns. It records execution events, publishes task messages, and stores shared state for team runs.",
  },
  {
    icon: FiZap,
    title: "Ultra-Fast vLLM Integration",
    description: "Native vLLM support delivers 5-10x faster inference. Auto multi-GPU tensor parallelism and PagedAttention.",
    features: ["5-10x faster inference", "PagedAttention memory efficiency", "Auto multi-GPU support"],
    accent: "#ffd700",

    code: 'pip install effgen[vllm]',
    expandedContent: "vLLM integration provides continuous batching, PagedAttention for efficient memory usage, and automatic tensor parallelism across multiple GPUs. Install with pip install effgen[vllm] and get 5-10x speedup instantly.",
  },
  {
    icon: FiTool,
    title: "Universal Tool Integration",
    description: "66 local tools plus provider-native tools across OpenAI, Gemini, and experimental Anthropic adapter specs.",
    features: ["66 local tools (docs, OCR, audio, image, geo, comms, research, news, social, finance, DevOps, RAG…)", "OpenAI / Gemini Agent-native tools", "Anthropic experimental tool specs"],
    accent: "#00e5ff",

    code: 'tools=[OCRTool(), PDFTool(), WeatherTool(), SlackWebhookTool()]\n# tool_calling_mode="auto"',
    expandedContent: "effGen ships 58+ production-ready local tools and layers in provider-native tools where the API offers first-party execution. v0.2.6 added 14 new tools: OCRTool (Tesseract + OCR.space fallback), AudioTranscribeTool (faster-whisper + HF Inference fallback), ImageInfoTool + ImageCaptionTool (Pillow + vision router), PDFTool / DOCXTool / ExcelTool (pypdf + pdfplumber, python-docx, openpyxl + pandas), WeatherTool (Open-Meteo) + GeocodeTool (Nominatim) + MapsTool (OSM static), and live communication tools EmailSMTPTool + EmailIMAPTool + SlackWebhookTool + DiscordWebhookTool. v0.2.5 added 13 free / no-auth tools spanning academic (PubMed, ArXiv, SemanticScholar), news/RSS, YouTube, social (Reddit, HN), translation (LibreTranslate + argostranslate offline fallback), language detection (langdetect, 55+ languages), and QR codes. v0.2.1 OpenAI native tools route web_search, code_interpreter, and file_search through the Responses API. v0.2.2+ Gemini native tools enable Google Search grounding, URL context, and server-side code execution; Anthropic exposes experimental Bash, text editor, and computer-use specs for direct AnthropicAdapter calls. The Agent loop supports ReAct, native, and hybrid modes.",
  },
  {
    icon: FiShield,
    title: "Guardrails & Safety",
    description: "Offline, ML-free guardrails for toxicity, PII, prompt injection, topics, length, and tool safety. Composable chains with four presets.",
    features: ["PII (SSN/email/phone/CC-Luhn), Toxicity, Topic, Length", "Prompt-injection detection (low/med/high)", "Tool input/output/permission guardrails"],
    accent: "#ff6b6b",
    code: 'from effgen.guardrails import get_guardrail_preset\nguardrails=get_guardrail_preset("standard")',
    expandedContent: "The effgen.guardrails module provides composable input/output validation that runs entirely offline — no external APIs, no ML models. Content guardrails handle toxicity, PII (SSN, email, phone, credit cards with Luhn validation, IP), topic allow/deny, and length. A dedicated PromptInjectionGuardrail ships with low/medium/high sensitivity tiers. Tool-safety guardrails cover input validation, output PII stripping, and permission control (allow/deny/require_approval). Wire them into any Agent via AgentConfig.guardrails or pick a preset: strict, standard, minimal, none.",
  },
  {
    icon: FiBookOpen,
    title: "Advanced RAG Pipeline",
    description: "Hybrid search (dense + BM25 + keyword) with RRF fusion, semantic/code/table/hierarchical chunkers, rerankers, and inline citations.",
    features: ["DocumentIngester (txt/md/pdf/docx/html/…)", "HybridSearchEngine with RRF", "CrossEncoder / LLM / rule-based rerankers"],
    accent: "#00e5ff",
    code: 'from effgen.presets import create_agent\nagent = create_agent("rag", model, knowledge_base="./docs/")',
    expandedContent: "effgen.rag is a complete retrieval pipeline: DocumentIngester loads TXT/MD/JSON/JSONL/CSV/HTML out of the box (plus optional PDF/DOCX/EPUB) with SHA-256 dedup. Chunk with SemanticChunker, CodeChunker (py/js/ts/go/rust/java), TableChunker, or HierarchicalChunker. HybridSearchEngine fuses dense embeddings, BM25, keyword, and metadata filtering via Reciprocal Rank Fusion. Rerank with CrossEncoderReranker, LLMReranker, or RuleBasedReranker. ContextBuilder handles token budgeting and inline [N] citations, exposed on AgentResponse.citations and .sources.",
  },
  {
    icon: FiGitBranch,
    title: "DAG Workflow Engine",
    description: "Define multi-agent pipelines as a DAG with cycle detection, conditional branching, and auto-parallel execution.",
    features: ["WorkflowDAG with Kahn's topo sort", "Auto-parallelize independent nodes", "YAML workflow definitions + CLI"],
    accent: "#a78bfa",
    code: 'dag = WorkflowDAG()\ndag.add_node("research", agent=r)\ndag.add_node("write", agent=w)\ndag.add_edge("research", "write")',
    expandedContent: "WorkflowDAG validates your multi-agent graph via Kahn's topological sort (rejecting cycles), then executes it — parallelising independent nodes via asyncio.gather. Edges carry typed data between nodes and can be conditional (skipped when a predicate returns False). Workflows can be defined in YAML and run via `effgen workflow run <file>`. Pairs with MessageBus (pub/sub between agents) and SharedState (thread-safe per-namespace KV store with snapshots and event log).",
  },
  {
    icon: FiCheckCircle,
    title: "Evaluation & Regression",
    description: "Built-in test suites, LLM-judge scoring, and baseline regression tracking. Compare multiple models side-by-side.",
    features: ["5 suites: Math, ToolUse, Reasoning, Safety, Conversation", "EXACT / CONTAINS / REGEX / SEMANTIC / LLM_JUDGE", "Nightly CI opens GitHub issues on drift"],
    accent: "#ffd700",
    code: 'from effgen.eval import AgentEvaluator, MathSuite\nresults = AgentEvaluator(agent).run(MathSuite())',
    expandedContent: "effgen.eval ships with five test suites totaling 270 cases: MathSuite (77), ToolUseSuite (93), ReasoningSuite (40), SafetySuite (40), ConversationSuite (20). AgentEvaluator supports exact match, substring, regex, semantic similarity (via optional sentence-transformers), and LLM-as-judge scoring. RegressionTracker diffs runs against stored baselines with warning/high/critical severity thresholds. ModelComparison runs a matrix of models × suites with markdown/JSON export. All wired into CLI: `effgen eval --suite math` and `effgen compare --models qwen,phi --suite reasoning`.",
  },
  {
    icon: FiServer,
    title: "Production API Server",
    description: "OpenAI-compatible /v1/chat/completions + /v1/embeddings with streaming, priority queue, agent pool, and multi-tenant API keys.",
    features: ["OpenAI-compatible with model aliases", "RequestQueue (priority + backpressure)", "TenantManager with rate limits"],
    accent: "#00ff88",
    code: 'effgen serve --host 0.0.0.0 --port 8000\n# /v1/chat/completions (stream + tools)',
    expandedContent: "API Server v2 exposes OpenAI-compatible /v1/chat/completions (with tools and SSE streaming) and /v1/completions, plus an /v1/embeddings endpoint backed by SentenceTransformers (with TFIDFEmbedder fallback) and LRU + SQLite caching. Model aliases let callers use names like gpt-4 → Qwen2.5-7B. Under the hood: a priority RequestQueue with fair scheduling and QueueFullError backpressure, an AgentPool (min/max, idle TTL, health checks), and TenantManager for per-tenant rate limits, model restrictions, tool permissions, and hashed API keys. Production middleware adds request IDs, CORS, gzip, and graceful shutdown.",
  },
  {
    icon: FiCpu,
    title: "Local + Cloud Model Routing",
    description: "Route across local SLM engines and 9 cloud providers with explicit provider names or provider:model_id prefixes.",
    features: ["Local: Transformers, vLLM, MLX, MLX-VLM, GGUF", "Cloud: OpenAI, Anthropic, Gemini, Cerebras, Groq, Together, Fireworks, Replicate, HF", "v0.2.3 ProviderRegistry-backed lookup"],
    accent: "#00e5ff",
    code: 'pip install "effgen[groq]"\npip install "effgen[all]"',
    expandedContent: "Model loading can stay local with Transformers, vLLM, MLX, MLX-VLM, or GGUF, or it can route to any registered cloud provider. v0.2.1 added Cerebras; v0.2.3 added Groq, Together AI, Fireworks, Replicate, and HuggingFace Inference. Use provider prefixes such as groq:llama-3.3-70b-versatile when model IDs overlap across providers.",
  },
  {
    icon: FiTarget,
    title: "Model Router & Multi-Model",
    description: "Smart routing by task complexity and model capabilities, with speculative execution across two models for best-first-wins.",
    features: ["Complexity analysis (< 1ms heuristic)", "Capability registry for 12+ SLMs", "Speculative multi-model execution"],
    accent: "#a78bfa",
    code: 'config = AgentConfig(name="router", model=fast,\n  models=[fast, strong], speculative_execution=True)',
    expandedContent: "ModelRouter picks the best model for each query: estimate_complexity() runs sub-millisecond keyword analysis across math/code/reasoning/multilingual axes; MODEL_CAPABILITIES pre-populates profiles for 12+ SLMs (Qwen 0.5B-7B, Llama 1B-3B, Phi-3/3.5/4, Mistral 7B, Gemma 2B/9B). Set multiple models on AgentConfig.models and flip speculative_execution=True to run two models concurrently via asyncio.wait(FIRST_COMPLETED). ModelPool provides LRU + GPU-memory eviction and hot-swap, with CLI: `effgen models load|unload|status`.",
  },
  {
    icon: FiActivity,
    title: "Observability & Debug",
    description: "OpenTelemetry tracing, Prometheus histograms, Grafana dashboard, and a step-through DebugAgent with rich TUI.",
    features: ["OTLP/Jaeger/Zipkin exporters", "Latency p50/p95/p99, GPU memory, error rate", "DebugAgent captures per-iteration trace"],
    accent: "#ffd700",
    code: 'agent.run("task", debug=True)\n# or: DebugAgent(config).run(task)',
    expandedContent: "Full observability stack: OpenTelemetry with Resource, BatchSpanProcessor, and OTLP/Jaeger/Zipkin/console exporters; cross-agent trace propagation via LogRunContext; structured JSON logging with run_id/workflow_id/agent_name/session_id correlation. Prometheus metrics include response_latency and token_usage histograms with p50/p95/p99, tool_execution_time, GPU memory gauges, and labels. A Grafana dashboard (12 panels) ships in the repo. For debugging, wrap any agent in DebugAgent or pass debug=True — each iteration's raw_prompt, raw_response, thought, action, observation, tokens, and latency are captured in DebugTrace, inspectable via `effgen debug`.",
  },
  {
    icon: FiUsers,
    title: "Human-in-the-Loop",
    description: "Approval workflows, clarification, free-text input, and feedback collection — with per-tool approval modes and timeouts.",
    features: ["approval_mode: always / first_time / dangerous_only", "ClarificationRequest with ambiguity detection", "FeedbackCollector (thumbs/rate/comment)"],
    accent: "#00ff88",
    code: 'AgentConfig(name="reviewed", model=model,\n  approval_mode="dangerous_only", approval_callback=my_cb)',
    expandedContent: "Pause agents for human interaction wherever needed. HumanApproval / HumanInput / HumanChoice all support timeouts via ThreadPoolExecutor. Per-tool approval is driven by ToolMetadata.requires_approval together with AgentConfig.approval_mode (always, first_time, never, dangerous_only) and approval_callback. ClarificationRequest offers options + free-text; ClarificationDetector uses heuristics (short query, vague words, multi-tool ambiguity) to decide when to ask. FeedbackCollector captures thumbs/rate/comment per turn and exports to JSONL.",
  },
  {
    icon: FiDatabase,
    title: "Checkpointing & Sessions",
    description: "Persist full agent state — scratchpad, memory, tool states — to JSON or SQLite. Resume multi-hour runs across processes.",
    features: ["CheckpointManager: filesystem + SQLite", "Session/SessionManager with UUIDs + expiry", "BackgroundTaskRunner (priority queue, pause/resume)"],
    accent: "#a78bfa",
    code: 'agent.run("task", checkpoint_interval=3)\nagent.resume(checkpoint_id="...")',
    expandedContent: "CheckpointManager serializes Agent state (scratchpad, iteration, partial_output, tool_calls, tokens, memory, tool_states) to JSON (human-readable — no pickle) or SQLite. Periodic saves via agent.run(..., checkpoint_interval=N) and resume via agent.resume(checkpoint_id=...). The Session / SessionManager layer provides persistent conversations keyed by UUID (or user-supplied id) under ~/.effgen/sessions with expiry and cleanup. For long-running jobs, BackgroundTaskRunner offers a priority queue with pause/resume/cancel and threading workers — plus Agent.run_background() / get_task_status() / cancel_task().",
  },
  /* ── Row 2: Supporting features ── */
  {
    icon: FiCode,
    title: "Prompt Library + SLM-Optimized Prompts",
    description: "The Prompt Library \u2014 35 curated, domain-organized templates across research, coding, data/SQL, legal, medical, creative, business, and education \u2014 plus the established Jinja2 templating and ReAct loop tuned for SLMs.",
    features: ["35 LibraryPrompt templates \u00b7 8 domains", "Golden + live eval harness (sqlglot.parse / ast.parse)", "effgen prompts CLI + playground REPL"],
    accent: "#a78bfa",

    code: 'p = registry.get("data.sql_from_nl.v1")\np.template(schema_ddl=..., question=...)',
    expandedContent: "The Prompt Library lives at effgen.prompts.library \u2014 35 LibraryPrompt callables grouped by domain: research (5), coding (5), data/SQL (5), legal (3), medical (3), creative (5), business (5), education (4). Each prompt declares its name (e.g. 'data.sql_from_nl.v1'), domain, variant (zero_shot / cot / few_shot / tool / structured), description, deterministic render template, JSON-Schema input_schema, fixture, optional expected_shape, and tags. The PromptRegistry auto-discovers domain packages on first access; PromptEval ships both eval_golden (compares against stored .txt golden files) and eval_live (runs through a model and validates expected_shape \u2014 including sqlglot.parse() for SQL and ast.parse() for generated Python). All 35 templates are surfaced through effgen prompts list / show / eval / render / run and an interactive playground REPL. Mandatory non-advice disclaimers are rendered verbatim in every legal and medical template's system prompt and enforced by unit tests. The older Jinja2 TemplateManager, PromptChain, and PromptOptimizer all remain available unchanged.",
  },
  {
    icon: FiLayers,
    title: "Agent Presets",
    description: "One-line agent creation with ready-to-use configurations. Math, research, coding, general, rag, media, notify, multimodal, and minimal.",
    features: ["create_agent() factory", "9 built-in presets (incl. media, notify, multimodal)", "CLI --preset for non-RAG presets"],
    accent: "#ff9500",
    code: 'agent = create_agent("multimodal", model)\nagent.run("Describe /tmp/chart.png")',
    visual: "presets",
    expandedContent: "Create production-ready agents with a single line: create_agent('research', model). Each preset bundles the optimal tools, temperature, max_iterations, and system prompt for its use case. The rag preset auto-ingests a knowledge_base directory and wires the Retrieval tool with hybrid search and inline [N] citations. v0.2.6 adds media (AudioTranscribeTool + ImageCaptionTool) for audio + vision pipelines and notify (EmailSMTP + EmailIMAP + SlackWebhook + DiscordWebhook) for alerts and digests; v0.2.8 adds multimodal, which wires the MultimodalDescribeTool to auto-route image / audio / video inputs to the right tool.",
  },
  {
    icon: FiDatabase,
    title: "Integrated Memory System",
    description: "Short-term, long-term, and vector memory connected to every agent. Persistent multi-turn context.",
    features: ["ShortTerm + LongTerm memory", "Vector store (FAISS/Chroma)", "Auto-summarization"],
    accent: "#a78bfa",
    code: 'agent.memory.store("key fact")\nagent.memory.recall("context")',
    visual: "memory",
    expandedContent: "Three-tier memory: ShortTermMemory for current conversation, LongTermMemory for persistent facts, and VectorMemory (FAISS/Chroma) for semantic retrieval. Auto-summarization keeps context compact across long sessions.",
  },
  {
    icon: FiImage,
    title: "Multimodal Input",
    description: "v0.2.8 makes image, audio, and video first-class input types across 6 cloud providers plus local MLX-VLM, with a unified ContentPart Message schema and capability gating.",
    features: ["Image / audio / video across Gemini, OpenAI, Groq, Anthropic, Together, HF + MLX-VLM", "Unified ContentPart schema (back-compatible)", "multimodal preset + MultimodalDescribeTool"],
    accent: "#f472b6",
    code: 'agent = create_agent("multimodal", model)\nagent.run("Describe /tmp/chart.png")',
    expandedContent: "v0.2.8 introduces a typed ContentPart union (TextPart, ImagePart, AudioPart, VideoPart, ToolCallPart, ToolResultPart) — Message(role, 'text') still works. image_from / audio_from / video_from helpers accept bytes, paths, URLs, PIL.Image, and np.ndarray; per-provider preprocessing handles resize, downsample, and ffmpeg keyframe sampling. Every adapter raises CapabilityNotSupportedError instead of silently downcasting an image. The multimodal preset and MultimodalDescribeTool auto-route image / audio / video inputs. Five cookbook walkthroughs cover image Q&A, audio transcribe + reason, video summarize, OCR + LLM, and chart reading.",
  },
  {
    icon: FiBarChart2,
    title: "Observability & Reliability",
    description: "v0.2.9 turns effGen into something you can operate in production: structured logging with secret redaction, Prometheus metrics, SLO tracking, OTel tracing, retries, circuit breakers, and bulkheads.",
    features: ["Structured logging + encoder-level secret redaction", "Prometheus /metrics + SLO /slo burn-rate", "Timeouts, jittered retries, circuit breakers, bulkheads"],
    accent: "#00e5ff",
    code: 'log.event("agent.started", preset="general")\nprint(export_metrics())  # Prometheus',
    expandedContent: "v0.2.9 ships effgen.observability and effgen.reliability. Structured JSON logging redacts secrets at the encoder so every path is covered. Prometheus histograms + token counters are exposed at GET /metrics; SLO + SLOTracker burn-rate tracking at GET /slo. OTel tracing uses explicit samplers (no implicit head=1.0) and a canonical span-attribute spec. Reliability primitives add explicit timeouts (no timeout=None), jittered retries, three-state per-provider circuit breakers, and bulkheads — validated by a deterministic Chaos(seed) harness (6 fault types, 4 scenarios) and a Hypothesis fuzz suite. The effgen loadtest CLI reports throughput + p50/p95/p99, and 6 Alertmanager rules ship with an AlertWebhook.",
  },
  {
    icon: FiLock,
    title: "Security, Edge & Developer Experience",
    description: "v0.2.10 hardens effGen end-to-end: a sandboxed CodeExecutor, OIDC auth + RBAC + audit log, supply-chain scanning, four production deploy targets, and three DX surfaces.",
    features: ["Sandboxed CodeExecutor (Docker / unprivileged-namespace subprocess)", "OIDC auth + RBAC + audit log; gitleaks + SBOM + pip-audit", "Docker / Helm / Lambda / Cloudflare + VSCode / Jupyter / dashboard"],
    accent: "#a78bfa",
    code: 'docker build -f deploy/docker/Dockerfile -t effgen:0.3.1 .\neffgen serve --port 8000  # OIDC auth on',
    expandedContent: "v0.2.10 sandboxes the CodeExecutor by default — DockerSandbox (--read-only, --network=none, --cap-drop=ALL, 256m) when Docker is available, otherwise an unprivileged user-namespace SubprocessSandbox. The API server validates Bearer JWTs via OIDC, enforces RBAC with daily cost caps (403 / 429), and writes a per-request redacted audit log. Supply-chain hardening adds gitleaks pre-commit + CI, a CycloneDX SBOM, pip-audit, and EFFGEN_VERIFY_HASHES hash verification. Four deploy targets ship under deploy/: a multi-stage Dockerfile, a Helm chart (HPA / PDB / NetworkPolicy), an AWS Lambda handler (Mangum + SAM), and a Cloudflare Worker edge proxy. Three DX surfaces round it out: a VSCode extension, Jupyter magics, and a live local dashboard at /dashboard.",
  },
  {
    icon: FiShield,
    title: "Production Infrastructure",
    description: "Sandboxed code execution, structured logging with secret redaction, OpenTelemetry tracing, Prometheus metrics, SLO tracking, and persistent state for production workloads.",
    features: ["Sandboxed CodeExecutor (v0.2.10)", "OpenTelemetry tracing + Prometheus /metrics (v0.2.9)", "Docker / Helm / Lambda / Cloudflare deploys (v0.2.10)"],
    accent: "#00ff88",

    code: 'docker run effgen:0.3.1\n# + OIDC auth, metrics & tracing',
    expandedContent: "Deploy with confidence using the v0.2.10 sandboxed CodeExecutor, OIDC/JWT auth with RBAC and a per-request audit log, v0.2.9 structured logging with encoder-level secret redaction, OpenTelemetry distributed tracing, Prometheus metrics + SLO burn-rate endpoints, and persistent state management. Ship to Docker, Kubernetes (Helm), AWS Lambda, or a Cloudflare Worker edge proxy.",
  },
];

/* ── Feature Card ── */

function FeatureCard({ feature, index, onExpand }: { feature: FeatureItem; index: number; onExpand: () => void }) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const [mousePos, setMousePos] = useState({ x: 50, y: 50 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const rotateX = ((y - centerY) / centerY) * -6;
    const rotateY = ((x - centerX) / centerX) * 6;
    setTilt({ x: rotateX, y: rotateY });
    setMousePos({ x: (x / rect.width) * 100, y: (y / rect.height) * 100 });
  }, []);

  const handleMouseLeave = useCallback(() => {
    setTilt({ x: 0, y: 0 });
    setIsHovered(false);
  }, []);

  return (
    <div className="tilt-parent h-full">
      <motion.div
        ref={cardRef}
        initial={{ y: 40, opacity: 0, scale: 0.95 }}
        animate={{ y: 0, opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" as const, delay: Math.min(index, 8) * 0.05 }}
        className="group relative p-6 rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 backdrop-blur-sm overflow-hidden cursor-pointer shadow-sm dark:shadow-none tilt-card h-full flex flex-col"
        style={{
          transform: `perspective(1000px) rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)`,
          transformStyle: "preserve-3d",
          transition: isHovered ? "none" : "transform 0.4s ease-out",
        }}
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={handleMouseLeave}
        onClick={onExpand}
      >
        {/* Mouse-tracking radial glow */}
        <div
          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl pointer-events-none"
          style={{
            background: `radial-gradient(circle at ${mousePos.x}% ${mousePos.y}%, ${feature.accent}15 0%, transparent 60%)`,
          }}
        />

        {/* Border glow on hover */}
        <motion.div
          className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"
          style={{ boxShadow: `inset 0 0 0 1px ${feature.accent}30` }}
        />

        {/* Top section: icon + title */}
        <div className="flex flex-col">
          {/* Icon */}
          <motion.div
            className="relative w-12 h-12 rounded-xl flex items-center justify-center overflow-hidden flex-shrink-0"
            style={{
              background: `${feature.accent}15`,
              border: `1px solid ${feature.accent}30`,
            }}
            whileHover={{ rotate: 360, scale: 1.1 }}
            transition={{ duration: 0.5 }}
          >
            <feature.icon style={{ color: feature.accent }} size={22} />
            <motion.div
              className="absolute inset-[-2px] rounded-xl pointer-events-none"
              style={{
                background: `conic-gradient(from 0deg, transparent 60%, ${feature.accent}40 80%, transparent 100%)`,
              }}
              animate={{ rotate: 360 }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
            />
          </motion.div>

          <div className="mt-4">
            <h3 className="text-lg font-bold mb-2 text-gray-900 dark:text-white group-hover:text-green-800 dark:group-hover:text-green-50 transition-colors">
              {feature.title}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-500 leading-relaxed group-hover:text-gray-700 dark:group-hover:text-gray-400 transition-colors">
              {feature.description}
            </p>
          </div>
        </div>

        {/* Code snippets only shown in expanded modal */}

        {/* Feature bullets + click hint pushed to bottom */}
        <div className="mt-auto pt-4">
          <ul className="space-y-1.5">
            {feature.features.map((item, idx) => (
              <motion.li
                key={idx}
                className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-500"
                whileHover={{ x: 4 }}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ background: feature.accent, boxShadow: `0 0 6px ${feature.accent}60` }}
                />
                {item}
              </motion.li>
            ))}
          </ul>

          <div className="mt-3 text-[10px] text-gray-400 dark:text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity font-medium tracking-wide uppercase">
            Click for details
          </div>
        </div>

        {/* Bottom accent line */}
        <motion.div
          className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ background: `linear-gradient(90deg, transparent, ${feature.accent}, transparent)` }}
        />
      </motion.div>
    </div>
  );
}

/* ── Expanded Detail Modal ── */

function ExpandedDetail({ feature, onClose }: { feature: FeatureItem; onClose: () => void }) {
  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      />
      <motion.div
        className="relative w-full max-w-xl rounded-2xl bg-white dark:bg-[#0a1a0f] border border-gray-200 dark:border-green-500/20 shadow-2xl overflow-hidden"
        style={{ boxShadow: `0 0 60px ${feature.accent}15` }}
        initial={{ scale: 0.9, opacity: 0, y: 30 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.9, opacity: 0, y: 30 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
      >
        {/* Header bar */}
        <motion.div
          className="h-1"
          style={{ background: `linear-gradient(90deg, transparent, ${feature.accent}, transparent)` }}
        />

        <div className="p-6">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 hover:text-gray-900 dark:hover:text-white transition-colors z-10"
          >
            <FiX size={16} />
          </button>

          <div className="flex items-center gap-3 mb-4">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ background: `${feature.accent}15`, border: `1px solid ${feature.accent}30` }}
            >
              <feature.icon style={{ color: feature.accent }} size={20} />
            </div>
            <h3 className="text-xl font-black text-gray-900 dark:text-white">{feature.title}</h3>
          </div>

          <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-5">
            {feature.expandedContent}
          </p>

          {/* Interactive visual (only shown in expanded view) */}
          {feature.visual === "decomp" && (
            <div className="mb-5">
              <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Live Visualization</h4>
              <div className="rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-3">
                <TaskDecompAnimation />
              </div>
            </div>
          )}
          {feature.visual === "orchestration" && (
            <div className="mb-5">
              <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Live Visualization</h4>
              <div className="rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-3">
                <OrchestrationAnimation />
              </div>
            </div>
          )}
          {feature.visual === "presets" && (
            <div className="mb-5">
              <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Available Presets</h4>
              <div className="rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-3">
                <PresetStripVisual />
              </div>
            </div>
          )}
          {feature.visual === "memory" && (
            <div className="mb-5">
              <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Memory Flow</h4>
              <div className="rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-3">
                <MemoryFlowVisual />
              </div>
            </div>
          )}

          {/* Features list */}
          <div className="mb-4">
            <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Key Capabilities</h4>
            <div className="space-y-1.5">
              {feature.features.map((f, i) => (
                <div key={i} className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: feature.accent }} />
                  {f}
                </div>
              ))}
            </div>
          </div>

          {/* Code example if exists */}
          {feature.code && (
            <div className="px-4 py-3 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 font-mono text-xs overflow-x-auto">
              <pre className="leading-relaxed text-gray-700 dark:text-gray-300">
                <code
                  className="syntax-code"
                  dangerouslySetInnerHTML={{
                    __html: highlightCode(
                      feature.code,
                      /^\s*(pip|effgen|export|cd|sudo|brew|apt|docker|curl|sh|bash|#!)/m.test(feature.code) ? "bash" : "python"
                    ),
                  }}
                />
              </pre>
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ── Main Features Section ── */

const FEATURES_PREVIEW_COUNT = 8;

export default function Features() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [expandedFeature, setExpandedFeature] = useState<FeatureItem | null>(null);
  const [showAll, setShowAll] = useState(false);
  const visibleFeatures = showAll ? features : features.slice(0, FEATURES_PREVIEW_COUNT);
  const hiddenCount = features.length - FEATURES_PREVIEW_COUNT;

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.08 } },
  };

  return (
    <>
      <section id="features" className="py-24 bg-gray-50 dark:bg-[#020c08] relative overflow-hidden noise-overlay" ref={ref}>
        {/* Background elements */}
        <div className="absolute inset-0 grid-pattern" />
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/30 to-transparent" />
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

        <motion.div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full opacity-[0.03] pointer-events-none"
          style={{ background: "radial-gradient(circle, #00ff88 0%, transparent 70%)" }}
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ duration: 8, repeat: Infinity }}
        />

        <Container className="relative z-10">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <motion.span
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-600 dark:text-green-400 text-sm font-semibold mb-6"
              whileHover={{ borderColor: "rgba(0,255,136,0.6)" }}
            >
              <FiZap size={14} />
              Features
            </motion.span>
            <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
              Everything You Need to Build
              <br />
              <span className="gradient-text">Production-Ready AI Agents</span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Optimized for Small Language Models with production-grade features
            </p>
          </motion.div>

          {/* Bento Grid */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate={inView ? "visible" : "hidden"}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5"
          >
            {visibleFeatures.map((feature, index) => (
              <FeatureCard
                key={feature.title}
                feature={feature}
                index={index}
                onExpand={() => setExpandedFeature(feature)}
              />
            ))}
          </motion.div>

          {/* Show more / less + CTA */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.8 }}
            className="flex flex-col items-center gap-4 mt-12"
          >
            {hiddenCount > 0 && (
              <motion.button
                onClick={() => setShowAll((v) => !v)}
                aria-expanded={showAll}
                className="group inline-flex items-center gap-2 px-6 py-3 rounded-full font-semibold text-sm text-green-600 dark:text-green-300 border border-green-500/30 bg-green-500/5 hover:border-green-400/60 hover:bg-green-500/10 transition-all"
                whileHover={{ y: -2 }}
                whileTap={{ scale: 0.97 }}
              >
                <motion.span
                  animate={{ rotate: showAll ? 180 : 0 }}
                  transition={{ duration: 0.3 }}
                  className="inline-flex"
                >
                  <FiZap size={14} />
                </motion.span>
                {showAll ? "Show fewer features" : `Show ${hiddenCount} more features`}
              </motion.button>
            )}

            <motion.a
              href="#quickstart"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
              style={{
                background: "linear-gradient(135deg, #00ff88, #00c96e)",
                boxShadow: "0 0 30px rgba(0,255,136,0.3)",
              }}
              whileHover={{ scale: 1.05, y: -2, boxShadow: "0 0 50px rgba(0,255,136,0.5)" }}
              whileTap={{ scale: 0.95 }}
            >
              Get Started
              <FiZap />
            </motion.a>
          </motion.div>
        </Container>
      </section>

      {/* Expanded Detail Modal */}
      <AnimatePresence>
        {expandedFeature && (
          <ExpandedDetail
            feature={expandedFeature}
            onClose={() => setExpandedFeature(null)}
          />
        )}
      </AnimatePresence>
    </>
  );
}
