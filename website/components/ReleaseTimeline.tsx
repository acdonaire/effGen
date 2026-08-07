"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import {
  FiActivity,
  FiArrowRight,
  FiBookOpen,
  FiCpu,
  FiFileText,
  FiGitBranch,
  FiImage,
  FiLock,
  FiBarChart2,
  FiServer,
  FiShield,
  FiTool,
  FiZap,
} from "react-icons/fi";
import Container from "./Container";
import { withBasePath } from "./basePath";

const releases = [
  {
    version: "0.2.5",
    date: "May 18, 2026",
    title: "13 New Free Tools · 44+ Total",
    accent: "#ffd700",
    icon: FiTool,
    summary:
      "Adds 13 new free / no-auth tools spanning academic research, news & RSS, YouTube, social media, translation, language detection, and QR codes — bringing the built-in tool count to 44+.",
    bullets: [
      "Academic: PubMedTool, ArXivTool, SemanticScholarTool (NCBI / arXiv / Semantic Scholar APIs with built-in rate limiting)",
      "News & social: RSSFeedTool, NewsTool, RedditTool, HackerNewsTool — no auth required for reads",
      "YouTube: YouTubeTranscriptTool (no Google API key) and YouTubeMetadataTool via yt-dlp",
      "Translate + LanguageDetect (LibreTranslate / argostranslate offline fallback, langdetect for 55+ languages), QRGenerateTool & QRReadTool — fully local",
      "All wired into research and general presets with new gallery.md and per-tool docs",
    ],
  },
  {
    version: "0.2.6",
    date: "May 19, 2026",
    title: "Docs · Media · Comms · 58+ Tools",
    accent: "#00ff88",
    icon: FiFileText,
    summary:
      "Adds 14 new built-in tools across OCR, audio transcription, image analysis, document parsing (PDF/DOCX/Excel), geo/weather, and email/webhook communication — raising the built-in tool count to 58+. Two new presets (media, notify) join the existing roster.",
    bullets: [
      "OCR + audio: OCRTool (Tesseract + OCR.space fallback), AudioTranscribeTool (faster-whisper + HF Inference fallback)",
      "Documents: PDFTool (pypdf + pdfplumber), DOCXTool (python-docx), ExcelTool (openpyxl + pandas) — added to research + general presets",
      "Media + geo: ImageInfoTool (Pillow), ImageCaptionTool (vision-router), WeatherTool (Open-Meteo), GeocodeTool (Nominatim), MapsTool (OSM static)",
      "Comms: EmailSMTPTool (TLS-on by default), EmailIMAPTool, SlackWebhookTool, DiscordWebhookTool — webhook URLs redacted in logs",
      "New presets: media (audio + image caption) and notify (email + Slack + Discord); 5 new error classes for missing creds / vision provider / system deps",
    ],
  },
  {
    version: "0.2.7",
    date: "May 20, 2026",
    title: "Prompt Library · 31 Templates · CLI + Playground",
    accent: "#22d3ee",
    icon: FiBookOpen,
    summary:
      "Adds the Prompt Library — 31 curated, domain-organized prompt templates across research, coding, data/SQL, legal, medical, creative, and business — with a golden + live eval harness, a rich CLI, and an interactive playground.",
    bullets: [
      "31 templates across 7 domains: research (5), coding (5), data/SQL (5), legal (3), medical (3), creative (5), business (5)",
      "PromptRegistry auto-discovers domain packages; variants: zero_shot / cot / few_shot / tool / structured",
      "PromptEval harness: eval_golden compares against stored .txt golden; eval_live runs through a model and validates expected_shape (sqlglot.parse / ast.parse / regex / JSON-schema)",
      "effgen prompts list / show / eval / render / run + interactive effgen prompts playground REPL",
      "Mandatory legal + medical non-advice disclaimers rendered verbatim in every template's system prompt, enforced by unit tests",
    ],
  },
  {
    version: "0.2.8",
    date: "May 21, 2026",
    title: "Multimodal Input",
    accent: "#f472b6",
    icon: FiImage,
    summary:
      "Makes image, audio, and video first-class input types across 6 cloud providers plus local MLX-VLM, with a unified ContentPart Message schema, per-provider preprocessing, and capability gating.",
    bullets: [
      "Typed ContentPart union: TextPart, ImagePart, AudioPart, VideoPart, ToolCallPart, ToolResultPart — Message(role, 'text') still works",
      "image_from / audio_from / video_from helpers (bytes, path, URL, PIL.Image, np.ndarray); ffmpeg keyframe sampling for video",
      "Image / audio / video across Gemini, OpenAI, Groq, Anthropic, Together, HF + local MLX-VLM on Apple Silicon",
      "CapabilityNotSupportedError gating — no silent downcast; new multimodal preset + MultimodalDescribeTool",
      "5 cookbook walkthroughs: image Q&A, audio transcribe + reason, video summarize, OCR + LLM, chart reading",
    ],
  },
  {
    version: "0.2.9",
    date: "May 23, 2026",
    title: "Observability & Reliability",
    accent: "#00e5ff",
    icon: FiBarChart2,
    summary:
      "Turns effGen into something you can operate in production: structured logging with secret redaction, Prometheus metrics, SLO tracking, OTel tracing, retries, circuit breakers, bulkheads, chaos + fuzz harnesses, and load testing.",
    bullets: [
      "Structured JSON logging with encoder-level secret Redactor (every path covered)",
      "Prometheus histograms + token counters (GET /metrics); SLO + SLOTracker burn-rate (GET /slo)",
      "OTel tracing with explicit samplers (no implicit head=1.0) and a canonical span-attribute spec",
      "Reliability: explicit timeouts (no timeout=None), jittered retries, per-provider circuit breakers, bulkheads",
      "Deterministic Chaos(seed) harness (6 fault types, 4 scenarios), Hypothesis fuzz suite, effgen loadtest CLI, 6 Alertmanager rules",
    ],
  },
  {
    version: "0.2.10",
    date: "May 27, 2026",
    title: "Security, Edge & Developer Experience",
    accent: "#a78bfa",
    icon: FiLock,
    summary:
      "Hardens effGen end-to-end with a sandboxed CodeExecutor, OIDC auth + RBAC + audit log, supply-chain scanning, four production deploy targets, and three developer-experience surfaces.",
    bullets: [
      "Sandboxed CodeExecutor: DockerSandbox by default (--read-only, --network=none, --cap-drop=ALL), unprivileged-namespace subprocess fallback",
      "API server OIDC/JWT auth, RBAC with daily cost caps (403 / 429), per-request redacted audit log",
      "gitleaks pre-commit + CI, CycloneDX SBOM, pip-audit, EFFGEN_VERIFY_HASHES hash verification",
      "Deploy: multi-stage Dockerfile, Helm chart (HPA/PDB/NetworkPolicy), AWS Lambda (Mangum + SAM), Cloudflare Worker edge proxy",
      "DX: VSCode extension, Jupyter magics (%effgen_chat / %%effgen_agent / %effgen_metrics), live local dashboard at /dashboard",
    ],
  },
  {
    version: "0.3.0",
    date: "June 19, 2026",
    title: "Stabilization & Hardening",
    accent: "#00ff88",
    icon: FiShield,
    summary:
      "A major stabilization release: no new providers, tools, or presets — instead everything already in effGen becomes robust, predictable, fast, secure, and pleasant to use. Failures are now loud and typed, the model catalog updates itself, local GPUs work out of the box, the server fails closed, the built-in tools are sandboxed, and import effgen is effectively instant. No breaking API changes.",
    bullets: [
      "Fail-closed: Agent.run() never returns success=True with empty output; a typed error taxonomy (auth / not_found / rate_limited / transient / timeout / fatal) drives smart retries and nearest-model suggestions",
      "Self-updating catalog: every provider ships a priced, dated snapshot; effgen models refresh diffs the live API and check_drift() warns once when stale",
      "Real GPU support: driver-compatible torch install, temperature=0 decodes greedily, NVML-aware allocator that no longer deadlocks",
      "Security: server rejects forged JWTs (fails closed), one shared SSRF guard on every URL tool, PythonREPL sandbox with out-of-process timeout, path-confined file tools, eval()/pickle paths removed",
      "Performance & DX: import effgen ~7.5s → ~20ms (lazy), faster incremental streaming, agent loop stops early (6 calls/66s → 1), quiet --json CLI, live thinking UX, pip-audit clean",
    ],
  },
  {
    version: "0.3.1",
    date: "June 29, 2026",
    title: "Real-World Usability & Polish",
    accent: "#00e5ff",
    icon: FiZap,
    summary:
      "A real-world usability & polish release driven by living with the framework as eleven professionals do. No new providers or subsystems — instead grounded results carry their sources, reasoning models finish token-heavy work, custom personas are honored on every path, multi-agent teams fail honestly, the OpenAI-compatible server stops silently downgrading, and a knowledge domain becomes a runnable agent in one call. No breaking API changes.",
    bullets: [
      "Traceable evidence: response.sources / .citations are populated from the URLs a run actually retrieved (plus provider-native grounding) — never from the model's prose",
      "Reasoning models (gpt-5 family, o-series) finish token-heavy tasks instead of empty, billed results; cost, tokens, and latency land on every result; readable sub-cent costs",
      "One-call domain agents — LegalDomain().to_agent('gpt-5-nano') wires prompt + tools + guardrails; custom personas honored on every path; a pre-built VectorMemoryStore as a RAG knowledge_base",
      "Honest orchestration & server: teams/DAGs fail closed and route by name; no silent client-tool drop (clear 400) or TF-IDF embedding fallback; physical GPU memory in models status",
      "Automation & security: no MCP deadlock on sync run(), plugin auto-discovery, effgen run --json to stdout, grammar-constrained local output; the REPL sandbox toggle out of the model's hands; exhaustive secret strip",
    ],
  },
];

const stats = [
  { label: "Built-in tools", value: "66", icon: FiTool, accent: "#ffd700" },
  { label: "Prompt templates", value: "35", icon: FiBookOpen, accent: "#22d3ee" },
  { label: "Agent presets", value: "9", icon: FiShield, accent: "#a78bfa" },
  { label: "Cloud providers", value: "9", icon: FiCpu, accent: "#00ff88" },
  { label: "Total backends", value: "14", icon: FiActivity, accent: "#00e5ff" },
];

export default function ReleaseTimeline() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <section className="py-20 bg-white dark:bg-[#020c08] relative overflow-hidden" ref={ref}>
      <div className="absolute inset-0 grid-pattern opacity-40" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

      <Container className="relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.55 }}
          className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-8 mb-12"
        >
          <div className="max-w-3xl">
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-600 dark:text-green-400 text-sm font-semibold mb-5">
              <FiZap size={14} />
              Release train
            </span>
            <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white leading-tight">
              Updated from v0.2.0 to <span className="gradient-text">v0.3.1</span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed">
              The site now reflects every post-0.2.0 release: the policy-based ModelRouter with
              transparent failover and <code>effgen cost</code> CLI in 0.2.4,{" "}
              <strong>27 new tools</strong> in 0.2.5 + 0.2.6 (bringing the built-in count to{" "}
              <strong>58+</strong>, plus the <code>media</code> and <code>notify</code> presets),
              the <strong>Prompt Library</strong> in 0.2.7, <strong>multimodal</strong> image /
              audio / video input in 0.2.8, a full <strong>observability &amp; reliability</strong>{" "}
              stack in 0.2.9, <strong>security, edge &amp; developer-experience</strong> features
              in 0.2.10, the <strong>v0.3.0 stabilization &amp; hardening</strong> release, and —
              most recently — the <strong>v0.3.1 real-world usability &amp; polish</strong> release:
              grounded <code>response.sources</code> / <code>.citations</code>, reasoning models that
              finish token-heavy work, custom personas honored everywhere, one-call domain agents,
              honest multi-agent teams and an honest OpenAI-compatible server, and{" "}
              <code>effgen run --json</code>.
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 lg:w-[600px]">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 18 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: 0.1 + index * 0.08 }}
                className="rounded-2xl bg-gray-50 dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-4 text-center"
              >
                <stat.icon className="mx-auto mb-2" size={18} style={{ color: stat.accent }} />
                <div className="text-3xl font-black" style={{ color: stat.accent }}>
                  {stat.value}
                </div>
                <div className="text-[10px] uppercase tracking-wide text-gray-500 font-semibold">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-5">{/* 5 most recent releases */}
          {releases.slice(-5).map((release, index) => (
            <motion.article
              key={release.version}
              initial={{ opacity: 0, y: 28 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: index * 0.12 }}
              className="group relative rounded-2xl bg-white dark:bg-gray-900/65 border border-gray-200 dark:border-gray-800 p-6 overflow-hidden shadow-sm dark:shadow-none"
            >
              <div
                className="absolute left-0 top-0 bottom-0 w-1"
                style={{ background: release.accent }}
              />
              <div className="flex items-start justify-between gap-4 mb-5">
                <div>
                  <div className="text-xs font-bold uppercase tracking-widest text-gray-500">
                    v{release.version} · {release.date}
                  </div>
                  <h3 className="text-xl font-black text-gray-900 dark:text-white mt-2">
                    {release.title}
                  </h3>
                </div>
                <div
                  className="w-11 h-11 rounded-xl flex items-center justify-center border flex-shrink-0"
                  style={{ borderColor: `${release.accent}45`, background: `${release.accent}18` }}
                >
                  <release.icon size={20} style={{ color: release.accent }} />
                </div>
              </div>

              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-5">
                {release.summary}
              </p>

              <ul className="space-y-2">
                {release.bullets.map((bullet) => (
                  <li key={bullet} className="flex gap-2 text-xs text-gray-600 dark:text-gray-500 leading-relaxed">
                    <span className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0" style={{ background: release.accent }} />
                    <span>{bullet}</span>
                  </li>
                ))}
              </ul>
            </motion.article>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.45 }}
          className="mt-10 flex flex-wrap gap-3 justify-center"
        >
          <a
            href={withBasePath("/docs/releases")}
            className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-bold text-black"
            style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)" }}
          >
            Read Release Notes
            <FiArrowRight size={16} />
          </a>
          <a
            href={withBasePath("/docs/providers")}
            className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-bold text-green-600 dark:text-green-300 border border-green-500/30 bg-green-500/5 hover:bg-green-500/10"
          >
            Provider Matrix
          </a>
        </motion.div>
      </Container>
    </section>
  );
}
