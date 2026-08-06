"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useState } from "react";
import { FiX, FiCopy, FiCheck, FiZap } from "react-icons/fi";
import Container from "./Container";

interface PresetData {
  name: string;
  description: string;
  tools: string[];
  code: string;
  accent: string;
  emoji: string;
  fullDescription: string;
  useCases: string[];
  examplePrompts: string[];
  temperature: number;
  maxIterations: number;
  promptDesc: string;
  workflowIcons: string[];
}

const presets: PresetData[] = [
  {
    name: "math",
    description: "Mathematical computations",
    tools: ["Calculator", "PythonREPL"],
    code: 'create_agent("math", model)',
    accent: "#00ff88",
    emoji: "\u{1F9EE}",
    fullDescription: "Optimized for mathematical reasoning with Calculator and PythonREPL. Uses low temperature (0.3) for precise answers.",
    useCases: ["Arithmetic and algebra", "Statistical analysis", "Unit conversions", "Financial calculations"],
    examplePrompts: ['agent.run("What is 24344 * 334?")', 'agent.run("Calculate the standard deviation of [12, 45, 23, 67, 89]")', 'agent.run("Convert 72\u00B0F to Celsius")'],
    temperature: 0.3,
    maxIterations: 8,
    promptDesc: "Precise mathematical reasoning agent",
    workflowIcons: ["\u{1F522}", "\u{1F9EE}", "\u{1F4CA}", "\u2705"],
  },
  {
    name: "research",
    description: "Web \u00b7 academic \u00b7 news \u00b7 social \u00b7 video \u00b7 docs",
    tools: ["WebSearch", "URLFetch", "Wikipedia", "ArXiv", "PubMed", "SemanticScholar", "RSSFeed", "News", "YouTubeTranscript", "YouTubeMetadata", "Reddit", "HackerNews", "PDF", "DOCX", "Excel"],
    code: 'create_agent("research", model)',
    accent: "#00e5ff",
    emoji: "\u{1F52C}",
    fullDescription: "Research preset (15 tools): WebSearch, URLFetch, Wikipedia, academic search (ArXiv, PubMed, Semantic Scholar), RSS feeds, news headlines, YouTube transcript/metadata, Reddit, Hacker News, and document parsing (PDF, DOCX, Excel \u2014 added in v0.2.6). Prefers academic sources for scientific / medical questions, news + RSS + Reddit + HN for current events, YouTube transcripts for video content, and PDF/DOCX/Excel for local document research. Cites sources and synthesises findings.",
    useCases: ["Academic research (arXiv / PubMed / Semantic Scholar)", "Current events (News + RSS + HN + Reddit)", "Video research (YouTube transcript + metadata)", "Local document research (PDF / DOCX / Excel)", "Fact-checking across multiple sources"],
    examplePrompts: ['agent.run("Find recent arXiv papers on small language models for agents")', 'agent.run("Summarize today\'s top Hacker News stories about AI")', 'agent.run("Extract the tables from /tmp/paper.pdf and summarize")'],
    temperature: 0.5,
    maxIterations: 10,
    promptDesc: "Thorough research agent (web + academic + news + social + video + docs)",
    workflowIcons: ["\u{1F50D}", "\u{1F4D8}", "\u{1F4F0}", "\u2728"],
  },
  {
    name: "coding",
    description: "Code execution & development",
    tools: ["CodeExecutor", "PythonREPL", "FileOps", "Bash"],
    code: 'create_agent("coding", model)',
    accent: "#a78bfa",
    emoji: "\u{1F4BB}",
    fullDescription: "Full development toolkit with sandboxed code execution, file operations, and shell access. Tests code before presenting results.",
    useCases: ["Write and debug code", "File manipulation", "System automation", "Script generation"],
    examplePrompts: ['agent.run("Write a Python function to find palindromic substrings")', 'agent.run("Read data.csv and create a summary")', 'agent.run("List all Python files in the current directory")'],
    temperature: 0.4,
    maxIterations: 12,
    promptDesc: "Expert coding agent",
    workflowIcons: ["\u{1F4AD}", "\u2328\uFE0F", "\u{1F9EA}", "\u{1F680}"],
  },
  {
    name: "general",
    description: "All 31 general-purpose tools",
    tools: ["Calculator", "PythonREPL", "WebSearch", "CodeExecutor", "FileOperations", "JSONTool", "DateTimeTool", "TextProcessing", "URLFetch", "Wikipedia", "RSSFeed", "News", "Reddit", "HackerNews", "Translate", "LanguageDetect", "QRGenerate", "QRRead", "OCR", "AudioTranscribe", "ImageInfo", "PDF", "DOCX", "Excel", "Weather", "Geocode", "Maps", "EmailSMTP", "EmailIMAP", "SlackWebhook", "DiscordWebhook"],
    code: 'create_agent("general", model)',
    accent: "#ffd700",
    emoji: "\u{1F680}",
    fullDescription: "The Swiss-Army-knife preset \u2014 31 general-purpose tools spanning OCR, audio transcription, image info, document parsing (PDF, DOCX, Excel), geo/weather (Open-Meteo + Nominatim + OSM static maps), and email/webhook communication (SMTP, IMAP, Slack, Discord) on top of the base set (RSS, News, Reddit, HN, Translate, LanguageDetect, QR). As of v0.3.1 the unsandboxed bash tool is no longer bundled here \u2014 it lives only in the coding preset, where you opt into a dev environment. Automatically picks the best tool for each task.",
    useCases: ["Multi-step workflows that span tools", "OCR a scan, summarise it, email the result", "Read a PDF/DOCX/Excel and post a summary to Slack", "Fetch the weather, render a map, and DM a Discord channel"],
    examplePrompts: ['agent.run("OCR /tmp/scan.png, summarize the text, and email it to alice@example.com")', 'agent.run("What\'s the weather at 1600 Amphitheatre Pkwy? Send a Slack alert if it\'s raining")', 'agent.run("Read tables from quarterly.xlsx and post the top 5 rows to #finance")'],
    temperature: 0.7,
    maxIterations: 10,
    promptDesc: "Versatile AI assistant (31 tools)",
    workflowIcons: ["\u{1F4DD}", "\u{1F527}", "\u{1F4CA}", "\u2705"],
  },
  {
    name: "rag",
    description: "Retrieval-augmented Q&A over your docs",
    tools: ["Retrieval (hybrid dense + BM25)"],
    code: 'create_agent("rag", model,\n  knowledge_base="./docs/")',
    accent: "#00c896",
    emoji: "\u{1F4D6}",
    fullDescription: "Ingests a knowledge base on creation via DocumentIngester (txt/md/json/jsonl/csv/html out of the box, plus optional pdf/docx/epub). Retrieves with hybrid dense + BM25 search and returns answers with inline [N] citations backed by AgentResponse.citations and .sources. Says 'I don\'t know' when the KB has no match, rather than hallucinating.",
    useCases: ["Docs Q&A with citations", "Policy / compliance lookups", "Codebase-aware assistants", "Knowledge-grounded chat"],
    examplePrompts: ['agent.run("What does the config file support?")', 'agent.run("Find the section about rate limiting")', 'agent.run("Summarize the v0.2.3 provider updates")'],
    temperature: 0.3,
    maxIterations: 8,
    promptDesc: "Retrieval-augmented assistant with citations",
    workflowIcons: ["\u{1F50D}", "\u{1F4D6}", "\u270F\uFE0F", "\u2705"],
  },
  {
    name: "media",
    description: "Audio transcription + vision captioning",
    tools: ["AudioTranscribe", "ImageCaption"],
    code: 'create_agent("media", model)',
    accent: "#f59e0b",
    emoji: "\u{1F39E}\uFE0F",
    fullDescription: "v0.2.6 media preset: bundles AudioTranscribeTool (speech-to-text via faster-whisper with HuggingFace Inference fallback) and ImageCaptionTool (vision captioning via OpenAI / Gemini / MLX-VLM through the model router). Handles audio files (MP3, WAV, OGG, FLAC) and images (PNG, JPEG, WEBP). For long recordings, summarizes key points after transcribing.",
    useCases: ["Transcribe podcasts and meetings", "Caption / describe images", "Summarize long audio recordings", "Multi-modal content pipelines"],
    examplePrompts: ['agent.run("Transcribe /tmp/meeting.mp3 and list the action items")', 'agent.run("Describe what is in /tmp/photo.jpg")', 'agent.run("Caption these three product images for an e-commerce listing")'],
    temperature: 0.3,
    maxIterations: 8,
    promptDesc: "Media processing assistant (audio + vision, v0.2.6)",
    workflowIcons: ["\u{1F3A4}", "\u{1F5BC}\uFE0F", "\u{1F4DD}", "\u2705"],
  },
  {
    name: "notify",
    description: "Email + Slack + Discord notifications",
    tools: ["EmailSMTP", "EmailIMAP", "SlackWebhook", "DiscordWebhook"],
    code: 'create_agent("notify", model)',
    accent: "#ec4899",
    emoji: "\u{1F4E2}",
    fullDescription: "v0.2.6 notify preset: sends email via SMTP (TLS-on by default), reads email via IMAP, and posts to Slack/Discord via incoming webhook URLs. Configure credentials via env vars (SMTP_HOST/SMTP_USER/SMTP_PASSWORD, IMAP_HOST/IMAP_USER/IMAP_PASSWORD, SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL). Webhook URLs are redacted in all logs. Tools that lack credentials raise MissingCredentialsError when called.",
    useCases: ["Deploy / incident alerts to Slack or Discord", "Email digests + status summaries", "Confirm delivery via IMAP after sending", "Multi-channel notifications from CI"],
    examplePrompts: ['agent.run("Email alice@example.com with the deploy summary")', 'agent.run("Post a status update to #ops on Slack")', 'agent.run("Check INBOX for any failed-job notifications")'],
    temperature: 0.3,
    maxIterations: 6,
    promptDesc: "Notification agent (email + Slack + Discord, v0.2.6)",
    workflowIcons: ["\u{1F4E7}", "\u{1F4AC}", "\u{1F47E}", "\u2705"],
  },
  {
    name: "multimodal",
    description: "Image, audio & video understanding",
    tools: ["MultimodalDescribe", "ImageCaption", "OCR", "AudioTranscribe", "PDF"],
    code: 'create_agent("multimodal", model)',
    accent: "#f472b6",
    emoji: "\u{1F5BC}\ufe0f",
    fullDescription: "v0.2.8 multimodal preset: reasons over images, audio, and video. Uses Gemini Flash (primary) with OpenAI gpt-4o-mini and HuggingFace fallbacks, and wires MultimodalDescribeTool (auto-dispatch for any media type) alongside ImageInfo, ImageCaption, OCR, AudioTranscribe, PDF, and Weather. Built on the unified ContentPart Message schema with per-provider preprocessing and capability gating \u2014 no silent downcast when a model lacks vision/audio/video.",
    useCases: ["Visual question answering over an image", "Transcribe audio then reason over it", "Summarize a video from sampled keyframes", "OCR a document then extract structured fields"],
    examplePrompts: ['agent.run("Describe /tmp/chart.png and read off the largest bar")', 'agent.run("Transcribe /tmp/talk.mp3 and summarize the key points")', 'agent.run("Summarize what happens in /tmp/clip.mp4")'],
    temperature: 0.3,
    maxIterations: 10,
    promptDesc: "Multimodal reasoning assistant (image + audio + video, v0.2.8)",
    workflowIcons: ["\u{1F5BC}\ufe0f", "\u{1F3A4}", "\u{1F50D}", "\u2705"],
  },
  {
    name: "minimal",
    description: "Direct inference, no tools",
    tools: ["No tools \u2014 pure LLM"],
    code: 'create_agent("minimal", model)',
    accent: "#ff9500",
    emoji: "\u26A1",
    fullDescription: "Pure language model inference with no tool overhead. Fastest response time, ideal for conversational AI and text generation.",
    useCases: ["Conversational AI", "Text generation", "Summarization", "Creative writing"],
    examplePrompts: ['agent.run("Explain quantum entanglement in simple terms")', 'agent.run("Write a haiku about programming")', 'agent.run("Summarize the key points of this article: ...")'],
    temperature: 0.7,
    maxIterations: 1,
    promptDesc: "Direct helpful assistant",
    workflowIcons: ["\u{1F4AC}", "\u2728"],
  },
];

/* ── Temperature Indicator ── */

function TempIndicator({ temp, accent }: { temp: number; accent: string }) {
  const tempColor = temp <= 0.3 ? "#00e5ff" : temp <= 0.5 ? "#00ff88" : temp <= 0.7 ? "#ffd700" : "#ff6b6b";
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[9px] text-gray-500 font-mono">temp</span>
      <div className="w-12 h-1.5 rounded-full bg-gray-200 dark:bg-gray-800 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, #00e5ff, ${tempColor})`, width: `${temp * 100}%` }}
          initial={{ width: 0 }}
          animate={{ width: `${temp * 100}%` }}
          transition={{ duration: 0.8, delay: 0.3 }}
        />
      </div>
      <span className="text-[9px] font-mono" style={{ color: tempColor }}>{temp}</span>
    </div>
  );
}

/* ── Tool Dots ── */


/* ── Animated Workflow ── */


export default function PresetShowcase() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [selectedPreset, setSelectedPreset] = useState<PresetData | null>(null);
  const [copied, setCopied] = useState(false);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.5, ease: "easeOut" as const } },
  };

  const copyCode = (code: string) => {
    const fullCode = `from effgen import load_model\nfrom effgen.presets import create_agent\n\nmodel = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")\nagent = ${code}`;
    navigator.clipboard.writeText(fullCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      <section className="py-24 bg-gray-50 dark:bg-[#030f07] relative overflow-hidden" ref={ref}>
        <div className="absolute inset-0 grid-pattern" />
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6 }}
            className="text-center mb-14"
          >
            <motion.span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-600 dark:text-green-400 text-sm font-semibold mb-6">
              <FiZap size={14} />
              Agent Presets
            </motion.span>
            <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
              One-Line Agent Creation with{" "}
              <span className="gradient-text">9 Presets</span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Ready-to-use configurations optimized for common use cases.{" "}
              <span className="text-green-500 font-semibold">v0.2.6 adds <code>media</code> and <code>notify</code>; v0.2.8 adds <code>multimodal</code>.</span>
            </p>
          </motion.div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate={inView ? "visible" : "hidden"}
            className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-3 xl:grid-cols-3 gap-3 sm:gap-4"
          >
            {presets.map((preset, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ y: -8, scale: 1.03 }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
                onClick={() => setSelectedPreset(preset)}
                className="group relative p-4 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none backdrop-blur-sm overflow-hidden cursor-pointer flex flex-col h-full"
              >
                {/* Animated border */}
                <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none overflow-hidden">
                  <motion.div
                    className="absolute inset-[-50%] w-[200%] h-[200%]"
                    style={{
                      background: `conic-gradient(from 0deg, transparent 60%, ${preset.accent}30 80%, transparent 100%)`,
                    }}
                    animate={{ rotate: 360 }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                  />
                  <div className="absolute inset-[1px] rounded-2xl bg-white dark:bg-gray-900/70" />
                </div>

                {/* Hover glow */}
                <motion.div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                  style={{ background: `radial-gradient(circle at 30% 30%, ${preset.accent}10 0%, transparent 70%)` }}
                />

                {/* TOP — emoji, name, blurb */}
                <div className="relative z-10 text-center">
                  <motion.div
                    className="w-11 h-11 rounded-xl flex items-center justify-center mb-2 text-xl overflow-hidden mx-auto"
                    style={{ background: `${preset.accent}15`, border: `1px solid ${preset.accent}30` }}
                    whileHover={{ rotate: 15, scale: 1.1 }}
                  >
                    {preset.emoji}
                  </motion.div>

                  <h3 className="text-sm font-bold mb-1 text-gray-900 dark:text-white capitalize">{preset.name}</h3>
                  <p className="text-[11px] text-gray-600 dark:text-gray-500 leading-snug line-clamp-2">{preset.description}</p>
                </div>

                {/* BOTTOM — temp/itr + code one-liner */}
                <div className="mt-auto pt-3 relative z-10">
                  <div className="flex items-center justify-between">
                    <TempIndicator temp={preset.temperature} accent={preset.accent} />
                    <span className="text-[9px] font-mono text-gray-500">
                      <span className="text-gray-400">itr:</span> {preset.maxIterations}
                    </span>
                  </div>

                  <div className="px-2 py-1.5 mt-2 rounded-md bg-gray-50 dark:bg-[#0a1a0f] border border-gray-200 dark:border-gray-800 overflow-hidden">
                    <code className="text-[10px] font-mono whitespace-nowrap block overflow-hidden text-ellipsis" style={{ color: preset.accent }}>
                      {preset.code}
                    </code>
                  </div>

                  <div className="mt-1.5 text-[9px] text-center text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity font-medium tracking-wider">
                    CLICK FOR DETAILS
                  </div>
                </div>

                {/* Bottom accent */}
                <motion.div
                  className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity z-10"
                  style={{ background: `linear-gradient(90deg, transparent, ${preset.accent}, transparent)` }}
                />
              </motion.div>
            ))}
          </motion.div>
        </Container>
      </section>

      {/* Enhanced Modal */}
      <AnimatePresence>
        {selectedPreset && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="absolute inset-0 bg-black/80 backdrop-blur-md"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={() => setSelectedPreset(null)}
            />

            <motion.div
              className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-2xl bg-white dark:bg-[#0a1a0f] border border-gray-200 dark:border-green-500/20 shadow-2xl"
              style={{ boxShadow: `0 0 60px ${selectedPreset.accent}20` }}
              initial={{ scale: 0.96, opacity: 0, y: 16 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.97, opacity: 0, y: 10 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
            >
              {/* Header bar — static base + smooth sliding sheen */}
              <div
                className="relative h-1 rounded-t-2xl overflow-hidden"
                style={{ background: `linear-gradient(90deg, transparent, ${selectedPreset.accent}80, transparent)` }}
              >
                <motion.div
                  className="absolute inset-y-0 w-1/3"
                  style={{
                    background: `linear-gradient(90deg, transparent, ${selectedPreset.accent}, transparent)`,
                  }}
                  initial={{ x: "-100%" }}
                  animate={{ x: "300%" }}
                  transition={{ duration: 2.4, repeat: Infinity, ease: "linear" }}
                />
              </div>

              <div className="p-8">
                <button
                  onClick={() => setSelectedPreset(null)}
                  className="absolute top-5 right-5 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors z-10"
                >
                  <FiX size={18} />
                </button>

                <div className="flex items-center gap-4 mb-4">
                  <div
                    className="w-14 h-14 rounded-xl flex items-center justify-center text-2xl"
                    style={{ background: `${selectedPreset.accent}15`, border: `1px solid ${selectedPreset.accent}30` }}
                  >
                    {selectedPreset.emoji}
                  </div>
                  <div>
                    <h3 className="text-2xl font-black text-gray-900 dark:text-white capitalize">{selectedPreset.name}</h3>
                    <p className="text-gray-500 text-sm">{selectedPreset.promptDesc}</p>
                  </div>
                </div>

                {/* Config summary */}
                <div className="flex flex-wrap gap-3 mb-5">
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800">
                    <span className="text-[10px] text-gray-500 uppercase font-bold">Temp</span>
                    <span className="text-sm font-mono font-bold" style={{ color: selectedPreset.accent }}>{selectedPreset.temperature}</span>
                  </div>
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800">
                    <span className="text-[10px] text-gray-500 uppercase font-bold">Max Itr</span>
                    <span className="text-sm font-mono font-bold" style={{ color: selectedPreset.accent }}>{selectedPreset.maxIterations}</span>
                  </div>
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800">
                    <span className="text-[10px] text-gray-500 uppercase font-bold">Tools</span>
                    <span className="text-sm font-mono font-bold" style={{ color: selectedPreset.accent }}>{selectedPreset.tools.length}</span>
                  </div>
                </div>

                <p className="text-gray-600 dark:text-gray-400 mb-6 text-sm leading-relaxed">{selectedPreset.fullDescription}</p>

                <div className="mb-5">
                  <h4 className="text-xs font-bold text-gray-500 dark:text-gray-500 uppercase tracking-widest mb-3">Tools Included</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedPreset.tools.map((tool, idx) => (
                      <span key={idx} className="px-3 py-1.5 rounded-full text-xs font-bold text-black" style={{ background: selectedPreset.accent }}>
                        {tool}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="mb-5">
                  <h4 className="text-xs font-bold text-gray-500 dark:text-gray-500 uppercase tracking-widest mb-3">Use Cases</h4>
                  <div className="grid grid-cols-2 gap-2">
                    {selectedPreset.useCases.map((uc, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-gray-600 dark:text-gray-400 text-sm">
                        <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: selectedPreset.accent }} />
                        {uc}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mb-5">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest">Quick Start</h4>
                    <button
                      onClick={() => copyCode(selectedPreset.code)}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 transition-colors border border-gray-200 dark:border-gray-700"
                    >
                      {copied ? <><FiCheck className="text-green-400" size={12} /> Copied</> : <><FiCopy size={12} /> Copy</>}
                    </button>
                  </div>
                  <div className="p-4 rounded-xl bg-gray-50 dark:bg-[#0a1a0f] border border-gray-300 dark:border-gray-800 overflow-x-auto relative">
                    <motion.div
                      className="absolute inset-0 rounded-xl pointer-events-none opacity-0 hover:opacity-100 transition-opacity"
                      style={{ boxShadow: `inset 0 0 30px ${selectedPreset.accent}08` }}
                    />
                    <pre className="text-xs font-mono text-gray-700 dark:text-gray-300 leading-relaxed code-preview-block">
                      <div className="flex"><span className="w-6 text-right pr-3 text-gray-400 dark:text-gray-600 select-none">1</span><span><span className="code-keyword">from </span><span className="code-module">effgen </span><span className="code-keyword">import </span><span className="code-text">load_model</span></span></div>
                      <div className="flex"><span className="w-6 text-right pr-3 text-gray-400 dark:text-gray-600 select-none">2</span><span><span className="code-keyword">from </span><span className="code-module">effgen.presets </span><span className="code-keyword">import </span><span className="code-text">create_agent</span></span></div>
                      <div className="flex"><span className="w-6 text-right pr-3 text-gray-400 dark:text-gray-600 select-none">3</span><span></span></div>
                      <div className="flex"><span className="w-6 text-right pr-3 text-gray-400 dark:text-gray-600 select-none">4</span><span><span className="code-text">model = </span><span className="code-func">load_model</span><span className="code-string">{`("Qwen/Qwen2.5-3B-Instruct"`}</span><span className="code-text">{`, quantization=`}</span><span className="code-string">{`"4bit"`}</span><span className="code-text">)</span></span></div>
                      <div className="flex bg-green-500/5 -mx-2 px-2 rounded border-l-2 border-green-500/30"><span className="w-6 text-right pr-3 text-gray-400 dark:text-gray-600 select-none">5</span><span><span className="code-text">agent = {selectedPreset.code}</span></span></div>
                    </pre>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-bold text-gray-500 dark:text-gray-500 uppercase tracking-widest mb-3">Example Prompts</h4>
                  <div className="space-y-2">
                    {selectedPreset.examplePrompts.map((prompt, idx) => (
                      <motion.div
                        key={idx}
                        className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 hover:border-green-500/30 transition-colors"
                        whileHover={{ x: 4, boxShadow: `0 0 15px ${selectedPreset.accent}08` }}
                      >
                        <code className="text-xs font-mono text-gray-600 dark:text-gray-400">{prompt}</code>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
