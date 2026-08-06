"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiCopy, FiCheck, FiZap, FiTerminal, FiChevronDown } from "react-icons/fi";
import { useState } from "react";
import Container from "./Container";
import { highlightCode } from "./syntaxHighlight";

export default function QuickStart() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const steps = [
    {
      number: "01",
      title: "Install",
      description: "One command for the core; add an extra only when you need that provider.",
      code: `pip install -U effgen

# Optional extras
pip install "effgen[vllm]"       # local CUDA throughput
pip install "effgen[cerebras]"   # Cerebras
pip install "effgen[groq]"       # Groq`,
      language: "bash",
      accent: "#00ff88",
    },
    {
      number: "02",
      title: "Set Keys & Check",
      description: "Export the keys you have, then run the doctor to confirm what's wired up.",
      code: `export OPENAI_API_KEY="..."
export GROQ_API_KEY="..."
export CEREBRAS_API_KEY="..."

effgen doctor               # table of providers + key status
effgen doctor --json        # machine-readable`,
      language: "bash",
      accent: "#00e5ff",
    },
    {
      number: "03",
      title: "Create an Agent",
      description: "Same Agent API across local SLMs and any registered cloud provider.",
      code: `from effgen import load_model
from effgen.presets import create_agent

# Cloud
model = load_model("gpt-5.4-nano", provider="openai")
# model = load_model("groq:llama-3.3-70b-versatile")

# Or local
# model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

agent = create_agent("general", model)
print(agent.run("What is (17 * 23) + sqrt(144)?").output)  # 403`,
      language: "python",
      accent: "#a78bfa",
    },
  ];

  const advancedSteps = [
    {
      number: "04",
      title: "Route With a Budget",
      description: "Let the router pick the cheapest provider that fits your budget and fail over on errors.",
      code: `import effgen.models  # register all cloud adapters
from effgen import (
    PolicyBasedRouter, RoutingContext, CostBasedPolicy, load_model,
)
from effgen.models.capabilities import Capability

router = PolicyBasedRouter(policies=[CostBasedPolicy()], failover_hops=3)

def call(pair):
    return load_model(f"{pair.provider}:{pair.model_id}").generate("Hello")

answer = router.route_and_execute(
    RoutingContext(
        user_budget_usd=0.01,
        required_capabilities={Capability.chat},
    ),
    call,
)`,
      language: "python",
      accent: "#ff6b6b",
    },
    {
      number: "05",
      title: "Thinking, Grounding & Caching",
      description: "Gemini thinking + grounding + Files and Anthropic prompt caching are GenerationConfig additions.",
      code: `from effgen import load_model
from effgen.models import GenerationConfig, mark_cached, upload_file

doc = upload_file("brief.txt")
gemini = load_model("gemini-2.5-pro", provider="gemini")
gemini.generate(
    "Use the file and search grounding.",
    config=GenerationConfig(thinking_budget=4096, include_thoughts=True, grounding=True),
    files=[doc],
)

claude = load_model("claude-sonnet-4-6", provider="anthropic")
claude.generate(
    "Reason about migration risks.",
    system_prompt=[mark_cached("Reusable system context", ttl="1h")],
    config=GenerationConfig(thinking={"type": "enabled", "budget_tokens": 4096}),
)`,
      language: "python",
      accent: "#ffd700",
    },
    {
      number: "06",
      title: "Track Spend & Set a Budget",
      description: "Every paid call is persisted. The CLI prints daily/weekly summaries and sets a daily cap.",
      code: `effgen cost today           # last 24h, per provider/model
effgen cost week            # rolling 7 days
effgen cost by-provider     # lifetime totals
effgen cost set-budget 1.0  # $1/day cap (warn at 80%, fail over at 100%)
effgen cost clear-budget`,
      language: "bash",
      accent: "#00e5ff",
    },
  ];

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <section id="quickstart" className="py-24 bg-gray-50 dark:bg-[#020c08] relative overflow-hidden" ref={ref}>
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

      {/* Background glow */}
      <motion.div
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-[0.04] pointer-events-none"
        style={{ background: "radial-gradient(circle, #00ff88 0%, transparent 70%)" }}
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ duration: 8, repeat: Infinity }}
      />

      <Container className="relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <motion.span
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-600 dark:text-green-400 text-sm font-semibold mb-6"
          >
            <FiTerminal size={14} />
            Quick Start
          </motion.span>
          <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
            Up and Running in{" "}
            <span className="gradient-text neon-text">60 Seconds</span>
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Three practical steps to your first local or cloud-backed AI agent. Expand the
            advanced section for budget-aware routing, thinking/grounding/caching, and persistent
            cost tracking.
          </p>
        </motion.div>

        {/* Progress bar */}
        <div className="max-w-4xl mx-auto mb-8 hidden lg:block">
          <div className="relative h-1 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className="absolute left-0 top-0 h-full rounded-full"
              style={{ background: "linear-gradient(90deg, #00ff88, #00e5ff, #a78bfa)" }}
              initial={{ width: "0%" }}
              animate={inView ? { width: "100%" } : {}}
              transition={{ duration: 2, delay: 0.5, ease: "easeOut" }}
            />
          </div>
          <div className="flex justify-between mt-2">
            {steps.map((step, i) => (
              <motion.div
                key={i}
                className="text-xs font-mono font-bold"
                style={{ color: step.accent }}
                initial={{ opacity: 0 }}
                animate={inView ? { opacity: 1 } : {}}
                transition={{ delay: 0.5 + i * 0.7 }}
              >
                Step {step.number}
              </motion.div>
            ))}
          </div>
        </div>

        <div className="max-w-4xl mx-auto space-y-6">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: index % 2 === 0 ? -40 : 40 }}
              animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ duration: 0.6, delay: index * 0.15 }}
              className="relative"
            >
              <motion.div
                className="group flex flex-col lg:flex-row items-start gap-6 p-6 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 backdrop-blur-sm relative overflow-hidden shadow-sm dark:shadow-none"
                whileHover={{ borderColor: step.accent + "50", boxShadow: `0 0 40px ${step.accent}10`, y: -4 }}
                transition={{ duration: 0.3 }}
              >
                {/* Hover glow bg */}
                <motion.div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                  style={{ background: `radial-gradient(circle at 10% 50%, ${step.accent}05 0%, transparent 60%)` }}
                />

                {/* Step number */}
                <motion.div
                  className="flex-shrink-0 relative z-10"
                  whileHover={{ scale: 1.15, boxShadow: `0 0 20px ${step.accent}30` }}
                >
                  <div
                    className="relative w-16 h-16 rounded-xl flex items-center justify-center overflow-hidden"
                    style={{
                      background: `${step.accent}15`,
                      border: `1px solid ${step.accent}30`,
                    }}
                  >
                    <motion.div
                      className="absolute inset-0"
                      animate={{ opacity: [0.3, 0.6, 0.3] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      style={{ background: `radial-gradient(circle, ${step.accent}20 0%, transparent 70%)` }}
                    />
                    <span
                      className="text-2xl font-black relative z-10 font-mono"
                      style={{ color: step.accent }}
                    >
                      {step.number}
                    </span>
                  </div>
                </motion.div>

                {/* Content */}
                <div className="flex-1 relative z-10">
                  <h3 className="text-xl font-bold mb-1.5 text-gray-900 dark:text-white flex items-center gap-2">
                    {step.title}
                    <motion.div
                      animate={{ rotate: [0, 360] }}
                      transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                    >
                      <FiZap style={{ color: step.accent }} size={16} />
                    </motion.div>
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-500 mb-4">{step.description}</p>

                  {/* Code block with enhanced hover glow */}
                  <div className="relative rounded-xl bg-gray-50 dark:bg-[#0a1a0f] border border-gray-300 dark:border-gray-800 overflow-hidden group/code">
                    {/* Top line */}
                    <div
                      className="absolute top-0 left-0 right-0 h-px"
                      style={{ background: `linear-gradient(90deg, transparent, ${step.accent}60, transparent)` }}
                    />

                    {/* Enhanced glow on hover */}
                    <motion.div
                      className="absolute inset-0 opacity-0 group-hover/code:opacity-100 transition-opacity duration-500 pointer-events-none rounded-xl"
                      style={{ boxShadow: `inset 0 0 40px ${step.accent}06, 0 0 30px ${step.accent}06` }}
                    />

                    {/* Shimmer animation */}
                    <motion.div
                      className="absolute inset-0 pointer-events-none"
                      style={{
                        background: `linear-gradient(90deg, transparent, ${step.accent}05, transparent)`,
                        backgroundSize: "200% 100%",
                      }}
                      animate={{ backgroundPosition: ["200% 0", "-200% 0"] }}
                      transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    />

                    <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-200 dark:border-gray-700/50">
                      <span
                        className="text-xs font-mono font-semibold px-2 py-0.5 rounded"
                        style={{
                          color: step.accent,
                          background: `${step.accent}15`,
                          border: `1px solid ${step.accent}30`,
                        }}
                      >
                        {step.language}
                      </span>
                      <motion.button
                        onClick={() => copyToClipboard(step.code, index)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-all text-xs font-medium border border-gray-300 dark:border-gray-700"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        {copiedIndex === index ? (
                          <><FiCheck className="text-green-400" size={12} /> Copied</>
                        ) : (
                          <><FiCopy size={12} /> Copy</>
                        )}
                      </motion.button>
                    </div>
                    <pre className="p-4 text-sm font-mono text-gray-800 dark:text-gray-200 overflow-x-auto leading-relaxed relative">
                      <code
                        className="syntax-code"
                        dangerouslySetInnerHTML={{ __html: highlightCode(step.code, step.language) }}
                      />
                    </pre>

                    {/* Shine animation */}
                    <motion.div
                      className="absolute inset-0 opacity-0 group-hover/code:opacity-100 pointer-events-none"
                      initial={{ x: "-100%" }}
                      whileHover={{ x: "100%" }}
                      transition={{ duration: 0.8 }}
                    >
                      <div className="h-full w-1/4 bg-gradient-to-r from-transparent via-white/3 to-transparent skew-x-12" />
                    </motion.div>
                  </div>
                </div>
              </motion.div>

              {/* Animated connector line using SVG dash animation */}
              {index < steps.length - 1 && (
                <div className="hidden lg:flex absolute -bottom-6 left-7 items-center gap-0">
                  <svg width="2" height="24" className="overflow-visible">
                    <motion.line
                      x1="1" y1="0" x2="1" y2="24"
                      stroke={step.accent}
                      strokeWidth="2"
                      strokeDasharray="24"
                      initial={{ strokeDashoffset: 24 }}
                      animate={inView ? { strokeDashoffset: 0 } : {}}
                      transition={{ duration: 0.6, delay: 0.5 + index * 0.3 }}
                    />
                  </svg>
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Advanced section toggle */}
        <div className="max-w-4xl mx-auto mt-10 flex justify-center">
          <motion.button
            onClick={() => setShowAdvanced((v) => !v)}
            aria-expanded={showAdvanced}
            className="group inline-flex items-center gap-2 px-6 py-3 rounded-full font-semibold text-sm text-green-600 dark:text-green-300 border border-green-500/30 bg-green-500/5 hover:border-green-400/60 hover:bg-green-500/10 transition-all"
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.97 }}
          >
            <motion.span
              animate={{ rotate: showAdvanced ? 180 : 0 }}
              transition={{ duration: 0.3 }}
              className="inline-flex"
            >
              <FiChevronDown size={16} />
            </motion.span>
            {showAdvanced ? "Hide advanced steps" : "Show advanced: routing, thinking & cost tracking"}
          </motion.button>
        </div>

        {showAdvanced && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="max-w-4xl mx-auto space-y-6 mt-8"
          >
            {advancedSteps.map((step, index) => {
              const realIndex = steps.length + index;
              return (
                <div key={index} className="relative">
                  <div
                    className="group flex flex-col lg:flex-row items-start gap-6 p-6 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 backdrop-blur-sm relative overflow-hidden shadow-sm dark:shadow-none"
                    style={{ borderColor: undefined }}
                  >
                    <div
                      className="flex-shrink-0 relative z-10 w-16 h-16 rounded-xl flex items-center justify-center"
                      style={{ background: `${step.accent}15`, border: `1px solid ${step.accent}30` }}
                    >
                      <span className="text-2xl font-black font-mono" style={{ color: step.accent }}>
                        {step.number}
                      </span>
                    </div>
                    <div className="flex-1 relative z-10">
                      <h3 className="text-xl font-bold mb-1.5 text-gray-900 dark:text-white">{step.title}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-500 mb-4">{step.description}</p>
                      <div className="relative rounded-xl bg-gray-50 dark:bg-[#0a1a0f] border border-gray-300 dark:border-gray-800 overflow-hidden">
                        <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-200 dark:border-gray-700/50">
                          <span
                            className="text-xs font-mono font-semibold px-2 py-0.5 rounded"
                            style={{ color: step.accent, background: `${step.accent}15`, border: `1px solid ${step.accent}30` }}
                          >
                            {step.language}
                          </span>
                          <motion.button
                            onClick={() => copyToClipboard(step.code, realIndex)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-all text-xs font-medium border border-gray-300 dark:border-gray-700"
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                          >
                            {copiedIndex === realIndex ? (
                              <><FiCheck className="text-green-400" size={12} /> Copied</>
                            ) : (
                              <><FiCopy size={12} /> Copy</>
                            )}
                          </motion.button>
                        </div>
                        <pre className="p-4 text-sm font-mono text-gray-800 dark:text-gray-200 overflow-x-auto leading-relaxed">
                          <code
                            className="syntax-code"
                            dangerouslySetInnerHTML={{ __html: highlightCode(step.code, step.language) }}
                          />
                        </pre>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </motion.div>
        )}
      </Container>
    </section>
  );
}
