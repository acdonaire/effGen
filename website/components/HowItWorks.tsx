"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiMessageSquare, FiCpu, FiTool, FiPlay, FiEye, FiCheckCircle, FiX } from "react-icons/fi";
import { useState } from "react";
import Container from "./Container";

const expandedDetails = [
  { explanation: "The agent receives a natural language query via agent.run(). Input is validated, tokenized, and formatted into the ReAct prompt template with available tool descriptions.", code: 'result = agent.run("Calculate 24344 * 334")\n# Input formatted into ReAct template' },
  { explanation: "The SLM generates a Thought step, analyzing task requirements. It identifies what information or computation is needed and plans which tool to use.", code: 'Thought: I need to multiply these numbers.\nI should use the Calculator tool.' },
  { explanation: "Based on reasoning, the agent selects the best tool from 66 built-in options. Selection considers task type, required parameters, and fallback options.", code: 'Action: Calculator\nAction Input: {"expression": "24344 * 334"}' },
  { explanation: "The selected tool executes in a sandboxed environment with configurable timeouts, memory limits, and security controls.", code: 'await calculator.execute(expression="24344 * 334")\n# Executed with 5s timeout -> Result: 8130896' },
  { explanation: "Tool output is captured and formatted as an Observation. The agent validates the result and decides whether to continue or provide a final answer.", code: 'Observation: 8130896\n# Agent validates: result is a number\n# Decision: task complete' },
  { explanation: "The agent synthesizes the observation into a natural language response. If the task isn't complete, steps 2-5 repeat up to max_iterations.", code: 'Final Answer: The result of\n24344 \u00d7 334 is 8,130,896' },
];

const steps = [
  {
    icon: FiMessageSquare,
    title: "User Input",
    description: "Natural language task or query is received by the agent",
    accent: "#00ff88",
    detail: 'agent.run("Calculate 24344 * 334")',
  },
  {
    icon: FiCpu,
    title: "Reasoning",
    description: "Agent analyzes the task using ReAct-style thinking",
    accent: "#00e5ff",
    detail: "Thought: I need to multiply these numbers...",
  },
  {
    icon: FiTool,
    title: "Tool Selection",
    description: "Best tool is selected from 66 built-in options",
    accent: "#a78bfa",
    detail: "Action: Calculator(24344 * 334)",
  },
  {
    icon: FiPlay,
    title: "Execution",
    description: "Tool runs in a sandboxed environment with safety controls",
    accent: "#ffd700",
    detail: "Executing Calculator...",
  },
  {
    icon: FiEye,
    title: "Observation",
    description: "Agent observes and validates the tool output",
    accent: "#ff9500",
    detail: "Observation: 8130896",
  },
  {
    icon: FiCheckCircle,
    title: "Final Answer",
    description: "Synthesized response returned to the user",
    accent: "#00ff88",
    detail: "Answer: 8,130,896",
  },
];

function FlowPulse() {
  return (
    <motion.div
      className="absolute top-0 left-0 w-3 h-3 rounded-full z-20 pointer-events-none hidden lg:block"
      style={{
        background: "radial-gradient(circle, #00ff88, transparent)",
        boxShadow: "0 0 12px #00ff88, 0 0 24px rgba(0,255,136,0.4)",
      }}
      animate={{
        left: ["0%", "100%"],
        opacity: [0, 1, 1, 1, 1, 0],
      }}
      transition={{
        duration: 4,
        repeat: Infinity,
        ease: "easeInOut",
        repeatDelay: 1,
      }}
    />
  );
}

export default function HowItWorks() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  return (
    <section className="py-24 bg-white dark:bg-[#020c08] relative overflow-hidden noise-overlay" ref={ref}>
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

      <motion.div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-[0.04] pointer-events-none"
        style={{ background: "radial-gradient(circle, #00e5ff 0%, transparent 70%)" }}
        animate={{ scale: [1, 1.15, 1] }}
        transition={{ duration: 10, repeat: Infinity }}
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
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-cyan-500/30 bg-cyan-500/5 text-cyan-600 dark:text-cyan-400 text-sm font-semibold mb-6"
            whileHover={{ borderColor: "rgba(0,229,255,0.6)" }}
          >
            <FiCpu size={14} />
            Architecture
          </motion.span>
          <h2 className="text-4xl md:text-5xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
            How <span className="gradient-text">effGen</span> Works
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            The ReAct agent loop — reasoning and acting in perfect harmony
          </p>
        </motion.div>

        {/* Flow diagram */}
        <div className="relative max-w-6xl mx-auto">
          {/* Connection line (desktop) */}
          <div className="hidden lg:block absolute top-[60px] left-[8%] right-[8%] h-px z-0">
            <div className="w-full h-full bg-gradient-to-r from-green-500/30 via-cyan-500/20 to-green-500/30" />
            <div className="relative w-full" style={{ marginTop: "-7px" }}>
              <FlowPulse />
            </div>
          </div>

          {/* Steps */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6 lg:gap-4">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 40 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: index * 0.12 }}
                className="relative group"
              >
                {/* Step card */}
                <div
                  className="relative p-5 rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 backdrop-blur-sm overflow-hidden text-center h-full cursor-pointer"
                  onClick={() => setSelectedStep(index)}
                >
                  {/* Hover glow */}
                  <motion.div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                    style={{ background: `radial-gradient(circle at 50% 30%, ${step.accent}15 0%, transparent 70%)` }}
                  />

                  {/* Step number */}
                  <div
                    className="absolute top-2 right-3 text-[10px] font-mono font-bold opacity-40"
                    style={{ color: step.accent }}
                  >
                    {String(index + 1).padStart(2, "0")}
                  </div>

                  {/* Icon */}
                  <motion.div
                    className="relative w-12 h-12 rounded-xl flex items-center justify-center mb-4 mx-auto overflow-hidden"
                    style={{
                      background: `${step.accent}15`,
                      border: `1px solid ${step.accent}30`,
                    }}
                    whileHover={{ scale: 1.1, rotate: 5 }}
                  >
                    <step.icon style={{ color: step.accent }} size={20} />
                    <motion.div
                      className="absolute inset-[-2px] rounded-xl pointer-events-none"
                      style={{
                        background: `conic-gradient(from 0deg, transparent 70%, ${step.accent}40 90%, transparent 100%)`,
                      }}
                      animate={{ rotate: 360 }}
                      transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    />
                  </motion.div>

                  {/* Title */}
                  <h3 className="text-sm font-bold mb-2 text-gray-900 dark:text-white">
                    {step.title}
                  </h3>

                  {/* Description */}
                  <p className="text-xs text-gray-500 dark:text-gray-500 leading-relaxed mb-3">
                    {step.description}
                  </p>

                  {/* Code detail */}
                  <div
                    className="px-2 py-1.5 rounded-lg bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800"
                  >
                    <code className="text-[10px] font-mono block truncate" style={{ color: step.accent }}>
                      {step.detail}
                    </code>
                  </div>

                  {/* Click hint */}
                  <div className="mt-2 text-[9px] text-gray-400 dark:text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity font-medium tracking-wide uppercase">
                    Click for details
                  </div>

                  {/* Bottom accent */}
                  <motion.div
                    className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ background: `linear-gradient(90deg, transparent, ${step.accent}, transparent)` }}
                  />
                </div>

                {/* Arrow between steps on mobile/tablet */}
                {index < steps.length - 1 && (
                  <div className="flex justify-center py-2 lg:hidden">
                    <motion.div
                      className="w-px h-6 bg-gradient-to-b from-green-500/40 to-transparent"
                      initial={{ scaleY: 0 }}
                      animate={inView ? { scaleY: 1 } : {}}
                      transition={{ delay: index * 0.12 + 0.3 }}
                    />
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          {/* Loop indicator */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 1 }}
            className="mt-8 text-center"
          >
            <div className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-green-500/5 border border-green-500/20">
              <motion.div
                className="w-2 h-2 rounded-full bg-green-400"
                animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
              <span className="text-xs font-mono text-green-600 dark:text-green-400">
                Steps 2-5 repeat until the task is complete (max_iterations configurable)
              </span>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                className="text-green-400"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                  <polyline points="21 3 21 12 12 12" />
                </svg>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </Container>

      {/* Expanded Step Modal */}
      <AnimatePresence>
        {selectedStep !== null && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="absolute inset-0 bg-black/70 backdrop-blur-sm"
              onClick={() => setSelectedStep(null)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            />
            <motion.div
              className="relative w-full max-w-xl rounded-2xl bg-white dark:bg-[#0a1a0f] border border-gray-200 dark:border-green-500/20 shadow-2xl overflow-hidden"
              style={{ boxShadow: `0 0 60px ${steps[selectedStep].accent}15` }}
              initial={{ scale: 0.9, opacity: 0, y: 30 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 30 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
            >
              {/* Header bar */}
              <motion.div
                className="h-1"
                style={{ background: `linear-gradient(90deg, transparent, ${steps[selectedStep].accent}, transparent)` }}
              />

              <div className="p-6">
                <button
                  onClick={() => setSelectedStep(null)}
                  className="absolute top-4 right-4 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 hover:text-gray-900 dark:hover:text-white transition-colors z-10"
                >
                  <FiX size={16} />
                </button>

                <div className="flex items-center gap-3 mb-4">
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center"
                    style={{ background: `${steps[selectedStep].accent}15`, border: `1px solid ${steps[selectedStep].accent}30` }}
                  >
                    {(() => {
                      const StepIcon = steps[selectedStep].icon;
                      return <StepIcon style={{ color: steps[selectedStep].accent }} size={20} />;
                    })()}
                  </div>
                  <div>
                    <h3 className="text-xl font-black text-gray-900 dark:text-white">{steps[selectedStep].title}</h3>
                    <span className="text-xs font-mono" style={{ color: steps[selectedStep].accent }}>
                      Step {String(selectedStep + 1).padStart(2, "0")}
                    </span>
                  </div>
                </div>

                <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-5">
                  {expandedDetails[selectedStep].explanation}
                </p>

                {/* Code block */}
                <div
                  className="px-4 py-3 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 font-mono text-xs"
                  style={{ borderColor: `${steps[selectedStep].accent}25` }}
                >
                  {expandedDetails[selectedStep].code.split("\n").map((line, i) => (
                    <div key={i} style={{ color: steps[selectedStep].accent }}>{line}</div>
                  ))}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
