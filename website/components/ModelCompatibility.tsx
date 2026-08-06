"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useState } from "react";
import { FiAward, FiCpu, FiChevronDown, FiChevronUp, FiStar, FiZap } from "react-icons/fi";
import Container from "./Container";

interface ModelData {
  name: string;
  size: string;
  score: number;
  passRate: number;
  note: string;
  tier: "perfect" | "excellent" | "good" | "fair" | "poor";
  recommended?: boolean;
}

const models: ModelData[] = [
  { name: "Qwen2.5-1.5B-Instruct", size: "1.5B", score: 10, passRate: 100, note: "Best value", tier: "perfect", recommended: true },
  { name: "Qwen2.5-3B-Instruct", size: "3B", score: 10, passRate: 100, note: "Recommended default", tier: "perfect", recommended: true },
  { name: "Phi-4-mini-instruct", size: "3.8B", score: 10, passRate: 100, note: "Cross-family validated", tier: "perfect", recommended: true },
  { name: "Qwen3-1.7B", size: "1.7B", score: 9.5, passRate: 95, note: "Strong alternative", tier: "excellent" },
  { name: "Qwen2.5-7B-Instruct", size: "7B", score: 9.0, passRate: 80, note: "When 7B quality needed", tier: "excellent" },
  { name: "Qwen3-4B", size: "4B", score: 9.0, passRate: 80, note: "Good 4B option", tier: "excellent" },
  { name: "Llama-3.2-3B-Instruct", size: "3B", score: 8.5, passRate: 85, note: "Single-turn tasks", tier: "good" },
  { name: "Llama-3.1-8B-Instruct", size: "8B", score: 8.5, passRate: 70, note: "General purpose 8B", tier: "good" },
  { name: "gemma-3-4b-it", size: "4B", score: 8.0, passRate: 70, note: "Single-turn only", tier: "good" },
  { name: "Qwen2.5-0.5B-Instruct", size: "0.5B", score: 7.0, passRate: 40, note: "Lightweight fallback", tier: "fair" },
  { name: "SmolLM2-1.7B-Instruct", size: "1.7B", score: 4.0, passRate: 30, note: "Not recommended", tier: "poor" },
];

const tierColors: Record<string, string> = {
  perfect: "#00ff88",
  excellent: "#00e5ff",
  good: "#ffd700",
  fair: "#ff9500",
  poor: "#ff6b6b",
};

const tierLabels: Record<string, string> = {
  perfect: "Perfect (10/10)",
  excellent: "Excellent (9+)",
  good: "Good (8+)",
  fair: "Fair (7+)",
  poor: "Poor (<7)",
};

function ScoreBar({ score, inView, delay }: { score: number; inView: boolean; delay: number }) {
  const widthPercent = (score / 10) * 100;
  const color = score >= 10 ? "#00ff88" : score >= 9 ? "#00e5ff" : score >= 8 ? "#ffd700" : score >= 7 ? "#ff9500" : "#ff6b6b";

  return (
    <div className="w-full h-2 rounded-full bg-gray-200 dark:bg-gray-800 overflow-hidden">
      <motion.div
        className="h-full rounded-full"
        style={{
          background: `linear-gradient(90deg, ${color}90, ${color})`,
          boxShadow: `0 0 8px ${color}40`,
        }}
        initial={{ width: 0 }}
        animate={inView ? { width: `${widthPercent}%` } : { width: 0 }}
        transition={{ duration: 1, delay, ease: "easeOut" }}
      />
    </div>
  );
}

export default function ModelCompatibility() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [expandedTier, setExpandedTier] = useState<string | null>(null);
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);

  const tiers = ["perfect", "excellent", "good", "fair", "poor"];

  const toggleTier = (tier: string) => {
    setExpandedTier(expandedTier === tier ? null : tier);
  };

  return (
    <section className="py-24 bg-gray-50 dark:bg-[#030f07] relative overflow-hidden" ref={ref}>
      <div className="absolute inset-0 grid-pattern" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

      <Container className="relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-14"
        >
          <motion.span
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-600 dark:text-green-400 text-sm font-semibold mb-6"
            whileHover={{ borderColor: "rgba(0,255,136,0.6)" }}
          >
            <FiCpu size={14} />
            Tested Models
          </motion.span>
          <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
            <span className="gradient-text">11 Models</span> Tested & Scored
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Comprehensive benchmarking across model families and sizes
          </p>
        </motion.div>

        {/* Tier summary cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-10"
        >
          {tiers.map((tier) => {
            const count = models.filter((m) => m.tier === tier).length;
            const color = tierColors[tier];
            return (
              <motion.button
                key={tier}
                onClick={() => toggleTier(tier)}
                className="relative p-4 rounded-xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 text-center overflow-hidden group"
                whileHover={{ y: -4 }}
              >
                <motion.div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl"
                  style={{ background: `radial-gradient(circle, ${color}10 0%, transparent 70%)` }}
                />
                <div className="text-2xl font-black mb-1" style={{ color }}>
                  {count}
                </div>
                <div className="text-[10px] font-bold uppercase tracking-wider text-gray-500">
                  {tierLabels[tier]}
                </div>
                <motion.div
                  className="absolute bottom-0 left-0 right-0 h-0.5"
                  style={{ background: color }}
                  initial={{ scaleX: 0 }}
                  animate={inView ? { scaleX: 1 } : {}}
                  transition={{ duration: 0.6, delay: 0.3 }}
                />
              </motion.button>
            );
          })}
        </motion.div>

        {/* Model table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 overflow-hidden"
        >
          {/* Table header */}
          <div className="grid grid-cols-12 gap-2 px-4 md:px-6 py-3 bg-gray-50 dark:bg-gray-900/80 border-b border-gray-200 dark:border-gray-800 text-xs font-bold uppercase tracking-wider text-gray-500">
            <div className="col-span-4 md:col-span-3">Model</div>
            <div className="col-span-2 md:col-span-1 text-center">Size</div>
            <div className="col-span-2 md:col-span-1 text-center">Score</div>
            <div className="hidden md:block md:col-span-1 text-center">Pass %</div>
            <div className="col-span-4 md:col-span-4">Performance</div>
            <div className="hidden md:block md:col-span-2">Note</div>
          </div>

          {/* Table rows */}
          {models.map((model, index) => {
            const color = tierColors[model.tier];
            const isHovered = hoveredRow === index;

            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={inView ? { opacity: 1, x: 0 } : {}}
                transition={{ duration: 0.4, delay: 0.1 * index }}
                className="relative grid grid-cols-12 gap-2 px-4 md:px-6 py-4 border-b border-gray-100 dark:border-gray-800/50 items-center cursor-default group"
                onMouseEnter={() => setHoveredRow(index)}
                onMouseLeave={() => setHoveredRow(null)}
              >
                {/* Hover background */}
                <motion.div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: `linear-gradient(90deg, ${color}08 0%, transparent 100%)` }}
                />

                {/* Model name */}
                <div className="col-span-4 md:col-span-3 flex items-center gap-2 relative z-10">
                  {model.recommended && (
                    <motion.div
                      animate={{ rotate: [0, 10, -10, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      <FiAward size={14} style={{ color }} />
                    </motion.div>
                  )}
                  <span className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                    {model.name}
                  </span>
                  {model.recommended && (
                    <span
                      className="hidden md:inline-flex px-1.5 py-0.5 text-[9px] font-bold rounded uppercase"
                      style={{ background: `${color}20`, color, border: `1px solid ${color}30` }}
                    >
                      Top Pick
                    </span>
                  )}
                </div>

                {/* Size */}
                <div className="col-span-2 md:col-span-1 text-center relative z-10">
                  <span className="text-xs font-mono text-gray-600 dark:text-gray-400 px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-800">
                    {model.size}
                  </span>
                </div>

                {/* Score */}
                <div className="col-span-2 md:col-span-1 text-center relative z-10">
                  <span className="text-sm font-black" style={{ color }}>
                    {model.score}
                  </span>
                  <span className="text-xs text-gray-500">/10</span>
                </div>

                {/* Pass rate (desktop) */}
                <div className="hidden md:block md:col-span-1 text-center relative z-10">
                  <span className="text-xs font-semibold" style={{ color }}>{model.passRate}%</span>
                </div>

                {/* Score bar */}
                <div className="col-span-4 md:col-span-4 relative z-10">
                  <ScoreBar score={model.score} inView={inView} delay={0.1 * index + 0.5} />
                </div>

                {/* Note (desktop) */}
                <div className="hidden md:block md:col-span-2 relative z-10">
                  <span className="text-xs text-gray-500">{model.note}</span>
                </div>

                {/* Hover detail tooltip (mobile - shows pass rate & note) */}
                <AnimatePresence>
                  {isHovered && (
                    <motion.div
                      className="absolute right-4 top-full z-30 md:hidden"
                      initial={{ opacity: 0, y: -4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -4 }}
                    >
                      <div className="px-3 py-2 rounded-lg bg-gray-900 text-white text-xs shadow-xl border border-gray-700">
                        <div>Pass rate: {model.passRate}%</div>
                        <div className="text-gray-400">{model.note}</div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Bottom recommendation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 1.2 }}
          className="mt-8 text-center"
        >
          <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-green-500/5 border border-green-500/20">
            <FiStar className="text-green-400" size={16} />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-bold text-green-600 dark:text-green-400">Qwen2.5-3B-Instruct</span> is the recommended default for the best balance of quality and speed
            </span>
          </div>
        </motion.div>
      </Container>
    </section>
  );
}
