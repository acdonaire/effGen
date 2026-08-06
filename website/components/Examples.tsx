"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiCode, FiSearch, FiBarChart2, FiUsers, FiArrowRight, FiCloud, FiBookOpen, FiZap } from "react-icons/fi";
import Link from "next/link";
import Container from "./Container";

export default function Examples() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  const examples = [
    {
      id: "code-assistant",
      icon: FiCode,
      emoji: "\u{1F4BB}",
      title: "Code Assistant",
      description: "Generate, execute, and debug code with sandboxed CodeExecutor and PythonREPL. Real-time streaming.",
      tags: ["CodeExecutor", "Python", "Streaming"],
      accent: "#00e5ff",
    },
    {
      id: "research-agent",
      icon: FiSearch,
      emoji: "\u{1F52C}",
      title: "Research Agent",
      description: "Comprehensive research using WebSearch, Wikipedia, and URLFetch. Detailed reports with citations.",
      tags: ["WebSearch", "Wikipedia", "Memory"],
      accent: "#a78bfa",
    },
    {
      id: "data-analysis",
      icon: FiBarChart2,
      emoji: "\u{1F4CA}",
      title: "Data Analysis Pipeline",
      description: "Load, clean, and analyze data with PythonREPL and FileOps. Create visualizations and insights.",
      tags: ["PythonREPL", "FileOps", "Presets"],
      accent: "#00ff88",
    },
    {
      id: "multi-agent",
      icon: FiUsers,
      emoji: "\u{1F91D}",
      title: "Multi-Agent System",
      description: "Coordinate specialized agents for complex workflows. A2A protocol with streaming callbacks.",
      tags: ["Multi-Agent", "A2A", "Presets"],
      accent: "#ff6b6b",
    },
    {
      id: "weather-json-pipeline",
      icon: FiCloud,
      emoji: "\u{1F324}\uFE0F",
      title: "Weather & JSON Pipeline",
      description: "Fetch real-time weather data, process JSON responses, and format reports using WeatherTool and JSONTool.",
      tags: ["Weather API", "JSON", "Pipeline"],
      accent: "#ffd700",
    },
    {
      id: "rag-knowledge-base",
      icon: FiBookOpen,
      emoji: "\u{1F4DA}",
      title: "RAG Knowledge Base",
      description: "Build a knowledge base with document loaders, chunking strategies, and hybrid search (vector + BM25).",
      tags: ["RAG", "BM25", "Hybrid Search"],
      accent: "#ff9500",
    },
  ];

  return (
    <section id="examples" className="py-24 bg-gray-50 dark:bg-[#020c08] relative overflow-hidden" ref={ref}>
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
            Examples
          </motion.span>
          <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
            See <span className="gradient-text">effGen</span> in Action
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Real-world examples showcasing the power and versatility
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5 mb-12">
          {examples.map((example, index) => (
            <Link key={index} href={`/examples/${example.id}`}>
              <motion.div
                initial={{ opacity: 0, y: 40 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: index * 0.08 }}
                whileHover={{ y: -8, scale: 1.02 }}
                className="group relative overflow-hidden p-6 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 backdrop-blur-sm cursor-pointer h-full shadow-sm dark:shadow-none"
                style={{
                  boxShadow: "0 1px 3px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.03), 0 8px 24px rgba(0,0,0,0.02)",
                }}
              >
                {/* Animated rotating border on hover */}
                <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none overflow-hidden">
                  <motion.div
                    className="absolute inset-[-50%] w-[200%] h-[200%]"
                    style={{ background: `conic-gradient(from 0deg, transparent 60%, ${example.accent}30 80%, transparent 100%)` }}
                    animate={{ rotate: 360 }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                  />
                  <div className="absolute inset-[1px] rounded-2xl bg-white dark:bg-gray-900/70" />
                </div>

                {/* Hover glow */}
                <motion.div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                  style={{ background: `radial-gradient(circle at 30% 30%, ${example.accent}10 0%, transparent 70%)` }}
                />
                <motion.div
                  className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ boxShadow: `inset 0 0 0 1px ${example.accent}30` }}
                />

                {/* Circuit pattern on hover */}
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-700 circuit-pattern pointer-events-none" />

                {/* Icon */}
                <div className="flex items-center gap-3 mb-5 relative z-10">
                  <motion.div
                    className="w-14 h-14 rounded-xl flex items-center justify-center text-2xl"
                    style={{ background: `${example.accent}15`, border: `1px solid ${example.accent}25` }}
                    whileHover={{ rotate: 10, scale: 1.1 }}
                  >
                    {example.emoji}
                  </motion.div>

                  {/* Pulsing status dot */}
                  <motion.div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ background: example.accent, boxShadow: `0 0 8px ${example.accent}` }}
                    animate={{ scale: [1, 1.3, 1], opacity: [1, 0.6, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </div>

                <h3 className="text-lg font-bold mb-2 text-gray-900 dark:text-white relative z-10 group-hover:text-green-800 dark:group-hover:text-green-50 transition-colors">
                  {example.title}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-500 mb-5 leading-relaxed relative z-10 group-hover:text-gray-700 dark:group-hover:text-gray-400 transition-colors">
                  {example.description}
                </p>

                {/* Tags with shimmer on hover */}
                <div className="flex flex-wrap gap-1.5 mb-5 relative z-10">
                  {example.tags.map((tag, idx) => (
                    <motion.span
                      key={idx}
                      className="px-2 py-0.5 rounded text-[10px] font-medium bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700 group-hover:border-opacity-50 transition-all relative overflow-hidden"
                      whileHover={{ scale: 1.05, borderColor: `${example.accent}40` }}
                    >
                      {tag}
                      <motion.div
                        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700"
                      />
                    </motion.span>
                  ))}
                </div>

                <motion.div
                  className="flex items-center gap-1.5 text-sm font-semibold relative z-10 transition-colors"
                  style={{ color: example.accent }}
                >
                  View Example
                  <motion.span
                    className="inline-block"
                    animate={{ x: [0, 0] }}
                    whileHover={{ x: 4 }}
                  >
                    <FiArrowRight size={14} className="group-hover:translate-x-1.5 transition-transform duration-300" />
                  </motion.span>
                </motion.div>

                {/* Bottom accent */}
                <motion.div
                  className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: `linear-gradient(90deg, transparent, ${example.accent}, transparent)` }}
                />
              </motion.div>
            </Link>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="text-center"
        >
          <Link href="/examples">
            <motion.span
              className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black cursor-pointer"
              style={{
                background: "linear-gradient(135deg, #00ff88, #00c96e)",
                boxShadow: "0 0 30px rgba(0,255,136,0.3)",
              }}
              whileHover={{ scale: 1.05, y: -2, boxShadow: "0 0 50px rgba(0,255,136,0.5)" }}
              whileTap={{ scale: 0.95 }}
            >
              View All Examples
              <FiArrowRight size={16} />
            </motion.span>
          </Link>
        </motion.div>
      </Container>
    </section>
  );
}
