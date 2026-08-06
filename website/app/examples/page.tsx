"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiArrowRight, FiGithub, FiZap } from "react-icons/fi";
import Link from "next/link";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const examples = [
  {
    id: "code-assistant",
    icon: "💻",
    title: "AI Code Assistant",
    description: "A powerful coding companion that helps you write better code faster. Supports multiple programming languages with intelligent code suggestions and debugging.",
    features: ["Multi-language code completion", "Intelligent bug detection", "Code refactoring suggestions", "Natural language to code"],
    stats: [{ value: "50+", label: "Languages" }, { value: "95%", label: "Accuracy" }, { value: "3x", label: "Faster" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/tools/coding_agent.py",
    accent: "#00e5ff",
  },
  {
    id: "research-agent",
    icon: "🔍",
    title: "Research Agent",
    description: "An intelligent research assistant that gathers, analyzes, and synthesizes information from multiple sources to provide comprehensive insights.",
    features: ["Multi-source information gathering", "Automatic fact-checking", "Citation management", "Summary generation"],
    stats: [{ value: "10+", label: "Sources" }, { value: "90%", label: "Accuracy" }, { value: "5x", label: "Faster" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/web_retrieval/web_agent.py",
    accent: "#a78bfa",
  },
  {
    id: "data-analysis",
    icon: "📊",
    title: "Data Analysis Agent",
    description: "Automate complex data analysis workflows with an intelligent agent that processes, visualizes, and extracts insights from large datasets.",
    features: ["Automated data cleaning", "Statistical analysis", "Interactive visualizations", "Pattern recognition"],
    stats: [{ value: "1M+", label: "Rows/sec" }, { value: "20+", label: "Formats" }, { value: "10x", label: "Faster" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/advanced/data_processing_agent.py",
    accent: "#00ff88",
  },
  {
    id: "multi-agent",
    icon: "🤖",
    title: "Multi-Agent System",
    description: "Build complex workflows with multiple specialized agents that collaborate to solve sophisticated problems and handle multi-step tasks.",
    features: ["Agent orchestration", "Task delegation", "Parallel execution", "Result synthesis"],
    stats: [{ value: "5+", label: "Agents" }, { value: "100%", label: "Parallel" }, { value: "15x", label: "Faster" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/advanced/multi_agent_pipeline.py",
    accent: "#ff6b6b",
  },
  {
    id: "weather-json-pipeline",
    icon: "🌤️",
    title: "Weather & JSON Pipeline",
    description: "Fetch real-time weather data, process JSON responses, and format reports using the new WeatherTool and JSONTool. No API keys required.",
    features: ["Real-time weather data", "JSON processing & validation", "Pipeline automation", "Free Open-Meteo API"],
    stats: [{ value: "Free", label: "No API Key" }, { value: "31", label: "Tools" }, { value: "Real-time", label: "Data" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/web_retrieval/weather_agent.py",
    accent: "#ffd700",
  },
  {
    id: "rag-knowledge-base",
    icon: "📚",
    title: "RAG Knowledge Base",
    description: "Build a knowledge base with document loaders, chunking strategies, and hybrid search combining vector similarity and BM25 keyword matching.",
    features: ["Multi-format document loading", "Hybrid search (vector + BM25)", "Smart document chunking", "Semantic understanding"],
    stats: [{ value: "5+", label: "Formats" }, { value: "Hybrid", label: "Search" }, { value: "BM25", label: "Keywords" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/web_retrieval/retrieval_agent.py",
    accent: "#ff9500",
  },
];

export default function ExamplesPage() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />

      {/* Hero */}
      <section className="relative pt-32 pb-20 overflow-hidden">
        <div className="absolute inset-0 grid-pattern" />
        <motion.div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] rounded-full opacity-[0.06] pointer-events-none"
          style={{ background: "radial-gradient(ellipse, #00ff88 0%, transparent 70%)" }}
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 6, repeat: Infinity }}
        />
        {/* Scan line */}
        <motion.div
          className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-400/30 to-transparent pointer-events-none z-10"
          animate={{ y: ["-50vh", "50vh"] }}
          transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
        />
        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="text-center max-w-4xl mx-auto"
          >
            <motion.div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-600 dark:text-green-400 text-sm font-semibold mb-8">
              <FiZap size={14} />
              Examples
            </motion.div>
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
              <span className="gradient-text">Real-World Examples</span>
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 leading-relaxed max-w-2xl mx-auto">
              Explore production-ready examples showcasing the power and flexibility of effGen.
              From simple code assistants to complex multi-agent systems.
            </p>
          </motion.div>
        </Container>
      </section>

      {/* Examples Grid */}
      <section className="py-12 pb-24 relative">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <Container className="relative z-10">
          <motion.div
            ref={ref}
            initial="hidden"
            animate={inView ? "visible" : "hidden"}
            variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.1 } } }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {examples.map((example) => (
              <motion.div
                key={example.id}
                variants={{ hidden: { opacity: 0, y: 30 }, visible: { opacity: 1, y: 0 } }}
                whileHover={{ y: -8, scale: 1.01 }}
                transition={{ duration: 0.3 }}
                className="group relative overflow-hidden rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none backdrop-blur-sm"
              >
                {/* Animated rotating border */}
                <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none overflow-hidden">
                  <motion.div
                    className="absolute inset-[-50%] w-[200%] h-[200%]"
                    style={{ background: `conic-gradient(from 0deg, transparent 60%, ${example.accent}25 80%, transparent 100%)` }}
                    animate={{ rotate: 360 }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                  />
                  <div className="absolute inset-[1px] rounded-2xl bg-white dark:bg-gray-900/70" />
                </div>

                {/* Hover glow */}
                <motion.div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                  style={{ background: `radial-gradient(circle at 30% 30%, ${example.accent}08 0%, transparent 70%)` }}
                />
                <motion.div
                  className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ boxShadow: `inset 0 0 0 1px ${example.accent}25` }}
                />

                <div className="p-8 relative z-10">
                  {/* Icon */}
                  <motion.div
                    className="w-16 h-16 rounded-xl flex items-center justify-center text-3xl mb-6"
                    style={{ background: `${example.accent}15`, border: `1px solid ${example.accent}25` }}
                    whileHover={{ scale: 1.15, rotate: 8 }}
                    transition={{ duration: 0.3 }}
                  >
                    {example.icon}
                  </motion.div>

                  {/* Title with pulsing dot */}
                  <div className="flex items-center gap-2 mb-3">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">{example.title}</h3>
                    <motion.div
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: example.accent, boxShadow: `0 0 8px ${example.accent}` }}
                      animate={{ scale: [1, 1.4, 1], opacity: [1, 0.5, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                  </div>
                  <p className="text-gray-600 dark:text-gray-500 mb-5 leading-relaxed text-sm">{example.description}</p>

                  {/* Features */}
                  <ul className="space-y-2 mb-6">
                    {example.features.map((feature, idx) => (
                      <li key={idx} className="flex items-center gap-2.5 text-sm text-gray-500">
                        <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: example.accent, boxShadow: `0 0 6px ${example.accent}60` }} />
                        <span className="text-gray-600 dark:text-gray-500">{feature}</span>
                      </li>
                    ))}
                  </ul>

                  {/* Stats */}
                  <div className="grid grid-cols-3 gap-3 mb-6 p-4 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                    {example.stats.map((stat, idx) => (
                      <div key={idx} className="text-center">
                        <div className="text-xl font-black mb-0.5" style={{ color: example.accent }}>{stat.value}</div>
                        <div className="text-[10px] uppercase tracking-wider text-gray-600 font-medium">{stat.label}</div>
                      </div>
                    ))}
                  </div>

                  {/* CTAs */}
                  <div className="flex gap-3">
                    <Link href={`/examples/${example.id}`} className="flex-1">
                      <motion.span
                        className="group/btn flex items-center justify-center gap-2 w-full px-5 py-3 rounded-xl font-bold text-sm text-black cursor-pointer"
                        style={{ background: `linear-gradient(135deg, ${example.accent}, ${example.accent}bb)`, boxShadow: `0 0 20px ${example.accent}30` }}
                        whileHover={{ scale: 1.02, boxShadow: `0 0 30px ${example.accent}50` }}
                        whileTap={{ scale: 0.97 }}
                      >
                        View Example
                        <span className="group-hover/btn:translate-x-1 transition-transform inline-block">
                          <FiArrowRight size={14} />
                        </span>
                      </motion.span>
                    </Link>
                    <motion.a
                      href={example.githubUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-5 py-3 rounded-xl font-semibold text-sm border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800/60 text-gray-700 dark:text-gray-300 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all flex items-center gap-2"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.97 }}
                    >
                      <FiGithub size={14} />
                      Code
                    </motion.a>
                  </div>
                </div>

                {/* Bottom accent */}
                <div
                  className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: `linear-gradient(90deg, transparent, ${example.accent}, transparent)` }}
                />
              </motion.div>
            ))}
          </motion.div>

          {/* CTA */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.4 }}
            className="relative overflow-hidden rounded-2xl p-10 text-center mt-12 bg-gray-50 dark:bg-gray-950/80 border border-gray-200 dark:border-green-500/15 shadow-sm dark:shadow-none"
            style={{ boxShadow: "0 0 40px rgba(0,255,136,0.04)" }}
          >
            <motion.div
              className="absolute top-0 left-0 right-0 h-px"
              style={{ background: "linear-gradient(90deg, transparent, #00ff88, transparent)" }}
              animate={{ opacity: [0.3, 0.8, 0.3] }}
              transition={{ duration: 3, repeat: Infinity }}
            />
            <div className="absolute inset-0 grid-pattern opacity-50" />

            {/* Floating geometric shapes */}
            <motion.div
              className="absolute top-10 left-10 w-20 h-20 border border-green-500/10 rounded-lg pointer-events-none"
              animate={{ rotate: 360, scale: [1, 1.1, 1] }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            />
            <motion.div
              className="absolute bottom-10 right-10 w-16 h-16 border border-cyan-500/10 rounded-full pointer-events-none"
              animate={{ rotate: -360, scale: [1, 1.2, 1] }}
              transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
            />
            <motion.div
              className="absolute top-1/2 right-20 w-12 h-12 border border-violet-500/10 pointer-events-none"
              style={{ clipPath: "polygon(50% 0%, 100% 100%, 0% 100%)" }}
              animate={{ rotate: 360 }}
              transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
            />

            <div className="relative z-10">
              <div className="text-4xl mb-5">🚀</div>
              <h2 className="text-2xl md:text-3xl font-black mb-3 text-gray-900 dark:text-white">
                <span className="gradient-text">Share Your Example</span>
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-xl mx-auto text-sm leading-relaxed">
                Built something amazing with effGen? We&apos;d love to feature your example! Join our community and share your creation.
              </p>
              <div className="flex flex-wrap gap-4 justify-center">
                <Link href="/community">
                  <motion.span
                    className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black cursor-pointer"
                    style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)", boxShadow: "0 0 25px rgba(0,255,136,0.3)" }}
                    whileHover={{ scale: 1.05, boxShadow: "0 0 40px rgba(0,255,136,0.5)" }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Join Community
                    <FiArrowRight size={14} />
                  </motion.span>
                </Link>
                <motion.a
                  href="https://github.com/ctrl-gaurav/effGen/tree/main/examples"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-900/60 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Browse on GitHub
                </motion.a>
              </div>
            </div>
          </motion.div>
        </Container>
      </section>

      <Footer />
    </div>
  );
}
