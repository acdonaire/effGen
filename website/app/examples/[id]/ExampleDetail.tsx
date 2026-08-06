"use client";

import { motion } from "framer-motion";
import { FiArrowRight, FiGithub, FiExternalLink, FiCopy, FiCheck, FiZap, FiChevronRight } from "react-icons/fi";
import Link from "next/link";
import { useState, useEffect } from "react";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { highlightCode } from "@/components/syntaxHighlight";
import { examplesData } from "./examplesData";

export default function ExampleDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const [id, setId] = useState<string>("");
  const [copiedCode, setCopiedCode] = useState(false);

  useEffect(() => {
    params.then((p) => setId(p.id));
  }, [params]);

  const example = id ? examplesData[id] : null;

  if (!example && id) {
    return (
      <div className="min-h-screen bg-white dark:bg-[#020c08] flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4 text-gray-900 dark:text-white">Example Not Found</h1>
          <Link href="/examples" className="text-green-600 dark:text-green-400 hover:text-green-500 dark:hover:text-green-300 transition-colors">
            &larr; Back to Examples
          </Link>
        </div>
      </div>
    );
  }

  if (!example) {
    return (
      <div className="min-h-screen bg-white dark:bg-[#020c08] flex items-center justify-center">
        <div className="w-8 h-8 rounded-full border-2 border-green-400 border-t-transparent animate-spin" />
      </div>
    );
  }

  const copyCode = () => {
    navigator.clipboard.writeText(example.codeExample);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />

      {/* Hero */}
      <section className="relative pt-32 pb-20 overflow-hidden">
        <div className="absolute inset-0 grid-pattern" />
        <motion.div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[300px] rounded-full opacity-[0.06] pointer-events-none"
          style={{ background: `radial-gradient(ellipse, ${example.accent} 0%, transparent 70%)` }}
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 6, repeat: Infinity }}
        />
        {/* Scan line */}
        <motion.div
          className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-400/20 to-transparent pointer-events-none z-10"
          animate={{ y: ["-50vh", "50vh"] }}
          transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
        />
        {/* Floating geometric shapes */}
        <motion.div
          className="absolute top-20 right-20 w-24 h-24 border border-green-500/10 rounded-lg pointer-events-none"
          animate={{ rotate: 360, scale: [1, 1.1, 1] }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="absolute bottom-20 left-20 w-16 h-16 border border-cyan-500/10 rounded-full pointer-events-none"
          animate={{ rotate: -360 }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
        />

        <Container className="relative z-10">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-600 mb-8 font-mono">
            <Link href="/" className="hover:text-green-600 dark:hover:text-green-400 transition-colors">Home</Link>
            <FiChevronRight size={12} />
            <Link href="/examples" className="hover:text-green-600 dark:hover:text-green-400 transition-colors">Examples</Link>
            <FiChevronRight size={12} />
            <span className="text-gray-600 dark:text-gray-400">{example.title}</span>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="max-w-4xl mx-auto text-center"
          >
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold mb-8"
              style={{ background: `${example.accent}20`, border: `1px solid ${example.accent}40`, color: example.accent }}
            >
              <span>{example.icon}</span>
              {example.badge}
            </motion.div>

            <h1 className="text-5xl md:text-6xl lg:text-7xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
              {example.title}
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 leading-relaxed mb-10">{example.subtitle}</p>

            <div className="flex flex-wrap gap-4 justify-center">
              <motion.a
                href={example.githubUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
                style={{ background: `linear-gradient(135deg, ${example.accent}, ${example.accent}bb)`, boxShadow: `0 0 25px ${example.accent}40` }}
                whileHover={{ scale: 1.05, boxShadow: `0 0 40px ${example.accent}60` }}
                whileTap={{ scale: 0.95 }}
              >
                <FiGithub size={16} />
                View on GitHub
                <FiArrowRight size={14} />
              </motion.a>
              <motion.a
                href={example.githubUrl}
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-900/60 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <FiExternalLink size={16} />
                Download Code
              </motion.a>
            </div>
          </motion.div>
        </Container>
      </section>

      {/* Overview */}
      <section className="py-16 relative">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <Container className="relative z-10">
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <h2 className="text-3xl font-black mb-5 text-gray-900 dark:text-white">Overview</h2>
              <p className="text-gray-600 dark:text-gray-400 leading-relaxed text-lg">{example.description}</p>
            </motion.div>

            {/* Stats Banner */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative overflow-hidden rounded-2xl p-8 md:p-10 mt-10 border"
              style={{
                background: `linear-gradient(135deg, ${example.accent}15 0%, transparent 60%)`,
                borderColor: `${example.accent}30`,
                boxShadow: `0 0 40px ${example.accent}10`,
              }}
            >
              <div
                className="absolute top-0 left-0 right-0 h-px"
                style={{ background: `linear-gradient(90deg, transparent, ${example.accent}, transparent)` }}
              />
              <h3 className="text-2xl font-black text-gray-900 dark:text-white text-center mb-8">Performance Metrics</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                {example.stats.map((stat: any, idx: number) => (
                  <div key={idx} className="text-center">
                    <div className="text-4xl font-black mb-1" style={{ color: example.accent }}>{stat.value}</div>
                    <div className="text-sm text-gray-500 font-medium">{stat.label}</div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </Container>
      </section>

      {/* Features */}
      <section className="py-16 relative">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
        <div className="absolute inset-0 grid-pattern" />
        <Container className="relative z-10">
          <div className="max-w-4xl mx-auto">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="text-3xl font-black mb-10 text-gray-900 dark:text-white"
            >
              Key Features
            </motion.h2>

            <div className="grid md:grid-cols-2 gap-4">
              {example.features.map((feature: any, idx: number) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: idx * 0.08 }}
                  viewport={{ once: true }}
                  whileHover={{ y: -4, scale: 1.02 }}
                  className="group relative p-5 rounded-xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none overflow-hidden"
                >
                  {/* Animated rotating border */}
                  <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none overflow-hidden">
                    <motion.div
                      className="absolute inset-[-50%] w-[200%] h-[200%]"
                      style={{ background: `conic-gradient(from 0deg, transparent 60%, ${example.accent}25 80%, transparent 100%)` }}
                      animate={{ rotate: 360 }}
                      transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    />
                    <div className="absolute inset-[1px] rounded-xl bg-white dark:bg-gray-900/70" />
                  </div>

                  <motion.div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-xl"
                    style={{ background: `radial-gradient(circle at 20% 50%, ${example.accent}08 0%, transparent 70%)` }}
                  />
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center text-xl mb-4 relative z-10"
                    style={{ background: `${example.accent}15`, border: `1px solid ${example.accent}25` }}
                  >
                    {feature.icon}
                  </div>
                  <h4 className="text-base font-bold mb-1.5 text-gray-900 dark:text-white relative z-10">{feature.title}</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-500 relative z-10">{feature.description}</p>
                  <div
                    className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ background: `linear-gradient(90deg, transparent, ${example.accent}60, transparent)` }}
                  />
                </motion.div>
              ))}
            </div>
          </div>
        </Container>
      </section>

      {/* Code Example */}
      <section className="py-16 relative">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <Container className="relative z-10">
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
            >
              <h2 className="text-3xl font-black mb-3 text-gray-900 dark:text-white">Quick Start</h2>
              <p className="text-gray-600 dark:text-gray-500 mb-8 text-sm">Get started with this example in just a few lines of code:</p>

              <div className="relative rounded-2xl bg-gray-50 dark:bg-[#0a1a0f] border border-gray-300 dark:border-green-500/20 overflow-hidden" style={{ boxShadow: `0 0 30px ${example.accent}08` }}>
                <div
                  className="h-px"
                  style={{ background: `linear-gradient(90deg, transparent, ${example.accent}80, transparent)` }}
                />
                <div className="flex items-center justify-between px-5 py-3 border-b border-gray-200 dark:border-gray-700/50">
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1.5">
                      {["#ff5f57", "#febc2e", "#28c840"].map((c, i) => (
                        <div key={i} className="w-3 h-3 rounded-full" style={{ background: c }} />
                      ))}
                    </div>
                    <span className="text-xs font-mono" style={{ color: example.accent }}>example.py</span>
                  </div>
                  <button
                    onClick={copyCode}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-all text-xs font-medium border border-gray-300 dark:border-gray-700"
                  >
                    {copiedCode ? <><FiCheck className="text-green-400" size={12} /> Copied</> : <><FiCopy size={12} /> Copy</>}
                  </button>
                </div>
                <div className="relative">
                  {/* Scanline overlay */}
                  <div
                    className="absolute inset-0 pointer-events-none z-10 opacity-[0.03]"
                    style={{
                      backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.1) 2px, rgba(0,255,136,0.1) 4px)",
                      backgroundSize: "100% 4px",
                    }}
                  />
                  <pre className="p-6 text-sm font-mono text-gray-700 dark:text-gray-300 overflow-x-auto leading-relaxed">
                    <code
                      className="syntax-code"
                      dangerouslySetInnerHTML={{ __html: highlightCode(example.codeExample, "python") }}
                    />
                  </pre>
                </div>
              </div>
            </motion.div>
          </div>
        </Container>
      </section>

      {/* Use Cases */}
      <section className="py-16 relative">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
        <div className="absolute inset-0 grid-pattern" />
        <Container className="relative z-10">
          <div className="max-w-4xl mx-auto">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="text-3xl font-black mb-10 text-gray-900 dark:text-white"
            >
              Use Cases
            </motion.h2>

            <div className="space-y-3">
              {example.useCases.map((useCase: any, idx: number) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: idx * 0.08 }}
                  viewport={{ once: true }}
                  whileHover={{ x: 6 }}
                  className="group relative flex items-start gap-4 p-5 rounded-xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none overflow-hidden"
                >
                  <motion.div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl"
                    style={{ background: `radial-gradient(circle at 5% 50%, ${example.accent}06 0%, transparent 50%)` }}
                  />
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <div
                      className="w-12 h-12 rounded-xl flex items-center justify-center text-xl"
                      style={{ background: `${example.accent}15`, border: `1px solid ${example.accent}25` }}
                    >
                      {useCase.icon}
                    </div>
                    {/* Pulsing dot */}
                    <motion.div
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: example.accent, boxShadow: `0 0 8px ${example.accent}` }}
                      animate={{ scale: [1, 1.4, 1], opacity: [1, 0.5, 1] }}
                      transition={{ duration: 2, repeat: Infinity, delay: idx * 0.3 }}
                    />
                  </div>
                  <div className="relative z-10">
                    <h4 className="text-base font-bold mb-1 text-gray-900 dark:text-white">{useCase.title}</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-500">{useCase.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </Container>
      </section>

      {/* CTA */}
      <section className="py-16 relative">
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="relative overflow-hidden rounded-2xl p-10 text-center max-w-3xl mx-auto bg-gray-50 dark:bg-gray-950/90 border border-gray-200 dark:border-green-500/15 shadow-sm dark:shadow-none"
            style={{ boxShadow: "0 0 50px rgba(0,255,136,0.05)" }}
          >
            <motion.div
              className="absolute top-0 left-0 right-0 h-px"
              style={{ background: "linear-gradient(90deg, transparent, #00ff88, transparent)" }}
              animate={{ opacity: [0.3, 0.8, 0.3] }}
              transition={{ duration: 3, repeat: Infinity }}
            />
            <div className="absolute inset-0 grid-pattern opacity-50" />
            <div className="relative z-10">
              <div className="text-4xl mb-5">🚀</div>
              <h2 className="text-2xl md:text-3xl font-black mb-3 text-gray-900 dark:text-white">
                <span className="gradient-text">Ready to Build?</span>
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-xl mx-auto text-sm">
                Start building with this example today. Check out the full source code on GitHub.
              </p>
              <div className="flex flex-wrap gap-4 justify-center">
                <motion.a
                  href={example.githubUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
                  style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)", boxShadow: "0 0 25px rgba(0,255,136,0.3)" }}
                  whileHover={{ scale: 1.05, boxShadow: "0 0 40px rgba(0,255,136,0.5)" }}
                  whileTap={{ scale: 0.95 }}
                >
                  <FiGithub size={16} />
                  View Full Code
                  <FiArrowRight size={14} />
                </motion.a>
                <Link href="/examples">
                  <motion.span
                    className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-900/60 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all cursor-pointer"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    More Examples
                  </motion.span>
                </Link>
              </div>
            </div>
          </motion.div>
        </Container>
      </section>

      <Footer />
    </div>
  );
}
