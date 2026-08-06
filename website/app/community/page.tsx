"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useState, useEffect } from "react";
import { FiGithub, FiTwitter, FiArrowRight, FiUsers, FiZap, FiCode, FiBookOpen, FiAlertCircle, FiAward } from "react-icons/fi";
import { SiDiscord } from "react-icons/si";
import { FaLinkedin as SiLinkedin } from "react-icons/fa6";
import Link from "next/link";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { useGitHubStats } from "@/components/GitHubStats";

function useSocialStats() {
  const [stats, setStats] = useState({
    discord: { members: "...", online: "..." },
    twitter: { followers: "...", posts: "..." },
    linkedin: { followers: "...", posts: "..." },
    loading: true,
  });

  useEffect(() => {
    const fetchDiscordStats = async () => {
      try {
        const res = await fetch("https://discord.com/api/v9/invites/jacn9ed3?with_counts=true");
        if (res.ok) {
          const data = await res.json();
          return { members: data.approximate_member_count || 0, online: data.approximate_presence_count || 0 };
        }
      } catch {}
      return { members: 0, online: 0 };
    };

    const load = async () => {
      const discord = await fetchDiscordStats();
      const fmt = (n: number) => (n >= 1000 ? (n / 1000).toFixed(1) + "k" : n.toString());
      setStats({
        discord: { members: discord.members > 0 ? fmt(discord.members) : "New", online: discord.online > 0 ? fmt(discord.online) : "Join!" },
        twitter: { followers: "New", posts: "Active" },
        linkedin: { followers: "New", posts: "Active" },
        loading: false,
      });
    };
    load();
  }, []);
  return stats;
}

const SECTION_DIVIDER = (
  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
);

export default function CommunityPage() {
  const githubStats = useGitHubStats();
  const socialStats = useSocialStats();
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });

  const fmt = (n: number) => (n >= 1000 ? (n / 1000).toFixed(1) + "k" : n.toString());

  const platforms = [
    {
      name: "GitHub",
      icon: FiGithub,
      description: "Star the repo, contribute code, report issues, and stay updated with the latest releases.",
      stats: [
        { label: "Stars", value: githubStats.loading ? "..." : fmt(githubStats.stars) },
        { label: "Contributors", value: githubStats.loading ? "..." : githubStats.contributors.toString() },
      ],
      link: "https://github.com/ctrl-gaurav/effGen",
      linkText: "Visit GitHub",
      accent: "#e8eaed",
    },
    {
      name: "LinkedIn",
      icon: SiLinkedin,
      description: "Connect with us professionally, follow company updates, and network with the team.",
      stats: [
        { label: "Followers", value: socialStats.loading ? "..." : socialStats.linkedin.followers },
        { label: "Status", value: socialStats.loading ? "..." : socialStats.linkedin.posts },
      ],
      link: "https://www.linkedin.com/company/111341317/",
      linkText: "Connect",
      accent: "#0a66c2",
    },
    {
      name: "Twitter/X",
      icon: FiTwitter,
      description: "Follow us for the latest updates, announcements, tips, and community highlights.",
      stats: [
        { label: "Followers", value: socialStats.loading ? "..." : socialStats.twitter.followers },
        { label: "Status", value: socialStats.loading ? "..." : socialStats.twitter.posts },
      ],
      link: "https://x.com/effGen_org",
      linkText: "Follow Us",
      accent: "#1d9bf0",
    },
    {
      name: "Discord",
      icon: SiDiscord,
      description: "Join our active community, get help, share projects, and discuss the future of AI agents.",
      stats: [
        { label: "Members", value: socialStats.loading ? "..." : socialStats.discord.members },
        { label: "Online", value: socialStats.loading ? "..." : socialStats.discord.online },
      ],
      link: "https://discord.com/invite/jacn9ed3",
      linkText: "Join Discord",
      accent: "#5865f2",
    },
  ];

  const contributions = [
    { icon: FiCode, emoji: "💻", title: "Code Contributions", desc: "Add features, fix bugs, improve performance, or enhance documentation.", accent: "#00ff88" },
    { icon: FiBookOpen, emoji: "📚", title: "Documentation", desc: "Improve guides, write tutorials, or translate documentation.", accent: "#00e5ff" },
    { icon: FiAward, emoji: "🎨", title: "Examples & Demos", desc: "Create showcases, build integrations, or share use cases.", accent: "#a78bfa" },
    { icon: FiAlertCircle, emoji: "🐛", title: "Issue Reports", desc: "Report bugs, suggest features, or provide feedback.", accent: "#ff6b6b" },
  ];

  const guidelines = [
    { num: "01", title: "Be Respectful", desc: "Treat everyone with respect. Harassment, trolling, and personal attacks are not tolerated." },
    { num: "02", title: "Be Helpful", desc: "Help others learn and grow. Share knowledge, provide constructive feedback, and support newcomers." },
    { num: "03", title: "Stay On Topic", desc: "Keep discussions relevant to effGen and AI development. Off-topic content may be removed." },
    { num: "04", title: "Quality Over Quantity", desc: "Focus on quality contributions. Well-researched, thoughtful posts are more valuable than many low-effort ones." },
    { num: "05", title: "Give Credit", desc: "Always credit original authors and sources. Plagiarism is not acceptable." },
  ];

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
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-600 dark:text-green-400 text-sm font-semibold mb-8"
              whileHover={{ borderColor: "rgba(0,255,136,0.6)" }}
            >
              <FiUsers size={14} />
              Community
            </motion.div>

            <h1 className="text-5xl md:text-6xl lg:text-7xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
              Join Our Growing{" "}
              <span className="gradient-text neon-text">Developer Community</span>
            </h1>

            <p className="text-xl text-gray-600 dark:text-gray-400 leading-relaxed">
              Connect with thousands of developers building the future of AI agents
            </p>
          </motion.div>
        </Container>
      </section>

      {/* Community Platforms */}
      <section className="py-20 relative">
        {SECTION_DIVIDER}
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <Container className="relative z-10">
          <motion.div
            ref={ref}
            initial="hidden"
            animate={inView ? "visible" : "hidden"}
            variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.1 } } }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5"
          >
            {platforms.map((platform, idx) => (
              <motion.div
                key={platform.name}
                variants={{ hidden: { opacity: 0, y: 30 }, visible: { opacity: 1, y: 0 } }}
                whileHover={{ y: -8, scale: 1.02 }}
                transition={{ duration: 0.3 }}
                className="group relative overflow-hidden p-6 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none backdrop-blur-sm"
              >
                {/* Animated rotating border */}
                <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none overflow-hidden">
                  <motion.div
                    className="absolute inset-[-50%] w-[200%] h-[200%]"
                    style={{ background: `conic-gradient(from 0deg, transparent 60%, ${platform.accent}25 80%, transparent 100%)` }}
                    animate={{ rotate: 360 }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                  />
                  <div className="absolute inset-[1px] rounded-2xl bg-white dark:bg-gray-900/70" />
                </div>

                {/* Hover glow */}
                <motion.div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                  style={{ background: `radial-gradient(circle at 30% 30%, ${platform.accent}10 0%, transparent 70%)` }}
                />
                <motion.div
                  className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ boxShadow: `inset 0 0 0 1px ${platform.accent}30` }}
                />

                {/* Icon */}
                <motion.div
                  className="w-14 h-14 rounded-xl flex items-center justify-center mb-5 relative z-10"
                  style={{ background: `${platform.accent}15`, border: `1px solid ${platform.accent}25` }}
                  whileHover={{ rotate: 10, scale: 1.1 }}
                >
                  <platform.icon style={{ color: platform.accent }} size={24} />
                </motion.div>

                {/* Name with pulsing status dot */}
                <div className="flex items-center gap-2 mb-2 relative z-10">
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">{platform.name}</h3>
                  <motion.div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ background: platform.accent, boxShadow: `0 0 8px ${platform.accent}` }}
                    animate={{ scale: [1, 1.4, 1], opacity: [1, 0.5, 1] }}
                    transition={{ duration: 2, repeat: Infinity, delay: idx * 0.3 }}
                  />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-500 mb-5 leading-relaxed relative z-10">{platform.description}</p>

                <div className="grid grid-cols-2 gap-3 mb-5 relative z-10">
                  {platform.stats.map((s, si) => (
                    <div key={si} className="text-center p-3 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800">
                      <div className="text-xl font-black" style={{ color: platform.accent }}>{s.value}</div>
                      <div className="text-[10px] text-gray-600 font-medium uppercase tracking-wide">{s.label}</div>
                    </div>
                  ))}
                </div>

                <motion.a
                  href={platform.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between p-3 rounded-xl font-semibold text-sm relative z-10 transition-all"
                  style={{ background: `${platform.accent}15`, border: `1px solid ${platform.accent}30`, color: platform.accent }}
                  whileHover={{ background: `${platform.accent}25` } as any}
                  whileTap={{ scale: 0.97 }}
                >
                  <span>{platform.linkText}</span>
                  <FiArrowRight size={14} />
                </motion.a>

                <div
                  className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: `linear-gradient(90deg, transparent, ${platform.accent}60, transparent)` }}
                />
              </motion.div>
            ))}
          </motion.div>
        </Container>
      </section>

      {/* Contribute */}
      <section className="py-20 relative">
        {SECTION_DIVIDER}
        <div className="absolute inset-0 grid-pattern" />
        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="max-w-4xl mx-auto text-center"
          >
            <motion.div
              className="text-5xl mb-6 inline-block"
              animate={{ y: [0, -8, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              🚀
            </motion.div>

            <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
              <span className="gradient-text">Contribute to effGen</span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 mb-12 leading-relaxed max-w-2xl mx-auto">
              We welcome contributions from developers of all skill levels. Code, docs, examples, or bug reports — every contribution helps!
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-12">
              {contributions.map((c, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: i * 0.1 }}
                  viewport={{ once: true }}
                  whileHover={{ y: -6, scale: 1.02 }}
                  className="group relative p-6 rounded-2xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 shadow-sm dark:shadow-none text-left overflow-hidden"
                >
                  {/* Animated rotating border */}
                  <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none overflow-hidden">
                    <motion.div
                      className="absolute inset-[-50%] w-[200%] h-[200%]"
                      style={{ background: `conic-gradient(from 0deg, transparent 60%, ${c.accent}25 80%, transparent 100%)` }}
                      animate={{ rotate: 360 }}
                      transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    />
                    <div className="absolute inset-[1px] rounded-2xl bg-white dark:bg-gray-900/70" />
                  </div>

                  <motion.div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
                    style={{ background: `radial-gradient(circle at 20% 50%, ${c.accent}08 0%, transparent 70%)` }}
                  />
                  <motion.div
                    className="w-11 h-11 rounded-xl flex items-center justify-center mb-4 text-xl relative z-10"
                    style={{ background: `${c.accent}15`, border: `1px solid ${c.accent}25` }}
                    whileHover={{ rotate: 360 }}
                    transition={{ duration: 0.5 }}
                  >
                    {c.emoji}
                  </motion.div>
                  <h3 className="text-base font-bold mb-2 text-gray-900 dark:text-white relative z-10">{c.title}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-500 relative z-10">{c.desc}</p>
                  <div
                    className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ background: `linear-gradient(90deg, transparent, ${c.accent}50, transparent)` }}
                  />
                </motion.div>
              ))}
            </div>

            <div className="flex flex-wrap gap-4 justify-center">
              <motion.a
                href="https://github.com/ctrl-gaurav/effGen/blob/main/CONTRIBUTING.md"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
                style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)", boxShadow: "0 0 25px rgba(0,255,136,0.3)" }}
                whileHover={{ scale: 1.05, boxShadow: "0 0 40px rgba(0,255,136,0.5)" }}
                whileTap={{ scale: 0.95 }}
              >
                Contributing Guide
                <FiArrowRight size={16} />
              </motion.a>
              <motion.a
                href="https://github.com/ctrl-gaurav/effGen/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-900/60 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                View Issues
              </motion.a>
            </div>
          </motion.div>
        </Container>
      </section>

      {/* Community Guidelines */}
      <section className="py-20 relative">
        {SECTION_DIVIDER}
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="max-w-3xl mx-auto"
          >
            <h2 className="text-4xl font-black mb-10 text-center text-gray-900 dark:text-white">Community Guidelines</h2>

            <div className="rounded-2xl bg-white dark:bg-gray-900/80 border border-gray-200 dark:border-green-500/20 shadow-sm dark:shadow-none p-8 space-y-6" style={{ boxShadow: "0 0 40px rgba(0,255,136,0.04)" }}>
              {guidelines.map((g, i) => (
                <motion.div
                  key={i}
                  className="flex items-start gap-4"
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: i * 0.08 }}
                  viewport={{ once: true }}
                >
                  <motion.div
                    className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 font-black text-sm font-mono"
                    style={{ background: "rgba(0,255,136,0.1)", border: "1px solid rgba(0,255,136,0.25)", color: "#00ff88" }}
                    whileHover={{ scale: 1.15, boxShadow: "0 0 20px rgba(0,255,136,0.3)" }}
                    transition={{ duration: 0.3 }}
                  >
                    {g.num}
                  </motion.div>
                  <div>
                    <h3 className="text-base font-bold mb-1 text-gray-900 dark:text-white">{g.title}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-500 leading-relaxed">{g.desc}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </Container>
      </section>

      {/* Join CTA */}
      <section className="py-20 relative">
        {SECTION_DIVIDER}
        <div className="absolute inset-0 grid-pattern" />
        <Container className="relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="relative overflow-hidden p-10 lg:p-14 rounded-2xl bg-gray-50 dark:bg-gray-950/90 border border-gray-200 dark:border-green-500/20 shadow-sm dark:shadow-none text-center max-w-3xl mx-auto"
            style={{ boxShadow: "0 0 60px rgba(0,255,136,0.06)" }}
          >
            <motion.div
              className="absolute top-0 left-0 right-0 h-px"
              style={{ background: "linear-gradient(90deg, transparent, #00ff88, transparent)" }}
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ duration: 3, repeat: Infinity }}
            />

            {/* Floating geometric shapes */}
            <motion.div
              className="absolute top-8 left-8 w-16 h-16 border border-green-500/10 rounded-lg pointer-events-none"
              animate={{ rotate: 360, scale: [1, 1.1, 1] }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            />
            <motion.div
              className="absolute bottom-8 right-8 w-12 h-12 border border-cyan-500/10 rounded-full pointer-events-none"
              animate={{ rotate: -360 }}
              transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
            />
            <motion.div
              className="absolute top-1/2 right-16 w-10 h-10 border border-violet-500/10 pointer-events-none"
              style={{ clipPath: "polygon(50% 0%, 100% 100%, 0% 100%)" }}
              animate={{ rotate: 360 }}
              transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
            />

            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="inline-block mb-6 relative z-10"
            >
              <FiUsers className="mx-auto text-green-400" size={48} />
            </motion.div>
            <h2 className="text-3xl md:text-4xl font-black mb-4 text-gray-900 dark:text-white relative z-10">
              <span className="gradient-text">Ready to Join?</span>
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-xl mx-auto relative z-10">
              Become part of a thriving community of developers pushing the boundaries of AI agents.
            </p>
            <div className="flex flex-wrap gap-4 justify-center relative z-10">
              <motion.a
                href="https://discord.com/invite/jacn9ed3"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black"
                style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)", boxShadow: "0 0 25px rgba(0,255,136,0.3)" }}
                whileHover={{ scale: 1.05, boxShadow: "0 0 40px rgba(0,255,136,0.5)" }}
                whileTap={{ scale: 0.95 }}
              >
                <SiDiscord size={16} />
                Join Discord
                <FiArrowRight size={14} />
              </motion.a>
              <motion.a
                href="https://github.com/ctrl-gaurav/effGen"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-900/60 hover:border-green-500/50 hover:text-green-600 dark:hover:text-green-400 transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <FiGithub size={16} />
                Star on GitHub
              </motion.a>
            </div>
          </motion.div>
        </Container>
      </section>

      <Footer />
    </div>
  );
}
