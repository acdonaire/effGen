"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useState, useEffect, useRef, useCallback } from "react";
import { FiGithub, FiTwitter, FiArrowRight, FiUsers } from "react-icons/fi";
import { SiDiscord } from "react-icons/si";
import { FaLinkedin as SiLinkedin } from "react-icons/fa6";
import Link from "next/link";
import Container from "./Container";
import { useGitHubStats } from "./GitHubStats";

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
      setStats({
        discord: {
          members: discord.members > 0 ? formatNum(discord.members) : "New",
          online: discord.online > 0 ? formatNum(discord.online) : "Join!",
        },
        twitter: { followers: "New", posts: "Active" },
        linkedin: { followers: "New", posts: "Active" },
        loading: false,
      });
    };
    load();
  }, []);
  return stats;
}

function formatNum(num: number): string {
  if (num >= 1000) return (num / 1000).toFixed(1) + "k";
  return num.toString();
}

// Brand color glow mapping
const brandGlows: Record<string, string> = {
  GitHub: "rgba(232,234,237,0.4)",
  LinkedIn: "rgba(10,102,194,0.4)",
  "Twitter/X": "rgba(29,155,240,0.4)",
  Discord: "rgba(88,101,242,0.4)",
};

function CommunityCard({ community, index, inView }: { community: any; index: number; inView: boolean }) {
  const cardRef = useRef<HTMLAnchorElement>(null);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    setTilt({
      x: ((y - centerY) / centerY) * -5,
      y: ((x - centerX) / centerX) * 5,
    });
  }, []);

  const handleMouseLeave = useCallback(() => {
    setTilt({ x: 0, y: 0 });
  }, []);

  return (
    <motion.a
      ref={cardRef}
      href={community.link}
      target="_blank"
      rel="noopener noreferrer"
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      whileHover={{ y: -8, scale: 1.02 }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className="group relative p-6 rounded-2xl bg-gray-50 dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 backdrop-blur-sm overflow-hidden block"
      style={{
        transform: `perspective(800px) rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)`,
        transition: "transform 0.15s ease-out",
      }}
    >
      {/* Hover glow */}
      <motion.div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
        style={{ background: `radial-gradient(circle at 30% 30%, ${community.accent}10 0%, transparent 70%)` }}
      />
      <motion.div
        className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"
        style={{ boxShadow: `inset 0 0 0 1px ${community.accent}30` }}
      />

      {/* Icon with brand-colored glow on hover */}
      <motion.div
        className="w-14 h-14 rounded-xl flex items-center justify-center mb-5 text-2xl relative z-10 transition-all duration-300"
        style={{
          background: `${community.accent}15`,
          border: `1px solid ${community.accent}25`,
        }}
        whileHover={{ rotate: 10, scale: 1.1, boxShadow: `0 0 20px ${brandGlows[community.name] || community.accent}` }}
      >
        <community.icon style={{ color: community.accent }} size={24} />
      </motion.div>

      <h3 className="text-lg font-bold mb-2 text-gray-900 dark:text-white relative z-10">{community.name}</h3>
      <p className="text-sm text-gray-600 dark:text-gray-500 mb-5 leading-relaxed relative z-10">{community.description}</p>

      <div className="grid grid-cols-2 gap-3 mb-5 relative z-10">
        {community.stats.map((stat: any, idx: number) => (
          <div key={idx} className="text-center p-3 rounded-xl bg-white dark:bg-black/40 border border-gray-200 dark:border-gray-800">
            <motion.div
              className="text-xl font-black"
              style={{ color: community.accent }}
              initial={{ opacity: 0, y: 5 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.5 + index * 0.1 + idx * 0.1 }}
            >
              {stat.value}
            </motion.div>
            <div className="text-[10px] text-gray-600 font-medium uppercase tracking-wide">{stat.label}</div>
          </div>
        ))}
      </div>

      <motion.div
        className="flex items-center gap-1.5 text-sm font-semibold relative z-10 transition-colors"
        style={{ color: community.accent }}
        animate={{ x: 0 }}
        whileHover={{ x: 4 }}
      >
        Visit {community.name}
        <FiArrowRight size={14} />
      </motion.div>

      {/* Bottom accent */}
      <div
        className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
        style={{ background: `linear-gradient(90deg, transparent, ${community.accent}60, transparent)` }}
      />
    </motion.a>
  );
}

export default function Community() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const githubStats = useGitHubStats();
  const socialStats = useSocialStats();

  const communities = [
    {
      icon: FiGithub,
      name: "GitHub",
      description: "Star the repo, contribute code, report issues, and stay updated with the latest releases.",
      stats: [
        { label: "Stars", value: githubStats.loading ? "..." : formatNum(githubStats.stars) },
        { label: "Contributors", value: githubStats.loading ? "..." : githubStats.contributors.toString() },
      ],
      link: "https://github.com/ctrl-gaurav/effGen",
      accent: "#e8eaed",
    },
    {
      icon: SiLinkedin,
      name: "LinkedIn",
      description: "Connect with us professionally, follow company updates, and network with the team.",
      stats: [
        { label: "Followers", value: socialStats.loading ? "..." : socialStats.linkedin.followers },
        { label: "Status", value: socialStats.loading ? "..." : socialStats.linkedin.posts },
      ],
      link: "https://www.linkedin.com/company/111341317/",
      accent: "#0a66c2",
    },
    {
      icon: FiTwitter,
      name: "Twitter/X",
      description: "Follow us for the latest updates, announcements, tips, and community highlights.",
      stats: [
        { label: "Followers", value: socialStats.loading ? "..." : socialStats.twitter.followers },
        { label: "Status", value: socialStats.loading ? "..." : socialStats.twitter.posts },
      ],
      link: "https://x.com/effGen_org",
      accent: "#1d9bf0",
    },
    {
      icon: SiDiscord,
      name: "Discord",
      description: "Join our active community, get help, share projects, and discuss the future of AI agents.",
      stats: [
        { label: "Members", value: socialStats.loading ? "..." : socialStats.discord.members },
        { label: "Online", value: socialStats.loading ? "..." : socialStats.discord.online },
      ],
      link: "https://discord.com/invite/jacn9ed3",
      accent: "#5865f2",
    },
  ];

  return (
    <section id="community" className="py-24 bg-white dark:bg-[#020c08] relative overflow-hidden" ref={ref}>
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
            <FiUsers size={14} />
            Community
          </motion.span>
          <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
            Join Our Growing{" "}
            <span className="gradient-text">Developer Community</span>
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Connect with developers building the future of AI agents
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {communities.map((community, index) => (
            <CommunityCard key={index} community={community} index={index} inView={inView} />
          ))}
        </div>

        {/* CTA Card */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="relative overflow-hidden p-8 lg:p-12 rounded-2xl bg-gray-50 dark:bg-gray-900/80 border border-green-500/20 text-center"
          style={{ boxShadow: "0 0 60px rgba(0,255,136,0.05)" }}
        >
          <div className="absolute inset-0 grid-pattern opacity-50" />
          <motion.div
            className="absolute top-0 left-0 right-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, #00ff88, transparent)" }}
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 3, repeat: Infinity }}
          />
          <motion.div
            className="absolute bottom-0 left-0 right-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, #00c96e, transparent)" }}
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 3, repeat: Infinity, delay: 1.5 }}
          />

          <div className="relative z-10">
            <h3 className="text-2xl md:text-3xl font-black mb-3 text-gray-900 dark:text-white">Want to learn more about our community?</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-xl mx-auto">
              Visit our dedicated community page for resources, events, and contribution guidelines.
            </p>
            <Link href="/community">
              <motion.span
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full font-bold text-black cursor-pointer"
                style={{
                  background: "linear-gradient(135deg, #00ff88, #00c96e)",
                  boxShadow: "0 0 30px rgba(0,255,136,0.3)",
                }}
                whileHover={{ scale: 1.05, boxShadow: "0 0 50px rgba(0,255,136,0.5)" }}
                whileTap={{ scale: 0.95 }}
              >
                Explore Community
                <FiArrowRight size={16} />
              </motion.span>
            </Link>
          </div>
        </motion.div>
      </Container>
    </section>
  );
}
