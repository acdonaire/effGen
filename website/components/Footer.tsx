"use client";

import { motion } from "framer-motion";
import { FiGithub, FiTwitter, FiHeart, FiZap, FiArrowUp } from "react-icons/fi";
import { SiDiscord } from "react-icons/si";
import { FaLinkedin as SiLinkedin } from "react-icons/fa6";
import HelpBot from "./HelpBot";
import { useState, useEffect } from "react";
import { withBasePath } from "./basePath";

// Brand color glows for social icons
const socialBrandColors: Record<string, string> = {
  GitHub: "#e8eaed",
  LinkedIn: "#0a66c2",
  Twitter: "#1d9bf0",
  Discord: "#5865f2",
};

export default function Footer() {
  const [showTop, setShowTop] = useState(false);

  useEffect(() => {
    const handleScroll = () => setShowTop(window.scrollY > 400);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const footerLinks = {
    Product: [
      { name: "Features", href: "#features" },
      { name: "Examples", href: "/examples" },
      { name: "Documentation", href: "/docs", external: false },
      { name: "GitHub", href: "https://github.com/ctrl-gaurav/effGen", external: true },
    ],
    Resources: [
      { name: "API Reference", href: "/docs/api-reference", external: false },
      { name: "PyPI", href: "https://pypi.org/project/effgen/", external: true },
      { name: "arXiv Paper", href: "https://arxiv.org/abs/2602.00887", external: true },
      { name: "Support", href: "https://github.com/ctrl-gaurav/effGen/issues", external: true },
    ],
    Company: [
      { name: "About", href: "/community" },
      { name: "Contributing", href: "https://github.com/ctrl-gaurav/effGen", external: true },
      { name: "License", href: "https://github.com/ctrl-gaurav/effGen/blob/main/LICENSE", external: true },
      { name: "Changelog", href: "https://github.com/ctrl-gaurav/effGen/blob/main/NEWS.md", external: true },
    ],
  };

  const socialLinks = [
    { icon: FiGithub, href: "https://github.com/ctrl-gaurav/effGen", label: "GitHub" },
    { icon: SiLinkedin, href: "https://www.linkedin.com/company/111341317/", label: "LinkedIn" },
    { icon: FiTwitter, href: "https://x.com/effGen_org", label: "Twitter" },
    { icon: SiDiscord, href: "https://discord.com/invite/jacn9ed3", label: "Discord" },
  ];

  return (
    <footer className="relative bg-gray-50 dark:bg-[#010a06] border-t border-gray-200 dark:border-green-500/10 overflow-hidden">
      {/* Animated flowing gradient divider */}
      <motion.div
        className="absolute top-0 left-0 right-0 h-px"
        animate={{
          background: [
            "linear-gradient(90deg, transparent 0%, #00ff88 30%, #00e5ff 50%, #00ff88 70%, transparent 100%)",
            "linear-gradient(90deg, transparent 0%, #00e5ff 30%, #00ff88 50%, #00e5ff 70%, transparent 100%)",
            "linear-gradient(90deg, transparent 0%, #00ff88 30%, #00e5ff 50%, #00ff88 70%, transparent 100%)",
          ],
        }}
        transition={{ duration: 4, repeat: Infinity }}
      />

      {/* Grid overlay */}
      <div className="absolute inset-0 grid-pattern opacity-30" />

      {/* Glow */}
      <motion.div
        className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[600px] h-[200px] rounded-full opacity-[0.03] pointer-events-none"
        style={{ background: "radial-gradient(ellipse, #00ff88 0%, transparent 70%)" }}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Main grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-8 mb-12">
          {/* Brand */}
          <div className="lg:col-span-2">
            <motion.div className="flex items-center gap-3 mb-5" whileHover={{ scale: 1.03 }}>
              <div className="relative">
                <motion.div
                  className="absolute inset-0 rounded-full bg-green-400/20 blur-md"
                  animate={{ scale: [1, 1.4, 1], opacity: [0.4, 0.7, 0.4] }}
                  transition={{ duration: 2.5, repeat: Infinity }}
                />
                <img src={withBasePath("/favicon.svg")} alt="effGen logo" className="w-9 h-9 relative z-10" />
              </div>
              <div>
                <span className="text-xl font-black">
                  <span className="gradient-text">eff</span>
                  <span className="text-gray-900 dark:text-white">Gen</span>
                </span>
              </div>
            </motion.div>
            <p className="text-gray-600 dark:text-gray-500 text-sm mb-6 max-w-xs leading-relaxed">
              Build powerful AI agents with Small Language Models. Production-ready framework optimized for speed and efficiency.
            </p>

            {/* Terminal snippet */}
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gray-100 dark:bg-[#0a1a0f] border border-gray-300 dark:border-green-500/20 font-mono text-xs mb-6 w-fit">
              <FiZap className="text-green-600 dark:text-green-400 flex-shrink-0" size={12} />
              <span className="text-green-600 dark:text-green-400">$</span>
              <span className="text-gray-700 dark:text-gray-300">pip install effgen</span>
            </div>

            {/* Social Links with brand-colored glows */}
            <div className="flex gap-3">
              {socialLinks.map((social, index) => {
                const brandColor = socialBrandColors[social.label] || "#00ff88";
                return (
                  <motion.a
                    key={index}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={social.label}
                    className="w-9 h-9 rounded-lg bg-gray-100 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 flex items-center justify-center text-gray-500 hover:text-green-600 dark:hover:text-green-400 hover:border-green-500/40 hover:bg-green-500/5 transition-all"
                    whileHover={{
                      scale: 1.1,
                      y: -2,
                      boxShadow: `0 0 15px ${brandColor}40`,
                    }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <social.icon size={16} />
                  </motion.a>
                );
              })}
            </div>
          </div>

          {/* Links with animated underlines */}
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h3 className="text-xs font-bold text-green-600 dark:text-green-400 uppercase tracking-widest mb-4">{category}</h3>
              <ul className="space-y-2.5">
                {links.map((link, index) => (
                  <li key={index}>
                    <motion.a
                      href={withBasePath(link.href)}
                      target={link.external ? "_blank" : undefined}
                      rel={link.external ? "noopener noreferrer" : undefined}
                      className="text-sm text-gray-600 dark:text-gray-500 hover:text-green-600 dark:hover:text-green-400 transition-colors hover-underline inline-block"
                      whileHover={{ x: 4 }}
                    >
                      {link.name}
                    </motion.a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom with animated gradient divider */}
        <div className="pt-8 border-t border-gray-200 dark:border-gray-900 relative">
          <motion.div
            className="absolute top-0 left-0 right-0 h-px"
            style={{
              background: "linear-gradient(90deg, transparent, rgba(0,255,136,0.2), transparent)",
            }}
          />
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-xs text-gray-600">
              &copy; 2026 effGen. Released under the Apache 2.0 License.
            </p>

            <motion.div className="flex items-center gap-1 text-xs text-gray-600" whileHover={{ scale: 1.05 }}>
              Made with
              <motion.div animate={{ scale: [1, 1.3, 1] }} transition={{ duration: 1, repeat: Infinity }}>
                <FiHeart className="text-green-500 mx-1" size={12} />
              </motion.div>
              by the effGen team
            </motion.div>

            <div className="flex items-center gap-4 text-xs text-gray-600">
              {[
                { name: "GitHub", href: "https://github.com/ctrl-gaurav/effGen" },
                { name: "License", href: "https://github.com/ctrl-gaurav/effGen/blob/main/LICENSE" },
                { name: "Discord", href: "https://discord.com/invite/jacn9ed3" },
              ].map((item) => (
                <a
                  key={item.name}
                  href={withBasePath(item.href)}
                  className="hover:text-green-600 dark:hover:text-green-400 transition-colors hover-underline"
                >
                  {item.name}
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Help Bot */}
      <HelpBot />

      {/* Scroll to top */}
      <motion.a
        href="#home"
        className="fixed bottom-8 right-8 w-11 h-11 rounded-full flex items-center justify-center text-black z-40"
        style={{
          background: "linear-gradient(135deg, #00ff88, #00c96e)",
          boxShadow: "0 0 20px rgba(0,255,136,0.4)",
        }}
        whileHover={{ scale: 1.1, y: -2, boxShadow: "0 0 30px rgba(0,255,136,0.6)" }}
        whileTap={{ scale: 0.95 }}
        initial={{ opacity: 0, scale: 0 }}
        animate={{ opacity: showTop ? 1 : 0, scale: showTop ? 1 : 0 }}
        transition={{ duration: 0.3 }}
      >
        <FiArrowUp size={18} />
      </motion.a>
    </footer>
  );
}
