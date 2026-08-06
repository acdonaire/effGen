"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useTheme } from "next-themes";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { FiSun, FiMoon, FiMenu, FiX, FiGithub, FiZap } from "react-icons/fi";
import { usePyPIVersion } from "./PyPIVersion";

export default function Navbar() {
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState("");
  const { theme, setTheme } = useTheme();
  const pathname = usePathname();
  const pypiVersion = usePyPIVersion();

  useEffect(() => {
    setMounted(true);
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
      // Track active section
      const sections = ["home", "features", "quickstart", "examples", "community"];
      for (const id of sections.reverse()) {
        const el = document.getElementById(id);
        if (el && window.scrollY >= el.offsetTop - 200) {
          setActiveSection(id);
          break;
        }
      }
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const isHomePage = pathname === "/";

  const navLinks = [
    { name: "Home", href: "/", isPage: true, isExternal: false, sectionId: "home" },
    { name: "Features", href: isHomePage ? "#features" : "/#features", isPage: false, isExternal: false, sectionId: "features" },
    { name: "Quick Start", href: isHomePage ? "#quickstart" : "/#quickstart", isPage: false, isExternal: false, sectionId: "quickstart" },
    { name: "Examples", href: "/examples", isPage: true, isExternal: false, sectionId: "examples" },
    { name: "Leaderboard", href: "/leaderboard", isPage: true, isExternal: false, sectionId: "" },
    { name: "Community", href: "/community", isPage: true, isExternal: false, sectionId: "community" },
    { name: "Docs", href: "/docs", isPage: true, isExternal: false, sectionId: "" },
  ];

  const isActive = (link: typeof navLinks[0]) => {
    if (link.isPage && pathname === link.href) return true;
    if (!link.isPage && isHomePage && activeSection === link.sectionId) return true;
    return false;
  };

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled
          ? "bg-white/80 dark:bg-black/40 backdrop-blur-[20px] border-b border-gray-200/60 dark:border-green-500/15 shadow-lg dark:shadow-[0_0_30px_rgba(0,255,136,0.06)]"
          : "bg-transparent"
      }`}
    >
      {/* Top accent line */}
      <motion.div
        className="absolute top-0 left-0 right-0 h-[1px]"
        style={{
          background: "linear-gradient(90deg, transparent, #00ff88, #00c96e, transparent)",
          opacity: scrolled ? 1 : 0,
          transition: "opacity 0.3s",
        }}
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 md:h-20">

          {/* Logo with enhanced glow */}
          <Link href="/" className="flex items-center space-x-3 group">
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="flex items-center space-x-3"
            >
              <div className="relative">
                <motion.div
                  className="absolute inset-0 rounded-full bg-green-400/30 blur-md"
                  animate={{
                    scale: [1, 1.4, 1],
                    opacity: [0.4, 0.8, 0.4],
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
                <motion.div
                  className="absolute inset-[-4px] rounded-full bg-green-400/10 blur-lg"
                  animate={{
                    scale: [1.2, 1.6, 1.2],
                    opacity: [0.2, 0.5, 0.2],
                  }}
                  transition={{ duration: 3, repeat: Infinity }}
                />
                <img src="/favicon.svg" alt="effGen logo" className="w-9 h-9 relative z-10" />
              </div>
              <span className="text-xl font-black tracking-tight">
                <span className="gradient-text">eff</span>
                <span className="text-gray-900 dark:text-white">Gen</span>
              </span>
              <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/30 font-mono">
                <FiZap size={10} />
                v{pypiVersion.loading ? "..." : pypiVersion.version}
              </span>
            </motion.div>
          </Link>

          {/* Desktop Navigation with active indicator */}
          <div className="hidden md:flex items-center space-x-1">
            {navLinks.map((link, index) => {
              const active = isActive(link);
              return link.isExternal ? (
                <motion.a
                  key={link.name}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="relative px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 transition-colors font-medium rounded-lg hover:bg-green-500/5"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.07 }}
                  whileHover={{ y: -1 }}
                >
                  {link.name}
                </motion.a>
              ) : link.isPage ? (
                <Link key={link.name} href={link.href}>
                  <motion.span
                    className={`relative px-3 py-2 text-sm font-medium rounded-lg cursor-pointer transition-colors block ${
                      active
                        ? "text-green-600 dark:text-green-400 bg-green-500/10"
                        : "text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 hover:bg-green-500/5"
                    }`}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.07 }}
                    whileHover={{ y: -1 }}
                  >
                    {link.name}
                    {/* Active dot indicator */}
                    {active && (
                      <motion.div
                        className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-green-400"
                        layoutId="nav-dot"
                        style={{ boxShadow: "0 0 6px rgba(0,255,136,0.8)" }}
                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                      />
                    )}
                  </motion.span>
                </Link>
              ) : (
                <motion.a
                  key={link.name}
                  href={link.href}
                  className={`relative px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                    active
                      ? "text-green-600 dark:text-green-400 bg-green-500/10"
                      : "text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 hover:bg-green-500/5"
                  }`}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.07 }}
                  whileHover={{ y: -1 }}
                >
                  {link.name}
                  {/* Active dot indicator */}
                  {active && (
                    <motion.div
                      className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-green-400"
                      layoutId="nav-dot"
                      style={{ boxShadow: "0 0 6px rgba(0,255,136,0.8)" }}
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                  )}
                </motion.a>
              );
            })}
          </div>

          {/* Right side */}
          <div className="hidden md:flex items-center gap-3">
            <motion.a
              href="https://github.com/ctrl-gaurav/effGen"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 hover:bg-green-500/10 transition-all"
              whileHover={{ scale: 1.1, rotate: 15 }}
              transition={{ duration: 0.2 }}
            >
              <FiGithub size={18} />
            </motion.a>

            {mounted && (
              <motion.button
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                className="p-2 rounded-lg bg-green-500/10 border border-green-500/20 text-green-600 dark:text-green-400 hover:bg-green-500/20 transition-all"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                {theme === "dark" ? <FiSun size={16} /> : <FiMoon size={16} />}
              </motion.button>
            )}

            <Link href={isHomePage ? "#quickstart" : "/#quickstart"}>
              <motion.span
                className="relative inline-flex items-center gap-2 px-5 py-2 rounded-full text-sm font-bold cursor-pointer overflow-hidden group"
                style={{
                  background: "linear-gradient(135deg, #00ff88, #00c96e)",
                  color: "#000",
                  boxShadow: "0 0 20px rgba(0,255,136,0.3)",
                }}
                whileHover={{
                  scale: 1.05,
                  boxShadow: "0 0 30px rgba(0,255,136,0.5)",
                }}
                whileTap={{ scale: 0.95 }}
              >
                <span className="absolute inset-0 bg-white/0 group-hover:bg-white/10 transition-colors" />
                <FiZap size={14} />
                Get Started
              </motion.span>
            </Link>
          </div>

          {/* Mobile */}
          <div className="flex md:hidden items-center gap-2">
            {mounted && (
              <motion.button
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                className="p-2 rounded-lg bg-green-500/10 border border-green-500/20 text-green-600 dark:text-green-400"
                whileTap={{ scale: 0.9 }}
              >
                {theme === "dark" ? <FiSun size={16} /> : <FiMoon size={16} />}
              </motion.button>
            )}
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300"
            >
              {isOpen ? <FiX size={20} /> : <FiMenu size={20} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-white/95 dark:bg-black/95 backdrop-blur-xl border-t border-gray-200 dark:border-green-500/20"
          >
            <div className="px-4 py-6 space-y-1">
              {navLinks.map((link) => (
                link.isPage ? (
                  <Link key={link.name} href={link.href} onClick={() => setIsOpen(false)}>
                    <motion.span
                      className={`flex items-center px-4 py-3 rounded-xl font-medium cursor-pointer transition-colors ${
                        pathname === link.href
                          ? "text-green-600 dark:text-green-400 bg-green-500/10"
                          : "text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 hover:bg-green-500/5"
                      }`}
                      whileHover={{ x: 6 }}
                    >
                      {link.name}
                    </motion.span>
                  </Link>
                ) : (
                  <motion.a
                    key={link.name}
                    href={link.href}
                    className="flex items-center px-4 py-3 rounded-xl text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 hover:bg-green-500/5 font-medium"
                    onClick={() => setIsOpen(false)}
                    whileHover={{ x: 6 }}
                  >
                    {link.name}
                  </motion.a>
                )
              ))}
              <div className="pt-2">
                <Link href={isHomePage ? "#quickstart" : "/#quickstart"} onClick={() => setIsOpen(false)}>
                  <motion.span
                    className="flex items-center justify-center gap-2 w-full px-6 py-3 rounded-full font-bold text-black cursor-pointer"
                    style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)" }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <FiZap size={14} />
                    Get Started
                  </motion.span>
                </Link>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}
