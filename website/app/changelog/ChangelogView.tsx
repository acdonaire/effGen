"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { FiAlertTriangle, FiArrowRight, FiExternalLink, FiGitCommit, FiTag } from "react-icons/fi";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import CodeSample from "@/components/ui/CodeSample";
import RouteLink from "@/components/ui/RouteLink";
import { siteData, version } from "@/components/siteData";
import {
  COMMITS_SINCE_0_3_2,
  RELEASE_DATE,
  breakingChanges,
  earlierReleases,
  newPublicNames,
  oneZeroGroups,
} from "./changelogData";
import { accentTextStyle } from "@/components/accentText";

const CHANGELOG_URL = "https://github.com/ctrl-gaurav/effGen/blob/main/CHANGELOG.md";
const RELEASES_URL = "https://github.com/ctrl-gaurav/effGen/releases";

const SECTION_DIVIDER = (
  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
);

/* ── One themed group of changes ── */

function Group({ group, index }: { group: (typeof oneZeroGroups)[number]; index: number }) {
  const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <motion.section
      ref={ref}
      id={group.id}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5 }}
      className="scroll-mt-28"
      aria-labelledby={`${group.id}-heading`}
    >
      <div className="flex items-baseline gap-3 mb-2">
        <span
          className="text-xs font-mono font-bold tabular-nums"
          style={accentTextStyle(group.accent)}
        >
          {String(index + 1).padStart(2, "0")}
        </span>
        <h3
          id={`${group.id}-heading`}
          className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white"
        >
          {group.title}
        </h3>
      </div>
      <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-3xl">{group.lede}</p>

      <div className="space-y-6">
        {group.items.map((item) => (
          <article
            key={item.title}
            className="relative rounded-2xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-6 shadow-sm dark:shadow-none"
          >
            <div
              className="absolute left-0 top-6 bottom-6 w-0.5 rounded-full"
              style={{ background: group.accent }}
            />
            <h4 className="text-base font-bold text-gray-900 dark:text-white mb-2 pl-3">
              {item.title}
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed pl-3">
              {item.body}
            </p>
            {item.code && (
              <div className="mt-4 pl-3">
                <CodeSample
                  code={item.code.source}
                  language={item.code.language ?? "python"}
                  accent={group.accent}
                  output={item.code.output}
                />
              </div>
            )}
          </article>
        ))}
      </div>
    </motion.section>
  );
}

export default function ChangelogView() {
  const [heroRef, heroInView] = useInView({ triggerOnce: true, threshold: 0.05 });

  const headline = [
    { value: version, label: "Version", accent: "#00ff88", icon: FiTag },
    { value: RELEASE_DATE.replace(" 2026", ""), label: "Released", accent: "#00e5ff", icon: FiTag },
    { value: String(COMMITS_SINCE_0_3_2), label: "Commits since 0.3.2", accent: "#a78bfa", icon: FiGitCommit },
    { value: "3", label: "Breaking changes", accent: "#ff9500", icon: FiAlertTriangle },
  ];

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />
      <main id="main">
        {/* Hero */}
        <section className="relative pt-32 pb-16 overflow-hidden">
          <div className="absolute inset-0 grid-pattern" />
          <Container className="relative z-10">
            <motion.div
              ref={heroRef}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              className="max-w-4xl mx-auto text-center"
            >
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-8">
                <FiTag size={14} />
                Changelog
              </span>
              <h1 className="text-5xl md:text-6xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                <span className="gradient-text">effGen {version}</span> — the first stable release
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl mx-auto">
                Released {RELEASE_DATE}, {COMMITS_SINCE_0_3_2} commits after 0.3.2. The theme running
                through it is control over where a model runs and visibility into what a run did:
                drive a server you already operate, read back the calls a run made, wrap the agent
                loop in middleware, hand one agent many conversations, choose how history is
                compacted, and resume a workflow that died half way through. The public surface grew
                from 204 names to {siteData.public_names}, and nothing was removed or renamed.
              </p>
            </motion.div>
          </Container>
        </section>

        {/* Headline figures */}
        <section className="py-6 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {headline.map((stat, idx) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={heroInView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: idx * 0.08 }}
                  className="rounded-2xl p-5 bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 text-center"
                >
                  <stat.icon className="mx-auto mb-2" style={accentTextStyle(stat.accent)} size={20} />
                  <div className="text-2xl font-black mb-0.5" style={accentTextStyle(stat.accent)}>
                    {stat.value}
                  </div>
                  <div className="text-[10px] text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider">
                    {stat.label}
                  </div>
                </motion.div>
              ))}
            </div>
          </Container>
        </section>

        {/* The three breaking changes */}
        <section id="breaking" className="py-16 relative scroll-mt-24">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-orange-500/25 bg-orange-500/[0.04] p-6 lg:p-10">
              <div className="flex items-center gap-3 mb-2">
                <FiAlertTriangle className="text-orange-500 dark:text-orange-400" size={22} />
                <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white">
                  Three things change when you upgrade
                </h2>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-8 max-w-3xl">
                Everything else in {version} is additive. Each of the three carries the line that
                migrates it, and what that line printed when it was run.
              </p>

              <ol className="space-y-6">
                {breakingChanges.map((change, i) => (
                  <li
                    key={change.title}
                    className="rounded-xl bg-white dark:bg-black/40 border border-gray-200 dark:border-gray-800 p-6"
                  >
                    <span className="text-[10px] font-mono uppercase tracking-widest text-orange-700 dark:text-orange-400">
                      Breaking {String(i + 1).padStart(2, "0")}
                    </span>
                    <h3 className="mt-2 text-lg font-bold text-gray-900 dark:text-white">
                      {change.title}
                    </h3>
                    <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                      {change.why}
                    </p>

                    <div className="mt-4">
                      <CodeSample
                        code={change.migration.source}
                        language={change.migration.language}
                        accent="#ff9500"
                        output={change.migration.output}
                      />
                    </div>

                    {change.note && (
                      <p className="mt-3 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                        {change.note}
                      </p>
                    )}
                  </li>
                ))}
              </ol>

              <div className="mt-8">
                <RouteLink
                  to="/docs/migration"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  The migration guide
                  <FiArrowRight size={14} />
                </RouteLink>
              </div>
            </div>
          </Container>
        </section>

        {/* What 1.0.0 added, grouped the way the changelog groups it */}
        <section className="py-16 relative bg-gray-50 dark:bg-[#030f07]">
          {SECTION_DIVIDER}
          <div className="absolute inset-0 grid-pattern opacity-50" />
          <Container className="relative z-10">
            <div className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-black text-gray-900 dark:text-white mb-4">
                What {version} <span className="gradient-text">added</span>
              </h2>
              <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
                {oneZeroGroups.length} themes, in the order the changelog files them. Jump to one:
              </p>
              <nav aria-label="Changes by theme" className="mt-6 flex flex-wrap justify-center gap-2">
                {oneZeroGroups.map((group) => (
                  <a
                    key={group.id}
                    href={`#${group.id}`}
                    className="px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors"
                    style={{
                      ...accentTextStyle(group.accent),
                      borderColor: `${group.accent}40`,
                      backgroundColor: `${group.accent}0d`,
                    }}
                  >
                    {group.title}
                  </a>
                ))}
              </nav>
            </div>

            <div className="max-w-4xl mx-auto space-y-16">
              {oneZeroGroups.map((group, index) => (
                <Group key={group.id} group={group} index={index} />
              ))}
            </div>
          </Container>
        </section>

        {/* The 19 new names */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-3">
                {newPublicNames.length} new names on the{" "}
                <code className="font-mono gradient-text">effgen</code> package
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                The top-level surface grew from 204 names to {siteData.public_names}. Nothing was
                removed and nothing was renamed. <code className="font-mono">BaseModel</code> also
                gained <code className="font-mono">build_assistant_message</code> and{" "}
                <code className="font-mono">build_tool_result_message</code>, and{" "}
                <code className="font-mono">SandboxResult</code> gained{" "}
                <code className="font-mono">credential_reads_masked</code> and{" "}
                <code className="font-mono">process_table_isolated</code>.
              </p>
              <ul className="flex flex-wrap gap-2">
                {newPublicNames.map((name) => (
                  <li
                    key={name}
                    className="px-3 py-1.5 rounded-lg text-xs font-mono bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-gray-700 dark:text-gray-300"
                  >
                    {name}
                  </li>
                ))}
              </ul>
            </div>
          </Container>
        </section>

        {/* Earlier releases */}
        <section id="earlier" className="py-16 relative bg-gray-50 dark:bg-[#030f07] scroll-mt-24">
          {SECTION_DIVIDER}
          <div className="absolute inset-0 grid-pattern opacity-50" />
          <Container className="relative z-10">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-3">
                Earlier releases
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-8">
                Every release before {version}, with the date and the headline the changelog gives
                it. The full entry for each — every addition, change and fix — is in{" "}
                <a
                  href={CHANGELOG_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-green-700 dark:text-green-400 font-semibold inline-flex items-center gap-1"
                >
                  CHANGELOG.md
                  <FiExternalLink size={12} />
                </a>
                .
              </p>

              <ol className="relative border-l border-gray-200 dark:border-gray-800 ml-3">
                {earlierReleases.map((release) => (
                  <li key={release.version} className="relative pl-8 pb-8 last:pb-0">
                    <span
                      className="absolute -left-[5px] top-1.5 w-2.5 h-2.5 rounded-full bg-gray-300 dark:bg-gray-700"
                      aria-hidden="true"
                    />
                    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                      <h3 className="text-lg font-bold text-gray-900 dark:text-white font-mono">
                        v{release.version}
                      </h3>
                      <span className="text-xs text-gray-600 dark:text-gray-400">{release.date}</span>
                      <span className="text-sm font-semibold text-green-700 dark:text-green-400">
                        {release.title}
                      </span>
                    </div>
                    <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                      {release.summary}
                    </p>
                  </li>
                ))}
              </ol>
            </div>
          </Container>
        </section>

        {/* Where to go next */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="max-w-3xl mx-auto text-center">
              <h2 className="text-2xl md:text-3xl font-black text-gray-900 dark:text-white mb-4">
                Upgrade
              </h2>
              <div className="max-w-md mx-auto text-left mb-8">
                <CodeSample code="pip install -U effgen" language="bash" />
              </div>
              <div className="flex flex-wrap gap-4 justify-center">
                <a
                  href={RELEASES_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-bold text-black"
                  style={{ background: "linear-gradient(135deg, #00ff88, #00c96e)" }}
                >
                  Releases on GitHub
                  <FiExternalLink size={14} />
                </a>
                <RouteLink
                  to="/docs/migration"
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-bold text-green-700 dark:text-green-300 border border-green-500/30 bg-green-500/5 hover:bg-green-500/10 transition-colors"
                >
                  Migrating from 0.3.x
                  <FiArrowRight size={14} />
                </RouteLink>
              </div>
            </div>
          </Container>
        </section>
      </main>
      <Footer />
    </div>
  );
}
