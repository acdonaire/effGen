"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FiStar, FiGitBranch, FiUsers, FiDownload } from "react-icons/fi";

interface GitHubStatsProps {
  className?: string;
  showForks?: boolean;
  showContributors?: boolean;
}

interface Stats {
  stars: number;
  forks: number;
  contributors: number;
  loading: boolean;
  error: boolean;
}

export default function GitHubStats({
  className = "",
  showForks = true,
  showContributors = true
}: GitHubStatsProps) {
  const [stats, setStats] = useState<Stats>({
    stars: 0,
    forks: 0,
    contributors: 0,
    loading: true,
    error: false,
  });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        // Fetch repository data
        const repoResponse = await fetch(
          "https://api.github.com/repos/ctrl-gaurav/effGen"
        );

        if (!repoResponse.ok) {
          throw new Error("Failed to fetch repo data");
        }

        const repoData = await repoResponse.json();

        // Fetch contributors count
        let contributorCount = 0;
        try {
          const contributorsResponse = await fetch(
            "https://api.github.com/repos/ctrl-gaurav/effGen/contributors?per_page=1"
          );

          if (contributorsResponse.ok) {
            // Parse the Link header to get total count
            const linkHeader = contributorsResponse.headers.get("Link");
            if (linkHeader) {
              const match = linkHeader.match(/page=(\d+)>; rel="last"/);
              if (match) {
                contributorCount = parseInt(match[1], 10);
              }
            } else {
              const contributors = await contributorsResponse.json();
              contributorCount = contributors.length;
            }
          }
        } catch {
          contributorCount = 1; // Default to at least 1 contributor
        }

        setStats({
          stars: repoData.stargazers_count || 0,
          forks: repoData.forks_count || 0,
          contributors: contributorCount || 1,
          loading: false,
          error: false,
        });
      } catch (error) {
        console.error("Error fetching GitHub stats:", error);
        setStats({
          stars: 0,
          forks: 0,
          contributors: 0,
          loading: false,
          error: true,
        });
      }
    };

    fetchStats();
  }, []);

  const formatNumber = (num: number): string => {
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + "k";
    }
    return num.toString();
  };

  if (stats.loading) {
    return (
      <div className={`flex items-center gap-4 ${className}`}>
        <div className="animate-pulse flex items-center gap-2">
          <div className="w-4 h-4 bg-gray-300 dark:bg-gray-700 rounded"></div>
          <div className="w-8 h-4 bg-gray-300 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (stats.error) {
    return null; // Don't show anything if there's an error
  }

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {/* Stars */}
      <motion.a
        href="https://github.com/ctrl-gaurav/effGen"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-yellow-400/10 text-yellow-600 dark:text-yellow-400 hover:bg-yellow-400/20 border border-yellow-400/20 hover:border-yellow-400/40 transition-all text-xs font-semibold font-mono"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <FiStar size={12} />
        {formatNumber(stats.stars)} stars
      </motion.a>

      {/* Forks */}
      {showForks && (
        <motion.a
          href="https://github.com/ctrl-gaurav/effGen/fork"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-green-400/10 text-green-600 dark:text-green-400 hover:bg-green-400/20 border border-green-400/20 hover:border-green-400/40 transition-all text-xs font-semibold font-mono"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FiGitBranch size={12} />
          {formatNumber(stats.forks)} forks
        </motion.a>
      )}

      {/* Contributors */}
      {showContributors && stats.contributors > 0 && (
        <motion.a
          href="https://github.com/ctrl-gaurav/effGen/graphs/contributors"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-cyan-400/10 text-cyan-600 dark:text-cyan-400 hover:bg-cyan-400/20 border border-cyan-400/20 hover:border-cyan-400/40 transition-all text-xs font-semibold font-mono"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FiUsers size={12} />
          {stats.contributors} contributors
        </motion.a>
      )}
    </div>
  );
}

// Export a hook for use in other components
export function useGitHubStats() {
  const [stats, setStats] = useState<Stats>({
    stars: 0,
    forks: 0,
    contributors: 0,
    loading: true,
    error: false,
  });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [repoResponse, contributorsResponse] = await Promise.all([
          fetch("https://api.github.com/repos/ctrl-gaurav/effGen"),
          fetch("https://api.github.com/repos/ctrl-gaurav/effGen/contributors?per_page=1"),
        ]);

        if (!repoResponse.ok) {
          throw new Error("Failed to fetch repo data");
        }

        const repoData = await repoResponse.json();

        let contributorCount = 1;
        if (contributorsResponse.ok) {
          const linkHeader = contributorsResponse.headers.get("Link");
          if (linkHeader) {
            const match = linkHeader.match(/page=(\d+)>; rel="last"/);
            if (match) {
              contributorCount = parseInt(match[1], 10);
            }
          } else {
            const contributors = await contributorsResponse.json();
            contributorCount = contributors.length || 1;
          }
        }

        setStats({
          stars: repoData.stargazers_count || 0,
          forks: repoData.forks_count || 0,
          contributors: contributorCount,
          loading: false,
          error: false,
        });
      } catch (error) {
        setStats({
          stars: 0,
          forks: 0,
          contributors: 0,
          loading: false,
          error: true,
        });
      }
    };

    fetchStats();
  }, []);

  return stats;
}
