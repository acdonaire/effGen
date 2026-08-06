"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

interface PyPIVersionProps {
  className?: string;
  showLink?: boolean;
}

interface VersionInfo {
  version: string;
  loading: boolean;
  error: boolean;
}

export default function PyPIVersion({ className = "", showLink = true }: PyPIVersionProps) {
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    version: "0.3.1",
    loading: true,
    error: false,
  });

  useEffect(() => {
    const fetchVersion = async () => {
      try {
        const response = await fetch("https://pypi.org/pypi/effgen/json");

        if (!response.ok) {
          throw new Error("Failed to fetch PyPI data");
        }

        const data = await response.json();

        setVersionInfo({
          version: data.info?.version || "0.3.1",
          loading: false,
          error: false,
        });
      } catch (error) {
        console.error("Error fetching PyPI version:", error);
        setVersionInfo({
          version: "0.3.1", // Fallback version
          loading: false,
          error: true,
        });
      }
    };

    fetchVersion();
  }, []);

  if (versionInfo.loading) {
    return (
      <span className={`inline-block px-2 py-1 text-xs rounded-full bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 animate-pulse ${className}`}>
        v...
      </span>
    );
  }

  const versionBadge = (
    <span className={`inline-block px-2 py-1 text-xs rounded-full bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 font-semibold ${className}`}>
      v{versionInfo.version}
    </span>
  );

  if (showLink) {
    return (
      <motion.a
        href="https://pypi.org/project/effgen/"
        target="_blank"
        rel="noopener noreferrer"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {versionBadge}
      </motion.a>
    );
  }

  return versionBadge;
}

// Export a hook for use in other components
export function usePyPIVersion() {
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    version: "0.3.1",
    loading: true,
    error: false,
  });

  useEffect(() => {
    const fetchVersion = async () => {
      try {
        const response = await fetch("https://pypi.org/pypi/effgen/json");

        if (!response.ok) {
          throw new Error("Failed to fetch PyPI data");
        }

        const data = await response.json();

        setVersionInfo({
          version: data.info?.version || "0.3.1",
          loading: false,
          error: false,
        });
      } catch (error) {
        setVersionInfo({
          version: "0.3.1",
          loading: false,
          error: true,
        });
      }
    };

    fetchVersion();
  }, []);

  return versionInfo;
}
