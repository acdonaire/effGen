import { useState, useEffect } from 'react';

interface VersionInfo {
  version: string;
  loading: boolean;
  error: boolean;
}

export function usePyPIVersion() {
  const [versionInfo, setVersionInfo] = useState<VersionInfo>({
    version: "0.3.0",
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
          version: data.info?.version || "0.3.0",
          loading: false,
          error: false,
        });
      } catch (error) {
        setVersionInfo({
          version: "0.3.0",
          loading: false,
          error: true,
        });
      }
    };

    fetchVersion();
  }, []);

  return versionInfo;
}
