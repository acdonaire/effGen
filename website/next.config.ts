import type { NextConfig } from "next";

// Set when building the static bundle GitHub Pages serves. A project page lives
// under /<repo>, so every asset and route needs that prefix; a server-rendered
// build (a host that runs `next start`) sets neither and behaves as before.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const isStaticExport = process.env.NEXT_STATIC_EXPORT === "1";

const nextConfig: NextConfig = {
  ...(isStaticExport
    ? {
        output: "export" as const,
        // A static host serves /path/ as /path/index.html.
        trailingSlash: true,
        // The export has no server to optimise images on request.
        images: { unoptimized: true },
      }
    : {
        // Rewrites need a server, so they are declared only for that build.
        // The docs SPA is emitted into public/docs and served from there; this
        // hands its client-side routes back to its own index.html.
        async rewrites() {
          return [
            {
              source: "/docs",
              destination: "/docs/index.html",
            },
            {
              source: "/docs/:path((?!assets/).*)",
              destination: "/docs/index.html",
            },
          ];
        },
      }),
  ...(basePath ? { basePath, assetPrefix: basePath } : {}),
};

export default nextConfig;
