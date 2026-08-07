import type { Metadata } from "next";
import { Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/ThemeProvider";
import LaunchModal from "@/components/LaunchModal";
import { withBasePath } from "@/components/basePath";

const inter = Inter({
  subsets: ["latin"],
  variable: '--font-inter',
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: '--font-space-grotesk',
});

export const metadata: Metadata = {
  title: "effGen - Production AI Agent Framework for SLMs",
  description: "Build production AI agents with Small Language Models, 14 inference backends, provider-supported native tools, RAG, guardrails, evaluation, and an OpenAI-compatible API server.",
  icons: {
    icon: withBasePath("/favicon.svg"),
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className="overflow-x-hidden">
      <body className={`${inter.variable} ${spaceGrotesk.variable} antialiased overflow-x-hidden`} suppressHydrationWarning>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          {children}
          <LaunchModal />
        </ThemeProvider>
      </body>
    </html>
  );
}
