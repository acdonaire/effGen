"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import { useRef, useState, useCallback } from "react";
import { FiX, FiCopy, FiCheck, FiChevronDown, FiChevronUp } from "react-icons/fi";
import Container from "./Container";
import { highlightCode } from "./syntaxHighlight";

interface ToolData {
  emoji: string;
  name: string;
  description: string;
  params: string[];
  category: string;
  accent: string;
  fullDescription: string;
  codeExample: string;
}

const categories: { key: string; label: string; accent: string }[] = [
  { key: "ALL", label: "All", accent: "#00ff88" },
  { key: "COMPUTATION", label: "Computation", accent: "#ffd700" },
  { key: "CODE_EXECUTION", label: "Code Execution", accent: "#00ff88" },
  { key: "INFORMATION_RETRIEVAL", label: "Information", accent: "#00e5ff" },
  { key: "ACADEMIC", label: "Academic", accent: "#22d3ee" },
  { key: "DOCUMENTS", label: "Documents", accent: "#22d3ee" },
  { key: "MEDIA", label: "News & Media", accent: "#f59e0b" },
  { key: "MEDIA_PROCESSING", label: "Audio / Image", accent: "#f59e0b" },
  { key: "SOCIAL", label: "Social", accent: "#ec4899" },
  { key: "GEO", label: "Geo / Weather", accent: "#10b981" },
  { key: "DATA_PROCESSING", label: "Data Processing", accent: "#a78bfa" },
  { key: "FILE_OPERATIONS", label: "Files", accent: "#ff9500" },
  { key: "SYSTEM", label: "System", accent: "#ff9500" },
  { key: "COMMUNICATION", label: "Communication", accent: "#ec4899" },
  { key: "UTILITIES", label: "Utilities", accent: "#10b981" },
  { key: "EXTERNAL_API", label: "External API", accent: "#ff6b6b" },
];

const tools: ToolData[] = [
  {
    emoji: "\u{1F9EE}",
    name: "Calculator",
    description: "Perform mathematical calculations, evaluate expressions, and convert units",
    params: ["expression", "operation", "from_unit", "to_unit", "precision"],
    category: "COMPUTATION",
    accent: "#ffd700",
    fullDescription: "A versatile calculator supporting arithmetic, algebra, unit conversions, and statistical operations. Handles complex expressions with operator precedence and parentheses.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import Calculator\ntool = Calculator()\nresult = asyncio.run(tool.execute(expression="24344 * 334"))',
  },
  {
    emoji: "\u{1F550}",
    name: "DateTimeTool",
    description: "Date/time operations, timezone conversion, date arithmetic",
    params: ["operation", "timezone", "date", "days", "hours", "to_timezone"],
    category: "COMPUTATION",
    accent: "#ffd700",
    fullDescription: "Handles date/time operations including timezone conversions, date arithmetic (add/subtract days), formatting, and comparisons between dates across timezones.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import DateTimeTool\ntool = DateTimeTool()\nresult = asyncio.run(tool.execute(operation="now", timezone="Asia/Tokyo"))',
  },
  {
    emoji: "\u26A1",
    name: "CodeExecutor",
    description: "Execute code in a secure sandboxed environment (Python, JS, Bash)",
    params: ["code", "language", "timeout", "memory_limit", "network_enabled", "files", "env_vars"],
    category: "CODE_EXECUTION",
    accent: "#00ff88",
    fullDescription: "Runs code securely in a sandboxed Docker environment. Supports Python, JavaScript, and Bash with configurable timeouts and memory limits. Output is captured and returned.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import CodeExecutor\ntool = CodeExecutor()\nresult = asyncio.run(tool.execute(code="print(2+2)", language="python"))',
  },
  {
    emoji: "\u{1F40D}",
    name: "PythonREPL",
    description: "Execute Python code in a persistent REPL session",
    params: ["code", "session_id", "reset_session", "return_variables", "restricted_mode"],
    category: "CODE_EXECUTION",
    accent: "#00ff88",
    fullDescription: "A persistent Python REPL that maintains state across calls. Variables and imports persist between executions within the same session. Supports restricted mode for safety.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import PythonREPL\ntool = PythonREPL()\nasyncio.run(tool.execute(code="x = 42"))\nasyncio.run(tool.execute(code="print(x * 2)"))  # 84',
  },
  {
    emoji: "\u{1F50D}",
    name: "WebSearch",
    description: "Search the web using DuckDuckGo, SerpAPI, or Google",
    params: ["query", "num_results", "backend", "time_range", "language", "region"],
    category: "INFORMATION_RETRIEVAL",
    accent: "#00e5ff",
    fullDescription: "Multi-backend web search supporting DuckDuckGo (free, no API key), SerpAPI, and Google Custom Search. Returns structured results with titles, URLs, and snippets.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import WebSearch\ntool = WebSearch(backend="duckduckgo")\nresults = asyncio.run(tool.execute(query="effGen framework"))',
  },
  {
    emoji: "\u{1F310}",
    name: "URLFetchTool",
    description: "Fetch webpage content and extract readable text",
    params: ["url", "extract_links"],
    category: "INFORMATION_RETRIEVAL",
    accent: "#00e5ff",
    fullDescription: "Fetches web pages and extracts clean, readable text content. Strips HTML, handles encoding, and respects max_length limits. Can optionally preserve links.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import URLFetchTool\ntool = URLFetchTool()\ntext = asyncio.run(tool.execute(url="https://example.com", max_length=5000))',
  },
  {
    emoji: "\u{1F4DA}",
    name: "WikipediaTool",
    description: "Search Wikipedia and get article summaries",
    params: ["query", "operation", "sentences"],
    category: "INFORMATION_RETRIEVAL",
    accent: "#00e5ff",
    fullDescription: "Search Wikipedia articles and retrieve summaries or full content. Supports search (find articles) and summary (get article overview) operations.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import WikipediaTool\ntool = WikipediaTool()\nsummary = asyncio.run(tool.execute(query="Machine Learning", sentences=3))',
  },
  {
    emoji: "\u{1F4D6}",
    name: "Retrieval",
    description: "RAG \u2014 retrieve documents using embeddings and BM25",
    params: ["query", "top_k", "score_threshold", "filter_metadata"],
    category: "INFORMATION_RETRIEVAL",
    accent: "#00e5ff",
    fullDescription: "Retrieval-Augmented Generation tool that searches document collections using hybrid semantic (embeddings) and keyword (BM25) search for maximum recall and precision.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import Retrieval\ntool = Retrieval(knowledge_base_path="./docs")\nresults = asyncio.run(tool.execute(query="agent configuration"))',
  },
  {
    emoji: "\u{1F50E}",
    name: "AgenticSearch",
    description: "Search files using grep/ripgrep for exact matches",
    params: ["query", "context_lines", "case_sensitive", "use_regex", "max_results", "search_mode", "file_type"],
    category: "INFORMATION_RETRIEVAL",
    accent: "#00e5ff",
    fullDescription: "File search tool that uses grep/ripgrep for fast, exact pattern matching across codebases. Supports regex, file type filtering, and recursive directory search.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import AgenticSearch\ntool = AgenticSearch()\nresults = asyncio.run(tool.execute(query="def run", data_path="./src"))',
  },
  {
    emoji: "\u{1F4CB}",
    name: "JSONTool",
    description: "Parse, query (JSONPath), validate, and format JSON data",
    params: ["data", "operation", "query"],
    category: "DATA_PROCESSING",
    accent: "#a78bfa",
    fullDescription: "Comprehensive JSON toolkit supporting parsing, JSONPath queries, schema validation, formatting (pretty-print/minify), and structural transformations.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import JSONTool\ntool = JSONTool()\nresult = asyncio.run(tool.execute(data=\'{"a": 1}\', operation="query", query="$.a"))',
  },
  {
    emoji: "\u{1F4DD}",
    name: "TextProcessingTool",
    description: "Text analysis: word count, summarize, regex, compare",
    params: ["operation", "text", "pattern", "replacement", "transform_type", "num_sentences"],
    category: "DATA_PROCESSING",
    accent: "#a78bfa",
    fullDescription: "Swiss-army knife for text operations: word/character counting, regex matching and replacement, text comparison (diff), case conversion, and basic summarization.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import TextProcessingTool\ntool = TextProcessingTool()\nresult = asyncio.run(tool.execute(operation="word_count", text="Hello world"))',
  },
  {
    emoji: "\u{1F4C1}",
    name: "FileOperations",
    description: "Safe file system operations: read, write, search, convert",
    params: ["operation", "path", "content", "format", "encoding", "pattern", "recursive", "target_format"],
    category: "FILE_OPERATIONS",
    accent: "#ff9500",
    fullDescription: "Secure file system operations with built-in safety controls. Read, write, list, search, and convert files. Path traversal prevention and configurable allowed directories.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import FileOperations\ntool = FileOperations()\ncontent = asyncio.run(tool.execute(operation="read", path="./data.txt"))',
  },
  {
    emoji: "\u{1F4BB}",
    name: "BashTool",
    description: "Execute shell commands with security controls",
    params: ["command"],
    category: "SYSTEM",
    accent: "#ff9500",
    fullDescription: "Execute shell commands with configurable security controls. Supports command whitelisting/blacklisting, timeout enforcement, and working directory specification.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import BashTool\ntool = BashTool()\nresult = asyncio.run(tool.execute(command="ls -la", timeout=30))',
  },
  {
    emoji: "\u{1F324}\uFE0F",
    name: "WeatherTool",
    description: "Get current / forecast / historical weather for any location (free API)",
    params: ["operation", "location", "units"],
    category: "EXTERNAL_API",
    accent: "#ff6b6b",
    fullDescription: "Get current weather data, forecasts, or historical observations for any location worldwide using the free Open-Meteo API. No API key required. Accepts either a place name (resolved via the geocoding service) or explicit lat/lon. Returns temperature, humidity, wind, and weather codes.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import WeatherTool\ntool = WeatherTool()\nweather = asyncio.run(tool.execute(operation="current", location="San Francisco", units="metric"))',
  },
  {
    emoji: "\u{1F4C8}",
    name: "StockPriceTool",
    description: "Fetch stock prices (yfinance + Yahoo Finance v8 fallback)",
    params: ["symbol"],
    category: "EXTERNAL_API",
    accent: "#ff6b6b",
    fullDescription: "Real-time stock quotes via yfinance with a Yahoo Finance v8 fallback. Always returns a 'not financial advice' disclaimer. No API key required for the fallback.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import StockPriceTool\ntool = StockPriceTool()\nresult = asyncio.run(tool.execute(symbol="AAPL"))',
  },
  {
    emoji: "\u{1F4B1}",
    name: "CurrencyConverterTool",
    description: "Convert between currencies via frankfurter.app / ECB",
    params: ["amount", "from_currency", "to_currency"],
    category: "EXTERNAL_API",
    accent: "#ff6b6b",
    fullDescription: "Convert amounts between 30+ currencies using the free frankfurter.app API backed by European Central Bank rates. No API key required.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import CurrencyConverterTool\ntool = CurrencyConverterTool()\nresult = asyncio.run(tool.execute(amount=100, from_currency="USD", to_currency="EUR"))',
  },
  {
    emoji: "\u20BF",
    name: "CryptoTool",
    description: "Crypto prices and market data via CoinGecko",
    params: ["coin", "vs_currency"],
    category: "EXTERNAL_API",
    accent: "#ff6b6b",
    fullDescription: "Cryptocurrency prices, market cap, and 24h change via the public CoinGecko API. Includes a 'not financial advice' disclaimer. No API key required.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import CryptoTool\ntool = CryptoTool()\nresult = asyncio.run(tool.execute(coin="bitcoin", vs_currency="usd"))',
  },
  {
    emoji: "\u{1F4CA}",
    name: "DataFrameTool",
    description: "Pandas DataFrame: load, head, describe, filter, aggregate",
    params: ["file_path", "operation", "n", "column", "op", "value", "agg", "group_by"],
    category: "COMPUTATION",
    accent: "#a78bfa",
    fullDescription: "Load CSV or JSON/JSONL into pandas DataFrames and run head/describe/info/filter/aggregate operations. Auto-detects format from file extension.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import DataFrameTool\ntool = DataFrameTool()\nresult = asyncio.run(tool.execute(file_path="./data.csv", operation="describe"))',
  },
  {
    emoji: "\u{1F4C9}",
    name: "PlotTool",
    description: "Matplotlib plots: line, bar, scatter, hist → PNG",
    params: ["chart_type", "x", "y", "title", "xlabel", "ylabel", "output_path"],
    category: "DATA_PROCESSING",
    accent: "#a78bfa",
    fullDescription: "Render matplotlib plots (line, bar, scatter, histogram) directly to PNG. Useful for data-science agents that need to generate visualisations.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import PlotTool\ntool = PlotTool()\nasyncio.run(tool.execute(chart_type="line", x=[1,2,3], y=[4,5,6], output_path="out.png"))',
  },
  {
    emoji: "\u{1F4CF}",
    name: "StatsTool",
    description: "NumPy stats: mean, median, std, variance, summary, correlation, regression",
    params: ["operation", "data", "y"],
    category: "DATA_PROCESSING",
    accent: "#a78bfa",
    fullDescription: "NumPy-backed statistics: mean, median, std, variance, min, max, summary, correlation, and simple linear regression. Works on lists or arrays.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import StatsTool\ntool = StatsTool()\nresult = asyncio.run(tool.execute(operation="correlation", data=[1,2,3], y=[2,4,6]))',
  },
  {
    emoji: "\u{1F500}",
    name: "GitTool",
    description: "Read-only Git: status, log, diff, branch, show, remote",
    params: ["operation", "cwd", "n", "ref"],
    category: "SYSTEM",
    accent: "#ff9500",
    fullDescription: "Read-only Git operations — status, log, diff, branch, show, remote. Never mutates repo state so it is safe to expose to agents.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import GitTool\ntool = GitTool()\nresult = asyncio.run(tool.execute(operation="status", cwd="."))',
  },
  {
    emoji: "\u{1F433}",
    name: "DockerTool",
    description: "Read-only Docker: ps, images, logs, version, info",
    params: ["operation", "container", "tail"],
    category: "SYSTEM",
    accent: "#ff9500",
    fullDescription: "Read-only Docker operations for inspecting running containers, images, logs, version, and info. Never starts, stops, or removes containers.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import DockerTool\ntool = DockerTool()\nresult = asyncio.run(tool.execute(operation="ps"))',
  },
  {
    emoji: "\u{1F5A5}\uFE0F",
    name: "SystemInfoTool",
    description: "System info via psutil: cpu, memory, disk, network",
    params: ["kind"],
    category: "SYSTEM",
    accent: "#ff9500",
    fullDescription: "Query live system metrics via psutil: CPU load, memory, disk usage, and network counters. Useful for diagnostic agents.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import SystemInfoTool\ntool = SystemInfoTool()\nresult = asyncio.run(tool.execute(kind="memory"))',
  },
  {
    emoji: "\u{1F310}",
    name: "HTTPTool",
    description: "urllib-based HTTP GET/POST with headers and body",
    params: ["url", "method", "headers", "params", "json_body", "timeout"],
    category: "EXTERNAL_API",
    accent: "#ff6b6b",
    fullDescription: "Lightweight HTTP GET/POST tool built on urllib — no extra dependencies. Supports custom headers, JSON bodies, query params, and configurable timeouts.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import HTTPTool\ntool = HTTPTool()\nresult = asyncio.run(tool.execute(url="https://api.github.com", method="GET"))',
  },
  {
    emoji: "\u{1F4DD}",
    name: "ArxivTool",
    description: "Search arXiv papers via Atom feed",
    params: ["query", "max_results"],
    category: "INFORMATION_RETRIEVAL",
    accent: "#00e5ff",
    fullDescription: "Search arXiv using its public Atom feed — titles, authors, abstracts, and PDF links. No API key required.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import ArxivTool\ntool = ArxivTool()\nresult = asyncio.run(tool.execute(query="agentic SLM", max_results=5))',
  },
  {
    emoji: "\u{1F6A7}",
    name: "StackOverflowTool",
    description: "Query Stack Overflow via the Stack Exchange API",
    params: ["query", "max_results"],
    category: "INFORMATION_RETRIEVAL",
    accent: "#00e5ff",
    fullDescription: "Search Stack Overflow using the public Stack Exchange API — question titles, scores, answer counts, tags, and links.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import StackOverflowTool\ntool = StackOverflowTool()\nresult = asyncio.run(tool.execute(query="pandas groupby", max_results=3))',
  },
  {
    emoji: "\u{1F419}",
    name: "GitHubTool",
    description: "Search GitHub public repos, code, and issues",
    params: ["query", "kind", "max_results"],
    category: "INFORMATION_RETRIEVAL",
    accent: "#00e5ff",
    fullDescription: "Search GitHub using the public search API for repositories, code, and issues. No auth required for low-volume use.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import GitHubTool\ntool = GitHubTool()\nresult = asyncio.run(tool.execute(query="effgen agent", kind="repositories"))',
  },
  {
    emoji: "\u{1F9E0}",
    name: "WolframAlphaTool",
    description: "WolframAlpha computational knowledge (optional, API key)",
    params: ["query"],
    category: "EXTERNAL_API",
    accent: "#ff6b6b",
    fullDescription: "Query WolframAlpha's computational knowledge engine for math, science, units, and facts. Requires a free AppID set via the WOLFRAM_ALPHA_APPID env var, or passed as app_id=... to the constructor.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import WolframAlphaTool\n# Provide AppID via env: WOLFRAM_ALPHA_APPID=...\ntool = WolframAlphaTool()\nresult = asyncio.run(tool.execute(query="integrate x^2 dx"))',
  },
  {
    emoji: "\u2709\uFE0F",
    name: "EmailDraftTool",
    description: "Draft emails (does NOT send) — safe by design",
    params: ["to", "subject", "body", "cc", "bcc", "from_address"],
    category: "COMMUNICATION",
    accent: "#ec4899",
    fullDescription: "Compose email drafts with to/cc/bcc/subject/body. Does NOT send — returns a structured draft for human review.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import EmailDraftTool\ntool = EmailDraftTool()\nresult = asyncio.run(tool.execute(to=["alice@example.com"], subject="Hi", body="..."))',
  },
  {
    emoji: "\u{1F4AC}",
    name: "SlackDraftTool",
    description: "Draft Slack messages (does NOT send)",
    params: ["channel", "text", "thread_ts", "mentions"],
    category: "COMMUNICATION",
    accent: "#ec4899",
    fullDescription: "Compose Slack message drafts with channel, thread, and markdown-aware text. Draft-only — never posts to Slack.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import SlackDraftTool\ntool = SlackDraftTool()\nresult = asyncio.run(tool.execute(channel="#eng", text="Deploy finished \u2705"))',
  },
  {
    emoji: "\u{1F514}",
    name: "NotificationTool",
    description: "Local desktop notifications (plyer, optional)",
    params: ["title", "message", "timeout"],
    category: "COMMUNICATION",
    accent: "#ec4899",
    fullDescription: "Send local desktop notifications via plyer. Works on macOS / Windows / Linux. Purely local — no network access.",
    codeExample: 'import asyncio\nfrom effgen.tools.builtin import NotificationTool\ntool = NotificationTool()\nasyncio.run(tool.execute(title="Done", message="Agent finished", timeout=5))',
  },
  // ── v0.2.5: 13 new free / no-auth tools ──
  {
    emoji: "\u{1F9EC}",
    name: "PubMedTool",
    description: "Search PubMed via NCBI E-utilities; fetch metadata + abstracts",
    params: ["operation", "query", "max_results", "pmid"],
    category: "ACADEMIC",
    accent: "#22d3ee",
    fullDescription: "Search PubMed via NCBI E-utilities, fetch article metadata, and retrieve abstracts. Built-in token-bucket rate limiter (3 req/s without key, 10/s with NCBI_API_KEY). Operations: search, fetch, abstract. New in v0.2.5; part of the research preset.",
    codeExample: 'from effgen.tools.builtin.pubmed import PubMedTool\ntool = PubMedTool()\nresult = await tool.execute({"operation": "search", "query": "CRISPR gene editing", "max_results": 5})\nfor a in result["data"]["articles"]:\n    print(a["pmid"], a["title"])',
  },
  {
    emoji: "\u{1F4D8}",
    name: "ArXivTool",
    description: "Search arXiv Atom feed; fetch by ID; download PDFs",
    params: ["operation", "query", "arxiv_id", "max_results"],
    category: "ACADEMIC",
    accent: "#22d3ee",
    fullDescription: "Search arXiv via Atom feed, fetch paper metadata by ID, and download PDFs. No auth required. Operations: search, fetch, download_pdf. New in v0.2.5; part of the research preset. (ArxivTool alias retained for back-compat.)",
    codeExample: 'from effgen.tools.builtin.arxiv import ArXivTool\ntool = ArXivTool()\nresult = await tool.execute({"operation": "search", "query": "attention is all you need", "max_results": 3})\nfor p in result["data"]["papers"]:\n    print(p["id"], p["title"])',
  },
  {
    emoji: "\u{1F50D}",
    name: "SemanticScholarTool",
    description: "Search papers, fetch details, citations & references",
    params: ["operation", "query", "paper_id", "limit"],
    category: "ACADEMIC",
    accent: "#22d3ee",
    fullDescription: "Search papers, fetch paper details, and retrieve citations / references via the Semantic Scholar Graph API. Built-in backoff (100 req / 5 min unauthenticated). Operations: search, paper, citations, references. New in v0.2.5; part of the research preset.",
    codeExample: 'from effgen.tools.builtin.semantic_scholar import SemanticScholarTool\ntool = SemanticScholarTool()\nresult = await tool.execute({"operation": "search", "query": "large language models survey"})\nfor p in result["data"]["papers"][:3]:\n    print(p.get("paperId"), p.get("title"))',
  },
  {
    emoji: "\u{1F4E1}",
    name: "RSSFeedTool",
    description: "Fetch, browse, full-text search any RSS / Atom feed",
    params: ["operation", "url", "n", "query"],
    category: "MEDIA",
    accent: "#f59e0b",
    fullDescription: "Fetch, browse, and full-text search any RSS / Atom feed by URL. Handles malformed feeds gracefully. Operations: fetch, latest, search_in_feed. New in v0.2.5; part of the research and general presets.",
    codeExample: 'from effgen.tools.builtin.rss import RSSFeedTool\ntool = RSSFeedTool()\nresult = await tool.execute({"operation": "latest", "url": "https://hnrss.org/frontpage", "n": 5})\nfor item in result["data"]["items"]:\n    print(item["title"])',
  },
  {
    emoji: "\u{1F4F0}",
    name: "NewsTool",
    description: "Aggregate headlines from curated reputable RSS sources",
    params: ["operation", "query", "category", "n"],
    category: "MEDIA",
    accent: "#f59e0b",
    fullDescription: "Aggregate top headlines and search news across curated reputable RSS sources (Reuters, BBC, HN, NPR, Al Jazeera, …); optional NEWS_API_KEY for NewsAPI.org. Operations: top_headlines, search. New in v0.2.5; part of the research and general presets.",
    codeExample: 'from effgen.tools.builtin.news import NewsTool\ntool = NewsTool()\nresult = await tool.execute({"operation": "top_headlines"})\nfor a in result["data"]["articles"][:5]:\n    print(a["title"], "-", a["source"])',
  },
  {
    emoji: "\u{1F39E}️",
    name: "YouTubeTranscriptTool",
    description: "YouTube captions / transcripts — no Google API key",
    params: ["operation", "video_id", "lang", "target_lang"],
    category: "MEDIA",
    accent: "#f59e0b",
    fullDescription: "Fetch YouTube captions / transcripts without a Google API key via youtube-transcript-api. Operations: get_transcript, list_available_languages, translated. Handles watch?v=, youtu.be/, and shorts/ URL formats. New in v0.2.5; part of the research preset.",
    codeExample: 'from effgen.tools.builtin.youtube_transcript import YouTubeTranscriptTool\ntool = YouTubeTranscriptTool()\nresult = await tool.execute({\n    "operation": "get_transcript",\n    "video_id": "dQw4w9WgXcQ",\n    "lang": "en",\n})\nprint(result["data"]["transcript"][:500])',
  },
  {
    emoji: "\u{1F4FA}",
    name: "YouTubeMetadataTool",
    description: "Video / channel metadata via yt-dlp (metadata-only)",
    params: ["operation", "video_id", "channel"],
    category: "MEDIA",
    accent: "#f59e0b",
    fullDescription: "Fetch video / channel metadata using yt-dlp in metadata-only mode. Operations: metadata, channel. No auth required for public content. New in v0.2.5; part of the research preset.",
    codeExample: 'from effgen.tools.builtin.youtube_metadata import YouTubeMetadataTool\ntool = YouTubeMetadataTool()\nresult = await tool.execute({\n    "operation": "metadata",\n    "video_id": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",\n})\nprint(result["data"].get("title"), result["data"].get("uploader"))',
  },
  {
    emoji: "\u{1F47D}",
    name: "RedditTool",
    description: "Reddit top/hot/user/thread via public JSON (no OAuth)",
    params: ["operation", "subreddit", "time", "user", "thread_id", "n"],
    category: "SOCIAL",
    accent: "#ec4899",
    fullDescription: "Access Reddit top/hot posts, user submissions, and thread comments via public JSON endpoints — no OAuth required for reads. Sets effGen/<version> User-Agent; exponential backoff on 429. Operations: subreddit_top, subreddit_hot, user_submissions, thread_comments. New in v0.2.5; part of research + general presets.",
    codeExample: 'from effgen.tools.builtin.reddit import RedditTool\ntool = RedditTool()\nresult = await tool.execute({"operation": "subreddit_top", "subreddit": "python", "time": "day", "n": 5})\nfor post in result["data"]["posts"]:\n    print(post["title"])',
  },
  {
    emoji: "\u{1F9E1}",
    name: "HackerNewsTool",
    description: "HN top/new stories, story details, user profiles",
    params: ["operation", "n", "story_id", "user"],
    category: "SOCIAL",
    accent: "#ec4899",
    fullDescription: "Fetch top / new stories, story details, and user profiles from the Hacker News Firebase API. No auth required. Operations: top_stories, new_stories, story, user. New in v0.2.5; part of research + general presets.",
    codeExample: 'from effgen.tools.builtin.hackernews import HackerNewsTool\ntool = HackerNewsTool()\nresult = await tool.execute({"operation": "top_stories", "n": 5})\nfor s in result["data"]["stories"]:\n    print(s["title"], s.get("url", ""))',
  },
  {
    emoji: "\u{1F310}",
    name: "TranslateTool",
    description: "Translate via LibreTranslate; offline argostranslate fallback",
    params: ["operation", "text", "source", "target"],
    category: "UTILITIES",
    accent: "#10b981",
    fullDescription: "Translate text between languages with LibreTranslate as the primary backend (configurable via LIBRE_TRANSLATE_URL) and argostranslate as an offline fallback. Language pack cache at ~/.effgen/argos/. Operations: translate, available_pairs. New in v0.2.5; part of the general preset.",
    codeExample: 'from effgen.tools.builtin.translate import TranslateTool\ntool = TranslateTool()\nresult = await tool.execute({"operation": "translate", "text": "Hello, world!", "source": "en", "target": "fr"})\nprint(result["data"]["translated_text"])',
  },
  {
    emoji: "\u{1F1FA}\u{1F1F3}",
    name: "LanguageDetectTool",
    description: "Detect language offline (55+ langs via langdetect)",
    params: ["operation", "text", "texts"],
    category: "UTILITIES",
    accent: "#10b981",
    fullDescription: "Detect the language of text or a batch of texts, fully offline via langdetect (55+ languages). Operations: detect, detect_batch. New in v0.2.5; part of the general preset.",
    codeExample: 'from effgen.tools.builtin.language_detect import LanguageDetectTool\ntool = LanguageDetectTool()\nresult = await tool.execute({"operation": "detect", "text": "Bonjour le monde"})\nprint(result["data"]["language"], result["data"]["confidence"])',
  },
  {
    emoji: "\u{1F532}",
    name: "QRGenerateTool",
    description: "Generate QR codes locally (PNG / base64 data URL)",
    params: ["operation", "data", "data_url_return", "output_path"],
    category: "UTILITIES",
    accent: "#10b981",
    fullDescription: "Generate QR codes locally from any text or URL; returns base64 PNG or saves to a file path. Supports data_url_return=True for inline embedding. No network required. Operations: generate. New in v0.2.5; part of the general preset.",
    codeExample: 'from effgen.tools.builtin.qr_generate import QRGenerateTool\ntool = QRGenerateTool()\nresult = await tool.execute({"operation": "generate", "data": "https://effgen.org", "data_url_return": True})\nprint(result["data"].get("data_url", "")[:40] + "...")',
  },
  {
    emoji: "\u{1F4F7}",
    name: "QRReadTool",
    description: "Decode QR / barcodes from images (pyzbar + OpenCV fallback)",
    params: ["operation", "image_path", "image_base64"],
    category: "UTILITIES",
    accent: "#10b981",
    fullDescription: "Decode QR codes and barcodes from image files or base64 PNG using pyzbar + Pillow, with OpenCV QR fallback when libzbar is unavailable. Fully local. Operations: read. New in v0.2.5; part of the general preset.",
    codeExample: 'from effgen.tools.builtin.qr_read import QRReadTool\ntool = QRReadTool()\nresult = await tool.execute({"operation": "read", "image_path": "/tmp/qr.png"})\nfor code in result["data"]["codes"]:\n    print(code["data"])',
  },
  // ── v0.2.6: 14 new document / media / comms tools ──
  {
    emoji: "\u{1F4DD}",
    name: "OCRTool",
    description: "Extract text from images via Tesseract (local) + OCR.space fallback",
    params: ["operation", "image_path", "lang"],
    category: "DOCUMENTS",
    accent: "#22d3ee",
    fullDescription: "Extract text from images using Tesseract (local, primary) with OCR.space free API as fallback (OCR_SPACE_API_KEY). Raises OCRBackendUnavailable with per-OS install instructions when no backend is available. Operations: extract, extract_regions. New in v0.2.6; part of the general preset.",
    codeExample: 'from effgen.tools.builtin.ocr import OCRTool\ntool = OCRTool()\nresult = await tool.execute({"operation": "extract", "image_path": "/tmp/scan.png", "lang": "eng"})\nprint(result["data"]["text"])',
  },
  {
    emoji: "\u{1F3A4}",
    name: "AudioTranscribeTool",
    description: "Local audio transcription via faster-whisper + HF Inference fallback",
    params: ["operation", "audio_path", "model_size", "language"],
    category: "MEDIA_PROCESSING",
    accent: "#f59e0b",
    fullDescription: "Transcribe audio files locally via faster-whisper (CPU/GPU auto-detected) with HuggingFace Inference fallback (HF_TOKEN). Detects GPU via nvidia-smi; warns when model_size > 'base' on CPU. Operations: transcribe. New in v0.2.6; part of the media preset.",
    codeExample: 'from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool\ntool = AudioTranscribeTool()\nresult = await tool.execute({"operation": "transcribe", "audio_path": "/tmp/clip.mp3", "model_size": "base"})\nprint(result["data"]["text"])',
  },
  {
    emoji: "\u{1F5BC}️",
    name: "ImageInfoTool",
    description: "Local image metadata + resize/thumbnail via Pillow (zero network)",
    params: ["operation", "image_path", "width", "height"],
    category: "MEDIA_PROCESSING",
    accent: "#f59e0b",
    fullDescription: "Extract image metadata (size, format, mode, EXIF, color histogram) and perform local resize/thumbnail operations using Pillow. Zero network calls. Operations: info, resize, thumbnail. New in v0.2.6; part of the general preset.",
    codeExample: 'from effgen.tools.builtin.image_info import ImageInfoTool\ntool = ImageInfoTool()\nresult = await tool.execute({"operation": "info", "image_path": "/tmp/photo.jpg"})\nprint(result["data"]["size"], result["data"]["format"])',
  },
  {
    emoji: "\u{1F441}️",
    name: "ImageCaptionTool",
    description: "Vision-router captioning (Gemini / OpenAI / MLX-VLM)",
    params: ["operation", "image_path"],
    category: "MEDIA_PROCESSING",
    accent: "#f59e0b",
    fullDescription: "Generate natural-language descriptions of images via the effGen model router (selects a vision-capable provider: Gemini, OpenAI, or MLX-VLM). Raises NoVisionProviderAvailable when no vision-capable provider is configured. Operations: caption, describe. New in v0.2.6; part of the media preset.",
    codeExample: 'from effgen.tools.builtin.image_caption import ImageCaptionTool\ntool = ImageCaptionTool()\nresult = await tool.execute({"operation": "caption", "image_path": "/tmp/photo.jpg"})\nprint(result["data"]["caption"])',
  },
  {
    emoji: "\u{1F4D5}",
    name: "PDFTool",
    description: "Extract text, tables, metadata from PDFs (pypdf + pdfplumber)",
    params: ["operation", "path", "pages"],
    category: "DOCUMENTS",
    accent: "#22d3ee",
    fullDescription: "Extract text, tables, and metadata from PDF files using pypdf (primary) with pdfplumber for structured table extraction. Operations: text, metadata, tables, extract_images. New in v0.2.6; part of the research and general presets.",
    codeExample: 'from effgen.tools.builtin.pdf import PDFTool\ntool = PDFTool()\nresult = await tool.execute({"operation": "text", "path": "/tmp/paper.pdf"})\nprint(result["data"]["text"][:500])',
  },
  {
    emoji: "\u{1F4C4}",
    name: "DOCXTool",
    description: "Parse Word documents (.docx) via python-docx",
    params: ["operation", "path"],
    category: "DOCUMENTS",
    accent: "#22d3ee",
    fullDescription: "Parse Word documents (.docx) using python-docx. Operations: text, paragraphs, tables, metadata. New in v0.2.6; part of the research and general presets.",
    codeExample: 'from effgen.tools.builtin.docx import DOCXTool\ntool = DOCXTool()\nresult = await tool.execute({"operation": "text", "path": "/tmp/report.docx"})\nprint(result["data"]["text"])',
  },
  {
    emoji: "\u{1F4CA}",
    name: "ExcelTool",
    description: "Read Excel workbooks (.xlsx) — openpyxl + pandas",
    params: ["operation", "path", "sheet_name"],
    category: "DOCUMENTS",
    accent: "#22d3ee",
    fullDescription: "Read Excel workbooks (.xlsx) using openpyxl with tabular DataFrame output via pandas. Operations: sheets, read_sheet, headers. New in v0.2.6; part of the research and general presets.",
    codeExample: 'from effgen.tools.builtin.excel import ExcelTool\ntool = ExcelTool()\nresult = await tool.execute({"operation": "read_sheet", "path": "/tmp/data.xlsx", "sheet_name": "Sheet1"})\nprint(result["data"]["rows"][:3])',
  },
  {
    emoji: "⛅",
    name: "WeatherTool (v0.2.6)",
    description: "Current / forecast / historical weather via Open-Meteo (no auth)",
    params: ["operation", "lat", "lon", "location", "days", "units"],
    category: "GEO",
    accent: "#10b981",
    fullDescription: "Fetch current conditions, forecasts, and historical weather data from Open-Meteo (free, no auth required). Integrates with GeocodeTool for place-name → lat/lon resolution. Operations: current, forecast, historical. New backend in v0.2.6; part of the general preset. (Coexists with the older WeatherTool entry; the new implementation backs the general preset wiring.)",
    codeExample: 'from effgen.tools.builtin.weather import WeatherTool\ntool = WeatherTool()\nresult = await tool.execute({"operation": "current", "lat": 37.42, "lon": -122.08})\nprint(result["data"]["temperature_c"], result["data"]["weather_description"])',
  },
  {
    emoji: "\u{1F4CD}",
    name: "GeocodeTool",
    description: "Forward / reverse geocoding via Nominatim (OSM), polite rate-limited",
    params: ["operation", "address", "lat", "lon"],
    category: "GEO",
    accent: "#10b981",
    fullDescription: "Forward/reverse geocoding using Nominatim (OpenStreetMap). Sets effGen/<version> User-Agent as required; built-in 1 req/s token-bucket rate limiter. Operations: geocode, reverse. New in v0.2.6; part of the general preset.",
    codeExample: 'from effgen.tools.builtin.geocode import GeocodeTool\ntool = GeocodeTool()\nresult = await tool.execute({"operation": "geocode", "address": "1600 Amphitheatre Pkwy, Mountain View, CA"})\nprint(result["data"]["lat"], result["data"]["lon"])',
  },
  {
    emoji: "\u{1F5FA}️",
    name: "MapsTool",
    description: "Render static PNG maps from OpenStreetMap tiles (staticmap)",
    params: ["operation", "lat", "lon", "zoom", "dest"],
    category: "GEO",
    accent: "#10b981",
    fullDescription: "Render static PNG maps from OpenStreetMap tiles using the staticmap library. Operations: render, bounding_box. New in v0.2.6; part of the general preset.",
    codeExample: 'from effgen.tools.builtin.maps import MapsTool\ntool = MapsTool()\nresult = await tool.execute({"operation": "render", "lat": 37.42, "lon": -122.08, "zoom": 13, "dest": "/tmp/map.png"})\nprint(result["data"]["path"])',
  },
  {
    emoji: "\u{1F4E7}",
    name: "EmailSMTPTool",
    description: "Send email via SMTP — stdlib smtplib, TLS-on by default",
    params: ["operation", "to", "subject", "body"],
    category: "COMMUNICATION",
    accent: "#ec4899",
    fullDescription: "Send email via SMTP using stdlib smtplib. TLS-on by default. Config: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM. Raises MissingCredentialsError when config is absent. Operations: send. New in v0.2.6; part of the notify preset.",
    codeExample: 'from effgen.tools.builtin.email_smtp import EmailSMTPTool\ntool = EmailSMTPTool()\nresult = await tool.execute({"operation": "send", "to": "alice@example.com", "subject": "Hello", "body": "Hi there!"})',
  },
  {
    emoji: "\u{1F4E5}",
    name: "EmailIMAPTool",
    description: "Read email via IMAP — list folders, fetch recent, search",
    params: ["operation", "folder", "n", "query"],
    category: "COMMUNICATION",
    accent: "#ec4899",
    fullDescription: "Read email via IMAP using stdlib imaplib. Config: IMAP_HOST, IMAP_PORT, IMAP_USER, IMAP_PASSWORD. Operations: list_folders, fetch_recent, search, get. New in v0.2.6; part of the notify preset.",
    codeExample: 'from effgen.tools.builtin.email_imap import EmailIMAPTool\ntool = EmailIMAPTool()\nresult = await tool.execute({"operation": "fetch_recent", "folder": "INBOX", "n": 5})\nfor msg in result["data"]["messages"]:\n    print(msg["subject"], msg["from"])',
  },
  {
    emoji: "\u{1F4AC}",
    name: "SlackWebhookTool",
    description: "Post Slack messages via incoming webhook URL (no OAuth)",
    params: ["operation", "text", "channel"],
    category: "COMMUNICATION",
    accent: "#ec4899",
    fullDescription: "Post messages to Slack via incoming webhook URL (no OAuth required). Config: SLACK_WEBHOOK_URL. URL is redacted in all logs. Operations: post. New in v0.2.6; part of the notify preset.",
    codeExample: 'from effgen.tools.builtin.slack_webhook import SlackWebhookTool\ntool = SlackWebhookTool()\nresult = await tool.execute({"operation": "post", "text": "Deploy complete!"})',
  },
  {
    emoji: "\u{1F47E}",
    name: "DiscordWebhookTool",
    description: "Post Discord messages via webhook URL (no auth)",
    params: ["operation", "content"],
    category: "COMMUNICATION",
    accent: "#ec4899",
    fullDescription: "Post messages to Discord via webhook URL. Config: DISCORD_WEBHOOK_URL. URL is redacted in all logs. Operations: post. New in v0.2.6; part of the notify preset.",
    codeExample: 'from effgen.tools.builtin.discord_webhook import DiscordWebhookTool\ntool = DiscordWebhookTool()\nresult = await tool.execute({"operation": "post", "content": "Deployment succeeded!"})',
  },
];

const heroToolNames = ["Calculator", "CodeExecutor", "WebSearch", "PythonREPL", "FileOperations", "BashTool", "URLFetchTool", "JSONTool"];

/* ── Tool Card ── */

function ToolCard({ tool, onClick }: { tool: ToolData; onClick: () => void }) {
  const [ripple, setRipple] = useState<{ x: number; y: number } | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseEnter = useCallback((e: React.MouseEvent) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    setRipple({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  }, []);

  return (
    <motion.div
      ref={cardRef}
      variants={{
        hidden: { y: 20, opacity: 0, scale: 0.95 },
        visible: { y: 0, opacity: 1, scale: 1, transition: { duration: 0.4, ease: "easeOut" as const } },
      }}
      whileHover={{
        y: -6,
        scale: 1.02,
        transition: { type: "spring", stiffness: 400, damping: 25 },
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setRipple(null)}
      onClick={onClick}
      className="group relative p-5 rounded-xl bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 backdrop-blur-sm overflow-hidden cursor-pointer"
      style={{ transformStyle: "preserve-3d" }}
    >
      {/* Ripple effect */}
      {ripple && (
        <motion.div
          className="absolute rounded-full pointer-events-none"
          style={{
            left: ripple.x,
            top: ripple.y,
            width: 10,
            height: 10,
            marginLeft: -5,
            marginTop: -5,
            background: `${tool.accent}20`,
          }}
          initial={{ scale: 0, opacity: 0.5 }}
          animate={{ scale: 6, opacity: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        />
      )}

      {/* Hover glow */}
      <motion.div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl"
        style={{ background: `radial-gradient(circle at 50% 30%, ${tool.accent}12 0%, transparent 70%)` }}
      />

      {/* Top row: emoji + category badge */}
      <div className="flex items-start justify-between mb-3">
        <motion.div
          className="text-2xl"
          whileHover={{ scale: 1.3, rotate: [0, -10, 10, 0] }}
          transition={{ duration: 0.4 }}
        >
          {tool.emoji}
        </motion.div>
        <span
          className="px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider"
          style={{ background: `${tool.accent}15`, color: tool.accent, border: `1px solid ${tool.accent}25` }}
        >
          {tool.category === "INFORMATION_RETRIEVAL" ? "INFO" : tool.category === "DATA_PROCESSING" ? "DATA" : tool.category === "CODE_EXECUTION" ? "CODE" : tool.category === "EXTERNAL_API" ? "API" : tool.category === "COMMUNICATION" ? "COMM" : tool.category === "FILE_OPERATIONS" ? "FILES" : tool.category === "ACADEMIC" ? "ACADEMIC" : tool.category === "MEDIA" ? "MEDIA" : tool.category === "SOCIAL" ? "SOCIAL" : tool.category === "UTILITIES" ? "UTIL" : tool.category}
        </span>
      </div>

      {/* Name */}
      <h3 className="text-sm font-bold text-gray-900 dark:text-white mb-1.5">{tool.name}</h3>

      {/* Description */}
      <p className="text-xs text-gray-600 dark:text-gray-500 leading-relaxed mb-3 group-hover:text-gray-700 dark:group-hover:text-gray-400 transition-colors line-clamp-2">
        {tool.description}
      </p>

      {/* Param tags */}
      <div className="flex flex-wrap gap-1">
        {tool.params.map((param, idx) => (
          <span
            key={idx}
            className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-500 border border-gray-200 dark:border-gray-700"
          >
            {param}
          </span>
        ))}
      </div>

      {/* Bottom glow line */}
      <motion.div
        className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
        style={{ background: `linear-gradient(90deg, transparent, ${tool.accent}, transparent)` }}
      />

      {/* Corner dot */}
      <div
        className="absolute top-3 right-3 w-1.5 h-1.5 rounded-full opacity-0 group-hover:opacity-100 transition-opacity hidden"
        style={{ background: tool.accent, boxShadow: `0 0 8px ${tool.accent}` }}
      />
    </motion.div>
  );
}

/* ── Tool Detail Panel ── */

function makeRunnableToolExample(code: string) {
  const indented = code.split("\n").map((line) => `    ${line}`).join("\n");
  return `import asyncio\n\nasync def main():\n${indented}\n\nasyncio.run(main())`;
}

function ToolDetail({ tool, onClose }: { tool: ToolData; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const runnableCode = makeRunnableToolExample(tool.codeExample);

  const handleCopy = () => {
    navigator.clipboard.writeText(runnableCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const relatedTools = tools.filter(t => t.category === tool.category && t.name !== tool.name);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      <motion.div
        className="relative w-full max-w-lg max-h-[85vh] overflow-y-auto rounded-2xl bg-white dark:bg-[#0a1a0f] border border-gray-200 dark:border-green-500/20 shadow-2xl"
        style={{ boxShadow: `0 0 50px ${tool.accent}15` }}
        initial={{ scale: 0.9, opacity: 0, y: 30 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.9, opacity: 0, y: 30 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
      >
        {/* Header bar */}
        <motion.div
          className="h-1 rounded-t-2xl"
          style={{ background: `linear-gradient(90deg, transparent, ${tool.accent}, transparent)` }}
        />

        <div className="p-6">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 hover:text-gray-900 dark:hover:text-white transition-colors z-10"
          >
            <FiX size={16} />
          </button>

          {/* Tool header */}
          <div className="flex items-center gap-3 mb-4">
            <span className="text-3xl">{tool.emoji}</span>
            <div>
              <h3 className="text-xl font-black text-gray-900 dark:text-white">{tool.name}</h3>
              <span
                className="px-2 py-0.5 rounded text-[10px] font-bold uppercase"
                style={{ background: `${tool.accent}15`, color: tool.accent }}
              >
                {tool.category.replace("_", " ")}
              </span>
            </div>
          </div>

          {/* Full description */}
          <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed mb-5">
            {tool.fullDescription}
          </p>

          {/* Parameters table */}
          <div className="mb-5">
            <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Parameters</h4>
            <div className="flex flex-wrap gap-2">
              {tool.params.map((param, idx) => (
                <div
                  key={idx}
                  className="px-3 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800"
                >
                  <code className="text-xs font-mono" style={{ color: tool.accent }}>{param}</code>
                </div>
              ))}
            </div>
          </div>

          {/* Code example */}
          <div className="mb-5">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest">Example</h4>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 transition-colors"
              >
                {copied ? <><FiCheck size={10} className="text-green-400" /> Copied</> : <><FiCopy size={10} /> Copy</>}
              </button>
            </div>
            <div className="p-3 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-gray-800 overflow-x-auto">
              <pre className="text-xs font-mono leading-relaxed text-gray-700 dark:text-gray-300">
                <code
                  className="syntax-code"
                  dangerouslySetInnerHTML={{ __html: highlightCode(runnableCode, "python") }}
                />
              </pre>
            </div>
          </div>

          {/* Related tools */}
          {relatedTools.length > 0 && (
            <div>
              <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Related Tools</h4>
              <div className="flex flex-wrap gap-2">
                {relatedTools.map((rt, idx) => (
                  <span
                    key={idx}
                    className="px-2.5 py-1 rounded-full text-xs bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700"
                  >
                    {rt.emoji} {rt.name}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

/* ── Main ToolShowcase ── */

export default function ToolShowcase() {
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.05 });
  const [activeCategory, setActiveCategory] = useState("ALL");
  const [selectedTool, setSelectedTool] = useState<ToolData | null>(null);
  const [showAll, setShowAll] = useState(false);

  const filteredTools = activeCategory === "ALL"
    ? (showAll ? tools : tools.filter(t => heroToolNames.includes(t.name)))
    : tools.filter(t => t.category === activeCategory);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.04 } },
  };

  return (
    <>
      <section className="py-24 bg-white dark:bg-[#030f07] relative overflow-hidden" ref={ref}>
        <div className="absolute inset-0 grid-pattern" />
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />

        <Container className="relative z-10">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6 }}
            className="text-center mb-10"
          >
            <motion.span
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-600 dark:text-green-400 text-sm font-semibold mb-6"
              whileHover={{ borderColor: "rgba(0,255,136,0.6)" }}
            >
              Built-in Tools
            </motion.span>
            <h2 className="text-4xl md:text-5xl font-black mb-4 text-gray-900 dark:text-white">
              <span className="gradient-text">66 Tools</span> Ready to Use
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              From finance, data science, DevOps, and academic research to news, social, translation,
              QR codes, OCR, audio transcription, image analysis, document parsing (PDF/DOCX/Excel),
              geo/weather, and email/webhook communication — everything your agent needs, built in.{" "}
              <span className="text-green-500 font-semibold">14 new tools landed in v0.2.6.</span>
            </p>
          </motion.div>

          {/* Category tabs */}
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex flex-wrap justify-center gap-2 mb-10"
          >
            {categories.map((cat) => {
              const isActive = activeCategory === cat.key;
              const count = cat.key === "ALL" ? tools.length : tools.filter(t => t.category === cat.key).length;
              return (
                <motion.button
                  key={cat.key}
                  onClick={() => { setActiveCategory(cat.key); if (cat.key !== "ALL") setShowAll(false); }}
                  className={`relative px-4 py-2 rounded-full text-sm font-semibold transition-all ${
                    isActive
                      ? "text-black"
                      : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/50"
                  }`}
                  style={isActive ? {
                    background: `linear-gradient(135deg, ${cat.accent}, ${cat.accent}cc)`,
                    boxShadow: `0 0 20px ${cat.accent}30`,
                  } : {}}
                  whileHover={{ scale: 1.05, y: -1 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {cat.label}
                  <span className={`ml-1.5 text-[10px] ${isActive ? "text-black/60" : "text-gray-400"}`}>
                    {count}
                  </span>
                  {isActive && (
                    <motion.div
                      className="absolute -bottom-0.5 left-1/4 right-1/4 h-0.5 rounded-full"
                      style={{ background: "rgba(0,0,0,0.3)" }}
                      layoutId="activeTab"
                      transition={{ type: "spring", stiffness: 400, damping: 30 }}
                    />
                  )}
                </motion.button>
              );
            })}
          </motion.div>

          {/* Tool cards grid */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeCategory + (showAll ? "-all" : "-hero")}
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              exit={{ opacity: 0, transition: { duration: 0.2 } }}
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
            >
              {filteredTools.map((tool) => (
                <ToolCard
                  key={tool.name}
                  tool={tool}
                  onClick={() => setSelectedTool(tool)}
                />
              ))}
            </motion.div>
          </AnimatePresence>

          {/* View All / Show Less buttons */}
          {activeCategory === "ALL" && !showAll && (
            <motion.div
              className="text-center mt-8"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <motion.button
                onClick={() => setShowAll(true)}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-bold text-sm border border-green-500/30 bg-green-500/5 text-green-600 dark:text-green-400 hover:bg-green-500/10 hover:border-green-500/50 transition-all"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
              >
                View All {tools.length} Tools
                <FiChevronDown size={16} />
              </motion.button>
            </motion.div>
          )}

          {activeCategory === "ALL" && showAll && (
            <motion.div
              className="text-center mt-8"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <motion.button
                onClick={() => setShowAll(false)}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-full font-bold text-sm border border-gray-300 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-green-500/50 transition-all"
                whileHover={{ scale: 1.05 }}
              >
                Show Less
                <FiChevronUp size={16} />
              </motion.button>
            </motion.div>
          )}
        </Container>
      </section>

      {/* Tool detail modal */}
      <AnimatePresence>
        {selectedTool && (
          <ToolDetail
            tool={selectedTool}
            onClose={() => setSelectedTool(null)}
          />
        )}
      </AnimatePresence>
    </>
  );
}
