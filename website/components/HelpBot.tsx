"use client";

import { useState, useEffect, useRef, useCallback, type ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FiMessageCircle, FiX, FiSend, FiMail, FiGithub, FiChevronRight, FiSearch, FiArrowLeft } from "react-icons/fi";
import { usePyPIVersion } from "./PyPIVersion";
import { highlightCode } from "./syntaxHighlight";

// ---------------------------------------------------------------------------
// FAQ database – every topic pulled from the actual site content
// ---------------------------------------------------------------------------
interface FAQ {
  id: number;
  category: string;
  question: string;
  answer: string;
  tags: string[]; // broad semantic tags used for matching
}

const faqs: FAQ[] = [
  // ── Installation & Setup ──────────────────────────────────────────────────
  {
    id: 1,
    category: "Installation",
    question: "How do I install effGen?",
    answer: "Install effGen with pip:\n\n```bash\npip install -U effgen\n```\n\nFor optional backends, install only the extras you need:\n\n```bash\npip install \"effgen[vllm]\"       # local CUDA throughput\npip install \"effgen[cerebras]\"   # v0.2.1 Cerebras\npip install \"effgen[groq]\"       # v0.2.3 Groq\npip install \"effgen[together]\"   # v0.2.3 Together AI\npip install \"effgen[fireworks]\"  # v0.2.3 Fireworks\npip install \"effgen[replicate]\"  # v0.2.3 Replicate\npip install \"effgen[hf]\"         # v0.2.3 HuggingFace Inference\npip install \"effgen[all]\"        # most extras\n```\n\n`effgen doctor` is available in v0.2.3+ for checking cloud provider API keys. On v0.2.1, set the relevant environment variables directly.",
    tags: ["install", "pip", "setup", "getting started", "installation", "how to install", "download", "package manager"],
  },
  {
    id: 2,
    category: "Installation",
    question: "What are the system requirements?",
    answer: "Minimum requirements:\n\n- **Python**: 3.10+\n- **RAM**: 8 GB (16 GB recommended)\n- **GPU VRAM**: 4 GB for 3B models, 8 GB for 7B models (4-bit quantization)\n- **CUDA**: 11.8+ for GPU support\n\nCPU-only mode works but is noticeably slower. Cloud providers only need the matching API key and optional provider extra.",
    tags: ["requirements", "system", "hardware", "gpu", "ram", "cuda", "minimum", "specs", "cpu", "vram", "python version"],
  },
  {
    id: 3,
    category: "Installation",
    question: "How do I upgrade effGen to the latest version?",
    answer: "Upgrade using pip:\n\n```bash\npip install --upgrade effgen\n```\n\nTo see your current version:\n\n```bash\npip show effgen\n```\n\nCheck `NEWS.md` (release notes) and `CHANGELOG.md` in the GitHub repo for what changed between versions.",
    tags: ["upgrade", "update", "version", "latest", "pip upgrade", "changelog", "release", "new version"],
  },
  {
    id: 4,
    category: "Installation",
    question: "I'm getting a version conflict / dependency error on install",
    answer: "Try installing in a fresh virtual environment:\n\n```bash\npython -m venv effgen-env\nsource effgen-env/bin/activate   # Windows: effgen-env\\Scripts\\activate\npip install effgen\n```\n\nIf the conflict persists, pin the specific version:\n\n```bash\npip install effgen==<version>\n```\n\nYou can find all released versions on PyPI. If it still fails, open a GitHub issue with your full error log.",
    tags: ["version conflict", "dependency error", "install error", "virtual environment", "venv", "pip error", "conflict", "broken install", "fix install"],
  },

  // ── Models & Backends ─────────────────────────────────────────────────────
  {
    id: 5,
    category: "Models",
    question: "What models does effGen support?",
    answer: "For v0.2.1, local engines work directly and cloud loading supports `provider=`. OpenAI, Anthropic, and Gemini are also auto-detected from model names; Cerebras should use explicit `provider=\"cerebras\"`:\n\n```python\nmodel = load_model(\"gpt-5.4-nano\", provider=\"openai\")\nmodel = load_model(\"llama3.1-8b\", provider=\"cerebras\")\n```\n\nv0.2.3 expanded this to **14 inference backends**: Transformers, vLLM, MLX, MLX-VLM, GGUF, OpenAI, Anthropic, Gemini, Cerebras, Groq, Together AI, Fireworks, Replicate, and HuggingFace Inference. Provider-prefixed IDs such as `groq:...` are v0.2.3+.",
    tags: ["models", "supported", "gpt", "claude", "qwen", "llama", "which models", "model support", "backend", "deepseek", "mistral", "gemini", "openai", "anthropic"],
  },
  {
    id: 6,
    category: "Models",
    question: "How do I load a local model?",
    answer: "Use `load_model` with your Hugging Face model name:\n\n```python\nfrom effgen import load_model\n\n# Default (Transformers engine)\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\n\n# With 4-bit quantization to save VRAM\nmodel = load_model(\n    \"Qwen/Qwen2.5-7B-Instruct\",\n    quantization=\"4bit\"\n)\n```\n\nAny model on Hugging Face that supports the `transformers` library can be loaded this way.",
    tags: ["load model", "local model", "hugging face", "transformers", "quantization", "4bit", "model loading", "download model"],
  },
  {
    id: 7,
    category: "Models",
    question: "Can I use an API-based model like GPT-4 or Claude?",
    answer: "Yes. Pass `provider=` for version-compatible cloud loading:\n\n```python\nfrom effgen import load_model\n\n# v0.2.1-compatible\nmodel = load_model(\"gpt-5.4-nano\", provider=\"openai\")\nmodel = load_model(\"claude-3-5-sonnet-20241022\", provider=\"anthropic\")\nmodel = load_model(\"gemini-2.0-flash-lite\", provider=\"gemini\")\nmodel = load_model(\"llama3.1-8b\", provider=\"cerebras\")\n\n# v0.2.3+ provider prefix\nmodel = load_model(\"groq:llama-3.3-70b-versatile\")\n```\n\nFor v0.2.1 set `OPENAI_API_KEY`, optional `OPENAI_ORG_ID`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, and `CEREBRAS_API_KEY`. Groq, Together, Fireworks, Replicate, HF, ProviderRegistry, and `effgen doctor` are v0.2.3+.",
    tags: ["api model", "gpt-4", "claude", "gemini", "api key", "openai api", "anthropic api", "remote model", "cloud model"],
  },
  {
    id: 8,
    category: "Models",
    question: "Which model size should I use?",
    answer: "It depends on your hardware and task complexity:\n\n- **1.5B**: about 2 GB VRAM; good for simple Q&A and classification\n- **3B**: about 4 GB VRAM; good for general-purpose agents\n- **7B**: about 8 GB VRAM with 4-bit quantization; good for harder reasoning and coding\n- **14B+**: about 16 GB+ VRAM; best for research-grade accuracy\n\nStart with **Qwen2.5-3B-Instruct** if you're unsure — it's a good balance of speed and quality.",
    tags: ["model size", "which model", "3b", "7b", "1.5b", "choose model", "best model", "small model", "slm"],
  },

  // ── Core Concepts & Architecture ─────────────────────────────────────────
  {
    id: 9,
    category: "Concepts",
    question: "How do I create my first agent?",
    answer: "Here's a complete minimal example:\n\n```python\nfrom effgen import Agent, load_model\nfrom effgen.core.agent import AgentConfig\nfrom effgen.tools.builtin import Calculator\n\nmodel = load_model(\"Qwen/Qwen2.5-3B-Instruct\")\n\nagent = Agent(config=AgentConfig(\n    name=\"my_agent\",\n    model=model,\n    tools=[Calculator()]\n))\n\nresult = agent.run(\"What is 25 * 17?\")\nprint(result.output)  # 425\n```\n\nThat's it — three imports, one model load, one agent, one run.",
    tags: ["create agent", "first agent", "example", "quick start", "tutorial", "basic", "hello world", "getting started", "beginner", "simple agent"],
  },
  {
    id: 10,
    category: "Concepts",
    question: "What is the ReAct loop and how does it work?",
    answer: "ReAct (Reasoning + Acting) is the core loop every effGen agent uses:\n\n1. **Think** – The model reasons about the current state and the goal\n2. **Act** – It decides which tool to call (or whether to answer directly)\n3. **Observe** – The tool runs and its output is fed back\n4. **Repeat** – Steps 1-3 continue until the task is done or max iterations is reached\n\nThis lets agents solve complex, multi-step tasks that require multiple tool calls in sequence.",
    tags: ["react", "reasoning", "acting", "loop", "how it works", "paradigm", "architecture", "think act observe", "agent loop", "reasoning loop"],
  },
  {
    id: 11,
    category: "Concepts",
    question: "What is task decomposition?",
    answer: "Task decomposition automatically breaks a complex user request into smaller sub-tasks:\n\n- effGen scores the **complexity** of the input\n- If complexity is high, it spawns **specialized sub-agents** for each piece\n- Sub-agents run (optionally in parallel) and their results are synthesized\n\nThis means you can hand the agent a big, vague request like *\"Research AI trends and write a report\"* and it will handle the breakdown for you.",
    tags: ["task decomposition", "sub-agent", "complex task", "automatic breakdown", "sub tasks", "splitting", "orchestration", "complexity"],
  },
  {
    id: 12,
    category: "Concepts",
    question: "How does agent memory work?",
    answer: "Enable memory in the agent config:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\n\nconfig = AgentConfig(\n    name=\"memory_agent\",\n    model=model,\n    enable_memory=True\n)\n\nagent = Agent(config=config)\nagent.run(\"My name is Alice\")\nagent.run(\"What is my name?\")  # \"Alice\"\n```\n\neffGen supports both **short-term memory** (within a session) and **long-term memory** (persisted across sessions). Short-term is on by default when memory is enabled.",
    tags: ["memory", "remember", "context", "conversation", "history", "short-term", "long-term", "persist", "state", "session"],
  },
  {
    id: 13,
    category: "Concepts",
    question: "What are the supported agent-to-agent protocols?",
    answer: "effGen supports three major protocols for agent communication:\n\n1. **MCP (Model Context Protocol)** – Pass context and tools between models\n2. **A2A (Agent-to-Agent)** – Direct communication between specialized agents\n3. **ACP (Agent Communication Protocol)** – Standardized message passing for multi-agent pipelines\n\nThese let you build complex, collaborative systems where agents hand off work to each other seamlessly.",
    tags: ["protocol", "mcp", "a2a", "acp", "agent communication", "agent to agent", "model context protocol", "inter-agent"],
  },

  // ── Tools ─────────────────────────────────────────────────────────────────
  {
    id: 14,
    category: "Tools",
    question: "What built-in tools are available?",
    answer: "effGen ships with **66 built-in tools** across many categories. v0.2.6 added 14 new tools (docs, media, comms); v0.2.5 added 13 free / no-auth tools. (As of **v0.2.10** the `CodeExecutor` runs inside a sandbox by default — DockerSandbox if available, else an unprivileged-namespace subprocess.) Highlights:\n\n- **Computation** – Calculator, DateTimeTool, StatsTool\n- **Code** – CodeExecutor (sandboxed by default — v0.2.10), PythonREPL\n- **Information** – WebSearch, WikipediaTool, URLFetchTool, Retrieval (RAG), AgenticSearch, StackOverflowTool, GitHubTool\n- **Academic (v0.2.5)** – ArXivTool, PubMedTool, SemanticScholarTool\n- **News & RSS (v0.2.5)** – RSSFeedTool, NewsTool\n- **YouTube (v0.2.5)** – YouTubeTranscriptTool, YouTubeMetadataTool\n- **Social (v0.2.5)** – RedditTool, HackerNewsTool\n- **Translation / Language (v0.2.5)** – TranslateTool (LibreTranslate + argostranslate offline fallback), LanguageDetectTool (offline, 55+ languages)\n- **QR Codes (v0.2.5)** – QRGenerateTool, QRReadTool (fully local)\n- **OCR (v0.2.6)** – OCRTool (Tesseract local + OCR.space fallback)\n- **Audio (v0.2.6)** – AudioTranscribeTool (faster-whisper + HF Inference fallback)\n- **Image (v0.2.6)** – ImageInfoTool (Pillow, zero network), ImageCaptionTool (Gemini / OpenAI / MLX-VLM via router)\n- **Documents (v0.2.6)** – PDFTool (pypdf + pdfplumber), DOCXTool (python-docx), ExcelTool (openpyxl + pandas)\n- **Geo / Weather (v0.2.6)** – WeatherTool (Open-Meteo, no auth), GeocodeTool (Nominatim, rate-limited), MapsTool (OSM static)\n- **Email (v0.2.6, live)** – EmailSMTPTool (TLS-on by default), EmailIMAPTool (read/search inbox)\n- **Webhooks (v0.2.6)** – SlackWebhookTool, DiscordWebhookTool (URLs redacted in logs)\n- **Data Processing** – DataFrameTool, PlotTool, JSONTool, TextProcessingTool\n- **File / System / DevOps** – FileOperations, BashTool, GitTool (read-only), DockerTool (read-only), SystemInfoTool, HTTPTool\n- **Finance** – StockPriceTool, CurrencyConverterTool, CryptoTool (all carry a 'not financial advice' disclaimer)\n- **Knowledge & External** – WolframAlphaTool\n- **Communication (draft-only, legacy)** – EmailDraftTool, SlackDraftTool, NotificationTool\n- **Provider-native** – OpenAI (web_search, code_interpreter, file_search), Gemini (GoogleSearch, UrlContext, CodeExecution), experimental Anthropic (Bash, TextEditor, Computer)\n\nYou can also build your own custom tools.",
    tags: ["tools", "built-in", "available tools", "calculator", "web search", "code executor", "file", "retrieval", "python repl", "wikipedia", "list tools"],
  },
  {
    id: 15,
    category: "Tools",
    question: "How do I create a custom tool?",
    answer: "Extend `BaseTool` and supply `ToolMetadata` in `__init__`:\n\n```python\nfrom effgen.tools import BaseTool\nfrom effgen.tools.base_tool import ToolMetadata, ToolCategory, ParameterSpec, ParameterType\n\nclass MyTool(BaseTool):\n    def __init__(self):\n        super().__init__(\n            metadata=ToolMetadata(\n                name=\"my_tool\",\n                description=\"Does something useful\",\n                category=ToolCategory.COMPUTATION,\n                parameters=[\n                    ParameterSpec(\n                        name=\"param\",\n                        type=ParameterType.STRING,\n                        description=\"Input value\",\n                        required=True,\n                    ),\n                ],\n            )\n        )\n\n    async def _execute(self, param: str, **kwargs) -> dict:\n        return {\"result\": f\"Processed: {param}\"}\n```\n\nThen pass it to your agent:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\n\nagent = Agent(config=AgentConfig(\n    name=\"my_agent\",\n    model=model,\n    tools=[MyTool()],\n))\n```\n\nThe agent uses the `description` and `parameters` from `ToolMetadata` to decide when/how to call the tool.",
    tags: ["custom tool", "create tool", "extend", "basetool", "new tool", "own tool", "build tool", "tool development", "write tool"],
  },
  {
    id: 16,
    category: "Tools",
    question: "How do I use the Calculator tool?",
    answer: "The Calculator is a built-in tool — just include it and let the agent decide when to use it:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\nfrom effgen.tools.builtin import Calculator\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\n\nagent = Agent(config=AgentConfig(\n    name=\"calculator_agent\",\n    model=model,\n    tools=[Calculator()]\n))\n\nresult = agent.run(\"What is 24344 * 334?\")\nprint(result.output)  # 8,130,896\n```\n\nIt handles arithmetic, unit conversions, and basic algebra automatically.",
    tags: ["calculator", "math", "arithmetic", "unit conversion", "calculator tool", "numbers"],
  },
  {
    id: 17,
    category: "Tools",
    question: "How do I use the CodeExecutor / PythonREPL tool?",
    answer: "**CodeExecutor** runs code in an isolated Docker sandbox:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\nfrom effgen.tools.builtin import CodeExecutor\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\n\nagent = Agent(config=AgentConfig(\n    name=\"code_agent\",\n    model=model,\n    tools=[CodeExecutor()]\n))\n\nresult = agent.run(\"Write and run a script that prints the first 10 Fibonacci numbers\")\n```\n\n**PythonREPL** keeps state between calls (variables persist):\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\nfrom effgen.tools.builtin import PythonREPL\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\n\nagent = Agent(config=AgentConfig(\n    name=\"repl_agent\",\n    model=model,\n    tools=[PythonREPL()]\n))\n```\n\nUse PythonREPL for multi-step data analysis where variables need to carry over.",
    tags: ["code executor", "python repl", "run code", "execute code", "sandbox", "docker", "coding tool", "script"],
  },
  {
    id: 18,
    category: "Tools",
    question: "How do I use WebSearch or Wikipedia tools?",
    answer: "Just add them to your agent's tool list:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\nfrom effgen.tools.builtin import WebSearch, WikipediaTool\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\n\nagent = Agent(config=AgentConfig(\n    name=\"research_agent\",\n    model=model,\n    tools=[WebSearch(), WikipediaTool()]\n))\n\nresult = agent.run(\"What are the latest developments in quantum computing?\")\n```\n\n- **WebSearch** queries DuckDuckGo and returns top results\n- **WikipediaTool** fetches and summarizes the relevant Wikipedia article\n\nThe agent automatically picks the right tool based on your question.",
    tags: ["web search", "wikipedia", "search", "internet", "online", "browsing", "research tool", "duckduckgo"],
  },
  {
    id: 19,
    category: "Tools",
    question: "How do I use FileOperations?",
    answer: "FileOperations lets the agent read and write files:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\nfrom effgen.tools.builtin import FileOperations\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\n\nagent = Agent(config=AgentConfig(\n    name=\"file_agent\",\n    model=model,\n    tools=[FileOperations()]\n))\n\nresult = agent.run(\"Read sales_data.csv and summarize the key trends\")\n```\n\nSupports common formats including CSV, JSON, TXT, and more. File access is sandboxed to a configured directory for safety.",
    tags: ["file operations", "read file", "write file", "csv", "file tool", "file access", "read write"],
  },
  {
    id: 20,
    category: "Tools",
    question: "How does RAG / Retrieval tool work?",
    answer: "The Retrieval tool enables **Retrieval-Augmented Generation** — it searches your own documents using semantic similarity:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\nfrom effgen.tools.builtin import Retrieval\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\nretrieval = Retrieval(knowledge_base_path=\"./my_docs\")\n\nagent = Agent(config=AgentConfig(\n    name=\"retrieval_agent\",\n    model=model,\n    tools=[retrieval]\n))\n\nresult = agent.run(\"What does our refund policy say?\")\n```\n\nIt embeds your documents at startup, then does a vector similarity search at query time to find the most relevant chunks.",
    tags: ["rag", "retrieval", "semantic search", "vector search", "documents", "knowledge base", "retrieval augmented generation", "embeddings"],
  },

  // ── Performance & vLLM ────────────────────────────────────────────────────
  {
    id: 21,
    category: "Performance",
    question: "How do I use vLLM for faster inference?",
    answer: "Switch the engine to `vllm` in `load_model`:\n\n```python\nfrom effgen import load_model\n\nmodel = load_model(\n    \"Qwen/Qwen2.5-7B-Instruct\",\n    engine=\"vllm\",\n    tensor_parallel_size=2  # spread across 2 GPUs\n)\n```\n\nMake sure you installed the vLLM extras:\n\n```bash\npip install effgen[vllm]\n```\n\nvLLM delivers **5–10× faster inference** compared to standard Transformers thanks to continuous batching and PagedAttention.",
    tags: ["vllm", "fast", "faster", "performance", "speed", "inference", "gpu", "production", "tensor parallel", "paged attention", "optimize"],
  },
  {
    id: 22,
    category: "Performance",
    question: "How do I use multiple GPUs?",
    answer: "Set `tensor_parallel_size` to the number of GPUs you want to use:\n\n```python\nmodel = load_model(\n    \"Qwen/Qwen2.5-14B-Instruct\",\n    engine=\"vllm\",\n    tensor_parallel_size=4  # 4 GPUs\n)\n```\n\neffGen handles the parallelism automatically via vLLM. You need at least as many GPUs as the value you set, and each GPU must have enough VRAM to hold its shard of the model.",
    tags: ["multiple gpu", "multi gpu", "tensor parallel", "gpu scaling", "parallel gpu", "distributed", "4 gpu", "2 gpu"],
  },
  {
    id: 23,
    category: "Performance",
    question: "How do I use quantization to reduce memory usage?",
    answer: "Pass `quantization` to `load_model`:\n\n```python\nmodel = load_model(\n    \"Qwen/Qwen2.5-7B-Instruct\",\n    quantization=\"4bit\"   # also supports \"8bit\"\n)\n```\n\n- **4-bit**: ~60% less VRAM, slight quality trade-off — great for consumer GPUs\n- **8-bit**: ~30% less VRAM, minimal quality loss\n\nQuantization is automatic — no extra libraries needed.",
    tags: ["quantization", "4bit", "8bit", "memory", "reduce memory", "vram", "low memory", "optimize memory"],
  },

  // ── Multi-Agent Systems ───────────────────────────────────────────────────
  {
    id: 24,
    category: "Multi-Agent",
    question: "How do I build a multi-agent system?",
    answer: "Use a **WorkflowDAG** to wire specialized agents together — one node per agent, edges to pass data between them:\n\n```python\nfrom effgen import load_model\nfrom effgen.presets import create_agent\nfrom effgen.core.workflow import WorkflowDAG, WorkflowNode\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\", engine=\"vllm\")\n\nresearcher = create_agent(\"research\", model)\nanalyst    = create_agent(\"math\",     model)\nwriter     = create_agent(\"general\",  model)\n\ndag = WorkflowDAG(name=\"research_to_report\")\ndag.add_node(WorkflowNode(id=\"research\", agent=researcher, output_key=\"facts\"))\ndag.add_node(WorkflowNode(id=\"analyze\",  agent=analyst,    input_keys=[\"facts\"]))\ndag.add_node(WorkflowNode(id=\"write\",    agent=writer,     input_keys=[\"facts\", \"analysis\"], output_key=\"report\"))\n\ndag.connect(\"research\", \"analyze\", key=\"facts\")\ndag.connect(\"research\", \"write\",   key=\"facts\")\ndag.connect(\"analyze\",  \"write\",   key=\"analysis\")\n\nresult = dag.run(initial_inputs={\"research\": \"Research AI trends and write a report\"})\nprint(result.outputs[\"write\"])\n```\n\nIndependent nodes auto-parallelise via `asyncio.gather`. For looser pub/sub patterns, see `MessageBus`. Built-in single-agent task decomposition is also available via `enable_sub_agents=True` on `AgentConfig`.",
    tags: ["multi agent", "multiple agents", "orchestration", "multi-agent system", "agent collaboration", "mas", "specialized agents"],
  },
  {
    id: 25,
    category: "Multi-Agent",
    question: "How does agent orchestration work?",
    answer: "effGen has two complementary orchestration paths:\n\n1. **Built-in sub-agents** — set `enable_sub_agents=True` on `AgentConfig` and a single agent will decompose complex tasks, spawn specialists in parallel, and synthesise the result. Tunable via `router_config`, `sub_agent_config`, and `max_sub_agent_depth` (default 3).\n2. **`WorkflowDAG`** — for known pipelines you define explicitly, with cycle detection (Kahn's topological sort), conditional edges, auto-parallel execution at each level, and YAML support (`effgen workflow run/validate`).\n\nFor looser pub/sub between agents, use `MessageBus`; share mutable state with `SharedState`; serve under load with `AgentPool` behind the API server.",
    tags: ["orchestration", "orchestrator", "routing", "task routing", "agent routing", "how orchestration works", "sub-task"],
  },

  // ── Configuration & Deployment ────────────────────────────────────────────
  {
    id: 26,
    category: "Configuration",
    question: "How do I configure effGen with YAML?",
    answer: "effGen supports YAML configuration files via `ConfigLoader`:\n\n```yaml\n# config.yaml\nmodel:\n  name: Qwen/Qwen2.5-7B-Instruct\n  engine: vllm\n  quantization: 4bit\n\nagent:\n  name: my_agent\n  max_iterations: 10\n  enable_memory: true\n  tools:\n    - calculator\n    - web_search\n    - code_executor\n```\n\nLoad and assemble in Python:\n\n```python\nfrom effgen import Agent, AgentConfig, ConfigLoader, load_model\nfrom effgen.tools.builtin import Calculator, WebSearch, CodeExecutor\n\nloader = ConfigLoader()\nloader.load_config(\"config.yaml\")  # populates loader.config\nmodel = load_model(\n    loader.get(\"model.name\"),\n    engine=loader.get(\"model.engine\"),\n    quantization=loader.get(\"model.quantization\"),\n)\n\nagent = Agent(config=AgentConfig(\n    name=loader.get(\"agent.name\"),\n    model=model,\n    max_iterations=loader.get(\"agent.max_iterations\", 10),\n    enable_memory=loader.get(\"agent.enable_memory\", True),\n    tools=[Calculator(), WebSearch(), CodeExecutor()],\n))\n```\n\nGreat for production deployments where you want reproducible, version-controlled configs. You can also use `effgen run --config config.yaml` from the CLI.",
    tags: ["yaml", "config", "configuration", "config file", "yaml config", "settings", "reproducible"],
  },
  {
    id: 27,
    category: "Configuration",
    question: "How do I set max iterations for an agent?",
    answer: "Set `max_iterations` in the agent config:\n\n```python\nfrom effgen import AgentConfig, load_model\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\n\nconfig = AgentConfig(\n    name=\"my_agent\",\n    model=model,\n    max_iterations=15  # default is 10\n)\n```\n\nWhen the agent hits the limit it stops and returns whatever partial result it has. Inspect `result.iterations` and `result.success` on the returned `AgentResponse` to detect the cutoff and recover gracefully.",
    tags: ["max iterations", "iteration limit", "max steps", "stop condition", "iteration cap", "limit"],
  },
  {
    id: 28,
    category: "Configuration",
    question: "How do I use secret management / environment variables?",
    answer: "API keys and secrets should never be hardcoded. For v0.2.1, use environment variables or a `.env` file:\n\n```bash\nexport OPENAI_API_KEY=\"sk-...\"\nexport OPENAI_ORG_ID=\"org_...\"  # optional\nexport ANTHROPIC_API_KEY=\"...\"\nexport GOOGLE_API_KEY=\"...\"\nexport CEREBRAS_API_KEY=\"...\"\n```\n\nv0.2.3+ adds `GROQ_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`, `REPLICATE_API_TOKEN`, `HF_TOKEN`, and the ProviderRegistry doctor command:\n\n```bash\neffgen doctor\neffgen doctor --json\n```",
    tags: ["secrets", "api key", "environment variable", "env", "dotenv", "secret management", "security", "credentials"],
  },

  // ── Error Handling & Debugging ────────────────────────────────────────────
  {
    id: 29,
    category: "Troubleshooting",
    question: "How do I handle errors and exceptions?",
    answer: "Agent errors surface through `AgentResponse` rather than raised exceptions — `agent.run()` catches and reports failures so you can keep going. Inspect the response:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\nagent = Agent(config=AgentConfig(\n    name=\"error_demo\",\n    model=model,\n    max_iterations=5,\n))\n\nresult = agent.run(\"complex task\")\n\nif not result.success:\n    print(\"Failed:\", result.metadata.get(\"error\") or result.output)\n\nif result.iterations >= agent.config.max_iterations:\n    print(\"Hit max_iterations — partial:\", result.output)\n\n# Guardrail blocks are also surfaced via metadata:\nif result.metadata.get(\"guardrail_blocked\"):\n    print(\"Blocked:\", result.metadata[\"guardrail_reason\"])\n```\n\nFor the API client, typed exceptions live in `effgen.client`: `EffGenClientError`, `EffGenAPIError`, `EffGenAuthError`, `EffGenRateLimitError`, `EffGenServerError`, `EffGenConnectionError`, `EffGenTimeoutError`.",
    tags: ["error", "exception", "handle error", "try catch", "debug", "troubleshoot", "exceptions", "error handling", "crash", "fix error"],
  },
  {
    id: 30,
    category: "Troubleshooting",
    question: "My agent keeps looping / hitting max iterations. What do I do?",
    answer: "A few things to check:\n\n1. **Increase max_iterations** if the task genuinely needs more steps\n2. **Simplify the prompt** – vague instructions cause more back-and-forth\n3. **Use a larger model** – bigger models reason better and loop less\n4. **Check tool outputs** – if a tool is returning unhelpful results the agent can get stuck\n5. **Enable logging** to see exactly what the agent is doing each iteration\n\nIf none of that helps, open a GitHub issue with your config and logs.",
    tags: ["looping", "max iterations", "stuck", "infinite loop", "too many iterations", "agent stuck", "not finishing"],
  },
  {
    id: 31,
    category: "Troubleshooting",
    question: "How do I enable logging / debug mode?",
    answer: "Use the logging helper or request a debug trace for a specific run:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\nfrom effgen.utils.logging import set_log_level\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\nagent = Agent(config=AgentConfig(name=\"debug_agent\", model=model))\n\nset_log_level(\"DEBUG\")\nresult = agent.run(\"Debug this task\", debug=True)\ntrace = result.metadata.get(\"debug_trace\")\n```\n\nFor CLI runs, use:\n\n```bash\neffgen run \"Debug this task\" --verbose\n```\n\nThe debug trace records the ReAct loop iterations and tool calls when available.",
    tags: ["logging", "debug", "log level", "verbose", "trace", "debug mode", "monitor", "diagnostics"],
  },
  {
    id: 32,
    category: "Troubleshooting",
    question: "I'm getting an out-of-memory (OOM) error",
    answer: "Try these in order:\n\n1. **Use quantization**: `quantization=\"4bit\"` cuts VRAM by ~60%\n2. **Use a smaller model**: e.g. 3B instead of 7B\n3. **Enable vLLM**: it uses PagedAttention which is far more memory-efficient\n4. **Reduce batch size** if you're running multiple requests\n5. **Close other GPU processes** that might be eating VRAM\n\nIf you're already doing all of this and still OOMing, you may need a GPU with more VRAM.",
    tags: ["out of memory", "oom", "memory error", "not enough memory", "vram full", "gpu memory", "memory issue"],
  },
  {
    id: 33,
    category: "Troubleshooting",
    question: "CUDA is not available / no GPU detected",
    answer: "1. Make sure your NVIDIA driver is up to date\n2. Install the right PyTorch version for your CUDA toolkit:\n\n```bash\npip install torch --index-url https://download.pytorch.org/whl/cu118\n```\n\n3. Verify with Python:\n\n```python\nimport torch\nprint(torch.cuda.is_available())  # should print True\nprint(torch.cuda.get_device_name(0))\n```\n\n4. If you're on a cloud instance, make sure you selected a GPU-enabled machine type.\n\neffGen falls back to CPU automatically if no GPU is found, but it will be slow.",
    tags: ["cuda", "no gpu", "gpu not found", "cuda not available", "pytorch", "driver", "nvidia", "gpu setup"],
  },

  // ── Examples & Use Cases ──────────────────────────────────────────────────
  {
    id: 34,
    category: "Examples",
    question: "How do I build an AI Code Assistant?",
    answer: "effGen has a full example for this:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\nfrom effgen.tools.builtin import CodeExecutor, FileOperations, WebSearch\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\", engine=\"vllm\", tensor_parallel_size=2)\n\nassistant = Agent(config=AgentConfig(\n    model=model,\n    name=\"code_assistant\",\n    system_prompt=\"You are an expert coding assistant. Provide clear, efficient, well-documented code.\",\n    tools=[CodeExecutor(), WebSearch(), FileOperations()],\n    enable_memory=True,\n))\n\nresult = assistant.run(\"Create a Python function for the longest palindromic substring.\")\nprint(result.output)\n```\n\nFull source: https://github.com/ctrl-gaurav/effGen/blob/main/examples/tools/coding_agent.py",
    tags: ["code assistant", "coding assistant", "code generation", "code example", "programming helper", "ide", "developer tool"],
  },
  {
    id: 35,
    category: "Examples",
    question: "How do I build a Research Agent?",
    answer: "The Research Agent gathers info from multiple sources and synthesizes a report:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\nfrom effgen.tools.builtin import Calculator, WebSearch, WikipediaTool\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\")\n\nresearcher = Agent(config=AgentConfig(\n    model=model,\n    name=\"research_agent\",\n    system_prompt=\"You are a thorough research assistant. Always verify facts and cite sources.\",\n    tools=[WebSearch(), WikipediaTool(), Calculator()],\n    enable_memory=True,\n    max_iterations=15,\n))\n\nresult = researcher.run(\"Research quantum computing breakthroughs in 2024.\")\n```\n\nFull source: https://github.com/ctrl-gaurav/effGen/blob/main/examples/web_retrieval/web_agent.py",
    tags: ["research agent", "research", "information gathering", "fact checking", "citations", "report generation"],
  },
  {
    id: 36,
    category: "Examples",
    question: "How do I build a Data Analysis Agent?",
    answer: "The Data Analysis Agent can clean, analyze, and visualize data:\n\n```python\nfrom effgen import Agent, AgentConfig, load_model\nfrom effgen.tools.builtin import Calculator, CodeExecutor, FileOperations\n\nmodel = load_model(\"Qwen/Qwen2.5-7B-Instruct\", engine=\"vllm\")\n\nanalyst = Agent(config=AgentConfig(\n    model=model,\n    name=\"data_analyst\",\n    system_prompt=\"You are an expert data scientist. Provide thorough analysis with visualizations.\",\n    tools=[CodeExecutor(), FileOperations(), Calculator()],\n    enable_memory=True,\n))\n\nresult = analyst.run(\"Analyze sales_data.csv: clean it, find trends, create charts.\")\n```\n\nFull source: https://github.com/ctrl-gaurav/effGen/blob/main/examples/advanced/data_processing_agent.py",
    tags: ["data analysis", "data science", "csv", "visualization", "pandas", "analytics", "data agent", "charts"],
  },

  // ── Production & Docker ───────────────────────────────────────────────────
  {
    id: 37,
    category: "Production",
    question: "How do I run effGen in Docker?",
    answer: "effGen's CodeExecutor already uses Docker sandboxing internally. For deploying your whole app:\n\n1. Use the official base image or build your own\n2. Install effGen inside the container\n3. Copy your config and scripts\n4. Mount any data volumes\n\nThe production infrastructure feature includes Docker-sandboxed execution, comprehensive logging, state persistence, and secret management out of the box.",
    tags: ["docker", "container", "deployment", "production", "sandbox", "containerize", "deploy"],
  },
  {
    id: 38,
    category: "Production",
    question: "How do I monitor my agents in production?",
    answer: "effGen includes built-in monitoring and metrics:\n\n- **Request logs** – every agent.run() is logged with timing\n- **Tool usage metrics** – which tools are called, how often, success/failure rates\n- **Iteration counts** – how many ReAct loops each task took\n- **Error tracking** – all exceptions are captured with context\n\nUse `set_log_level(\"INFO\")` for a clean production log stream, or `set_log_level(\"DEBUG\")` plus `agent.run(..., debug=True)` for full traces.",
    tags: ["monitoring", "metrics", "production", "logging", "observability", "track", "analytics", "performance monitoring"],
  },

  // ── Community & Contributing ──────────────────────────────────────────────
  {
    id: 39,
    category: "Community",
    question: "How do I contribute to effGen?",
    answer: "Contributions are welcome at every level:\n\n1. **Fork** the GitHub repo\n2. **Create a branch** for your change\n3. **Make your changes** – code, docs, examples, bug fixes all count\n4. **Submit a pull request**\n\nCheck the **CONTRIBUTING.md** in the repo for detailed guidelines. You can also report bugs or suggest features via GitHub Issues.",
    tags: ["contribute", "contribution", "pull request", "pr", "fork", "open source", "help out", "contributing guide"],
  },
  {
    id: 40,
    category: "Community",
    question: "Where can I get help or ask questions?",
    answer: "You have several options:\n\n- **This Help Bot** – ask away, it knows a lot!\n- **GitHub Issues** – github.com/ctrl-gaurav/effGen/issues\n- **Discord** – discord.com/invite/jacn9ed3\n- **Email** – gks@vt.edu\n- **Twitter/X** – @effGen_org\n\nThe community is active and friendly — don't hesitate to reach out.",
    tags: ["help", "support", "ask question", "where to ask", "community", "discord", "contact", "forum"],
  },
  {
    id: 41,
    category: "Community",
    question: "Where can I find examples and tutorials?",
    answer: "Examples and learning resources:\n\n1. **Examples page** – effgen.org/examples (six full example walkthroughs)\n2. **GitHub examples folder** – github.com/ctrl-gaurav/effGen/tree/main/examples\n3. **Quick Start** – the landing page walkthrough covers install to first agent in 3 steps\n4. **Example detail pages** – each example has a full code walkthrough\n\nExamples cover: Code Assistant, Research Agent, Data Analysis, Multi-Agent Systems, Weather and JSON, and RAG Knowledge Base.",
    tags: ["examples", "tutorials", "sample", "demo", "learn", "code examples", "use cases", "getting started guide"],
  },

  // ── Licensing & General ───────────────────────────────────────────────────
  {
    id: 42,
    category: "General",
    question: "What license is effGen under?",
    answer: "effGen is released under the **Apache 2.0 License** — you can use it in personal and commercial projects, modify it, and distribute it freely. Just keep the copyright notice and NOTICE file.\n\nSee the LICENSE file in the GitHub repo for the full text.",
    tags: ["license", "apache", "apache 2.0", "open source", "commercial", "copyright", "usage rights", "legal"],
  },
  {
    id: 46,
    category: "General",
    question: "What changed in effGen v0.2.1?",
    answer: "v0.2.1 added the Cerebras backend and modern OpenAI support.\n\nKey details:\n- Cerebras has 4 registered models, but only `llama3.1-8b` and `qwen-3-235b-a22b-instruct-2507` are reliably free-tier callable\n- Cerebras native tools are model-dependent; `zai-glm-4.7` is registered without native tool support\n- OpenAI has 35 registered models covering gpt-5/gpt-5.4/gpt-4.1/gpt-4o/chat legacy and o-series reasoning models\n- Use `load_model(\"gpt-5.4-nano\", provider=\"openai\")` or `load_model(\"llama3.1-8b\", provider=\"cerebras\")` for v0.2.1-compatible loading\n- `reasoning_effort` supports `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`; v0.2.1 validates membership and passes the value through\n- OpenAI structured outputs v2 uses `generate_structured()`, `to_openai_schema()`, `$ref` inlining, `additionalProperties: false`, and `ModelRefusalError`",
    tags: ["v0.2.1", "0.2.1", "cerebras", "openai", "structured outputs", "prompt caching", "reasoning"],
  },
  {
    id: 47,
    category: "General",
    question: "What changed in effGen v0.2.2?",
    answer: [
      "v0.2.2 modernized Gemini and Anthropic while keeping v0.2.1 calls backward-compatible.",
      "",
      "Key details:",
      "- Gemini moved to `google-genai>=1.0.0` and the `google.genai` namespace",
      "- Gemini gained Gemini 3.x/2.5/2.0 and Gemma registry entries, `thinking_budget`, `include_thoughts`, `grounding`, Files API inputs, and native tools",
      "- Anthropic gained Claude 4.x registry entries, extended thinking, prompt caching, typed streaming chunks, raw/redacted thinking preservation, and experimental computer-use tool specs",
      "- New fields default to safe values: `thinking_budget=None`, `include_thoughts=False`, `grounding=False`, and `thinking=None`",
      "",
      "Release notes: https://www.effgen.org/docs/releases",
    ].join("\n"),
    tags: ["v0.2.2", "0.2.2", "release notes", "gemini", "anthropic", "upgrade", "thinking", "grounding", "files", "caching"],
  },
  {
    id: 48,
    category: "Models",
    question: "How do I use Gemini thinking, grounding, and Files API?",
    answer: [
      "Use a Gemini model that supports thinking, such as `gemini-2.5-pro`, then opt in through `GenerationConfig`:",
      "",
      "```python",
      "from pathlib import Path",
      "from effgen import load_model",
      "from effgen.models import GenerationConfig, upload_file",
      "",
      "Path(\"brief.txt\").write_text(\"Summarize this release note.\")",
      "doc = upload_file(\"brief.txt\")",
      "",
      "model = load_model(\"gemini-2.5-pro\", provider=\"gemini\")",
      "result = model.generate(",
      "    \"Use this file and Google Search grounding.\",",
      "    config=GenerationConfig(thinking_budget=4096, include_thoughts=True, grounding=True),",
      "    files=[doc],",
      ")",
      "print(result.metadata.get(\"thinking\"))",
      "print(result.metadata.get(\"grounding_chunks\"))",
      "```",
      "",
      "Docs: https://www.effgen.org/docs/native-provider-tools",
    ].join("\n"),
    tags: ["gemini", "thinking_budget", "include_thoughts", "grounding", "files api", "upload_file", "google-genai", "v0.2.2"],
  },
  {
    id: 49,
    category: "Models",
    question: "How do I use Anthropic thinking and prompt caching?",
    answer: [
      "Use Claude 4.x and pass Anthropic thinking plus cache markers through `GenerationConfig` and `mark_cached`:",
      "",
      "```python",
      "from effgen import load_model",
      "from effgen.models import GenerationConfig, mark_cached",
      "",
      "model = load_model(\"claude-sonnet-4-6\", provider=\"anthropic\")",
      "result = model.generate(",
      "    \"Reason about migration risks.\",",
      "    system_prompt=[mark_cached(\"Reusable system context\", ttl=\"1h\")],",
      "    config=GenerationConfig(thinking={\"type\": \"enabled\", \"budget_tokens\": 4096}),",
      ")",
      "print(result.metadata.get(\"cached_input_tokens\"))",
      "print(result.metadata.get(\"cache_creation_tokens\"))",
      "```",
      "",
      "`generate_stream_full()` returns typed chunks for text, thinking, redacted_thinking, and tool_use. `build_assistant_message(result)` preserves raw content blocks for multi-turn flows.",
    ].join("\n"),
    tags: ["anthropic", "claude", "claude-sonnet-4-6", "thinking", "prompt caching", "mark_cached", "generate_stream_full", "redacted_thinking", "v0.2.2"],
  },
  {
    id: 50,
    category: "Tools",
    question: "How do provider-native tools work in v0.2.2?",
    answer: [
      "Provider-native tools run on the provider side, not as local effGen tools.",
      "",
      "- OpenAI native tools arrived in v0.2.1: web search, code interpreter, and file search through the Responses API",
      "- Gemini native tools arrived in v0.2.2 and can be passed through `AgentConfig.tools`: `GoogleSearchTool`, `GeminiUrlContextTool`, and `GeminiCodeExecutionTool`",
      "- Anthropic v0.2.2 computer-use wrappers are experimental adapter-level specs; pass `AnthropicBashTool().to_anthropic_tool_spec()` to `AnthropicAdapter.generate_with_tools()` with the required beta header",
      "",
      "Native tools docs: https://www.effgen.org/docs/native-provider-tools",
    ].join("\n"),
    tags: ["provider-native tools", "native provider tools", "gemini native tools", "anthropic native tools", "openai native tools", "ToolIncompatibleError", "v0.2.2"],
  },
  {
    id: 51,
    category: "General",
    question: "What changed in effGen v0.2.3?",
    answer: [
      "v0.2.3 expanded effGen from 4 to **9 cloud inference providers** and documented the full 14-backend surface.",
      "",
      "Key details:",
      "- New cloud providers: Groq, Together AI, Fireworks, Replicate, and HuggingFace Inference",
      "- New extras: `effgen[groq]`, `effgen[together]`, `effgen[fireworks]`, `effgen[replicate]`, and `effgen[hf]`",
      "- `ProviderRegistry` adds `list_providers()`, `list_models(provider)`, `lookup(model_id)`, and `provider:model_id` prefixes",
      "- `effgen doctor` checks provider API keys from `~/.effgen/.env`, the project `.env`, and exported environment variables",
      "- Backend parity recorded 7/8 providers correct on the canonical calculator task, 7/7 validated streaming providers, and 9/9 `ModelAuthError` behavior on bad credentials",
      "",
      "Release notes: https://www.effgen.org/docs/releases",
    ].join("\n"),
    tags: ["v0.2.3", "0.2.3", "release notes", "groq", "together", "fireworks", "replicate", "huggingface", "providerregistry", "doctor", "parity"],
  },
  {
    id: 52,
    category: "Models",
    question: "How do I use ProviderRegistry and effgen doctor?",
    answer: [
      "ProviderRegistry and `effgen doctor` are v0.2.3+ features for explicit cloud routing and auth checks.",
      "",
      "```python",
      "from effgen import load_model",
      "from effgen.models.registry import list_providers, list_models, lookup",
      "from effgen.models.auth import check_keys",
      "",
      "print(list_providers())",
      "print(list_models(\"groq\")[:3])",
      "",
      "model = load_model(\"groq:llama-3.3-70b-versatile\")",
      "provider, adapter_cls, info = lookup(\"groq:llama-3.3-70b-versatile\")",
      "keys = check_keys()",
      "```",
      "",
      "CLI:",
      "",
      "```bash",
      "effgen doctor",
      "effgen doctor --provider groq",
      "effgen doctor --json",
      "```",
      "",
      "Docs: https://www.effgen.org/docs/providers",
    ].join("\n"),
    tags: ["providerregistry", "provider registry", "effgen doctor", "doctor", "provider prefix", "provider:model", "list_providers", "list_models", "lookup", "check_keys", "v0.2.3"],
  },
  {
    id: 53,
    category: "Releases",
    question: "What changed in effGen v0.2.4?",
    answer: [
      "v0.2.4 introduces an opt-in **policy-based ModelRouter** on top of the existing 9 cloud providers, plus persistent cost and rate-limit coordination.",
      "",
      "Highlights:",
      "",
      "- `PolicyBasedRouter` runs an ordered chain of policies — `FirstAvailablePolicy`, `CostBasedPolicy`, `LatencyBasedPolicy` — over all provider/model pairs in `ProviderRegistry`.",
      "- `RoutingContext(prompt_tokens_estimate, user_budget_usd, latency_budget_ms, required_capabilities)` describes the request; `RouterDecision` returns the chosen `ProviderModelPair` plus every eliminated candidate with a reason (`no_key`, `cost_exceeds_budget`, `latency_exceeds_sla`, `missing capabilities: tools`, `requires dedicated endpoint`, etc.).",
      "- `router.route_and_execute(context, fn)` auto-retries on `RateLimitExceeded`, `ProviderTransientError`, `ModelTimeoutError`, and `BudgetExceededError`, emitting a `RouterEvent` per failover hop to any `router.subscribe(callback)`.",
      "- `SQLiteCostStore` persists every paid call to `~/.effgen/costs.sqlite`; `SQLiteRateLimitStore` coordinates worker processes via `~/.effgen/rate_limits.sqlite` (WAL mode, `BEGIN IMMEDIATE`).",
      "- New `effgen cost today | week | by-provider | set-budget | clear-budget` CLI; 80% of budget emits a `UserWarning`, 100% raises `BudgetExceededError`, which the router treats as retriable failover to a free-tier provider.",
      "- `LatencyTracker` records p50 total + p50 time-to-first-token per (provider, model) for SLA routing; populated automatically by every `generate()` / `generate_stream()` call.",
      "",
      "No breaking changes — all v0.2.0-v0.2.3 code keeps working. The router is opt-in and the SQLite stores are off by default unless you instantiate them.",
      "",
      "Docs: https://www.effgen.org/docs/releases",
    ].join("\n"),
    tags: ["v0.2.4", "0.2.4", "release notes", "model router", "policy", "policy-based", "PolicyBasedRouter", "CostBasedPolicy", "LatencyBasedPolicy", "FirstAvailablePolicy", "failover", "RouterDecision", "RouterEvent", "BudgetExceededError", "effgen cost", "SQLiteCostStore", "SQLiteRateLimitStore"],
  },
  {
    id: 54,
    category: "Models",
    question: "How do I use the v0.2.4 ModelRouter?",
    answer: [
      "v0.2.4 ships an opt-in `PolicyBasedRouter` that picks the best `(provider, model_id)` pair for a request and transparently fails over on retriable errors. Compose any of `FirstAvailablePolicy`, `CostBasedPolicy`, or `LatencyBasedPolicy`:",
      "",
      "```python",
      "import effgen.models  # registers all 9 cloud adapters",
      "from effgen import (",
      "    PolicyBasedRouter, RoutingContext,",
      "    CostBasedPolicy, LatencyBasedPolicy,",
      "    load_model,",
      ")",
      "from effgen.models.capabilities import Capability",
      "",
      "router = PolicyBasedRouter(",
      "    policies=[LatencyBasedPolicy(), CostBasedPolicy()],",
      "    failover_hops=3,",
      ")",
      "router.subscribe(lambda ev: print(\"failover:\", ev.as_dict()))",
      "",
      "context = RoutingContext(",
      "    prompt_tokens_estimate=500,",
      "    user_budget_usd=0.01,",
      "    latency_budget_ms=3000,",
      "    required_capabilities={Capability.chat, Capability.tools},",
      ")",
      "",
      "decision = router.route(context)",
      "print(decision.chosen.provider, decision.chosen.model_id, decision.policy_name)",
      "for pair, reason in decision.eliminated:",
      "    print(\" -\", pair.provider, pair.model_id, \"->\", reason)",
      "",
      "def call_model(pair):",
      "    return load_model(f\"{pair.provider}:{pair.model_id}\").generate(\"Hello\")",
      "",
      "answer = router.route_and_execute(context, call_model)",
      "```",
      "",
      "`route_and_execute` retries on `RateLimitExceeded`, `ProviderTransientError`, `ModelTimeoutError`, and `BudgetExceededError`, and emits a `RouterEvent(from_provider, from_model, to_provider, to_model, reason, hop)` per failover hop.",
      "",
      "Docs: https://www.effgen.org/docs/models · https://www.effgen.org/docs/api-reference",
    ].join("\n"),
    tags: ["model router", "policy router", "PolicyBasedRouter", "RoutingContext", "RouterDecision", "RouterEvent", "Capability", "route_and_execute", "failover", "v0.2.4"],
  },
  {
    id: 55,
    category: "CLI",
    question: "How do I track costs and set a budget with effgen cost?",
    answer: [
      "v0.2.4 adds an `effgen cost` CLI backed by SQLiteCostStore at `~/.effgen/costs.sqlite`. Every paid API call writes a row, and budgets are stored in `~/.effgen/budget.json`.",
      "",
      "```bash",
      "effgen cost today          # per-provider/model summary for the last 24h",
      "effgen cost week           # rolling 7-day spend",
      "effgen cost by-provider    # lifetime totals grouped by provider",
      "effgen cost set-budget 1.0 # $1/day cap",
      "effgen cost clear-budget   # remove caps",
      "```",
      "",
      "At >= 80% of budget effGen emits a `UserWarning`; at >= 100% it raises `BudgetExceededError(period=\"daily\"|\"monthly\")`. Zero-cost calls remain allowed so the router can fail over to free-tier providers.",
      "",
      "Programmatic access:",
      "",
      "```python",
      "from effgen.models._cost import CostTracker",
      "from effgen.models._cost_store import SQLiteCostStore",
      "",
      "tracker = CostTracker.get()           # singleton (SQLite-backed)",
      "store = SQLiteCostStore()             # ~/.effgen/costs.sqlite",
      "for ev in store.query_today():",
      "    print(f\"{ev.provider}/{ev.model}: ${ev.cost_usd:.6f}\")",
      "```",
      "",
      "Docs: https://www.effgen.org/docs/releases",
    ].join("\n"),
    tags: ["effgen cost", "cost CLI", "budget", "BudgetExceededError", "SQLiteCostStore", "CostTracker", "v0.2.4"],
  },
  {
    id: 63,
    category: "Releases",
    question: "What changed in effGen v0.3.1?",
    answer: [
      "**effGen v0.3.1** is a **real-world usability & polish** release, driven by living with the framework as real professionals do. It adds no new providers or subsystems — instead it seals the sharp edges those users hit first. There are **no breaking API changes**; every addition is additive.",
      "",
      "**Traceable evidence on every result:**",
      "- `response.sources` and `response.citations` are populated from the URLs a run **actually retrieved** (`web_search`, `url_fetch`, `news`, `wikipedia`) and from provider-native grounding (OpenAI `url_citation` annotations, surfaced as `metadata[\"grounding_chunks\"]`; Gemini search grounding). Only tool-returned URLs land here — **never** URLs scraped from the model's prose — so a caller can verify and link them programmatically. The research preset cites only a URL one of its tools returned this run.",
      "",
      "**Reasoning models finish the job:**",
      "- Reasoning models (the `gpt-5` family, `o`-series) **no longer return empty, billed results** on token-heavy tasks. They now get a larger default output budget (4096) across every path (direct, ReAct, streaming, speculative, native-tool, structured); an empty result with `finish_reason=\"length\"` is treated as truncation — the budget grows once and retries, or fails with an actionable \"increase `max_tokens`\" message. `effgen batch` gained `--max-tokens`.",
      "",
      "**Honest, costed, measurable results:**",
      "- `AgentResponse.metadata` now carries `cost_usd` and prompt/completion/total token counts (summed across the run, tool loops included); local models report no `cost_usd` key rather than a fake `$0`. Per-run `latency_ms` / `duration_s` is folded onto both `AgentResponse` and raw `GenerationResult` metadata. A shared adaptive formatter shows real sub-cent SLM costs (e.g. `$0.000049`) instead of rounding to `$0.0000`. Team and workflow results report summed `cost_usd` / tokens.",
      "- Passing a Pydantic class to `output_schema=` now also populates `metadata[\"parsed\"]` with a typed instance.",
      "",
      "**Your persona is honored everywhere:**",
      "- A custom `system_prompt` now steers **every** response. It was silently dropped on the no-tool direct path, the no-tool streaming path that `chat` uses, and the native/hybrid tool path. New `chat --system-prompt/--persona`, an `education.*` prompt set (`socratic_tutor`, `lesson_plan`, `quiz_generate`, `explain_simply`), and `prompts list --json`.",
      "",
      "**Trustworthy multi-agent teams & workflows:**",
      "- Collaborative teams **fail closed** (a failed collaborator sets team `success=False` with a discoverable per-agent error). Hierarchical teams **route by name** to the worker the manager named, and every subtask runs. Workflow DAGs don't run downstream of a failed/skipped node. `effgen chat --session-id/--resume` continues a persisted conversation.",
      "",
      "**One-call agents from a knowledge domain:**",
      "- `LegalDomain().to_agent(\"gpt-5-nano\")` (or `create_agent(domain=...)`) wires the domain's system prompt, recommended tools, and guardrails into an agent. A RAG agent accepts a pre-built `VectorMemoryStore` as its `knowledge_base`. The everyday guardrail classes (`PIIGuardrail`, `GuardrailChain`, presets) are exported at the top level.",
      "",
      "**Honest OpenAI-compatible server:**",
      "- `/v1/chat/completions` no longer silently drops a client-defined tool it doesn't host — it's rejected with a clear `400` (`unknown_tool`). `/v1/embeddings` strips a `provider:` prefix and reflects its real backend (or fails closed under `EFFGEN_EMBEDDINGS_STRICT=1`) instead of serving near-zero hash vectors under a neural model's name. Auth / validation / rate-limit / RBAC errors share the OpenAI error envelope; empty `messages` → `400`; per-call `cost_usd` in the `effgen` extension.",
      "",
      "**Local-first truth:** `models status` shows **physical GPU memory** (the driver's view across all processes) plus utilization; `models info` recognizes a model in the local HuggingFace cache; local Transformers batch is thread-safe; the optional `effgen[grammar]` extra (`outlines`) lets small local models emit schema-valid JSON in one constrained pass.",
      "",
      "**Automation & integration:** sync `Agent.run()` no longer hangs forever on an MCP stdio tool (clear `TimeoutError` pointing at `run_async()`); installed tool plugins auto-discover via the `effgen.plugins` entry point; `effgen run --json` emits the full result document to stdout (add `-q` for pristine stdout), with `--json` also on `eval`, `compare`, `workflow`, and `sessions list`.",
      "",
      "**Security:** the PythonREPL sandbox toggle is out of the model's hands (`restricted_mode` removed from the model-facing schema; unrestricted execution is a developer-only opt-in); the `bash` env scrub is exhaustive and refuses reads of secret files; the prompt-injection and PII guardrails are sharpened.",
      "",
      "Upgrade with `pip install --upgrade effgen`. Docs: https://www.effgen.org/docs/releases",
    ].join("\n"),
    tags: ["v0.3.1", "0.3.1", "release notes", "usability", "polish", "sources", "citations", "response.sources", "response.citations", "grounding", "grounding_chunks", "reasoning models", "gpt-5", "o-series", "max_tokens", "cost_usd", "latency_ms", "tokens", "persona", "system_prompt", "education", "socratic_tutor", "lesson_plan", "to_agent", "LegalDomain", "domain agent", "VectorMemoryStore", "knowledge_base", "fail closed", "hierarchical", "route by name", "workflow DAG", "unknown_tool", "embeddings", "EFFGEN_EMBEDDINGS_STRICT", "models status", "physical GPU memory", "grammar", "outlines", "MCP deadlock", "TimeoutError", "plugin auto-discovery", "effgen.plugins", "run --json", "restricted_mode", "PythonREPL", "bash", "prompt injection", "PII"],
  },
  {
    id: 62,
    category: "Releases",
    question: "What changed in effGen v0.3.0?",
    answer: [
      "**effGen v0.3.0** is a major **stabilization & hardening** release. It adds no new providers, tools, prompt templates, or subsystems — instead it makes everything already in effGen **robust, predictable, fast, secure, and pleasant to use**. There are no breaking API changes; every ergonomic improvement is an additive alias.",
      "",
      "**Robust failures, never silent:**",
      "- `Agent.run()` can no longer return `success=True` with empty output. The direct and tool paths return the **same** shape on failure: `success=False`, a coarse `metadata[\"reason\"]` stage label (e.g. `generation_failed`), and a typed redacted `metadata[\"error\"]` dict.",
      "- `classify_provider_error()` populates `metadata[\"error\"][\"category\"]` with a stable taxonomy: `auth`, `not_found`, `rate_limited`, `transient`, `timeout`, `fatal`. Retries fire only on transient / rate-limited errors; auth and not-found fast-stop. `AgentConfig.raise_on_error` opts into exceptions.",
      "- A wrong / 404 model id suggests the nearest live alternative instead of crashing.",
      "",
      "**A model catalog that updates itself:**",
      "```bash",
      "effgen models refresh                 # pull the live list, report added / removed / changed",
      "effgen models list --json             # catalog-backed: status, price, context, deprecation, verified-on",
      "```",
      "- Every provider ships a local snapshot (id, context window, max output, input/output price per 1M tokens, tool/vision/audio support, free-tier flag, rate limits) with a count and a \"verified on\" date. `check_drift()` warns once (never spammily) when the catalog looks stale. Private `ft:` / embeddings / audio / image ids are never persisted.",
      "",
      "**Real GPU support:** the documented GPU install selects a driver-compatible torch (a runtime guard warns once when NVML sees GPUs but `torch.cuda` cannot). `temperature=0` now decodes greedily (`do_sample=False`) instead of raising; the GPU allocator no longer deadlocks and reads real free memory via NVML.",
      "",
      "**Server fails closed:** outside dev mode with no configured issuer/JWKS, the server rejects all bearer tokens — a forged HS256 JWT can't reach `/whoami` or `/v1/chat/completions`. CORS no longer combines wildcard origin with credentials; metrics + dashboard require auth; the `viewer` role can't run tools; budget reserves then reconciles; upstream/auth/missing-key failures map to **502/503**, not a misleading 401.",
      "",
      "**Hardened built-in tools:** PythonREPL runs user code in a worker subprocess with an **out-of-process** wall-clock timeout, process-group kill, and memory/output caps. One shared SSRF guard (`tools/builtin/_net.py`) re-validates on every redirect for **every** URL tool. File tools are path-confined; the un-gated `pickle.load` path is gone (JSON-only state); prompt-chain conditions use an AST-whitelist comparator, not `eval()`.",
      "",
      "**Fast, consistent, and a joy to use:** `import effgen` dropped from ~7.5 s / ~800 MB to **~0.02 s / ~12 MB** (lazy loading). Streaming is truly incremental with typed mid-stream errors; the agent loop stops calling tools once it has a confident answer (a task that took 6 tool calls / 66 s now takes 1). The CLI is quiet and scriptable (`--json` everywhere, `--provider` on `run`/`chat`/`debug`, non-zero exit codes), the obvious constructor calls just work, and a live \"thinking\" UX, rotating tips, \"did you mean?\" suggestions, rich Markdown rendering, and a polished `effgen chat` make the first five minutes easy.",
      "",
      "**Packaging:** `pip-audit` is clean across the documented extras and `pypdf` is patched to `>=6.13.3` (GHSA-jm82-fx9c-mx94).",
      "",
      "Upgrade with `pip install --upgrade effgen`. Docs: https://www.effgen.org/docs/releases",
    ].join("\n"),
    tags: ["v0.3.0", "0.3.0", "release notes", "stabilization", "hardening", "fail closed", "fail-closed", "silent success", "classify_provider_error", "model catalog", "effgen models refresh", "drift", "check_drift", "GPU", "torch.cuda", "temperature 0", "greedy decoding", "SSRF", "PythonREPL", "sandbox", "path confinement", "pickle", "eval", "lazy import", "streaming", "agent loop", "quiet CLI", "--json", "pip-audit", "pypdf"],
  },
  {
    id: 61,
    category: "Releases",
    question: "What changed in effGen v0.2.10?",
    answer: [
      "**effGen v0.2.10** is the **Security, Edge & Developer Experience** release — hardening effGen end-to-end and adding production deployment targets plus three developer-experience surfaces. No breaking API changes; every security and DX feature is additive.",
      "",
      "**Security — sandboxed CodeExecutor:**",
      "- `DockerSandbox` is the default when Docker is available — `--read-only`, `--network=none`, `--cap-drop=ALL`, `--no-new-privileges`, `--pids-limit=100`, `--memory=256m`, non-root user.",
      "- `SubprocessSandbox` is the fallback — unprivileged user namespaces (`unshare --map-root-user --net --mount`), private tmpfs over `/tmp`, network blocked, no `CAP_SYS_ADMIN` required. Loud warning when it falls back from Docker.",
      "- Env-driven: `EFFGEN_SANDBOX_BACKEND=docker|subprocess|off`, `EFFGEN_SANDBOX_TIMEOUT=10`.",
      "",
      "**Auth, RBAC & audit (API server):**",
      "- OIDC/JWT validation on every non-public endpoint via `authlib`. Configure `EFFGEN_OIDC_ISSUER`, `EFFGEN_OIDC_CLIENT_ID`, `EFFGEN_OIDC_JWKS_URI`. `EFFGEN_DEV_MODE=1` disables auth with a loud warning.",
      "- RBAC with union-of-roles semantics; `RBACBudgetMiddleware` enforces tool allow-lists (403) and a daily cost cap (429 `BudgetExceeded`). Built-in roles: `admin`, `researcher`, `viewer`, `reader`.",
      "- Per-request audit log at `~/.effgen/audit/<date>.jsonl` (redacted, never contains secrets).",
      "",
      "**Supply-chain hardening:** gitleaks pre-commit + CI (working tree + full history), a CycloneDX 1.5 SBOM (`sbom.cdx.json`), `pip-audit` CI (fails on HIGH/CRITICAL), hash-verified lockfiles, and `EFFGEN_VERIFY_HASHES=1` startup hash verification.",
      "",
      "**Deploy targets (under `deploy/`):**",
      "```bash",
      "docker build -f deploy/docker/Dockerfile --build-arg EXTRAS=server -t effgen:0.2.10 .",
      "helm install effgen deploy/k8s/helm/effgen/ -f deploy/k8s/helm/effgen/values.yaml",
      "cd deploy/aws_lambda && sam build && sam deploy   # Mangum adapter",
      "cd deploy/cloudflare && wrangler deploy           # edge proxy: CORS + JWT + rate-limit",
      "```",
      "",
      "**Developer experience:** a VSCode extension (prompt-template completion, run code lens, hover docs), three Jupyter magics (`%effgen_chat`, `%%effgen_agent`, `%effgen_metrics`), and a live local dashboard at `/dashboard` (metrics, SLO burn rates, recent runs, SSE span stream).",
      "",
      "Full regression suite: **3721 passed, 0 failed**. Upgrade with `pip install --upgrade effgen`.",
    ].join("\n"),
    tags: ["v0.2.10", "0.2.10", "release notes", "security", "sandbox", "CodeExecutor", "DockerSandbox", "SubprocessSandbox", "OIDC", "auth", "RBAC", "audit log", "gitleaks", "SBOM", "pip-audit", "docker", "helm", "kubernetes", "lambda", "cloudflare", "vscode", "jupyter", "dashboard"],
  },
  {
    id: 60,
    category: "Releases",
    question: "What changed in effGen v0.2.9?",
    answer: [
      "**effGen v0.2.9** is the **Observability & Reliability** release — turning effGen into something you can operate in production. Every telemetry path is async / non-blocking, so a failed export never fails inference. No breaking API changes.",
      "",
      "**Observability:**",
      "- **Structured logging** — `StructuredFormatter` emits JSON lines `{ts, level, module, event, attributes, trace_id, span_id}`; agent loop, adapters, router, and tools migrated off ad-hoc `print`.",
      "- **Secret redaction** — `Redactor` strips OpenAI/Anthropic/Cerebras/Google/HF/Groq keys, Bearer tokens, and Slack/Discord webhook URLs **at the log encoder**, so every path is covered.",
      "- **Metrics** — Prometheus histograms (`effgen_model_call_latency_seconds`, tool, agent) + `effgen_tokens_total` counter; `GET /metrics`.",
      "- **SLO tracking** — `SLO` + `SLOTracker` rolling-window error budgets; `burn_rate()`; `GET /slo`.",
      "- **Tracing** — OTel with explicit samplers (`AlwaysOn/Off`, `TraceIdRatio`, `RateLimited`, `ParentBased`) and a canonical span-attribute spec. No implicit `head=1.0` in production.",
      "",
      "**Reliability:**",
      "```python",
      "from effgen.reliability import Retry, retryable, CircuitBreaker, Bulkhead",
      "",
      "@retryable(Retry(max_attempts=3, base_delay=0.5, jitter=True))",
      "def call_model(prompt): ...",
      "",
      "breaker = CircuitBreaker(\"cerebras\", failure_threshold=5, recovery_timeout=30)",
      "bulkhead = Bulkhead(\"cerebras\", max_concurrency=10, queue_size=50)",
      "```",
      "- Explicit timeouts on every I/O boundary (no `timeout=None` — audited at test time), jittered retries, three-state per-provider circuit breakers, and bulkheads.",
      "",
      "**Testing & ops:** a deterministic `Chaos(seed)` harness (6 fault types, 4 canonical scenarios), a Hypothesis fuzz suite (66 tools, messages, router), the `effgen loadtest` CLI (throughput + p50/p95/p99 + error rate), and 6 Alertmanager-compatible rules with `AlertWebhook`.",
      "",
      "Upgrade with `pip install --upgrade effgen`.",
    ].join("\n"),
    tags: ["v0.2.9", "0.2.9", "release notes", "observability", "reliability", "logging", "redaction", "Redactor", "metrics", "prometheus", "SLO", "tracing", "OpenTelemetry", "OTel", "retries", "circuit breaker", "bulkhead", "timeout", "chaos", "fuzz", "loadtest", "alerting"],
  },
  {
    id: 59,
    category: "Releases",
    question: "What changed in effGen v0.2.8?",
    answer: [
      "**effGen v0.2.8** is the **Multimodal Input** release — image, audio, and video are now first-class input types across 6 cloud providers plus local MLX-VLM. No breaking API changes; the old `Message(role, \"text\")` constructor still works.",
      "",
      "**Unified Message schema:** `Message.content` is a typed `List[ContentPart]` — `TextPart`, `ImagePart`, `AudioPart`, `VideoPart`, `ToolCallPart`, `ToolResultPart`. Invalid MIME types or empty video frames raise `InvalidMultimodalContent`.",
      "",
      "**Helpers:**",
      "```python",
      "from effgen import image_from, audio_from, video_from",
      "img = image_from(\"/tmp/photo.jpg\")     # bytes, path, URL, PIL.Image, np.ndarray",
      "aud = audio_from(\"/tmp/recording.mp3\")",
      "vid = video_from(\"/tmp/clip.mp4\", fps=1)  # ffmpeg keyframe sampling",
      "```",
      "",
      "**Provider support:** Gemini (image + audio + native video), OpenAI gpt-4o (image + Whisper audio), Groq (Llama 4 / 3.2-vision image), Anthropic (image, code-only), Together (vision image), HuggingFace (image + ASR), plus local MLX-VLM on Apple Silicon. Providers without native video receive sampled frames.",
      "",
      "**Capability gating:** every adapter raises `CapabilityNotSupportedError` when the model lacks `vision` / `audio_input` / `video_input` — no silent downcast to `\"[image not supported]\"`.",
      "",
      "**New preset + tool:** the `multimodal` preset (Gemini Flash-Lite primary; gpt-4o-mini / BLIP fallback) wires `MultimodalDescribeTool`, which auto-routes image / audio / video inputs to the right tool. Five cookbook walkthroughs cover image Q&A, audio transcribe + reason, video summarize, OCR + LLM, and chart reading.",
      "",
      "Upgrade with `pip install --upgrade effgen`.",
    ].join("\n"),
    tags: ["v0.2.8", "0.2.8", "release notes", "multimodal", "image", "audio", "video", "ContentPart", "ImagePart", "AudioPart", "VideoPart", "image_from", "audio_from", "video_from", "MLX-VLM", "multimodal preset", "MultimodalDescribeTool", "capability gating", "Whisper"],
  },
  {
    id: 58,
    category: "Releases",
    question: "What changed in effGen v0.2.7?",
    answer: [
      "**effGen v0.2.7** ships the **Prompt Library** — a curated, domain-organized catalog of **31 reusable prompt templates** across research, coding, data/SQL, legal, medical, creative, and business. Every template is a Python callable that renders deterministically, ships with a fixture and golden evaluation test, and is accessible through a rich CLI and an interactive playground. No breaking API changes.",
      "",
      "**31 templates across 7 domains:**",
      "- **Research (5)** — `research.literature_review.v1.zero_shot`, `research.literature_review.v1.cot`, `research.paper_summary.v1`, `research.citation_extract.v1`, `research.methodology_critique.v1`",
      "- **Coding (5)** — `coding.code_review.v1`, `coding.bug_diagnose.v1`, `coding.refactor_plan.v1`, `coding.test_generate.v1`, `coding.docstring_fill.v1`",
      "- **Data / SQL (5)** — `data.sql_from_nl.v1`, `data.sql_explain.v1`, `data.sql_optimize.v1`, `data.data_profile.v1`, `data.etl_plan.v1`",
      "- **Legal (3)** — `legal.contract_summarize.v1`, `legal.clause_classify.v1`, `legal.legal_research_brief.v1` (each carries a verbatim non-advice disclaimer)",
      "- **Medical (3)** — `medical.symptom_triage.v1`, `medical.drug_interaction_query.v1`, `medical.medical_literature.v1` (each carries a verbatim non-advice disclaimer)",
      "- **Creative (5)** — `creative.story_continuation.v1.zero_shot`, `creative.story_continuation.v1.few_shot`, `creative.poetry_forms.v1`, `creative.character_bio.v1`, `creative.world_building.v1`",
      "- **Business (5)** — `business.meeting_summary.v1`, `business.email_draft.v1`, `business.okr_generate.v1`, `business.swot_analysis.v1`, `business.elevator_pitch.v1`",
      "",
      "**Core API:**",
      "```python",
      "from effgen.prompts.library import registry",
      "",
      "p = registry.get(\"data.sql_from_nl.v1\")",
      "text = p.template(schema_ddl=\"CREATE TABLE orders (id INT, total FLOAT)\", question=\"Total orders this month\", dialect=\"sqlite\")",
      "",
      "for prompt in registry.search(domain=\"coding\", variant=\"structured\"):",
      "    print(prompt.name, prompt.description)",
      "```",
      "",
      "**Variants:** `zero_shot`, `cot`, `few_shot`, `tool`, `structured` (validated by the registry).",
      "",
      "**Golden + live eval harness (`PromptEval`):**",
      "- `eval_golden` renders with `fixture` and compares against a stored `.txt` golden (writes on first run).",
      "- `eval_live` renders, runs through a model, and validates `expected_shape` — including `sqlglot.parse()` for SQL templates and `ast.parse()` for generated Python.",
      "- `eval_all_golden` produces a pass/fail table for the entire registry.",
      "",
      "**CLI:**",
      "```bash",
      "effgen prompts list                                    # table by default",
      "effgen prompts list --domain research --variant cot",
      "effgen prompts list --format markdown                  # regenerate the gallery",
      "effgen prompts show research.literature_review.v1.cot",
      "effgen prompts eval                                    # golden evals only",
      "effgen prompts eval --domain coding --live --model llama3.1-8b",
      "effgen prompts render data.sql_from_nl.v1 --input '{\"schema_ddl\": \"...\", \"question\": \"...\", \"dialect\": \"sqlite\"}'",
      "effgen prompts run    data.sql_from_nl.v1 --input ... --model groq:llama-3.3-70b-versatile",
      "effgen prompts playground                              # interactive REPL",
      "```",
      "",
      "**Interactive playground commands:** `select <name>`, `set <key> <value>`, `unset <key>`, `render`, `run [--model <id>]`, `save [<path>]`, `load <path>`, `reload`, `list [--domain D]`, `show <name>`, `help`, `exit`.",
      "",
      "**Auto-generated gallery:** `docs/prompts/gallery.md` lists every template with its variant and description; regenerate with `effgen prompts list --format markdown`.",
      "",
      "**Upgrade:** `pip install --upgrade effgen`. No breaking API changes — all prompt library classes are opt-in additions; v0.2.0–v0.2.6 code keeps working.",
    ].join("\n"),
    tags: ["v0.2.7", "0.2.7", "release notes", "prompt library", "PromptRegistry", "LibraryPrompt", "PromptEval", "playground", "effgen prompts", "literature_review", "code_review", "sql_from_nl", "legal disclaimer", "medical disclaimer", "31 templates", "31 prompts"],
  },
  {
    id: 57,
    category: "Releases",
    question: "What changed in effGen v0.2.6?",
    answer: [
      "**effGen v0.2.6** adds **14 new built-in tools** across six categories — OCR, audio transcription, image analysis, document parsing, geo/weather, and email/webhook communication — raising the total built-in tool count from 44 to **58+**. Two new presets (`media`, `notify`) join the existing roster. No breaking API changes.",
      "",
      "**OCR:**",
      "- `OCRTool` — Tesseract (local, primary) + OCR.space free API fallback (`OCR_SPACE_API_KEY`). Raises `OCRBackendUnavailable` with per-OS install instructions. Operations: `extract`, `extract_regions`. Added to `general` preset.",
      "",
      "**Audio Transcription:**",
      "- `AudioTranscribeTool` — `faster-whisper` (CPU/GPU auto-detected) + HuggingFace Inference fallback (`HF_TOKEN`). Warns when `model_size > base` on CPU. Operations: `transcribe`. Added to new `media` preset.",
      "",
      "**Image Analysis:**",
      "- `ImageInfoTool` — Pillow-based metadata, EXIF, color histogram, resize/thumbnail. Zero network. Operations: `info`, `resize`, `thumbnail`. Added to `general` preset.",
      "- `ImageCaptionTool` — natural-language captions via effGen model router (Gemini / OpenAI / MLX-VLM). Raises `NoVisionProviderAvailable` if no vision provider is configured. Operations: `caption`, `describe`. Added to `media` preset.",
      "",
      "**Document Parsing:**",
      "- `PDFTool` — `pypdf` (primary) + `pdfplumber` (tables). Operations: `text`, `metadata`, `tables`, `extract_images`. Added to `research` and `general` presets.",
      "- `DOCXTool` — `python-docx`. Operations: `text`, `paragraphs`, `tables`, `metadata`. Added to `research` and `general` presets.",
      "- `ExcelTool` — `openpyxl` + `pandas`. Operations: `sheets`, `read_sheet`, `headers`. Added to `research` and `general` presets.",
      "",
      "**Geo / Weather:**",
      "- `WeatherTool` — Open-Meteo (free, no auth). Operations: `current`, `forecast`, `historical`. Added to `general` preset.",
      "- `GeocodeTool` — Nominatim (OSM), 1 req/s token bucket, sets `effGen/<version>` User-Agent. Operations: `geocode`, `reverse`. Added to `general` preset.",
      "- `MapsTool` — static PNG maps from OSM tiles via the `staticmap` library. Operations: `render`, `bounding_box`. Added to `general` preset.",
      "",
      "**Email (live send/read, replaces draft-only):**",
      "- `EmailSMTPTool` — stdlib `smtplib`, TLS-on by default. Env: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`. Raises `MissingCredentialsError` when absent. Operations: `send`. Added to new `notify` preset.",
      "- `EmailIMAPTool` — stdlib `imaplib`. Env: `IMAP_HOST`, `IMAP_PORT`, `IMAP_USER`, `IMAP_PASSWORD`. Operations: `list_folders`, `fetch_recent`, `search`, `get`. Added to `notify` preset.",
      "",
      "**Webhooks:**",
      "- `SlackWebhookTool` — incoming webhook URL (no OAuth). Env: `SLACK_WEBHOOK_URL`. URL redacted in all logs. Operations: `post`. Added to `notify` preset.",
      "- `DiscordWebhookTool` — webhook URL. Env: `DISCORD_WEBHOOK_URL`. URL redacted in all logs. Operations: `post`. Added to `notify` preset.",
      "",
      "**New presets:**",
      "- `media` — bundles `AudioTranscribeTool` + `ImageCaptionTool`.",
      "- `notify` — bundles `EmailSMTPTool` + `EmailIMAPTool` + `SlackWebhookTool` + `DiscordWebhookTool`.",
      "",
      "**New errors:** `OCRBackendUnavailable`, `MissingSystemDependency`, `NoVisionProviderAvailable`, `MissingCredentialsError`, `CorruptDocumentError`.",
      "",
      "**Install extras:** `pip install -U \"effgen[documents]\"` (PDF/DOCX/Excel), `pip install -U \"effgen[audio]\"` (AudioTranscribe), `pip install -U \"effgen[tools]\"` (OCR/Image/and more), or grab everything with `pip install -U \"effgen[all]\"`.",
      "",
      "**System deps (per-tool primary paths):** `apt-get install tesseract-ocr` / `brew install tesseract` (OCR); `apt-get install ffmpeg` / `brew install ffmpeg` (non-WAV audio).",
      "",
      "No breaking API changes — all v0.2.0–v0.2.5 code keeps working.",
    ].join("\n"),
    tags: ["v0.2.6", "0.2.6", "release notes", "OCR", "AudioTranscribe", "ImageInfo", "ImageCaption", "PDF", "DOCX", "Excel", "Weather", "Geocode", "Maps", "EmailSMTP", "EmailIMAP", "Slack", "Discord", "webhook", "media preset", "notify preset", "58 tools", "58+ tools"],
  },
  {
    id: 56,
    category: "Releases",
    question: "What changed in effGen v0.2.5?",
    answer: [
      "**effGen v0.2.5** adds **13 new free / no-auth tools** spanning academic research, news & RSS, YouTube, social, translation, language detection, and QR codes — bringing the built-in tool count to **44+**.",
      "",
      "All tools are `BaseTool` subclasses with the structured `{success, data, error}` output shape, integrated into the `research` and `general` presets, and covered by unit + integration tests.",
      "",
      "**Academic research:**",
      "- `PubMedTool` — NCBI E-utilities; token-bucket rate limiter (3 req/s without key, 10/s with `NCBI_API_KEY`). Operations: `search`, `fetch`, `abstract`.",
      "- `ArXivTool` — arXiv Atom feed search, ID fetch, PDF download. Operations: `search`, `fetch`, `download_pdf`.",
      "- `SemanticScholarTool` — Semantic Scholar Graph API. Operations: `search`, `paper`, `citations`, `references`; built-in backoff (100 req / 5 min unauth).",
      "",
      "**News & RSS:**",
      "- `RSSFeedTool` — any RSS/Atom feed by URL. Operations: `fetch`, `latest`, `search_in_feed`.",
      "- `NewsTool` — curated reputable sources (Reuters, BBC, HN, NPR…); optional `NEWS_API_KEY` for NewsAPI.org. Operations: `top_headlines`, `search`.",
      "",
      "**YouTube:**",
      "- `YouTubeTranscriptTool` — captions/transcripts via `youtube-transcript-api`, no Google API key. Handles watch?v=, youtu.be/, shorts/. Operations: `get_transcript`, `list_available_languages`, `translated`.",
      "- `YouTubeMetadataTool` — yt-dlp metadata-only mode. Operations: `metadata`, `channel`.",
      "",
      "**Social:**",
      "- `RedditTool` — public Reddit JSON (no OAuth for reads). Operations: `subreddit_top`, `subreddit_hot`, `user_submissions`, `thread_comments`. Sets `effGen/<version>` User-Agent, exponential backoff on 429.",
      "- `HackerNewsTool` — HN Firebase API. Operations: `top_stories`, `new_stories`, `story`, `user`.",
      "",
      "**Translation & Language Detection:**",
      "- `TranslateTool` — LibreTranslate primary backend (configurable via `LIBRE_TRANSLATE_URL`), `argostranslate` offline fallback. Language pack cache at `~/.effgen/argos/`. Operations: `translate`, `available_pairs`.",
      "- `LanguageDetectTool` — fully offline `langdetect` (55+ languages). Operations: `detect`, `detect_batch`.",
      "",
      "**QR Codes (fully local):**",
      "- `QRGenerateTool` — local QR generation; base64 PNG or file path; `data_url_return=True` for inline. Operations: `generate`.",
      "- `QRReadTool` — decode QR / barcodes from image path or base64 PNG via `pyzbar` + Pillow, with OpenCV QR fallback. Operations: `read`.",
      "",
      "**Preset updates:**",
      "- `research` now includes PubMed, ArXiv, SemanticScholar, RSS, News, YouTubeTranscript, YouTubeMetadata, Reddit, HackerNews (alongside WebSearch / URLFetch / Wikipedia).",
      "- `general` now includes RSS, News, Reddit, HackerNews, Translate, LanguageDetect, QRGenerate, QRRead (alongside the core tools) — 19 tools total.",
      "",
      "**Docs:** new `docs/tools/gallery.md` with a one-line description + quickstart per tool, plus per-tool docs (`pubmed.md`, `arxiv.md`, `semantic_scholar.md`, `rss.md`, `news.md`, `youtube.md`, `reddit.md`, `hackernews.md`, `translate.md`, `language_detect.md`, `qr.md`).",
      "",
      "No breaking changes — all v0.2.0–v0.2.4 code keeps working.",
    ].join("\n"),
    tags: ["v0.2.5", "0.2.5", "release notes", "PubMed", "ArXiv", "SemanticScholar", "RSS", "News", "YouTube", "Reddit", "HackerNews", "Translate", "LanguageDetect", "QR", "QRGenerate", "QRRead", "44 tools", "44+ tools", "free tools"],
  },
  {
    id: 43,
    category: "General",
    question: "What is effGen exactly?",
    answer: "effGen is a **production-grade Python framework for building AI agents using Small Language Models (SLMs)** — v__VERSION__ is the latest release.\n\nKey highlights:\n- **5–10x faster** local inference with vLLM; native MLX & MLX-VLM on Apple Silicon\n- **14 inference backends**: 5 local engines plus 9 cloud providers\n- **66 built-in tools** across computation, code execution, web, academic research, news/RSS, YouTube, social, translation, language detection, QR codes, OCR, audio transcription, image analysis, document parsing (PDF/DOCX/Excel), geo/weather, email, webhooks, data science, DevOps, and finance\n- **9 built-in agent presets** — `math`, `research`, `coding`, `general`, `rag`, `media`, `notify`, `multimodal`, `minimal`\n- **v0.3.1 — Real-World Usability & Polish** — grounded `response.sources` / `.citations` filled from the URLs a run actually retrieved (never the model's prose), reasoning models (`gpt-5` family, `o`-series) that finish token-heavy work instead of empty billed results, cost + tokens + latency on every result, custom personas honored on every path, one-call domain agents (`LegalDomain().to_agent()`), honest multi-agent teams + workflow DAGs (fail closed, route by name), an honest OpenAI-compatible server (no silent tool/embedding downgrade), physical GPU memory in `models status`, grammar-constrained local structured output, no MCP deadlock on sync `run()`, tool-plugin auto-discovery, and `effgen run --json` — no breaking changes\n- **v0.3.0 — Stabilization & Hardening** — fail-closed typed errors (no more silent `success=True`), a self-updating drift-aware model catalog (`effgen models refresh`), real GPU support (`temperature=0` greedy decoding, NVML-aware allocator), a fail-closed API server (forged JWTs rejected, 502 not 401), sandboxed built-in tools (shared SSRF guard, out-of-process PythonREPL timeout, path confinement, no `eval`/`pickle`), a near-instant `import effgen` (~7.5s → ~20ms), faster streaming + an early-stopping agent loop, and a quiet `--json` CLI — no breaking changes\n- **v0.2.10 — Security, Edge & DX** — sandboxed `CodeExecutor` (Docker by default, unprivileged-namespace subprocess fallback), OIDC/JWT auth + RBAC + per-request audit log on the API server, gitleaks + CycloneDX SBOM + pip-audit supply-chain hardening, production deploys (Docker, Helm, AWS Lambda, Cloudflare Worker), a VSCode extension, Jupyter magics, and a live `/dashboard`\n- **v0.2.9 — Observability & Reliability** — structured JSON logging with secret redaction, Prometheus metrics + SLO burn-rate tracking, OTel tracing with samplers, timeouts / jittered retries / circuit breakers / bulkheads, a deterministic chaos harness, a Hypothesis fuzz suite, the `effgen loadtest` CLI, and an Alertmanager rule pack\n- **v0.2.8 — Multimodal Input** — image / audio / video as first-class types across Gemini, OpenAI, Groq, Anthropic, Together, HF (+ local MLX-VLM), with a unified `ContentPart` `Message` schema, per-provider preprocessing, capability gating, the `multimodal` preset, and the `MultimodalDescribeTool`\n- **Prompt Library (v0.2.7)** — 31 curated, domain-organized templates across research / coding / data-SQL / legal / medical / creative / business, with a golden + live eval harness, a CLI (`effgen prompts list / show / eval / render / run`), and an interactive playground REPL\n- **v0.2.6** added 14 new tools (OCR, AudioTranscribe, ImageInfo, ImageCaption, PDF, DOCX, Excel, Weather, Geocode, Maps, EmailSMTP, EmailIMAP, SlackWebhook, DiscordWebhook) plus the `media` and `notify` presets\n- **v0.2.5** added 13 new free / no-auth tools (PubMed, ArXiv, SemanticScholar, RSS, News, YouTubeTranscript, YouTubeMetadata, Reddit, HackerNews, Translate, LanguageDetect, QRGenerate, QRRead), wired into the `research` and `general` presets\n- **v0.2.4** introduced a policy-based **ModelRouter** (FirstAvailable / CostBased / LatencyBased) with transparent failover, SQLite-backed cost/rate-limit stores, and the **`effgen cost`** CLI for daily/weekly spend dashboards and budget guardrails\n- **v0.2.3** expanded to 9 cloud providers with **ProviderRegistry + effgen doctor** for lookup, auth readiness, provider prefixes, and uniform ModelAuthError handling\n- **v0.2.2** added Gemini thinking/grounding/Files/native tools plus Anthropic Claude 4.x thinking, caching, streaming, and experimental tool specs\n- **v0.2.1** added Cerebras plus modern OpenAI reasoning-model controls, cached token metadata, structured outputs v2, and native OpenAI tools\n- **Guardrails, RAG, DAG workflows, evaluation, OpenAI-compatible API server**\n- 100% open source (Apache 2.0)",
    tags: ["what is effgen", "about", "overview", "what does it do", "introduction", "summary", "effgen"],
  },
  {
    id: 44,
    category: "General",
    question: "Is effGen free to use?",
    answer: "Yes — effGen is 100% free and open source under the Apache 2.0 License. You only pay for any external API calls you make (e.g. OpenAI, Anthropic) if you choose to use cloud models. Running local models is completely free.",
    tags: ["free", "cost", "pricing", "paid", "money", "open source", "free to use"],
  },
  {
    id: 45,
    category: "General",
    question: "How does effGen compare to LangChain or AutoGen?",
    answer: "effGen is specifically **optimized for Small Language Models**, which the others are not:\n\n- **SLM-tuned prompts** – prompt templates designed for 1.5B–7B models\n- **vLLM-first** – native integration for maximum speed with local models\n- **Lighter weight** – simpler API, fewer abstractions, faster to learn\n- **Task decomposition** – automatic complexity scoring and sub-agent spawning\n\nIf you want to run powerful agents on consumer hardware without paying for API calls, effGen is built for exactly that.",
    tags: ["comparison", "langchain", "autogen", "vs", "alternative", "compare", "difference", "why effgen"],
  },
];

const getFAQById = (id: number, fallbackIndex = 0): FAQ =>
  faqs.find((faq) => faq.id === id) ?? faqs[fallbackIndex];

// ---------------------------------------------------------------------------
// Matching engine – TF-IDF-style scoring with bigrams + exact-phrase bonus
// ---------------------------------------------------------------------------

// Simple stemming: strip common English suffixes so "installing" matches "install"
const suffixes = ["ing", "tion", "ment", "ness", "able", "ible", "ous", "ive", "ful", "less", "ly", "ed", "er", "es", "s"];
function stem(word: string): string {
  for (const s of suffixes) {
    if (word.length > s.length + 2 && word.endsWith(s)) {
      return word.slice(0, -s.length);
    }
  }
  return word;
}

// English stop-words to ignore
const STOP = new Set([
  "the","a","an","is","it","in","on","to","of","and","or","for","with",
  "that","this","what","how","do","does","i","my","me","can","you","your",
  "be","are","was","were","have","has","had","will","would","could","should",
  "if","but","not","no","yes","so","as","at","by","from","up","out","about",
  "which","who","when","where","why","they","them","their","we","our",
  "can","may","might","must","shall","am","been","being","get","got",
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 1 && !STOP.has(w))
    .map(stem);
}

// Build bigrams from a token array
function bigrams(tokens: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < tokens.length - 1; i++) {
    out.push(tokens[i] + " " + tokens[i + 1]);
  }
  return out;
}

// Pre-compute IDF weights across the FAQ corpus
function buildIDF(docs: string[][]): Map<string, number> {
  const df = new Map<string, number>();
  const N = docs.length;
  for (const tokens of docs) {
    const unique = new Set(tokens);
    unique.forEach((t) => df.set(t, (df.get(t) ?? 0) + 1));
  }
  const idf = new Map<string, number>();
  df.forEach((count, term) => {
    idf.set(term, Math.log(N / count));
  });
  return idf;
}

// Tokenised FAQ corpus (unigrams + bigrams) – built once at module load
const faqTokenised = faqs.map((faq) => {
  const base = tokenize(faq.question + " " + faq.tags.join(" ") + " " + faq.answer);
  return [...base, ...bigrams(base)];
});
const IDF = buildIDF(faqTokenised);

// Cosine-style TF-IDF score between query and a single FAQ
function scoreFAQ(queryTokens: string[], faqTokens: string[]): number {
  const tfFaq = new Map<string, number>();
  faqTokens.forEach((t) => tfFaq.set(t, (tfFaq.get(t) ?? 0) + 1));

  let dot = 0, magQ = 0, magF = 0;
  const allTerms = new Set([...queryTokens, ...faqTokens]);
  allTerms.forEach((t) => {
    const idf = IDF.get(t) ?? 1;
    const qTF = queryTokens.filter((x) => x === t).length;
    const fTF = tfFaq.get(t) ?? 0;
    const qW = qTF * idf;
    const fW = fTF * idf;
    dot += qW * fW;
    magQ += qW * qW;
    magF += fW * fW;
  });

  return magQ === 0 || magF === 0 ? 0 : dot / (Math.sqrt(magQ) * Math.sqrt(magF));
}

// Exact-phrase bonus: if any multi-word tag appears verbatim in the query, boost hard
function exactPhraseBonus(queryLower: string, faq: FAQ): number {
  let bonus = 0;
  for (const tag of faq.tags) {
    if (tag.includes(" ") && queryLower.includes(tag)) {
      bonus += 0.25; // significant boost per exact multi-word match
    }
  }
  return bonus;
}

interface ScoredFAQ {
  faq: FAQ;
  score: number;
}

function findBestMatches(query: string, topK = 3): ScoredFAQ[] {
  const queryLower = query.toLowerCase();
  const queryTokens = tokenize(query);
  const queryBi = bigrams(queryTokens);
  const allQueryTokens = [...queryTokens, ...queryBi];

  const scored: ScoredFAQ[] = faqs.map((faq, i) => ({
    faq,
    score: scoreFAQ(allQueryTokens, faqTokenised[i]) + exactPhraseBonus(queryLower, faq),
  }));

  scored.sort((a, b) => b.score - a.score);

  return scored.filter((s) => s.score > 0.08).slice(0, topK);
}

function renderLinks(text: string, keyPrefix: string): ReactNode[] {
  const urlPattern = /(https?:\/\/[^\s<>()]+|(?:github\.com|discord\.gg|effgen\.org)\/[^\s<>()]+)/g;
  return text.split(urlPattern).filter(Boolean).map((part, index) => {
    const isUrl = /^(https?:\/\/|github\.com\/|discord\.gg\/|effgen\.org\/)/.test(part);
    if (!isUrl) return part;
    const href = part.startsWith("http") ? part : `https://${part}`;
    return (
      <a
        key={`${keyPrefix}-link-${index}`}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-green-600 dark:text-green-400 underline underline-offset-2"
      >
        {part}
      </a>
    );
  });
}

function renderInlineMarkdown(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  text.split("**").forEach((part, index) => {
    if (index % 2 === 1) {
      nodes.push(
      <strong key={`${keyPrefix}-strong-${index}`}>
        {renderLinks(part, `${keyPrefix}-strong-${index}`)}
      </strong>
      );
    } else {
      nodes.push(...renderLinks(part, `${keyPrefix}-text-${index}`));
    }
  });
  return nodes;
}

function renderMarkdownText(text: string, keyPrefix: string): ReactNode[] {
  return text
    .split(/\n{2,}/)
    .filter((block) => block.trim().length > 0)
    .map((block, blockIndex) => {
      const lines = block.split("\n").filter((line) => line.trim().length > 0);
      const allBullets = lines.every((line) => /^\s*[-*]\s+/.test(line));
      const allOrdered = lines.every((line) => /^\s*\d+\.\s+/.test(line));

      if (allBullets) {
        return (
          <ul key={`${keyPrefix}-ul-${blockIndex}`} className="my-2 ml-5 list-disc space-y-1">
            {lines.map((line, lineIndex) => (
              <li key={`${keyPrefix}-li-${blockIndex}-${lineIndex}`}>
                {renderInlineMarkdown(line.replace(/^\s*[-*]\s+/, ""), `${keyPrefix}-li-${blockIndex}-${lineIndex}`)}
              </li>
            ))}
          </ul>
        );
      }

      if (allOrdered) {
        return (
          <ol key={`${keyPrefix}-ol-${blockIndex}`} className="my-2 ml-5 list-decimal space-y-1">
            {lines.map((line, lineIndex) => (
              <li key={`${keyPrefix}-oli-${blockIndex}-${lineIndex}`}>
                {renderInlineMarkdown(line.replace(/^\s*\d+\.\s+/, ""), `${keyPrefix}-oli-${blockIndex}-${lineIndex}`)}
              </li>
            ))}
          </ol>
        );
      }

      return (
        <p key={`${keyPrefix}-p-${blockIndex}`} className="my-1.5 whitespace-pre-line leading-relaxed">
          {renderInlineMarkdown(block, `${keyPrefix}-p-${blockIndex}`)}
        </p>
      );
    });
}

// ---------------------------------------------------------------------------
// Category grouping helper for the browse view
// ---------------------------------------------------------------------------
const CATEGORIES = [...new Set(faqs.map((f) => f.category))];

// ---------------------------------------------------------------------------
// Chat UI state
// ---------------------------------------------------------------------------
type MessageType = "user" | "bot" | "suggestions" | "contact-prompt" | "contact-options";

interface Message {
  type: MessageType;
  content: string;
  suggestions?: ScoredFAQ[];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function HelpBot() {
  const { version: pypiVersion } = usePyPIVersion();
  const [isOpen, setIsOpen] = useState(false);
  const [view, setView] = useState<"chat" | "browse">("chat"); // chat vs category browse
  const [browseCategory, setBrowseCategory] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      type: "bot",
      content: "Hi! I'm the effGen Help Bot. Ask me anything or browse topics below.",
    },
    {
      type: "suggestions",
      content: "",
	      suggestions: [
	        { faq: getFAQById(1), score: 1 },
	        { faq: getFAQById(9), score: 1 },
	        { faq: getFAQById(47), score: 1 },
	        { faq: getFAQById(50), score: 1 },
	        { faq: getFAQById(51), score: 1 },
	        { faq: getFAQById(52), score: 1 },
	      ],
    },
  ]);
  const [input, setInput] = useState("");
  const [isWaitingForContact, setIsWaitingForContact] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 100);
  }, [isOpen]);

  // ── core send handler ─────────────────────────────────────────────────────
  const handleSend = useCallback(() => {
    if (!input.trim()) return;
    const userQuery = input.trim();
    setInput("");

    // If we just asked "would you like to contact support?" and the user says yes/no
    if (isWaitingForContact) {
      setIsWaitingForContact(false);
      setMessages((prev) => [...prev, { type: "user", content: userQuery }]);
      const yes = /^(yes|y|yeah|yep|sure|ok|okay|please)\b/i.test(userQuery);
      if (yes) {
        setTimeout(() => {
          setMessages((prev) => [
            ...prev,
            { type: "contact-options", content: "" },
          ]);
        }, 300);
      } else {
        setTimeout(() => {
          setMessages((prev) => [
            ...prev,
            { type: "bot", content: "No problem! Feel free to ask another question anytime." },
          ]);
        }, 300);
      }
      return;
    }

    // Normal flow
    setMessages((prev) => [...prev, { type: "user", content: userQuery }]);

    const matches = findBestMatches(userQuery, 3);

    setTimeout(() => {
      if (matches.length > 0 && matches[0].score > 0.12) {
        // Good match – show the best answer
        setMessages((prev) => [
          ...prev,
          { type: "bot", content: matches[0].faq.answer },
        ]);
        // Show related if we have them
        if (matches.length > 1) {
          setTimeout(() => {
            setMessages((prev) => [
              ...prev,
              { type: "bot", content: "**Related questions:**" },
              {
                type: "suggestions",
                content: "",
                suggestions: matches.slice(1),
              },
            ]);
          }, 400);
        }
      } else {
        // No good match – offer contact, but DON'T loop
        setIsWaitingForContact(true);
        setMessages((prev) => [
          ...prev,
          {
            type: "bot",
            content: "I don't have a great answer for that one. You can:\n\n1. **Rephrase** your question and try again\n2. **Browse topics** using the menu above\n3. **Contact us** directly (I can show you how)",
          },
          { type: "contact-prompt", content: "Would you like me to show you the contact options?" },
        ]);
      }
    }, 300);
  }, [input, isWaitingForContact]);

  const handleSuggestionClick = (faq: FAQ) => {
    setIsWaitingForContact(false);
    setMessages((prev) => [
      ...prev,
      { type: "user", content: faq.question },
      { type: "bot", content: faq.answer },
    ]);
  };

  // ── render helpers ────────────────────────────────────────────────────────
  const renderBotContent = (rawContent: string) => {
    const content = rawContent.replace(/__VERSION__/g, pypiVersion);
    return content.split("```").flatMap((part, i) => {
      if (i % 2 === 1) {
        const lines = part.split("\n");
        const hasLang = !!lines[0].match(/^[a-z]+$/);
        const lang = hasLang ? lines[0] : "python";
        const code = hasLang ? lines.slice(1).join("\n") : part;
        return (
          <pre key={i} className="bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-100 p-2.5 rounded-lg my-2 text-xs overflow-x-auto font-mono">
            <code
              className="syntax-code"
              dangerouslySetInnerHTML={{ __html: highlightCode(code, lang) }}
            />
          </pre>
        );
      }
      return renderMarkdownText(part, `part-${i}`);
    });
  };

  // ── browse view ───────────────────────────────────────────────────────────
  const renderBrowse = () => {
    if (browseCategory) {
      const items = faqs.filter((f) => f.category === browseCategory);
      return (
        <div className="flex flex-col h-full">
          <button
            onClick={() => setBrowseCategory(null)}
            className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400 hover:underline px-4 pt-3 pb-1"
          >
            <FiArrowLeft size={12} /> Back to topics
          </button>
          <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1.5">
            {items.map((faq) => (
              <button
                key={faq.id}
                onClick={() => {
                  setView("chat");
                  handleSuggestionClick(faq);
                }}
                className="w-full text-left px-3 py-2.5 bg-gray-50 dark:bg-gray-800/50 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-xl text-sm text-gray-700 dark:text-gray-300 transition-colors flex items-start gap-2 group"
              >
                <FiChevronRight className="text-green-500 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" size={13} />
                <span className="group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors">{faq.question}</span>
              </button>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="flex flex-col h-full">
        <p className="text-xs text-gray-500 dark:text-gray-400 px-4 pt-3 pb-1">Choose a topic</p>
        <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-1.5">
          {CATEGORIES.map((cat) => {
            const count = faqs.filter((f) => f.category === cat).length;
            return (
              <button
                key={cat}
                onClick={() => setBrowseCategory(cat)}
                className="w-full text-left px-3 py-2.5 bg-gray-50 dark:bg-gray-800/50 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-xl text-sm text-gray-700 dark:text-gray-300 transition-colors flex items-center justify-between group"
              >
                <span className="group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors font-medium">{cat}</span>
                <span className="text-xs text-gray-400 dark:text-gray-500 bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded-full">{count}</span>
              </button>
            );
          })}
        </div>
      </div>
    );
  };

  // ── main render ───────────────────────────────────────────────────────────
  return (
    <>
      {/* Trigger button */}
      <motion.button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-24 right-8 w-14 h-14 rounded-full bg-gradient-to-r from-green-600 to-emerald-600 flex items-center justify-center text-white shadow-lg hover:shadow-xl hover:shadow-emerald-500/30 transition-all z-40"
        whileHover={{ scale: 1.1, y: -2 }}
        whileTap={{ scale: 0.95 }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
      >
        <FiMessageCircle size={24} />
      </motion.button>

      {/* Chat window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-24 right-8 w-96 h-[540px] bg-white dark:bg-gray-900 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 flex flex-col overflow-hidden z-50"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white flex-shrink-0">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">
                  <FiMessageCircle size={16} />
                </div>
                <div>
                  <h3 className="font-semibold text-sm">effGen Help</h3>
                  <p className="text-xs text-white/70">{faqs.length} topics · Ask anything</p>
                </div>
              </div>
              <div className="flex items-center gap-1">
                {/* Chat / Browse toggle */}
                <div className="flex bg-white/15 rounded-lg p-0.5 mr-1">
                  <button
                    onClick={() => setView("chat")}
                    className={`px-2 py-0.5 rounded-md text-xs font-medium transition-colors ${view === "chat" ? "bg-white text-green-700" : "text-white/80 hover:text-white"}`}
                  >
                    Chat
                  </button>
                  <button
                    onClick={() => { setView("browse"); setBrowseCategory(null); }}
                    className={`px-2 py-0.5 rounded-md text-xs font-medium transition-colors ${view === "browse" ? "bg-white text-green-700" : "text-white/80 hover:text-white"}`}
                  >
                    Browse
                  </button>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1.5 rounded-lg hover:bg-white/20 transition-colors"
                >
                  <FiX size={18} />
                </button>
              </div>
            </div>

            {/* Body – swaps between chat and browse */}
            {view === "browse" ? (
              <div className="flex-1 overflow-hidden">{renderBrowse()}</div>
            ) : (
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.map((message, index) => (
                  <div key={index}>
                    {message.type === "user" && (
                      <div className="flex justify-end">
                        <div className="max-w-[80%] bg-gradient-to-r from-green-600 to-emerald-600 text-white px-4 py-2 rounded-2xl rounded-br-md text-sm">
                          {message.content}
                        </div>
                      </div>
                    )}

                    {message.type === "bot" && (
                      <div className="flex justify-start">
                        <div className="max-w-[85%] bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 px-4 py-2.5 rounded-2xl rounded-bl-md text-sm whitespace-pre-wrap">
                          {renderBotContent(message.content)}
                        </div>
                      </div>
                    )}

                    {message.type === "suggestions" && message.suggestions && (
                      <div className="space-y-1.5 mt-1">
                        {message.suggestions.map((scored, i) => (
                          <motion.button
                            key={i}
                            onClick={() => handleSuggestionClick(scored.faq)}
                            className="w-full text-left px-3 py-2 bg-gray-50 dark:bg-gray-800/50 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-xl text-sm text-gray-700 dark:text-gray-300 transition-colors flex items-center gap-2 group border border-gray-200 dark:border-gray-700/50 hover:border-green-300 dark:hover:border-green-700/50"
                            whileHover={{ x: 3 }}
                          >
                            <FiChevronRight className="text-green-500 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" size={14} />
                            <span className="group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors">
                              {scored.faq.question}
                            </span>
                          </motion.button>
                        ))}
                      </div>
                    )}

                    {message.type === "contact-prompt" && (
                      <div className="flex justify-start mt-1">
                        <div className="max-w-[85%] bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-700/40 text-green-800 dark:text-green-200 px-4 py-2.5 rounded-2xl rounded-bl-md text-sm">
                          {message.content}
                        </div>
                      </div>
                    )}

                    {message.type === "contact-options" && (
                      <div className="flex justify-start mt-1">
                        <div className="max-w-[90%] bg-gray-100 dark:bg-gray-800 px-4 py-3 rounded-2xl rounded-bl-md text-sm space-y-2">
                          <p className="text-gray-700 dark:text-gray-300 font-medium">Here's how to reach us:</p>
                          <a
                            href="mailto:gks@vt.edu"
                            className="flex items-center gap-2 px-3 py-2 bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-green-400 dark:hover:border-green-500 transition-colors group"
                          >
                            <FiMail size={14} className="text-green-600 dark:text-green-400" />
                            <div className="text-left">
                              <p className="text-xs font-semibold text-gray-800 dark:text-gray-200">Email</p>
                              <p className="text-xs text-gray-500 dark:text-gray-400 group-hover:text-green-500 dark:group-hover:text-green-400 transition-colors">gks@vt.edu</p>
                            </div>
                          </a>
                          <a
                            href="https://github.com/ctrl-gaurav/effGen/issues/new"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-3 py-2 bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-green-400 dark:hover:border-green-500 transition-colors group"
                          >
                            <FiGithub size={14} className="text-green-600 dark:text-green-400" />
                            <div className="text-left">
                              <p className="text-xs font-semibold text-gray-800 dark:text-gray-200">GitHub Issue</p>
                              <p className="text-xs text-gray-500 dark:text-gray-400 group-hover:text-green-500 dark:group-hover:text-green-400 transition-colors">Report a bug or request a feature</p>
                            </div>
                          </a>
                          <p className="text-xs text-gray-400 dark:text-gray-500 pt-1">We typically respond within 24–48 hours.</p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}

            {/* Bottom contact strip – always visible */}
            <div className="px-3 py-2 border-t border-gray-200 dark:border-gray-700 flex gap-2 flex-shrink-0">
              <a
                href="mailto:gks@vt.edu"
                className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 rounded-lg text-xs text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <FiMail size={12} /> Email
              </a>
              <a
                href="https://github.com/ctrl-gaurav/effGen/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 rounded-lg text-xs text-gray-600 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <FiGithub size={12} /> Issues
              </a>
            </div>

            {/* Input row – hidden in browse mode */}
            {view === "chat" && (
              <div className="px-3 py-2.5 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
                <div className="flex items-center gap-2">
                  <input
                    ref={inputRef}
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    placeholder="Type your question..."
                    className="flex-1 px-4 py-2 bg-gray-100 dark:bg-gray-800 rounded-xl text-sm text-gray-800 dark:text-gray-200 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:bg-white dark:focus:bg-gray-700 transition-colors"
                  />
                  <motion.button
                    onClick={handleSend}
                    disabled={!input.trim()}
                    className="p-2 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl disabled:opacity-40 disabled:cursor-not-allowed shadow-sm hover:shadow-md transition-shadow"
                    whileHover={input.trim() ? { scale: 1.08 } : {}}
                    whileTap={input.trim() ? { scale: 0.95 } : {}}
                  >
                    <FiSend size={16} />
                  </motion.button>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
