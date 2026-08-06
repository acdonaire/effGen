import React from 'react';
import { Link } from 'react-router-dom';
import { Wrench } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';

export default function Tools() {
  const toolFlowDiagram = `
flowchart LR
    A[Agent] --> B{Tool Needed?}
    B -->|Yes| C[Select Tool]
    B -->|No| F[Direct Response]
    C --> D[Execute Tool]
    D --> E[Return Result]
    E --> A
    F --> G[Output]
`;

  return (
    <DocPage
      title="Tools"
      subtitle="Tools extend agent capabilities by connecting to external systems, APIs, and functions. Learn to use built-in tools and create custom ones."
      icon={<Wrench size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Core Concepts', path: '/agents' },
        { label: 'Tools' },
      ]}
    >
      <h2>Overview</h2>
      <p>
        Tools are the primary way agents interact with the external world. When an agent needs to
        perform calculations, search the web, execute code, or access APIs, it uses tools.
      </p>

      <InfoBox type="success" title="v0.3.0 — every built-in tool is hardened">
        <p>
          As of v0.3.0 the built-in tools are sandboxed and SSRF-safe by default: every URL-taking
          tool shares one SSRF guard (<code>tools/builtin/_net.py</code>) that re-validates on each
          redirect and blocks private / loopback / metadata addresses; <code>PythonREPL</code>{' '}
          enforces its timeout out-of-process with memory / output caps; file tools are
          path-confined (<code>tools/builtin/_fs.py</code>); and the unsafe <code>pickle</code> /{' '}
          <code>eval</code> paths are gone. The tool count is unchanged — see{' '}
          <Link to="/security">Security</Link>. You can also turn any plain function into a tool with{' '}
          <code>@tool</code> / <code>Tool.from_function()</code>.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.1 — plugin auto-discovery &amp; a model-proof REPL sandbox">
        <p>
          Installed tool plugins now <strong>auto-discover</strong>: a package published with an{' '}
          <code>effgen.plugins</code> entry point (exactly what <code>effgen create-plugin</code>{' '}
          scaffolds) has its tools folded into the registry on first use, so they appear in{' '}
          <code>effgen tools list</code>, <code>effgen run -t &lt;tool&gt;</code>, the API, and over
          MCP with no manual <code>register()</code> call (<code>EFFGEN_DISABLE_PLUGINS=1</code>{' '}
          opts out). The Python REPL&apos;s <code>restricted_mode</code> sandbox toggle is now{' '}
          <strong>out of the model-facing schema</strong> — unrestricted execution is a
          developer-only opt-in (<code>PythonREPL(allow_unrestricted=True)</code> or{' '}
          <code>EFFGEN_REPL_ALLOW_UNRESTRICTED</code>), and a model-supplied{' '}
          <code>restricted_mode=False</code> is ignored. The <code>bash</code> env scrub now strips
          every provider credential and is no longer bundled in the <code>general</code> preset
          (it stays in <code>coding</code>), and <code>extra_tools</code> accepts tool-name strings.
        </p>
      </InfoBox>

      <MermaidDiagram chart={toolFlowDiagram} title="Tool Execution Flow" />

      <h2>Tool Calling Modes (v0.2.0)</h2>
      <p>
        effGen supports three strategies for how agents invoke tools, selected via
        <code> AgentConfig.tool_calling_mode</code>:
      </p>
      <ul>
        <li>
          <strong><code>"react"</code></strong> — free-text <code>Thought: ... Action: ... Action Input: {`{...}`}</code>
          parsed by <code>ReActStrategy</code>. Works with any model; the legacy default.
        </li>
        <li>
          <strong><code>"native"</code></strong> — the model's own function-calling format (OpenAI tools / Anthropic
          tool_use / Qwen / Llama / Mistral / generic). Uses <code>NativeFunctionCallingStrategy</code>.
          effGen probes <code>model.supports_tool_calling()</code> and passes
          <code>ToolDefinition</code> JSON Schemas via the chat template's <code>tools</code> parameter.
        </li>
        <li>
          <strong><code>"hybrid"</code></strong> — try native first; if parsing fails, fall back to ReAct.
          Safe default for mixed model fleets.
        </li>
        <li>
          <strong><code>"auto"</code></strong> (default) — pick native if the model reports
          <code>supports_tool_calling()</code>, otherwise ReAct.
        </li>
      </ul>

      <CodeBlock
        code={`from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin import Calculator

config = AgentConfig(
    name="calc_agent",
    model=model,
    tools=[Calculator()],
    tool_calling_mode="hybrid",   # "auto" | "native" | "react" | "hybrid"
)
agent = Agent(config)`}
        language="python"
        filename="tool_calling_mode.py"
      />

      <InfoBox type="info" title="Structured Output">
        <p>
          The same v0.2.0 work enabled JSON-schema-constrained structured output. Set
          <code> output_format="json"</code> or pass <code>output_schema</code> /
          <code>output_model</code> (Pydantic) on either <code>AgentConfig</code> or
          <code>Agent.run()</code> to validate model output against a schema.
        </p>
      </InfoBox>

      <h2>Provider-Native Tools (v0.2.1-v0.2.2)</h2>
      <p>
        In addition to local built-in tools, effGen exposes selected provider server-side
        tools as tool classes. These run inside the provider infrastructure and are guarded
        by <code>ToolIncompatibleError</code> if paired with the wrong model adapter.
      </p>
      <ApiTable
        headers={['Provider', 'Classes', 'Use cases']}
        rows={[
          ['OpenAI', <code>OpenAIWebSearchTool, OpenAICodeInterpreterTool, OpenAIFileSearchTool</code>, 'Responses API web search, hosted Python execution, vector store search'],
          ['Gemini (v0.2.2+)', <code>GoogleSearchTool, GeminiUrlContextTool, GeminiCodeExecutionTool</code>, 'Google Search grounding, server-side URL fetch, server-side code execution'],
          ['Anthropic (v0.2.2+)', <code>AnthropicBashTool, AnthropicTextEditorTool, AnthropicComputerTool</code>, 'Experimental computer-use tools behind Anthropic beta headers'],
        ]}
      />
      <CodeBlock
        code={`from effgen.core.agent import Agent, AgentConfig
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

model = OpenAIAdapter(model_name="gpt-5.4-nano")
model.load()

agent = Agent(AgentConfig(
    name="native-search",
    model=model,
    tools=[OpenAIWebSearchTool()],
    tool_calling_mode="native",
))`}
        language="python"
        filename="provider_native_tools.py"
      />
      <p>
        See <Link to="/native-provider-tools">Native Provider Tools</Link> for the full
        OpenAI, Gemini, and Anthropic reference.
      </p>

      <h2>Built-in Tools</h2>
      <p>
        effGen ships <strong>66 built-in tools</strong> spanning computation, code execution, information retrieval,
        academic research, news &amp; RSS, YouTube, social, translation, language detection, QR codes, OCR, audio transcription,
        image analysis, document parsing (PDF/DOCX/Excel), geo/weather (Open-Meteo + Nominatim + OSM static), email (SMTP/IMAP),
        webhooks (Slack, Discord), data processing, file ops, system, DevOps, finance, knowledge, and communication. All are
        lazy-loaded; optional deps are noted inline. v0.2.6 added <strong>14 new tools</strong>; v0.2.5 added
        <strong> 13 free / no-auth tools</strong>. (v0.2.7-v0.2.10 focus on the
        <a href="/docs/prompts">Prompt Library</a>, <a href="/docs/multimodal">multimodal input</a>,{' '}
        <a href="/docs/observability">observability &amp; reliability</a>, and{' '}
        <a href="/docs/security">security / deploy / DX</a> rather than new built-in tools — though as of
        v0.2.10 the code-execution tool runs inside a sandbox by default.)
      </p>

      <ApiTable
        headers={['Tool', 'Class', 'Description']}
        rows={[
          [<code>Calculator</code>, 'Calculator', 'Mathematical calculations, unit conversions, statistics'],
          [<code>CodeExecutor</code>, 'CodeExecutor', 'Execute Python code in a secure sandbox'],
          [<code>PythonREPL</code>, 'PythonREPL', 'Interactive Python with state persistence'],
          [<code>FileOperations</code>, 'FileOperations', 'Read, write, list, and manipulate files'],
          [<code>WebSearch</code>, 'WebSearch', 'Search the web using DuckDuckGo, SerpAPI, or Google'],
          [<code>Retrieval</code>, 'Retrieval', 'RAG-based semantic search with BM25 hybrid'],
          [<code>AgenticSearch</code>, 'AgenticSearch', 'ripgrep-based search for precise queries'],
          [<code>BashTool</code>, 'BashTool', 'Execute shell commands with security controls (whitelist/blacklist)'],
          [<code>WeatherTool</code>, 'WeatherTool', 'Weather data via Open-Meteo API (free, no API key)'],
          [<code>JSONTool</code>, 'JSONTool', 'Parse, query (JSONPath), transform, and validate JSON'],
          [<code>DateTimeTool</code>, 'DateTimeTool', 'Current time, timezone conversion, date arithmetic'],
          [<code>TextProcessingTool</code>, 'TextProcessingTool', 'Word count, regex operations, text comparison'],
          [<code>URLFetchTool</code>, 'URLFetchTool', 'Fetch and extract text from web pages'],
          [<code>WikipediaTool</code>, 'WikipediaTool', 'Search and retrieve Wikipedia articles (free API)'],
          [<code>StockPriceTool</code>, 'StockPriceTool', 'Stock quotes via yfinance + Yahoo Finance v8 fallback (disclaimer included)'],
          [<code>CurrencyConverterTool</code>, 'CurrencyConverterTool', 'FX conversion via frankfurter.app / ECB rates'],
          [<code>CryptoTool</code>, 'CryptoTool', 'Crypto prices and market data via CoinGecko'],
          [<code>DataFrameTool</code>, 'DataFrameTool', 'Pandas DataFrames: load / head / describe / filter / aggregate'],
          [<code>PlotTool</code>, 'PlotTool', 'Matplotlib line / bar / scatter / hist → PNG'],
          [<code>StatsTool</code>, 'StatsTool', 'NumPy mean / median / std / variance / summary / correlation / regression'],
          [<code>GitTool</code>, 'GitTool', 'Read-only Git: status, log, diff, branch, show, remote'],
          [<code>DockerTool</code>, 'DockerTool', 'Read-only Docker: ps, images, logs, version, info'],
          [<code>SystemInfoTool</code>, 'SystemInfoTool', 'System metrics via psutil (CPU, memory, disk, network)'],
          [<code>HTTPTool</code>, 'HTTPTool', 'HTTP GET/POST via urllib'],
          [<code>ArXivTool</code>, 'ArXivTool', 'Search arXiv papers, fetch by ID, download PDFs (Atom feed; no auth)'],
          [<code>PubMedTool</code>, 'PubMedTool', 'NCBI E-utilities: search / fetch / abstract (3 req/s; 10/s with NCBI_API_KEY) — v0.2.5'],
          [<code>SemanticScholarTool</code>, 'SemanticScholarTool', 'Semantic Scholar Graph API: search / paper / citations / references (100 req/5 min unauth) — v0.2.5'],
          [<code>RSSFeedTool</code>, 'RSSFeedTool', 'Fetch / browse / full-text search any RSS or Atom feed — v0.2.5'],
          [<code>NewsTool</code>, 'NewsTool', 'Curated reputable RSS sources (Reuters/BBC/HN/NPR/…); optional NEWS_API_KEY — v0.2.5'],
          [<code>YouTubeTranscriptTool</code>, 'YouTubeTranscriptTool', 'YouTube captions via youtube-transcript-api (no Google API key) — v0.2.5'],
          [<code>YouTubeMetadataTool</code>, 'YouTubeMetadataTool', 'Video / channel metadata via yt-dlp metadata-only mode — v0.2.5'],
          [<code>RedditTool</code>, 'RedditTool', 'Reddit top/hot/user/thread via public JSON (no OAuth for reads) — v0.2.5'],
          [<code>HackerNewsTool</code>, 'HackerNewsTool', 'HN top/new stories, story details, user profiles (Firebase API) — v0.2.5'],
          [<code>TranslateTool</code>, 'TranslateTool', 'LibreTranslate (configurable URL) + offline argostranslate fallback — v0.2.5'],
          [<code>LanguageDetectTool</code>, 'LanguageDetectTool', 'Offline language detection via langdetect (55+ languages) — v0.2.5'],
          [<code>QRGenerateTool</code>, 'QRGenerateTool', 'Local QR generation; PNG file or base64 data URL (no network) — v0.2.5'],
          [<code>QRReadTool</code>, 'QRReadTool', 'Local QR / barcode decode via pyzbar + Pillow, OpenCV QR fallback — v0.2.5'],
          [<code>OCRTool</code>, 'OCRTool', 'Tesseract (local) + OCR.space fallback; per-OS install instructions on OCRBackendUnavailable — v0.2.6'],
          [<code>AudioTranscribeTool</code>, 'AudioTranscribeTool', 'faster-whisper CPU/GPU auto-detect + HuggingFace Inference fallback (HF_TOKEN) — v0.2.6'],
          [<code>ImageInfoTool</code>, 'ImageInfoTool', 'Pillow: size / format / mode / EXIF / histogram + resize / thumbnail (zero network) — v0.2.6'],
          [<code>ImageCaptionTool</code>, 'ImageCaptionTool', 'Vision-capable provider via model router (Gemini / OpenAI / MLX-VLM) — v0.2.6'],
          [<code>PDFTool</code>, 'PDFTool', 'pypdf text/metadata + pdfplumber tables + extract_images — v0.2.6'],
          [<code>DOCXTool</code>, 'DOCXTool', 'python-docx: text / paragraphs / tables / metadata — v0.2.6'],
          [<code>ExcelTool</code>, 'ExcelTool', 'openpyxl + pandas: sheets / read_sheet / headers — v0.2.6'],
          [<code>GeocodeTool</code>, 'GeocodeTool', 'Nominatim (OSM) forward / reverse with 1 req/s token bucket and effGen User-Agent — v0.2.6'],
          [<code>MapsTool</code>, 'MapsTool', 'Static PNG maps from OSM tiles via the staticmap library — v0.2.6'],
          [<code>EmailSMTPTool</code>, 'EmailSMTPTool', 'Live SMTP send (stdlib smtplib, TLS-on) — SMTP_HOST/USER/PASSWORD/FROM env — v0.2.6'],
          [<code>EmailIMAPTool</code>, 'EmailIMAPTool', 'IMAP read: list_folders / fetch_recent / search / get — IMAP_* env — v0.2.6'],
          [<code>SlackWebhookTool</code>, 'SlackWebhookTool', 'Post to Slack via incoming webhook URL — SLACK_WEBHOOK_URL (redacted in logs) — v0.2.6'],
          [<code>DiscordWebhookTool</code>, 'DiscordWebhookTool', 'Post to Discord via webhook URL — DISCORD_WEBHOOK_URL (redacted in logs) — v0.2.6'],
          [<code>StackOverflowTool</code>, 'StackOverflowTool', 'Stack Exchange API search'],
          [<code>GitHubTool</code>, 'GitHubTool', 'Public GitHub search: repos, code, issues'],
          [<code>WolframAlphaTool</code>, 'WolframAlphaTool', 'WolframAlpha (requires free AppID)'],
          [<code>EmailDraftTool</code>, 'EmailDraftTool', 'Draft emails — does NOT send (legacy)'],
          [<code>SlackDraftTool</code>, 'SlackDraftTool', 'Draft Slack messages — does NOT post (legacy)'],
          [<code>NotificationTool</code>, 'NotificationTool', 'Local desktop notifications via plyer (optional)'],
        ]}
      />

      <h3>Calculator</h3>
      <CodeBlock
        code={`from effgen.tools.builtin import Calculator
import asyncio

calc = Calculator()

async def demo():
    # Basic math
    result = await calc.execute(expression="sqrt(144) + 8")
    print(result.output)  # {"result": 20.0}

    # Unit conversion
    result = await calc.execute(expression="convert(100, 'km', 'miles')")
    print(result.output)  # {"result": 62.137...}

    # Statistics
    result = await calc.execute(expression="mean([1, 2, 3, 4, 5])")
    print(result.output)  # {"result": 3.0}

asyncio.run(demo())`}
        language="python"
        filename="calculator_tool.py"
      />

      <h3>CodeExecutor</h3>
      <CodeBlock
        code={`from effgen.tools.builtin import CodeExecutor
import asyncio

executor = CodeExecutor()

async def demo():
    result = await executor.execute(
        code='''
import math
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print([fibonacci(i) for i in range(10)])
''',
        language="python",
        timeout=30,
        memory_limit="512m",
        network_enabled=False
    )
    print(result.output)  # {"stdout": "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\\n"}

asyncio.run(demo())`}
        language="python"
        filename="code_executor_tool.py"
      />

      <h3>PythonREPL</h3>
      <p>
        Unlike CodeExecutor, PythonREPL maintains state between calls:
      </p>

      <CodeBlock
        code={`from effgen.tools.builtin import PythonREPL
import asyncio

repl = PythonREPL()

async def demo():
    # First call - define variable
    await repl.execute(code='x = 42')

    # Second call - use variable (state persisted)
    result = await repl.execute(code='print(x * 2)')
    print(result.output)  # {"stdout": "84\\n"}

    # Third call - modify and use
    await repl.execute(code='x = x + 8')
    result = await repl.execute(code='print(x)')
    print(result.output)  # {"stdout": "50\\n"}

asyncio.run(demo())`}
        language="python"
        filename="python_repl_tool.py"
      />

      <h3>FileOperations</h3>
      <CodeBlock
        code={`from effgen.tools.builtin import FileOperations
import asyncio
import tempfile

file_tool = FileOperations()

# Add allowed directory for security
tmpdir = tempfile.mkdtemp()
file_tool.add_allowed_directory(tmpdir)

async def demo():
    # Write file
    await file_tool.execute(
        operation="write",
        path=f"{tmpdir}/test.txt",
        content="Hello, World!"
    )

    # Read file
    result = await file_tool.execute(
        operation="read",
        path=f"{tmpdir}/test.txt"
    )
    print(result.output)  # {"content": "Hello, World!"}

    # List directory
    result = await file_tool.execute(
        operation="list",
        path=tmpdir
    )
    print(result.output)  # {"files": ["test.txt"]}

asyncio.run(demo())`}
        language="python"
        filename="file_operations_tool.py"
      />

      <h3>Retrieval (RAG Search)</h3>
      <p>
        The <code>Retrieval</code> tool provides semantic search over a knowledge base using embeddings.
        This is ideal for finding relevant documents based on meaning rather than exact matches.
      </p>

      <InfoBox type="info" title="When to use Retrieval">
        <ul>
          <li>Searching for conceptually similar content</li>
          <li>Question-answering over documents</li>
          <li>Finding related information across large knowledge bases</li>
          <li>When exact keywords may not be known</li>
        </ul>
      </InfoBox>

      <CodeBlock
        code={`from effgen.tools.builtin import Retrieval
import asyncio

# Create retrieval tool
retrieval = Retrieval(
    chunk_size=500,        # Size of document chunks in characters
    chunk_overlap=100,     # Overlap between chunks
    index_path="./my_index.pkl"  # Optional: persist index to disk
)

# Add documents to the index
retrieval.add_documents([
    {"content": "Python is a high-level programming language...", "id": "doc1"},
    {"content": "Machine learning is a subset of AI...", "id": "doc2"},
    {"content": "Neural networks are inspired by the brain...", "id": "doc3"},
])

# Or add from files
retrieval.add_from_file("knowledge_base.txt", chunk=True)
retrieval.add_from_file("data.json", file_type="json")

async def demo():
    # Search the knowledge base
    result = await retrieval.execute(
        query="What is machine learning?",
        top_k=5,
        score_threshold=0.3
    )

    print(f"Found {result['total_found']} results")
    for r in result['results']:
        print(f"Score: {r['score']:.2f}")
        print(f"Content: {r['content'][:100]}...")
        print(f"Metadata: {r['metadata']}")
        print("---")

asyncio.run(demo())

# Save the index for later use
retrieval.save_index("./my_index.pkl")`}
        language="python"
        filename="retrieval_tool.py"
      />

      <h4>Retrieval Parameters</h4>
      <ApiTable
        headers={['Parameter', 'Type', 'Description']}
        rows={[
          [<code>query</code>, 'string', 'The search query to find relevant documents'],
          [<code>top_k</code>, 'integer', 'Number of top results to return (default: 5)'],
          [<code>score_threshold</code>, 'float', 'Minimum similarity score 0-1 (default: 0.0)'],
          [<code>filter_metadata</code>, 'object', 'Filter results by metadata fields'],
        ]}
      />

      <h4>Embedding Providers</h4>
      <p>
        The Retrieval tool supports multiple embedding providers:
      </p>

      <CodeBlock
        code={`from effgen.tools.builtin import Retrieval
from effgen.tools.builtin.retrieval import (
    SentenceTransformerEmbedding,
    SimpleEmbedding
)

# Use Sentence Transformers (best quality, requires sentence-transformers)
embedding = SentenceTransformerEmbedding(
    model_name="all-MiniLM-L6-v2"  # Fast and lightweight
)

# Or use TF-IDF fallback (no extra dependencies)
embedding = SimpleEmbedding()

# Create retrieval with custom embedding
retrieval = Retrieval(embedding_provider=embedding)`}
        language="python"
        filename="retrieval_embeddings.py"
      />

      <h3>AgenticSearch (Exact Match)</h3>
      <p>
        The <code>AgenticSearch</code> tool uses ripgrep (<code>rg</code>) with grep fallback for precise text matching.
        Unlike embedding-based RAG, this finds exact phrases, technical terms, and specific patterns.
      </p>

      <InfoBox type="info" title="When to use AgenticSearch">
        <ul>
          <li>Finding exact phrases, numbers, or formulas</li>
          <li>Technical queries (code snippets, specific terms)</li>
          <li>When semantic search might miss precise answers</li>
          <li>Large knowledge bases where indexing is impractical</li>
          <li>Case-sensitive or regex pattern matching</li>
        </ul>
      </InfoBox>

      <CodeBlock
        code={`from effgen.tools.builtin import AgenticSearch
import asyncio

# Create agentic search tool
search = AgenticSearch(
    data_path="./knowledge_base",  # Directory or file to search
    context_lines=5,                # Lines of context around matches
    max_results=10,                 # Maximum results to return
    supported_extensions=[".txt", ".md", ".json", ".csv"]
)

async def demo():
    # Exact phrase search
    result = await search.execute(
        query="photosynthesis converts",
        search_mode="exact",         # "exact", "keywords", or "any"
        context_lines=5,
        case_sensitive=False
    )

    print(f"Found {result['total_matches']} matches")
    print(f"Query terms: {result['query_terms']}")

    for r in result['results']:
        print(f"File: {r['file']}")
        print(f"Line: {r['line_number']}")
        print(f"Score: {r['score']:.2f}")
        print(f"Match: {r['match_line']}")
        print(f"Context:\\n{r['content']}")
        print("---")

asyncio.run(demo())`}
        language="python"
        filename="agentic_search_tool.py"
      />

      <h4>AgenticSearch Parameters</h4>
      <ApiTable
        headers={['Parameter', 'Type', 'Description']}
        rows={[
          [<code>query</code>, 'string', 'Search query (exact text, keywords, or regex)'],
          [<code>context_lines</code>, 'integer', 'Lines of context before/after match (default: 5)'],
          [<code>case_sensitive</code>, 'boolean', 'Case-sensitive search (default: false)'],
          [<code>use_regex</code>, 'boolean', 'Treat query as regex pattern (default: false)'],
          [<code>max_results</code>, 'integer', 'Maximum results (default: 10)'],
          [<code>search_mode</code>, 'string', 'Mode: "exact", "keywords", or "any" (default: "keywords")'],
          [<code>file_type</code>, 'string', 'Filter: "all", "python", "json", "markdown", or "text"'],
        ]}
      />

      <h4>Search Modes</h4>
      <ApiTable
        headers={['Mode', 'Description', 'Use Case']}
        rows={[
          [<code>exact</code>, 'Search for exact phrase as-is', 'Finding specific sentences or quotes'],
          [<code>keywords</code>, 'Find results containing ALL keywords (AND logic)', 'Technical queries with multiple terms'],
          [<code>any</code>, 'Find results containing ANY keyword (OR logic)', 'Broad searches with alternatives'],
        ]}
      />

      <CodeBlock
        code={`# Search modes example
async def search_modes():
    search = AgenticSearch(data_path="./docs")

    # Exact phrase - finds "machine learning algorithms" verbatim
    await search.execute(
        query="machine learning algorithms",
        search_mode="exact"
    )

    # Keywords (AND) - finds lines containing BOTH "neural" AND "network"
    await search.execute(
        query="neural network",
        search_mode="keywords"
    )

    # Any (OR) - finds lines containing "GPU" OR "CUDA" OR "parallel"
    await search.execute(
        query="GPU CUDA parallel",
        search_mode="any"
    )

    # Regex pattern - finds function definitions
    await search.execute(
        query="def\\\\s+\\\\w+\\\\(",
        use_regex=True
    )`}
        language="python"
        filename="search_modes.py"
      />

      <h4>Comparing Retrieval vs AgenticSearch</h4>
      <ApiTable
        headers={['Aspect', 'Retrieval (RAG)', 'AgenticSearch (Grep)']}
        rows={[
          ['Search Type', 'Semantic/conceptual similarity', 'Exact text matching'],
          ['Best For', 'Q&A, conceptual queries', 'Technical terms, exact phrases'],
          ['Speed', 'Requires pre-indexing', 'Direct file search'],
          ['Accuracy', 'May miss exact matches', 'Precise but no semantic understanding'],
          ['Setup', 'Needs embedding model', 'Uses ripgrep/grep exact matching'],
        ]}
      />

      <h3>BashTool</h3>
      <CodeBlock
        code={`from effgen.tools.builtin import BashTool
import asyncio

bash = BashTool(
    allowed_commands=["ls", "cat", "echo", "grep"],  # Whitelist mode
    timeout=30,
    working_directory="/tmp"
)

async def demo():
    result = await bash.execute(command="echo 'Hello from shell!'")
    print(result.output)  # {"stdout": "Hello from shell!\\n", "exit_code": 0}

    result = await bash.execute(command="ls -la /tmp")
    print(result.output)

asyncio.run(demo())`}
        language="python"
        filename="bash_tool.py"
      />

      <h3>WeatherTool</h3>
      <CodeBlock
        code={`from effgen.tools.builtin import WeatherTool
import asyncio

weather = WeatherTool()  # Uses free Open-Meteo API, no key needed

async def demo():
    result = await weather.execute(
        location="San Francisco",
        units="metric"
    )
    print(result.output)
    # {"temperature": 18.5, "condition": "Partly Cloudy", ...}

asyncio.run(demo())`}
        language="python"
        filename="weather_tool.py"
      />

      <h3>JSONTool</h3>
      <CodeBlock
        code={`from effgen.tools.builtin import JSONTool
import asyncio

json_tool = JSONTool()

async def demo():
    data = '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'

    # Query with JSONPath
    result = await json_tool.execute(
        data=data,
        operation="query",
        query="$.users[0].name"
    )
    print(result.output)  # {"result": "Alice"}

    # Validate JSON
    result = await json_tool.execute(data=data, operation="validate")
    print(result.output)  # {"valid": true}

asyncio.run(demo())`}
        language="python"
        filename="json_tool.py"
      />

      <h3>DateTimeTool</h3>
      <CodeBlock
        code={`from effgen.tools.builtin import DateTimeTool
import asyncio

dt = DateTimeTool()

async def demo():
    # Get current time in a timezone
    result = await dt.execute(operation="now", timezone_param="JST")
    print(result.output)  # {"datetime": "2026-03-01T12:00:00+09:00", ...}

    # Date arithmetic
    result = await dt.execute(
        operation="add",
        date="2026-03-01",
        days=30, hours=5
    )
    print(result.output)  # {"result": "2026-03-31T05:00:00"}

    # Timezone conversion
    result = await dt.execute(
        operation="convert",
        date="2026-03-01T12:00:00",
        timezone_param="EST",
        to_timezone="JST"
    )
    print(result.output)

asyncio.run(demo())`}
        language="python"
        filename="datetime_tool.py"
      />

      <h3>TextProcessingTool</h3>
      <CodeBlock
        code={`from effgen.tools.builtin import TextProcessingTool
import asyncio

text_tool = TextProcessingTool()

async def demo():
    # Word count
    result = await text_tool.execute(
        operation="word_count",
        text="Hello world, this is a test."
    )
    print(result.output)  # {"word_count": 6, "char_count": 27, ...}

    # Regex search
    result = await text_tool.execute(
        operation="regex_search",
        text="Email: user@example.com and admin@test.com",
        pattern=r"[\\w.]+@[\\w.]+"
    )
    print(result.output)  # {"matches": ["user@example.com", "admin@test.com"]}

    # Text comparison
    result = await text_tool.execute(
        operation="compare",
        text="Hello World",
        text2="Hello World!"
    )
    print(result.output)

asyncio.run(demo())`}
        language="python"
        filename="text_processing_tool.py"
      />

      <h3>URLFetchTool</h3>
      <CodeBlock
        code={`from effgen.tools.builtin import URLFetchTool
import asyncio

url_fetch = URLFetchTool(
    max_content_length=10000,
    timeout=15
)

async def demo():
    result = await url_fetch.execute(
        url="https://example.com",
        extract_links=True
    )
    print(result.output)
    # {"content": "...", "title": "Example Domain", "links": [...]}

asyncio.run(demo())`}
        language="python"
        filename="url_fetch_tool.py"
      />

      <h3>WikipediaTool</h3>
      <CodeBlock
        code={`from effgen.tools.builtin import WikipediaTool
import asyncio

wiki = WikipediaTool()

async def demo():
    # Search Wikipedia
    result = await wiki.execute(
        query="quantum computing",
        operation="search"
    )
    print(result.output)  # {"results": ["Quantum computing", ...]}

    # Get article summary
    result = await wiki.execute(
        query="Quantum computing",
        operation="summary",
        sentences=3
    )
    print(result.output)  # {"title": "Quantum computing", "summary": "..."}

asyncio.run(demo())`}
        language="python"
        filename="wikipedia_tool.py"
      />

      <h2>Tool Fallback Chains</h2>
      <p>
        effGen supports automatic fallback chains for resilient tool execution. If a primary tool
        fails, the system automatically tries the next tool in the chain:
      </p>

      <CodeBlock
        code={`# Built-in fallback chain: calculator → python_repl → code_executor
# If Calculator fails on a complex expression, PythonREPL is tried next,
# and if that fails, CodeExecutor handles it.

from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL, CodeExecutor

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

config = AgentConfig(
    name="math_agent",
    model=model,
    tools=[Calculator(), PythonREPL(), CodeExecutor()],
    # Fallback chains are configured automatically
)

agent = Agent(config=config)
# If "integrate sin(x) from 0 to pi" fails in Calculator,
# it automatically falls back to PythonREPL with scipy
result = agent.run("Integrate sin(x) from 0 to pi")`}
        language="python"
        filename="fallback_chains.py"
      />

      <h2>Circuit Breaker</h2>
      <p>
        The circuit breaker pattern tracks tool failures and temporarily disables tools that are
        consistently failing, preventing cascading errors:
      </p>

      <InfoBox type="info" title="How Circuit Breaker Works">
        <ul>
          <li>Tracks consecutive failures per tool</li>
          <li>After a configurable threshold (default: 3 failures), the tool is temporarily disabled</li>
          <li>After a cooldown period, the tool is re-enabled for a trial run</li>
          <li>If the trial succeeds, the tool is fully restored; otherwise it stays disabled</li>
        </ul>
      </InfoBox>

      <h2>Plugin System</h2>
      <p>
        Create custom tool plugins that can be discovered and loaded automatically:
      </p>

      <CodeBlock
        code={`from effgen.tools.plugin import ToolPlugin, PluginManager

# Create a plugin
class MyPlugin(ToolPlugin):
    name = "my-plugin"
    version = "1.0.0"
    description = "Custom tools for my project"
    tools = [MyCustomTool, AnotherTool]

# Discover plugins via entry points (setup.py / pyproject.toml)
# [project.entry-points."effgen.plugins"]
# my_plugin = "my_package:MyPlugin"

# Or discover from a directory
manager = PluginManager()
manager.discover_user_dir()     # ~/.effgen/plugins/
manager.discover_env_dir()      # EFFGEN_PLUGINS_DIR env var
manager.discover_entry_points() # Python entry points
manager.discover_all()          # All of the above`}
        language="python"
        filename="plugin_system.py"
      />

      <h2>Using Tools with Agents</h2>

      <CodeBlock
        code={`from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL, FileOperations

model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

# Create agent with multiple tools
config = AgentConfig(
    name="multi_tool_agent",
    model=model,
    tools=[
        Calculator(),
        PythonREPL(),
        FileOperations()
    ],
    system_prompt="""You have access to:
- Calculator: for math operations
- PythonREPL: for Python code execution
- FileOperations: for file management

Choose the appropriate tool for each task."""
)

agent = Agent(config=config)

# Agent will automatically select appropriate tools
result = agent.run("Calculate sqrt(256), then write a Python script to generate prime numbers up to 50")`}
        language="python"
        filename="agent_with_tools.py"
      />

      <h2>Creating Custom Tools</h2>
      <p>
        Create your own tools by extending <code>BaseTool</code>:
      </p>

      <CodeBlock
        code={`from effgen import AgentConfig, load_model
from effgen.tools import BaseTool
from effgen.tools.base_tool import (
    ToolMetadata, ToolCategory, ParameterSpec, ParameterType,
)
from effgen.tools.builtin import Calculator
import aiohttp

class WeatherTool(BaseTool):
    """Custom tool to fetch weather information."""

    def __init__(self, api_key: str):
        super().__init__(
            metadata=ToolMetadata(
                name="weather",
                description="Get current weather for a city",
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="city",
                        type=ParameterType.STRING,
                        description="City name (e.g., 'Tokyo', 'New York')",
                        required=True,
                    ),
                    ParameterSpec(
                        name="units",
                        type=ParameterType.STRING,
                        description="Temperature units",
                        required=False,
                        default="celsius",
                        enum=["celsius", "fahrenheit"],
                    ),
                ],
            )
        )
        self.api_key = api_key

    async def _execute(self, city: str, units: str = "celsius", **kwargs) -> dict:
        """Execute the weather lookup."""
        async with aiohttp.ClientSession() as session:
            url = f"https://api.weather.com/v1/current?city={city}&units={units}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "city": city,
                        "temperature": data["temp"],
                        "condition": data["condition"],
                        "units": units,
                    }
                raise Exception(f"API error: {response.status}")

# Use custom tool
model = load_model("Qwen/Qwen2.5-7B-Instruct")
weather = WeatherTool(api_key="your-api-key")
config = AgentConfig(
    name="weather_agent",
    model=model,
    tools=[weather, Calculator()],
)`}
        language="python"
        filename="custom_tool.py"
      />

      <h2>Tool Parameters</h2>
      <p>
        Define tool parameters with full validation:
      </p>

      <CodeBlock
        code={`from effgen.tools import BaseTool
from effgen.tools.base_tool import (
    ToolMetadata, ToolCategory, ParameterSpec, ParameterType,
)

class DatabaseTool(BaseTool):
    """Query a database."""

    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="database",
                description="Execute database queries",
                category=ToolCategory.SYSTEM,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="SQL query to execute",
                        required=True,
                    ),
                    ParameterSpec(
                        name="database",
                        type=ParameterType.STRING,
                        description="Database name",
                        required=True,
                        enum=["users", "products", "orders"],
                    ),
                    ParameterSpec(
                        name="limit",
                        type=ParameterType.INTEGER,
                        description="Maximum rows to return",
                        required=False,
                        default=100,
                        min_value=1,
                        max_value=1000,
                    ),
                ],
            )
        )

    async def _execute(self, query: str, database: str, limit: int = 100, **kwargs) -> dict:
        # Validate and execute query
        # ... implementation
        return {"rows": [], "count": 0}`}
        language="python"
        filename="parameter_validation.py"
      />

      <h2>Tool Registry</h2>
      <p>
        Use the tool registry for dynamic tool discovery:
      </p>

      <CodeBlock
        code={`import asyncio
from effgen.tools import get_registry

async def main():
    # Get the global registry
    registry = get_registry()

    # Discover all built-in tools
    registry.discover_builtin_tools()

    # Register a custom no-arg tool class
    # registry.register_tool(WeatherTool)

    # Get tool by name
    calculator = await registry.get_tool("calculator")
    result = await calculator.execute(expression="2 + 2")

    # List all available tools and metadata
    for name in registry.list_tools():
        metadata = registry.get_metadata(name)
        print(f"{name}: {metadata.description}")

asyncio.run(main())`}
        language="python"
        filename="tool_registry.py"
      />

      <h2>v0.2.0 Domain Tools</h2>
      <p>
        17 domain-specific tools added in v0.2.0. All external libraries are optional and checked at import
        time with clear install hints if missing. Draft-only tools (email, Slack) deliberately never send.
      </p>

      <h3>Finance</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin import StockPriceTool, CurrencyConverterTool, CryptoTool

async def main():
    stock = StockPriceTool()
    print((await stock.execute(symbol="AAPL")).output)
    # {"symbol": "AAPL", "price": 195.12, "disclaimer": "...Not financial advice."}

    fx = CurrencyConverterTool()
    print((await fx.execute(amount=100, from_currency="USD", to_currency="EUR")).output)

    crypto = CryptoTool()
    print((await crypto.execute(coin="bitcoin", vs_currency="usd")).output)

asyncio.run(main())`}
        language="python"
        filename="finance_tools.py"
      />

      <h3>Data Science</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin import DataFrameTool, PlotTool, StatsTool

async def main():
    df = DataFrameTool()
    print((await df.execute(operation="describe", file_path="./sales.csv")).output)

    plot = PlotTool()
    await plot.execute(chart_type="line", x=[1,2,3], y=[4,5,6], output_path="out.png")

    stats = StatsTool()
    print((await stats.execute(operation="correlation", data=[1,2,3], y=[2,4,6])).output)

asyncio.run(main())`}
        language="python"
        filename="data_science_tools.py"
      />

      <h3>DevOps</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin import GitTool, DockerTool, SystemInfoTool, HTTPTool

async def main():
    git = GitTool()
    print((await git.execute(operation="status", cwd=".")).output)         # read-only

    docker = DockerTool()
    print((await docker.execute(operation="ps")).output)                   # read-only

    sysinfo = SystemInfoTool()
    print((await sysinfo.execute(kind="memory")).output)                   # psutil-backed

    http = HTTPTool()
    print((await http.execute(method="GET", url="https://api.github.com")).output)  # stdlib urllib

asyncio.run(main())`}
        language="python"
        filename="devops_tools.py"
      />

      <h3>Knowledge</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin import ArxivTool, StackOverflowTool, GitHubTool, WolframAlphaTool

async def main():
    arxiv = ArxivTool()
    print((await arxiv.execute(query="small language model agents", max_results=5)).output)

    so = StackOverflowTool()
    print((await so.execute(query="pandas groupby apply", max_results=3)).output)

    gh = GitHubTool()
    print((await gh.execute(query="effgen agent", kind="repositories")).output)

    # Optional: requires free AppID from wolframalpha.com
    wolfram = WolframAlphaTool(app_id="YOUR_APPID")
    print((await wolfram.execute(query="integrate x^2 dx")).output)

asyncio.run(main())`}
        language="python"
        filename="knowledge_tools.py"
      />

      <h3>Communication (Draft-Only)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin import EmailDraftTool, SlackDraftTool, NotificationTool

async def main():
    email = EmailDraftTool()
    print((await email.execute(to=["alice@example.com"], subject="Hi", body="...")).output)
    # Returns a structured draft; does NOT send.

    slack = SlackDraftTool()
    print((await slack.execute(channel="#eng", text="Deploy finished")).output)
    # Draft only; never posts to Slack.

    notify = NotificationTool()       # local desktop, via plyer (optional)
    await notify.execute(title="Done", message="Agent finished", timeout=5)

asyncio.run(main())`}
        language="python"
        filename="communication_tools.py"
      />

      <h3>Academic Research (v0.2.5)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.pubmed import PubMedTool
from effgen.tools.builtin.arxiv import ArXivTool
from effgen.tools.builtin.semantic_scholar import SemanticScholarTool

async def main():
    # PubMed via NCBI E-utilities. 3 req/s without key; 10/s with NCBI_API_KEY.
    pubmed = PubMedTool()
    r = await pubmed.execute({"operation": "search", "query": "CRISPR gene editing", "max_results": 5})
    for a in r["data"]["articles"]:
        print(a["pmid"], a["title"])

    # arXiv Atom feed — free, no auth.
    arxiv = ArXivTool()
    r = await arxiv.execute({"operation": "search", "query": "attention is all you need", "max_results": 3})
    for p in r["data"]["papers"]:
        print(p["id"], p["title"])

    # Semantic Scholar Graph API — backoff on 100 req / 5 min unauth.
    s2 = SemanticScholarTool()
    r = await s2.execute({"operation": "search", "query": "large language models survey"})
    for p in r["data"]["papers"][:3]:
        print(p.get("paperId"), p.get("title"))

asyncio.run(main())`}
        language="python"
        filename="academic_tools.py"
      />

      <h3>News &amp; RSS (v0.2.5)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.rss import RSSFeedTool
from effgen.tools.builtin.news import NewsTool

async def main():
    rss = RSSFeedTool()
    r = await rss.execute({"operation": "latest", "url": "https://hnrss.org/frontpage", "n": 5})
    for item in r["data"]["items"]:
        print(item["title"])

    news = NewsTool()  # curated reputable sources (Reuters, BBC, HN, NPR, Al Jazeera, …)
    r = await news.execute({"operation": "top_headlines"})
    for a in r["data"]["articles"][:5]:
        print(a["title"], "-", a["source"])

asyncio.run(main())`}
        language="python"
        filename="news_rss_tools.py"
      />

      <h3>YouTube (v0.2.5)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.youtube_transcript import YouTubeTranscriptTool
from effgen.tools.builtin.youtube_metadata import YouTubeMetadataTool

async def main():
    # Captions / transcripts via youtube-transcript-api — no Google API key.
    yt = YouTubeTranscriptTool()
    r = await yt.execute({
        "operation": "get_transcript",
        "video_id": "dQw4w9WgXcQ",  # accepts watch?v=, youtu.be/, shorts/ URLs too
        "lang": "en",
    })
    print(r["data"]["transcript"][:500])

    # Video / channel metadata via yt-dlp metadata-only mode.
    ytm = YouTubeMetadataTool()
    r = await ytm.execute({"operation": "metadata", "video_id": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
    print(r["data"].get("title"), r["data"].get("uploader"))

asyncio.run(main())`}
        language="python"
        filename="youtube_tools.py"
      />

      <h3>Social — Reddit &amp; Hacker News (v0.2.5)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.reddit import RedditTool
from effgen.tools.builtin.hackernews import HackerNewsTool

async def main():
    reddit = RedditTool()  # public JSON — no OAuth for reads; exponential backoff on 429
    r = await reddit.execute({"operation": "subreddit_top", "subreddit": "python", "time": "day", "n": 5})
    for post in r["data"]["posts"]:
        print(post["title"])

    hn = HackerNewsTool()  # HN Firebase API — no auth
    r = await hn.execute({"operation": "top_stories", "n": 5})
    for s in r["data"]["stories"]:
        print(s["title"], s.get("url", ""))

asyncio.run(main())`}
        language="python"
        filename="social_tools.py"
      />

      <h3>Translation &amp; Language Detection (v0.2.5)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.translate import TranslateTool
from effgen.tools.builtin.language_detect import LanguageDetectTool

async def main():
    # LibreTranslate primary (configurable via LIBRE_TRANSLATE_URL);
    # argostranslate offline fallback (language packs cached in ~/.effgen/argos/).
    tr = TranslateTool()
    r = await tr.execute({"operation": "translate", "text": "Hello, world!", "source": "en", "target": "fr"})
    print(r["data"]["translated_text"])  # Bonjour le monde!

    # langdetect — fully offline, 55+ languages
    ld = LanguageDetectTool()
    r = await ld.execute({"operation": "detect", "text": "Bonjour le monde"})
    print(r["data"]["language"], r["data"]["confidence"])  # fr 0.99...

asyncio.run(main())`}
        language="python"
        filename="translate_lang_tools.py"
      />

      <h3>QR Codes — Fully Local (v0.2.5)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.qr_generate import QRGenerateTool
from effgen.tools.builtin.qr_read import QRReadTool

async def main():
    # Local QR generation — no network. Returns base64 PNG or saves to file.
    gen = QRGenerateTool()
    r = await gen.execute({
        "operation": "generate",
        "data": "https://effgen.org",
        "data_url_return": True,
    })
    print(r["data"].get("data_url", "")[:40] + "...")  # data:image/png;base64,...

    # Decode QR / barcodes from image path or base64 PNG (pyzbar + Pillow; OpenCV QR fallback).
    rd = QRReadTool()
    r = await rd.execute({"operation": "read", "image_path": "/tmp/qr.png"})
    for code in r["data"]["codes"]:
        print(code["data"])

asyncio.run(main())`}
        language="python"
        filename="qr_tools.py"
      />

      <h3>OCR (v0.2.6)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.ocr import OCRTool

async def main():
    ocr = OCRTool()  # Tesseract (local, primary) + OCR.space fallback (OCR_SPACE_API_KEY)
    r = await ocr.execute({"operation": "extract", "image_path": "/tmp/scan.png", "lang": "eng"})
    print(r["data"]["text"])

# System dep: apt-get install tesseract-ocr  /  brew install tesseract
# Raises OCRBackendUnavailable with per-OS install instructions if no backend is available.
asyncio.run(main())`}
        language="python"
        filename="ocr_tool.py"
      />

      <h3>Audio Transcription (v0.2.6)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

async def main():
    # Local: faster-whisper (auto CPU/GPU). Falls back to HuggingFace Inference if HF_TOKEN is set.
    tool = AudioTranscribeTool()
    r = await tool.execute({"operation": "transcribe", "audio_path": "/tmp/clip.mp3", "model_size": "base"})
    print(r["data"]["text"])

# System dep for non-WAV: apt-get install ffmpeg  /  brew install ffmpeg
asyncio.run(main())`}
        language="python"
        filename="audio_transcribe_tool.py"
      />

      <h3>Image Analysis (v0.2.6)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.image_info import ImageInfoTool
from effgen.tools.builtin.image_caption import ImageCaptionTool

async def main():
    # Zero-network metadata + resize via Pillow
    info = ImageInfoTool()
    r = await info.execute({"operation": "info", "image_path": "/tmp/photo.jpg"})
    print(r["data"]["size"], r["data"]["format"])

    # Natural-language caption via model router (Gemini / OpenAI / MLX-VLM).
    # Raises NoVisionProviderAvailable if no vision-capable provider is configured.
    cap = ImageCaptionTool()
    r = await cap.execute({"operation": "caption", "image_path": "/tmp/photo.jpg"})
    print(r["data"]["caption"])

asyncio.run(main())`}
        language="python"
        filename="image_tools.py"
      />

      <h3>Document Parsing — PDF / DOCX / Excel (v0.2.6)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.pdf import PDFTool
from effgen.tools.builtin.docx import DOCXTool
from effgen.tools.builtin.excel import ExcelTool

async def main():
    # PDF: pypdf (primary) + pdfplumber (tables)
    r = await PDFTool().execute({"operation": "text", "path": "/tmp/paper.pdf"})
    print(r["data"]["text"][:500])

    # DOCX: python-docx
    r = await DOCXTool().execute({"operation": "text", "path": "/tmp/report.docx"})
    print(r["data"]["text"])

    # Excel: openpyxl + pandas
    r = await ExcelTool().execute({"operation": "read_sheet", "path": "/tmp/data.xlsx", "sheet_name": "Sheet1"})
    print(r["data"]["rows"][:3])

asyncio.run(main())`}
        language="python"
        filename="document_tools.py"
      />

      <h3>Geo / Weather (v0.2.6)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.weather import WeatherTool
from effgen.tools.builtin.geocode import GeocodeTool
from effgen.tools.builtin.maps import MapsTool

async def main():
    # Open-Meteo — free, no auth
    r = await WeatherTool().execute({"operation": "current", "lat": 37.42, "lon": -122.08})
    print(r["data"]["temperature_c"], r["data"]["weather_description"])

    # Nominatim — 1 req/s token bucket; sets effGen/<version> User-Agent
    r = await GeocodeTool().execute({"operation": "geocode", "address": "1600 Amphitheatre Pkwy, Mountain View, CA"})
    print(r["data"]["lat"], r["data"]["lon"])

    # OSM static tiles via the staticmap library
    r = await MapsTool().execute({"operation": "render", "lat": 37.42, "lon": -122.08, "zoom": 13, "dest": "/tmp/map.png"})
    print(r["data"]["path"])

asyncio.run(main())`}
        language="python"
        filename="geo_tools.py"
      />

      <h3>Email (v0.2.6, live)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.email_smtp import EmailSMTPTool
from effgen.tools.builtin.email_imap import EmailIMAPTool

async def main():
    # SMTP send — TLS-on by default. Env: SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / SMTP_FROM
    r = await EmailSMTPTool().execute({
        "operation": "send",
        "to": "alice@example.com",
        "subject": "Hello",
        "body": "Hi there!",
    })
    print(r["success"])

    # IMAP read — Env: IMAP_HOST / IMAP_PORT / IMAP_USER / IMAP_PASSWORD
    r = await EmailIMAPTool().execute({"operation": "fetch_recent", "folder": "INBOX", "n": 5})
    for msg in r["data"]["messages"]:
        print(msg["subject"], msg["from"])

# Raises MissingCredentialsError when required env vars are absent.
asyncio.run(main())`}
        language="python"
        filename="email_tools.py"
      />

      <h3>Webhooks — Slack &amp; Discord (v0.2.6)</h3>
      <CodeBlock
        code={`import asyncio
from effgen.tools.builtin.slack_webhook import SlackWebhookTool
from effgen.tools.builtin.discord_webhook import DiscordWebhookTool

async def main():
    # Webhook URLs are redacted in all logs.
    await SlackWebhookTool().execute({"operation": "post", "text": "Deploy complete!"})
    await DiscordWebhookTool().execute({"operation": "post", "content": "Deployment succeeded!"})

# Env: SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL
asyncio.run(main())`}
        language="python"
        filename="webhook_tools.py"
      />

      <h2>Tool Categories</h2>

      <FeatureList
        features={[
          { icon: '🔍', title: 'INFORMATION_RETRIEVAL', description: 'Web, Wikipedia, RAG retrieval, AgenticSearch, Stack Overflow, GitHub' },
          { icon: '🧬', title: 'ACADEMIC (v0.2.5)', description: 'PubMed, ArXiv, Semantic Scholar' },
          { icon: '📰', title: 'NEWS & RSS (v0.2.5)', description: 'RSSFeed, News (curated reputable sources)' },
          { icon: '📺', title: 'YOUTUBE (v0.2.5)', description: 'Transcript (no API key), Metadata (yt-dlp)' },
          { icon: '💬', title: 'SOCIAL (v0.2.5)', description: 'Reddit (no OAuth), Hacker News' },
          { icon: '🌐', title: 'TRANSLATION (v0.2.5)', description: 'Translate (LibreTranslate + offline fallback), LanguageDetect (55+ langs)' },
          { icon: '🔲', title: 'QR CODES (v0.2.5)', description: 'QRGenerate, QRRead — fully local' },
          { icon: '📝', title: 'OCR (v0.2.6)', description: 'OCRTool — Tesseract local + OCR.space fallback' },
          { icon: '🎤', title: 'AUDIO (v0.2.6)', description: 'AudioTranscribeTool — faster-whisper + HF Inference fallback' },
          { icon: '🖼️', title: 'IMAGE (v0.2.6)', description: 'ImageInfoTool (Pillow), ImageCaptionTool (vision router)' },
          { icon: '📑', title: 'DOCUMENTS (v0.2.6)', description: 'PDFTool (pypdf + pdfplumber), DOCXTool (python-docx), ExcelTool (openpyxl + pandas)' },
          { icon: '🗺️', title: 'GEO / WEATHER (v0.2.6)', description: 'WeatherTool (Open-Meteo), GeocodeTool (Nominatim), MapsTool (OSM static)' },
          { icon: '📧', title: 'EMAIL (v0.2.6, live)', description: 'EmailSMTPTool (TLS-on), EmailIMAPTool (read/search inbox)' },
          { icon: '🔔', title: 'WEBHOOKS (v0.2.6)', description: 'SlackWebhookTool, DiscordWebhookTool (URLs redacted in logs)' },
          { icon: '💻', title: 'CODE_EXECUTION', description: 'Sandboxed Python, JavaScript, Bash' },
          { icon: '📊', title: 'DATA_PROCESSING', description: 'JSON, text, DataFrame, Plot' },
          { icon: '🌎', title: 'EXTERNAL_API', description: 'Weather, stocks, FX, crypto, HTTP, WolframAlpha' },
          { icon: '📁', title: 'FILE_OPERATIONS', description: 'File read/write/list/search/convert operations' },
          { icon: '🖥️', title: 'SYSTEM', description: 'Bash, Git, Docker, system metrics' },
          { icon: '✉️', title: 'COMMUNICATION (legacy)', description: 'Email and Slack drafts plus local notifications' },
          { icon: '🧮', title: 'COMPUTATION', description: 'Calculator, DateTime, Stats' },
        ]}
      />

      <h2>Best Practices</h2>

      <InfoBox type="info" title="Tool Design Guidelines">
        <ul>
          <li><strong>Clear descriptions:</strong> Write detailed descriptions so the model knows when to use the tool</li>
          <li><strong>Validate inputs:</strong> Use parameter specifications to validate inputs before execution</li>
          <li><strong>Handle errors:</strong> Return meaningful error messages for debugging</li>
          <li><strong>Security:</strong> Implement proper sandboxing for code execution tools</li>
          <li><strong>Async:</strong> Use async/await for I/O-bound operations</li>
        </ul>
      </InfoBox>

      <InfoBox type="success" title="Next Steps">
        <p>
          Learn about <Link to="/memory">Memory Systems</Link> for context persistence,
          or explore <Link to="/protocols">Protocols</Link> for MCP and A2A integration.
        </p>
      </InfoBox>
    </DocPage>
  );
}
