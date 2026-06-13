# Tool Gallery

Quick-reference for every built-in tool in effGen. Each entry has a one-line description and a minimal working snippet.

> All tools accept a plain `dict` via `tool.execute({...})` or can be wired into an `Agent` for agentic use.

---

## Core Utilities

### Calculator
**Math expressions, unit conversions, and basic financial calculations.**

```python
from effgen.tools.builtin.calculator import Calculator
result = Calculator().execute({"operation": "evaluate", "expression": "2 ** 10 + sqrt(144)"})
print(result["data"])  # 1036.0
```

### PythonREPL
**Interactive Python execution with persistent state across calls.**

```python
from effgen.tools.builtin.python_repl import PythonREPL
repl = PythonREPL()
repl.execute({"code": "x = [i**2 for i in range(5)]"})
result = repl.execute({"code": "print(sum(x))"})
print(result["data"]["output"])  # 30
```

### CodeExecutor
**Sandboxed multi-language code execution (Python, JavaScript, Bash).**

```python
from effgen.tools.builtin.code_executor import CodeExecutor
result = CodeExecutor().execute({"language": "python", "code": "print('hello')"})
print(result["data"]["stdout"])
```

### BashTool
**Restricted shell command execution with allow/deny lists.**

```python
from effgen.tools.builtin.bash_tool import BashTool
result = BashTool().execute({"command": "ls -lh /tmp"})
print(result["data"]["stdout"])
```

### FileOps
**Read, write, list, and search files on the local filesystem.**

```python
from effgen.tools.builtin.file_ops import FileOperations
result = FileOperations().execute({"operation": "read", "path": "README.md"})
print(result["data"]["content"][:200])
```

---

## Web & Search

### WebSearch
**DuckDuckGo search with caching; no API key required.**

```python
from effgen.tools.builtin.web_search import WebSearch
result = WebSearch().execute({"query": "effGen AI framework", "max_results": 5})
for r in result["data"]["results"]:
    print(r["title"], r["url"])
```

### URLFetch
**Fetch and extract text content from any public URL.**

```python
from effgen.tools.builtin.url_fetch import URLFetchTool
result = URLFetchTool().execute({"url": "https://example.com"})
print(result["data"]["text"][:500])
```

### WikipediaTool
**Search Wikipedia and retrieve article summaries; free API, no key needed.**

```python
from effgen.tools.builtin.wikipedia_tool import WikipediaTool
result = WikipediaTool().execute({"operation": "search", "query": "transformer neural network"})
print(result["data"]["summary"][:300])
```

### HTTPTool
**Generic HTTP GET/POST requests with headers and JSON body support.**

```python
from effgen.tools.builtin.devops import HTTPTool
result = HTTPTool().execute({"method": "GET", "url": "https://httpbin.org/get"})
print(result["data"]["status_code"])
```

---

## Academic Research

### PubMedTool
**Search PubMed via NCBI E-utilities, fetch metadata, and retrieve abstracts. Built-in rate limiter (3 req/s; 10/s with `NCBI_API_KEY`).**

```python
from effgen.tools.builtin.pubmed import PubMedTool
result = PubMedTool().execute({"operation": "search", "query": "CRISPR gene editing", "max_results": 5})
for article in result["data"]["articles"]:
    print(article["pmid"], article["title"])
```

### ArXivTool
**Search arXiv papers, fetch metadata by ID, or download the PDF. Free, no auth.**

```python
from effgen.tools.builtin.arxiv import ArXivTool
result = ArXivTool().execute({"operation": "search", "query": "attention is all you need", "max_results": 3})
for paper in result["data"]["papers"]:
    print(paper["id"], paper["title"])
```

### SemanticScholarTool
**Search papers, get details, retrieve citations and references from Semantic Scholar Graph API. Built-in backoff (100 req/5 min unauth).**

```python
from effgen.tools.builtin.semantic_scholar import SemanticScholarTool
result = SemanticScholarTool().execute({"operation": "search", "query": "large language models survey"})
for paper in result["data"]["papers"][:3]:
    print(paper.get("paperId"), paper.get("title"))
```

---

## News & RSS

### RSSFeedTool
**Fetch, browse, and full-text search any RSS/Atom feed by URL. Handles malformed feeds gracefully.**

```python
from effgen.tools.builtin.rss import RSSFeedTool
result = RSSFeedTool().execute({"operation": "latest", "url": "https://hnrss.org/frontpage", "n": 5})
for item in result["data"]["items"]:
    print(item["title"])
```

### NewsTool
**Aggregate top headlines across curated reputable sources (BBC, Reuters, HN, NPR, Al Jazeera, etc.). Optional `NEWS_API_KEY` for NewsAPI.org.**

```python
from effgen.tools.builtin.news import NewsTool
result = NewsTool().execute({"operation": "top_headlines"})
for article in result["data"]["articles"][:5]:
    print(article["title"], "-", article["source"])
```

---

## YouTube

### YouTubeTranscriptTool
**Fetch YouTube video captions/transcripts without a Google API key. Supports watch?v=, youtu.be/, and shorts/ URL formats.**

```python
from effgen.tools.builtin.youtube_transcript import YouTubeTranscriptTool
result = YouTubeTranscriptTool().execute({
    "operation": "get_transcript",
    "video_id": "dQw4w9WgXcQ",
    "lang": "en"
})
print(result["data"]["transcript"][:500])
```

### YouTubeMetadataTool
**Fetch video or channel metadata using yt-dlp in metadata-only mode. No auth required for public content.**

```python
from effgen.tools.builtin.youtube_metadata import YouTubeMetadataTool
result = YouTubeMetadataTool().execute({
    "operation": "metadata",
    "video_id": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
})
print(result["data"].get("title"), result["data"].get("uploader"))
```

---

## Social Media

### RedditTool
**Access Reddit top/hot posts, user submissions, and thread comments via public JSON endpoints. No OAuth required for reads. Sets `effGen/<version>` User-Agent; exponential backoff on 429.**

```python
from effgen.tools.builtin.reddit import RedditTool
result = RedditTool().execute({"operation": "subreddit_top", "subreddit": "python", "time": "day", "n": 5})
for post in result["data"]["posts"]:
    print(post["title"])
```

### HackerNewsTool
**Fetch top/new stories, story details, and user profiles from the Hacker News Firebase API. No auth required.**

```python
from effgen.tools.builtin.hackernews import HackerNewsTool
result = HackerNewsTool().execute({"operation": "top_stories", "n": 5})
for story in result["data"]["stories"]:
    print(story["title"], story.get("url", ""))
```

---

## Translation & Language Detection

### TranslateTool
**Translate text between languages. Primary backend: LibreTranslate (configurable via `LIBRE_TRANSLATE_URL`). Offline fallback: `argostranslate` with language packs cached in `~/.effgen/argos/`.**

```python
from effgen.tools.builtin.translate import TranslateTool
result = TranslateTool().execute({"operation": "translate", "text": "Hello, world!", "source": "en", "target": "fr"})
print(result["data"]["translated_text"])  # Bonjour le monde!
```

### LanguageDetectTool
**Detect the language of text or a batch of texts. Fully offline via `langdetect` — supports 55+ languages.**

```python
from effgen.tools.builtin.language_detect import LanguageDetectTool
result = LanguageDetectTool().execute({"operation": "detect", "text": "Bonjour le monde"})
print(result["data"]["language"], result["data"]["confidence"])  # fr 0.99...
```

---

## QR Codes

### QRGenerateTool
**Generate QR codes locally from any text or URL. Returns a base64 PNG or saves to a file. No network required.**

```python
from effgen.tools.builtin.qr_generate import QRGenerateTool
result = QRGenerateTool().execute({"operation": "generate", "data": "https://effgen.org", "data_url_return": True})
# result["data"]["data_url"]  →  "data:image/png;base64,..."
print("QR generated:", result["data"].get("data_url", "")[:40] + "...")
```

### QRReadTool
**Decode QR codes and barcodes from an image file path or base64 PNG using `pyzbar` + Pillow, with OpenCV QR fallback when zbar is unavailable. Fully local.**

```python
from effgen.tools.builtin.qr_read import QRReadTool
result = QRReadTool().execute({"operation": "read", "image_path": "/tmp/qr.png"})
for code in result["data"]["codes"]:
    print(code["data"])
```

---

## Data Science

### DataFrameTool
**Load CSV/JSON, inspect with head/describe, filter rows, and aggregate with pandas.**

```python
from effgen.tools.builtin.data_analysis import DataFrameTool
result = DataFrameTool().execute({"operation": "load", "path": "data.csv"})
print(result["data"]["shape"])
```

### PlotTool
**Create line, bar, scatter, and histogram charts with matplotlib; returns a PNG file path.**

```python
from effgen.tools.builtin.data_analysis import PlotTool
result = PlotTool().execute({
    "operation": "line",
    "x": [1, 2, 3, 4],
    "y": [1, 4, 9, 16],
    "title": "Squares"
})
print(result["data"]["file"])
```

### StatsTool
**Compute mean, median, std, correlation, and linear regression with NumPy.**

```python
from effgen.tools.builtin.data_analysis import StatsTool
result = StatsTool().execute({"operation": "mean", "data": [1, 2, 3, 4, 5]})
print(result["data"]["mean"])  # 3.0
```

---

## Finance

### StockPriceTool
**Fetch real-time stock prices and historical data via yfinance. Not financial advice.**

```python
from effgen.tools.builtin.finance import StockPriceTool
result = StockPriceTool().execute({"operation": "price", "ticker": "AAPL"})
print(result["data"]["price"])
```

### CurrencyConverterTool
**Convert between 170+ currencies using frankfurter.app (ECB rates). No API key needed.**

```python
from effgen.tools.builtin.finance import CurrencyConverterTool
result = CurrencyConverterTool().execute({"amount": 100, "from_currency": "USD", "to_currency": "EUR"})
print(result["data"]["converted"])
```

### CryptoTool
**Fetch cryptocurrency prices and market data from CoinGecko. No API key for basic use.**

```python
from effgen.tools.builtin.finance import CryptoTool
result = CryptoTool().execute({"operation": "price", "coin_id": "bitcoin", "vs_currency": "usd"})
print(result["data"]["price"])
```

---

## DevOps

### GitTool
**Read-only Git operations: status, log, diff, branch list, show.**

```python
from effgen.tools.builtin.devops import GitTool
result = GitTool().execute({"operation": "log", "repo_path": ".", "max_count": 5})
for commit in result["data"]["commits"]:
    print(commit["hash"][:8], commit["message"])
```

### DockerTool
**Read-only Docker introspection: list containers, images, and fetch logs.**

```python
from effgen.tools.builtin.devops import DockerTool
result = DockerTool().execute({"operation": "ps"})
for c in result["data"]["containers"]:
    print(c["name"], c["status"])
```

### SystemInfoTool
**CPU, memory, disk, and network usage via psutil.**

```python
from effgen.tools.builtin.devops import SystemInfoTool
result = SystemInfoTool().execute({"operation": "cpu"})
print(result["data"]["cpu_percent"])
```

---

## Utilities

### JSONTool
**Parse, query (JSONPath), transform, and validate JSON data.**

```python
from effgen.tools.builtin.json_tool import JSONTool
result = JSONTool().execute({"operation": "query", "json_str": '{"a": [1,2,3]}', "path": "$.a[*]"})
print(result["data"]["result"])  # [1, 2, 3]
```

### DateTimeTool
**Current time, timezone conversion, and date arithmetic.**

```python
from effgen.tools.builtin.datetime_tool import DateTimeTool
result = DateTimeTool().execute({"operation": "now", "timezone": "US/Eastern"})
print(result["data"]["datetime"])
```

### TextProcessingTool
**Word count, regex find/replace, text comparison, and basic NLP operations.**

```python
from effgen.tools.builtin.text_processing import TextProcessingTool
result = TextProcessingTool().execute({"operation": "word_count", "text": "Hello world!"})
print(result["data"]["word_count"])  # 2
```

### WeatherTool
**Current weather and forecasts from Open-Meteo (free, no API key).**

```python
from effgen.tools.builtin.weather import WeatherTool
result = WeatherTool().execute({"operation": "current", "location": "San Francisco"})
print(result["data"]["temperature"], result["data"]["conditions"])
```

---

## Knowledge

### StackOverflowTool
**Search Stack Overflow questions and answers via the Stack Exchange API.**

```python
from effgen.tools.builtin.knowledge import StackOverflowTool
result = StackOverflowTool().execute({"operation": "search", "query": "python async await", "limit": 3})
for q in result["data"]["questions"]:
    print(q["title"])
```

### GitHubTool
**Search GitHub repositories, issues, and code via the public API.**

```python
from effgen.tools.builtin.knowledge import GitHubTool
result = GitHubTool().execute({"operation": "search_repos", "query": "effGen", "limit": 3})
for repo in result["data"]["repositories"]:
    print(repo["full_name"], repo["stargazers_count"])
```

### WolframAlphaTool
**Query Wolfram Alpha for computation and factual answers. Requires `WOLFRAM_API_KEY`.**

```python
from effgen.tools.builtin.knowledge import WolframAlphaTool
result = WolframAlphaTool().execute({"query": "integrate x^2 from 0 to 1"})
print(result["data"]["result"])
```

---

## Communication (Draft Only)

### EmailDraftTool
**Compose email drafts. Does NOT send — returns the draft for review.**

```python
from effgen.tools.builtin.communication import EmailDraftTool
result = EmailDraftTool().execute({
    "to": ["alice@example.com"],
    "subject": "Meeting tomorrow",
    "body": "Hi Alice, can we meet at 10am?"
})
print(result["data"]["draft"])
```

### SlackDraftTool
**Compose Slack message drafts. Does NOT send — returns the draft for review.**

```python
from effgen.tools.builtin.communication import SlackDraftTool
result = SlackDraftTool().execute({"channel": "#general", "text": "Deployment complete!"})
print(result["data"]["draft"])
```

---

## Provider-Native Tools

### OpenAI Native Tools
**Activate OpenAI-hosted capabilities within an Agent: web_search, code_interpreter, file_search. Requires an OpenAI model.**

```python
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool
from effgen import load_model, Agent
from effgen.core.agent import AgentConfig

model = load_model("gpt-4o-mini", provider="openai")
agent = Agent(config=AgentConfig(name="a", model=model, tools=[OpenAIWebSearchTool()]))
```

See [openai_native.md](openai_native.md) for full docs.

### Gemini Native Tools
**Activate Gemini server-side capabilities: GoogleSearchTool, GeminiUrlContextTool, GeminiCodeExecutionTool. Requires a Gemini model.**

```python
from effgen.tools.builtin.gemini_native import GoogleSearchTool
from effgen import load_model, Agent
from effgen.core.agent import AgentConfig

model = load_model("gemini-3.1-flash-lite", provider="gemini")
agent = Agent(config=AgentConfig(name="a", model=model, tools=[GoogleSearchTool()]))
```

See [gemini_native.md](gemini_native.md) for full docs.

### Anthropic Native Tools
**Experimental computer-use tools (bash, text_editor, computer). Requires an Anthropic model.**

See [anthropic_native.md](anthropic_native.md) for full docs.

---

## OCR

### OCRTool
**Extract text from images using Tesseract (local) with OCR.space free API fallback.**

```python
from effgen.tools.builtin.ocr import OCRTool
result = OCRTool().execute({"operation": "extract", "image_path": "/tmp/scan.png", "lang": "eng"})
print(result["data"]["text"])
```

System dep: `sudo apt-get install tesseract-ocr` / `brew install tesseract` / `choco install tesseract`.
See [ocr.md](ocr.md) for full docs.

---

## Audio Transcription

### AudioTranscribeTool
**Transcribe audio files locally via faster-whisper (CPU/GPU auto-detected); HuggingFace Inference fallback with HF_TOKEN.**

```python
from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool
result = AudioTranscribeTool().execute({"operation": "transcribe", "audio_path": "/tmp/clip.mp3", "model_size": "base"})
print(result["data"]["text"])
```

System dep (non-WAV formats): `sudo apt-get install ffmpeg` / `brew install ffmpeg`.
See [audio_transcribe.md](audio_transcribe.md) for full docs.

---

## Image Analysis

### ImageInfoTool
**Extract image metadata (size, format, mode, EXIF, histogram) and perform local resize/thumbnail operations. Zero network.**

```python
from effgen.tools.builtin.image_info import ImageInfoTool
result = ImageInfoTool().execute({"operation": "info", "image_path": "/tmp/photo.jpg"})
print(result["data"]["size"], result["data"]["format"], result["data"]["mode"])
```

### ImageCaptionTool
**Generate natural-language image descriptions via the effGen vision model router (Gemini / OpenAI / MLX-VLM).**

```python
from effgen.tools.builtin.image_caption import ImageCaptionTool
result = ImageCaptionTool().execute({"operation": "caption", "image_path": "/tmp/photo.jpg"})
print(result["data"]["caption"])
```

See [image.md](image.md) for full docs.

---

## Document Parsing

### PDFTool
**Extract text, tables, and metadata from PDF files using pypdf (primary) + pdfplumber (table fallback).**

```python
from effgen.tools.builtin.pdf import PDFTool
result = PDFTool().execute({"operation": "text", "path": "/tmp/paper.pdf"})
print(result["data"]["text"][:500])
# Also: metadata, tables, extract_images
```

### DOCXTool
**Parse Word documents (.docx) — text, paragraphs, tables, and metadata via python-docx.**

```python
from effgen.tools.builtin.docx import DOCXTool
result = DOCXTool().execute({"operation": "text", "path": "/tmp/report.docx"})
print(result["data"]["text"])
# Also: paragraphs, tables, metadata
```

### ExcelTool
**Read Excel workbooks (.xlsx) — sheets, headers, and row data via openpyxl + pandas.**

```python
from effgen.tools.builtin.excel import ExcelTool
# List sheets
sheets = ExcelTool().execute({"operation": "sheets", "path": "/tmp/data.xlsx"})
# Read a sheet
result = ExcelTool().execute({"operation": "read_sheet", "path": "/tmp/data.xlsx", "sheet_name": "Sheet1"})
print(result["data"]["rows"][:3])
```

See [documents.md](documents.md) for full docs.

---

## Geo / Weather

### WeatherTool
**Fetch current conditions, 7-day forecasts, or historical weather from Open-Meteo (free, no auth).**

```python
from effgen.tools.builtin.weather import WeatherTool
result = WeatherTool().execute({"operation": "current", "lat": 37.42, "lon": -122.08})
print(result["data"]["temperature_c"], result["data"]["weather_description"])
# Also: forecast (days=7), historical (start_date, end_date)
```

### GeocodeTool
**Forward/reverse geocoding via Nominatim (OpenStreetMap). Honors 1 req/s rate limit; sets effGen/<version> User-Agent.**

```python
from effgen.tools.builtin.geocode import GeocodeTool
result = GeocodeTool().execute({"operation": "geocode", "address": "1600 Amphitheatre Pkwy, Mountain View, CA"})
print(result["data"]["lat"], result["data"]["lon"])
# Also: reverse (lat, lon) → address
```

### MapsTool
**Render static PNG maps from OpenStreetMap tiles using the staticmap library.**

```python
from effgen.tools.builtin.maps import MapsTool
result = MapsTool().execute({
    "operation": "render",
    "lat": 37.42, "lon": -122.08,
    "zoom": 13,
    "dest": "/tmp/map.png"
})
print(result["data"]["path"])
# Also: bounding_box (south, west, north, east)
```

See [weather.md](weather.md), [geocode.md](geocode.md), [maps.md](maps.md) for full docs.

---

## Email

### EmailSMTPTool
**Send email via SMTP (stdlib smtplib, TLS on by default). Config: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM.**

```python
from effgen.tools.builtin.email_smtp import EmailSMTPTool
result = EmailSMTPTool().execute({
    "operation": "send",
    "to": "alice@example.com",
    "subject": "Hello from effGen",
    "body": "This message was sent by an AI agent."
})
print(result["data"]["status"])
```

### EmailIMAPTool
**Read email via IMAP (stdlib imaplib). Config: IMAP_HOST, IMAP_PORT, IMAP_USER, IMAP_PASSWORD.**

```python
from effgen.tools.builtin.email_imap import EmailIMAPTool
result = EmailIMAPTool().execute({"operation": "fetch_recent", "folder": "INBOX", "n": 5})
for msg in result["data"]["messages"]:
    print(msg["subject"], msg["from"])
# Also: list_folders, search, get
```

See [email.md](email.md) for full docs.

---

## Webhooks

### SlackWebhookTool
**Post messages to Slack via incoming webhook URL (no OAuth). Config: SLACK_WEBHOOK_URL. URL is redacted in logs.**

```python
from effgen.tools.builtin.slack_webhook import SlackWebhookTool
result = SlackWebhookTool().execute({"operation": "post", "text": "Deployment complete!"})
print(result["data"]["status"])
```

### DiscordWebhookTool
**Post messages to Discord via webhook URL. Config: DISCORD_WEBHOOK_URL. URL is redacted in logs.**

```python
from effgen.tools.builtin.discord_webhook import DiscordWebhookTool
result = DiscordWebhookTool().execute({
    "operation": "post",
    "content": "Build passed!",
    "username": "effGen Bot"
})
print(result["data"]["status"])
```

See [webhooks.md](webhooks.md) for full docs.

---

## Using Tools in an Agent

Any tool above can be wired into an Agent for agentic use:

```python
from effgen import load_model, Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin.arxiv import ArXivTool
from effgen.tools.builtin.translate import TranslateTool
from effgen.tools.builtin.hackernews import HackerNewsTool

model = load_model("gpt-oss-120b", provider="cerebras")
agent = Agent(config=AgentConfig(
    name="researcher",
    model=model,
    tools=[ArXivTool(), TranslateTool(), HackerNewsTool()],
    system_prompt="You are a research assistant."
))
result = agent.run("Find recent HN posts about AI agents and summarize in French.")
print(result.output)
```

## Using Presets

```python
from effgen import load_model
from effgen.presets import create_agent

model = load_model("gpt-oss-120b", provider="cerebras")

research_agent = create_agent("research", model)  # ArXiv, PubMed, SemanticScholar, RSS, News, YouTube, Reddit, HN, Wikipedia, WebSearch, PDF, DOCX, Excel
general_agent  = create_agent("general", model)   # All of the above + OCR, ImageInfo, Weather, Geocode, Maps, Email, Webhooks, Translate, QR, ...
media_agent    = create_agent("media", model)     # AudioTranscribeTool + ImageCaptionTool
notify_agent   = create_agent("notify", model)    # EmailSMTPTool + EmailIMAPTool + SlackWebhookTool + DiscordWebhookTool
```
