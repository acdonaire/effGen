"""Snippet tests for ``docs/tools/gallery.md``.

Every Python block in the gallery is checked for form (compiles; ``execute()``
called with keyword arguments only, never a positional dict), and the runnable
ones are executed verbatim in a subprocess:

* offline snippets run everywhere (file paths under ``/tmp/`` are rewritten to
  a per-test temp directory and any input fixture is created there first);
* snippets that reach a public network service are marked ``live``;
* snippets that need credentials, a webhook URL, or a system binary are
  skipped unless that requirement is present.

Each gallery ``###`` heading must be categorized below — an uncategorized
snippet fails ``test_every_gallery_snippet_is_categorized`` so new entries
cannot silently go untested.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent.parent
_GALLERY = _REPO_ROOT / "docs/tools/gallery.md"
_AUDIO_FIXTURE = _REPO_ROOT / "tests/fixtures/multimodal/sample_audio.mp3"

load_dotenv(_REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# Gallery parsing
# ---------------------------------------------------------------------------

def _gallery_snippets() -> list[tuple[str, str]]:
    """Return ``(heading, code)`` for every ```python block, in file order.

    The heading is the nearest preceding ``###`` title; headings repeat when a
    tool has several blocks (or appears in two sections), so consumers index
    by position within a heading where it matters.
    """
    text = _GALLERY.read_text(encoding="utf-8")
    snippets: list[tuple[str, str]] = []
    heading = ""
    in_block = False
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("### ") and not in_block:
            heading = line[4:].strip()
        elif line.startswith("## ") and not in_block:
            heading = line[3:].strip()
        elif line.strip() == "```python" and not in_block:
            in_block = True
            lines = []
        elif line.strip() == "```" and in_block:
            in_block = False
            snippets.append((heading, "\n".join(lines)))
        elif in_block:
            lines.append(line)
    return snippets


def _blocks_for(heading: str) -> list[str]:
    return [code for h, code in _gallery_snippets() if h == heading]


# ---------------------------------------------------------------------------
# Categorization: every heading with at least one snippet appears exactly once
# ---------------------------------------------------------------------------

# Runnable offline with no credentials; fixture files are created on the fly.
# (QRGenerateTool/QRReadTool are also offline but form a two-step sequence —
# generate writes the PNG that read decodes — so they have a dedicated test.)
OFFLINE = [
    "Calculator",
    "PythonREPL",
    "CodeExecutor",
    "BashTool",
    "FileOps",
    "DataFrameTool",
    "PlotTool",
    "StatsTool",
    "GitTool",
    "SystemInfoTool",
    "JSONTool",
    "DateTimeTool",
    "TextProcessingTool",
    "LanguageDetectTool",
    "EmailDraftTool",
    "SlackDraftTool",
    "ImageInfoTool",
    "PDFTool",
    "DOCXTool",
    "ExcelTool",
]

# Reach a public, unauthenticated network service.
NETWORK = [
    "WebSearch",
    "URLFetch",
    "WikipediaTool",
    "HTTPTool",
    "PubMedTool",
    "ArXivTool",
    "SemanticScholarTool",
    "RSSFeedTool",
    "NewsTool",
    "RedditTool",
    "HackerNewsTool",
    "TranslateTool",
    "StockPriceTool",
    "CurrencyConverterTool",
    "CryptoTool",
    "StackOverflowTool",
    "GitHubTool",
    "WeatherTool",        # appears twice (Utilities + Geo/Weather); both run
    "GeocodeTool",
    "MapsTool",
    "YouTubeTranscriptTool",
    "YouTubeMetadataTool",
]

# Need a credential, webhook URL, system binary, or model API key; each has a
# dedicated test below with its own skip condition.
GATED = [
    "DockerTool",
    "WolframAlphaTool",
    "EmailSMTPTool",
    "EmailIMAPTool",
    "SlackWebhookTool",
    "DiscordWebhookTool",
    "OCRTool",
    "AudioTranscribeTool",
    "ImageCaptionTool",
    "OpenAI Native Tools",
    "Gemini Native Tools",
    "Using Tools in an Agent",
    "Using Presets",
]

# Offline two-step sequences with their own dedicated test.
SEQUENCES = ["QRGenerateTool", "QRReadTool"]

# Optional third-party imports per heading; missing ones skip the test.
_OPTIONAL_DEPS = {
    "QRGenerateTool": ["qrcode"],
    "QRReadTool": ["pyzbar"],
    "DataFrameTool": ["pandas"],
    "PlotTool": ["matplotlib"],
    "LanguageDetectTool": ["langdetect"],
    "ImageInfoTool": ["PIL"],
    "PDFTool": ["pypdf", "matplotlib"],
    "DOCXTool": ["docx"],
    "ExcelTool": ["openpyxl"],
    "MapsTool": ["staticmap"],
    "YouTubeTranscriptTool": ["youtube_transcript_api"],
    "YouTubeMetadataTool": ["yt_dlp"],
    "StockPriceTool": ["yfinance"],
    "RSSFeedTool": ["feedparser"],
}


def test_every_gallery_snippet_is_categorized():
    headings = {h for h, _ in _gallery_snippets()}
    categorized = set(OFFLINE) | set(NETWORK) | set(GATED) | set(SEQUENCES)
    missing = headings - categorized
    assert not missing, f"Uncategorized gallery snippets: {sorted(missing)}"
    stale = categorized - headings
    assert not stale, f"Categorized but absent from the gallery: {sorted(stale)}"


# ---------------------------------------------------------------------------
# Form: every block compiles and calls execute() with keywords only
# ---------------------------------------------------------------------------

def test_gallery_snippets_compile_and_use_keyword_execute():
    problems: list[str] = []
    for heading, code in _gallery_snippets():
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            problems.append(f"{heading}: does not parse: {exc}")
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
            ):
                if node.args:
                    problems.append(
                        f"{heading}: execute() called with positional argument(s); "
                        "the tool API takes keyword arguments"
                    )
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "result"
            ):
                problems.append(
                    f"{heading}: subscripts `result[...]`; ToolResult exposes "
                    ".success/.output/.error attributes"
                )
    assert not problems, "\n".join(problems)


def test_gallery_snippet_imports_resolve():
    """Every module imported by a snippet is importable (no stale paths)."""
    import importlib

    modules: set[str] = set()
    for _, code in _gallery_snippets():
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("effgen"):
                    modules.add(node.module)
    assert modules, "expected effgen imports in the gallery"
    for mod in sorted(modules):
        importlib.import_module(mod)


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _run_snippet(code: str, tmp_path: Path, cwd: Path | None = None) -> None:
    """Execute a snippet verbatim in a subprocess; /tmp/ paths land in tmp_path."""
    rewritten = code.replace("/tmp/", f"{tmp_path}/")
    script = tmp_path / "snippet.py"
    script.write_text(rewritten, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=cwd or _REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=110,
        check=False,
    )
    assert proc.returncode == 0, (
        f"snippet failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


def _require_deps(heading: str) -> None:
    for dep in _OPTIONAL_DEPS.get(heading, []):
        pytest.importorskip(dep)


def _make_fixtures(heading: str, tmp_path: Path) -> None:
    """Create the input file a snippet reads, at the path it will see."""
    if heading == "DataFrameTool":
        (tmp_path / "data.csv").write_text("name,score\na,1\nb,2\nc,3\n")
    elif heading == "ImageInfoTool" or heading == "ImageCaptionTool":
        from PIL import Image

        Image.new("RGB", (400, 300), (70, 130, 180)).save(tmp_path / "photo.jpg")
    elif heading == "PDFTool":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(tmp_path / "paper.pdf") as pdf:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.1, 0.8, "A short sample document.")
            pdf.savefig(fig)
            plt.close(fig)
    elif heading == "DOCXTool":
        import docx

        doc = docx.Document()
        doc.add_heading("Quarterly Report", 0)
        doc.add_paragraph("Revenue grew 12% quarter over quarter.")
        doc.save(str(tmp_path / "report.docx"))
    elif heading == "ExcelTool":
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["name", "score"])
        ws.append(["alpha", 91])
        wb.save(str(tmp_path / "data.xlsx"))
    elif heading == "OCRTool":
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (600, 120), (255, 255, 255))
        ImageDraw.Draw(img).text((20, 40), "Invoice 2026-001", fill=(0, 0, 0))
        img.save(tmp_path / "scan.png")
    elif heading == "AudioTranscribeTool":
        shutil.copy(_AUDIO_FIXTURE, tmp_path / "clip.mp3")


# ---------------------------------------------------------------------------
# Offline snippets
# ---------------------------------------------------------------------------

class TestOfflineSnippets:
    @pytest.mark.cookbook
    @pytest.mark.parametrize("heading", OFFLINE)
    def test_snippet_runs(self, heading, tmp_path):
        _require_deps(heading)
        _make_fixtures(heading, tmp_path)
        blocks = _blocks_for(heading)
        assert blocks, f"no snippet found under heading {heading!r}"
        # Only the first block per heading is a standalone program; later
        # blocks (e.g. the PythonREPL max_sessions line) are fragments.
        _run_snippet(blocks[0], tmp_path)

    @pytest.mark.cookbook
    def test_qr_generate_then_read(self, tmp_path):
        """The QR snippets form a sequence: generate writes, read decodes."""
        _require_deps("QRGenerateTool")
        _require_deps("QRReadTool")
        gen, read = _blocks_for("QRGenerateTool")[0], _blocks_for("QRReadTool")[0]
        _run_snippet(gen + "\n" + read, tmp_path)


# ---------------------------------------------------------------------------
# Network snippets (public services, no credentials)
# ---------------------------------------------------------------------------

class TestNetworkSnippets:
    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.parametrize("heading", sorted(set(NETWORK)))
    def test_snippet_runs(self, heading, tmp_path):
        _require_deps(heading)
        for code in _blocks_for(heading):
            _run_snippet(code, tmp_path)


# ---------------------------------------------------------------------------
# Gated snippets: credentials / binaries / model keys
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    probe = subprocess.run(
        ["docker", "ps", "--format", "{{.ID}}"], capture_output=True, timeout=15
    )
    return probe.returncode == 0


class TestGatedSnippets:
    @pytest.mark.cookbook
    @pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable")
    def test_docker(self, tmp_path):
        _run_snippet(_blocks_for("DockerTool")[0], tmp_path)

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not os.getenv("WOLFRAM_ALPHA_APPID"), reason="WOLFRAM_ALPHA_APPID not set")
    def test_wolfram_alpha(self, tmp_path):
        _run_snippet(_blocks_for("WolframAlphaTool")[0], tmp_path)

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not os.getenv("SMTP_HOST"), reason="SMTP not configured")
    def test_email_smtp(self, tmp_path):
        _run_snippet(_blocks_for("EmailSMTPTool")[0], tmp_path)

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not os.getenv("IMAP_HOST"), reason="IMAP not configured")
    def test_email_imap(self, tmp_path):
        _run_snippet(_blocks_for("EmailIMAPTool")[0], tmp_path)

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not os.getenv("SLACK_WEBHOOK_URL"), reason="SLACK_WEBHOOK_URL not set")
    def test_slack_webhook(self, tmp_path):
        _run_snippet(_blocks_for("SlackWebhookTool")[0], tmp_path)

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not os.getenv("DISCORD_WEBHOOK_URL"), reason="DISCORD_WEBHOOK_URL not set")
    def test_discord_webhook(self, tmp_path):
        _run_snippet(_blocks_for("DiscordWebhookTool")[0], tmp_path)

    @pytest.mark.cookbook
    @pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract not installed")
    def test_ocr(self, tmp_path):
        _make_fixtures("OCRTool", tmp_path)
        _run_snippet(_blocks_for("OCRTool")[0], tmp_path)

    @pytest.mark.cookbook
    def test_audio_transcribe(self, tmp_path):
        # faster-whisper decodes MP3 itself (bundled PyAV); ffmpeg not needed.
        pytest.importorskip("faster_whisper")
        if not _AUDIO_FIXTURE.exists():
            pytest.skip("sample_audio.mp3 fixture not found")
        _make_fixtures("AudioTranscribeTool", tmp_path)
        _run_snippet(_blocks_for("AudioTranscribeTool")[0], tmp_path)

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(
        not (os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        reason="No vision API key",
    )
    def test_image_caption(self, tmp_path):
        _make_fixtures("ImageCaptionTool", tmp_path)
        _run_snippet(_blocks_for("ImageCaptionTool")[0], tmp_path)

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_openai_native_agent_constructs(self, tmp_path):
        _run_snippet(_blocks_for("OpenAI Native Tools")[0], tmp_path)

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
    def test_gemini_native_agent_constructs(self, tmp_path):
        _run_snippet(_blocks_for("Gemini Native Tools")[0], tmp_path)

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not os.getenv("CEREBRAS_API_KEY"), reason="CEREBRAS_API_KEY not set")
    def test_agent_with_tools(self, tmp_path):
        _run_snippet(_blocks_for("Using Tools in an Agent")[0], tmp_path)

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not os.getenv("CEREBRAS_API_KEY"), reason="CEREBRAS_API_KEY not set")
    def test_presets_construct(self, tmp_path):
        _run_snippet(_blocks_for("Using Presets")[0], tmp_path)
