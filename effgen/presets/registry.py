"""
Preset registry — defines agent presets and the create_agent factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from effgen.core.agent import Agent, AgentConfig
from effgen.models import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class PresetConfig:
    """Definition of an agent preset."""

    name: str
    description: str
    tool_names: list[str]
    system_prompt: str
    max_iterations: int = 10
    temperature: float = 0.7
    enable_sub_agents: bool = False
    enable_memory: bool = True
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

_MATH_PRESET = PresetConfig(
    name="math",
    description="Mathematical reasoning agent with Calculator and PythonREPL.",
    tool_names=["calculator", "python_repl"],
    system_prompt=(
        "You are a precise mathematical reasoning agent. "
        "Use the calculator tool for arithmetic and the python_repl tool "
        "for complex computations. Always show your work and verify results."
    ),
    max_iterations=8,
    temperature=0.3,
    tags=["math", "computation"],
)

_RESEARCH_PRESET = PresetConfig(
    name="research",
    description=(
        "Research agent with WebSearch, URLFetch, Wikipedia, academic "
        "search (arXiv, PubMed, Semantic Scholar), RSS feeds, news, "
        "YouTube transcript/metadata, Reddit, Hacker News, and document "
        "parsing (PDF, DOCX, Excel) tools."
    ),
    tool_names=[
        "web_search",
        "url_fetch",
        "wikipedia",
        "arxiv",
        "pubmed",
        "semantic_scholar",
        "rss_feed",
        "news",
        "youtube_transcript",
        "youtube_metadata",
        "reddit",
        "hackernews",
        "pdf",
        "docx",
        "excel",
    ],
    system_prompt=(
        "You are a thorough research agent. Search the web, fetch URLs, and "
        "consult Wikipedia, arXiv, PubMed, Semantic Scholar, RSS feeds, news "
        "sources, YouTube videos, Reddit, and Hacker News to gather accurate information. "
        "Prefer academic sources for scientific or medical questions. For current events, "
        "use news, rss_feed, reddit, or hackernews tools. For video content, use "
        "youtube_transcript to read captions and youtube_metadata for video details. "
        "Cite your sources and synthesize findings into clear answers."
    ),
    max_iterations=10,
    temperature=0.5,
    tags=["research", "search", "information", "academic"],
)

_CODING_PRESET = PresetConfig(
    name="coding",
    description="Coding agent with CodeExecutor, PythonREPL, FileOperations, and BashTool.",
    tool_names=["code_executor", "python_repl", "file_operations", "bash"],
    system_prompt=(
        "You are an expert coding agent. Write, execute, and debug code to "
        "solve programming tasks. Use file operations to read/write files and "
        "bash for system commands. Always test your code before presenting results."
    ),
    max_iterations=12,
    temperature=0.4,
    tags=["coding", "programming", "development"],
)

_GENERAL_PRESET = PresetConfig(
    name="general",
    description=(
        "General-purpose agent with all available built-in tools, including QR, OCR, "
        "audio transcription, image analysis, document parsing (PDF, DOCX, Excel), "
        "weather/geo, email (SMTP/IMAP), Slack, and Discord webhooks."
    ),
    tool_names=[
        "calculator",
        "python_repl",
        "web_search",
        "code_executor",
        "file_operations",
        "bash",
        "json_tool",
        "datetime_tool",
        "text_processing",
        "url_fetch",
        "wikipedia",
        "rss_feed",
        "news",
        "reddit",
        "hackernews",
        "translate",
        "language_detect",
        "qr_generate",
        "qr_read",
        "ocr",
        "audio_transcribe",
        "image_info",
        "pdf",
        "docx",
        "excel",
        "weather",
        "geocode",
        "maps",
        "email_smtp",
        "email_imap",
        "slack_webhook",
        "discord_webhook",
    ],
    system_prompt=(
        "You are a versatile AI assistant with access to many tools. "
        "Choose the most appropriate tool for each task. "
        "Think step by step and use tools when they will help you "
        "give a more accurate or complete answer."
    ),
    max_iterations=10,
    temperature=0.7,
    tags=["general", "all-purpose"],
)

_RAG_PRESET = PresetConfig(
    name="rag",
    description="Retrieval-Augmented Generation agent with hybrid search over a knowledge base.",
    tool_names=["retrieval"],
    system_prompt=(
        "You are a retrieval-augmented assistant. When answering, ALWAYS "
        "consult the knowledge base first using the retrieval tool. Cite "
        "sources inline using [1], [2], ... markers matching the returned "
        "citation list. If the knowledge base does not contain the answer, "
        "say so explicitly rather than guessing."
    ),
    max_iterations=8,
    temperature=0.3,
    tags=["rag", "retrieval", "knowledge-base"],
)

_MINIMAL_PRESET = PresetConfig(
    name="minimal",
    description="Minimal agent with no tools — direct model inference only.",
    tool_names=[],
    system_prompt="You are a helpful AI assistant. Answer questions directly.",
    max_iterations=1,
    temperature=0.7,
    tags=["minimal", "no-tools"],
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRESETS: dict[str, PresetConfig] = {
    "math": _MATH_PRESET,
    "research": _RESEARCH_PRESET,
    "coding": _CODING_PRESET,
    "general": _GENERAL_PRESET,
    "rag": _RAG_PRESET,
    "minimal": _MINIMAL_PRESET,
    # media preset is registered by effgen/presets/media.py on import
}


def list_presets() -> dict[str, str]:
    """Return a mapping of preset name → description."""
    return {name: p.description for name, p in PRESETS.items()}


class UnknownPresetError(ValueError, KeyError):
    """Raised when an unknown preset name is requested.

    Subclasses both :class:`ValueError` and :class:`KeyError` so existing
    ``except KeyError`` / ``except ValueError`` handlers keep working, while the
    explicit ``__str__`` keeps the message clean (a bare ``KeyError`` would
    wrap it in repr-quotes and read like an internal bug).
    """

    def __str__(self) -> str:  # noqa: D105 - clean message, no KeyError quoting
        return self.args[0] if self.args else ""


def get_preset(name: str) -> PresetConfig:
    """Get a preset configuration by name.

    Raises:
        UnknownPresetError: If the preset name is not found. The error message
            lists the available presets and a fuzzy "did you mean" suggestion.
            It subclasses both ``ValueError`` and ``KeyError`` for compatibility.
    """
    if name not in PRESETS:
        import difflib

        available = ", ".join(sorted(PRESETS.keys()))
        close = difflib.get_close_matches(str(name), list(PRESETS.keys()), n=1, cutoff=0.6)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise UnknownPresetError(
            f"Unknown preset '{name}'.{hint} Available presets: {available}."
        )
    return PRESETS[name]


def _instantiate_tools(tool_names: list[str]) -> list:
    """Instantiate tool objects from their registry names.

    Tools that fail to load are skipped with a warning rather than
    raising — this keeps the agent usable even when optional tools
    (e.g. web_search without API keys) cannot be initialised.
    """
    from effgen.tools.builtin import (
        AgenticSearch,
        ArXivTool,
        AudioTranscribeTool,
        BashTool,
        Calculator,
        CodeExecutor,
        DateTimeTool,
        DiscordWebhookTool,
        DOCXTool,
        EmailIMAPTool,
        EmailSMTPTool,
        ExcelTool,
        FileOperations,
        GeocodeTool,
        HackerNewsTool,
        ImageCaptionTool,
        ImageInfoTool,
        JSONTool,
        LanguageDetectTool,
        MapsTool,
        MultimodalDescribeTool,
        NewsTool,
        OCRTool,
        PDFTool,
        PubMedTool,
        PythonREPL,
        QRGenerateTool,
        QRReadTool,
        RedditTool,
        Retrieval,
        RSSFeedTool,
        SemanticScholarTool,
        SlackWebhookTool,
        TextProcessingTool,
        TranslateTool,
        URLFetchTool,
        WeatherTool,
        WebSearch,
        WikipediaTool,
        YouTubeMetadataTool,
        YouTubeTranscriptTool,
    )

    _TOOL_MAP: dict[str, type] = {
        "calculator": Calculator,
        "python_repl": PythonREPL,
        "web_search": WebSearch,
        "code_executor": CodeExecutor,
        "file_operations": FileOperations,
        "bash": BashTool,
        "json_tool": JSONTool,
        "datetime_tool": DateTimeTool,
        "text_processing": TextProcessingTool,
        "url_fetch": URLFetchTool,
        "wikipedia": WikipediaTool,
        "agentic_search": AgenticSearch,
        "retrieval": Retrieval,
        "arxiv": ArXivTool,
        "pubmed": PubMedTool,
        "semantic_scholar": SemanticScholarTool,
        "rss_feed": RSSFeedTool,
        "news": NewsTool,
        "youtube_transcript": YouTubeTranscriptTool,
        "youtube_metadata": YouTubeMetadataTool,
        "reddit": RedditTool,
        "hackernews": HackerNewsTool,
        "translate": TranslateTool,
        "language_detect": LanguageDetectTool,
        "qr_generate": QRGenerateTool,
        "qr_read": QRReadTool,
        "ocr": OCRTool,
        "audio_transcribe": AudioTranscribeTool,
        "image_info": ImageInfoTool,
        "image_caption": ImageCaptionTool,
        "multimodal_describe": MultimodalDescribeTool,
        "pdf": PDFTool,
        "docx": DOCXTool,
        "excel": ExcelTool,
        "weather": WeatherTool,
        "geocode": GeocodeTool,
        "maps": MapsTool,
        "email_smtp": EmailSMTPTool,
        "email_imap": EmailIMAPTool,
        "slack_webhook": SlackWebhookTool,
        "discord_webhook": DiscordWebhookTool,
    }

    tools = []
    for name in tool_names:
        cls = _TOOL_MAP.get(name)
        if cls is None:
            logger.warning("Unknown tool '%s' in preset — skipping.", name)
            continue
        try:
            tools.append(cls())
        except Exception as exc:
            logger.warning("Failed to instantiate tool '%s': %s", name, exc)
    return tools


def _resolve_default_model(preset: str) -> str:
    """Resolve a model when the caller omitted one.

    Honours the ``EFFGEN_DEFAULT_MODEL`` environment variable so a user can set
    a zero-config default once; otherwise raises a clear, actionable error that
    points at the discovery commands instead of a cryptic ``TypeError``.
    """
    import os

    env_default = os.environ.get("EFFGEN_DEFAULT_MODEL", "").strip()
    if env_default:
        return env_default
    raise ValueError(
        f"create_agent('{preset}') needs a model — there is no built-in default "
        "(effGen never silently picks a paid cloud model). Pass one explicitly, "
        "e.g. create_agent('" + preset + "', 'gpt-5-nano') for a cheap cloud "
        "model or create_agent('" + preset + "', 'Qwen/Qwen2.5-1.5B-Instruct') "
        "for a local model. Run `effgen models list` to see options or "
        "`effgen doctor` to check which providers are usable. You can also set "
        "EFFGEN_DEFAULT_MODEL to choose a default once."
    )


def create_agent(
    preset: str,
    model: BaseModel | str | None = None,
    *,
    agent_name: str | None = None,
    extra_tools: list | None = None,
    knowledge_base: str | None = None,
    system_prompt: str | None = None,
    max_iterations: int | None = None,
    temperature: float | None = None,
    enable_memory: bool | None = None,
    **config_overrides: Any,
) -> Agent:
    """Create an agent from a named preset.

    Args:
        preset: Preset name. Available: {PRESET_LIST}.
            New to effGen? Start with ``math`` or ``minimal`` (small, fast);
            ``general`` is the "kitchen sink" with every tool. See
            ``list_presets()`` for descriptions.
        model: A loaded model instance or a model identifier string. If omitted,
            ``EFFGEN_DEFAULT_MODEL`` is used when set, otherwise a clear error
            tells you how to pick one (effGen never silently picks a paid model).
        agent_name: Optional override for the agent name.
        extra_tools: Additional tool instances to add beyond the preset.
        system_prompt: Override the preset's system prompt.
        max_iterations: Override max iterations.
        temperature: Override temperature.
        enable_memory: Override memory setting.
        **config_overrides: Extra keyword arguments forwarded to AgentConfig.

    Returns:
        A configured Agent ready to run.

    Example:
        >>> from effgen.presets import create_agent
        >>> # cheap cloud model:
        >>> agent = create_agent("math", "gpt-5-nano")
        >>> result = agent.run("What is 12 * 12?")
        >>> # or a local small model:
        >>> agent = create_agent("math", "Qwen/Qwen2.5-1.5B-Instruct")
    """
    if model is None:
        model = _resolve_default_model(preset)

    cfg = get_preset(preset)

    tools = _instantiate_tools(cfg.tool_names)

    # Special handling for RAG preset: ingest knowledge base on creation
    if preset == "rag" and knowledge_base:
        try:
            from effgen.rag import DocumentIngester, HybridSearchEngine  # noqa: F401
            from effgen.tools.builtin import Retrieval

            ingester = DocumentIngester(show_progress=False)
            chunks = ingester.ingest(knowledge_base)

            # Find the Retrieval tool in tools and populate it
            retrieval_tool = next(
                (t for t in tools if t.metadata.name == "retrieval"), None
            )
            if retrieval_tool is None:
                retrieval_tool = Retrieval()
                tools.append(retrieval_tool)

            docs = [
                {
                    "content": c.content,
                    "id": c.id,
                    "metadata": c.metadata,
                }
                for c in chunks
            ]
            if docs:
                retrieval_tool.add_documents(docs, chunk=False)
                logger.info(
                    "RAG preset: ingested %d chunks from %s",
                    len(docs),
                    knowledge_base,
                )
        except Exception as exc:
            logger.warning("RAG knowledge base ingestion failed: %s", exc)

    if extra_tools:
        tools.extend(extra_tools)

    agent_config = AgentConfig(
        name=agent_name or f"{cfg.name}-agent",
        model=model,
        tools=tools,
        system_prompt=system_prompt or cfg.system_prompt,
        max_iterations=max_iterations if max_iterations is not None else cfg.max_iterations,
        temperature=temperature if temperature is not None else cfg.temperature,
        enable_sub_agents=cfg.enable_sub_agents,
        enable_memory=enable_memory if enable_memory is not None else cfg.enable_memory,
        **config_overrides,
    )

    logger.info(
        "Created '%s' preset agent with %d tools: %s",
        cfg.name,
        len(tools),
        ", ".join(t.metadata.name for t in tools) if tools else "(none)",
    )

    return Agent(agent_config)


# Keep the template (with the ``{PRESET_LIST}`` placeholder) so the docstring can
# be regenerated from the registry whenever the set of presets changes — the
# bundled extra presets (media/multimodal/notify) register by side-effect on
# import, which can happen after this module finishes (see presets/__init__.py).
_CREATE_AGENT_DOC_TEMPLATE = create_agent.__doc__ or ""


def _refresh_create_agent_doc() -> None:
    """Regenerate ``create_agent``'s preset list from the live registry (U1-12).

    Idempotent and safe to call repeatedly; the preset list can never drift from
    the actual presets. Called once here and again after the bundled extra
    presets are imported.
    """
    if _CREATE_AGENT_DOC_TEMPLATE:
        create_agent.__doc__ = _CREATE_AGENT_DOC_TEMPLATE.replace(
            "{PRESET_LIST}", ", ".join(sorted(PRESETS.keys()))
        )


# Best-effort: register the bundled extra presets so a *direct* import of this
# module (bypassing effgen.presets) still sees all of them. When imported via
# effgen.presets, __init__ re-runs the refresh after its own side-effect imports.
for _extra in ("media", "multimodal", "notify"):
    try:  # pragma: no cover - defensive; the modules ship with the package
        __import__(f"effgen.presets.{_extra}")
    except Exception:  # noqa: BLE001
        pass

_refresh_create_agent_doc()
