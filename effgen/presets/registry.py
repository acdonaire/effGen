"""
Preset registry — defines agent presets and the create_agent factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
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
        "Synthesize findings into clear answers. When you cite a source URL, use "
        "ONLY a URL that one of your tools actually returned in this conversation — "
        "copy it verbatim. Never invent, guess, or reconstruct a URL from memory; "
        "if you have no tool-returned URL for a claim, say so rather than fabricate one."
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


def _normalize_extra_tools(extra_tools: list, *, _arg: str = "extra_tools") -> list:
    """Resolve ``extra_tools`` entries to tool instances.

    Each entry may be a ready tool instance **or** a registry tool *name* string
    (the same names the CLI's ``-t/--tools`` accepts, e.g. ``"calculator"``).
    A name is resolved through the global registry; an unknown name raises a
    ``ValueError`` with close-match suggestions instead of crashing later with the
    opaque ``'str' object has no attribute 'metadata'``.
    """
    if not extra_tools:
        return list(extra_tools or [])
    from effgen.tools import get_registry

    registry = None
    resolved = []
    for entry in extra_tools:
        if isinstance(entry, str):
            if registry is None:
                registry = get_registry()
            try:
                resolved.append(registry.get_tool_sync(entry))
            except KeyError:
                import difflib

                try:
                    available = registry.list_tools()
                except Exception:  # noqa: BLE001
                    available = []
                close = difflib.get_close_matches(entry, available, n=3, cutoff=0.5)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                raise ValueError(
                    f"{_arg}: unknown tool name {entry!r}.{hint} "
                    "Pass a registered tool name (see get_registry().list_tools() "
                    "or `effgen tools list`) or a Tool instance."
                ) from None
        else:
            resolved.append(entry)
    return resolved


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


def _looks_like_path(s: str) -> bool:
    """Heuristic: does ``s`` look like a file/dir path the user expected to
    exist (rather than an inline text document)?

    Used to fail loudly on a *typo'd* path instead of silently indexing the
    literal string as a one-line document. Prose has whitespace, so anything
    with internal whitespace is treated as inline text. A whitespace-free
    string is path-like if it has a path separator, a home/relative prefix, or
    a file-ish extension. A bare single word (e.g. ``"Canberra"``) is NOT
    path-like and stays inline.
    """
    s = s.strip()
    if not s or any(ch.isspace() for ch in s):
        return False
    if "/" in s or "\\" in s:
        return True
    if s.startswith(("~", ".")):
        return True
    suffix = Path(s).suffix.lower()
    return bool(suffix) and len(suffix) <= 6 and suffix[1:].isalnum()


def _ingest_rag_knowledge_base(
    tools: list, knowledge_base: str | list[str]
) -> None:
    """Populate the RAG preset's Retrieval tool from ``knowledge_base``.

    Each entry is a file path, a directory path, **or** a raw text document:
    an entry that exists on disk is loaded as a file/dir; any other string is
    treated as an inline document and chunked directly. An entry that *looks
    like* a path (a separator/extension, no whitespace) but does not exist is
    rejected loudly as a likely typo rather than indexed as a literal string.
    Raises ``ValueError`` if nothing ingestable was found (so a misuse fails
    loudly instead of silently producing an empty index).
    """
    try:
        from effgen.rag import DocumentIngester
        from effgen.tools.builtin import Retrieval
    except ImportError as exc:  # pragma: no cover - exercised via clear error
        raise ImportError(
            "The 'rag' preset needs RAG dependencies. Install with "
            "`pip install effgen[rag]`."
        ) from exc

    if isinstance(knowledge_base, str | Path):
        sources: list = [knowledge_base]
    else:
        sources = list(knowledge_base)

    ingester = DocumentIngester(show_progress=False)
    chunks = []
    n_files = 0
    n_inline = 0
    skipped: list[tuple[str, str]] = []
    for src in sources:
        s = str(src)
        # A blank entry has no content; skip it (and never let "" become
        # Path(".") and silently ingest the whole working directory).
        if not s.strip():
            continue
        path = Path(s).expanduser()
        if path.exists():
            chunks.extend(ingester.ingest(path))
            skipped.extend(ingester.last_skipped)
            n_files += 1
        elif _looks_like_path(s):
            # Looks like a path but isn't on disk -> almost certainly a typo.
            # Fail loudly instead of indexing the literal path string.
            raise ValueError(
                f"knowledge_base entry {s!r} looks like a file/directory path "
                "but does not exist. Fix the path, or pass raw text (text has "
                "spaces and no file extension) to index it inline."
            )
        else:
            # Not a path on disk -> treat the string as an inline document.
            chunks.extend(ingester.ingest_text(s))
            n_inline += 1

    docs = [
        {"content": c.content, "id": c.id, "metadata": c.metadata}
        for c in chunks
    ]
    if not docs:
        reason = ""
        if skipped:
            # Surface *why* each file was skipped (e.g. a missing PDF backend)
            # instead of a bare "0 documents" — the actionable hint already
            # exists, it was just being logged and thrown away.
            details = "; ".join(f"{p} — {why}" for p, why in skipped)
            reason = f" Skipped: {details}."
        raise ValueError(
            "knowledge_base produced 0 documents to index. Pass existing file "
            "or directory paths, or raw text strings with content. Got "
            f"{len(sources)} entr{'y' if len(sources) == 1 else 'ies'} "
            f"({n_files} treated as path(s), {n_inline} as inline text)."
            + reason
        )

    retrieval_tool = next(
        (t for t in tools if t.metadata.name == "retrieval"), None
    )
    if retrieval_tool is None:
        retrieval_tool = Retrieval()
        tools.append(retrieval_tool)

    retrieval_tool.add_documents(docs, chunk=False)
    logger.info(
        "RAG preset: indexed %d chunk(s) from %d path(s) + %d inline document(s)",
        len(docs),
        n_files,
        n_inline,
    )


def create_agent(
    preset: str,
    model: BaseModel | str | None = None,
    *,
    agent_name: str | None = None,
    extra_tools: list | None = None,
    knowledge_base: str | list[str] | None = None,
    system_prompt: str | None = None,
    max_iterations: int | None = None,
    temperature: float | None = None,
    enable_memory: bool | None = None,
    session_id: str | None = None,
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
        agent_name: Optional override for the agent name (the keyword ``name``
            is accepted as an alias).
        extra_tools: Additional tools to add beyond the preset. Each entry may be
            a tool instance **or** a registry tool *name* string (the same names
            the CLI's ``-t/--tools`` accepts, e.g. ``"calculator"``); an unknown
            name raises a clear ``ValueError`` with suggestions. The keyword
            ``tools`` is accepted as an alias for ``extra_tools``.
        knowledge_base: Only used by the ``rag`` preset. A document source, or
            a list of them, indexed into the agent's retrieval tool at
            creation. Each entry may be a file path, a directory path, **or**
            raw text (an entry that does not exist on disk is treated as an
            inline document and chunked directly). Raises ``ValueError`` if the
            sources yield zero documents, so a typo'd path or empty text fails
            loudly rather than producing a silent empty-index agent.
        system_prompt: Override the preset's system prompt.
        max_iterations: Override max iterations.
        temperature: Override temperature.
        enable_memory: Override memory setting.
        session_id: Optional persistent session id. When set, the agent loads or
            creates a stored conversation session so multi-turn context carries
            across processes (the same `--session-id` the CLI exposes).
        **config_overrides: Extra keyword arguments. Model-loading options
            (``engine``, ``quantization``, ``tensor_parallel_size``,
            ``gpu_memory_utilization``, ``apply_chat_template``,
            ``trust_remote_code``) are routed to ``load_model`` when ``model``
            is an id string; everything else is forwarded to ``AgentConfig``.
            An unrecognized keyword raises a clear ``TypeError`` listing what is
            accepted (rather than a cryptic dataclass error).

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
    # `name` is the natural way to name an agent (and a real AgentConfig field),
    # so accept it as an alias for `agent_name`. Without this it would fall into
    # **config_overrides and collide with the internally-supplied name, raising a
    # cryptic "got multiple values for keyword argument 'name'".
    if "name" in config_overrides:
        _alias_name = config_overrides.pop("name")
        if agent_name is None:
            agent_name = _alias_name

    # `tools=` is the natural LangChain-style guess for "give the agent these
    # tools". create_agent's parameter is `extra_tools` (added on top of the
    # preset's own tools). Accept `tools=` as an alias so the obvious call works,
    # rather than letting it collide with the internally-built `tools` list and
    # raise a cryptic "got multiple values for keyword argument 'tools'".
    if "tools" in config_overrides:
        _alias_tools = config_overrides.pop("tools")
        if extra_tools is not None:
            raise TypeError(
                "create_agent() got both `tools=` and `extra_tools=`. `tools=` is "
                "an alias for `extra_tools` (tools added on top of the preset) — "
                "pass only one."
            )
        extra_tools = _alias_tools

    if model is None:
        model = _resolve_default_model(preset)

    # Route load_model-only kwargs (engine, quantization, ...) to load_model
    # instead of letting them fall into AgentConfig and raise a cryptic
    # "unexpected keyword argument 'engine'". These only apply to a model *id*
    # string; for a pre-loaded instance the user already chose the engine.
    _load_kwargs = {
        k: config_overrides.pop(k)
        for k in (
            "engine", "engine_config", "tensor_parallel_size",
            "gpu_memory_utilization", "apply_chat_template",
            "quantization", "trust_remote_code",
        )
        if k in config_overrides
    }
    if _load_kwargs:
        if not isinstance(model, str):
            raise TypeError(
                f"{sorted(_load_kwargs)} only apply when `model` is a model-id "
                "string; you passed an already-loaded model. Load it yourself "
                "with load_model(id, engine=...) and pass the instance."
            )
        from ..models import load_model
        _prov = config_overrides.pop("provider", None)
        model = load_model(model, provider=_prov, **_load_kwargs)

    cfg = get_preset(preset)

    tools = _instantiate_tools(cfg.tool_names)

    # Special handling for RAG preset: ingest knowledge base on creation.
    # Failures here are LOUD: a knowledge_base that ingests to zero documents
    # raises instead of silently producing an empty-index agent that answers
    # "No documents in index." with success=True.
    if preset == "rag" and knowledge_base is not None:
        _ingest_rag_knowledge_base(tools, knowledge_base)

    if extra_tools:
        tools.extend(_normalize_extra_tools(extra_tools))

    try:
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
    except TypeError as exc:
        # Turn the raw dataclass "unexpected keyword argument" into an
        # actionable message listing what create_agent actually accepts.
        import dataclasses
        accepted = sorted(
            f.name for f in dataclasses.fields(AgentConfig)
            if f.name not in {"name", "model", "tools"}
        )
        raise TypeError(
            f"create_agent() got an unsupported keyword: {exc}. "
            "Pass a model-loading option (engine, quantization, "
            "tensor_parallel_size, gpu_memory_utilization, apply_chat_template, "
            "trust_remote_code) to choose the backend, or an AgentConfig field: "
            f"{', '.join(accepted)}."
        ) from exc

    logger.info(
        "Created '%s' preset agent with %d tools: %s",
        cfg.name,
        len(tools),
        ", ".join(t.metadata.name for t in tools) if tools else "(none)",
    )

    return Agent(agent_config, session_id=session_id)


# Keep the template (with the ``{PRESET_LIST}`` placeholder) so the docstring can
# be regenerated from the registry whenever the set of presets changes — the
# bundled extra presets (media/multimodal/notify) register by side-effect on
# import, which can happen after this module finishes (see presets/__init__.py).
_CREATE_AGENT_DOC_TEMPLATE = create_agent.__doc__ or ""


def _refresh_create_agent_doc() -> None:
    """Regenerate ``create_agent``'s preset list from the live registry.

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
