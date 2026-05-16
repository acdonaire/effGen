"""
Built-in tools for the effGen framework.

This module contains the standard set of tools that ship with effGen.
"""

# Import built-in tools (lazy loading handled by registry)
__all__ = [
    # OpenAI native (server-side) tools
    "OpenAIWebSearchTool",
    "OpenAICodeInterpreterTool",
    "OpenAIFileSearchTool",
    # Gemini native (server-side) tools
    "GoogleSearchTool",
    "GeminiUrlContextTool",
    "GeminiCodeExecutionTool",
    # Core tools
    "CodeExecutor",
    "PythonREPL",
    "WebSearch",
    "FileOperations",
    "Calculator",
    "Retrieval",
    "AgenticSearch",
    "BashTool",
    "WeatherTool",
    "JSONTool",
    "DateTimeTool",
    "TextProcessingTool",
    "URLFetchTool",
    "WikipediaTool",
    # Finance
    "StockPriceTool",
    "CurrencyConverterTool",
    "CryptoTool",
    # Data Science
    "DataFrameTool",
    "PlotTool",
    "StatsTool",
    # DevOps
    "GitTool",
    "DockerTool",
    "SystemInfoTool",
    "HTTPTool",
    # Knowledge
    "ArxivTool",
    "ArXivTool",
    "StackOverflowTool",
    "GitHubTool",
    "WolframAlphaTool",
    # Academic research
    "PubMedTool",
    "SemanticScholarTool",
    # Communication
    "EmailDraftTool",
    "SlackDraftTool",
    "NotificationTool",
]


def __getattr__(name):
    """Lazy import of tools."""
    if name in ("GoogleSearchTool", "GeminiUrlContextTool", "GeminiCodeExecutionTool"):
        from .gemini_native import (
            GeminiCodeExecutionTool,
            GeminiUrlContextTool,
            GoogleSearchTool,
        )
        if name == "GoogleSearchTool":
            return GoogleSearchTool
        elif name == "GeminiUrlContextTool":
            return GeminiUrlContextTool
        else:
            return GeminiCodeExecutionTool
    elif name in ("OpenAIWebSearchTool", "OpenAICodeInterpreterTool", "OpenAIFileSearchTool"):
        from .openai_native import (
            OpenAICodeInterpreterTool,
            OpenAIFileSearchTool,
            OpenAIWebSearchTool,
        )
        if name == "OpenAIWebSearchTool":
            return OpenAIWebSearchTool
        elif name == "OpenAICodeInterpreterTool":
            return OpenAICodeInterpreterTool
        else:
            return OpenAIFileSearchTool
    elif name in ("AnthropicBashTool", "AnthropicTextEditorTool", "AnthropicComputerTool"):
        from .anthropic_native import (
            AnthropicBashTool,
            AnthropicComputerTool,
            AnthropicTextEditorTool,
        )
        if name == "AnthropicBashTool":
            return AnthropicBashTool
        elif name == "AnthropicTextEditorTool":
            return AnthropicTextEditorTool
        else:
            return AnthropicComputerTool
    elif name == "CodeExecutor":
        from .code_executor import CodeExecutor
        return CodeExecutor
    elif name == "PythonREPL":
        from .python_repl import PythonREPL
        return PythonREPL
    elif name == "WebSearch":
        from .web_search import WebSearch
        return WebSearch
    elif name == "FileOperations":
        from .file_ops import FileOperations
        return FileOperations
    elif name == "Calculator":
        from .calculator import Calculator
        return Calculator
    elif name == "Retrieval":
        from .retrieval import Retrieval
        return Retrieval
    elif name == "AgenticSearch":
        from .agentic_search import AgenticSearch
        return AgenticSearch
    elif name == "BashTool":
        from .bash_tool import BashTool
        return BashTool
    elif name == "WeatherTool":
        from .weather import WeatherTool
        return WeatherTool
    elif name == "JSONTool":
        from .json_tool import JSONTool
        return JSONTool
    elif name == "DateTimeTool":
        from .datetime_tool import DateTimeTool
        return DateTimeTool
    elif name == "TextProcessingTool":
        from .text_processing import TextProcessingTool
        return TextProcessingTool
    elif name == "URLFetchTool":
        from .url_fetch import URLFetchTool
        return URLFetchTool
    elif name == "WikipediaTool":
        from .wikipedia_tool import WikipediaTool
        return WikipediaTool
    # Finance
    elif name == "StockPriceTool":
        from .finance import StockPriceTool
        return StockPriceTool
    elif name == "CurrencyConverterTool":
        from .finance import CurrencyConverterTool
        return CurrencyConverterTool
    elif name == "CryptoTool":
        from .finance import CryptoTool
        return CryptoTool
    # Data Science
    elif name == "DataFrameTool":
        from .data_analysis import DataFrameTool
        return DataFrameTool
    elif name == "PlotTool":
        from .data_analysis import PlotTool
        return PlotTool
    elif name == "StatsTool":
        from .data_analysis import StatsTool
        return StatsTool
    # DevOps
    elif name == "GitTool":
        from .devops import GitTool
        return GitTool
    elif name == "DockerTool":
        from .devops import DockerTool
        return DockerTool
    elif name == "SystemInfoTool":
        from .devops import SystemInfoTool
        return SystemInfoTool
    elif name == "HTTPTool":
        from .devops import HTTPTool
        return HTTPTool
    # Knowledge
    elif name == "ArxivTool":
        from .knowledge import ArxivTool
        return ArxivTool
    elif name == "ArXivTool":
        from .arxiv import ArXivTool
        return ArXivTool
    elif name == "PubMedTool":
        from .pubmed import PubMedTool
        return PubMedTool
    elif name == "SemanticScholarTool":
        from .semantic_scholar import SemanticScholarTool
        return SemanticScholarTool
    elif name == "StackOverflowTool":
        from .knowledge import StackOverflowTool
        return StackOverflowTool
    elif name == "GitHubTool":
        from .knowledge import GitHubTool
        return GitHubTool
    elif name == "WolframAlphaTool":
        from .knowledge import WolframAlphaTool
        return WolframAlphaTool
    # Communication
    elif name == "EmailDraftTool":
        from .communication import EmailDraftTool
        return EmailDraftTool
    elif name == "SlackDraftTool":
        from .communication import SlackDraftTool
        return SlackDraftTool
    elif name == "NotificationTool":
        from .communication import NotificationTool
        return NotificationTool
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
