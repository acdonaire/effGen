"""
effGen model error types.

Defines exceptions raised by model adapters beyond the standard
RuntimeError/ValueError hierarchy.
"""

from __future__ import annotations


class ModelRefusalError(Exception):
    """Raised when a model refuses to answer a structured-output request.

    OpenAI structured outputs may return a ``refusal`` field instead of
    valid JSON content.  The adapter raises this exception so callers can
    distinguish a genuine refusal (policy / safety) from a malformed
    response or a network error.

    Attributes:
        refusal_message: The raw refusal string returned by the model.
        model_name: The model that issued the refusal.
    """

    def __init__(self, refusal_message: str, model_name: str = "") -> None:
        self.refusal_message = refusal_message
        self.model_name = model_name
        suffix = f" (model={model_name!r})" if model_name else ""
        super().__init__(f"Model refused to generate structured output{suffix}: {refusal_message}")


class ModelAuthError(Exception):
    """Raised when a model adapter fails to authenticate with its provider.

    Distinguishes auth failures (bad/missing API key, revoked credentials)
    from other transport or rate-limit errors so callers get a clear
    "fix your key" signal instead of a generic 500.

    Attributes:
        provider: Provider name (e.g. ``"groq"``).
        model_name: The model that was requested.
        message: Human-readable cause.
    """

    def __init__(self, provider: str, model_name: str = "", message: str = "") -> None:
        self.provider = provider
        self.model_name = model_name
        self.message = message
        suffix = f" (model={model_name!r})" if model_name else ""
        body = message or "authentication failed"
        super().__init__(f"{provider} auth error{suffix}: {body}")


class ToolIncompatibleError(Exception):
    """Raised when a tool cannot be used with the configured model.

    Detected at Agent init time (not mid-run) so users get a clear error
    before any API calls are made.

    Attributes:
        tool_name: The tool that triggered the error.
        model_name: The model that does not support the tool.
        reason: Human-readable explanation.
    """

    def __init__(self, tool_name: str, model_name: str, reason: str = "") -> None:
        self.tool_name = tool_name
        self.model_name = model_name
        self.reason = reason
        parts = [f"Tool '{tool_name}' is incompatible with model '{model_name}'."]
        if reason:
            parts.append(reason)
        super().__init__(" ".join(parts))
