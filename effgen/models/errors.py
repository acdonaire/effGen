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


class ModelTimeoutError(Exception):
    """Raised when a model prediction does not complete within the timeout.

    Replicate and other async-polling providers may take variable time to
    complete.  This error is raised when the configurable timeout is exceeded
    so callers can distinguish a slow/stuck prediction from a network error.

    Attributes:
        provider: Provider name (e.g. ``"replicate"``).
        model_name: The model that was requested.
        timeout_seconds: The timeout that was exceeded.
        prediction_id: The prediction ID (if available) so it can be cancelled.
    """

    def __init__(
        self,
        provider: str,
        model_name: str = "",
        timeout_seconds: float = 300,
        prediction_id: str = "",
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.prediction_id = prediction_id
        suffix = f" (model={model_name!r})" if model_name else ""
        pid = f", prediction_id={prediction_id!r}" if prediction_id else ""
        super().__init__(
            f"{provider} prediction timed out after {timeout_seconds}s{suffix}{pid}. "
            f"Increase timeout= on the adapter or check provider status."
        )


class ModelUnavailableError(Exception):
    """Raised when a model is temporarily unavailable on the provider's serverless tier.

    HuggingFace Serverless Inference rotates model availability.  This error
    surfaces the 503/404 with actionable suggestions so callers know what to try
    instead of getting a raw HTTP error.

    Attributes:
        provider: Provider name (e.g. ``"hf"``).
        model_name: The model that is unavailable.
        suggestions: List of alternative model IDs to try.
        message: Human-readable cause.
    """

    def __init__(
        self,
        provider: str,
        model_name: str = "",
        suggestions: list[str] | None = None,
        message: str = "",
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.suggestions = suggestions or []
        self.message = message
        suffix = f" (model={model_name!r})" if model_name else ""
        body = message or "model is not available on the serverless tier"
        suggest_str = ""
        if self.suggestions:
            suggest_str = "  Try one of: " + ", ".join(self.suggestions)
        super().__init__(f"{provider} unavailable{suffix}: {body}.{suggest_str}")


class ModelNotFoundError(Exception):
    """Raised when a requested model ID does not exist on the provider.

    Attributes:
        provider: Provider name.
        model_name: The model that was not found.
    """

    def __init__(self, provider: str, model_name: str = "", message: str = "") -> None:
        self.provider = provider
        self.model_name = model_name
        self.message = message
        suffix = f" (model={model_name!r})" if model_name else ""
        body = message or "model not found"
        super().__init__(f"{provider} error{suffix}: {body}")


class AmbiguousModelError(Exception):
    """Raised when a model ID exists in multiple providers and no provider prefix is given.

    Example: ``"llama-3.3-70b"`` could be Groq, Together, or Fireworks.
    Callers should disambiguate with ``"groq:llama-3.3-70b-versatile"`` syntax.

    Attributes:
        model_id: The ambiguous model identifier.
        providers: List of providers that each claim this model ID.
    """

    def __init__(self, model_id: str, providers: list[str]) -> None:
        self.model_id = model_id
        self.providers = list(providers)
        plist = ", ".join(f'"{p}"' for p in providers)
        super().__init__(
            f"Model {model_id!r} is available on multiple providers: [{plist}]. "
            f"Disambiguate with 'provider:model_id' syntax, e.g. "
            f'"{providers[0]}:{model_id}".'
        )


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
