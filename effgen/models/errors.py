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


class NoCandidateWithinBudgetError(Exception):
    """Raised when no provider/model fits within the user's budget.

    Attributes:
        user_budget_usd:    The budget that was not satisfiable.
        cheapest_cost_usd:  The lowest estimated cost among available candidates.
        cheapest_pair:      ``(provider, model_id)`` for the cheapest option.
    """

    def __init__(
        self,
        user_budget_usd: float,
        cheapest_cost_usd: float,
        cheapest_pair: "tuple[str, str] | None" = None,
    ) -> None:
        self.user_budget_usd = user_budget_usd
        self.cheapest_cost_usd = cheapest_cost_usd
        self.cheapest_pair = cheapest_pair
        pair_str = f" ({cheapest_pair[0]}/{cheapest_pair[1]})" if cheapest_pair else ""
        super().__init__(
            f"No candidate fits budget ${user_budget_usd:.6f} USD. "
            f"Cheapest available: ${cheapest_cost_usd:.6f}{pair_str}. "
            f"Increase user_budget_usd or use a free-tier provider."
        )


class NoCandidateWithinLatencyError(Exception):
    """Raised when no provider/model fits within the latency budget.

    Attributes:
        latency_budget_ms:    The SLA budget that was not satisfiable.
        fastest_latency_ms:   The lowest p50 latency among available candidates.
        fastest_pair:         ``(provider, model_id)`` for the fastest option.
    """

    def __init__(
        self,
        latency_budget_ms: float,
        fastest_latency_ms: float,
        fastest_pair: "tuple[str, str] | None" = None,
    ) -> None:
        self.latency_budget_ms = latency_budget_ms
        self.fastest_latency_ms = fastest_latency_ms
        self.fastest_pair = fastest_pair
        pair_str = f" ({fastest_pair[0]}/{fastest_pair[1]})" if fastest_pair else ""
        super().__init__(
            f"No candidate fits latency budget {latency_budget_ms:.0f}ms. "
            f"Fastest available p50: {fastest_latency_ms:.0f}ms{pair_str}. "
            "Increase latency_budget_ms or seed the tracker with faster providers."
        )


class ProviderTransientError(Exception):
    """Raised when a provider returns a transient server-side error (5xx).

    Attributes:
        provider:   Provider name (e.g. ``"groq"``).
        model_name: The model that was requested.
        status_code: HTTP status code (e.g. 500, 503).
        message:    Human-readable cause.
    """

    def __init__(
        self,
        provider: str,
        model_name: str = "",
        status_code: int = 500,
        message: str = "",
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.status_code = status_code
        self.message = message
        suffix = f" (model={model_name!r})" if model_name else ""
        body = message or "transient server error"
        super().__init__(f"{provider} {status_code}{suffix}: {body}")


class AllCandidatesExhaustedError(Exception):
    """Raised when all failover hops have been exhausted without a successful response.

    Every candidate in the router's ordered list has either failed with a
    retriable error or been skipped.  The caller should surface this to the
    user with the list of failures so they can diagnose the root cause.

    Attributes:
        failures: List of ``(provider, model_id, exception)`` triples for every
                  attempted hop.
        hop_limit: The maximum number of hops that was configured.
    """

    def __init__(
        self,
        failures: "list[tuple[str, str, Exception]]",
        hop_limit: int = 3,
    ) -> None:
        self.failures = failures
        self.hop_limit = hop_limit
        self.attempts = len(failures)
        summary = "; ".join(
            f"{prov}/{model}: {type(exc).__name__}({exc})"
            for prov, model, exc in failures
        )
        super().__init__(
            f"All {len(failures)} candidate attempts exhausted after "
            f"{hop_limit} failover hops. "
            f"Failures: [{summary}]"
        )


class InvalidRequestError(Exception):
    """Raised when the request itself is malformed and cannot be retried.

    Examples: invalid JSON schema, unsupported parameter, prompt too long.

    Attributes:
        provider:   Provider name.
        model_name: The model that rejected the request.
        message:    Human-readable cause.
    """

    def __init__(self, provider: str, model_name: str = "", message: str = "") -> None:
        self.provider = provider
        self.model_name = model_name
        self.message = message
        suffix = f" (model={model_name!r})" if model_name else ""
        body = message or "invalid request"
        super().__init__(f"{provider} invalid request{suffix}: {body}")


class BudgetExceededError(Exception):
    """Raised when cumulative spend crosses the configured daily/monthly budget.

    The router treats this as a retriable error and attempts failover to a
    free-tier provider when one is available.

    Attributes:
        budget_usd:   The budget limit that was crossed.
        actual_usd:   Current cumulative spend.
        period:       ``"daily"`` or ``"monthly"``.
        provider:     Provider that triggered the alert (if known).
        model:        Model that triggered the alert (if known).
    """

    def __init__(
        self,
        budget_usd: float,
        actual_usd: float,
        period: str = "daily",
        provider: str = "",
        model: str = "",
    ) -> None:
        self.budget_usd = budget_usd
        self.actual_usd = actual_usd
        self.period = period
        self.provider = provider
        self.model = model
        ctx = f" (provider={provider!r}, model={model!r})" if provider else ""
        super().__init__(
            f"{period.capitalize()} budget ${budget_usd:.4f} exceeded: "
            f"actual=${actual_usd:.4f}{ctx}. "
            "Router will attempt failover to a free-tier provider."
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
