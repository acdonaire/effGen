"""
effGen model error types.

Defines exceptions raised by model adapters beyond the standard
RuntimeError/ValueError hierarchy, plus :func:`classify_provider_error`, a
single classification layer the agent/retry logic uses to decide whether an
error is worth retrying and how to report it.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Structured error context (shared by typed errors below and by adapters via
# effgen.models._adapter_utils). One source of truth for the per-category
# retry status + remediation text so every provider failure reports the same
# machine-readable shape: {provider, model, request_type, retry_status,
# remediation, category}.
# ---------------------------------------------------------------------------

# retry_status values (stable, machine-readable).
RETRY_WILL_RETRY = "will_retry"
RETRY_RATE_LIMITED = "rate_limited"
RETRY_NON_RETRYABLE = "non_retryable"

# category -> retry_status
_RETRY_STATUS_BY_CATEGORY: dict[str, str] = {
    "auth": RETRY_NON_RETRYABLE,
    "not_found": RETRY_NON_RETRYABLE,
    "refusal": RETRY_NON_RETRYABLE,
    "invalid_request": RETRY_NON_RETRYABLE,
    "fatal": RETRY_NON_RETRYABLE,
    "rate_limited": RETRY_RATE_LIMITED,
    "transient": RETRY_WILL_RETRY,
    "timeout": RETRY_WILL_RETRY,
    "unknown": RETRY_WILL_RETRY,
}

# category -> remediation hint (no secrets; safe to log/surface).
REMEDIATION_BY_CATEGORY: dict[str, str] = {
    "auth": "Check the provider API key (present, correct, not expired, and has access to this model).",
    "not_found": "Model id not found — run `effgen models list` to see ids, `effgen models refresh` to update the catalog, and verify the id/provider prefix.",
    "rate_limited": "Rate limited — lower concurrency or configure a rate limit; the client backs off and retries.",
    "timeout": "Request timed out — increase the adapter timeout or retry.",
    "transient": "Transient provider error — retry shortly; check the provider status page if it persists.",
    "invalid_request": "Request rejected as invalid — check parameters, prompt size, and any JSON schema.",
    "refusal": "The model refused the request — rephrase or adjust the content.",
    "fatal": "Unrecoverable error — check configuration and provider status.",
    "unknown": "Unexpected provider error — check the provider status and effGen logs (set EFFGEN_LOG_LEVEL=DEBUG).",
}


def error_context_dict(
    provider: str,
    model: str,
    request_type: str,
    category: str,
) -> dict[str, str]:
    """Build the canonical structured error-context dict for *category*.

    The returned dict is always free of secret material (the fields are
    static/derived), so it is safe to attach to exceptions, log, and surface.
    """
    return {
        "provider": provider or "",
        "model": model or "",
        "request_type": request_type or "request",
        "retry_status": _RETRY_STATUS_BY_CATEGORY.get(category, RETRY_WILL_RETRY),
        "remediation": REMEDIATION_BY_CATEGORY.get(category, REMEDIATION_BY_CATEGORY["unknown"]),
        "category": category,
    }


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
        self.error_context = error_context_dict("", model_name, "request", "refusal")
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
        self.error_context = error_context_dict(provider, model_name, "request", "auth")
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
        self.error_context = error_context_dict(provider, model_name, "request", "timeout")
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
        self.error_context = error_context_dict(provider, model_name, "request", "not_found")
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
        self.error_context = error_context_dict(provider, model_name, "request", "not_found")
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
        self.error_context = error_context_dict(provider, model_name, "request", "transient")
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
        self.error_context = error_context_dict(provider, model_name, "request", "invalid_request")
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


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorClass:
    """Classification of a provider/model error for retry + reporting.

    ``category`` is the single human-readable label (``"auth"``,
    ``"not_found"``, ``"rate_limited"``, ``"transient"``, ``"timeout"``,
    ``"refusal"``, ``"invalid_request"``, ``"fatal"``, or ``"unknown"``).
    The boolean flags let callers branch without string-matching ``category``.

    Only ``retryable`` errors (transient/timeout/unknown) and ``rate_limited``
    errors should be retried. ``auth``, ``not_found``, ``refusal`` and
    ``fatal`` errors will not succeed on retry and must fail fast.
    """

    category: str
    retryable: bool = False
    rate_limited: bool = False
    auth: bool = False
    not_found: bool = False
    refusal: bool = False
    fatal: bool = False

    @property
    def should_retry(self) -> bool:
        """True when retrying the call could plausibly succeed."""
        return self.retryable or self.rate_limited


# Reusable instances (frozen → safe to share).
_AUTH = ErrorClass("auth", auth=True, fatal=True)
_NOT_FOUND = ErrorClass("not_found", not_found=True)
_RATE_LIMITED = ErrorClass("rate_limited", rate_limited=True, retryable=True)
_TRANSIENT = ErrorClass("transient", retryable=True)
_TIMEOUT = ErrorClass("timeout", retryable=True)
_REFUSAL = ErrorClass("refusal", refusal=True)
_INVALID = ErrorClass("invalid_request", fatal=True)
_FATAL = ErrorClass("fatal", fatal=True)
_UNKNOWN = ErrorClass("unknown", retryable=True)


def _status_code_of(exc: Exception) -> int | None:
    """Best-effort HTTP status extraction across SDK exception shapes."""
    for attr in ("status_code", "status", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def classify_provider_error(exc: Exception) -> ErrorClass:
    """Classify a provider/model exception into an :class:`ErrorClass`.

    Recognises effGen's own typed errors first, then falls back to inspecting
    SDK exception class names, HTTP status codes, and message text so raw
    provider-SDK errors (openai/groq/gemini/…) are still classified correctly.

    Args:
        exc: The exception raised by an adapter or provider SDK.

    Returns:
        An :class:`ErrorClass`. Unknown errors default to ``retryable`` so a
        genuine transient blip is not turned into a hard failure, while the
        recognised auth/not-found/refusal/invalid classes fail fast.
    """
    # 1. effGen typed errors (adapters map most provider errors to these).
    if isinstance(exc, ModelAuthError):
        return _AUTH
    if isinstance(exc, ModelNotFoundError | ModelUnavailableError):
        return _NOT_FOUND
    if isinstance(exc, ModelRefusalError):
        return _REFUSAL
    if isinstance(exc, ModelTimeoutError):
        return _TIMEOUT
    if isinstance(exc, ProviderTransientError):
        return _TRANSIENT
    if isinstance(exc, InvalidRequestError):
        return _INVALID
    if isinstance(
        exc,
        BudgetExceededError
        | NoCandidateWithinBudgetError
        | NoCandidateWithinLatencyError
        | AllCandidatesExhaustedError
        | AmbiguousModelError,
    ):
        return _FATAL

    # 2. HTTP status code (raw SDK errors carry one).
    status = _status_code_of(exc)
    if status is not None:
        if status in (401, 403):
            return _AUTH
        if status == 404:
            return _NOT_FOUND
        if status == 429:
            return _RATE_LIMITED
        if status in (400, 422):
            return _INVALID
        if status == 408 or status >= 500:
            return _TRANSIENT

    # 3. Exception class name heuristics.
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "toomanyrequests" in name:
        return _RATE_LIMITED
    if any(k in name for k in ("authentication", "permission", "unauthorized", "forbidden")):
        return _AUTH
    if "notfound" in name:
        return _NOT_FOUND
    if "timeout" in name or "timedout" in name:
        return _TIMEOUT
    if any(k in name for k in ("connection", "serviceunavailable", "internalserver", "apistatus", "overloaded")):
        return _TRANSIENT
    if any(k in name for k in ("badrequest", "invalidrequest", "unprocessable", "validation")):
        return _INVALID

    # 4. Message-text heuristics (last resort).
    msg = str(exc).lower()
    if any(k in msg for k in ("rate limit", "rate-limit", "too many requests", "quota exceeded", "429")):
        return _RATE_LIMITED
    if any(k in msg for k in (
        "invalid api key", "incorrect api key", "invalid_api_key", "no api key",
        "authentication", "unauthorized", "permission denied", "api key not",
    )):
        return _AUTH
    if "unknown provider" in msg:
        return _INVALID
    if any(k in msg for k in ("model_not_found", "does not exist", "not found", "no such model", "unknown model")):
        return _NOT_FOUND
    if "timed out" in msg or "timeout" in msg:
        return _TIMEOUT
    if any(k in msg for k in ("connection error", "service unavailable", "temporarily unavailable", "overloaded", "503", "502", "500")):
        return _TRANSIENT

    return _UNKNOWN
