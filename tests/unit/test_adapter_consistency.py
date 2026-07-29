"""Provider-adapter consistency contracts.

These tests pin the uniform behaviour every adapter must share regardless of
the provider SDK's native quirks:

* ``finish_reason`` is normalized to one canonical set (OpenAI returns
  ``"stop"``, Anthropic ``"end_turn"``, Gemini ``FinishReason.STOP`` — all must
  collapse to ``"stop"``).
* provider failures carry a structured, **redacted** ``error_context`` with
  ``{provider, model, request_type, retry_status, remediation, category}``.

The finish-reason cases below are *golden fixtures* of the raw values each
provider SDK actually emits, captured so a provider SDK change is caught here
instead of silently leaking a non-canonical token downstream.
"""

from __future__ import annotations

import pytest

from effgen.models._adapter_utils import (
    CANONICAL_FINISH_REASONS,
    build_error_context,
    normalize_finish_reason,
    provider_runtime_error,
)
from effgen.models.errors import (
    RETRY_NON_RETRYABLE,
    RETRY_RATE_LIMITED,
    RETRY_WILL_RETRY,
    InvalidRequestError,
    ModelAuthError,
    ModelNotFoundError,
    ModelTimeoutError,
    ProviderTransientError,
    error_context_dict,
)


class _Enum:
    """Mimics google-genai FinishReason: str(enum) == 'FinishReason.NAME'."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:  # pragma: no cover - exercised indirectly
        return f"FinishReason.{self.name}"


# (raw_value, expected_canonical) — golden fixtures per provider SDK.
FINISH_REASON_CASES = [
    # OpenAI / OpenAI-compatible (openai, groq, cerebras, together, fireworks, hf)
    ("stop", "stop"),
    ("length", "length"),
    ("tool_calls", "tool_calls"),
    ("function_call", "tool_calls"),
    ("content_filter", "content_filter"),
    ("eos_token", "stop"),  # together/fireworks-ism
    # Anthropic stop_reason
    ("end_turn", "stop"),
    ("max_tokens", "length"),
    ("tool_use", "tool_calls"),
    ("stop_sequence", "stop"),
    ("refusal", "content_filter"),
    # Gemini FinishReason (str(enum) form and .name form)
    (_Enum("STOP"), "stop"),
    (_Enum("MAX_TOKENS"), "length"),
    (_Enum("SAFETY"), "content_filter"),
    (_Enum("MALFORMED_FUNCTION_CALL"), "tool_calls"),
    ("FinishReason.STOP", "stop"),
    ("FinishReason.RECITATION", "content_filter"),
    # Gemini numeric enum fallback
    (1, "stop"),
    (2, "length"),
    (3, "content_filter"),
    # local transformers engine
    # missing / unknown
    (None, "stop"),
    ("", "stop"),
]


@pytest.mark.parametrize("raw,expected", FINISH_REASON_CASES)
def test_normalize_finish_reason(raw, expected):
    assert normalize_finish_reason(raw) == expected


@pytest.mark.parametrize("raw,expected", FINISH_REASON_CASES)
def test_normalized_value_is_canonical(raw, expected):
    assert normalize_finish_reason(raw) in CANONICAL_FINISH_REASONS


def test_unknown_finish_reason_is_preserved_lowercased_not_dropped():
    # A novel value we don't map is returned lowercased (never silently lost).
    assert normalize_finish_reason("SOME_NEW_REASON") == "some_new_reason"


def test_bool_is_not_treated_as_int_enum():
    # bool is an int subclass; must not map True->1->"stop" by accident.
    assert normalize_finish_reason(True) == "stop"  # falls back to default


# ---------------------------------------------------------------------------
# Structured, redacted error context
# ---------------------------------------------------------------------------

_REQUIRED_CONTEXT_KEYS = {
    "provider",
    "model",
    "request_type",
    "retry_status",
    "remediation",
    "category",
}


def test_error_context_dict_shape():
    ctx = error_context_dict("openai", "gpt-5-nano", "generate", "auth")
    assert set(ctx) == _REQUIRED_CONTEXT_KEYS
    assert ctx["provider"] == "openai"
    assert ctx["model"] == "gpt-5-nano"
    assert ctx["request_type"] == "generate"
    assert ctx["retry_status"] == RETRY_NON_RETRYABLE
    assert "API key" in ctx["remediation"]


@pytest.mark.parametrize(
    "exc,expected_status",
    [
        (ModelAuthError("openai"), RETRY_NON_RETRYABLE),
        (ModelNotFoundError("openai", "x"), RETRY_NON_RETRYABLE),
        (InvalidRequestError("openai"), RETRY_NON_RETRYABLE),
        (ProviderTransientError("openai", "x", 503), RETRY_WILL_RETRY),
        (ModelTimeoutError("replicate", "x"), RETRY_WILL_RETRY),
    ],
)
def test_build_error_context_classifies_retry_status(exc, expected_status):
    ctx = build_error_context("openai", "x", "generate", exc)
    assert ctx["retry_status"] == expected_status
    assert set(ctx) == _REQUIRED_CONTEXT_KEYS


def test_rate_limit_status():
    err = RuntimeError("HTTP 429 rate limit exceeded")
    ctx = build_error_context("groq", "x", "generate", err)
    assert ctx["retry_status"] == RETRY_RATE_LIMITED


@pytest.mark.parametrize(
    "exc",
    [
        ModelAuthError("openai", "m"),
        ModelNotFoundError("openai", "m"),
        ProviderTransientError("openai", "m", 500),
        ModelTimeoutError("replicate", "m"),
        InvalidRequestError("openai", "m"),
    ],
)
def test_typed_errors_carry_error_context(exc):
    ctx = getattr(exc, "error_context", None)
    assert isinstance(ctx, dict)
    assert set(ctx) == _REQUIRED_CONTEXT_KEYS
    assert ctx["model"] == "m"


def test_provider_runtime_error_is_redacted():
    # A secret leaking into the SDK message must be scrubbed from the wrapped
    # RuntimeError and never reach logs / users.
    secret = "sk-abcdEFGH0123456789abcdEFGH0123456789abcdEF12"
    src = RuntimeError(f"connection error with key {secret}")
    err = provider_runtime_error("openai", "gpt-5-nano", "generate", src)
    assert secret not in str(err)
    assert err.error_context["provider"] == "openai"
    assert err.error_context["request_type"] == "generate"


def test_provider_runtime_error_keeps_classification():
    src = RuntimeError("invalid api key provided")
    err = provider_runtime_error("openai", "m", "generate", src)
    assert err.error_context["category"] == "auth"
    assert err.error_context["retry_status"] == RETRY_NON_RETRYABLE


# ---------------------------------------------------------------------------
# Cost metadata: one canonical per-call key, no duplicate alias
# ---------------------------------------------------------------------------

# ``cost_usd`` (this call's cost) and ``total_cost`` (the adapter's cumulative
# session cost — a genuinely different number) are both legitimate. A bare
# ``"cost"`` key was a third, always-equal-to-cost_usd duplicate with no
# consumer of its own; guard against it creeping back into a metadata dict.
_COST_METADATA_ADAPTER_MODULES = [
    "effgen.models.openai_adapter",
    "effgen.models.anthropic_adapter",
    "effgen.models.gemini_adapter",
]


@pytest.mark.parametrize("module_name", _COST_METADATA_ADAPTER_MODULES)
def test_adapter_source_has_no_redundant_cost_key(module_name):
    import importlib
    import inspect
    import re

    source = inspect.getsource(importlib.import_module(module_name))
    # Matches a literal `"cost": <expr>,` dict entry — not `"cost_usd"` or
    # `"total_cost"`, both of which are intentionally kept.
    assert not re.search(r'"cost"\s*:\s*\w', source), (
        f"{module_name} reintroduced a bare 'cost' metadata key; "
        "cost_usd is the canonical per-call key."
    )


# ---------------------------------------------------------------------------
# Error contract: every adapter reports a failure the same way
# ---------------------------------------------------------------------------
#
# The two rules under test, for every provider adapter and local engine:
#   1. a provider/runtime failure raises an error carrying ``.error_context``
#      (so the retry layer, the agent's error record and the server envelope
#      all read one classified shape) and never leaks a credential;
#   2. a call made before ``load()`` fails fast instead of being retried.
#
# The failure is injected at the client boundary rather than mocked at the
# adapter's own API, so each adapter's real error-handling code runs.

_SECRET = "sk-proj-abcdEFGH0123456789abcdEFGH0123456789abcdEF12"


class _FakeSDKError(Exception):
    """A provider SDK's HTTP error: status code plus a body echoing the key."""

    def __init__(self) -> None:
        super().__init__(
            "Error code: 401 - {'error': {'message': 'Incorrect API key "
            f"provided: {_SECRET}'}}}}"
        )
        self.status_code = 401


class _ExplodingClient:
    """Any attribute access returns another one; any call raises."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _ExplodingClient()

    def __call__(self, *args, **kwargs):
        raise _FakeSDKError()

    def __iter__(self):
        raise _FakeSDKError()


def _adapter_cases():
    """(provider, class, ctor kwargs) for every keyed provider adapter."""
    from effgen.models.anthropic_adapter import AnthropicAdapter
    from effgen.models.cerebras_adapter import CerebrasAdapter
    from effgen.models.fireworks_adapter import FireworksAdapter
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.models.groq_adapter import GroqAdapter
    from effgen.models.hf_inference_adapter import HFInferenceAdapter
    from effgen.models.openai_adapter import OpenAIAdapter
    from effgen.models.replicate_adapter import ReplicateAdapter
    from effgen.models.together_adapter import TogetherAdapter

    key = "unit-test-key"
    return [
        ("openai", OpenAIAdapter, {"model_name": "gpt-5-nano", "api_key": key}),
        ("groq", GroqAdapter, {"model_name": "llama-3.1-8b-instant", "api_key": key}),
        ("gemini", GeminiAdapter, {"model_name": "gemini-3.1-flash-lite", "api_key": key}),
        ("cerebras", CerebrasAdapter, {"model_name": "gpt-oss-120b", "api_key": key}),
        ("together", TogetherAdapter,
         {"model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "api_key": key}),
        ("fireworks", FireworksAdapter,
         {"model_name": "accounts/fireworks/models/llama-v3p1-8b-instruct", "api_key": key}),
        ("replicate", ReplicateAdapter,
         {"model_name": "meta/meta-llama-3-8b-instruct", "api_key": key}),
        ("hf_inference", HFInferenceAdapter,
         {"model_name": "meta-llama/Llama-3.1-8B-Instruct", "api_key": key}),
        ("anthropic", AnthropicAdapter,
         {"model_name": "claude-3-5-haiku-latest", "api_key": key}),
    ]


_TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
        },
    },
}]


def _invoke(adapter, method: str):
    """Call *method* on *adapter* with minimal valid arguments."""
    from effgen.models.base import GenerationConfig

    config = GenerationConfig(max_tokens=16)
    if method == "generate":
        return adapter.generate("hi", config)
    if method == "generate_stream":
        return list(adapter.generate_stream("hi", config))
    if method == "generate_with_tools":
        return adapter.generate_with_tools("hi", _TOOL_SCHEMA, config)
    if method == "generate_structured":
        return adapter.generate_structured("hi", {"type": "object", "properties": {}}, config)
    if method == "generate_with_system_prompt":
        return adapter.generate_with_system_prompt("hi", "sys", config)
    if method == "generate_with_history":
        return adapter.generate_with_history([{"role": "user", "content": "hi"}], config)
    raise AssertionError(f"unhandled method {method}")


_GENERATION_METHODS = (
    "generate",
    "generate_stream",
    "generate_with_tools",
    "generate_structured",
    "generate_with_system_prompt",
    "generate_with_history",
)


def _methods_of(cls):
    return [m for m in _GENERATION_METHODS if hasattr(cls, m)]


_PROVIDER_METHOD_CASES = [
    pytest.param(provider, cls, kwargs, method, id=f"{provider}-{method}")
    for provider, cls, kwargs in _adapter_cases()
    for method in _methods_of(cls)
]


@pytest.mark.parametrize("provider,cls,kwargs,method", _PROVIDER_METHOD_CASES)
def test_provider_failure_is_classified_and_redacted(provider, cls, kwargs, method):
    adapter = cls(**kwargs)
    adapter._is_loaded = True
    # Adapters name their SDK handle differently (``_client``/``client``).
    for attr in ("_client", "client", "_model", "model"):
        if hasattr(adapter, attr):
            setattr(adapter, attr, _ExplodingClient())

    with pytest.raises(Exception) as exc_info:  # noqa: PT011 - type varies per adapter
        _invoke(adapter, method)

    exc = exc_info.value
    ctx = getattr(exc, "error_context", None)
    assert isinstance(ctx, dict), f"{provider}.{method} raised {type(exc).__name__} with no error_context"
    assert set(ctx) == _REQUIRED_CONTEXT_KEYS
    assert _SECRET not in str(exc), f"{provider}.{method} leaked the submitted key"
    assert ctx["retry_status"] == RETRY_NON_RETRYABLE
    if isinstance(exc, NotImplementedError):
        # The adapter refuses the call before reaching the provider (this
        # model has no native tool calling); that refusal is classified too.
        assert ctx["category"] == "invalid_request"
    else:
        assert ctx["category"] == "auth"


@pytest.mark.parametrize("provider,cls,kwargs,method", _PROVIDER_METHOD_CASES)
def test_call_before_load_fails_fast(provider, cls, kwargs, method):
    from effgen.models.errors import classify_provider_error

    adapter = cls(**kwargs)  # never loaded

    with pytest.raises(Exception) as exc_info:  # noqa: PT011 - type varies per adapter
        _invoke(adapter, method)

    exc = exc_info.value
    ctx = getattr(exc, "error_context", None)
    assert isinstance(ctx, dict), f"{provider}.{method} raised {type(exc).__name__} with no error_context"
    assert ctx["retry_status"] == RETRY_NON_RETRYABLE
    assert not classify_provider_error(exc).should_retry, (
        f"{provider}.{method} would be retried although the model is not loaded"
    )


class _ExplodingLocalModel:
    """A local runtime that fails on the device (e.g. out of memory)."""

    device = "cuda:0"
    config = None

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _ExplodingLocalModel()

    def __call__(self, *args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def generate(self, *args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


def _local_engine_cases():
    from effgen.models.transformers_engine import TransformersEngine
    from effgen.models.vllm_engine import VLLMEngine

    return [
        ("transformers", TransformersEngine),
        ("vllm", VLLMEngine),
    ]


@pytest.mark.parametrize("engine_name,cls", _local_engine_cases())
@pytest.mark.parametrize("method", ["generate", "generate_stream"])
def test_local_engine_failure_is_classified(engine_name, cls, method):
    engine = cls(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    engine._is_loaded = True
    engine.model = _ExplodingLocalModel()
    engine.tokenizer = _ExplodingLocalModel()
    if hasattr(engine, "llm"):
        engine.llm = _ExplodingLocalModel()

    with pytest.raises(RuntimeError) as exc_info:
        _invoke(engine, method)

    ctx = getattr(exc_info.value, "error_context", None)
    assert isinstance(ctx, dict), f"{engine_name}.{method} raised no classified error"
    assert ctx["provider"] == engine_name
    assert set(ctx) == _REQUIRED_CONTEXT_KEYS


@pytest.mark.parametrize("engine_name,cls", _local_engine_cases())
@pytest.mark.parametrize("method", ["generate", "generate_stream"])
def test_local_engine_call_before_load_fails_fast(engine_name, cls, method):
    from effgen.models.errors import classify_provider_error

    engine = cls(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    with pytest.raises(RuntimeError) as exc_info:
        _invoke(engine, method)

    ctx = getattr(exc_info.value, "error_context", None)
    assert isinstance(ctx, dict)
    assert ctx["category"] == "not_loaded"
    assert not classify_provider_error(exc_info.value).should_retry


def test_device_out_of_memory_names_what_to_change():
    from effgen.models._adapter_utils import provider_runtime_error

    err = provider_runtime_error(
        "transformers", "Qwen/Qwen2.5-1.5B-Instruct", "generate",
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    )
    message = str(err)
    assert "max_tokens" in message
    assert "quantization_bits" in message


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("MPS backend out of memory (MPS allocated: 8.00 GB)"),
        RuntimeError("llama_model_load: failed to allocate buffer"),
    ],
)
def test_device_out_of_memory_is_not_retried(exc):
    """Reissuing the same request against a full device cannot succeed.

    Classified as ``unknown``, the retry layer would re-run a generation that
    is guaranteed to fail again and the remediation would point at the
    provider's status page instead of the device.
    """
    from effgen.models.errors import classify_provider_error

    cls = classify_provider_error(exc)
    assert cls.category == "resource_exhausted"
    assert not cls.should_retry


def test_provider_500_out_of_memory_stays_transient():
    """A cloud 5xx keeps its status-derived class — that one can be retried."""
    from effgen.models.errors import classify_provider_error

    assert classify_provider_error(
        _StatusError(500, "server out of memory")
    ).category == "transient"


def test_not_loaded_error_shape():
    from effgen.models._adapter_utils import not_loaded_error

    err = not_loaded_error("openai", "gpt-5-nano", "generate")
    assert err.error_context["category"] == "not_loaded"
    assert err.error_context["retry_status"] == RETRY_NON_RETRYABLE
    assert "load()" in str(err)


# ---------------------------------------------------------------------------
# Classification of failures providers report ambiguously
# ---------------------------------------------------------------------------


class _StatusError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.status_code = status


def test_rejected_key_reported_as_400_is_still_auth():
    """Gemini answers a rejected API key with HTTP 400 INVALID_ARGUMENT.

    Classifying that by status alone labels the same failure ``auth`` on the
    non-streaming path and ``invalid_request`` on the streaming one.
    """
    from effgen.models.errors import classify_provider_error

    exc = _StatusError(
        400,
        "400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': "
        "'API key not valid. Please pass a valid API key.', "
        "'status': 'INVALID_ARGUMENT'}}",
    )
    assert classify_provider_error(exc).category == "auth"


def test_generic_invalid_argument_is_not_auth():
    from effgen.models.errors import classify_provider_error

    exc = _StatusError(400, "400 INVALID_ARGUMENT. Unsupported value for 'top_k'")
    assert classify_provider_error(exc).category == "invalid_request"


@pytest.mark.parametrize(
    "exc",
    [
        _StatusError(402, "Payment required"),
        RuntimeError("Replicate account has insufficient credits. Add billing"),
    ],
)
def test_billing_failures_are_not_retried(exc):
    from effgen.models.errors import classify_provider_error

    assert not classify_provider_error(exc).should_retry


def test_typed_error_message_is_redacted():
    err = ModelAuthError(
        "openai", "gpt-5-nano",
        f"Incorrect API key provided: {_SECRET}",
    )
    assert _SECRET not in str(err)
    assert _SECRET not in err.message


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        ("effgen.models.hf_inference_adapter:HFInferenceAdapter",
         {"model_name": "meta-llama/Llama-3.1-8B-Instruct"}),
        ("effgen.models.replicate_adapter:ReplicateAdapter",
         {"model_name": "meta/meta-llama-3-8b-instruct"}),
    ],
)
def test_api_key_alias_is_used_not_dropped(cls, kwargs):
    """These two adapters name the credential ``api_token``.

    ``api_key`` is the name the other seven use, so it is accepted here too —
    otherwise it lands in ``**kwargs`` and the ambient environment credential
    is used instead of the one the caller passed.
    """
    import importlib

    module_name, class_name = cls.split(":")
    adapter_cls = getattr(importlib.import_module(module_name), class_name)
    adapter = adapter_cls(api_key="explicit-caller-credential", **kwargs)
    assert adapter._api_token == "explicit-caller-credential"


@pytest.mark.parametrize(
    "cls",
    [
        "effgen.models.hf_inference_adapter:HFInferenceAdapter",
        "effgen.models.replicate_adapter:ReplicateAdapter",
    ],
)
def test_api_key_alias_does_not_shift_positional_arguments(cls):
    """The alias sits last, so an existing positional call still means what it did."""
    import importlib
    import inspect

    module_name, class_name = cls.split(":")
    adapter_cls = getattr(importlib.import_module(module_name), class_name)
    names = [
        p.name
        for p in inspect.signature(adapter_cls.__init__).parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert names[-1] == "api_key", names


# ---------------------------------------------------------------------------
# Reasoning-only turns: one contract every OpenAI-compatible adapter runs
# ---------------------------------------------------------------------------

_REASONING_CHAIN = "Thinking Process:\n1. Analyze the request.\n2. Compute.\n"


def _reasoning_only_response(finish_reason: str = "length"):
    """An SDK response whose whole output was reasoning: no visible token."""
    from unittest.mock import MagicMock

    message = MagicMock()
    message.content = ""
    message.reasoning = _REASONING_CHAIN
    message.tool_calls = []
    message.refusal = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = 559
    usage.completion_tokens = 90
    usage.total_tokens = 649
    usage.completion_tokens_details.reasoning_tokens = 90
    usage.prompt_tokens_details.cached_tokens = 0

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _loaded_adapter(provider: str):
    """Build one adapter of *provider* with a client that never leaves the process."""
    import importlib
    from unittest.mock import MagicMock

    specs = {
        "together": ("together_adapter", "TogetherAdapter", {}),
        "groq": ("groq_adapter", "GroqAdapter", {}),
        "cerebras": ("cerebras_adapter", "CerebrasAdapter", {}),
        "fireworks": ("fireworks_adapter", "FireworksAdapter", {}),
        "openai": ("openai_adapter", "OpenAIAdapter", {"model_name": "gpt-5-nano"}),
        "hf_inference": ("hf_inference_adapter", "HFInferenceAdapter",
                         {"model_name": "meta-llama/Llama-3.1-8B-Instruct"}),
    }
    module_name, class_name, kwargs = specs[provider]
    module = importlib.import_module(f"effgen.models.{module_name}")
    if provider == "hf_inference":
        adapter = getattr(module, class_name)(api_key="test-key", **kwargs)
    else:
        adapter = getattr(module, class_name)(
            api_key="test-key", enable_rate_limiting=False,
            enable_cost_tracking=False, **kwargs,
        )
    client = MagicMock()
    client.chat.completions.create.return_value = _reasoning_only_response()
    client.chat_completion.return_value = _reasoning_only_response()
    adapter._client = client
    if hasattr(adapter, "client"):
        adapter.client = client
    adapter._is_loaded = True
    return adapter


def _set_response(adapter, response) -> None:
    """Point *adapter*'s fake client at *response*, whichever call it makes."""
    adapter._client.chat.completions.create.return_value = response
    adapter._client.chat_completion.return_value = response


@pytest.mark.parametrize(
    "provider",
    ["together", "groq", "cerebras", "fireworks", "openai", "hf_inference"],
)
def test_every_adapter_reports_a_reasoning_only_turn(provider):
    """A turn billed entirely to reasoning is never a bare empty string.

    Each adapter flags it and carries a message naming the model, the cap in
    force and the reasoning budget spent, so a direct ``generate()`` caller —
    which has no agent loop watching — can tell it apart from a flaky empty.
    """
    from effgen.models.base import GenerationConfig

    adapter = _loaded_adapter(provider)
    result = adapter.generate("Question: 234 * 567?",
                              config=GenerationConfig(max_tokens=64))

    assert result.text == ""
    assert result.metadata["reasoning_only"] is True
    assert result.metadata["reasoning_tokens"] == 90
    assert result.metadata["reasoning_chars"] == len(_REASONING_CHAIN)
    reason = result.metadata["empty_response_reason"]
    assert adapter.model_name in reason
    assert "64" in reason  # the cap the request asked for
    assert "90 reasoning tokens" in reason


@pytest.mark.parametrize(
    "provider",
    ["together", "groq", "cerebras", "fireworks", "openai", "hf_inference"],
)
def test_an_answer_is_never_flagged_reasoning_only(provider):
    adapter = _loaded_adapter(provider)
    response = _reasoning_only_response(finish_reason="stop")
    response.choices[0].message.content = "132678"
    _set_response(adapter, response)

    result = adapter.generate("Question: 234 * 567?")

    assert result.text == "132678"
    assert "reasoning_only" not in result.metadata
    assert result.metadata["reasoning_tokens"] == 90


def _reasoning_only_stream(finish_reason: str = "length"):
    """Streamed chunks carrying a reasoning chain and not one visible token."""
    from unittest.mock import MagicMock

    chunks = []
    for fragment in ("Thinking Process:\n", "1. Analyze the request.\n"):
        delta = MagicMock()
        delta.content = None
        delta.reasoning = fragment
        delta.tool_calls = None
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = None
        chunk = MagicMock()
        chunk.choices = [choice]
        chunk.usage = None
        chunks.append(chunk)

    final_delta = MagicMock()
    final_delta.content = None
    final_delta.reasoning = None
    final_delta.tool_calls = None
    final_choice = MagicMock()
    final_choice.delta = final_delta
    final_choice.finish_reason = finish_reason
    final = MagicMock()
    final.choices = [final_choice]
    final.usage = MagicMock()
    final.usage.prompt_tokens = 20
    final.usage.completion_tokens = 48
    final.usage.total_tokens = 68
    final.usage.completion_tokens_details.reasoning_tokens = 48
    final.usage.prompt_tokens_details.cached_tokens = 0
    chunks.append(final)
    return chunks


@pytest.mark.parametrize(
    "provider",
    ["together", "groq", "cerebras", "fireworks", "openai", "hf_inference"],
)
def test_a_stream_that_yields_nothing_visible_reports_why(provider, caplog):
    """An empty iterator would otherwise read as "the model had nothing to say".

    A streamed turn has no metadata channel back to the caller, so the reason
    is logged — naming the cap in force and the reasoning budget spent.
    """
    import logging

    from effgen.models.base import GenerationConfig

    adapter = _loaded_adapter(provider)
    adapter._client.chat.completions.create.return_value = _reasoning_only_stream()
    adapter._client.chat_completion.return_value = _reasoning_only_stream()

    with caplog.at_level(logging.WARNING):
        chunks = list(adapter.generate_stream(
            "Question: 234 * 567?", config=GenerationConfig(max_tokens=48),
        ))

    assert chunks == [] or "".join(chunks) == ""
    reported = [r.message for r in caplog.records if "no visible text" in r.message]
    assert len(reported) == 1, [r.message for r in caplog.records]
    assert adapter.model_name in reported[0]
    assert "48" in reported[0]
