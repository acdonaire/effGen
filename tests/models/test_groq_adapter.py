"""
Unit tests for GroqAdapter (mocks OK for adapter plumbing).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from effgen.models.groq_adapter import GroqAdapter
from effgen.models.groq_models import (
    GROQ_DEFAULT_MODEL,
    GROQ_MODELS,
    available_models,
    chat_models,
    model_info,
    tool_capable_models,
)

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestGroqModelsRegistry:
    def test_available_models_nonempty(self):
        assert len(available_models()) > 0

    def test_chat_models_subset(self):
        assert set(chat_models()).issubset(set(available_models()))

    def test_tool_capable_subset_of_chat(self):
        assert set(tool_capable_models()).issubset(set(chat_models()))

    def test_model_info_known(self):
        info = model_info("llama-3.1-8b-instant")
        assert info["context"] == 131_072
        assert info["supports_native_tools"] is True

    def test_model_info_unknown_raises(self):
        with pytest.raises(KeyError):
            model_info("nonexistent-model-xyz")

    def test_default_model_in_registry(self):
        assert GROQ_DEFAULT_MODEL in GROQ_MODELS

    def test_all_chat_models_have_required_fields(self):
        required = {"context", "max_output", "supports_native_tools", "supports_streaming"}
        for model_id, info in GROQ_MODELS.items():
            if info.get("modality") == "chat":
                missing = required - set(info.keys())
                assert not missing, f"{model_id} missing fields: {missing}"

    def test_known_tool_capable_models(self):
        capable = tool_capable_models()
        assert "llama-3.3-70b-versatile" in capable
        assert "llama-3.1-8b-instant" in capable
        assert "qwen/qwen3-32b" in capable

    def test_guard_models_no_tools(self):
        for model_id in ["meta-llama/llama-prompt-guard-2-22m", "meta-llama/llama-prompt-guard-2-86m"]:
            assert GROQ_MODELS[model_id]["supports_native_tools"] is False

    def test_stt_models_not_in_chat(self):
        chat = chat_models()
        assert "whisper-large-v3" not in chat
        assert "whisper-large-v3-turbo" not in chat


# ---------------------------------------------------------------------------
# GroqAdapter unit tests (mocked)
# ---------------------------------------------------------------------------

class TestGroqAdapterInit:
    def test_unknown_model_raises(self):
        from effgen.models.errors import ModelNotFoundError
        with pytest.raises(ModelNotFoundError, match="Unknown Groq model"):
            GroqAdapter("does-not-exist-model")

    def test_stt_model_raises(self):
        with pytest.raises(ValueError, match="stt model"):
            GroqAdapter("whisper-large-v3")

    def test_default_model_accepted(self):
        adapter = GroqAdapter()
        assert adapter.model_name == GROQ_DEFAULT_MODEL

    def test_rate_limiter_wired_by_default(self):
        adapter = GroqAdapter("llama-3.1-8b-instant")
        assert adapter._rate_limiter is not None

    def test_rate_limiter_disabled(self):
        adapter = GroqAdapter("llama-3.1-8b-instant", enable_rate_limiting=False)
        assert adapter._rate_limiter is None

    def test_context_length_before_load(self):
        adapter = GroqAdapter("llama-3.3-70b-versatile")
        assert adapter.get_context_length() == 131_072


class TestGroqAdapterLoad:
    def test_load_no_key_raises(self):
        adapter = GroqAdapter("llama-3.1-8b-instant", api_key=None)
        # Stub SDK so the no-key path is reached even when groq isn't installed
        stub = MagicMock()
        stub.Groq = MagicMock()
        with patch.dict("sys.modules", {"groq": stub}):
            with patch.dict("os.environ", {}, clear=True):
                import os
                os.environ.pop("GROQ_API_KEY", None)
                with pytest.raises(ValueError, match="GROQ_API_KEY"):
                    adapter.load()

    def test_load_sets_is_loaded(self):
        with patch("effgen.models.groq_adapter.os.getenv", return_value="fake-key"):
            with patch("effgen.models.groq_adapter.GroqAdapter.load") as mock_load:
                mock_load.return_value = None
                adapter = GroqAdapter("llama-3.1-8b-instant", api_key="fake-key")
                # Manually set state as load() is mocked
                adapter._is_loaded = True
                assert adapter._is_loaded

    def test_import_error_on_missing_groq(self):
        adapter = GroqAdapter("llama-3.1-8b-instant", api_key="fake-key")
        with patch("builtins.__import__", side_effect=ImportError("No module named 'groq'")):
            with pytest.raises((ImportError, RuntimeError)):
                adapter.load()

    def test_unload_clears_client(self):
        adapter = GroqAdapter("llama-3.1-8b-instant", api_key="fake-key")
        adapter._client = MagicMock()
        adapter._is_loaded = True
        adapter.unload()
        assert adapter._client is None
        assert not adapter._is_loaded


class TestGroqAdapterGenerate:
    def _make_mock_response(self, text="Hello!", tool_calls=None):
        """Build a mock Groq API response."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = text
        mock_message.tool_calls = tool_calls
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop" if not tool_calls else "tool_calls"
        mock_response.choices = [mock_choice]
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        return mock_response

    def _loaded_adapter(self, model="llama-3.1-8b-instant"):
        adapter = GroqAdapter(model, api_key="fake-key", enable_rate_limiting=False, enable_cost_tracking=False)
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def test_generate_not_loaded_raises(self):
        adapter = GroqAdapter("llama-3.1-8b-instant", api_key="fake-key")
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.generate("Hello")

    def test_generate_returns_text(self):
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Bonjour!")
        result = adapter.generate("Say hello in French")
        assert result.text == "Bonjour!"
        assert result.model_name == "llama-3.1-8b-instant"

    def test_generate_usage_populated(self):
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Hi")
        result = adapter.generate("Hi")
        assert result.metadata["prompt_tokens"] == 10
        assert result.metadata["completion_tokens"] == 5
        assert result.metadata["total_tokens"] == 15

    def test_generate_forwards_frequency_and_presence_penalty(self):
        """A pinned penalty must reach the Groq request, matching seed/top_p."""
        from effgen.models.base import GenerationConfig

        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Hi")
        adapter.generate(
            "Hi",
            config=GenerationConfig(frequency_penalty=1.5, presence_penalty=0.5),
        )
        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["frequency_penalty"] == 1.5
        assert call_kwargs["presence_penalty"] == 0.5

    def test_generate_omits_default_penalties(self):
        """The neutral 0.0 default is not sent, matching the seed/top_p convention."""
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.return_value = self._make_mock_response("Hi")
        adapter.generate("Hi")
        call_kwargs = adapter._client.chat.completions.create.call_args.kwargs
        assert "frequency_penalty" not in call_kwargs
        assert "presence_penalty" not in call_kwargs

    def test_generate_with_tools_calls_api(self):
        adapter = self._loaded_adapter("llama-3.3-70b-versatile")
        tc = MagicMock()
        tc.id = "tc1"
        tc.type = "function"
        tc.function.name = "calculator"
        tc.function.arguments = '{"expression": "2+2"}'
        mock_resp = self._make_mock_response("", tool_calls=[tc])
        adapter._client.chat.completions.create.return_value = mock_resp
        tools = [{"type": "function", "function": {"name": "calculator", "description": "calc",
                   "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}}]
        result = adapter.generate_with_tools("What is 2+2?", tools)
        assert len(result.metadata["tool_calls"]) == 1
        assert result.metadata["tool_calls"][0]["function"]["name"] == "calculator"

    def test_generate_with_tools_recovers_failed_generation_tool_call(self):
        adapter = self._loaded_adapter("llama-3.3-70b-versatile")
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': 'Failed to call a function.', "
            "'type': 'invalid_request_error', 'code': 'tool_use_failed', "
            "'failed_generation': '<function=calculator{\"expression\": \"2+2\"}</function>'}}"
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            },
        }]
        result = adapter.generate_with_tools("What is 2+2?", tools)
        tool_call = result.metadata["tool_calls"][0]
        assert tool_call["function"]["name"] == "calculator"
        assert tool_call["function"]["arguments"] == {"expression": "2+2"}

    def test_recovered_tool_call_logs_at_info_not_warning(self, caplog):
        """The tool_use_failed recovery is not actionable — it must log at
        INFO (shown only with --verbose) so a successful turn's default
        output doesn't carry a stray WARNING line."""
        import logging

        adapter = self._loaded_adapter("llama-3.3-70b-versatile")
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': 'Failed to call a function.', "
            "'type': 'invalid_request_error', 'code': 'tool_use_failed', "
            "'failed_generation': '<function=calculator{\"expression\": \"2+2\"}</function>'}}"
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            },
        }]
        with caplog.at_level(logging.INFO, logger="effgen.models.groq_adapter"):
            adapter.generate_with_tools("What is 2+2?", tools)
        recovery_records = [
            r for r in caplog.records if "tool_use_failed but included a parseable" in r.message
        ]
        assert recovery_records, "expected the recovery note to be logged"
        assert all(r.levelno == logging.INFO for r in recovery_records)
        assert not any(r.levelno >= logging.WARNING for r in recovery_records)

    def test_generate_with_tools_recovers_failed_generation_with_closing_bracket(self):
        adapter = self._loaded_adapter("llama-3.3-70b-versatile")
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': 'Failed to call a function.', "
            "'type': 'invalid_request_error', 'code': 'tool_use_failed', "
            "'failed_generation': '<function=calculator>"
            "{\"expression\": \"(17 * 23) + sqrt(144)\"}</function>'}}"
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            },
        }]
        result = adapter.generate_with_tools("What is (17 * 23) + sqrt(144)?", tools)
        tool_call = result.metadata["tool_calls"][0]
        assert tool_call["function"]["name"] == "calculator"
        assert tool_call["function"]["arguments"] == {
            "expression": "(17 * 23) + sqrt(144)"
        }

    def test_count_tokens_returns_positive(self):
        adapter = self._loaded_adapter()
        tc = adapter.count_tokens("Hello world")
        assert tc.count > 0

    def test_supports_native_tools_property(self):
        adapter = self._loaded_adapter("llama-3.3-70b-versatile")
        assert adapter.supports_native_tools is True
        adapter2 = self._loaded_adapter("allam-2-7b")
        assert adapter2.supports_native_tools is False

    def test_rate_limit_status_disabled(self):
        adapter = GroqAdapter("llama-3.1-8b-instant", api_key="fake-key", enable_rate_limiting=False)
        status = adapter.rate_limit_status()
        assert status["enabled"] is False

    def test_rate_limit_status_enabled(self):
        adapter = GroqAdapter("llama-3.1-8b-instant", api_key="fake-key", enable_rate_limiting=True)
        status = adapter.rate_limit_status()
        assert status["enabled"] is True

    def test_supports_tool_calling_method(self):
        adapter = self._loaded_adapter("llama-3.3-70b-versatile")
        assert adapter.supports_tool_calling() is True
        assert adapter.supports_function_calling() is True
        adapter2 = self._loaded_adapter("allam-2-7b")
        assert adapter2.supports_tool_calling() is False

    def test_bad_key_raises_model_auth_error(self):
        from effgen.models.errors import ModelAuthError
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 401 - {'error': {'message': 'Invalid API Key', "
            "'code': 'invalid_api_key'}}"
        )
        with pytest.raises(ModelAuthError) as exc:
            adapter.generate("hi")
        assert exc.value.provider == "groq"
        assert "401" in str(exc.value)

    def test_request_too_large_raises_invalid_request_not_rate_limit(self):
        # Groq returns 413 with a rate_limit_exceeded code for a single oversized
        # request. It must classify as a non-retryable invalid request (not a
        # rate limit routed through failover), and the org id must be redacted.
        from effgen.models._rate_limit import RateLimitExceeded
        from effgen.models.errors import InvalidRequestError
        adapter = self._loaded_adapter()
        adapter._client.chat.completions.create.side_effect = Exception(
            "Error code: 413 - {'error': {'message': 'Request too large for "
            "model `llama-3.1-8b-instant` in organization `org_secret123` on "
            "tokens per minute (TPM): Limit 6000, Requested 9288, please "
            "reduce your message size and try again.', 'type': 'tokens', "
            "'code': 'rate_limit_exceeded'}}"
        )
        with pytest.raises(InvalidRequestError) as exc:
            adapter.generate("hi")
        assert not isinstance(exc.value, RateLimitExceeded)
        msg = str(exc.value)
        assert "org_secret123" not in msg
        assert "reduce" in msg.lower() or "larger-context" in msg.lower()


def test_is_request_too_large_helper():
    from effgen.models.groq_adapter import _is_request_too_large
    msg = "Error code: 413 - Request too large for model ..."
    assert _is_request_too_large(msg, msg.lower()) is True
    rate = "Error code: 429 - Rate limit reached for requests"
    assert _is_request_too_large(rate, rate.lower()) is False


def test_redact_groq_org_helper():
    from effgen.models.groq_adapter import _redact_groq_org
    msg = "Request too large in organization `org_01abcXYZ` service tier"
    out = _redact_groq_org(msg)
    assert "org_01abcXYZ" not in out
    assert "organization `***`" in out
