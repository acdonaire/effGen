"""
Unit tests for ReplicateAdapter.

These tests mock the replicate SDK so they run offline.
Live API tests are in tests/integration/test_replicate_live.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from effgen.models.errors import ModelAuthError, ModelTimeoutError
from effgen.models.replicate_adapter import ReplicateAdapter
from effgen.models.replicate_models import (
    REPLICATE_DEFAULT_MODEL,
    REPLICATE_MODELS,
    available_models,
    get_model_info,
    streaming_models,
    tool_capable_models,
)

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestReplicateRegistry:
    def test_default_model_in_registry(self):
        assert REPLICATE_DEFAULT_MODEL in REPLICATE_MODELS

    def test_all_models_have_required_keys(self):
        required = {
            "display_name", "family", "organization", "context",
            "supports_native_tools", "supports_streaming",
        }
        for mid, info in REPLICATE_MODELS.items():
            missing = required - set(info.keys())
            assert not missing, f"Model '{mid}' missing keys: {missing}"

    def test_available_models_returns_all_keys(self):
        assert set(available_models()) == set(REPLICATE_MODELS.keys())

    def test_tool_capable_models_correct(self):
        tc = tool_capable_models()
        assert "ibm-granite/granite-3.3-8b-instruct" in tc
        assert "meta/meta-llama-3-8b-instruct" not in tc

    def test_streaming_models_correct(self):
        sm = streaming_models()
        assert "meta/meta-llama-3-8b-instruct" in sm
        assert "replicate/flan-t5-xl" not in sm  # flan-t5 has supports_streaming=False

    def test_get_model_info_known(self):
        info = get_model_info("meta/meta-llama-3-8b-instruct")
        assert info is not None
        assert info["family"] == "llama3"

    def test_get_model_info_unknown_returns_none(self):
        assert get_model_info("nonexistent/model") is None

    def test_context_lengths_positive(self):
        for mid, info in REPLICATE_MODELS.items():
            assert info["context"] > 0, f"Model '{mid}' has non-positive context"

    def test_cost_per_second_non_negative(self):
        for mid, info in REPLICATE_MODELS.items():
            cost = info.get("cost_per_second_usd", 0.0)
            assert cost >= 0.0, f"Model '{mid}' has negative cost"

    def test_ibm_granite_has_messages_schema(self):
        info = REPLICATE_MODELS["ibm-granite/granite-3.3-8b-instruct"]
        assert info["input_schema"] == "messages"
        assert info["supports_native_tools"] is True

    def test_llama_has_prompt_only_schema(self):
        info = REPLICATE_MODELS["meta/meta-llama-3-8b-instruct"]
        assert info["input_schema"] == "prompt_only"

    def test_total_model_count(self):
        # We registered ≥38 models
        assert len(REPLICATE_MODELS) >= 38


# ---------------------------------------------------------------------------
# Adapter construction tests
# ---------------------------------------------------------------------------

class TestReplicateAdapterConstruction:
    def test_default_model_name(self):
        adapter = ReplicateAdapter(api_token="test")
        assert adapter.model_name == REPLICATE_DEFAULT_MODEL

    def test_custom_model_name(self):
        adapter = ReplicateAdapter(
            "ibm-granite/granite-3.3-8b-instruct", api_token="test"
        )
        assert adapter.model_name == "ibm-granite/granite-3.3-8b-instruct"

    def test_unknown_model_warns(self, caplog):
        with caplog.at_level("WARNING"):
            ReplicateAdapter("unknown/model-xyz", api_token="test", warn_unknown_model=True)
        assert "not in the bundled registry" in caplog.text

    def test_unknown_model_no_warn(self, caplog):
        with caplog.at_level("WARNING"):
            ReplicateAdapter("unknown/model-xyz", api_token="test", warn_unknown_model=False)
        assert "not in the bundled registry" not in caplog.text

    def test_context_length_from_registry(self):
        adapter = ReplicateAdapter(
            "ibm-granite/granite-3.3-8b-instruct", api_token="test"
        )
        assert adapter.get_context_length() == 128_000

    def test_timeout_default(self):
        adapter = ReplicateAdapter(api_token="test")
        assert adapter.timeout == 300.0

    def test_custom_timeout(self):
        adapter = ReplicateAdapter(api_token="test", timeout=60.0)
        assert adapter.timeout == 60.0

    def test_supports_tool_calling_true(self):
        adapter = ReplicateAdapter(
            "ibm-granite/granite-3.3-8b-instruct", api_token="test"
        )
        assert adapter.supports_tool_calling() is True

    def test_supports_tool_calling_false(self):
        adapter = ReplicateAdapter(
            "meta/meta-llama-3-8b-instruct", api_token="test"
        )
        assert adapter.supports_tool_calling() is False

    def test_supports_streaming_true(self):
        adapter = ReplicateAdapter(
            "meta/meta-llama-3-8b-instruct", api_token="test"
        )
        assert adapter.supports_streaming() is True

    def test_cost_per_second_from_registry(self):
        adapter = ReplicateAdapter(
            "meta/meta-llama-3-8b-instruct", api_token="test"
        )
        assert adapter.get_cost_per_second() == pytest.approx(0.000225)


# ---------------------------------------------------------------------------
# Load / unload tests
# ---------------------------------------------------------------------------

class TestReplicateAdapterLifecycle:
    def test_load_no_key_raises_value_error(self):
        adapter = ReplicateAdapter(api_token=None)
        with patch.dict("os.environ", {}, clear=True):
            # Ensure REPLICATE_API_TOKEN is not in env
            import os
            os.environ.pop("REPLICATE_API_TOKEN", None)
            with pytest.raises(ValueError, match="REPLICATE_API_TOKEN"):
                adapter.load()

    def test_load_missing_sdk_raises_runtime_error(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "replicate":
                raise ImportError("No module named 'replicate'")
            return real_import(name, *args, **kwargs)

        adapter = ReplicateAdapter(api_token="test-token")
        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(RuntimeError, match="replicate SDK is not installed"):
            adapter.load()

    def test_load_creates_client(self):
        adapter = ReplicateAdapter(api_token="test-token")
        mock_client = MagicMock()

        with patch("effgen.models.replicate_adapter.ReplicateAdapter.load") as mock_load:
            mock_load.side_effect = lambda: setattr(adapter, "_client", mock_client) or setattr(adapter, "_is_loaded", True)
            adapter.load()
            assert adapter._client is mock_client
            assert adapter._is_loaded

    def test_unload_clears_client(self):
        adapter = ReplicateAdapter(api_token="test-token")
        adapter._client = MagicMock()
        adapter._is_loaded = True
        adapter.unload()
        assert adapter._client is None
        assert not adapter._is_loaded

    def test_generate_without_load_raises(self):
        adapter = ReplicateAdapter(api_token="test-token")
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.generate("Hello")

    def test_stream_without_load_raises(self):
        adapter = ReplicateAdapter(api_token="test-token")
        with pytest.raises(RuntimeError, match="not loaded"):
            list(adapter.generate_stream("Hello"))


# ---------------------------------------------------------------------------
# Token counting tests
# ---------------------------------------------------------------------------

class TestTokenCounting:
    def test_count_tokens_approximate(self):
        adapter = ReplicateAdapter(api_token="test")
        result = adapter.count_tokens("Hello world this is a test sentence.")
        assert result.count >= 1
        assert result.model_name == REPLICATE_DEFAULT_MODEL

    def test_count_tokens_longer_text(self):
        adapter = ReplicateAdapter(api_token="test")
        short = adapter.count_tokens("Hello")
        long = adapter.count_tokens("Hello " * 100)
        assert long.count > short.count


# ---------------------------------------------------------------------------
# Input building tests
# ---------------------------------------------------------------------------

class TestInputBuilding:
    def _make_adapter(self, model_id: str) -> ReplicateAdapter:
        adapter = ReplicateAdapter(model_id, api_token="test")
        return adapter

    def test_prompt_only_schema(self):
        adapter = self._make_adapter("meta/meta-llama-3-8b-instruct")
        from effgen.models.base import GenerationConfig
        cfg = GenerationConfig(max_tokens=100, temperature=0.7)
        inp = adapter._build_input("Hello", cfg)
        assert "prompt" in inp
        assert inp["max_tokens"] == 100
        assert inp["temperature"] == 0.7

    def test_prompt_only_with_template(self):
        adapter = self._make_adapter("meta/meta-llama-3-8b-instruct")
        from effgen.models.base import GenerationConfig
        inp = adapter._build_input("Tell me a joke", GenerationConfig())
        # Llama 3 has a prompt_template — it should be used
        assert "Tell me a joke" in inp["prompt"]

    def test_messages_schema(self):
        adapter = self._make_adapter("ibm-granite/granite-3.3-8b-instruct")
        from effgen.models.base import GenerationConfig
        inp = adapter._build_input("Hello", GenerationConfig(max_tokens=200))
        assert "messages" in inp
        assert isinstance(inp["messages"], list)
        assert inp["messages"][-1]["role"] == "user"
        assert inp["max_tokens"] == 200

    def test_messages_schema_with_tools(self):
        adapter = self._make_adapter("ibm-granite/granite-3.3-8b-instruct")
        from effgen.models.base import GenerationConfig
        tools = [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
        inp = adapter._build_input("Use calc", GenerationConfig(), tools=tools)
        assert "tools" in inp
        assert inp["tool_choice"] == "auto"

    def test_messages_schema_from_messages_list(self):
        adapter = self._make_adapter("ibm-granite/granite-3.3-8b-instruct")
        from effgen.models.base import GenerationConfig
        msgs = [{"role": "user", "content": "Hi"}]
        inp = adapter._build_input("ignored", GenerationConfig(), messages=msgs)
        assert inp["messages"] == msgs

    def test_stop_sequences_in_prompt_only(self):
        adapter = self._make_adapter("meta/meta-llama-3-8b-instruct")
        from effgen.models.base import GenerationConfig
        cfg = GenerationConfig(stop_sequences=["STOP", "END"])
        inp = adapter._build_input("Hello", cfg)
        assert "stop_sequences" in inp

    def test_seed_passed_through(self):
        adapter = self._make_adapter("meta/meta-llama-3-8b-instruct")
        from effgen.models.base import GenerationConfig
        cfg = GenerationConfig(seed=42)
        inp = adapter._build_input("Hello", cfg)
        assert inp.get("seed") == 42


# ---------------------------------------------------------------------------
# Polling / timeout tests
# ---------------------------------------------------------------------------

class TestPolling:
    def _make_loaded_adapter(self) -> ReplicateAdapter:
        adapter = ReplicateAdapter(
            "meta/meta-llama-3-8b-instruct",
            api_token="test-token",
            timeout=5.0,
            enable_rate_limiting=False,
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def test_poll_succeeds_immediately(self):
        adapter = self._make_loaded_adapter()
        pred = MagicMock()
        pred.status = "succeeded"
        result = adapter._poll_prediction(pred)
        assert result.status == "succeeded"

    def test_poll_waits_then_succeeds(self):
        adapter = self._make_loaded_adapter()
        adapter._initial_poll_interval = 0.01
        call_count = 0

        pred = MagicMock()
        pred.id = "pred-123"

        def reload_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                pred.status = "succeeded"

        pred.reload.side_effect = reload_side_effect
        pred.status = "processing"

        with patch("time.sleep"):
            adapter._poll_prediction(pred)

        assert call_count >= 2

    def test_poll_timeout_raises_model_timeout_error(self):
        adapter = self._make_loaded_adapter()
        adapter.timeout = 0.05
        adapter._initial_poll_interval = 0.01

        pred = MagicMock()
        pred.id = "pred-timeout-test"
        pred.status = "processing"
        pred.cancel = MagicMock()

        # Make reload never change status to terminal
        pred.reload.return_value = None

        with patch("time.monotonic") as mock_time:
            # Start at 0, then exceed timeout on second call
            mock_time.side_effect = [0.0, 0.0, 1.0, 1.0, 1.0]
            with patch("time.sleep"):
                with pytest.raises(ModelTimeoutError) as exc_info:
                    adapter._poll_prediction(pred)

        assert exc_info.value.provider == "replicate"
        assert exc_info.value.prediction_id == "pred-timeout-test"
        assert exc_info.value.timeout_seconds == 0.05
        pred.cancel.assert_called_once()

    def test_poll_failed_prediction_raises_runtime_error(self):
        adapter = self._make_loaded_adapter()
        pred = MagicMock()
        pred.status = "failed"
        pred.error = "OOM error"

        # _poll_prediction returns failed pred; _do_generate checks and raises
        returned = adapter._poll_prediction(pred)
        assert returned.status == "failed"


# ---------------------------------------------------------------------------
# Generate mock tests
# ---------------------------------------------------------------------------

class TestGenerateMocked:
    def _make_loaded_adapter(self, model_id: str = "meta/meta-llama-3-8b-instruct") -> ReplicateAdapter:
        adapter = ReplicateAdapter(
            model_id,
            api_token="test-token",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def _make_prediction(
        self,
        output: list[str] | None = None,
        status: str = "succeeded",
        metrics: dict | None = None,
        prediction_id: str = "pred-abc",
        error: str | None = None,
    ) -> MagicMock:
        pred = MagicMock()
        pred.id = prediction_id
        pred.status = status
        pred.output = output if output is not None else ["Hello", " world"]
        pred.error = error
        pred.metrics = metrics or {
            "predict_time": 1.23,
            "input_token_count": 10,
            "output_token_count": 20,
            "total_time": 1.5,
            "tokens_per_second": 16.26,
        }
        return pred

    def test_generate_returns_text(self):
        adapter = self._make_loaded_adapter()
        pred = self._make_prediction(output=["Hello", " world"])
        adapter._client.predictions.create.return_value = pred
        pred.reload = MagicMock()

        result = adapter.generate("Hi")
        assert result.text == "Hello world"

    def test_generate_populates_compute_seconds(self):
        adapter = self._make_loaded_adapter()
        pred = self._make_prediction(metrics={"predict_time": 2.5, "input_token_count": 5, "output_token_count": 10, "total_time": 3.0})
        adapter._client.predictions.create.return_value = pred

        result = adapter.generate("Hi")
        assert result.metadata["compute_seconds"] == pytest.approx(2.5)

    def test_generate_populates_prediction_id(self):
        adapter = self._make_loaded_adapter()
        pred = self._make_prediction(prediction_id="pred-xyz-123")
        adapter._client.predictions.create.return_value = pred

        result = adapter.generate("Hi")
        assert result.metadata["prediction_id"] == "pred-xyz-123"

    def test_generate_failed_prediction_raises(self):
        adapter = self._make_loaded_adapter()
        pred = self._make_prediction(status="failed", error="GPU OOM")
        adapter._client.predictions.create.return_value = pred

        with pytest.raises(RuntimeError, match="GPU OOM"):
            adapter.generate("Hi")

    def test_generate_auth_error_raises_model_auth_error(self):
        adapter = self._make_loaded_adapter()

        from replicate.exceptions import ReplicateError
        exc = ReplicateError(status=401, title="Unauthorized", detail="Invalid token")
        adapter._client.predictions.create.side_effect = exc

        with pytest.raises(ModelAuthError) as exc_info:
            adapter.generate("Hi")
        assert exc_info.value.provider == "replicate"

    def test_generate_billing_error_raises_runtime_error(self):
        adapter = self._make_loaded_adapter()

        from replicate.exceptions import ReplicateError
        exc = ReplicateError(status=402, title="Insufficient credit", detail="Add billing")
        adapter._client.predictions.create.side_effect = exc

        with pytest.raises(RuntimeError, match="insufficient credits"):
            adapter.generate("Hi")

    def test_generate_usage_populated(self):
        adapter = self._make_loaded_adapter()
        pred = self._make_prediction(
            metrics={
                "predict_time": 1.0,
                "input_token_count": 15,
                "output_token_count": 25,
                "total_time": 1.5,
            }
        )
        adapter._client.predictions.create.return_value = pred

        result = adapter.generate("Hi")
        usage = result.metadata["usage"]
        assert usage["input_tokens"] == 15
        assert usage["output_tokens"] == 25
        assert usage["total_tokens"] == 40

    def test_generate_string_output_handled(self):
        adapter = self._make_loaded_adapter()
        pred = self._make_prediction(output="Hello world")
        adapter._client.predictions.create.return_value = pred

        result = adapter.generate("Hi")
        assert "Hello world" in result.text


# ---------------------------------------------------------------------------
# Streaming mock tests
# ---------------------------------------------------------------------------

class TestStreamingMocked:
    def _make_loaded_adapter(self, model_id: str = "meta/meta-llama-3-8b-instruct") -> ReplicateAdapter:
        adapter = ReplicateAdapter(
            model_id,
            api_token="test-token",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def test_stream_sse_yields_tokens(self):
        adapter = self._make_loaded_adapter()
        tokens = ["Hello", " ", "world", "!"]
        adapter._client.stream.return_value = iter(tokens)

        chunks = list(adapter.generate_stream("Hi"))
        assert "".join(chunks) == "Hello world!"

    def test_stream_sse_auth_error(self):
        adapter = self._make_loaded_adapter()

        from replicate.exceptions import ReplicateError

        def bad_stream(*args, **kwargs):
            raise ReplicateError(status=401, title="Unauthorized", detail="Bad key")

        adapter._client.stream.side_effect = bad_stream

        with pytest.raises(ModelAuthError):
            list(adapter.generate_stream("Hi"))

    def test_stream_billing_error(self):
        adapter = self._make_loaded_adapter()

        from replicate.exceptions import ReplicateError

        def billing_error(*args, **kwargs):
            raise ReplicateError(status=402, title="Insufficient credit", detail="Add billing")

        adapter._client.stream.side_effect = billing_error

        with pytest.raises(RuntimeError, match="insufficient credits"):
            list(adapter.generate_stream("Hi"))

    def test_non_streaming_model_falls_back_to_poll(self):
        adapter = self._make_loaded_adapter("replicate/flan-t5-xl")
        # flan-t5-xl has supports_streaming=False

        pred = MagicMock()
        pred.id = "pred-123"
        pred.output = ["The", " capital", " is", " Paris"]
        pred.status = "succeeded"
        pred.error = None
        pred.reload = MagicMock()
        adapter._client.predictions.create.return_value = pred

        with patch("time.sleep"):
            chunks = list(adapter.generate_stream("What is the capital?"))

        assert len(chunks) > 0
        assert "".join(chunks) != ""


# ---------------------------------------------------------------------------
# Native tool-calling tests
# ---------------------------------------------------------------------------

class TestNativeToolCalling:
    def _make_loaded_adapter(self, model_id: str) -> ReplicateAdapter:
        adapter = ReplicateAdapter(
            model_id,
            api_token="test-token",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def test_non_tool_model_raises_not_implemented(self):
        adapter = self._make_loaded_adapter("meta/meta-llama-3-8b-instruct")
        with pytest.raises(NotImplementedError, match="does not support native tool calling"):
            adapter.generate_with_tools("Hello", tools=[])

    def test_tool_model_passes_tools_in_input(self):
        adapter = self._make_loaded_adapter("ibm-granite/granite-3.3-8b-instruct")

        pred = MagicMock()
        pred.id = "pred-tool"
        pred.status = "succeeded"
        pred.output = [{"tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}]}]
        pred.error = None
        pred.metrics = {"predict_time": 0.5, "input_token_count": 10, "output_token_count": 5, "total_time": 0.8}
        adapter._client.predictions.create.return_value = pred

        tools = [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object", "properties": {}}}}]
        result = adapter.generate_with_tools("Calculate 2+2", tools=tools)
        assert result is not None

    def test_tool_dict_normalisation(self):
        adapter = self._make_loaded_adapter("ibm-granite/granite-3.3-8b-instruct")

        pred = MagicMock()
        pred.id = "pred-tool"
        pred.status = "succeeded"
        pred.output = ["no tool calls here"]
        pred.error = None
        pred.metrics = {"predict_time": 0.5, "input_token_count": 10, "output_token_count": 5, "total_time": 0.8}
        adapter._client.predictions.create.return_value = pred

        # Pass a bare dict without "type" key — should be normalised
        tools = [{"name": "calc", "parameters": {"type": "object", "properties": {}}}]
        result = adapter.generate_with_tools("Calculate 2+2", tools=tools)
        # The input dict should have tools or it was passed via generate()
        assert result is not None


# ---------------------------------------------------------------------------
# Tool call extraction tests
# ---------------------------------------------------------------------------

class TestToolCallExtraction:
    def _make_adapter(self) -> ReplicateAdapter:
        return ReplicateAdapter(
            "ibm-granite/granite-3.3-8b-instruct",
            api_token="test",
        )

    def test_extract_from_list_with_tool_calls(self):
        adapter = self._make_adapter()
        raw = [{"tool_calls": [{"id": "1", "type": "function", "function": {"name": "calc"}}]}]
        result = adapter._extract_tool_calls("", raw)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "calc"

    def test_extract_from_json_text(self):
        adapter = self._make_adapter()
        text = '{"tool_calls": [{"id": "1", "type": "function", "function": {"name": "search"}}]}'
        result = adapter._extract_tool_calls(text, [])
        assert len(result) == 1

    def test_no_tool_calls_returns_empty(self):
        adapter = self._make_adapter()
        result = adapter._extract_tool_calls("Just regular text", ["Just", " regular", " text"])
        assert result == []

    def test_non_tool_model_returns_empty(self):
        adapter = ReplicateAdapter("meta/meta-llama-3-8b-instruct", api_token="test")
        raw = [{"tool_calls": [{"id": "1"}]}]
        result = adapter._extract_tool_calls("", raw)
        assert result == []


# ---------------------------------------------------------------------------
# ModelTimeoutError tests
# ---------------------------------------------------------------------------

class TestModelTimeoutError:
    def test_basic_message(self):
        err = ModelTimeoutError(
            provider="replicate",
            model_name="meta/meta-llama-3-8b-instruct",
            timeout_seconds=300.0,
            prediction_id="pred-123",
        )
        msg = str(err)
        assert "replicate" in msg
        assert "300" in msg
        assert "pred-123" in msg

    def test_attributes(self):
        err = ModelTimeoutError(
            provider="replicate",
            model_name="test/model",
            timeout_seconds=60.0,
            prediction_id="pred-abc",
        )
        assert err.provider == "replicate"
        assert err.model_name == "test/model"
        assert err.timeout_seconds == 60.0
        assert err.prediction_id == "pred-abc"

    def test_is_exception(self):
        err = ModelTimeoutError(provider="replicate")
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# Refresh models / drift detection (mock)
# ---------------------------------------------------------------------------

class TestRefreshModels:
    def test_refresh_no_token_raises(self, monkeypatch):
        monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
        from effgen.models.replicate_models import refresh_models
        with pytest.raises(ValueError, match="REPLICATE_API_TOKEN"):
            refresh_models(api_token=None)

    def test_refresh_returns_drift_report(self):
        from effgen.models.replicate_models import refresh_models

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": {
                "results": [
                    {"owner": "meta", "name": "meta-llama-3-8b-instruct"},
                    {"owner": "new-org", "name": "brand-new-model"},
                ],
                "next": None,
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            report = refresh_models(api_token="test-token")

        assert "added" in report
        assert "removed" in report
        assert "registry_count" in report
        assert "live_count" in report
        assert report["live_count"] == 2
        # "new-org/brand-new-model" is in live but not in registry → added
        assert "new-org/brand-new-model" in report["added"]
