"""``"engine:model_id"`` prefix parsing for local models.

``load_model("Qwen/Qwen2.5-7B-Instruct", engine="transformers")`` is the
documented way to pick a local engine, but a user reaching for the same
``"provider:model_id"`` shape already used for cloud models
(``"openai:gpt-5-nano"``) would naturally try
``"transformers:Qwen/Qwen2.5-7B-Instruct"``. That string used to be passed
whole to the HuggingFace repo-id validator, which rejects the embedded colon
with a cryptic ``HFValidationError`` instead of running the model locally.
These tests pin the fix offline (the heavy engine constructors are stubbed
out; only the routing/parsing is under test).
"""

from unittest.mock import MagicMock

from effgen.models.model_loader import ModelLoader


def _stubbed_loader() -> ModelLoader:
    loader = ModelLoader()
    loader._load_with_transformers = MagicMock(return_value=MagicMock(name="transformers-model"))
    loader._load_with_vllm = MagicMock(return_value=MagicMock(name="vllm-model"))
    loader._load_with_mlx = MagicMock(return_value=MagicMock(name="mlx-model"))
    loader._validate_model = MagicMock()
    return loader


class TestEnginePrefixParsing:
    def test_transformers_prefix_strips_and_routes(self):
        loader = _stubbed_loader()
        result = loader.load_model("transformers:Qwen/Qwen2.5-7B-Instruct")
        assert result is loader._load_with_transformers.return_value
        loader._load_with_transformers.assert_called_once()
        called_model_name = loader._load_with_transformers.call_args[0][0]
        assert called_model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_vllm_prefix_strips_and_routes(self):
        loader = _stubbed_loader()
        result = loader.load_model("vllm:Qwen/Qwen2.5-7B-Instruct")
        assert result is loader._load_with_vllm.return_value
        loader._load_with_vllm.assert_called_once()
        called_model_name = loader._load_with_vllm.call_args[0][0]
        assert called_model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_mlx_prefix_strips_and_routes(self):
        loader = _stubbed_loader()
        result = loader.load_model("mlx:mlx-community/Qwen2.5-7B-Instruct-4bit")
        assert result is loader._load_with_mlx.return_value
        loader._load_with_mlx.assert_called_once()
        called_model_name = loader._load_with_mlx.call_args[0][0]
        assert called_model_name == "mlx-community/Qwen2.5-7B-Instruct-4bit"

    def test_explicit_engine_kwarg_wins_over_prefix_mismatch(self):
        # force_engine set via the constructor (the engine= convenience
        # function) already wins; the prefix parser must not clobber it.
        loader = _stubbed_loader()
        loader.force_engine = "vllm"
        loader.load_model("transformers:Qwen/Qwen2.5-7B-Instruct")
        loader._load_with_vllm.assert_called_once()
        loader._load_with_transformers.assert_not_called()

    def test_gguf_prefix_strips_and_routes_to_gguf_engine(self):
        loader = ModelLoader()
        loader._validate_model = MagicMock()

        class _FakeGGUFEngine:
            def __init__(self, model_name, **kwargs):
                self.model_name = model_name

            def load(self):
                pass

        import effgen.models.gguf_engine as gguf_mod
        original = gguf_mod.GGUFEngine
        gguf_mod.GGUFEngine = _FakeGGUFEngine
        try:
            result = loader.load_model("gguf:/models/qwen.gguf")
        finally:
            gguf_mod.GGUFEngine = original
        assert isinstance(result, _FakeGGUFEngine)
        assert result.model_name == "/models/qwen.gguf"

    def test_cloud_provider_prefix_still_takes_precedence(self):
        # "openai:gpt-5-nano" must still route to the OpenAI adapter, not be
        # misread as a local-engine prefix (no overlap in practice, but this
        # pins that the two prefix checks don't interfere).
        loader = _stubbed_loader()
        loader._load_openai_model = MagicMock(return_value=MagicMock())
        loader.load_model("openai:gpt-5-nano")
        loader._load_openai_model.assert_called_once()
        called_model_name = loader._load_openai_model.call_args[0][0]
        assert called_model_name == "gpt-5-nano"
        loader._load_with_transformers.assert_not_called()

    def test_bare_hf_repo_id_unaffected(self):
        # No prefix at all -> ordinary HuggingFace routing, untouched.
        loader = _stubbed_loader()
        loader.load_model("Qwen/Qwen2.5-7B-Instruct")
        loader._load_with_transformers.assert_called_once()
        called_model_name = loader._load_with_transformers.call_args[0][0]
        assert called_model_name == "Qwen/Qwen2.5-7B-Instruct"
