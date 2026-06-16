"""Tests for preset registry and create_agent factory."""


import pytest

from effgen.presets.registry import (
    PresetConfig,
    create_agent,
    get_preset,
    list_presets,
)
from tests.fixtures.mock_models import MockModel

EXPECTED_PRESETS = {
    "math",
    "research",
    "coding",
    "general",
    "rag",
    "minimal",
    "media",
    "notify",
    "multimodal",
}


class TestListPresets:
    """Test list_presets()."""

    def test_returns_all_presets(self):
        presets = list_presets()
        assert set(presets.keys()) == EXPECTED_PRESETS

    def test_returns_descriptions(self):
        presets = list_presets()
        for _name, desc in presets.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestGetPreset:
    """Test get_preset()."""

    def test_valid_preset(self):
        cfg = get_preset("math")
        assert isinstance(cfg, PresetConfig)
        assert cfg.name == "math"

    def test_invalid_preset_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("invalid_preset_name")

    def test_math_has_calculator(self):
        cfg = get_preset("math")
        assert "calculator" in cfg.tool_names

    def test_minimal_has_no_tools(self):
        cfg = get_preset("minimal")
        assert cfg.tool_names == []


class TestCreateAgent:
    """Test create_agent() factory."""

    def _mock_model(self):
        return MockModel(responses=["Thought: done\nFinal Answer: ok"])

    def test_create_math_agent(self):
        model = self._mock_model()
        agent = create_agent("math", model)
        assert agent is not None
        assert agent.config.name == "math-agent"

    def test_create_minimal_agent(self):
        model = self._mock_model()
        agent = create_agent("minimal", model)
        assert agent.config.tools == []

    def test_invalid_preset_raises(self):
        model = self._mock_model()
        with pytest.raises(KeyError, match="Unknown preset"):
            create_agent("invalid_preset_name", model)

    def test_custom_agent_name(self):
        model = self._mock_model()
        agent = create_agent("minimal", model, agent_name="custom-name")
        assert agent.config.name == "custom-name"

    def test_custom_max_iterations(self):
        model = self._mock_model()
        agent = create_agent("minimal", model, max_iterations=99)
        assert agent.config.max_iterations == 99

    def test_create_all_presets(self):
        """Verify all presets can be created without error."""
        model = self._mock_model()
        for name in EXPECTED_PRESETS:
            agent = create_agent(name, model)
            assert agent is not None


class TestRagKnowledgeBase:
    """create_agent('rag', knowledge_base=...) must work or fail loudly."""

    def _mock_model(self):
        return MockModel(responses=["Thought: done\nFinal Answer: ok"])

    def _retrieval_tool(self, agent):
        return next(
            (t for t in agent.config.tools if t.metadata.name == "retrieval"),
            None,
        )

    def test_raw_text_list_is_indexed(self):
        """The obvious call — raw text strings — populates the index."""
        import asyncio

        agent = create_agent(
            "rag",
            self._mock_model(),
            knowledge_base=[
                "The Eiffel Tower is 330 meters tall and located in Paris.",
                "The Great Wall of China is over 21,000 kilometers long.",
            ],
        )
        rt = self._retrieval_tool(agent)
        assert rt is not None
        res = asyncio.run(
            rt.execute(operation="search", query="How tall is the Eiffel Tower?", top_k=2)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert results, "expected the inline knowledge base to be retrievable"
        assert any("330 meters" in r["content"] for r in results)

    def test_single_raw_string_is_indexed(self):
        import asyncio

        agent = create_agent(
            "rag",
            self._mock_model(),
            knowledge_base="The capital of Australia is Canberra.",
        )
        rt = self._retrieval_tool(agent)
        res = asyncio.run(
            rt.execute(operation="search", query="capital of Australia", top_k=1)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("Canberra" in r["content"] for r in results)

    def test_file_path_still_works(self, tmp_path):
        import asyncio

        p = tmp_path / "facts.txt"
        p.write_text("Mercury is the closest planet to the Sun.", encoding="utf-8")
        agent = create_agent(
            "rag", self._mock_model(), knowledge_base=str(p)
        )
        rt = self._retrieval_tool(agent)
        res = asyncio.run(
            rt.execute(operation="search", query="closest planet to the Sun", top_k=1)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("Mercury" in r["content"] for r in results)

    def test_empty_knowledge_base_fails_loud(self):
        """Blank entries must raise, never silently build an empty index."""
        with pytest.raises(ValueError, match="0 documents"):
            create_agent(
                "rag", self._mock_model(), knowledge_base=["", "   "]
            )

    def test_typod_path_fails_loud_not_indexed_literally(self):
        """A path-like entry that doesn't exist is a typo, not a document:
        it must raise, never index the literal path string as a 1-doc index."""
        with pytest.raises(ValueError, match="looks like a file"):
            create_agent(
                "rag", self._mock_model(),
                knowledge_base=["/nonexistent/path/xyz.txt"],
            )

    def test_single_word_inline_text_is_not_treated_as_path(self):
        """A bare word with no separator/extension stays inline (not a path)."""
        agent = create_agent(
            "rag", self._mock_model(), knowledge_base="Canberra"
        )
        rt = self._retrieval_tool(agent)
        assert rt is not None

    def test_empty_string_does_not_ingest_cwd(self):
        """A "" entry must not resolve to Path('.') and ingest the whole tree."""
        with pytest.raises(ValueError, match="0 documents"):
            create_agent("rag", self._mock_model(), knowledge_base="")

    def test_ingest_text_chunks_directly(self):
        from effgen.rag import DocumentIngester

        chunks = DocumentIngester(show_progress=False).ingest_text(
            "Hello world. This is an inline document."
        )
        assert chunks
        assert all(c.content.strip() for c in chunks)
        assert chunks[0].source.startswith("inline:")

    def test_ingest_text_blank_returns_empty(self):
        from effgen.rag import DocumentIngester

        assert DocumentIngester(show_progress=False).ingest_text("   ") == []
