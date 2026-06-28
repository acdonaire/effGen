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

    def test_unknown_preset_reported_before_missing_model(self):
        """create_agent('legal') (an invalid preset) must report the bad preset
        up front, not first demand a model for a preset that can't exist."""
        with pytest.raises(KeyError, match="Unknown preset"):
            create_agent("legal")  # no model passed on purpose

    def test_valid_preset_no_model_still_asks_for_model(self):
        with pytest.raises(ValueError, match="needs a model"):
            create_agent("math")

    def test_no_preset_and_no_domain_raises(self):
        with pytest.raises(TypeError, match="preset name"):
            create_agent(model=self._mock_model())


class TestDomainAgentBridge:
    """A Domain must have one obvious on-ramp to a runnable agent (E11-1)."""

    def _mock_model(self):
        return MockModel(responses=["Thought: done\nFinal Answer: ok"])

    def test_domain_to_agent_wires_prompt_tools_guardrails(self):
        from effgen.domains import LegalDomain

        agent = LegalDomain().to_agent(self._mock_model())
        assert agent.config.name == "legal-agent"
        assert agent.config.system_prompt.startswith("You are a legal")
        names = [t.metadata.name for t in agent.config.tools]
        assert "web_search" in names and "wikipedia" in names
        # The domain's guardrails ("standard") get their first real consumer.
        assert agent._guardrail_chain is not None

    def test_create_agent_domain_kwarg_equivalent(self):
        from effgen.domains import LegalDomain
        from effgen.presets import create_agent as ca

        d = LegalDomain()
        agent = ca(domain=d, model=self._mock_model())
        assert agent.config.system_prompt == d.system_prompt

    def test_domain_without_guardrails_has_no_chain(self):
        from effgen.domains import TechDomain

        agent = TechDomain().to_agent(self._mock_model())
        assert agent._guardrail_chain is None

    def test_to_agent_overrides_apply(self):
        from effgen.domains import LegalDomain

        agent = LegalDomain().to_agent(
            self._mock_model(), temperature=0.1, max_iterations=3,
            system_prompt="custom",
        )
        assert agent.config.temperature == 0.1
        assert agent.config.max_iterations == 3
        assert agent.config.system_prompt == "custom"

    def test_to_agent_extra_tools_by_name(self):
        from effgen.domains import LegalDomain

        agent = LegalDomain().to_agent(self._mock_model(), extra_tools=["calculator"])
        names = [t.metadata.name for t in agent.config.tools]
        assert "calculator" in names

    def test_preset_and_domain_together_raises(self):
        from effgen.domains import LegalDomain
        from effgen.presets import create_agent as ca

        with pytest.raises(TypeError, match="both"):
            ca("math", self._mock_model(), domain=LegalDomain())

    def test_domain_kwarg_rejects_non_domain(self):
        from effgen.presets import create_agent as ca

        with pytest.raises(TypeError, match="Domain instance"):
            ca(domain="legal", model=self._mock_model())

    def test_domain_no_model_raises_domain_aware(self):
        from effgen.domains import LegalDomain

        with pytest.raises(ValueError, match="to_agent"):
            LegalDomain().to_agent()


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

    def test_pdf_knowledge_base_is_indexed(self, tmp_path):
        """A PDF knowledge base ingests out of the box (pypdf/pdfplumber),
        not only when pymupdf happens to be installed."""
        import asyncio

        pytest.importorskip("reportlab")
        pytest.importorskip("pypdf")
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = tmp_path / "facts.pdf"
        c = canvas.Canvas(str(pdf), pagesize=letter)
        c.drawString(72, 720, "Neptune is the farthest planet from the Sun.")
        c.save()

        agent = create_agent("rag", self._mock_model(), knowledge_base=str(pdf))
        rt = self._retrieval_tool(agent)
        res = asyncio.run(
            rt.execute(operation="search", query="farthest planet", top_k=1)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("Neptune" in r["content"] for r in results)

    def test_unparseable_file_surfaces_skip_reason(self, tmp_path):
        """When nothing indexes, the error names *why* each file was skipped
        instead of a bare "0 documents"."""
        pytest.importorskip("pypdf")
        bad = tmp_path / "broken.pdf"
        bad.write_text("not a real pdf")
        with pytest.raises(ValueError, match="Skipped:.*broken.pdf"):
            create_agent("rag", self._mock_model(), knowledge_base=str(bad))

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

    def test_prebuilt_vector_memory_store_is_indexed(self):
        """A pre-built VectorMemoryStore connects straight to a RAG agent —
        the memory/ and rag/ retrieval subsystems link up."""
        import asyncio

        from effgen.memory import VectorMemoryStore

        store = VectorMemoryStore()
        store.add(
            "The Eiffel Tower is 330 meters tall and located in Paris.",
            metadata={"topic": "landmark"},
        )
        store.add("The Great Wall of China is over 21,000 kilometers long.")

        agent = create_agent("rag", self._mock_model(), knowledge_base=store)
        rt = self._retrieval_tool(agent)
        assert rt is not None
        res = asyncio.run(
            rt.execute(operation="search", query="How tall is the Eiffel Tower?", top_k=2)
        )
        assert res.success
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("330 meters" in r["content"] for r in results)

    def test_vector_store_mixed_with_text(self):
        """A VectorMemoryStore can be mixed with raw text in one list."""
        import asyncio

        from effgen.memory import VectorMemoryStore

        store = VectorMemoryStore()
        store.add("Mercury is the closest planet to the Sun.")
        agent = create_agent(
            "rag",
            self._mock_model(),
            knowledge_base=[store, "Neptune is the farthest planet from the Sun."],
        )
        rt = self._retrieval_tool(agent)
        res = asyncio.run(
            rt.execute(operation="search", query="closest planet", top_k=2)
        )
        results = res.output["results"] if isinstance(res.output, dict) else []
        assert any("Mercury" in r["content"] for r in results)

    def test_empty_vector_store_fails_loud(self):
        """An empty VectorMemoryStore must raise, never build an empty index."""
        from effgen.memory import VectorMemoryStore

        with pytest.raises(ValueError, match="0 documents"):
            create_agent("rag", self._mock_model(), knowledge_base=VectorMemoryStore())
