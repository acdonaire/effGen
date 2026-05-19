"""Tests for PromptRegistry: discovery and search."""

from __future__ import annotations

import pytest

from effgen.prompts.library import LibraryPrompt
from effgen.prompts.library.registry import PromptRegistry


def dummy_template(topic="hello"):
    return f"Tell me about {topic}."


def make_prompt(**kwargs) -> LibraryPrompt:
    defaults = {
        "name": "test.dummy.v1",
        "domain": "test",
        "variant": "zero_shot",
        "description": "A test prompt",
        "template": dummy_template,
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
        },
        "fixture": {"topic": "AI"},
        "expected_shape": None,
        "tags": ["test"],
    }
    defaults.update(kwargs)
    return LibraryPrompt(**defaults)


class TestRegistryBasics:
    def test_empty_registry_has_zero_length(self):
        r = PromptRegistry()
        r._discovered = True  # skip auto-discovery
        assert len(r) == 0

    def test_register_and_get(self):
        r = PromptRegistry()
        r._discovered = True
        p = make_prompt()
        r.register(p)
        assert r.get("test.dummy.v1") is p

    def test_get_missing_raises_key_error(self):
        r = PromptRegistry()
        r._discovered = True
        with pytest.raises(KeyError):
            r.get("does.not.exist")

    def test_search_by_domain(self):
        r = PromptRegistry()
        r._discovered = True
        p1 = make_prompt(name="a.dummy.v1", domain="alpha")
        p2 = make_prompt(name="b.dummy.v1", domain="beta")
        r.register(p1)
        r.register(p2)
        results = r.search(domain="alpha")
        assert len(results) == 1
        assert results[0].name == "a.dummy.v1"

    def test_search_by_variant(self):
        r = PromptRegistry()
        r._discovered = True
        p1 = make_prompt(name="c.cot.v1", domain="c", variant="cot")
        p2 = make_prompt(name="c.zs.v1", domain="c", variant="zero_shot")
        r.register(p1)
        r.register(p2)
        results = r.search(variant="cot")
        assert len(results) == 1
        assert results[0].name == "c.cot.v1"

    def test_search_by_tags(self):
        r = PromptRegistry()
        r._discovered = True
        p = make_prompt(name="d.dummy.v1", domain="d", tags=["nlp", "summarize"])
        r.register(p)
        assert r.search(tags=["nlp"]) == [p]
        assert r.search(tags=["nlp", "summarize"]) == [p]
        assert r.search(tags=["missing_tag"]) == []

    def test_all_returns_sorted(self):
        r = PromptRegistry()
        r._discovered = True
        r.register(make_prompt(name="z.dummy.v1", domain="z"))
        r.register(make_prompt(name="a.dummy.v1", domain="a"))
        names = [p.name for p in r.all()]
        assert names == sorted(names)

    def test_domains_returns_unique_sorted(self):
        r = PromptRegistry()
        r._discovered = True
        r.register(make_prompt(name="x.a.v1", domain="x"))
        r.register(make_prompt(name="x.b.v1", domain="x"))
        r.register(make_prompt(name="y.a.v1", domain="y"))
        assert r.domains() == ["x", "y"]

    def test_overwrite_warns(self, caplog):
        import logging
        r = PromptRegistry()
        r._discovered = True
        p1 = make_prompt()
        p2 = make_prompt()
        r.register(p1)
        with caplog.at_level(logging.WARNING):
            r.register(p2)
        assert any("Overwriting" in m for m in caplog.messages)

    def test_register_rejects_invalid_input_schema(self):
        r = PromptRegistry()
        r._discovered = True
        p = make_prompt(input_schema={"type": "definitely-not-a-json-schema-type"})
        with pytest.raises(ValueError, match="Invalid input_schema"):
            r.register(p)

    def test_register_rejects_fixture_that_does_not_match_schema(self):
        r = PromptRegistry()
        r._discovered = True
        p = make_prompt(
            input_schema={
                "type": "object",
                "required": ["topic"],
                "properties": {"topic": {"type": "string"}},
            },
            fixture={"topic": 42},
        )
        with pytest.raises(ValueError, match="does not match input_schema"):
            r.register(p)


class TestGlobalRegistry:
    def test_global_registry_is_prompt_registry(self):
        from effgen.prompts.library.registry import (
            PromptRegistry,
        )
        from effgen.prompts.library.registry import (
            registry as global_reg,
        )

        assert isinstance(global_reg, PromptRegistry)

    def test_global_registry_imports_cleanly(self):
        from effgen.prompts.library import registry as r
        # On an empty library, search returns empty list cleanly
        results = r.search(domain="__nonexistent__")
        assert isinstance(results, list)
        assert len(results) == 0
