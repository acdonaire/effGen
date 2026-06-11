"""
Domain base class for effGen.

A Domain bundles seed keywords, tools, system prompts, and guardrails
for a particular knowledge area. The ``expand_keywords`` method
delegates to :class:`~effgen.domains.expander.KeywordExpander` to
grow N seed keywords into 10N+ expanded terms.

Usage:
    from effgen.domains.base import Domain

    domain = Domain(
        name="tech",
        keywords=["Python", "machine learning"],
        system_prompt="You are a technology expert.",
    )
    expanded = domain.expand_keywords(factor=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Domain:
    """Configurable knowledge domain.

    Attributes:
        name: Domain identifier.
        keywords: Seed keywords for this domain.
        description: Human-readable description of the domain.
        system_prompt: System prompt tailored to the domain.
        tool_names: Names of tools relevant to this domain.
        guardrails: Optional guardrail preset name or GuardrailChain.
        templates: Query templates used by the template-based expander.
        metadata: Arbitrary extra metadata.
    """

    name: str
    keywords: list[str] = field(default_factory=list)
    description: str = ""
    system_prompt: str = "You are a helpful AI assistant."
    tool_names: list[str] = field(default_factory=list)
    guardrails: Any = None
    templates: list[str] = field(default_factory=lambda: [
        "best {kw}",
        "{kw} tutorial",
        "{kw} vs {alt}",
        "how to {kw}",
        "{kw} examples",
        "{kw} guide",
        "what is {kw}",
        "{kw} for beginners",
    ])
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Keyword expansion
    # ------------------------------------------------------------------

    def expand_keywords(
        self,
        factor: int = 10,
        *,
        use_wordnet: bool = False,
        use_templates: bool = True,
        use_llm: bool = False,
        model: Any = None,
    ) -> list[str]:
        """Expand seed keywords by the given factor.

        Combines multiple strategies (templates, optional LLM-based
        generation, optional WordNet synonyms) and deduplicates.

        Args:
            factor: Target multiplier — aim for ``len(keywords) * factor``
                expanded terms.
            use_wordnet: Enable WordNet synonym expansion (requires nltk).
                Off by default — it has no word-sense disambiguation and
                drifts off-domain for polysemous seeds.
            use_templates: Enable template-based expansion.
            use_llm: Enable LLM-based expansion (requires *model*).
            model: An effgen BaseModel instance for LLM expansion.

        Returns:
            Deduplicated list of expanded keyword strings.
        """
        from .expander import KeywordExpander

        expander = KeywordExpander(
            use_wordnet=use_wordnet,
            use_templates=use_templates,
            use_llm=use_llm,
            model=model,
            templates=self.templates,
        )
        expanded = expander.expand(self.keywords, factor=factor)
        logger.info(
            "Domain '%s': expanded %d seed keywords to %d terms (factor=%d)",
            self.name, len(self.keywords), len(expanded), factor,
        )
        return expanded

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "keywords": self.keywords,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tool_names": self.tool_names,
        }
