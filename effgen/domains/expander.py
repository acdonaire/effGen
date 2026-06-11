"""
Keyword Expander for effGen domains.

Expands N seed keywords to 10N+ using three strategies:
1. **Template-based** — deterministic pattern expansion (on by default).
2. **LLM-based** — uses the agent's own model to generate related terms.
3. **WordNet synonyms** — free, offline (requires ``nltk``; **opt-in**).

WordNet is off by default: it has no word-sense disambiguation, so polysemous
seeds drift off-domain (e.g. ``"python"`` expands toward snakes, not code). Turn
it on with ``use_wordnet=True`` when that trade-off is acceptable.

All enabled strategies are combined and deduplicated to produce the final list.

Usage:
    from effgen.domains.expander import KeywordExpander

    expander = KeywordExpander()
    expanded = expander.expand(["Python", "machine learning"], factor=10)
    # A bare string is treated as a single keyword (not iterated char-by-char):
    expanded = expander.expand("python")
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Default query templates — {kw} is the keyword, {alt} is another keyword
_DEFAULT_TEMPLATES: list[str] = [
    "best {kw}",
    "{kw} tutorial",
    "{kw} vs {alt}",
    "how to {kw}",
    "{kw} examples",
    "{kw} guide",
    "what is {kw}",
    "{kw} for beginners",
]


class KeywordExpander:
    """Expand seed keywords using multiple strategies.

    Strategies gracefully degrade:
    - Templates: always available; the sensible, on-domain default.
    - LLM: skipped if no model is provided.
    - WordNet: opt-in (``use_wordnet=True``); skipped with a note if ``nltk``
      is not installed.
    """

    def __init__(
        self,
        *,
        use_wordnet: bool = False,
        use_templates: bool = True,
        use_llm: bool = False,
        model: Any = None,
        templates: list[str] | None = None,
    ) -> None:
        self.use_wordnet = use_wordnet
        self.use_templates = use_templates
        self.use_llm = use_llm
        self.model = model
        self.templates = templates or _DEFAULT_TEMPLATES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_keywords(keywords: list[str] | str) -> list[str]:
        """Coerce input to a clean ``list[str]`` of seed keywords.

        A bare ``str`` is wrapped as a single keyword (the most common mistake,
        which previously char-iterated into nonsense). Other types raise a clear
        ``TypeError`` instead of silently producing garbage.
        """
        if isinstance(keywords, str):
            keywords = [keywords]
        elif isinstance(keywords, list | tuple | set):
            keywords = list(keywords)
        else:
            raise TypeError(
                "KeywordExpander.expand() expects a list of keyword strings "
                f"(or a single string), got {type(keywords).__name__}."
            )
        cleaned: list[str] = []
        for kw in keywords:
            if not isinstance(kw, str):
                raise TypeError(
                    "Each keyword must be a string, got "
                    f"{type(kw).__name__}: {kw!r}."
                )
            kw = kw.strip()
            if kw:
                cleaned.append(kw)
        return cleaned

    def expand(self, keywords: list[str] | str, factor: int = 10) -> list[str]:
        """Expand *keywords* aiming for ``len(keywords) * factor`` terms.

        Args:
            keywords: A list of seed keywords, or a single keyword string. A
                bare string is treated as **one** keyword — it is *not* iterated
                character by character.
            factor: Target multiplier (aim for ``len(keywords) * factor`` terms).

        Returns a deduplicated, sorted list of expanded terms.
        The original seed keywords are always included.
        """
        keywords = self._normalize_keywords(keywords)
        if not keywords:
            return []

        all_terms: set[str] = set()

        # Always include originals
        for kw in keywords:
            all_terms.add(kw.strip())

        # 1. WordNet synonyms
        if self.use_wordnet:
            wn_terms = self._expand_wordnet(keywords)
            all_terms.update(wn_terms)
            logger.debug("WordNet produced %d terms", len(wn_terms))

        # 2. Template-based
        if self.use_templates:
            tmpl_terms = self._expand_templates(keywords)
            all_terms.update(tmpl_terms)
            logger.debug("Templates produced %d terms", len(tmpl_terms))

        # 3. LLM-based
        if self.use_llm and self.model is not None:
            target = max(0, len(keywords) * factor - len(all_terms))
            if target > 0:
                llm_terms = self._expand_llm(keywords, target)
                all_terms.update(llm_terms)
                logger.debug("LLM produced %d terms", len(llm_terms))

        # Deduplicate (case-insensitive) keeping first-seen casing
        seen_lower: dict[str, str] = {}
        for t in all_terms:
            low = t.lower().strip()
            if low and low not in seen_lower:
                seen_lower[low] = t.strip()

        result = sorted(seen_lower.values(), key=str.lower)
        logger.info(
            "Expanded %d keywords -> %d terms (target %d)",
            len(keywords), len(result), len(keywords) * factor,
        )
        return result

    # ------------------------------------------------------------------
    # Strategy: WordNet
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_wordnet(keywords: list[str]) -> set[str]:
        """Generate synonyms using NLTK WordNet (optional dependency)."""
        try:
            from nltk.corpus import wordnet  # type: ignore[import-untyped]
        except ImportError:
            logger.info(
                "nltk not installed — skipping WordNet expansion. "
                "Install with: pip install nltk"
            )
            return set()

        # Ensure wordnet data is available
        try:
            wordnet.synsets("test")
        except LookupError:
            try:
                import nltk  # type: ignore[import-untyped]
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            except Exception:
                logger.warning("Could not download WordNet data — skipping.")
                return set()

        terms: set[str] = set()
        for kw in keywords:
            for word in kw.split():
                # Skip 1-2 char / non-alphabetic tokens: WordNet maps them to
                # off-domain noise (single letters -> chemical elements, etc.).
                if len(word) < 3 or not word.isalpha():
                    continue
                for syn in wordnet.synsets(word.lower()):
                    for lemma in syn.lemmas():
                        name = lemma.name().replace("_", " ")
                        terms.add(name)
                    # Also grab hypernyms and hyponyms (one level)
                    for hyper in syn.hypernyms():
                        for lemma in hyper.lemmas():
                            terms.add(lemma.name().replace("_", " "))
                    for hypo in syn.hyponyms():
                        for lemma in hypo.lemmas():
                            terms.add(lemma.name().replace("_", " "))
        return terms

    # ------------------------------------------------------------------
    # Strategy: Templates
    # ------------------------------------------------------------------

    def _expand_templates(self, keywords: list[str]) -> set[str]:
        """Generate queries from templates."""
        terms: set[str] = set()
        for i, kw in enumerate(keywords):
            # Pick an alternative keyword for {alt} templates
            alt = keywords[(i + 1) % len(keywords)] if len(keywords) > 1 else kw
            for tmpl in self.templates:
                expanded = tmpl.replace("{kw}", kw).replace("{alt}", alt)
                terms.add(expanded)
        return terms

    # ------------------------------------------------------------------
    # Strategy: LLM
    # ------------------------------------------------------------------

    def _expand_llm(self, keywords: list[str], target: int) -> set[str]:
        """Use the model to generate related terms."""
        if self.model is None:
            return set()

        terms: set[str] = set()
        per_kw = max(1, target // len(keywords))

        for kw in keywords:
            prompt = (
                f"Generate {per_kw} related terms, synonyms, or subtopics "
                f"for the keyword: \"{kw}\". "
                f"Return ONLY a numbered list, one term per line. "
                f"Do not include explanations."
            )
            try:
                from ..models.base import GenerationConfig
                result = self.model.generate(
                    prompt,
                    config=GenerationConfig(
                        max_tokens=256,
                        temperature=0.7,
                    ),
                )
                text = result.text if hasattr(result, "text") else str(result)
                # Parse numbered list
                for line in text.strip().split("\n"):
                    line = line.strip()
                    # Remove numbering like "1.", "1)", "- "
                    cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                    cleaned = re.sub(r"^[-*]\s*", "", cleaned).strip()
                    if cleaned and len(cleaned) < 100:
                        terms.add(cleaned)
            except Exception as exc:
                logger.warning("LLM expansion failed for '%s': %s", kw, exc)

        return terms
