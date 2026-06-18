"""Ergonomics + correctness tests for guardrails and keyword expansion.

Covers these reported issues:
- KeywordExpander.expand() char-iterates a bare string into off-domain nonsense.
- KeywordExpander.expand() emits grammatically broken query variants ("how to python").
- get_guardrail_preset("default") raises instead of mapping to "standard".
- PIIGuardrail redacts dotted-quad version numbers as IP addresses.

These exercise deterministic offline logic, so no live model is involved.
"""

from __future__ import annotations

import pytest

from effgen.domains.expander import KeywordExpander
from effgen.guardrails.content import PIIGuardrail
from effgen.guardrails.presets import get_guardrail_preset

# ----------------------------------------------------------------------
# KeywordExpander.expand() — input-type robustness + sensible terms
# ----------------------------------------------------------------------

class TestKeywordExpander:
    def test_bare_string_is_one_keyword_not_char_iterated(self):
        out = KeywordExpander().expand("python")
        # Every term must be about "python" — never single-char element junk.
        assert out, "expansion should not be empty"
        assert all("python" in t.lower() for t in out)
        joined = " ".join(out).lower()
        for junk in ("hydrogen", "oxygen", "atomic number", "alphabetic", "nitrogen"):
            assert junk not in joined

    def test_expand_python_yields_query_variants_not_chemistry(self):
        out = KeywordExpander().expand(["python"])
        assert "python" in out
        # Related search-query variants, on the seed's topic.
        assert "python tutorial" in out
        assert "what is python" in out
        # No chemistry / element noise.
        assert not any("atomic number" in t for t in out)

    def test_query_variants_read_naturally(self):
        # The expander produces search-query phrasings, which must read
        # naturally for an arbitrary noun keyword — never the grammatically
        # broken bare "how to <noun>".
        out = KeywordExpander().expand(["python"])
        assert "how to python" not in out
        assert "how to use python" in out

    def test_list_input_preserved(self):
        out = KeywordExpander().expand(["python", "machine learning"])
        assert "python" in out
        assert "machine learning" in out
        assert "machine learning tutorial" in out

    def test_tuple_and_set_accepted(self):
        assert "python" in KeywordExpander().expand(("python",))
        assert "python" in KeywordExpander().expand({"python"})

    def test_empty_inputs_return_empty(self):
        assert KeywordExpander().expand("") == []
        assert KeywordExpander().expand([]) == []
        assert KeywordExpander().expand(["   "]) == []

    def test_non_string_input_raises_typeerror(self):
        with pytest.raises(TypeError):
            KeywordExpander().expand(123)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            KeywordExpander().expand(None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            KeywordExpander().expand([1, 2, 3])  # type: ignore[list-item]

    def test_wordnet_opt_in_skips_short_token_noise(self):
        # Even when WordNet is forced on, 1-2 char tokens must not pull in
        # chemical-element synonyms.
        out = KeywordExpander(use_wordnet=True, use_templates=False).expand("ai")
        assert out == ["ai"]

    def test_wordnet_off_by_default(self):
        assert KeywordExpander().use_wordnet is False


# ----------------------------------------------------------------------
# get_guardrail_preset — "default" alias resolves
# ----------------------------------------------------------------------

class TestGuardrailPresets:
    def test_default_aliases_standard(self):
        default = get_guardrail_preset("default")
        standard = get_guardrail_preset("standard")
        assert len(default.guardrails) == len(standard.guardrails)
        assert [type(g).__name__ for g in default.guardrails] == \
               [type(g).__name__ for g in standard.guardrails]

    @pytest.mark.parametrize(
        "name", ["default", "balanced", "off", "disabled",
                 "strict", "standard", "minimal", "none", "DEFAULT", "Strict"],
    )
    def test_known_names_resolve(self, name):
        get_guardrail_preset(name)  # must not raise

    def test_off_aliases_none(self):
        assert len(get_guardrail_preset("off").guardrails) == 0

    def test_unknown_name_raises_with_available_list(self):
        with pytest.raises(ValueError, match="Available"):
            get_guardrail_preset("bogus")


# ----------------------------------------------------------------------
# PIIGuardrail — version strings must not redact as IPs
# ----------------------------------------------------------------------

class TestPIIVersionFalsePositive:
    @pytest.mark.parametrize("text", [
        "Version 1.2.3.4 released",
        "v1.2.3.4 is out",
        "Upgrade to release 2.0.1.5 now",
        "Build 10.0.0.1 patch",
        "schema 1.2.3.4",
        "protocol 1.0.0.1 spec",
        "range 1.2.3.4.5 here",          # five octets -> not an IP at all
    ])
    def test_version_not_redacted(self, text):
        r = PIIGuardrail(action="redact").check(text)
        types = (r.metadata or {}).get("pii_types", [])
        assert "IP_address" not in types
        # Content is unchanged when nothing else matched.
        assert (r.modified_content or text) == text

    @pytest.mark.parametrize("text", [
        "My server is 192.168.1.1",
        "Connect to 8.8.8.8 for DNS",
        "ssh 10.0.0.5 then run",
        "gateway 172.16.0.1 down",
    ])
    def test_real_ip_still_redacted(self, text):
        r = PIIGuardrail(action="redact").check(text)
        types = (r.metadata or {}).get("pii_types", [])
        assert "IP_address" in types
        assert "[IP REDACTED]" in (r.modified_content or "")

    def test_block_mode_still_blocks_real_ip(self):
        r = PIIGuardrail(action="block").check("host 10.1.2.3 is up")
        assert r.passed is False

    def test_other_pii_unaffected(self):
        g = PIIGuardrail(action="redact")
        assert "[EMAIL REDACTED]" in g.check("mail a@b.com").modified_content
        assert "[SSN REDACTED]" in g.check("ssn 123-45-6789").modified_content
