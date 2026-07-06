"""Guard against internal build-process scaffolding leaking into shipped source.

Shipped code, tests, docs, configs and workflows describe *behavior*, not the
internal process used to build the project, and they describe that behavior
plainly — without self-congratulatory adjectives. This gate scans every tracked
text file and fails on two classes of text:

**(a) Internal process jargon** that should never ship:

* internal tracking IDs (e.g. ``VF5``, ``GA4``, ``RA-N3``, ``SEC1``, ``FN-2``,
  ``Audit-2 #x``) and per-finding IDs (e.g. ``E1-1``, ``E11-3``),
* internal milestone references (``Phase 7``, ``Phase-7``, ``Phase_7``,
  ``build plan``, ``stabilization sprint``) — the separator between ``Phase``
  and its number may be a space, hyphen, or underscore,
* author/process breadcrumbs (``this phase``, ``as per audit``,
  ``fixed in phase``, ``builder/verifier added``),
* leftover debugging (``breakpoint()`` / ``pdb.set_trace()``),
* unresolved placeholder markers (``TODO`` / ``FIXME`` / ``XXX`` / ``HACK``).

**(b) Editorializing self-praise** — the code describing its *own* behavior with
approving adjectives/adverbs. State what the code does, not how good it is at it:
say "reports the failure with a typed error", not "fails honestly"; say
"degrades to plain output when rich is absent", not "degrades gracefully". The
gated vocabulary is the set that is essentially always editorializing when it
describes the software's behavior: ``honest``/``honestly``/``honesty``,
``gracefully``, ``delightful``/``delightfully``, ``beautifully``,
``elegant``/``elegantly``, ``robustly``.

``cleanly``/``properly``/``correctly`` are deliberately *not* machine-gated:
they carry a large volume of plainly factual technical uses ("imports cleanly",
"routes correctly", "properly configured") that a regex cannot separate from the
rare filler use, so blanket-gating them would force an unwieldy allowlist. Those
words are scrubbed by human review, not by this gate.

A small, documented allowlist excuses the handful of *legitimate* occurrences
(an SSN-format mask, files that intentionally exclude the internal planning
directory, the CI meta-files that themselves grep for these markers). The scan
is intentionally narrow and self-tested so it cannot silently no-op: a planted
violation of every pattern must be caught (see the ``test_detector_catches_*``
tests).

``build_plan/`` is deliberately *not* scanned — it is gitignored internal
scaffolding and is never shipped. The human-authored release narrative
(``CHANGELOG.md``/``NEWS.md``/``README.md``/``README_PYPI.md``) is scanned for
process jargon but *exempt from the editorializing check*: it is prose owned and
maintained by the release step, not source code.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files scanned by suffix (source/docs/config) plus these exact names.
_SOURCE_SUFFIXES = {
    ".py", ".pyi", ".md", ".rst", ".toml", ".cfg", ".ini",
    ".yaml", ".yml", ".sh", ".bash", ".ts", ".js", ".tsx", ".jsx",
}
_SOURCE_NAMES = {".gitignore", ".dockerignore", ".gitattributes", "MANIFEST.in", "Dockerfile"}

# Directories never scanned: internal planning scaffolding (gitignored) and
# bundled datasets (arbitrary natural-language payloads, not authored prose).
_SKIP_DIR_PREFIXES = ("build_plan/", "examples/data/")

# This gate embeds every forbidden pattern as a literal, so it cannot scan
# itself without self-tripping.
_SKIP_FILES = {"tests/unit/test_no_internal_scaffolding.py"}

# Human-authored release narrative. Scanned for process jargon, but exempt from
# the editorializing-self-praise check: this prose is owned by the release step
# (which is the only phase permitted to edit it), not by the source scrub.
_EDITORIALIZING_EXEMPT_FILES = {
    "CHANGELOG.md", "NEWS.md", "README.md", "README_PYPI.md",
}

# ── forbidden patterns: (a) internal process jargon ───────────────────────────
PATTERNS: dict[str, re.Pattern[str]] = {
    # internal tracking IDs from the build/audit process
    "internal-tracking-id": re.compile(
        r"\b(?:VF\d+|GA\d+|RA-[NC]\d+|SEC\d+|FN-\d+|E\d+-\d+)\b|Audit-2 #"
    ),
    # internal milestone / planning references. The separator between "Phase"
    # and the number may be a space, hyphen, or underscore ("Phase 7",
    # "Phase-7", "Phase_7") — an earlier hyphenated breadcrumb slipped past the
    # space-only form.
    "milestone-reference": re.compile(
        r"\bPhase[ _-]?\d+\b|\bbuild[ _-]?plan\b|stabilization sprint", re.IGNORECASE
    ),
    # author/process breadcrumbs
    "process-breadcrumb": re.compile(
        r"\bthis phase\b|\bas per audit\b|\bfixed in phase\b|"
        r"\bbuilder agent\b|\bverifier (?:added|fixed)\b",
        re.IGNORECASE,
    ),
    # leftover debugging
    "debug-leftover": re.compile(r"breakpoint\(\)|pdb\.set_trace\("),
    # unresolved placeholder markers
    "placeholder-marker": re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b"),
}

# ── forbidden patterns: (b) editorializing self-praise ────────────────────────
# Vocabulary that is essentially always the code praising its own behavior.
# See the module docstring for why cleanly/properly/correctly are excluded.
EDITORIALIZING_PATTERNS: dict[str, re.Pattern[str]] = {
    "praise-honest": re.compile(r"\bhonest(?:ly|y)?\b", re.IGNORECASE),
    "praise-gracefully": re.compile(r"\bgracefully\b", re.IGNORECASE),
    "praise-delightful": re.compile(r"\bdelightful(?:ly)?\b", re.IGNORECASE),
    "praise-beautifully": re.compile(r"\bbeautifully\b", re.IGNORECASE),
    "praise-elegant": re.compile(r"\belegant(?:ly)?\b", re.IGNORECASE),
    "praise-robustly": re.compile(r"\brobustly\b", re.IGNORECASE),
}

# ── documented allowlist of legitimate occurrences ────────────────────────────
# Each entry: (tracked path, substring that must appear on the matched line).
# A matched line is excused only if both the path and the substring match.
ALLOWLIST: list[tuple[str, str]] = [
    # An SSN-format mask used by the PII redactor, not a placeholder marker.
    ("effgen/guardrails/content.py", "XXX-XX-XXXX"),
    # Files that intentionally exclude the internal planning directory.
    (".gitignore", "build_plan"),
    ("MANIFEST.in", "build_plan"),
    ("deploy/docker/.dockerignore", "build_plan"),
    # The sibling jargon-check test carries the forbidden words as test data.
    ("tests/unit/test_onboarding.py", "for bad in"),
    # CI meta-files that themselves grep for placeholder markers.
    (".github/workflows/pr-check.yml", "TODO"),
    ("CONTRIBUTING.md", "without a tracking issue"),
    # Historical release-notes entry (changelog is human narrative, not source).
    ("CHANGELOG.md", "ACP TODO"),
]

# Documented allowlist for the editorializing check: genuinely-legitimate domain
# uses of an otherwise-gated word (empty today — every source occurrence was
# rephrased to describe behavior plainly). Same (path, substring) form as above.
EDITORIALIZING_ALLOWLIST: list[tuple[str, str]] = []


def _is_allowed(rel_path: str, line: str, allowlist: list[tuple[str, str]]) -> bool:
    return any(p == rel_path and needle in line for p, needle in allowlist)


def find_violations(rel_path: str, text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_no, pattern_name, line)`` for every unjustified jargon hit."""
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pat in PATTERNS.items():
            if pat.search(line) and not _is_allowed(rel_path, line, ALLOWLIST):
                out.append((lineno, name, line.strip()))
    return out


def find_editorializing(rel_path: str, text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_no, pattern_name, line)`` for every editorializing hit."""
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pat in EDITORIALIZING_PATTERNS.items():
            if pat.search(line) and not _is_allowed(rel_path, line, EDITORIALIZING_ALLOWLIST):
                out.append((lineno, name, line.strip()))
    return out


def _tracked_source_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available / not a git checkout")
    files: list[str] = []
    for rel in out.splitlines():
        if not rel or rel in _SKIP_FILES:
            continue
        if rel.startswith(_SKIP_DIR_PREFIXES):
            continue
        name = rel.rsplit("/", 1)[-1]
        if Path(rel).suffix in _SOURCE_SUFFIXES or name in _SOURCE_NAMES:
            files.append(rel)
    return files


def test_tracked_source_has_no_internal_scaffolding():
    violations: list[str] = []
    for rel in _tracked_source_files():
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, name, line in find_violations(rel, text):
            violations.append(f"{rel}:{lineno} [{name}] {line}")
    assert not violations, (
        "Internal build-process scaffolding found in tracked source. Describe "
        "the behavior, not the process (or add a documented allowlist entry):\n  "
        + "\n  ".join(violations)
    )


def test_tracked_source_has_no_editorializing_self_praise():
    violations: list[str] = []
    for rel in _tracked_source_files():
        if rel in _EDITORIALIZING_EXEMPT_FILES:
            continue
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, name, line in find_editorializing(rel, text):
            violations.append(f"{rel}:{lineno} [{name}] {line}")
    assert not violations, (
        "Editorializing self-praise found in tracked source. Describe what the "
        "code does plainly, not how well it does it (or add a documented "
        "EDITORIALIZING_ALLOWLIST entry for a genuine domain use):\n  "
        + "\n  ".join(violations)
    )


def test_detector_catches_planted_violation():
    """The jargon detector must actually fire — a no-op gate would pass forever."""
    samples = [
        "this is the VF5 regression case",
        "the E11-3 per-finding case",
        "fixes Audit-2 #42",
        "see Phase 7 of the build plan",
        "as per audit, this phase reworked it",
        "    breakpoint()  # debug",
        "x = 1  # TODO clean this up",
    ]
    for sample in samples:
        assert find_violations("some/source.py", sample), sample


def test_detector_catches_hyphenated_and_underscored_phase():
    """The milestone pattern must catch every ``Phase``/number separator."""
    for sample in ("a Phase-35 breadcrumb", "the Phase_18 guard", "Phase 8 catalog"):
        hits = find_violations("some/source.py", sample)
        assert hits and hits[0][1] == "milestone-reference", sample
    # A bare "phase" with no number is ordinary English, not a milestone ref.
    assert not any(
        n == "milestone-reference" for _, n, _ in find_violations("s.py", "a phased rollout")
    )


def test_detector_catches_planted_editorializing():
    """The editorializing detector must actually fire on each gated word."""
    samples = [
        "# fails honestly with a typed error",
        "the honesty of the cost label",
        "degrades gracefully when rich is absent",
        "a delightful first-run experience",
        "renders beautifully in the notebook",
        "an elegant fallback path",
        "handles arbitrary input robustly",
    ]
    for sample in samples:
        assert find_editorializing("some/source.py", sample), sample
    # Plainly-factual descriptions must NOT trip the editorializing gate.
    for ok in ("routes to the correct adapter", "the model imports cleanly", "properly typed"):
        assert not find_editorializing("some/source.py", ok), ok


def test_allowlist_excuses_legitimate_lines():
    # SSN mask is not flagged on its own file...
    assert not find_violations(
        "effgen/guardrails/content.py", "    # US Social Security Number: XXX-XX-XXXX"
    )
    # ...but the same text elsewhere is flagged.
    assert find_violations("effgen/elsewhere.py", "mask = 'XXX-XX-XXXX'")
