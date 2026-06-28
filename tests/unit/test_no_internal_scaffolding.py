"""Guard against internal build-process scaffolding leaking into shipped source.

Shipped code, tests, docs, configs and workflows describe *behavior*, not the
internal process used to build the project. This gate scans every tracked text
file and fails if it finds breadcrumbs that should never ship:

* internal tracking IDs (e.g. ``VF5``, ``GA4``, ``RA-N3``, ``SEC1``, ``FN-2``,
  ``Audit-2 #x``) and per-finding IDs (e.g. ``E1-1``, ``E11-3``),
* internal milestone references (``Phase 7``, ``build plan``,
  ``stabilization sprint``),
* author/process breadcrumbs (``this phase``, ``as per audit``,
  ``fixed in phase``, ``builder/verifier added``),
* leftover debugging (``breakpoint()`` / ``pdb.set_trace()``),
* unresolved placeholder markers (``TODO`` / ``FIXME`` / ``XXX`` / ``HACK``).

A small, documented allowlist excuses the handful of *legitimate* occurrences
(an SSN-format mask, files that intentionally exclude the internal planning
directory, the CI meta-files that themselves grep for these markers). The scan
is intentionally narrow and self-tested so it cannot silently no-op: a planted
violation must be caught (see ``test_detector_catches_planted_violation``).

``build_plan/`` is deliberately *not* scanned — it is gitignored internal
scaffolding and is never shipped.
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

# ── forbidden patterns ────────────────────────────────────────────────────────
PATTERNS: dict[str, re.Pattern[str]] = {
    # internal tracking IDs from the build/audit process
    "internal-tracking-id": re.compile(
        r"\b(?:VF\d+|GA\d+|RA-[NC]\d+|SEC\d+|FN-\d+|E\d+-\d+)\b|Audit-2 #"
    ),
    # internal milestone / planning references
    "milestone-reference": re.compile(
        r"\bPhase \d+\b|\bbuild[ _-]?plan\b|stabilization sprint", re.IGNORECASE
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


def _is_allowed(rel_path: str, line: str) -> bool:
    return any(p == rel_path and needle in line for p, needle in ALLOWLIST)


def find_violations(rel_path: str, text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_no, pattern_name, line)`` for every unjustified hit."""
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pat in PATTERNS.items():
            if pat.search(line) and not _is_allowed(rel_path, line):
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


def test_detector_catches_planted_violation():
    """The detector must actually fire — a no-op gate would pass forever."""
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


def test_allowlist_excuses_legitimate_lines():
    # SSN mask is not flagged on its own file...
    assert not find_violations(
        "effgen/guardrails/content.py", "    # US Social Security Number: XXX-XX-XXXX"
    )
    # ...but the same text elsewhere is flagged.
    assert find_violations("effgen/elsewhere.py", "mask = 'XXX-XX-XXXX'")
