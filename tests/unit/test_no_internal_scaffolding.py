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
``elegant``/``elegantly``, ``robustly``, ``seamless``/``seamlessly``,
``effortless``/``effortlessly``, ``magically``, ``blazing``/``blazingly``,
``production-grade``, ``production-ready``, ``world-class``,
``battle-tested``, ``bulletproof``, ``state-of-the-art``,
``flawless``/``flawlessly``, ``rock-solid``, ``industrial-strength``,
``enterprise-grade``, ``military-grade``, ``best-in-class``, ``turnkey``,
``lightning-fast``, ``hassle-free``, ``painless``/``painlessly``, ``sleek``,
``gorgeous``, ``stunning``.

``cleanly``/``properly``/``correctly`` are deliberately *not* machine-gated:
they carry a large volume of plainly factual technical uses ("imports cleanly",
"routes correctly", "properly configured") that a regex cannot separate from the
rare filler use, so blanket-gating them would force an unwieldy allowlist.
``first-class`` is excluded for the same reason: it is a term of art for a type
or value the language/framework supports natively, and separating that from the
promotional use ("a first-class experience") needs a human. ``cutting-edge`` and
``bleeding-edge`` are excluded too: they describe how recent a dependency or
branch is ("installs the bleeding-edge main branch") as often as they praise.
Those words are scrubbed by human review, not by this gate.

Both classes are matched against each line twice: as written, and with
identifiers broken into words, so a gated term hidden inside a function or class
name (``test_fails_gracefully``, ``TestExecutorHonesty``) is caught too.

A small, documented allowlist excuses the handful of *legitimate* occurrences
(an SSN-format mask, files that intentionally exclude the internal planning
directory, the CI meta-files that themselves grep for these markers). The scan
is intentionally narrow and self-tested so it cannot silently no-op: a planted
violation of every pattern must be caught (see the ``test_detector_catches_*``
tests).

Scanning is by suffix (see ``_SOURCE_SUFFIXES``) and covers every file type
effGen authors prose or markup in — including the bundled web surfaces
(``.html``/``.css``), the brand assets (``.svg``), and the scaffolding templates
the ``create-plugin`` command emits (``.tmpl``/``.tpl``), whose comments and
copy ship to users like any other source. A file is matched on *any* of its
dotted suffixes and on a variant name prefix, so the forms that park a real file
type behind another word — a parked workflow (``release.yml.disabled``), a
per-target image (``Dockerfile.sandbox``) — are scanned as what they are.

Data files are out of scope by design:

* ``.txt``/``.jsonl`` fixtures and golden files hold quoted third-party prose
  (paper abstracts) and recorded model output — neither is effGen describing
  itself, and gating them would mean allowlisting verbatim quotations;
* ``.json`` is dependency lockfiles, provider catalog snapshots and generated
  schemas — machine-written payloads with no authored prose;
* ``.sql`` is query fixtures for the prompt-library evals.

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
    # Authored markup that ships to users: the self-contained web surfaces,
    # the brand assets, and the plugin scaffolding templates.
    ".html", ".css", ".svg", ".tmpl", ".tpl",
}
_SOURCE_NAMES = {".gitignore", ".dockerignore", ".gitattributes", "MANIFEST.in", "Dockerfile"}

# Names whose variant forms are authored the same way as the base name: a
# per-target container image is conventionally ``Dockerfile.<variant>``.
_SOURCE_NAME_PREFIXES = ("Dockerfile.",)

# The only directory never scanned: internal planning scaffolding, which is
# gitignored and never shipped. Bundled datasets need no directory rule — their
# payload files are ``.txt``/``.json``/``.jsonl``, already out of scope by
# suffix, while the scripts that fetch them are ordinary authored source.
# ``website/`` holds the marketing site and its documentation app, kept as a
# mirror of the site they are published from so an update there is a straight
# copy. They are page copy rather than shipped library text, and rewording them
# to satisfy this gate would put the repository and the live site out of step.
_SKIP_DIR_PREFIXES = ("build_plan/", "website/")

# This gate embeds every forbidden pattern as a literal, so it cannot scan
# itself without self-tripping.
_SKIP_FILES = {"tests/unit/test_no_internal_scaffolding.py"}

# Dated release narrative. Scanned for process jargon, but exempt from the
# editorializing-self-praise check: these are append-only historical records of
# what each past release said, so rewording them would falsify the record.
# README.md/README_PYPI.md are deliberately NOT exempt — they describe the
# project as it is today and are held to the same standard as source.
_EDITORIALIZING_EXEMPT_FILES = {
    "CHANGELOG.md", "NEWS.md",
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
    # author/process breadcrumbs. The role words (builder/verifier/explorer)
    # are only flagged next to a process verb — "the verifier" is also the
    # ordinary name for a hash-checking component, and "explorer" for a file
    # browser.
    "process-breadcrumb": re.compile(
        r"\bthis phase\b|\bas per (?:audit|the report)\b|\bfixed in phase\b|"
        r"\bthe report (?:asked|requested|wanted)\b|\bbuilder agent\b|"
        r"\b(?:builder|verifier|explorer) (?:added|fixed|confirmed|noted|reported)\b",
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
    "praise-seamless": re.compile(r"\bseamless(?:ly)?\b", re.IGNORECASE),
    "praise-effortless": re.compile(r"\beffortless(?:ly)?\b", re.IGNORECASE),
    "praise-magically": re.compile(r"\bmagically\b", re.IGNORECASE),
    "praise-blazing": re.compile(r"\bblazing(?:ly)?\b", re.IGNORECASE),
    "praise-production-grade": re.compile(r"\bproduction[ -]grade\b", re.IGNORECASE),
    "praise-production-ready": re.compile(r"\bproduction[ -]ready\b", re.IGNORECASE),
    "praise-world-class": re.compile(r"\bworld[ -]class\b", re.IGNORECASE),
    "praise-battle-tested": re.compile(r"\bbattle[ -]tested\b", re.IGNORECASE),
    "praise-bulletproof": re.compile(r"\bbullet[ -]?proof\b", re.IGNORECASE),
    "praise-state-of-the-art": re.compile(r"\bstate[ -]of[ -]the[ -]art\b", re.IGNORECASE),
    "praise-flawless": re.compile(r"\bflawless(?:ly)?\b", re.IGNORECASE),
    "praise-rock-solid": re.compile(r"\brock[ -]solid\b", re.IGNORECASE),
    "praise-industrial-strength": re.compile(r"\bindustrial[ -]strength\b", re.IGNORECASE),
    "praise-enterprise-grade": re.compile(r"\benterprise[ -]grade\b", re.IGNORECASE),
    "praise-military-grade": re.compile(r"\bmilitary[ -]grade\b", re.IGNORECASE),
    "praise-best-in-class": re.compile(r"\bbest[ -]in[ -]class\b", re.IGNORECASE),
    "praise-turnkey": re.compile(r"\bturn[ -]?key\b", re.IGNORECASE),
    "praise-lightning-fast": re.compile(r"\blightning[ -]fast\b", re.IGNORECASE),
    "praise-hassle-free": re.compile(r"\bhassle[ -]free\b", re.IGNORECASE),
    "praise-painless": re.compile(r"\bpainless(?:ly)?\b", re.IGNORECASE),
    # Visual self-praise, aimed at the bundled web surfaces now in scope.
    "praise-sleek": re.compile(r"\bsleek\b", re.IGNORECASE),
    "praise-gorgeous": re.compile(r"\bgorgeous\b", re.IGNORECASE),
    "praise-stunning": re.compile(r"\bstunning\b", re.IGNORECASE),
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
    (".gitleaks.toml", "build_plan"),
    # The sibling jargon-check test carries the forbidden words as test data.
    ("tests/unit/test_onboarding.py", "for bad in"),
    # CI meta-files that themselves grep for placeholder markers.
    (".github/workflows/pr-check.yml", "TODO"),
    ("CONTRIBUTING.md", "without a tracking issue"),
    # Historical release-notes entry (changelog is human narrative, not source).
    ("CHANGELOG.md", "ACP TODO"),
]

# Documented allowlist for the editorializing check: genuinely-legitimate domain
# uses of an otherwise-gated word. Same (path, substring) form as above.
EDITORIALIZING_ALLOWLIST: list[tuple[str, str]] = [
    # Prompt-template payload: the wording is an instruction sent to the model
    # about the artifact *it* should design, not a claim about effGen itself.
    # Changing it would change the template's output.
    (
        "effgen/prompts/library/domains/data/etl_plan_v1.py",
        "You are a senior data engineer.",
    ),
]


def _is_allowed(rel_path: str, line: str, allowlist: list[tuple[str, str]]) -> bool:
    return any(p == rel_path and needle in line for p, needle in allowlist)


# Every pattern below is anchored on ``\b``, and ``_`` is a word character while
# a CamelCase hump is not a boundary at all — so a gated word sitting inside an
# identifier (``test_fails_gracefully``, ``TestExecutorHonesty``) would never
# match. Each line is therefore also searched in a form where identifiers are
# broken into their constituent words. Splitting on the hump rather than on
# every capital keeps acronyms intact, and words that merely *contain* a gated
# stem (``dishonest``) stay unsplit and so stay unmatched.
_CAMEL_HUMP = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _split_identifiers(line: str) -> str:
    return _CAMEL_HUMP.sub(" ", line).replace("_", " ")


def _scan(
    rel_path: str,
    text: str,
    patterns: dict[str, re.Pattern[str]],
    allowlist: list[tuple[str, str]],
) -> list[tuple[int, str, str]]:
    """Return ``(line_no, pattern_name, line)`` for every unjustified hit.

    The allowlist is matched against the line as written, so an entry keeps
    naming the real text of the line it excuses.
    """
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _is_allowed(rel_path, line, allowlist):
            continue
        split = _split_identifiers(line)
        for name, pat in patterns.items():
            if pat.search(line) or (split != line and pat.search(split)):
                out.append((lineno, name, line.strip()))
    return out


def find_violations(rel_path: str, text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_no, pattern_name, line)`` for every unjustified jargon hit."""
    return _scan(rel_path, text, PATTERNS, ALLOWLIST)


def find_editorializing(rel_path: str, text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_no, pattern_name, line)`` for every editorializing hit."""
    return _scan(rel_path, text, EDITORIALIZING_PATTERNS, EDITORIALIZING_ALLOWLIST)


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
        # Every dotted suffix is considered, not just the last one, so a file
        # parked behind a trailing marker (``release.yml.disabled``) is scanned
        # as the kind of file it is.
        if (
            any(s in _SOURCE_SUFFIXES for s in Path(rel).suffixes)
            or name in _SOURCE_NAMES
            or name.startswith(_SOURCE_NAME_PREFIXES)
        ):
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


def test_detector_catches_report_and_role_breadcrumbs():
    """Breadcrumbs naming the internal report or an internal role must fire."""
    for sample in (
        "as per the report, the retry budget was raised",
        "the report asked for a clearer error here",
        "the report requested a --json flag",
        "builder added the fallback path",
        "the verifier confirmed this on three families",
        "explorer reported a cryptic traceback here",
    ):
        hits = find_violations("some/source.py", sample)
        assert any(n == "process-breadcrumb" for _, n, _ in hits), sample
    # The same role words in their ordinary technical sense must stay clean:
    # a hash verifier, a file explorer, a package builder.
    for ok in (
        "EFFGEN_VERIFY_HASHES=1 must call the verifier",
        "opens the path in the system file explorer",
        "the wheel builder writes to dist/",
    ):
        assert not any(
            n == "process-breadcrumb" for _, n, _ in find_violations("s.py", ok)
        ), ok


def test_detector_catches_gated_words_inside_identifiers():
    """A gated word hidden in a snake_case or CamelCase name must still fire."""
    snake = [
        ("def test_init_with_string_model_fails_gracefully(self):", "praise-gracefully"),
        ("def test_empty_workflow_is_honest_failure():", "praise-honest"),
        ("def test_renders_beautifully_in_notebook():", "praise-beautifully"),
        ("production_ready = True", "praise-production-ready"),
    ]
    camel = [
        ("class TestPublicCodeExecutorHonesty:", "praise-honest"),
        ("class SeamlessMigrationHelper:", "praise-seamless"),
    ]
    for sample, expected in snake + camel:
        hits = find_editorializing("some/source.py", sample)
        assert any(n == expected for _, n, _ in hits), sample

    # Process jargon hidden in an identifier is caught the same way.
    assert any(
        n == "milestone-reference"
        for _, n, _ in find_violations("s.py", "def test_phase_7_regression():")
    )

    # A word that merely *contains* a gated stem is not split, so it stays clean.
    for ok in ("dishonest_input = True", "class Blazer:", "the production environment"):
        assert not find_editorializing("some/source.py", ok), ok


def test_allowlist_matches_the_line_as_written():
    """Splitting identifiers must not defeat an allowlist entry."""
    # ".gitignore" excuses the literal "build_plan"; the split form reads
    # "build plan", which the milestone pattern also matches.
    assert not find_violations(".gitignore", "build_plan/")
    assert find_violations("some/source.py", "build_plan/")


def test_scan_covers_authored_markup_suffixes():
    """The bundled web surfaces and scaffolding templates are in scope."""
    for suffix in (".html", ".css", ".svg", ".tmpl", ".tpl"):
        assert suffix in _SOURCE_SUFFIXES, suffix
    scanned = set(_tracked_source_files())
    assert "effgen/dashboard/static/index.html" in scanned
    assert "effgen/playground/static/index.html" in scanned
    assert "effgen/cli/_templates/plugin/tools.py.tmpl" in scanned
    # Recorded model output and third-party quotations stay out of scope.
    for suffix in (".txt", ".jsonl", ".json", ".sql"):
        assert suffix not in _SOURCE_SUFFIXES, suffix


def test_scan_covers_files_whose_type_is_not_the_last_suffix():
    """A parked workflow and a per-target image are scanned as what they are."""
    scanned = set(_tracked_source_files())
    assert ".github/workflows/release.yml.disabled" in scanned
    assert "deploy/sandbox/Dockerfile.sandbox" in scanned


def test_scan_covers_scripts_that_sit_beside_bundled_data():
    """Only the dataset payloads are out of scope, not the code next to them."""
    scanned = set(_tracked_source_files())
    assert "examples/data/download_arc.py" in scanned
    assert "examples/data/arc_easy_test.jsonl" not in scanned
    assert "examples/data/arc_easy_test.txt" not in scanned


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


def test_detector_catches_promotional_vocabulary():
    """Every promotional word carries its own pattern, in both spellings."""
    expected = {
        "praise-seamless": ["a seamless upgrade", "swaps seamlessly"],
        "praise-effortless": ["scales effortlessly", "an effortless setup"],
        "praise-magically": ["the cache magically warms"],
        "praise-blazing": ["blazing throughput", "blazingly fast startup"],
        "praise-production-grade": ["a production-grade pipeline", "production grade RAG"],
        "praise-production-ready": ["a production-ready chart", "production ready adapter"],
        "praise-world-class": ["world-class ergonomics", "world class tracing"],
        "praise-battle-tested": ["battle-tested retries", "battle tested sandbox"],
        "praise-bulletproof": ["bulletproof auth", "bullet-proof parsing"],
        "praise-state-of-the-art": ["state-of-the-art routing", "state of the art reranking"],
        "praise-flawless": ["flawless recovery", "replays flawlessly"],
        "praise-rock-solid": ["rock-solid checkpoints", "rock solid streaming"],
        "praise-industrial-strength": ["industrial-strength queueing"],
        "praise-enterprise-grade": ["enterprise-grade RBAC", "enterprise grade audit log"],
        "praise-military-grade": ["military-grade encryption"],
        "praise-best-in-class": ["best-in-class latency", "best in class recall"],
        "praise-turnkey": ["a turnkey deployment", "turn-key onboarding"],
        "praise-lightning-fast": ["lightning-fast cold starts"],
        "praise-hassle-free": ["hassle-free upgrades"],
        "praise-painless": ["a painless migration", "migrates painlessly"],
        "praise-sleek": ["a sleek dashboard"],
        "praise-gorgeous": ["gorgeous trace rendering"],
        "praise-stunning": ["a stunning topology view"],
    }
    for name, samples in expected.items():
        for sample in samples:
            hits = find_editorializing("some/source.py", sample)
            assert any(n == name for _, n, _ in hits), (name, sample)
    # Ordinary technical prose that merely shares a word stem must stay clean.
    for ok in (
        "the production environment variable",
        "class Blazer:",
        "artwork classification",
        "the enterprise directory endpoint",
        "flaws = validate(config)",
        "installs the bleeding-edge main branch",
    ):
        assert not find_editorializing("some/source.py", ok), ok


def test_editorializing_allowlist_is_scoped_to_its_file():
    """An allowlisted line is excused only on its own path."""
    line = "        \"You are a senior data engineer. Design a production-ready ETL pipeline.\\n\\n\""
    assert not find_editorializing(
        "effgen/prompts/library/domains/data/etl_plan_v1.py", line
    )
    assert find_editorializing("effgen/elsewhere.py", line)


def test_allowlist_excuses_legitimate_lines():
    # SSN mask is not flagged on its own file...
    assert not find_violations(
        "effgen/guardrails/content.py", "    # US Social Security Number: XXX-XX-XXXX"
    )
    # ...but the same text elsewhere is flagged.
    assert find_violations("effgen/elsewhere.py", "mask = 'XXX-XX-XXXX'")
