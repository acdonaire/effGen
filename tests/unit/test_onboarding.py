"""Onboarding, tips, did-you-mean and teaching-error contracts (no live calls).

Covers the CLI guidance layer: a non-spammy rotating tip system
(``EFFGEN_TIPS=0`` / ``--quiet`` / non-interactive suppression), a one-time
first-run welcome with a stored flag, a shared fuzzy "did you mean?" suggester,
and the shared "errors that teach" formatter.
"""

import io

import pytest

from effgen.cli import onboarding as ob


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Redirect effGen state to a throwaway directory via EFFGEN_HOME."""
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    monkeypatch.delenv("EFFGEN_TIPS", raising=False)
    return tmp_path


# --------------------------------------------------------------------------- #
# state dir
# --------------------------------------------------------------------------- #
def test_state_dir_honors_env(isolated_home):
    assert ob.state_dir() == isolated_home


# --------------------------------------------------------------------------- #
# tips (D-9)
# --------------------------------------------------------------------------- #
def test_tips_are_curated_and_nonempty():
    assert len(ob.TIPS) >= 5
    assert all(isinstance(t, str) and t for t in ob.TIPS)
    # No internal phase jargon leaks into user-facing tips.
    joined = " ".join(ob.TIPS).lower()
    for bad in ("phase", "v0.3.0", "stabiliz", "build plan"):
        assert bad not in joined


def test_pick_tip_rotates_and_wraps():
    assert ob.pick_tip(0) == ob.TIPS[0]
    assert ob.pick_tip(1) == ob.TIPS[1]
    assert ob.pick_tip(len(ob.TIPS)) == ob.TIPS[0]  # wraps


def test_tips_enabled_respects_quiet_env_and_tty(isolated_home, monkeypatch):
    monkeypatch.setattr(ob, "is_interactive", lambda: True)
    assert ob.tips_enabled(quiet=False) is True
    assert ob.tips_enabled(quiet=True) is False
    monkeypatch.setenv("EFFGEN_TIPS", "0")
    assert ob.tips_enabled(quiet=False) is False
    monkeypatch.setenv("EFFGEN_TIPS", "1")
    assert ob.tips_enabled(quiet=False) is True
    # Non-interactive always disables.
    monkeypatch.setattr(ob, "is_interactive", lambda: False)
    assert ob.tips_enabled(quiet=False) is False


def test_maybe_print_tip_suppressed_when_disabled(isolated_home, monkeypatch):
    monkeypatch.setattr(ob, "is_interactive", lambda: False)
    buf = io.StringIO()
    assert ob.maybe_print_tip(quiet=False, force=True, stream=buf) is False
    assert buf.getvalue() == ""


def test_maybe_print_tip_prints_and_rotates(isolated_home, monkeypatch):
    monkeypatch.setattr(ob, "is_interactive", lambda: True)
    buf = io.StringIO()
    assert ob.maybe_print_tip(quiet=False, force=True, stream=buf) is True
    out = buf.getvalue()
    assert "💡 Tip:" in out
    # A second forced tip rotates to a different body.
    buf2 = io.StringIO()
    ob.maybe_print_tip(quiet=False, force=True, stream=buf2)
    assert buf2.getvalue() != out


def test_maybe_print_tip_is_throttled(isolated_home, monkeypatch):
    """Without force, a tip shows only occasionally (every few runs)."""
    monkeypatch.setattr(ob, "is_interactive", lambda: True)
    shown = 0
    for _ in range(9):
        buf = io.StringIO()
        if ob.maybe_print_tip(quiet=False, stream=buf):
            shown += 1
    # 9 eligible runs, throttle of 3 -> exactly 3 tips.
    assert shown == 3


# --------------------------------------------------------------------------- #
# first-run welcome (D-11)
# --------------------------------------------------------------------------- #
def test_first_run_welcome_shows_once(isolated_home, monkeypatch):
    monkeypatch.setattr(ob, "is_interactive", lambda: True)
    assert ob.first_run_pending() is True
    buf = io.StringIO()
    assert ob.maybe_show_first_run_welcome(quiet=False, stream=buf) is True
    assert "Welcome to effGen" in buf.getvalue()
    assert "effgen doctor" in buf.getvalue()
    # Flag is now set; never shows again.
    assert ob.first_run_pending() is False
    buf2 = io.StringIO()
    assert ob.maybe_show_first_run_welcome(quiet=False, stream=buf2) is False
    assert buf2.getvalue() == ""


def test_first_run_welcome_silent_when_quiet_but_records_flag(isolated_home, monkeypatch):
    monkeypatch.setattr(ob, "is_interactive", lambda: True)
    buf = io.StringIO()
    assert ob.maybe_show_first_run_welcome(quiet=True, stream=buf) is False
    assert buf.getvalue() == ""
    # Even though it was silent, the flag is recorded so it won't surprise later.
    assert ob.first_run_pending() is False


# --------------------------------------------------------------------------- #
# did you mean (D-10)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,pool,expected",
    [
        ("maths", ["math", "research", "coding"], "math"),
        ("rnu", ["run", "resume", "chat"], "run"),
        ("gpt-5-nanoo", ["gpt-5-nano", "gpt-5-mini"], "gpt-5-nano"),
        ("budget.dialy", ["budget.daily", "budget.monthly"], "budget.daily"),
    ],
)
def test_suggest_finds_close_match(name, pool, expected):
    assert ob.suggest(name, pool, n=1) == [expected]


def test_suggest_returns_empty_for_far_input():
    assert ob.suggest("zzzzzz", ["run", "chat"]) == []
    assert ob.suggest("", ["run"]) == []


def test_did_you_mean_phrasing():
    assert ob.did_you_mean("rnu", ["run", "resume"]) == "Did you mean 'run'?"
    assert ob.did_you_mean("zzzzz", ["run"]) == ""
    multi = ob.did_you_mean("re", ["run", "resume", "reset"], n=3, cutoff=0.4)
    assert multi.startswith("Did you mean")


# --------------------------------------------------------------------------- #
# errors that teach (D-14)
# --------------------------------------------------------------------------- #
def test_teach_formats_cause_fix_doc():
    out = ob.teach("Something broke", "Do X", "docs/y.md")
    assert out.splitlines() == [
        "Something broke",
        "  Fix: Do X",
        "  Docs: docs/y.md",
    ]


def test_teach_cause_only():
    assert ob.teach("Just a cause") == "Just a cause"
