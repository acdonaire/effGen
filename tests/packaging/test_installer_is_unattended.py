"""The installer never waits for an answer nobody is there to give.

`./install.sh` runs from a pipeline, from another script, and a second time on
a machine that already has the environment. In each case stdin is not a
terminal, so a bare `read` returns non-zero at once — and the installer runs
under `set -e`, which turns that into "Installation failed" for a condition
that is not a failure. This has reached a release candidate before.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: Every shell file that can prompt during an install.
INSTALLER_SCRIPTS = (
    "install.sh",
    "scripts/install_effgen.sh",
    "scripts/setup_and_verify.sh",
)

#: A prompt is acceptable when a terminal check guards it. `[ -t 0 ]` is the
#: test; it has to appear close enough above the prompt to be guarding it.
TERMINAL_CHECK = re.compile(r"-t\s+0")
GUARD_WINDOW_LINES = 12

#: Reading a prompt. Matches `read -p`, and `read` with the prompt on the line
#: before it, while ignoring commented-out lines and `while read` loops that
#: consume a pipe rather than a person.
PROMPT = re.compile(r"^[^#]*\bread\s+(-[a-zA-Z]+\s+)*-p\b")


def _scripts() -> list[Path]:
    return [REPO / name for name in INSTALLER_SCRIPTS if (REPO / name).exists()]


def test_at_least_one_installer_script_is_present():
    """A vacuous pass here would hide every check below."""
    assert _scripts(), f"none of {INSTALLER_SCRIPTS} exist to check"


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_every_prompt_is_guarded_by_a_terminal_check(script: Path):
    """A prompt with no terminal check fails any unattended install."""
    lines = script.read_text().splitlines()
    unguarded = []
    for number, line in enumerate(lines, start=1):
        if not PROMPT.search(line):
            continue
        window = "\n".join(lines[max(0, number - GUARD_WINDOW_LINES - 1):number])
        if not TERMINAL_CHECK.search(window):
            unguarded.append(f"{script.name}:{number}: {line.strip()}")

    assert not unguarded, (
        "These prompts run whether or not anyone can answer them, so an "
        "install from CI, from another script, or with stdin redirected ends "
        "at `read` under `set -e`:\n  " + "\n  ".join(unguarded) +
        "\nGuard each with `if [ -t 0 ]` and pick a safe default without one."
    )


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_the_script_parses(script: Path):
    """A syntax error here is only ever found by someone installing."""
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"{script.name}: {result.stderr}"


def test_the_existing_environment_branch_keeps_what_is_there():
    """Without a terminal the installer must not delete an existing environment.

    Removing it is the destructive choice, and nobody asked for it — the branch
    exists precisely because a second run is a normal thing to do.
    """
    text = (REPO / "scripts/install_effgen.sh").read_text()
    start = text.find("already exists")
    assert start != -1, "the existing-environment branch has moved or gone"
    branch = text[start:start + 1500]
    assert "-t 0" in branch, "the existing-environment prompt is not guarded"
    assert "conda env remove" in branch, "the branch no longer offers a rebuild"
    remove_at = branch.find("conda env remove")
    guard_at = branch.find("-t 0")
    assert guard_at < remove_at, (
        "the environment is removed before the terminal check decides whether "
        "anyone asked for that"
    )
