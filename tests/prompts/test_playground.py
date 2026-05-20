"""
Tests for the prompt playground — both unit tests and pexpect REPL walk-throughs.

pexpect tests drive the REPL via a subprocess so they exercise the full CLI
stack including arg parsing, REPL dispatch, render, session save/load, and
exit.

Non-pexpect tests cover the session serialization and the cmd_render /
cmd_run helpers directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _effgen_bin() -> str:
    """Return the effgen executable in the currently active Python env."""
    scripts = Path(sys.executable).parent
    for name in ("effgen", "effgen.exe"):
        candidate = scripts / name
        if candidate.exists():
            return str(candidate)
    return "effgen"


EFFGEN = _effgen_bin()


# -------------------------------------------------------------------------
# Session unit tests
# -------------------------------------------------------------------------

class TestPlaygroundSession:
    def test_set_unset(self):
        from effgen.prompts.library.session import PlaygroundSession

        s = PlaygroundSession(prompt_name="x.v1")
        s.set_var("topic", "diffusion models")
        assert s.variables["topic"] == "diffusion models"
        removed = s.unset_var("topic")
        assert removed
        assert "topic" not in s.variables
        assert not s.unset_var("topic")  # idempotent

    def test_add_render_run(self):
        from effgen.prompts.library.session import PlaygroundSession

        s = PlaygroundSession(prompt_name="x.v1")
        s.add_render("Hello, world")
        assert len(s.render_history) == 1
        s.add_run("gpt-4", "Hello, world", "Hi there!")
        assert len(s.run_history) == 1
        assert s.run_history[0].model == "gpt-4"

    def test_save_load_roundtrip(self, tmp_path):
        from effgen.prompts.library.session import PlaygroundSession

        s = PlaygroundSession(prompt_name="research.v1", variables={"topic": "AI"})
        s.add_render("Sample rendered text")
        s.add_run("llama3", "Sample rendered text", "Model response")
        path = tmp_path / "session.json"
        saved = s.save(path)
        assert saved == path
        s2 = PlaygroundSession.load(path)
        assert s2.prompt_name == "research.v1"
        assert s2.variables["topic"] == "AI"
        assert len(s2.render_history) == 1
        assert len(s2.run_history) == 1
        assert s2.run_history[0].model == "llama3"

    def test_auto_save_path(self, tmp_path, monkeypatch):
        from effgen.prompts.library import session as sess_mod
        from effgen.prompts.library.session import PlaygroundSession

        monkeypatch.setattr(sess_mod, "_SESSIONS_DIR", tmp_path)
        s = PlaygroundSession(prompt_name="business.elevator_pitch.v1")
        path = s.save()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["prompt_name"] == "business.elevator_pitch.v1"

    def test_summary(self):
        from effgen.prompts.library.session import PlaygroundSession

        s = PlaygroundSession(prompt_name="research.v1", variables={"a": 1, "b": 2})
        s.add_render("x")
        summary = s.summary()
        assert "research.v1" in summary
        assert "vars=2" in summary
        assert "renders=1" in summary


# -------------------------------------------------------------------------
# cmd_render / cmd_run unit tests
# -------------------------------------------------------------------------

class TestNonInteractive:
    def test_cmd_render_known_prompt(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render("research.literature_review.v1.zero_shot", {})
        assert rc == 0
        captured = capsys.readouterr()
        assert "literature review" in captured.out.lower() or captured.out  # some output

    def test_cmd_render_unknown_prompt(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render("no.such.prompt.v99", {})
        assert rc == 1

    def test_cmd_render_with_inputs(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render(
            "research.literature_review.v1.zero_shot",
            {"topic": "quantum computing", "years_range": "2022-2024", "max_papers": 3},
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "quantum computing" in captured.out

    def test_cmd_render_business_prompt(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render("business.elevator_pitch.v1", {})
        assert rc == 0


# -------------------------------------------------------------------------
# CLI non-interactive tests (subprocess)
# -------------------------------------------------------------------------

class TestCLINonInteractive:
    def test_prompts_list(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "list"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "research" in result.stdout.lower() or "business" in result.stdout.lower()

    def test_prompts_render_stdout(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "research.literature_review.v1.zero_shot"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_prompts_render_with_input_file(self, tmp_path):
        import subprocess
        inp = tmp_path / "input.json"
        inp.write_text(json.dumps({
            "topic": "neural radiance fields",
            "years_range": "2021-2024",
            "max_papers": 5,
        }))
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "research.literature_review.v1.zero_shot",
             "--input", str(inp)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        # Rich may wrap "neural radiance fields" across panel border lines;
        # strip all non-alphanumeric separators to do a substring check.
        import re as _re
        cleaned = _re.sub(r"[│╭╰╮╯─\s]+", " ", result.stdout)
        assert "neural" in cleaned and "radiance fields" in cleaned

    def test_prompts_render_unknown(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "no.such.prompt.v99"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    def test_prompts_render_coding(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "coding.docstring_fill.v1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_prompts_render_legal(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "legal.clause_classify.v1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_prompts_render_medical(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "medical.symptom_triage.v1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_prompts_render_creative(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "creative.story_continuation.v1.zero_shot"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_prompts_render_data(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "data.sql_explain.v1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0


# -------------------------------------------------------------------------
# pexpect REPL tests
# -------------------------------------------------------------------------

pexpect = pytest.importorskip("pexpect", reason="pexpect not installed")


class TestPlaygroundREPL:
    """Scripted REPL walk-through via pexpect."""

    TIMEOUT = 30

    def _spawn(self):
        child = pexpect.spawn(
            EFFGEN,
            ["prompts", "playground"],
            encoding="utf-8",
            timeout=self.TIMEOUT,
        )
        child.expect(r"effGen Prompt Playground", timeout=self.TIMEOUT)
        return child

    def test_help_command(self):
        child = self._spawn()
        child.sendline("help")
        child.expect("select", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_select_and_render(self):
        child = self._spawn()
        child.sendline("select research.literature_review.v1.zero_shot")
        child.expect("Selected", timeout=self.TIMEOUT)
        child.sendline("render")
        child.expect("literature review", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_set_then_render(self):
        child = self._spawn()
        child.sendline("select research.literature_review.v1.zero_shot")
        child.expect("Selected", timeout=self.TIMEOUT)
        child.sendline('set topic "protein folding"')
        child.expect("Set topic", timeout=self.TIMEOUT)
        child.sendline("render")
        child.expect("protein folding", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_unset(self):
        child = self._spawn()
        child.sendline("select business.elevator_pitch.v1")
        child.expect("Selected", timeout=self.TIMEOUT)
        child.sendline("set product_name TestProduct")
        child.expect("Set product_name", timeout=self.TIMEOUT)
        child.sendline("unset product_name")
        child.expect("Unset product_name", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_select_unknown_prompt(self):
        child = self._spawn()
        child.sendline("select no.such.prompt.v99")
        child.expect("not found", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_render_without_select(self):
        child = self._spawn()
        child.sendline("render")
        child.expect("No prompt selected", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_save_and_load_session(self, tmp_path):
        session_file = str(tmp_path / "session.json")
        child = self._spawn()
        child.sendline("select research.literature_review.v1.zero_shot")
        child.expect("Selected", timeout=self.TIMEOUT)
        child.sendline('set topic "deep learning"')
        child.expect("Set topic", timeout=self.TIMEOUT)
        child.sendline(f"save {session_file}")
        child.expect("Session saved", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

        # Verify the file was written
        assert Path(session_file).exists()
        data = json.loads(Path(session_file).read_text())
        assert data["prompt_name"] == "research.literature_review.v1.zero_shot"
        assert data["variables"]["topic"] == "deep learning"

        # Load in a new session
        child2 = self._spawn()
        child2.sendline(f"load {session_file}")
        child2.expect("Session loaded", timeout=self.TIMEOUT)
        child2.sendline("exit")
        child2.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_list_command_in_repl(self):
        child = self._spawn()
        child.sendline("list")
        child.expect("research|business|coding", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_list_with_domain_filter(self):
        child = self._spawn()
        child.sendline("list --domain coding")
        child.expect("coding", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_show_prompt(self):
        child = self._spawn()
        child.sendline("show coding.code_review.v1")
        child.expect("Domain", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_unknown_command(self):
        child = self._spawn()
        child.sendline("foobar")
        child.expect("Unknown command", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_quit_alias(self):
        child = self._spawn()
        child.sendline("quit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_reload_without_select(self):
        child = self._spawn()
        child.sendline("reload")
        child.expect("No prompt selected", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_reload_with_selected_prompt(self):
        child = self._spawn()
        child.sendline("select coding.docstring_fill.v1")
        child.expect("Selected", timeout=self.TIMEOUT)
        child.sendline("reload")
        # Either reloaded or warned, but no crash
        child.expect(r"Reloaded|No matching|Warning", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_full_round_trip_multiple_domains(self):
        """REPL round-trip on prompts from research, coding, business, legal, data."""
        child = self._spawn()
        for name in [
            "research.literature_review.v1.zero_shot",
            "coding.docstring_fill.v1",
            "business.elevator_pitch.v1",
            "legal.clause_classify.v1",
            "data.sql_explain.v1",
            "medical.symptom_triage.v1",
            "creative.story_continuation.v1.zero_shot",
        ]:
            child.sendline(f"select {name}")
            child.expect("Selected", timeout=self.TIMEOUT)
            child.sendline("render")
            child.expect(r"Rendered Prompt|---", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)
