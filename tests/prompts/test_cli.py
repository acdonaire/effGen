"""Tests for 'effgen prompts' CLI subcommand."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "effgen.cli"] + list(args),
        capture_output=True,
        text=True,
    )


class TestPromptsListCLI:
    def test_prompts_list_exits_zero(self):
        result = run_cli("prompts", "list")
        assert result.returncode == 0

    def test_prompts_list_empty_registry(self):
        result = run_cli("prompts", "list")
        assert result.returncode == 0
        # Should mention 'Total' even if empty
        assert "Total" in result.stdout or "No prompts" in result.stdout

    def test_prompts_list_json_format(self):
        result = run_cli("prompts", "list", "--format", "json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)

    def test_prompts_list_markdown_format(self):
        result = run_cli("prompts", "list", "--format", "markdown")
        assert result.returncode == 0
        assert "| Name |" in result.stdout

    def test_prompts_list_domain_filter(self):
        result = run_cli("prompts", "list", "--domain", "research")
        assert result.returncode == 0

    def test_prompts_list_variant_filter(self):
        result = run_cli("prompts", "list", "--variant", "cot")
        assert result.returncode == 0


class TestPromptsShowCLI:
    def test_prompts_show_missing_exits_nonzero(self):
        result = run_cli("prompts", "show", "does.not.exist.v1")
        assert result.returncode != 0
        assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()

    def test_prompts_show_missing_suggests_close_match(self):
        # A near-miss typo of a real registered name gets a "did you mean".
        result = run_cli("prompts", "show", "business.elevater_pitch.v1")
        assert result.returncode != 0
        combined = (result.stdout + result.stderr).lower()
        assert "did you mean" in combined
        assert "business.elevator_pitch.v1" in combined

    def test_prompts_show_base_name_resolves_default_variant(self):
        # Omitting the trailing ".v1" (the id truncated in the table view)
        # still resolves when the base name is unambiguous.
        result = run_cli("prompts", "show", "business.elevator_pitch")
        assert result.returncode == 0
        assert "business.elevator_pitch.v1" in result.stdout

    def test_prompts_list_narrow_terminal_does_not_ellipsis_truncate_names(self):
        # At a narrow terminal a long, space-free name (e.g.
        # "business.elevator_pitch.v1") would previously be cut mid-word
        # with an ellipsis ("business.elevator_pitch…"), hiding the ".v1"
        # a user has to type back into `show`/`run`/`render`. It must now
        # wrap onto an extra line instead.
        env = {**os.environ, "COLUMNS": "80"}
        result = subprocess.run(
            [sys.executable, "-m", "effgen.cli", "prompts", "list"],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0
        assert "pitch…" not in result.stdout
        assert "summar…" not in result.stdout

    def test_prompts_table_name_column_uses_fold_overflow(self):
        # Regression guard tied to the actual fix: the Name column must use
        # "fold" (wraps a space-free id onto extra lines) rather than the
        # default ellipsis truncation.
        import inspect

        from effgen.cli import _main

        src = inspect.getsource(_main)
        assert 'add_column("Name", style="cyan", overflow="fold")' in src


class TestPromptsEvalCLI:
    def test_eval_empty_registry_exits_zero(self, tmp_path):
        result = run_cli("prompts", "eval")
        assert result.returncode == 0
        assert "empty" in result.stdout.lower() or "Total" in result.stdout

    def test_eval_with_output_file(self, tmp_path):
        out = str(tmp_path / "eval.txt")
        result = run_cli("prompts", "eval", "--output", out)
        assert result.returncode == 0
        assert Path(out).exists()
        content = Path(out).read_text()
        assert "Golden Eval" in content

    def test_eval_live_requires_model(self):
        result = run_cli("prompts", "eval", "--live")
        assert result.returncode != 0
        assert "--model" in result.stdout or "--model" in result.stderr
