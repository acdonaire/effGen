"""Tests for 'effgen prompts' CLI subcommand."""

from __future__ import annotations

import json
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
