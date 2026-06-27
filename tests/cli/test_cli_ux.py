"""CLI UX regression tests.

Covers: provider validation, quiet-by-default logging, registry-backed
`models list`/`info`, `--json` outputs, doctor system report and exit code,
examples-dir discovery, all-preset acceptance, and centralized `.env`
discovery. No live model calls here — live behavior (`run --provider`,
`doctor --live`) is exercised via end-to-end CLI smokes.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from effgen.cli import _main


# --------------------------------------------------------------------------- #
# Provider validation
# --------------------------------------------------------------------------- #
def test_resolve_provider_valid():
    assert _main.resolve_provider_name("groq") == ("groq", None)
    assert _main.resolve_provider_name("OpenAI") == ("openai", None)


def test_resolve_provider_alias():
    assert _main.resolve_provider_name("google") == ("gemini", None)
    assert _main.resolve_provider_name("huggingface") == ("hf", None)


def test_resolve_provider_none():
    assert _main.resolve_provider_name(None) == (None, None)


def test_resolve_provider_typo_suggests():
    resolved, err = _main.resolve_provider_name("grok")
    assert resolved is None
    assert err is not None
    assert "groq" in err  # fuzzy suggestion
    assert "grok" in err


def test_loader_rejects_unknown_provider():
    from effgen.models.model_loader import ModelLoader
    with pytest.raises(ValueError, match="Unknown provider"):
        ModelLoader().load_model("some-model", provider="grok")


# --------------------------------------------------------------------------- #
# Quiet-by-default logging
# --------------------------------------------------------------------------- #
def test_setup_logging_default_is_warning():
    _main.setup_logging(verbose=False, quiet=False)
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_verbose_is_debug():
    _main.setup_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_quiet_is_error():
    _main.setup_logging(quiet=True)
    assert logging.getLogger().level == logging.ERROR
    # restore a sane default for the rest of the suite
    _main.setup_logging(verbose=False)


# --------------------------------------------------------------------------- #
# Registry-backed models list / info
# --------------------------------------------------------------------------- #
def _cli():
    return _main.CLIInterface()


def test_models_list_json_is_registry_backed(capsys):
    args = SimpleNamespace(provider=None, free=False, tools=False, output_json=True)
    code = _cli()._models_list(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert code == 0
    assert "providers" in data and "local_cache" in data
    # The known 9 providers are present and registry-derived, not yaml.
    assert {"openai", "groq", "cerebras"} <= set(data["providers"])
    assert data["providers"]["openai"]["count"] > 0


def test_models_list_no_bogus_empty_config_ids(capsys):
    # The old empty-config fallback printed mistral-7b/llama-2-7b/
    # gemma-7b. Those must never appear now.
    args = SimpleNamespace(provider=None, free=False, tools=False, output_json=False)
    _cli()._models_list(args)
    out = capsys.readouterr().out
    for bogus in ("mistral-7b", "llama-2-7b", "gemma-7b"):
        assert bogus not in out


def test_models_info_known(capsys):
    args = SimpleNamespace(name="gpt-5-nano", output_json=True)
    code = _cli()._models_info(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert code == 0
    assert data["id"] == "gpt-5-nano"
    assert data["provider"] == "openai"


def test_models_info_unknown_suggests_and_exits_nonzero(capsys):
    args = SimpleNamespace(name="gpt-5-nanoo", output_json=False)
    code = _cli()._models_info(args)
    out = capsys.readouterr().out
    assert code == 1
    assert "not found" in out.lower()


# --------------------------------------------------------------------------- #
# Local-cache awareness (models info / list)
# --------------------------------------------------------------------------- #
def _fake_local(entries):
    """Build a fake _local_cached_models payload."""
    return [
        {"id": e["id"], "size_gb": e.get("size_gb", 1.0),
         "path": e.get("path", "/tmp/x"), "complete": e.get("complete", True)}
        for e in entries
    ]


def test_models_info_local_aware_when_not_in_catalog(capsys, monkeypatch):
    # A bare HF id that's downloaded locally but absent from the cloud catalog
    # must describe the local copy, not say "not found".
    cli = _cli()
    monkeypatch.setattr(cli, "_local_cached_models",
                        lambda: _fake_local([{"id": "meta-llama/Llama-3.2-3B-Instruct",
                                              "size_gb": 6.0}]))
    monkeypatch.setattr(cli, "_local_model_context_window", lambda path: 131072)
    args = SimpleNamespace(name="meta-llama/Llama-3.2-3B-Instruct", output_json=True)
    code = cli._models_info(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert code == 0
    assert data["local"]["cached"] is True
    assert "transformers" in data["local"]["engines"]
    assert data["local"]["context_window"] == 131072


def test_models_info_includes_local_block_alongside_cloud(capsys, monkeypatch):
    # An id present in BOTH the catalog and the local cache shows the cloud row
    # AND a local-copy block (non-JSON path), so the local copy isn't hidden.
    cli = _cli()
    monkeypatch.setattr(cli, "_local_cached_models",
                        lambda: _fake_local([{"id": "gpt-5-nano", "size_gb": 2.0}]))
    monkeypatch.setattr(cli, "_local_model_context_window", lambda path: 4096)
    args = SimpleNamespace(name="gpt-5-nano", output_json=True)
    code = cli._models_info(args)
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["provider"] == "openai"  # cloud row preserved
    assert data["local"] is not None and data["local"]["cached"] is True


def test_models_list_flags_incomplete_download(capsys, monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli, "_local_cached_models", lambda: _fake_local([
        {"id": "Qwen/Qwen2.5-1.5B-Instruct", "size_gb": 2.9, "complete": True},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "size_gb": 0.01, "complete": False},
    ]))
    args = SimpleNamespace(provider=None, free=False, tools=False, output_json=False)
    cli._models_list(args)
    out = capsys.readouterr().out
    assert "incomplete" in out.lower()
    # the "ready" count excludes the incomplete one.
    assert "(1 ready)" in out


def test_local_cached_models_carries_complete_flag():
    # Against the real HF cache on this host: every entry exposes a bool 'complete'.
    for entry in _cli()._local_cached_models():
        assert isinstance(entry.get("complete"), bool)
        assert "id" in entry and "size_gb" in entry


# --------------------------------------------------------------------------- #
# --json outputs
# --------------------------------------------------------------------------- #
def test_tools_list_json(capsys):
    args = SimpleNamespace(output_json=True, category=None)
    code = _cli()._tools_list(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert code == 0
    assert isinstance(data, list) and len(data) > 10
    assert {"name", "category", "description"} <= set(data[0])


# --------------------------------------------------------------------------- #
# Doctor system report
# --------------------------------------------------------------------------- #
def test_doctor_system_report_has_cuda_and_vllm():
    report = _main._doctor_system_report(include_pip_check=False)
    assert "torch.cuda.is_available()" in report
    assert "vLLM" in report
    assert "torch" in report


def test_doctor_exit_code_is_format_independent():
    # A keyed provider whose live probe failed must yield exit 1 regardless of
    # whether output is JSON or the human table.
    failed = {"openai": {"available": True, "live": {"ok": False}}}
    assert _main._doctor_exit_code(failed, live=True) == 1
    # No live probe requested → always 0 even if a key is missing.
    assert _main._doctor_exit_code(failed, live=False) == 0
    # All keyed providers usable → 0; a missing-key provider never fails the run.
    ok = {
        "openai": {"available": True, "live": {"ok": True}},
        "anthropic": {"available": False},
    }
    assert _main._doctor_exit_code(ok, live=True) == 0


# --------------------------------------------------------------------------- #
# Examples directory discovery
# --------------------------------------------------------------------------- #
def test_find_examples_dir_via_env(tmp_path, monkeypatch):
    ex = tmp_path / "examples"
    (ex / "basic").mkdir(parents=True)
    (ex / "basic" / "hello.py").write_text("def main():\n    pass\n")
    monkeypatch.setenv("EFFGEN_EXAMPLES_DIR", str(ex))
    found = _main.CLIInterface._find_examples_dir()
    assert found == ex


# --------------------------------------------------------------------------- #
# All 9 presets accepted
# --------------------------------------------------------------------------- #
def test_all_presets_accepted_by_run_parser():
    from effgen.presets import list_presets
    parser = _main.create_parser()
    all_presets = set(list_presets())
    for preset in all_presets:
        ns = parser.parse_args(["run", "task", "--preset", preset])
        assert ns.preset == preset
    # the previously-rejected four must be in there
    assert {"rag", "media", "multimodal", "notify"} <= all_presets


# --------------------------------------------------------------------------- #
# .env discovery is centralized + documented
# --------------------------------------------------------------------------- #
def test_env_search_paths_include_override_and_home(monkeypatch):
    monkeypatch.setenv("EFFGEN_DOTENV", "/tmp/custom.env")
    paths = [str(p) for p in _main._env_search_paths()]
    assert "/tmp/custom.env" == paths[0]
    assert any(p.endswith("/.effgen/.env") for p in paths)
    # walks up from cwd, so it includes a ./.env candidate
    assert any(p.endswith("/.env") for p in paths)


# --------------------------------------------------------------------------- #
# "Did you mean?" on mistyped subcommands and choice-based options
# --------------------------------------------------------------------------- #
def test_unknown_subcommand_suggests_and_exits_2(capsys):
    parser = _main.create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["rnu"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unknown command 'rnu'" in err
    assert "Did you mean 'run'?" in err
    assert "Available commands:" in err


def test_mistyped_preset_value_suggests_and_exits_2(capsys):
    parser = _main.create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["run", "task", "--preset", "codng"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid value 'codng' for --preset" in err
    assert "Did you mean 'coding'?" in err


def test_mistyped_completion_value_suggests(capsys):
    parser = _main.create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--completion", "bsh"])
    err = capsys.readouterr().err
    assert "Did you mean 'bash'?" in err
