"""The ``effgen doctor`` command: provider keys, the system, and coding readiness.

:mod:`effgen.cli._main` parses arguments and dispatches; it imports this at
module scope and re-exports these names, so ``effgen.cli._main._handle_doctor_command``
keeps resolving for ``effgen chat``'s and ``effgen code``'s ``/doctor``. Holds
the provider key table, the optional live usability probe, the CUDA/torch/vLLM
snapshot, the circuit-breaker/bulkhead view, and the coding-readiness section.

The report helpers are reached through :mod:`effgen.cli._main` rather than as
module-local names, so replacing one there — as the coding REPL's tests do —
still changes what the command renders.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from effgen._env import load_env as load_env_files
from effgen.ui.theme import get_console as _get_console

logger = logging.getLogger(__name__)


def _handle_doctor_command(args) -> int:
    """Handle the 'effgen doctor' subcommand — check API key availability."""
    import json as _json

    from effgen.cli import _main

    # Load .env from the documented search paths before checking keys.
    load_env_files()

    from effgen.models.auth import check_keys
    from effgen.models.registry import ProviderRegistry

    # Ensure all adapters are imported so they self-register
    try:
        import effgen.models.anthropic_adapter  # noqa: F401
        import effgen.models.cerebras_adapter  # noqa: F401
        import effgen.models.fireworks_adapter  # noqa: F401
        import effgen.models.gemini_adapter  # noqa: F401
        import effgen.models.groq_adapter  # noqa: F401
        import effgen.models.hf_inference_adapter  # noqa: F401
        import effgen.models.openai_adapter  # noqa: F401
        import effgen.models.replicate_adapter  # noqa: F401
        import effgen.models.together_adapter  # noqa: F401
    except Exception:
        pass

    provider_filter = getattr(args, 'doctor_provider', None)
    providers_to_check = [provider_filter] if provider_filter else None

    results = check_keys(providers_to_check)

    live = bool(getattr(args, 'live', False))

    # Optional live usability probe: a tiny call per keyed provider that tells
    # "key present" apart from "default model actually callable".
    live_results: dict[str, dict] = {}
    if live:
        live_results = _main._doctor_live_probe(
            [p for p in results if results[p].get("available")]
        )
        for prov, lr in live_results.items():
            results[prov]["live"] = lr

    # System / CUDA / vLLM / pip-check report.
    system_report = _main._doctor_system_report(include_pip_check=live)

    # Circuit-breaker/bulkhead state for any provider that has been routed
    # through effgen.reliability middleware this process — surfaces an open
    # circuit or a saturated bulkhead without the caller instrumenting their
    # own code.
    reliability_report = _main._doctor_reliability_report()

    # What `effgen code` needs from this machine: a writable workspace, a
    # sandbox backend for the code it runs, and git for repository context.
    code_report = _main._doctor_code_report(getattr(args, 'workspace', None))

    # Exit nonzero if a live probe was requested and a keyed provider failed.
    # Computed once so every output format (JSON and human) agrees.
    exit_code = _main._doctor_exit_code(results, live)

    if getattr(args, 'output_json', False):
        print(_json.dumps({
            "providers": results,
            "system": system_report,
            "reliability": reliability_report,
            "code": code_report,
        }, indent=2))
        return exit_code

    # Pretty-print
    if _main.RICH_AVAILABLE:
        console = _get_console()
        table = _main.Table(title="effgen doctor — Provider Status")
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Key", style="white")
        table.add_column("Env Var", style="dim")
        table.add_column("Models", style="dim", justify="right")
        if live:
            table.add_column("Live", style="white")
            table.add_column("Default Model", style="magenta", overflow="fold")

        for prov in sorted(results):
            info = results[prov]
            available = info.get("available", False)
            env_key = info.get("env_key") or "—"
            status = "[green]present[/green]" if available else "[red]missing[/red]"
            try:
                n_models = str(len(ProviderRegistry.list_models(prov)))
            except Exception:
                n_models = "?"
            row = [prov, status, env_key, n_models]
            if live:
                lr = info.get("live", {})
                if not available:
                    row += ["[dim]—[/dim]", "—"]
                elif lr.get("ok"):
                    row += ["[green]usable[/green]", lr.get("model", "—")]
                else:
                    row += [f"[red]{lr.get('status', 'fail')}[/red]", lr.get("model", "—")]
            table.add_row(*row)

        console.print(table)

        if live:
            console.print("\n[bold]Live probe[/bold] — a tiny call confirms the default "
                          "model is callable (not just that a key is set).", highlight=False)
            for prov in sorted(live_results):
                lr = live_results[prov]
                if not lr.get("ok") and lr.get("detail"):
                    console.print(f"  [yellow]{prov}[/yellow]: {lr['detail']}", highlight=False)

        # System section
        console.print("\n[bold cyan]System[/bold cyan]", highlight=False)
        sys_table = _main.Table(show_header=False)
        sys_table.add_column("Check", style="cyan")
        sys_table.add_column("Value", style="white", overflow="fold")
        for k, v in system_report.items():
            sys_table.add_row(k, str(v))
        console.print(sys_table)

        # Reliability section — only shown once a provider has actually been
        # routed through circuit-breaker/bulkhead middleware this process.
        if reliability_report:
            console.print("\n[bold cyan]Reliability[/bold cyan]", highlight=False)
            rel_table = _main.Table(show_header=True)
            rel_table.add_column("Provider", style="cyan", no_wrap=True)
            rel_table.add_column("Circuit", style="white")
            rel_table.add_column("Bulkhead", style="white")
            for prov, rec in sorted(reliability_report.items()):
                cb = rec.get("circuit_breaker")
                bh = rec.get("bulkhead")
                if cb is None:
                    circuit_cell = "[dim]—[/dim]"
                elif cb["state"] == "closed":
                    circuit_cell = "[green]closed[/green]"
                elif cb["state"] == "half_open":
                    circuit_cell = "[yellow]half_open[/yellow]"
                else:
                    circuit_cell = "[red]open[/red]"
                if bh is None:
                    bulkhead_cell = "[dim]—[/dim]"
                else:
                    bulkhead_cell = f"active={bh['active']}/{bh['max_concurrency']}, queued={bh['queued']}/{bh['queue_size']}"
                rel_table.add_row(prov, circuit_cell, bulkhead_cell)
            console.print(rel_table)

        # Coding section — what `effgen code` needs from this machine.
        console.print("\n[bold cyan]Coding (effgen code)[/bold cyan]", highlight=False)
        if code_report.get("error"):
            console.print(f"  [red]The coding checks could not run: {code_report['error']}[/red]", highlight=False)
        code_table = _main.Table(show_header=True)
        code_table.add_column("Check", style="cyan", no_wrap=True)
        code_table.add_column("Status", style="white", no_wrap=True)
        code_table.add_column("Detail", style="white", overflow="fold")
        _code_status_style = {"ready": "green", "limited": "yellow"}
        for check in code_report.get("checks", []):
            style = _code_status_style.get(
                check.get("status", ""), "green" if check.get("ok") else "red"
            )
            code_table.add_row(
                check.get("name", ""),
                f"[{style}]{check.get('status', '')}[/{style}]",
                check.get("detail", ""),
            )
        console.print(code_table)
        for check in code_report.get("checks", []):
            if check.get("fix"):
                console.print(f"  [yellow]{check.get('name')}[/yellow]: {check['fix']}", highlight=False)
        if code_report.get("ready"):
            console.print("  Try it: [bold]effgen code \"write fib.py and run it\"[/bold]", highlight=False)

        # Print hints for missing keys
        missing = [p for p, i in results.items() if not i.get("available")]
        if missing:
            console.print("\n[yellow]Missing keys — set in ~/.effgen/.env or export:[/yellow]", highlight=False)
            for prov in missing:
                keys = results[prov].get("env_keys_checked", [])
                key_str = " or ".join(keys) if keys else f"{prov.upper()}_API_KEY"
                console.print(f"  export {key_str}=<your-key>", highlight=False)
    else:
        print("effgen doctor — Provider Status")
        print("-" * 50)
        for prov in sorted(results):
            info = results[prov]
            available = info.get("available", False)
            env_key = info.get("env_key") or "not set"
            status = "key present" if available else "key missing"
            line = f"  {prov:12s} {status:12s}  (env: {env_key})"
            if live and available:
                lr = info.get("live", {})
                line += f"  live={'usable' if lr.get('ok') else lr.get('status', 'fail')}"
            print(line)
        print("\nSystem:")
        for k, v in system_report.items():
            print(f"  {k}: {v}")
        if reliability_report:
            print("\nReliability:")
            for prov, rec in sorted(reliability_report.items()):
                cb = rec.get("circuit_breaker")
                bh = rec.get("bulkhead")
                circuit_str = cb["state"] if cb else "—"
                bulkhead_str = (
                    f"active={bh['active']}/{bh['max_concurrency']}, queued={bh['queued']}/{bh['queue_size']}"
                    if bh else "—"
                )
                print(f"  {prov:12s} circuit={circuit_str:10s} bulkhead={bulkhead_str}")
        print("\nCoding (effgen code):")
        if code_report.get("error"):
            print(f"  The coding checks could not run: {code_report['error']}")
        for check in code_report.get("checks", []):
            print(f"  {check.get('name', ''):12s} {check.get('status', ''):14s} {check.get('detail', '')}")
            if check.get("fix"):
                print(f"    Fix: {check['fix']}")
        if code_report.get("ready"):
            print("  Try it: effgen code \"write fib.py and run it\"")
        missing = [p for p, i in results.items() if not i.get("available")]
        if missing:
            print("\nMissing keys — set in ~/.effgen/.env or export:")
            for prov in missing:
                keys = results[prov].get("env_keys_checked", [])
                key_str = " or ".join(keys) if keys else f"{prov.upper()}_API_KEY"
                print(f"  export {key_str}=<your-key>")

    return exit_code


def _doctor_exit_code(results: dict[str, dict], live: bool) -> int:
    """Exit code for `effgen doctor`: nonzero iff a live probe was requested and
    a keyed (key-present) provider's default model was not actually usable.

    Kept format-independent so `--json` and the human table return the same code
    for the same provider state.
    """
    if live and any(
        results[p].get("available") and not results[p].get("live", {}).get("ok")
        for p in results
    ):
        return 1
    return 0


def _doctor_reliability_report() -> dict[str, dict]:
    """Circuit-breaker/bulkhead state for providers routed through reliability
    middleware this process, keyed by provider name (empty if none have).

    A provider only appears once ``ProviderRegistry.get_circuit_breaker``/
    ``get_bulkhead`` has been used for it — no calls yet made means no state
    to report, which is the common case for a fresh CLI invocation.
    """
    try:
        from effgen.models.registry import ProviderRegistry

        stats = ProviderRegistry.reliability_stats()
    except Exception:
        return {}
    return {
        prov: rec
        for prov, rec in stats.items()
        if rec.get("circuit_breaker") is not None or rec.get("bulkhead") is not None
    }


def _doctor_code_report(workspace: str | None = None) -> dict[str, Any]:
    """Coding-agent readiness: the workspace, the sandbox backend and git.

    Reported for information only — a machine without Docker or without git can
    still run `effgen code`, so this never changes the exit code. A check that
    would actually stop a run (an unwritable workspace, a disabled sandbox)
    carries ``ok: false`` and the fix for it.
    """
    try:
        from effgen.cli.code.readiness import code_readiness

        return code_readiness(workspace).to_dict()
    except Exception as e:  # noqa: BLE001 - a diagnostic never breaks doctor
        return {"ready": False, "workspace": "", "checks": [], "error": str(e)}


def _doctor_system_report(*, include_pip_check: bool = False) -> dict[str, Any]:
    """Collect a CUDA / torch / vLLM / pip-check diagnostic snapshot."""
    report: dict[str, Any] = {}
    try:
        from effgen.gpu import cuda_compat
        status = cuda_compat.get_cuda_status()
        report["Physical GPUs (NVML)"] = status.physical_gpus
        report["Driver CUDA"] = status.driver_cuda or "n/a"
        report["torch CUDA build"] = (
            (status.torch_cuda or "cpu-only") if status.torch_installed else "not installed"
        )
        report["torch.cuda.is_available()"] = status.usable
        if status.mismatch:
            report["CUDA mismatch"] = (
                "YES — GPUs present but torch runs on CPU" if status.torch_installed
                else "n/a — GPUs present but PyTorch is not installed"
            )
    except Exception as e:  # noqa: BLE001
        report["CUDA"] = f"unavailable ({e})"

    # torch version
    try:
        import torch
        report["torch"] = torch.__version__
    except Exception:
        report["torch"] = "not installed"

    # vLLM import status (a frequent ABI casualty)
    try:
        import importlib.util
        if importlib.util.find_spec("vllm") is None:
            report["vLLM"] = "not installed"
        else:
            try:
                import vllm  # noqa: F401
                report["vLLM"] = f"importable ({getattr(vllm, '__version__', '?')})"
            except Exception as e:  # noqa: BLE001
                # Name what is actually missing or broken — "ModuleNotFoundError"
                # alone does not tell the reader which package to install.
                detail = str(e).strip() or type(e).__name__
                report["vLLM"] = f"installed but import failed ({detail[:80]})"
    except Exception:
        report["vLLM"] = "unknown"

    if include_pip_check:
        try:
            import subprocess
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0:
                report["pip check"] = "no broken requirements"
            else:
                lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
                report["pip check"] = f"{len(lines)} issue(s): " + "; ".join(lines[:3])
        except Exception as e:  # noqa: BLE001
            report["pip check"] = f"could not run ({e})"

    return report


def _doctor_live_probe(providers: list[str], *, timeout: float = 30.0) -> dict[str, dict]:
    """Make a tiny live call per provider to confirm its default model is usable.

    Returns ``{provider: {"ok": bool, "model": str, "status": str, "detail": str}}``.
    Runs providers concurrently with a bounded wall-clock budget so the command
    stays responsive even if one provider hangs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from effgen.models import _catalog

    def _probe_one(prov: str) -> dict:
        model_id = _catalog.default_model(prov)
        out = {"ok": False, "model": model_id or "—", "status": "no-default", "detail": ""}
        if not model_id:
            out["detail"] = "no default model in catalog"
            return out
        try:
            from effgen import load_model
            model = load_model(model_id, provider=prov)
            model.load()
            # Keep it minimal but DON'T force max_tokens: newer reasoning models
            # (e.g. OpenAI gpt-5.x) reject max_tokens and need a token budget for
            # reasoning, so a hard cap of 1 produces a false "error". A one-word
            # reply to "Reply with: ok" is already negligibly cheap.
            resp = model.generate("Reply with the single word: ok", temperature=0.0)
            text = getattr(resp, "content", None) or getattr(resp, "text", "") or str(resp)
            out["ok"] = True
            out["status"] = "usable"
            out["detail"] = (text or "").strip()[:40]
        except Exception as e:  # noqa: BLE001 - classify for a friendly status
            from effgen.models.errors import (
                ModelAuthError,
                ModelNotFoundError,
            )
            if isinstance(e, ModelAuthError):
                out["status"] = "auth-failed"
            elif isinstance(e, ModelNotFoundError):
                out["status"] = "model-404"
            else:
                out["status"] = "error"
            # Message is already redacted by the typed errors / adapters.
            out["detail"] = str(e)[:160]
        return out

    results: dict[str, dict] = {}
    if not providers:
        return results
    with ThreadPoolExecutor(max_workers=min(len(providers), 6)) as pool:
        futs = {pool.submit(_probe_one, p): p for p in providers}
        for fut in as_completed(futs, timeout=timeout + 5):
            prov = futs[fut]
            try:
                results[prov] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[prov] = {"ok": False, "model": "—", "status": "timeout", "detail": str(e)[:120]}
    return results
