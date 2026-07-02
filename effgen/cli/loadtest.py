"""
effgen loadtest — CLI sub-command for the load-test harness.

Wired into the ``effgen`` CLI by ``effgen/cli/_main.py`` and invoked when the
user types ``effgen loadtest …``.

Examples
--------
    # 30-second mock run, concurrency=10, report printed to stdout
    effgen loadtest --duration 30 --concurrency 10

    # Live run against Cerebras
    effgen loadtest --provider cerebras --model gpt-oss-120b \\
        --concurrency 5 --duration 60

    # Synthetic scenario, also save the report to a file
    effgen loadtest --scenario synthetic --output /tmp/my-report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..tools.loadgen import LoadConfig, LoadGenerator, LoadScenario


def add_loadtest_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``loadtest`` to an existing argparse subparsers group."""
    p = subparsers.add_parser(
        "loadtest",
        help="Run a load test against a local mock or a live provider",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        metavar="N",
        help="Number of concurrent virtual users (default: 10)",
    )
    p.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Test duration in seconds (default: 30)",
    )
    p.add_argument(
        "--scenario", "-s",
        choices=[s.value for s in LoadScenario],
        default=LoadScenario.FIXED.value,
        help="Workload scenario: fixed | synthetic | multi_tool (default: fixed)",
    )
    p.add_argument(
        "--ramp-up",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Ramp-up period: stagger VU starts over N seconds (default: 0)",
    )
    p.add_argument(
        "--think-time",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Simulated think-time between requests per VU (default: 0)",
    )
    p.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Per-request timeout in seconds (default: 60)",
    )
    p.add_argument(
        "--provider",
        default=None,
        metavar="NAME",
        help="Provider for live runs (e.g. cerebras, openai). Default: mock",
    )
    p.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help="Model id for live runs (e.g. gpt-oss-120b). Default: mock",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write the JSON report to PATH. If omitted, the report is only "
            "printed to stdout."
        ),
    )
    p.set_defaults(func=run_loadtest_command)


def _build_live_target(provider: str, model: str):
    """Return an async callable that queries a real provider model.

    The adapter is loaded **once** here (not per request) so a load test does
    not pay model-load cost on every call.  Each virtual-user request runs the
    sync adapter in a worker thread so it never blocks the event loop.  The
    per-request timeout is enforced by the load generator, not here.
    """
    import asyncio

    from effgen import load_model  # noqa: PLC0415

    # Load the model adapter a single time; reused across all requests.
    mdl = load_model(model, provider=provider)
    if hasattr(mdl, "load"):
        try:
            mdl.load()
        except Exception:
            # Some adapters are ready on construction; loading is best-effort.
            pass

    def _generate(prompt: str) -> str:
        # The per-request timeout is enforced by the load generator via
        # asyncio.wait_for, so it is not duplicated in the adapter call here.
        result = mdl.generate(prompt)
        # Adapters return a GenerationResult; fall back to str() for plain text.
        return getattr(result, "text", None) or str(result)

    async def _live_target(prompt: str) -> str:
        return await asyncio.to_thread(_generate, prompt)

    return _live_target


def run_loadtest_command(args: argparse.Namespace) -> int:
    """Execute the load test and print / save the report."""

    # Resolve output path: write a file only when --output is given,
    # otherwise the report is printed to stdout.
    output_path: Path | None = Path(args.output) if args.output else None

    scenario = LoadScenario(args.scenario)

    # --provider and --model must be given together for a live run.
    if bool(args.provider) != bool(args.model):
        print(
            "[loadtest] Error: --provider and --model must be supplied "
            "together for a live run. Omit both to use the local mock.",
            file=sys.stderr,
        )
        return 2

    # Build target callable
    target = None
    if args.provider and args.model:
        try:
            target = _build_live_target(args.provider, args.model)
            print(
                f"Live mode: provider={args.provider}  model={args.model}",
                flush=True,
            )
        except Exception as exc:
            print(f"[loadtest] Warning: could not build live target: {exc}", file=sys.stderr)
            print("[loadtest] Falling back to mock target.", file=sys.stderr)
            target = None

    cfg = LoadConfig(
        concurrency=args.concurrency,
        duration=args.duration,
        scenario=scenario,
        ramp_up=args.ramp_up,
        request_timeout=args.request_timeout,
        think_time=args.think_time,
        provider=args.provider,
        model=args.model,
        output_path=output_path,
    )

    print(
        f"Starting load test  scenario={scenario.value}  "
        f"concurrency={cfg.concurrency}  duration={cfg.duration}s",
        flush=True,
    )

    gen = LoadGenerator(cfg, target=target)
    report = gen.run()

    # Pretty-print report to stdout
    _print_report(report)

    return 0 if report.error_rate < 1.0 else 1


def _print_report(report) -> None:  # type: ignore[no-untyped-def]
    """Render a load report to stdout."""
    d = report.to_dict()
    sep = "─" * 52
    print(sep)
    print(f"  Load Test Report — {d['scenario']}")
    print(sep)
    print(f"  Concurrency   : {d['concurrency']}")
    # Only surface the drain note when the overshoot is large enough to show at
    # the 0.1s precision used below; sub-0.05s scheduling jitter would otherwise
    # print a self-contradictory "incl. 0.0s draining" line on every fast run.
    if d["drain_s"] >= 0.05:
        print(
            f"  Duration      : requested {d['requested_duration_s']:.1f}s, "
            f"wall {d['duration_s']:.1f}s (incl. {d['drain_s']:.1f}s draining "
            f"in-flight requests after the window closed)"
        )
    else:
        print(f"  Duration      : {d['duration_s']:.1f}s")
    print(f"  Total requests: {d['total_requests']}")
    print(f"  Successful    : {d['successful_requests']}")
    print(f"  Failed        : {d['failed_requests']}")
    print(f"  Error rate    : {d['error_rate'] * 100:.2f}%")
    if d["error_breakdown"]:
        breakdown = ", ".join(f"{k}={v}" for k, v in d["error_breakdown"].items())
        print(f"  Error types   : {breakdown}")
    print(f"  Throughput    : {d['throughput_rps']:.2f} req/s")
    lat = d["latency"]
    print(f"  Latency p50   : {lat['p50'] * 1000:.1f}ms")
    print(f"  Latency p95   : {lat['p95'] * 1000:.1f}ms")
    print(f"  Latency p99   : {lat['p99'] * 1000:.1f}ms")
    print(f"  Latency mean  : {lat['mean'] * 1000:.1f}ms")
    print(f"  Latency stdev : {lat['stdev'] * 1000:.1f}ms")
    if d.get("provider"):
        print(f"  Provider      : {d['provider']}")
        print(f"  Model         : {d['model']}")
    print(sep)
