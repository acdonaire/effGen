"""Live validation for LatencyBasedPolicy routing."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import effgen.models  # noqa: E402,F401  # register providers
from effgen.models.base import GenerationConfig  # noqa: E402
from effgen.models.capabilities import Capability  # noqa: E402
from effgen.models.errors import NoCandidateWithinLatencyError  # noqa: E402
from effgen.models.groq_adapter import GroqAdapter  # noqa: E402
from effgen.models.latency_tracker import LatencyTracker  # noqa: E402
from effgen.models.router import PolicyBasedRouter, RoutingContext  # noqa: E402
from effgen.models.routing._probe import reset_probe_cache  # noqa: E402
from effgen.models.routing.latency import LatencyBasedPolicy  # noqa: E402

OUTPUT_DIR = ROOT / "build_plan" / "v0.2.4" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "3-router-latency.txt"

GROQ_MODEL = "llama-3.1-8b-instant"
lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


def section(title: str) -> None:
    log()
    log("=" * 72)
    log(title)
    log("=" * 72)


def require_groq_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not loaded; cannot run live latency validation.")


def make_router(tracker: LatencyTracker) -> PolicyBasedRouter:
    return PolicyBasedRouter(policies=[LatencyBasedPolicy(tracker=tracker)])


def main() -> None:
    require_groq_key()

    section("1. Clear latency tracker state")
    LatencyTracker.reset()
    reset_probe_cache()
    tracker = LatencyTracker.get()
    log(f"Tracker empty: {tracker.is_empty()}")
    assert tracker.is_empty()

    section("2. Seed latency data with 5 live Groq calls")
    adapter = GroqAdapter(GROQ_MODEL)
    adapter.load()
    config = GenerationConfig(max_tokens=10, temperature=0.0)
    try:
        for idx in range(5):
            t0 = time.perf_counter()
            result = adapter.generate(
                "Reply with exactly: LATENCY_OK",
                config=config,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log(f"Groq call {idx + 1}: {elapsed_ms:.0f}ms, text={result.text[:40]!r}")
    finally:
        adapter.unload()

    groq_p50 = tracker.p50("groq", GROQ_MODEL)
    log(f'LatencyTracker.p50("groq", "{GROQ_MODEL}") = {groq_p50!r}')
    assert isinstance(groq_p50, float)

    section("3. Route with latency_budget_ms=500")
    router = make_router(tracker)
    decision_500 = router.route(RoutingContext(
        latency_budget_ms=500,
        required_capabilities={Capability.chat},
    ))
    log(f"Chosen: {decision_500.chosen.provider}/{decision_500.chosen.model_id}")
    log(f"Score: {decision_500.score:.0f}ms")
    assert decision_500.chosen.provider == "groq"
    assert decision_500.chosen.model_id == GROQ_MODEL
    log("PASS: router picked the fastest provider with real data.")

    section("4. Route with latency_budget_ms=100")
    try:
        decision_100 = router.route(RoutingContext(
            latency_budget_ms=100,
            required_capabilities={Capability.chat},
        ))
    except NoCandidateWithinLatencyError as exc:
        log(f"PASS: impossible budget raises NoCandidateWithinLatencyError: {exc}")
    else:
        log(
            "Budget was satisfiable with current data; router chose "
            f"{decision_100.chosen.provider}/{decision_100.chosen.model_id} "
            f"at {decision_100.score:.0f}ms."
        )
        impossible_budget = max(1, int(decision_100.score) - 1)
        try:
            router.route(RoutingContext(
                latency_budget_ms=impossible_budget,
                required_capabilities={Capability.chat},
            ))
        except NoCandidateWithinLatencyError as exc:
            log(
                "PASS: budget below observed p50 raises "
                f"NoCandidateWithinLatencyError: {exc}"
            )
        else:
            raise AssertionError(
                f"Expected NoCandidateWithinLatencyError for {impossible_budget}ms budget"
            )

    section("5. Empty tracker route triggers warm-up probes")
    LatencyTracker.reset()
    reset_probe_cache()
    warm_tracker = LatencyTracker.get()
    warm_router = make_router(warm_tracker)
    before = warm_tracker.all_stats()
    log(f"Stats before route: {len(before)}")
    warm_decision = warm_router.route(RoutingContext(
        latency_budget_ms=5_000,
        required_capabilities={Capability.chat},
    ))
    after = warm_tracker.all_stats()
    log(f"Stats after route: {len(after)}")
    for (provider, model), stats in sorted(after.items()):
        p50 = stats["p50_total_ms"]
        log(f"  {provider}/{model}: p50={p50:.0f}ms n={stats['n']}" if p50 else f"  {provider}/{model}: no total")
    log(f"Warm route chose: {warm_decision.chosen.provider}/{warm_decision.chosen.model_id}")
    assert len(after) > len(before)
    assert any(provider == "groq" for provider, _model in after)
    log("PASS: router-triggered warm-up probe seeded the tracker.")

    section("6. Streaming TTFT is distinct from total latency")
    LatencyTracker.reset()
    stream_tracker = LatencyTracker.get()
    stream_adapter = GroqAdapter(GROQ_MODEL)
    stream_adapter.load()
    try:
        chunks = list(stream_adapter.generate_stream(
            "Count to five in words.",
            config=GenerationConfig(max_tokens=20, temperature=0.0),
        ))
    finally:
        stream_adapter.unload()
    total = stream_tracker.p50("groq", GROQ_MODEL)
    ttft = stream_tracker.p50_ttft("groq", GROQ_MODEL)
    log(f"Stream chunks: {len(chunks)}")
    log(f"Total p50: {total!r}")
    log(f"TTFT p50: {ttft!r}")
    assert isinstance(total, float)
    assert isinstance(ttft, float)
    assert ttft != total
    log("PASS: streaming records TTFT separately from total latency.")

    section("SUMMARY")
    log("Latency routing works on live Groq data; warm-up probe seeds tracker.")
    OUTPUT_FILE.write_text("\n".join(lines))
    log(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
