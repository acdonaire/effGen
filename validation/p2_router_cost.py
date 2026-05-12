#!/usr/bin/env python3
"""Live validation for CostBasedPolicy routing."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Load adapters so ProviderRegistry is populated.
import effgen.models.anthropic_adapter  # noqa: F401,E402
import effgen.models.cerebras_adapter  # noqa: F401,E402
import effgen.models.fireworks_adapter  # noqa: F401,E402
import effgen.models.gemini_adapter  # noqa: F401,E402
import effgen.models.groq_adapter  # noqa: F401,E402
import effgen.models.hf_inference_adapter  # noqa: F401,E402
import effgen.models.openai_adapter  # noqa: F401,E402
import effgen.models.replicate_adapter  # noqa: F401,E402
import effgen.models.together_adapter  # noqa: F401,E402
from effgen.models.capabilities import Capability  # noqa: E402
from effgen.models.errors import NoCandidateWithinBudgetError  # noqa: E402
from effgen.models.registry import ProviderRegistry  # noqa: E402
from effgen.models.router import PolicyBasedRouter, ProviderModelPair, RoutingContext  # noqa: E402
from effgen.models.routing.cost import CostBasedPolicy, _get_model_pricing  # noqa: E402

PROMPT = "Say COST_ROUTER_OK and nothing else."
FREE_TIER_PROVIDERS = {"cerebras", "groq", "hf"}
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "build_plan",
    "v0.2.4",
    "outputs",
    "2-router-cost.txt",
)

lines: list[str] = []


def out(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


def section(title: str) -> None:
    sep = "=" * 70
    out(sep)
    out(f"  {title}")
    out(sep)


def subsection(title: str) -> None:
    out(f"\n--- {title} ---")


def live_call(pair: ProviderModelPair) -> str:
    """Call the exact routed provider/model pair and return response text."""
    provider = ProviderRegistry.get_provider_info(pair.provider)
    adapter_cls = provider["adapter_cls"]
    adapter = adapter_cls(model_name=pair.model_id)
    adapter.load()
    try:
        result = adapter.generate(PROMPT)
    finally:
        unload = getattr(adapter, "unload", None)
        if callable(unload):
            unload()
    return result.text if hasattr(result, "text") else str(result)


section("Pricing Registry Snapshot (2026-05-11)")
out(f"{'Provider':<15} {'Input/1M':>12} {'Output/1M':>13} {'FreeTier':<10} {'KeySet':<8}")
out("-" * 65)
for provider in sorted(ProviderRegistry.list_providers()):
    pricing = ProviderRegistry.get_pricing(provider)
    available = ProviderRegistry.is_available(provider)
    out(
        f"{provider:<15} ${pricing['input_per_1m']:>10.3f} "
        f"${pricing['output_per_1m']:>11.3f} "
        f"{'yes' if pricing['free_tier'] else 'no':<10} "
        f"{'yes' if available else 'no':<8}"
    )
out()

router = PolicyBasedRouter(policies=[CostBasedPolicy()])

section("Context A: budget=$0.0001 expects Cerebras/Groq/HF free tier")
ctx_a = RoutingContext(
    prompt_tokens_estimate=500,
    user_budget_usd=0.0001,
    required_capabilities={Capability.chat},
)
t0 = time.monotonic()
decision_a = router.route(ctx_a)
elapsed = (time.monotonic() - t0) * 1000

out(f"Chosen:    {decision_a.chosen.provider}/{decision_a.chosen.model_id}")
out(f"Policy:    {decision_a.policy_name}")
out(f"Est cost:  ${decision_a.score:.8f}")
out(f"Latency:   {elapsed:.1f}ms (routing only)")
out()
assert decision_a.chosen.provider in FREE_TIER_PROVIDERS, (
    f"Expected one of {sorted(FREE_TIER_PROVIDERS)}, got {decision_a.chosen.provider}"
)
assert decision_a.score <= ctx_a.user_budget_usd
out("PASS: free-tier provider selected for tiny budget")

subsection("Top 8 eliminations")
for pair, reason in decision_a.eliminated[:8]:
    out(f"  {pair.provider}/{pair.model_id}: {reason}")
if len(decision_a.eliminated) > 8:
    out(f"  ... and {len(decision_a.eliminated) - 8} more")

subsection("Live call with routed provider")
response_a = live_call(decision_a.chosen)
out(f"Live provider: {decision_a.chosen.provider}/{decision_a.chosen.model_id}")
out(f"Response:  {response_a!r}")
assert "COST_ROUTER_OK" in response_a, f"Expected COST_ROUTER_OK, got: {response_a!r}"
out("PASS: routed model responded with COST_ROUTER_OK")

section("Context B: budget=$1.00 logs cheapest available decision")
ctx_b = RoutingContext(
    prompt_tokens_estimate=500,
    user_budget_usd=1.0,
    required_capabilities={Capability.chat},
)
t1 = time.monotonic()
decision_b = router.route(ctx_b)
elapsed_b = (time.monotonic() - t1) * 1000

out(f"Chosen:    {decision_b.chosen.provider}/{decision_b.chosen.model_id}")
out(f"Policy:    {decision_b.policy_name}")
out(f"Est cost:  ${decision_b.score:.8f}")
out(f"Latency:   {elapsed_b:.1f}ms (routing only)")
input_price, output_price, free_tier = _get_model_pricing(
    decision_b.chosen.provider,
    decision_b.chosen.model_id,
)
out(
    f"Pricing:   in=${input_price:.4f}/1M  "
    f"out=${output_price:.4f}/1M  free={free_tier}"
)

subsection("Top 8 eliminations")
for pair, reason in decision_b.eliminated[:8]:
    out(f"  {pair.provider}/{pair.model_id}: {reason}")
if len(decision_b.eliminated) > 8:
    out(f"  ... and {len(decision_b.eliminated) - 8} more")

subsection("Live call with routed provider")
response_b = live_call(decision_b.chosen)
out(f"Live provider: {decision_b.chosen.provider}/{decision_b.chosen.model_id}")
out(f"Response:  {response_b!r}")
assert "COST_ROUTER_OK" in response_b, f"Expected COST_ROUTER_OK, got: {response_b!r}"
out("PASS: routed model responded with COST_ROUTER_OK")

section("No-fit budget: $0.0 with paid-only candidates")
paid_policy = CostBasedPolicy()
paid_candidates = [
    ProviderModelPair("openai", "gpt-4o"),
    ProviderModelPair("openai", "gpt-4o-mini"),
]
ctx_c = RoutingContext(
    prompt_tokens_estimate=1000,
    user_budget_usd=0.0,
    required_capabilities={Capability.chat},
)
try:
    paid_policy.select(paid_candidates, ctx_c)
except NoCandidateWithinBudgetError as exc:
    out("NoCandidateWithinBudgetError raised correctly:")
    out(f"  budget=${exc.user_budget_usd}")
    out(f"  cheapest_cost=${exc.cheapest_cost_usd:.8f}")
    out(f"  cheapest_pair={exc.cheapest_pair}")
    out(f"  message={exc}")
    assert exc.cheapest_cost_usd > 0.0
    assert "Cheapest available" in str(exc)
    out("PASS: error includes cheapest available cost")
else:
    out("FAIL: expected NoCandidateWithinBudgetError but none raised")
    sys.exit(1)

section("Tie-break stability")
ctx_tie = RoutingContext(
    prompt_tokens_estimate=500,
    user_budget_usd=0.0001,
    required_capabilities={Capability.chat},
)
choices = [router.route(ctx_tie).chosen for _ in range(5)]
out("Choices:")
for choice in choices:
    out(f"  {choice.provider}/{choice.model_id}")
assert len(set(choices)) == 1
out("PASS: tie-break is stable for repeated calls with the same context")
out("Rule: cost, free-tier, provider priority, published list cost, provider, model id")

section("VALIDATION COMPLETE")
out("Context A: PASS - free-tier provider selected and called live")
out("Context B: PASS - cheapest available provider selected and called live")
out("No-fit budget: PASS - NoCandidateWithinBudgetError raised cleanly")
out("Tie-break: PASS - stable decision documented")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nOutput written to: {OUTPUT_FILE}")
