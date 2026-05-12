# Model Router

effGen includes a policy-based router that automatically selects the best cloud
provider for each request based on capabilities, cost, and latency.

## Quick start

```python
from effgen.models.router import ModelRouter, RoutingContext
from effgen.models.capabilities import Capability
from effgen.models.routing.first_available import FirstAvailablePolicy

# Import adapters to register providers
import effgen.models.cerebras_adapter
import effgen.models.openai_adapter
import effgen.models.groq_adapter
# ... (or use effgen.models to import all at once)

router = ModelRouter(policies=[FirstAvailablePolicy()])
decision = router.route(RoutingContext(required_capabilities={Capability.chat}))

print(decision.chosen.provider, decision.chosen.model_id)
print("Eliminated:", [(p.provider, r) for p, r in decision.eliminated])
```

## RoutingContext

```python
@dataclass
class RoutingContext:
    prompt_tokens_estimate: int = 0          # estimated input tokens
    user_budget_usd: float | None = None     # max cost per call (None = unlimited)
    latency_budget_ms: int | None = None     # max latency in ms (None = unlimited)
    required_capabilities: set[Capability]   # features the provider must support
```

## Capability flags

| Flag          | Meaning                                          |
|---------------|--------------------------------------------------|
| `chat`        | Basic text completion / chat                     |
| `tools`       | Function / tool calling                          |
| `streaming`   | Token streaming via generate_stream()            |
| `vision`      | Image inputs                                     |
| `grounding`   | Web grounding / Google Search integration        |
| `thinking`    | Extended thinking / chain-of-thought reasoning   |
| `json_schema` | Structured output with JSON schema constraint    |

## RouterDecision

Every `router.route()` call returns a `RouterDecision`:

```python
@dataclass
class RouterDecision:
    chosen: ProviderModelPair          # the selected (provider, model_id)
    eliminated: list[tuple[...]]       # (pair, reason) for each rejected candidate
    policy_name: str                   # name of the winning policy
    score: float                       # numeric score (policy-defined)
```

Decisions are always explainable — `eliminated` records every candidate that was
rejected and why.

## Provider capability matrix

| Provider    | chat | tools | streaming | vision | grounding | thinking | json_schema |
|-------------|------|-------|-----------|--------|-----------|----------|-------------|
| cerebras    | ✓    | ✓     | ✓         |        |           |          | ✓           |
| openai      | ✓    | ✓     | ✓         | ✓      |           | ✓        | ✓           |
| groq        | ✓    | ✓     | ✓         |        |           |          | ✓           |
| anthropic   | ✓    | ✓     | ✓         | ✓      |           | ✓        |             |
| gemini      | ✓    | ✓     | ✓         | ✓      | ✓         | ✓        | ✓           |
| together    | ✓    | ✓     | ✓         |        |           |          | ✓           |
| fireworks   | ✓    | ✓     | ✓         |        |           |          | ✓           |
| replicate   | ✓    |       | ✓         | ✓      |           |          |             |
| hf          | ✓    |       | ✓         |        |           |          |             |

## Policies

### FirstAvailablePolicy

Returns the first provider (alphabetical) that has a configured API key and
supports all required capabilities.

```python
from effgen.models.routing.first_available import FirstAvailablePolicy
router = ModelRouter(policies=[FirstAvailablePolicy()])
```

`PolicyBasedRouter` is also exported for callers who prefer an explicit
provider-policy class name.

### CostBasedPolicy — cheapest-first routing

`CostBasedPolicy` selects the cheapest provider/model pair that:
1. Has a configured API key.
2. Supports all required capabilities.
3. Has an estimated cost ≤ `user_budget_usd` (if set).

**Cost estimate:**
```
cost = input_per_1m * prompt_tokens / 1_000_000
     + output_per_1m * expected_output_tokens / 1_000_000
```

Free-tier providers rank ahead of paid providers when cost is equal. Remaining
ties are deterministic: provider priority (`groq`, `cerebras`, `hf`, `gemini`,
`together`, `fireworks`, `openai`, `replicate`, `anthropic`), then published
list cost, provider name, and model id.

#### Cheapest-first recipe

```python
from effgen.models.routing.cost import CostBasedPolicy
from effgen.models.router import ModelRouter, RoutingContext
from effgen.models.capabilities import Capability
import effgen.models.cerebras_adapter  # register providers
import effgen.models.groq_adapter
import effgen.models.openai_adapter

# Always pick the free-tier provider if one is available
router = ModelRouter(policies=[CostBasedPolicy()])
decision = router.route(RoutingContext(
    prompt_tokens_estimate=500,
    user_budget_usd=0.001,      # $0.001 max per call
    required_capabilities={Capability.chat},
))
print(f"Chose {decision.chosen.provider}/{decision.chosen.model_id}")
print(f"Estimated cost: ${decision.score:.6f}")
```

#### Budget guard

If no candidate fits the budget, `NoCandidateWithinBudgetError` is raised
with the cheapest available option:

```python
from effgen.models.errors import NoCandidateWithinBudgetError

try:
    decision = router.route(RoutingContext(
        prompt_tokens_estimate=1_000_000,
        user_budget_usd=0.001,
        required_capabilities={Capability.chat},
    ))
except NoCandidateWithinBudgetError as e:
    print(f"Budget too tight. Cheapest available: ${e.cheapest_cost_usd:.6f}")
    print(f"  from {e.cheapest_pair[0]}/{e.cheapest_pair[1]}")
```

#### Provider Pricing (verified 2026-05-11)

| Provider   | Routing input/1M | Routing output/1M | Notes |
|------------|------------------|-------------------|-------|
| cerebras   | $0.00            | $0.00             | Free tier, rate-limited |
| groq       | $0.00            | $0.00             | Free developer tier; paid list prices retained per model |
| hf         | $0.00            | $0.00             | HF routed requests include free-tier credits |
| together   | $0.03            | $0.12             | Cheapest published serverless chat model |
| gemini     | $0.10            | $0.40             | Paid Flash-Lite price; free quota documented below |
| fireworks  | $0.10            | $0.10             | Cheapest serverless tier for <4B text models |
| openai     | $0.05            | $0.40             | Cheapest current text model, `gpt-5-nano` |
| anthropic  | $1.00            | $5.00             | Claude Haiku 4.5 default |
| replicate  | $0.80            | $4.00             | Fallback for token-priced LLMs; many public models bill by hardware time |

Provider defaults are used when per-model pricing is not available.
Per-model pricing in model registry dicts takes precedence.

**Gemini free-tier note:** Flash and Flash-Lite have free quotas, but the free
tier has stricter rate limits and data-use differences. `CostBasedPolicy`
therefore uses Gemini's published paid token prices for budget checks.

**Standard routing note:** Models marked as requiring a dedicated endpoint are
eliminated by `CostBasedPolicy`; start the endpoint explicitly before routing
to those models directly.

## Writing a custom policy

```python
from effgen.models.router import RoutingPolicy, NoCandidateError, RouterDecision

class MyPolicy(RoutingPolicy):
    @property
    def name(self) -> str:
        return "my_policy"

    def select(self, candidates, context):
        for pair in candidates:
            # custom selection logic
            return RouterDecision(chosen=pair, policy_name=self.name, score=1.0)
        raise NoCandidateError("no candidate found")
```

## Back-compat note

The complexity-based `ModelRouter(models=[...]).select(...)` path for local
model pools is preserved. Passing `policies=[...]` enables provider-policy
routing through `ModelRouter.route(...)`. `PolicyBasedRouter` remains exported
as the lower-level provider-policy class.
