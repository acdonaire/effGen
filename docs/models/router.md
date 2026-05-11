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
