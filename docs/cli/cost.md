# effgen cost - Spend Dashboard & Budget Management

The `effgen cost` subcommand gives you a live view of your API spend and lets you set budget guardrails that automatically trigger provider failover.

## Subcommands

| Subcommand | Description |
|---|---|
| `effgen cost today` | Per-provider/model summary for the last 24 hours |
| `effgen cost week` | Rolling 7-day spend summary |
| `effgen cost by-provider` | Lifetime totals grouped by provider |
| `effgen cost set-budget <amount>` | Set a daily budget in USD |
| `effgen cost clear-budget` | Remove configured budget limits |

## Viewing spend

```bash
# Today's spend (last 24 hours)
effgen cost today

# Last 7 days
effgen cost week

# All-time totals
effgen cost by-provider
```

Example output:

```
                  effGen Cost Summary - Last 24 hours
+----------+-----------------------+----------+---------------+---------------+------------+
| Provider | Model                 | Requests | Prompt Tokens |  Compl Tokens | Cost (USD) |
+----------+-----------------------+----------+---------------+---------------+------------+
| openai   | gpt-4o-mini           |       12 |        48,300 |         9,210 |  $0.008265 |
| groq     | llama-3.1-8b-instant  |       35 |       105,000 |        17,500 |  $0.000000 |
| cerebras | gpt-oss-120b           |       28 |        84,000 |        14,000 |  $0.000000 |
+----------+-----------------------+----------+---------------+---------------+------------+

Total: 75 requests  $0.008265 USD
Daily budget: [########............] $0.0083 / $1.0000 (1%)
```

## Setting budgets

```bash
# Set a $1 daily budget
effgen cost set-budget 1.0

# Or via the config subcommand
effgen config set budget.daily 1.0

# Optional rolling 30-day budget
effgen config set budget.monthly 20.0
```

Budget state is stored in `~/.effgen/budget.json`:

```json
{
  "daily": 1.0,
  "monthly": 20.0
}
```

## Budget alerts

When a budget is configured, every paid API call checks cumulative daily and monthly spend. Zero-cost calls are still allowed after a budget is exhausted so router failover can use free-tier providers.

| Threshold | Behaviour |
|---|---|
| >= 80% of budget | `UserWarning` emitted when the threshold is crossed |
| >= 100% of budget | `BudgetExceededError` raised for paid calls; router fails over |

### Failover on budget exceed

`BudgetExceededError` is classified as *retriable* by the router's `RetryPolicy`. When the router catches it, it:

1. Excludes the provider that triggered the error.
2. Re-routes to the next best candidate (e.g., a free-tier provider like Cerebras or Groq).
3. Emits a `RouterEvent` with `reason="budget_exceeded_daily"`.

```python
from effgen.models.router import PolicyBasedRouter, RoutingContext
from effgen.models.routing.first_available import FirstAvailablePolicy
from effgen.models.capabilities import Capability

router = PolicyBasedRouter(
    policies=[FirstAvailablePolicy()],
    failover_hops=3,
)

events = []
router.subscribe(lambda ev: events.append(ev))

def call_model(pair):
    from effgen import load_model
    m = load_model(f"{pair.provider}:{pair.model_id}")
    return m.generate("Hello")

result = router.route_and_execute(
    RoutingContext(required_capabilities={Capability.chat}),
    call_model,
)
```

If the daily budget is hit during `call_model`, the router automatically falls over to the next available provider without any changes to your calling code.

## Removing a budget

```bash
effgen cost clear-budget
```

## Where data is stored

Cost events are persisted in `~/.effgen/costs.sqlite`.  Each API call through effGen's adapters writes one row:

```
cost_events(provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp)
```

The file is written automatically when you use `CostTracker.get()` (the default singleton).

## Programmatic access

```python
from effgen.models._cost import CostTracker
from effgen.models._cost_store import SQLiteCostStore

# Global tracker (SQLite-backed)
tracker = CostTracker.get()

# Query today's events directly
store = SQLiteCostStore()          # ~/.effgen/costs.sqlite
events = store.query_today()       # last 24 hours
events = store.query_week()        # last 7 days
events = store.query_all()         # lifetime

for ev in events:
    print(f"{ev.provider}/{ev.model}: ${ev.cost_usd:.6f}")

# In-memory tracker (no persistence, back-compat)
mem_tracker = CostTracker(storage=None)
```
