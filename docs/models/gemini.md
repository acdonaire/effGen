# Gemini & Gemma in effGen

The `GeminiAdapter` calls Google's Generative Language API. effGen ships
a small registry of free / cheap text-out models so you can pick one
without leaving the framework.

## Quick start

```python
from effgen.models.gemini_adapter import GeminiAdapter
from effgen.models.base import GenerationConfig

model = GeminiAdapter(model_name="gemini-3.1-flash-lite-preview")
model.load()
result = model.generate(
    "Summarise the second law of thermodynamics in one sentence.",
    config=GenerationConfig(max_tokens=64, temperature=0.2),
)
print(result.text)
```

The adapter reads `GOOGLE_API_KEY` from `~/.effgen/.env` (or the project's
`.env`) automatically when `python-dotenv` has been initialized; you can
also pass `api_key=...` explicitly.

## Recommended models (text-out only)

```python
import effgen

# Models that are reliably callable on the free tier today.
for m in effgen.gemini_recommended_models(tier="free"):
    print(m["id"], m["family"], "RPM=", m["rpm"], "RPD=", m["rpd"])

# All registered models (free + premium):
effgen.gemini_recommended_models(tier="all")

# Inspect a single model:
effgen.gemini_model_info("gemini-3.1-flash-lite")  # alias resolves
```

| Model                                | Family       | Tier    | RPM | TPM     | RPD    | Tools | Thinking |
|--------------------------------------|--------------|---------|-----|---------|--------|-------|----------|
| `gemini-3.1-flash-lite-preview`      | flash-lite   | free    | 15  | 250 K   | 500    | yes   | yes      |
| `gemini-3-flash-preview`             | flash        | free    | 5   | 250 K   | 20     | yes   | yes      |
| `gemini-3-pro-preview`               | pro          | premium | —   | —       | —      | yes   | yes      |
| `gemini-3.1-pro-preview`             | pro          | premium | —   | —       | —      | yes   | yes      |
| `gemini-2.5-flash-lite`              | flash-lite   | free    | 10  | 250 K   | 20     | yes   | no       |
| `gemini-2.5-flash`                   | flash        | free    | 5   | 250 K   | 20     | yes   | no       |
| `gemini-2.5-pro`                     | pro          | premium | —   | —       | —      | yes   | yes      |
| `gemini-2.0-flash`                   | flash        | premium | —   | —       | —      | yes   | no       |
| `gemini-2.0-flash-lite`              | flash-lite   | premium | —   | —       | —      | yes   | no       |
| `gemma-3-1b` / `3-4b` / `3-12b` / `3-27b` | gemma   | free    | 30  | 15 K    | 14 400 | no    | no       |
| `gemma-4-26b` / `gemma-4-31b`        | gemma        | free    | 15  | unlim.  | 1 500  | no    | no       |

Limits reflect Google's free-tier defaults as of 2026-04-25 — check
[ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)
for the current numbers.

## Rate-limit handling

`GeminiAdapter` honors the `retry_delay` Google attaches to 429
`ResourceExhausted` errors. If you trip the per-minute quota the adapter
sleeps for the duration the SDK suggested, retries up to
`MAX_RATE_LIMIT_RETRIES` (default 5), and only then re-raises. Transient
5xx / `DEADLINE_EXCEEDED` errors fall back to exponential backoff.

For tight free-tier loops, prefer Gemma models (30 RPM, 14 400 RPD) or
batch your work with longer sleeps between bursts.

## Tool calling

Native function calling works end-to-end for Gemini 2.0+ and Gemini 3.x.
Gemma models do **not** support function calling; pair them with the
ReAct strategy or no tools.

```python
from effgen.core.agent import Agent, AgentConfig
from effgen.models.gemini_adapter import GeminiAdapter
from effgen.tools.builtin import Calculator

model = GeminiAdapter(model_name="gemini-3.1-flash-lite-preview")
model.load()
agent = Agent(config=AgentConfig(
    name="GeminiAgent",
    model=model,
    tools=[Calculator()],
    max_iterations=6,
))
print(agent.run("Use the calculator to compute (137 * 248) - 1024.").output)
```

Under the hood the adapter:

1. Converts effGen tool specs (OpenAI dict format) into Gemini
   `FunctionDeclaration` objects, stripping JSON-Schema fields Google
   doesn't accept (`minLength`, `maximum`, etc.).
2. Re-emits Gemini's structured `function_call` parts as Qwen-style
   `<tool_call>{...}</tool_call>` tokens so the agent's native parser
   can dispatch them with no Gemini-specific code path.

## Model aliases

These short forms resolve to the canonical preview ID in the registry:

| Alias                       | Canonical                          |
|-----------------------------|------------------------------------|
| `gemini-3.1-flash-lite`     | `gemini-3.1-flash-lite-preview`    |
| `gemini-3-flash`            | `gemini-3-flash-preview`           |
| `gemini-3-pro`              | `gemini-3-pro-preview`             |
| `gemini-3.1-pro`            | `gemini-3.1-pro-preview`           |
| `gemini-flash-lite-latest`  | `gemini-3.1-flash-lite-preview`    |

Both forms are callable directly against the API.
