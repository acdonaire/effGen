# effGen Prompt Library — Framework Overview

The prompt library is a curated, domain-organized catalog of reusable prompt templates.
Every template is a Python function that renders deterministically for fixed inputs, paired
with an evaluation harness and CLI.

## Design Principles

- **Prompts live in code.** Every template is a Python callable that renders to a string.
- **Every prompt has a test.** Golden rendering tests at minimum; live smoke tests where needed.
- **Domain-first taxonomy.** Templates live under `effgen/prompts/library/domains/<domain>/`.
- **Variants are explicit.** Each style (`zero_shot`, `cot`, `few_shot`, `tool`, `structured`) is its own named template.
- **Evaluation first.** Each template ships with a fixture and can be evaluated with `effgen prompts eval`.

## Key Classes

### `LibraryPrompt`

```python
from effgen.prompts.library import LibraryPrompt

@dataclass
class LibraryPrompt:
    name: str            # e.g. "research.literature_review.v1"
    domain: str          # e.g. "research"
    variant: str         # "zero_shot" | "cot" | "few_shot" | "tool" | "structured"
    description: str
    template: Callable[..., str]
    input_schema: dict
    fixture: dict        # default inputs for eval
    expected_shape: dict | None
    tags: list[str]
```

### `PromptRegistry`

Singleton that auto-discovers all domain packages:

```python
from effgen.prompts.library import registry

# List all prompts
for p in registry.all():
    print(p.name, p.domain, p.variant)

# Search
results = registry.search(domain="research", variant="cot")
p = registry.get("research.literature_review.v1")
```

### `PromptEval`

Harness for golden and live evaluation:

```python
from effgen.prompts.library import PromptEval, registry

ev = PromptEval()

# Golden eval (renders with fixture, compares against stored .txt)
result = ev.eval_golden(p)
print(result.passed, result.message)

# Live eval (runs prompt through a model, checks expected_shape)
result = ev.eval_live(p, model="llama3.1-8b")

# Batch eval with report table
report = ev.eval_all_golden(registry.all())
print(report.as_table())
```

## CLI

```bash
# List all templates
effgen prompts list

# Filter by domain and/or variant
effgen prompts list --domain research --variant cot

# Different output formats
effgen prompts list --format json
effgen prompts list --format markdown

# Inspect a specific template
effgen prompts show research.literature_review.v1

# Run golden eval (no model needed)
effgen prompts eval

# Run golden + live eval
effgen prompts eval --domain research --live --model llama3.1-8b

# Write eval table to file
effgen prompts eval --output outputs/eval.txt
```

## Adding a New Domain

1. Create `effgen/prompts/library/domains/<domain>/__init__.py`
2. Create one file per template, e.g. `my_prompt_v1.py`
3. In each file, construct a `LibraryPrompt` and call `registry.register(prompt)`
4. Add a fixture in `tests/prompts/fixtures/<domain>/`
5. Run `effgen prompts eval` to generate the golden file
6. Add tests in `tests/prompts/test_<domain>.py`

The registry auto-discovers packages in `effgen/prompts/library/domains/` at startup.

## Output Shape Specs

The `expected_shape` field controls live eval validation:

```python
# JSON schema validation
expected_shape = {
    "type": "json",
    "schema": {"required": ["summary", "key_points"]}
}

# Regex check
expected_shape = {
    "type": "regex",
    "pattern": r"(?i)introduction|abstract"
}

# Custom callable
expected_shape = {
    "type": "callable",
    "fn": lambda output: len(output.split()) >= 50 or "output too short"
}
```
