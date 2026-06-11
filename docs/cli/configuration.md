# CLI configuration & environment

This page documents how the `effgen` command-line tool finds your API keys,
controls log verbosity, and selects providers.

## API keys and `.env` discovery

`effgen` loads `.env` files automatically before running any command. The search
order is **documented and predictable** (earlier entries win; a real
environment variable always beats a file):

1. **`$EFFGEN_DOTENV`** — an explicit path you set, if any.
2. **`~/.effgen/.env`** — your per-user effGen config.
3. **`./.env` and each parent directory** up to the filesystem root — the nearest
   project `.env` to your current working directory (e.g. a checkout's repo root).

So a key works whether you keep it in `~/.effgen/.env`, in a project `.env`, or
exported in your shell. To point at a non-standard file:

```bash
export EFFGEN_DOTENV=/secure/keys/effgen.env
effgen doctor
```

Check what keys effGen can see (no secrets are printed):

```bash
effgen doctor              # key present / missing per provider, plus a system report
effgen doctor --live --cheap  # also makes a tiny call to confirm each default model is usable
```

## Selecting a provider and model

A model id can be given three ways:

```bash
# 1. Local HuggingFace repo (downloaded / on disk)
effgen run "What is 25 * 17?" -m Qwen/Qwen2.5-1.5B-Instruct

# 2. Provider-prefixed id
effgen run "Tell me a joke" -m groq:llama-3.1-8b-instant

# 3. Bare id plus --provider
effgen run "Summarize quantum computing" -m gpt-5-nano --provider openai
```

Valid providers: `openai`, `anthropic`, `gemini`, `cerebras`, `groq`,
`together`, `fireworks`, `replicate`, `hf`. A typo (e.g. `grok`) fails fast with a
suggestion instead of silently downloading a local model.

Browse what's available:

```bash
effgen models list                    # provider registry overview + local cache
effgen models list --provider groq    # full per-model detail (context, price, tools…)
effgen models list --free --tools      # filter to free, tool-capable models
effgen models info gpt-5-nano          # one model's full record
effgen models refresh                  # update the catalog from each keyed provider's live API
```

## Output control

- `-q` / `--quiet` — errors only (clean, scriptable output).
- `-v` / `--verbose` — show DEBUG/INFO diagnostics.
- `--log-file PATH` — capture full detail to a file while the console stays quiet.
- `--json` — machine-readable output for `doctor`, `models list`, `tools list`,
  and `cost`.

By default the CLI is quiet: tables and answers print cleanly with no library log
noise.

## Examples

`effgen examples list` / `effgen examples run <name>` work from a source checkout.
Examples ship with the repository (not the installed wheel); set
`EFFGEN_EXAMPLES_DIR=/path/to/examples` to point at them explicitly.

## Health checks

`effgen health` contacts external services (the project site and PyPI). These
network checks are **opt-in** — run `effgen health --remote` (or set
`EFFGEN_HEALTH_REMOTE=1`) to enable them. Without the flag, no network request is
made.
