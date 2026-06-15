# CLI developer-experience surfaces

Beyond `effgen run`/`chat`/`serve`, the CLI ships a few developer-focused
commands and a shell-completion generator. All of them stay in sync with the
real CLI automatically — the completion lists and preset choices are generated
by introspecting the live parser and registries, never hand-maintained.

## Shell completion

Generate a completion script for your shell and source it:

```bash
# Bash — add to ~/.bashrc
eval "$(effgen --completion bash)"

# Zsh — add to ~/.zshrc
eval "$(effgen --completion zsh)"

# Fish
effgen --completion fish | source
```

The generated scripts complete the current subcommands (`run`, `chat`, `serve`,
`models`, `tools`, `doctor`, `compare`, `debug`, `cost`, `eval`, …), the
`--preset` choices (from the preset registry), and `--tools` names (from the
tool registry). Because they are generated, they cannot drift out of date the
way a hardcoded list does.

## `effgen debug`

Run a task with full ReAct-step inspection — each iteration's thought, action,
tool input/observation, tokens and latency, plus a run summary.

```bash
effgen debug "What is 137 * 19? Use the calculator." -m gpt-5-nano
effgen debug "Plan a 3-step research task" --preset research --step
```

- `-m/--model` or `--preset` is required; with neither, the command prints an
  actionable message and exits `2`.
- `--step` pauses after each iteration so you can inspect the scratchpad.
- Exit code: `0` on success, `1` if the run failed, `2` for a config/usage error.

## `effgen compare`

Compare two or more models on a registered evaluation suite and print an
accuracy/latency matrix with a recommendation.

```bash
effgen compare \
  --models "gpt-5-nano,groq:llama-3.1-8b-instant" \
  --suite conversation \
  --scoring contains
```

- `--suite` accepts a registered suite name (`math`, `tool_use`, `reasoning`,
  `safety`, `conversation`); an unknown name lists the valid ones and exits `2`.
- `--preset` (registry-driven) wires each model into a preset agent.
- `-o/--output` writes the matrix to `.md` or `.json`.
- Exit code: `0` on success, `1` if no model loaded, `2` for an unknown suite.
