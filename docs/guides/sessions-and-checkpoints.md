# Sessions, Checkpoints & Background Tasks

This guide covers the durability surfaces effGen ships for long-lived work:
persistent **sessions** (multi-turn memory across processes), **checkpoints**
(resume a long task where it left off), and **background tasks** (run agent work
off the main thread).

## Sessions — persistent multi-turn memory

A session stores a conversation so a later process can pick it up and the model
*remembers* earlier turns.

```python
from effgen import Agent, AgentConfig

# Same session_id in two separate runs → the second remembers the first.
a1 = Agent(AgentConfig(model="gpt-5-nano", provider="openai"), session_id="user-123")
a1.run("My dog is named Pixel.")

a2 = Agent(AgentConfig(model="gpt-5-nano", provider="openai"), session_id="user-123")
print(a2.run("What is my dog's name?").output)   # -> "Pixel"
```

From a preset, pass `session_id=` the same way:

```python
from effgen import create_agent
agent = create_agent("math", "gpt-5-nano", session_id="user-123")
```

On the CLI, every command that runs an agent accepts `--session-id`:

```bash
effgen run --session-id user-123 "My favorite color is teal."
effgen run --session-id user-123 "What is my favorite color?"   # remembers
```

### Managing sessions

```bash
effgen sessions list                 # ids, message counts, timestamps, storage dir
effgen sessions export <id>          # JSON (default) — a documented, reloadable format
effgen sessions export <id> --format text
effgen sessions delete <id>          # exit 0 if removed, 1 if not found
effgen sessions cleanup --days 30    # delete sessions not updated in N days
```

Exit codes follow the CLI convention: `0` success, `1` user error (e.g. a
missing id), `2` a corrupt file (see below).

### Where sessions are stored

By default sessions live in `~/.effgen/sessions/<id>.json`. Override the
location with environment variables (resolved at call time):

| Variable | Effect |
|---|---|
| `EFFGEN_SESSIONS_DIR` | Exact directory for session files. |
| `EFFGEN_HOME` | Base effGen state dir; sessions go in `$EFFGEN_HOME/sessions`. |

Saves are **atomic** (written to a temp file then renamed) so a crash mid-write
can never leave a truncated session that fails to load later.

To cap unbounded growth, schedule `effgen sessions cleanup --days N` (e.g. from
cron) or call `SessionManager(...).cleanup(older_than_days=N)`.

## Checkpoints & resume

A checkpoint snapshots an in-progress run (scratchpad, iteration, memory, the
model id) so it can be resumed. Write checkpoints by passing a directory:

```bash
effgen run --checkpoint-dir ./checkpoints --checkpoint-interval 1 "Long task..."
```

Resume — by default the latest checkpoint in the directory:

```bash
effgen resume --checkpoint ./checkpoints           # newest checkpoint
effgen resume --checkpoint ./checkpoints/<id>.json # a specific file
```

Resume reuses the **model the checkpoint was created with**. Pass `-m/--model`
to override; if it differs from the saved model you get a clear warning. If the
checkpoint predates model tracking, resume falls back to a small local model and
tells you to pass `-m`.

Programmatically:

```python
from effgen import Agent, AgentConfig

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))
agent.run("Long task...", checkpoint_dir="./checkpoints", checkpoint_interval=1)

# Later, on a fresh agent built with the same tools:
resumed = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))
result = resumed.resume(checkpoint_dir="./checkpoints")
```

Checkpoints are **JSON only** (no pickle). A truncated or corrupt checkpoint
raises a clear `CorruptStateError` that names the file rather than a stack
trace; a missing one raises `FileNotFoundError`. A SQLite backend is also
available: `CheckpointManager(dir, backend="sqlite")`.

## Background tasks

Run agent tasks on worker threads without blocking:

```python
from effgen.core.background import BackgroundTaskRunner

with BackgroundTaskRunner(agent, max_workers=2) as runner:
    task_id = runner.submit("Summarize this report...")
    result = runner.get_result(task_id, wait=True, timeout=60)
    print(runner.get_status(task_id))   # COMPLETED / FAILED / CANCELLED
```

- `submit` / `get_status` / `get_result` / `cancel` / `pause` / `resume`.
- A failing task is reported as `FAILED` with a typed, **secret-redacted**
  error (never a silent success).
- The runner stops its worker threads when it is closed, used as a context
  manager, garbage-collected, or at interpreter exit — so a forgotten
  `shutdown()` never leaks threads. Prefer the `with` form for prompt cleanup.
