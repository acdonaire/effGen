# effgen code — the coding agent

`effgen code` runs a plan → act → observe loop over a workspace: the model
proposes an approach, writes files, executes code in the configured sandbox,
reads the **real** output, and iterates until the task is done or the iteration
cap is reached. Every file change is shown as a unified diff before it touches
disk, every write and command passes a permission gate, and the files the agent
edits stay inside the workspace root.

```bash
effgen code                                     # interactive session
effgen code "write fib.py with fib(n) and print fib(10)" --auto-edit
effgen code -p "add a --retries flag to cli.py" --auto-edit --json | jq .files_written
cat pytest.log | effgen code -p "why did this fail?"
effgen code --undo                              # revert the last applied edit
effgen code "fix the failing test" --auto-edit --commit
```

On a terminal with no task it opens an interactive session. A task argument,
`-p`, piped stdin, or `--json` runs once and exits.

## Stdin

When stdin is not a terminal it is read **to EOF before the run starts**. With a
task already given the piped text becomes context in front of it; with no task
the piped text *is* the task:

```bash
cat pytest.log | effgen code -p "why did this fail?"   # log becomes context
echo "add a --retries flag to cli.py" | effgen code    # stdin is the task
```

Reading to EOF is what lets a producer that writes slowly — a build log, `tail
-f` — be folded in whole. The consequence is that a pipe which never closes
holds the run: an open stdin inherited from a supervisor, an agent harness or a
job-control shell keeps the command waiting even though a task was supplied.
That wait is announced rather than silent — after about two seconds the command
prints a line to stderr naming what it is waiting for:

```
Reading piped stdin as context before starting; the stream has not closed.
Close it (Ctrl-D) or re-run with < /dev/null to skip it.
```

With no task given the piped text is the task, and the note names that instead:

```
Reading the task from piped stdin; the stream has not closed.
Close it (Ctrl-D), or pass the task with -p to skip the read.
```

Redirect stdin from `/dev/null` when a caller has no context to pipe in:

```bash
effgen code -p "write fib.py" -w WS --auto-edit --json < /dev/null
```

The note goes to stderr, so stdout still carries only the answer or the JSON
document.

## The workspace

The agent reads, edits and runs in one directory:

1. `-w/--workspace DIR` if given (created if it does not exist),
2. otherwise `EFFGEN_WORKSPACE`,
3. otherwise the directory the command was started in.

A file path outside that root is refused with the path named, not silently
redirected. The deny-list for credential files and sensitive locations
(`~/.ssh`, `~/.aws`, `.env`, …) applies inside the workspace too. Shell commands
and executed code start in the workspace, and how far they can reach is decided
by the sandbox backend below.

Executed code runs in the sandbox effGen already ships — Docker when its daemon
is reachable, otherwise a subprocess sandbox that isolates the network and
confines writes to the workspace, leaving the rest of the filesystem readable
but read-only. `EFFGEN_SANDBOX_BACKEND=docker|subprocess` pins one. A write
outside the workspace fails with a read-only-filesystem error naming where
writes are allowed, so the model corrects the path instead of silently
scattering files across the machine. Only Docker also confines *reads*.

Confinement is not assumed — the sandbox proves it can hold before claiming it,
and reports what each run actually enforced. A host that allows the namespaces
but refuses the locked read-only mounts keeps network isolation and says writes
are not confined; a host without unprivileged user namespaces at all isolates
neither, and says that too. On those hosts a workspace under the system temp
directory is written but not readable from executed code, and the command
reports that before the run rather than letting it fail mid-loop. `effgen
doctor` names whichever of the three states applies.

`effgen doctor` reports all of this — the workspace it would use, the sandbox
backend, and whether git is present — with the fix for anything that is not
ready.

## Permission modes

Pick at most one; naming two is an error rather than a silent precedence rule.

| Mode | Writes | Sandboxed runs | Shell commands | Commit |
|---|---|---|---|---|
| `--plan` | proposed only | no | no | no |
| default, with a terminal | confirm each | confirm each | confirm each | confirm |
| `--auto-edit` | applied | applied | confirm each | confirm |
| `--yes` | applied | applied | applied | applied |

Without a terminal there is nobody to confirm, so the default becomes `--plan`:
the run reports what it *would* do and writes nothing. Opt in explicitly with
`--auto-edit` or `--yes` to let a scripted run change files. A run that
completed but withheld its changes for that reason exits `2`.

## Diffs, apply/reject and undo

Each proposed edit is rendered as a colorized unified diff **before** the write
is decided, so a change is visible in every mode — including `--plan` and
including piped output (the diff goes to stderr, keeping stdout for the result).

In the interactive session `/plan` stages edits without writing them, `/diff`
shows what is staged, `/apply [n]` writes them (all, or one by number),
`/reject [n]` discards them. A hunk that no longer applies because the file
changed underneath is reported by name; the remaining hunks still apply.

Every applied edit is journaled per workspace, so it can be reversed:

```bash
effgen code --undo             # reverse the last applied edit
effgen code --undo --undo-count 3
```

`/undo [n]` does the same inside a session. A restored file returns to its
previous content; a file the run created is removed.

## Repository awareness

In a git repository the branch, a short `git status` and a bounded file layout
(ignored files excluded) are read before the first model call and become part of
the agent's context, along with an `AGENTS.md` in the workspace if there is one.
Outside a repository the same inventory is built from the workspace itself.

`/git` surfaces the read-only view (`status`, `diff`, `log`, `branch`, `show`,
`remote`) and `/git staged` pulls the staged patch into context.

The single repository change a session can make is a commit of the files it
wrote, and only after an explicit confirmation:

```bash
effgen code "fix the failing test" --auto-edit --commit
```

The plan — repository, exact paths, message — is printed before the y/N. Only
those paths are staged; work you had staged yourself stays staged and out of the
commit. Push, force-push, tag, reset, checkout, rebase and stash are refused in
every mode, including when the model tries to reach them through the shell.

## The interactive session

| Command | Does |
|---|---|
| `/plan` | Propose a change without writing it |
| `/diff` `/apply [n]` `/reject [n]` | Review and decide the staged edits |
| `/undo [n]` | Reverse the last applied edit(s) |
| `/run <cmd>` `/test [args]` | Run a command or the test suite in the workspace |
| `/context` `/add <file>` `/drop <file>` | Manage the files the agent sees, with a size estimate |
| `/clear` `/reset` `/compact` | Reset live context, clear memory, or summarize a long session |
| `/mode [ask\|auto-edit\|yes\|plan]` | Show or change the permission mode |
| `/model <id>` | Hot-swap the model, carrying the conversation |
| `/git [...]` | Repository status/diff/log, staged diff, confirmed commit |
| `/save` `/session` `/load` | Save, name and resume a coding session |
| `/cost` `/trace` `/tools` `/doctor` | Session totals, the last turn's steps, the tool list, an environment check |
| `/help` `/exit` | The command table; leave the session |

Type `/` on its own for the menu; tab completes command names.

## Scripting and non-TTY behavior

Piped or with `--json`, stdout carries only the result — the answer text, or one
JSON document — and everything a human reads goes to stderr:

```bash
echo "summarize what this module does" | effgen code
effgen code -p "add type hints to utils.py" --auto-edit --json | jq '.files_written, .cost_usd'
```

The JSON document carries the answer, `files_written`, the diffs, the full
action log (what was allowed, withheld, declined or refused, and why),
iterations, tool calls, tokens, cost and duration. Every proposed edit appears
in `diffs`; the ones that reached disk carry `"applied": true`, so
`--plan --json` reports the changes it would make without writing any of them.
`tool_calling` names the path the model's tool calls travelled on — `hybrid` and
`native` send the tool definitions to the provider's tool-calling API, `react`
reads the calls out of the model's text — and the report prints the same line
under its summary.

## When the model describes a call instead of making one

Some small models answer with the tool call written out as text:

```
<file_operations> {"operation": "write", "path": "greet.py", …} </file_operations>
```

Nothing is written and nothing runs, so that turn is reported as a failure
(`"reason": "written_tool_call"`, exit code 1) naming the tool whose call was
written out and what to do about it, rather than as an answer describing work
that did not happen. The remedy depends on the path the run used: a model that
was sent the tool definitions natively and still answered in prose needs
replacing with one that calls tools, while a run on the `react` text path can
ask for the native path instead. Larger instruct models and the current cloud
models complete the loop; `effgen models list` marks the models that advertise
tool calling.

An answer that *recaps* a call the run really made — asking the agent to report
the arguments it used, for instance — keeps its result: the turn is reported as
a failure only when the tool named in the block never ran, or when the answer is
nothing but the block. A call inside backticks or a fenced block is
documentation and is left alone either way.

## When the run stops at its iteration cap

A run that spends every step without writing a final answer has no answer to
report (`"reason": "max_iterations_partial"`, or `max_iterations_exhausted` when
nothing was recovered). The answer states what stopped it and what to do — raise
the cap, or run the task on a model that needs fewer steps — and whatever the
run had reached is reported separately, as `partial_output` in `--json` and
under a *Partial progress* heading in the terminal. It is tool output and
reasoning, never presented as a result.

Exit codes: `0` completed, `1` failed, `2` completed but changes were withheld
because there was no terminal to confirm on and no `--auto-edit`/`--yes` — which
includes a `--commit` that could not be confirmed.

`NO_COLOR`, `--no-animation` and a non-terminal stdout all render plain text
with no escape codes.

## First run

`effgen quickstart` (and its alias `effgen tutorial`) offers a coding step that
writes and runs a small program end to end inside `~/.effgen/quickstart-code`,
so the whole loop can be seen before it is pointed at real code. `--code` runs
that step without asking (needed alongside `--yes`, which otherwise skips it);
`--no-code` skips it.
