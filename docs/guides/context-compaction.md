# Context compaction

A conversation eventually outgrows the model's context window, and something
has to go. Which turns survive changes the answer more for a small model than
for a frontier one, and different tasks want different answers, so effGen makes
the choice a strategy rather than a fixed rule.

The default, `SummarizeOldest`, is what effGen has always done: once the history
passes a fraction of the window, everything but the most recent few turns
becomes a summary.

```python
from effgen import Agent, AgentConfig
from effgen.memory.compaction import KeepFirstAndLast

agent = Agent(AgentConfig(
    model="Qwen/Qwen2.5-7B-Instruct",
    compaction_strategy=KeepFirstAndLast(first=2, last=6),
))
```

Strategies can also be named:

```python
AgentConfig(model=..., compaction_strategy="drop_oldest")
```

## What ships

| Strategy | What survives | Model call? |
|---|---|---|
| `SummarizeOldest` *(default)* | The most recent turns verbatim; everything older becomes a summary. | Yes |
| `DropOldest` | The most recent turns. The rest is forgotten. | No |
| `KeepFirstAndLast` | The opening turns and the recent ones; the middle is summarized or dropped. | Optional |
| `KeepToolResults` | Tool results and the recent turns; older reasoning is compacted. | Optional |

**`DropOldest`** costs nothing and waits for nothing, and cannot invent
anything — a summary is a model output, with all that implies. Use it when
older turns genuinely do not matter.

**`KeepFirstAndLast`** exists because the opening turns usually carry the task
— the document, the instruction, the constraint everything else refers to — and
a summary of those is a poor substitute. The redundancy in a long conversation
is in the middle.

**`KeepToolResults`** suits a tool-heavy run, where the reasoning is most of the
tokens and the tool results are the evidence the answer rests on.

## Measuring the history

By default the history is measured with the model's own tokenizer when there is
one, and otherwise estimated at four characters per token. Supply a tokenizer to
measure it in the units the window is actually measured in:

```python
import tiktoken

AgentConfig(model=..., tokenizer=tiktoken.get_encoding("cl100k_base"))
```

Anything with `count_tokens(text)` or `encode(text)` works. A tokenizer that
raises falls back to the estimate rather than failing the run.

## Writing your own

Override what differs; the defaults are `SummarizeOldest`'s.

```python
from effgen.memory.compaction import CompactionStrategy

class DropFailedToolTurns(CompactionStrategy):
    """Compact the turns where a tool errored; keep everything else."""

    def messages_to_compact(self, memory):
        return [m for m in memory.messages if "Error executing tool" in m.content]

    def summarize(self, memory, messages):
        return None   # drop them rather than summarize them
```

The three methods, called in this order:

| Method | Answers | Default |
|---|---|---|
| `should_compact(memory)` | Is it time? | Past `summarization_threshold` × the window |
| `messages_to_compact(memory)` | Which messages leave? | Everything but the recent few |
| `summarize(memory, messages)` | What replaces them? | A generated summary; `None` drops them |

Returning an empty list from `messages_to_compact` cancels that round.

## Related

- [Sessions and checkpoints](sessions-and-checkpoints.md) — persisting a
  conversation across processes.
- [Architecture](../architecture/overview.md) — where short-term memory sits in the three-tier system.
