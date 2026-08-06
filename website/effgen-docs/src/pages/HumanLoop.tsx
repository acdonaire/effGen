import React from 'react';
import { Link } from 'react-router-dom';
import { Users } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function HumanLoop() {
  return (
    <DocPage
      title="Human-in-the-Loop"
      subtitle="Pause agents for approvals, clarifications, free-text input, or feedback — with per-tool approval modes and timeouts."
      icon={<Users size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Safety', path: '/guardrails' },
        { label: 'Human-in-the-Loop' },
      ]}
    >
      <h2>Interaction Points</h2>
      <FeatureList
        features={[
          { icon: '✅', title: 'HumanApproval', description: 'Ask a human to approve / deny before executing a tool.' },
          { icon: '❓', title: 'HumanInput', description: 'Pause for free-text input from a user.' },
          { icon: '🔘', title: 'HumanChoice', description: 'Present options; return the selected index.' },
          { icon: '🙋', title: 'ClarificationRequest', description: 'Bundle a question with options + free-text; detects ambiguity automatically.' },
        ]}
      />

      <p>All of the above support <code>timeout</code> via a background <code>ThreadPoolExecutor</code>.</p>

      <h2>Tool Approval</h2>
      <p>
        The most common use case — require human approval before a tool runs. Controlled by
        three <code>AgentConfig</code> fields and one <code>ToolMetadata</code> flag.
      </p>

      <ApiTable
        headers={['Field', 'Type', 'Meaning']}
        rows={[
          [<code>approval_mode</code>, 'str', '"always" | "first_time" | "never" | "dangerous_only"'],
          [<code>approval_callback</code>, 'Callable[[str, str], bool]', '(tool_name, tool_args) → bool'],
          [<code>approval_timeout</code>, 'float', 'Seconds to wait before applying default (0 = forever)'],
          [<code>ToolMetadata.requires_approval</code>, 'bool', 'Per-tool opt-in (used by "dangerous_only" mode)'],
        ]}
      />

      <CodeBlock
        code={`from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import BashTool, Calculator

def approve(tool_name: str, tool_args: str) -> bool:
    print(f"Approve {tool_name}({tool_args})? [y/N]")
    return input().strip().lower() == "y"

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

agent = Agent(AgentConfig(
    name="supervised",
    model=model,
    tools=[Calculator(), BashTool()],
    approval_mode="dangerous_only",   # only prompt for tools where requires_approval=True
    approval_callback=approve,
    approval_timeout=30.0,            # 30s to respond
))`}
        language="python"
        filename="approval.py"
      />

      <h3>Approval Modes</h3>
      <ApiTable
        headers={['Mode', 'Behaviour']}
        rows={[
          [<code>"always"</code>, 'Prompt on every tool call'],
          [<code>"first_time"</code>, 'Prompt once per (tool, args) pair — subsequent identical calls auto-approved'],
          [<code>"dangerous_only"</code>, 'Prompt only when ToolMetadata.requires_approval is True (e.g. BashTool, FileOperations write)'],
          [<code>"never"</code>, 'Never prompt (default)'],
        ]}
      />

      <h2>Clarification</h2>
      <CodeBlock
        code={`from effgen.core.clarification import ClarificationRequest, ClarificationDetector

def ask(question: str, options: list[str]) -> int:
    print(question)
    for i, o in enumerate(options):
        print(f"  {i+1}. {o}")
    return int(input("Choose: ")) - 1

agent = Agent(AgentConfig(
    name="clarifier",
    model=model,
    clarification_callback=ask,
))

# ClarificationDetector heuristics fire on:
#   - very short queries ("help")
#   - vague words ("stuff", "thing", "something")
#   - queries that match 2+ tools equally well`}
        language="python"
        filename="clarify.py"
      />

      <h2>Free-Text Input</h2>
      <CodeBlock
        code={`from effgen.core.human_loop import HumanInput

hi = HumanInput(prompt="What's the budget?", timeout=60.0)
answer = hi.request(callback=lambda p: input(p + " "))`}
        language="python"
        filename="human_input.py"
      />

      <h2>Feedback Collection</h2>
      <p>
        After a turn completes, collect thumbs / rating / comment from the user. Exports to
        JSONL for offline analysis or fine-tuning.
      </p>

      <CodeBlock
        code={`from effgen.core.feedback import FeedbackCollector

fc = FeedbackCollector(agent_name="chat")

# Use a stable response_id (e.g. a hash of agent_response.output, or response.metadata['run_id'])
fc.thumbs(response_id="r-1", thumbs_up=True,  query="What is X?")
fc.thumbs(response_id="r-2", thumbs_up=False, query="What is Y?")
fc.rate(response_id="r-3",   rating=4,        query="What is Z?")
fc.comment(response_id="r-3", text="minor typo")

# Iterate
for entry in fc.entries:
    print(entry.feedback_type, entry.value, entry.query)

# Export to JSONL for offline analysis / fine-tuning
fc.export_jsonl("./feedback.jsonl")`}
        language="python"
        filename="feedback.py"
      />

      <InfoBox type="success" title="Composable with everything">
        <p>
          Human-in-the-loop layers cleanly with <Link to="/guardrails">guardrails</Link>
          (which may also block or modify before the callback fires) and
          {' '}<Link to="/checkpointing">checkpointing</Link> (so a paused run can be resumed
          tomorrow).
        </p>
      </InfoBox>

      <h2>See Also</h2>
      <p>
        <Link to="/guardrails">Guardrails</Link> · <Link to="/agents">Agents</Link> ·
        {' '}<Link to="/workflows">Workflows</Link>
      </p>
    </DocPage>
  );
}
