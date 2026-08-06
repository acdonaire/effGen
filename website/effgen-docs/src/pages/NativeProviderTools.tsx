import React from 'react';
import { Wrench } from 'lucide-react';
import DocPage, { ApiTable, InfoBox } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function NativeProviderTools() {
  return (
    <DocPage
      title="Native Provider Tools"
      subtitle="OpenAI v0.2.1 server-side tools, Gemini v0.2.2 Agent-native tools, and Anthropic v0.2.2 experimental adapter specs."
      icon={<Wrench size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Native Provider Tools' },
      ]}
    >
      <p>
        Provider-native tools run inside the model provider's infrastructure. They are
        different from effGen's 31 local built-in tools. OpenAI and Gemini native
        tool wrappers can be passed through <code>AgentConfig.tools</code> and raise
        <code> ToolIncompatibleError</code> at agent initialization if paired with the
        wrong provider. Anthropic computer-use wrappers are experimental tool specs
        passed directly to <code>AnthropicAdapter.generate_with_tools()</code>.
      </p>

      <h2>Tool Matrix</h2>
      <ApiTable
        headers={['Provider', 'Class', 'Provider type', 'Purpose']}
        rows={[
          ['OpenAI', <code>OpenAIWebSearchTool</code>, <code>web_search_preview</code>, 'Live web search with citations through the Responses API'],
          ['OpenAI', <code>OpenAICodeInterpreterTool</code>, <code>code_interpreter</code>, 'Sandboxed Python runtime on OpenAI infrastructure'],
          ['OpenAI', <code>OpenAIFileSearchTool</code>, <code>file_search</code>, 'Vector search over OpenAI vector stores'],
          ['Gemini (v0.2.2+)', <code>GoogleSearchTool</code>, <code>google_search</code>, 'Google Search grounding with attribution chunks'],
          ['Gemini (v0.2.2+)', <code>GeminiUrlContextTool</code>, <code>url_context</code>, 'Server-side URL fetching'],
          ['Gemini (v0.2.2+)', <code>GeminiCodeExecutionTool</code>, <code>code_execution</code>, 'Server-side Python execution'],
          ['Anthropic (v0.2.2+)', <code>AnthropicBashTool</code>, <code>bash_20250124</code>, 'Experimental server-side bash'],
          ['Anthropic (v0.2.2+)', <code>AnthropicTextEditorTool</code>, <code>text_editor_20250728</code>, 'Experimental server-side file view/edit'],
          ['Anthropic (v0.2.2+)', <code>AnthropicComputerTool</code>, <code>computer_20251124</code>, 'Experimental computer-use actions'],
        ]}
      />

      <h2>OpenAI Native Tools</h2>
      <CodeBlock
        code={`from effgen.core.agent import Agent, AgentConfig
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

model = OpenAIAdapter(model_name="gpt-5.4-nano")
model.load()

agent = Agent(AgentConfig(
    name="openai-search",
    model=model,
    tools=[OpenAIWebSearchTool(search_context_size="medium")],
    tool_calling_mode="native",
))
result = agent.run("Search for recent model provider news and cite sources.")`}
        language="python"
        filename="openai_native.py"
      />
      <InfoBox type="warning" title="OpenAI billing">
        <p>
          <code>OpenAIWebSearchTool</code> can add a per-call search surcharge on top
          of token costs; v0.2.1 tracked the then-current web search preview note
          at roughly $30 per 1,000 calls for gpt-4o-class models. Monitor usage
          when enabling it in production agents.
        </p>
      </InfoBox>
      <ApiTable
        headers={['Metadata field', 'Meaning']}
        rows={[
          [<code>metadata["response_id"]</code>, 'Responses API response ID returned by OpenAI'],
          [<code>metadata["tool_calls"]</code>, 'Per-tool records for web_search_call, code_interpreter_call, file_search_call, and function_call items'],
          [<code>metadata["native_tool_results"]</code>, 'Same native OpenAI tool records exposed for the Agent loop'],
          [<code>metadata["cached_input_tokens"]</code>, 'Automatic prompt-cache hit tokens when the model reports them'],
        ]}
      />

      <h2>Gemini Native Tools (v0.2.2+)</h2>
      <p>
        v0.2.2 uses the modern <code>google-genai</code> package and
        <code> google.genai</code> namespace. Gemini supports first-party tools,
        Files API inputs, and multiple function calls in one model turn.
      </p>
      <CodeBlock
        code={`from effgen.core.agent import Agent, AgentConfig
from effgen.models.gemini_adapter import GeminiAdapter
from effgen.tools.builtin.gemini_native import (
    GoogleSearchTool,
    GeminiUrlContextTool,
    GeminiCodeExecutionTool,
)

model = GeminiAdapter(model_name="gemini-2.5-flash")
model.load()

agent = Agent(AgentConfig(
    name="gemini-grounded",
    model=model,
    tools=[GoogleSearchTool(), GeminiUrlContextTool(), GeminiCodeExecutionTool()],
))
result = agent.run("Use grounded search and code execution to answer the question.")`}
        language="python"
        filename="gemini_native.py"
      />
      <CodeBlock
        code={`from effgen.models.base import GenerationConfig
from effgen.models.gemini_adapter import GeminiAdapter
from effgen.models.gemini_files import FileRef, upload_file
from effgen.tools.builtin.gemini_native import GoogleSearchTool, GeminiCodeExecutionTool

doc: FileRef = upload_file("requirements.pdf")

with GeminiAdapter(model_name="gemini-2.5-pro") as model:
    response = model.generate(
        "Read the file, search for current context, and return any tool calls.",
        config=GenerationConfig(thinking_budget=4096, include_thoughts=True, grounding=True),
        files=[doc],
        tools=[
            GoogleSearchTool().to_gemini_tool(),
            GeminiCodeExecutionTool().to_gemini_tool(),
        ],
    )

print(response.metadata["thinking"])
print(response.metadata["grounding_chunks"])
print(response.metadata["tool_calls"])  # Parallel function calls are preserved here.`}
        language="python"
        filename="gemini_files_parallel.py"
      />
      <InfoBox type="info" title="Gemini Files API">
        <p>
          <code>upload_file(path)</code> returns a <code>FileRef</code>. Pass one or more
          refs to <code>generate(..., files=[...])</code>. effGen checks the 2 GiB
          Files API upload limit before sending the request.
        </p>
      </InfoBox>

      <h2>Anthropic Native Tools (v0.2.2+)</h2>
      <InfoBox type="warning" title="Experimental">
        <p>
          Anthropic native computer-use tools require the provider beta header and are
          not registered in default presets. Pass their Anthropic-format specs directly
          to <code>generate_with_tools()</code> and validate with your own key before deploying.
        </p>
      </InfoBox>
      <CodeBlock
        code={`from effgen.models import AnthropicAdapter
from effgen.tools.builtin.anthropic_native import AnthropicBashTool

tool = AnthropicBashTool()
spec = tool.to_anthropic_tool_spec()

with AnthropicAdapter(model_name="claude-sonnet-4-6") as adapter:
    result = adapter.generate_with_tools(
        "List files in /tmp.",
        tools=[spec],
        extra_headers={"anthropic-beta": "computer-use-2025-11-24"},
    )`}
        language="python"
        filename="anthropic_native.py"
      />
      <ApiTable
        headers={['API', 'What it preserves']}
        rows={[
          [<code>generate_stream_full()</code>, 'Typed StreamChunk objects for text, thinking, redacted_thinking, and tool_use deltas'],
          [<code>mark_cached(block)</code>, 'Anthropic cache_control markers on system, message, or tool blocks'],
          [<code>metadata["cached_input_tokens"]</code>, 'Prompt-cache hit tokens'],
          [<code>metadata["cache_creation_tokens"]</code>, 'Tokens written into a new cache entry'],
          [<code>metadata["raw_content_blocks"]</code>, 'Claude content blocks needed to replay redacted_thinking in multi-turn flows'],
          [<code>build_assistant_message(result)</code>, 'Assistant message reconstruction that keeps raw_content_blocks intact'],
        ]}
      />
      <CodeBlock
        code={`from effgen.models import AnthropicAdapter, mark_cached
from effgen.models.base import GenerationConfig

long_context = "..."  # Large reusable policy, spec, or repository summary.

with AnthropicAdapter(model_name="claude-sonnet-4-6") as adapter:
    result = adapter.generate(
        "Think through the migration plan.",
        system_prompt=[mark_cached({"type": "text", "text": long_context}, ttl="1h")],
        config=GenerationConfig(thinking={"type": "enabled", "budget_tokens": 4096}),
    )

    print(result.metadata["cached_input_tokens"])
    print(result.metadata["cache_creation_tokens"])

    history = [
        {"role": "user", "content": "Think through the migration plan."},
        adapter.build_assistant_message(result),
        {"role": "user", "content": "Now list rollout risks."},
    ]
    follow_up = adapter.generate_with_history(history)

    for chunk in adapter.generate_stream_full("Stream the implementation risks."):
        if chunk.type == "thinking":
            print("[thinking]", chunk.text)
        elif chunk.type == "redacted_thinking":
            print("[redacted]", chunk.data)
        elif chunk.type == "tool_use":
            print("[tool]", chunk.data)
        elif chunk.type == "text":
            print(chunk.text, end="")`}
        language="python"
        filename="anthropic_stream_cache.py"
      />

      <h2>Compatibility Guard</h2>
      <CodeBlock
        code={`from effgen.core.agent import Agent, AgentConfig
from effgen.models.errors import ToolIncompatibleError
from effgen.models.gemini_adapter import GeminiAdapter
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

model = GeminiAdapter(model_name="gemini-2.5-flash")

try:
    Agent(AgentConfig(name="bad", model=model, tools=[OpenAIWebSearchTool()]))
except ToolIncompatibleError as exc:
    print(exc)`}
        language="python"
        filename="tool_guard.py"
      />
    </DocPage>
  );
}
