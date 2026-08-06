import React, { useState, useRef } from 'react';
import { Book, ChevronRight, Brain, RefreshCw, Network, Zap, Cpu, Layers, Wrench, Radio } from 'lucide-react';
import PageNavbar from '../components/PageNavbar';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';
import './Guides.css';

interface Guide {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  content: React.ReactNode;
}

const guides: Guide[] = [
  {
    id: 'how-memory-works',
    title: 'How Memory Works',
    description: 'Understanding short-term and long-term memory systems',
    icon: <Brain size={20} />,
    content: (
      <>
        <MermaidDiagram
          title="Memory Flow"
          chart={`
flowchart TB
    U[User Input] --> A[Agent]
    A --> STM[Short-Term Memory]
    STM --> |"Context"| M[Model]
    M --> R[Response]
    R --> STM
    STM --> |"Important"| LTM[Long-Term Memory]
    LTM --> |"Recall"| STM

    style STM fill:#00c96e,color:#fff
    style LTM fill:#009950,color:#fff
`}
        />
        <h3>Short-Term Memory</h3>
        <p>
          Short-term memory maintains the current conversation context. It automatically
          manages token limits through truncation and summarization.
        </p>
        <CodeBlock
          code={`from effgen.memory.short_term import ShortTermMemory

memory = ShortTermMemory(
    max_tokens=4096,           # Context limit
    summarization_threshold=0.8  # Summarize at 80%
)

# Messages are automatically managed
memory.add_user_message("Hello!")
memory.add_assistant_message("Hi there!")

# When approaching limit, older messages are summarized`}
          language="python"
        />

        <h3>Long-Term Memory</h3>
        <p>
          Long-term memory persists important information across sessions using
          storage backends like JSON or SQLite.
        </p>
        <CodeBlock
          code={`from effgen.memory.long_term import LongTermMemory, MemoryType

memory.add_memory(
    content="User prefers Python",
    memory_type=MemoryType.OBSERVATION,
    importance=ImportanceLevel.HIGH,
    tags=["preference", "programming"]
)

# Later, retrieve relevant memories
results = memory.search("programming preferences")`}
          language="python"
        />
      </>
    ),
  },
  {
    id: 'react-loop',
    title: 'Understanding the ReAct Loop',
    description: 'How agents reason and act to solve problems',
    icon: <RefreshCw size={20} />,
    content: (
      <>
        <MermaidDiagram
          title="ReAct Loop"
          chart={`
flowchart LR
    T["Think"] --> D{Decision}
    D --> |"Use Tool"| A["Act"]
    D --> |"Answer Ready"| F["Final"]
    A --> O["Observe"]
    O --> T

    style T fill:#00c96e,color:#fff
    style A fill:#009950,color:#fff
    style O fill:#00b8d4,color:#fff
    style F fill:#22c55e,color:#fff
`}
        />
        <h3>The Reasoning Cycle</h3>
        <ol>
          <li><strong>Think:</strong> The agent reasons about the current state</li>
          <li><strong>Act:</strong> Execute a tool or provide final answer</li>
          <li><strong>Observe:</strong> Process tool results</li>
          <li><strong>Repeat:</strong> Continue until task is complete</li>
        </ol>
        <CodeBlock
          code={`# Example ReAct execution trace
result = agent.run("What is sqrt(144) + 10?")

for step in result.execution_trace:
    print(f"Iteration {step['iteration']}:")
    print(f"  Thought: {step['thought']}")
    print(f"  Action: {step['action']}")
    print(f"  Observation: {step['observation']}")

# Output:
# Iteration 1:
#   Thought: I need to calculate sqrt(144) first
#   Action: calculator(expression="sqrt(144)")
#   Observation: {"result": 12.0}
# Iteration 2:
#   Thought: Now I add 10 to 12
#   Action: calculator(expression="12 + 10")
#   Observation: {"result": 22.0}
# Iteration 3:
#   Thought: The answer is 22
#   Action: Final Answer: 22`}
          language="python"
        />
      </>
    ),
  },
  {
    id: 'orchestration-patterns',
    title: 'Multi-Agent Orchestration Patterns',
    description: 'Choosing the right pattern for your use case',
    icon: <Network size={20} />,
    content: (
      <>
        <h3>Sequential Pattern</h3>
        <MermaidDiagram
          title="Sequential Pattern"
          chart={`
flowchart LR
    I[Input] --> A1[Agent 1]
    A1 --> A2[Agent 2]
    A2 --> A3[Agent 3]
    A3 --> O[Output]
`}
        />
        <p><strong>Use for:</strong> Pipelines where each step depends on the previous</p>

        <h3>Parallel Pattern</h3>
        <MermaidDiagram
          title="Parallel Pattern"
          chart={`
flowchart TB
    I[Input] --> A1[Agent 1]
    I --> A2[Agent 2]
    I --> A3[Agent 3]
    A1 --> M[Merge]
    A2 --> M
    A3 --> M
    M --> O[Output]
`}
        />
        <p><strong>Use for:</strong> Independent sub-tasks that can run simultaneously</p>

        <h3>Hierarchical Pattern</h3>
        <MermaidDiagram
          title="Hierarchical Pattern"
          chart={`
flowchart TB
    I[Input] --> M[Manager]
    M --> W1[Worker 1]
    M --> W2[Worker 2]
    M --> W3[Worker 3]
    W1 --> M
    W2 --> M
    W3 --> M
    M --> O[Output]
`}
        />
        <p><strong>Use for:</strong> Complex tasks requiring coordination</p>

        <CodeBlock
          code={`from effgen.core.orchestrator import OrchestrationPattern

# Choose pattern based on task
if task.is_pipeline:
    pattern = OrchestrationPattern.SEQUENTIAL
elif task.has_independent_subtasks:
    pattern = OrchestrationPattern.PARALLEL
elif task.needs_coordination:
    pattern = OrchestrationPattern.HIERARCHICAL`}
          language="python"
        />
      </>
    ),
  },
  {
    id: 'tool-selection',
    title: 'Tool Selection Best Practices',
    description: 'How agents choose and use tools effectively',
    icon: <Zap size={20} />,
    content: (
      <>
        <MermaidDiagram
          title="Tool Selection Flow"
          chart={`
flowchart TB
    T[Task] --> A{Analyze}
    A --> |"Math needed"| C[Calculator]
    A --> |"Code needed"| P[PythonREPL]
    A --> |"Data needed"| S[WebSearch]
    A --> |"Simple question"| D[Direct Answer]

    C --> R[Result]
    P --> R
    S --> R
    D --> R
`}
        />
        <h3>Tips for Effective Tool Use</h3>
        <ul>
          <li><strong>Clear descriptions:</strong> Tools with clear descriptions are selected correctly</li>
          <li><strong>Minimal tools:</strong> Only provide necessary tools to avoid confusion</li>
          <li><strong>Parameter validation:</strong> Well-defined parameters help agents use tools correctly</li>
        </ul>
        <CodeBlock
          code={`from effgen.tools import BaseTool, ParameterSpec, ParameterType, ToolCategory, ToolMetadata

# Good: clear, specific metadata and an async _execute implementation.
class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__(metadata=ToolMetadata(
            name="weather",
            description="Get current weather for a city. Returns temperature, humidity, and conditions.",
            category=ToolCategory.EXTERNAL_API,
            parameters=[
                ParameterSpec(
                    name="city",
                    type=ParameterType.STRING,
                    description="City name",
                    required=True,
                )
            ],
        ))

    async def _execute(self, city: str) -> dict:
        return {"city": city, "temperature": 22, "condition": "Sunny"}

# Avoid: description is too vague for reliable tool selection.
class DataTool(BaseTool):
    def __init__(self):
        super().__init__(metadata=ToolMetadata(
            name="data",
            description="Get data",
            category=ToolCategory.DATA_PROCESSING,
        ))

    async def _execute(self, **kwargs) -> dict:
        return {"data": None}`}
          language="python"
        />
      </>
    ),
  },
  {
    id: 'optimization-slm',
    title: 'Optimizing for Small Language Models',
    description: 'Get the best results from 3B-7B parameter models',
    icon: <Cpu size={20} />,
    content: (
      <>
        <h3>Prompt Optimization</h3>
        <MermaidDiagram
          title="SLM Optimization Flow"
          chart={`
flowchart LR
    V[Verbose Prompt] --> O[Optimizer]
    O --> C[Concise]
    O --> S[Structured]
    O --> E[Examples]
    C --> SLM[SLM]
    S --> SLM
    E --> SLM
    SLM --> R[Better Results]
`}
        />
        <h3>Key Strategies</h3>
        <ol>
          <li><strong>Be concise:</strong> Remove unnecessary words and phrases</li>
          <li><strong>Use structure:</strong> Numbered lists and clear sections</li>
          <li><strong>Add examples:</strong> 2-3 relevant examples improve accuracy</li>
          <li><strong>Explicit format:</strong> Specify exact output format</li>
        </ol>
        <CodeBlock
          code={`# Before: Verbose prompt
prompt = """
I would like you to carefully analyze the following information
and provide a comprehensive summary considering all aspects...
"""

# After: Optimized for SLM
prompt = """
Analyze the data. Provide:
1. Key findings (3 bullets)
2. Main pattern
3. Recommendation

Format: Short sentences, bullet points."""`}
          language="python"
        />
      </>
    ),
  },
  {
    id: 'creating-presets',
    title: 'Creating Agent Presets',
    description: 'Build and customize agent presets for common use cases',
    icon: <Layers size={20} />,
    content: (
      <>
        <h3>Understanding Presets</h3>
        <p>
          Presets are pre-configured agent templates that bundle a system prompt with an optimized
          set of tools. effGen ships with 9 built-in presets (math, research, coding, general, rag, minimal, the <code>media</code> and <code>notify</code> presets added in v0.2.6, and the <code>multimodal</code> preset added in v0.2.8), and you can create your own.
        </p>

        <h3>Using Built-in Presets</h3>
        <CodeBlock
          code={`from effgen import load_model
from effgen.presets import create_agent, list_presets

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# List all available presets (returns name → description dict)
for name, description in list_presets().items():
    print(f"{name}: {description}")

# Create agents with presets
math_agent = create_agent("math", model)
research_agent = create_agent("research", model)
coding_agent = create_agent("coding", model)
general_agent = create_agent("general", model)
media_agent = create_agent("media", model)          # v0.2.6 — audio + vision
notify_agent = create_agent("notify", model)        # v0.2.6 — email + Slack + Discord
multimodal_agent = create_agent("multimodal", model)  # v0.2.8 — image / audio / video
minimal_agent = create_agent("minimal", model)`}
          language="python"
        />

        <h3>Creating Custom Presets</h3>
        <CodeBlock
          code={`from effgen.presets.registry import PresetConfig, PRESETS

# Define a custom preset
my_preset = PresetConfig(
    name="web_scraper",
    description="Web scraping and data extraction",
    tool_names=["web_search", "url_fetch", "json_tool"],
    system_prompt="""You are a web scraping specialist.
Extract structured data from web pages and format as JSON.""",
    temperature=0.1,
    max_iterations=10
)

# Register the preset
PRESETS["web_scraper"] = my_preset

# Now use it like any built-in preset
agent = create_agent("web_scraper", model)`}
          language="python"
        />
      </>
    ),
  },
  {
    id: 'building-plugins',
    title: 'Building Tool Plugins',
    description: 'Create reusable tool plugins for effGen',
    icon: <Wrench size={20} />,
    content: (
      <>
        <h3>Plugin Architecture</h3>
        <p>
          The plugin system allows you to package and distribute custom tools. Plugins are
          discovered automatically via Python entry points or directory scanning.
        </p>

        <h3>Creating a Plugin</h3>
        <CodeBlock
          code={`from effgen.tools.plugin import ToolPlugin
from effgen.tools import BaseTool, ParameterSpec, ParameterType, ToolCategory, ToolMetadata

class TranslationTool(BaseTool):
    def __init__(self):
        super().__init__(metadata=ToolMetadata(
            name="translate",
            description="Translate text between languages",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[
                ParameterSpec(
                    name="text",
                    type=ParameterType.STRING,
                    description="Text to translate",
                    required=True,
                ),
                ParameterSpec(
                    name="target_lang",
                    type=ParameterType.STRING,
                    description="Target language",
                    required=True,
                ),
            ],
        ))

    async def _execute(self, text: str, target_lang: str) -> dict:
        # Your translation logic here
        return {"translated": "...", "language": target_lang}

class MyPlugin(ToolPlugin):
    name = "translation-plugin"
    version = "1.0.0"
    description = "Translation tools for effGen"
    tools = [TranslationTool]`}
          language="python"
        />

        <h3>Distribution via Entry Points</h3>
        <CodeBlock
          code={`# In your pyproject.toml:
[project.entry-points."effgen.plugins"]
translation = "my_package.plugin:MyPlugin"

# Plugins are auto-discovered when installed
# pip install my-translation-plugin
# effGen will find and load it automatically`}
          language="toml"
        />

        <h3>Directory-Based Loading</h3>
        <CodeBlock
          code={`# Place plugin files in ~/.effgen/plugins/
# Or set EFFGEN_PLUGINS_DIR environment variable

from effgen.tools.plugin import PluginManager

manager = PluginManager()
manager.discover_user_dir()     # ~/.effgen/plugins/
manager.discover_env_dir()      # EFFGEN_PLUGINS_DIR
manager.discover_entry_points() # Python entry points`}
          language="python"
        />
      </>
    ),
  },
  {
    id: 'real-time-streaming',
    title: 'Real-Time Streaming',
    description: 'Implement token-by-token streaming and SSE endpoints',
    icon: <Radio size={20} />,
    content: (
      <>
        <h3>Token Streaming</h3>
        <p>
          effGen supports true token-by-token streaming via <code>generate_stream()</code>.
          This provides real-time output as the model generates tokens.
        </p>

        <CodeBlock
          code={`from effgen import load_model
from effgen.presets import create_agent

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = create_agent("general", model)

# Basic streaming
for token in agent.stream("Explain machine learning"):
    print(token, end="", flush=True)`}
          language="python"
        />

        <h3>Streaming Callbacks</h3>
        <p>
          Use callbacks to handle different types of streaming events:
        </p>
        <CodeBlock
          code={`# Four callback types:
# on_thought   - Agent's reasoning steps
# on_tool_call - When a tool is invoked
# on_observation - Tool execution results
# on_answer    - Final answer tokens

agent.stream(
    "What is the weather in Tokyo?",
    on_thought=lambda t: print(f"[Think] {t}"),
    on_tool_call=lambda name, args: print(f"[Tool] {name}({args})"),
    on_observation=lambda r: print(f"[Result] {r}"),
    on_answer=lambda a: print(f"[Answer] {a}", end="")
)`}
          language="python"
        />

        <h3>WebSocket API Endpoint</h3>
        <CodeBlock
          code={`# Start the API server
# effgen serve --port 8000

# The /ws endpoint streams responses over WebSocket
import websockets

async def consume_ws():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send("Explain quantum computing")
        async for chunk in ws:
            print(chunk, end="", flush=True)`}
          language="python"
        />
      </>
    ),
  },
];

export default function Guides() {
  const [selectedGuide, setSelectedGuide] = useState<Guide>(guides[0]);
  const contentRef = useRef<HTMLDivElement>(null);

  const handleGuideClick = (guide: Guide) => {
    setSelectedGuide(guide);
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
  };

  return (
    <div className="guides-page">
      <PageNavbar
        title="Guides"
        items={[]}
        rightContent={
          <span className="guides-count">{guides.length} guides available</span>
        }
      />

      <div className="guides-layout">
        <nav className="guides-nav">
          <div className="guides-nav-header">
            <Book size={18} />
            <span>Topics</span>
          </div>
          {guides.map(guide => (
            <button
              key={guide.id}
              className={`guide-nav-item ${selectedGuide.id === guide.id ? 'active' : ''}`}
              onClick={() => handleGuideClick(guide)}
            >
              <div className="guide-nav-icon">{guide.icon}</div>
              <div className="guide-nav-content">
                <span className="guide-nav-title">{guide.title}</span>
                <span className="guide-nav-desc">{guide.description}</span>
              </div>
              <ChevronRight size={16} className="guide-nav-arrow" />
            </button>
          ))}
        </nav>

        <article className="guide-content" ref={contentRef}>
          <div className="guide-content-header">
            <div className="guide-content-icon">{selectedGuide.icon}</div>
            <div>
              <h2>{selectedGuide.title}</h2>
              <p className="guide-description">{selectedGuide.description}</p>
            </div>
          </div>
          <div className="guide-content-body">
            {selectedGuide.content}
          </div>
        </article>
      </div>
    </div>
  );
}
