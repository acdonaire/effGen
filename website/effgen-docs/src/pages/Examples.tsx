import React, { useState, useRef, useEffect } from 'react';
import { BookOpen, Bot, Code, Database, Brain, GitBranch, Zap, FileText, Settings, Cpu, Play, Terminal, Layers, Network, Shield, RefreshCw, Workflow, Grid, List } from 'lucide-react';
import PageNavbar from '../components/PageNavbar';
import CodeBlock from '../components/CodeBlock';
import './Examples.css';

interface Example {
  id: string;
  title: string;
  description: string;
  category: string;
  icon: React.ReactNode;
  code: string;
  filename: string;
}

const examples: Example[] = [
  // Getting Started
  {
    id: 'model-loading',
    title: 'Model Loading',
    description: 'Load models with different engines and quantization',
    category: 'Getting Started',
    icon: <Cpu size={20} />,
    filename: 'test_model_loading.py',
    code: `"""Load models with different configurations."""
from effgen import load_model

# Load with Transformers engine (4-bit quantization)
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    engine="transformers",
    quantization="4bit"
)

# Load with vLLM engine (faster inference)
model_vllm = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="vllm",
    tensor_parallel_size=1
)

# Test generation
response = model.generate("What is Python?")
print(f"Response: {response}")`
  },
  {
    id: 'agent-creation',
    title: 'Agent Creation',
    description: 'Create and configure agents with custom settings',
    category: 'Getting Started',
    icon: <Bot size={20} />,
    filename: 'test_agent_creation.py',
    code: `"""Create agents with different configurations."""
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL, DateTimeTool

# Load model
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    engine="transformers",
    quantization="4bit"
)

# Create tools directly
tools = [Calculator(), PythonREPL(), DateTimeTool()]

# Create agent with default config
config = AgentConfig(
    name="test_agent",
    model=model,
    tools=tools
)
agent = Agent(config=config)

# Create agent with custom config
custom_config = AgentConfig(
    name="custom_agent",
    model=model,
    tools=tools,
    max_iterations=5,
    temperature=0.5,
    system_prompt="You are a helpful assistant."
)
custom_agent = Agent(config=custom_config)

# Run a simple task
result = agent.run("What is 2 + 2?")
print(f"Answer: {result.output}")`
  },
  {
    id: 'simple-task',
    title: 'Simple Agent Task',
    description: 'Run basic tasks with agents',
    category: 'Getting Started',
    icon: <Play size={20} />,
    filename: 'test_agent_simple_task.py',
    code: `"""Run simple tasks with an agent."""
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Setup
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

config = AgentConfig(
    name="math_agent",
    model=model,
    tools=[Calculator()],
    system_prompt="You are a helpful math assistant.",
    temperature=0.1,
    max_iterations=5
)

agent = Agent(config=config)

# Run task
result = agent.run("Calculate 25 * 17 + sqrt(144)")
print(f"Answer: {result.output}")
print(f"Success: {result.success}")
print(f"Tool calls: {result.tool_calls}")`
  },

  // Tools
  {
    id: 'calculator-tool',
    title: 'Calculator Tool',
    description: 'Use the built-in calculator for math operations',
    category: 'Tools',
    icon: <Code size={20} />,
    filename: 'test_calculator.py',
    code: `"""Use the Calculator tool for math operations."""
import asyncio
from effgen.tools.builtin import Calculator

async def main():
    calculator = Calculator()

    # Basic operations
    result = await calculator.execute(expression="2 + 3 * 4")
    print(f"2 + 3 * 4 = {result.output}")

    # Complex expressions
    result = await calculator.execute(expression="sqrt(16) + pow(2, 3)")
    print(f"sqrt(16) + pow(2, 3) = {result.output}")

    # Mathematical functions
    result = await calculator.execute(expression="sin(pi/2) + cos(0)")
    print(f"sin(pi/2) + cos(0) = {result.output}")

asyncio.run(main())`
  },
  {
    id: 'python-repl',
    title: 'Python REPL Tool',
    description: 'Execute Python code with state persistence',
    category: 'Tools',
    icon: <Terminal size={20} />,
    filename: 'test_python_repl.py',
    code: `"""Use Python REPL with state persistence."""
import asyncio
from effgen.tools.builtin import PythonREPL

async def main():
    repl = PythonREPL()

    # Execute code and store state
    await repl.execute(code='x = 42')

    # State persists across executions
    result = await repl.execute(code='print(x * 2)')
    print(f"x * 2 = {result.output}")  # 84

    # Define functions
    code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
'''
    result = await repl.execute(code=code)
    print(f"factorial(5) = {result.output}")  # 120

    # Use defined functions later
    result = await repl.execute(code='print(factorial(6))')
    print(f"factorial(6) = {result.output}")  # 720

asyncio.run(main())`
  },
  {
    id: 'custom-tool',
    title: 'Custom Tool Creation',
    description: 'Create your own tools for agents',
    category: 'Tools',
    icon: <Zap size={20} />,
    filename: 'test_custom_tool.py',
    code: `"""Create custom tools for agents."""
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools import BaseTool, ParameterSpec, ParameterType, ToolCategory, ToolMetadata

class WeatherTool(BaseTool):
    """Custom weather lookup tool."""

    def __init__(self, api_key: str):
        super().__init__(
            metadata=ToolMetadata(
                name="weather",
                description="Get current weather for a city",
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="city",
                        type=ParameterType.STRING,
                        description="City name",
                        required=True,
                    )
                ]
            )
        )
        self.api_key = api_key

    async def _execute(self, city: str) -> dict:
        # Implementation here
        return {
            "city": city,
            "temperature": 22,
            "condition": "Sunny",
            "humidity": 45
        }

# Use custom tool
model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")
weather = WeatherTool(api_key="your-api-key")

config = AgentConfig(
    name="weather_agent",
    model=model,
    tools=[weather]
)

agent = Agent(config=config)
result = agent.run("What's the weather in Tokyo?")`
  },
  {
    id: 'file-operations',
    title: 'File Operations Tool',
    description: 'Read and write files safely',
    category: 'Tools',
    icon: <FileText size={20} />,
    filename: 'test_file_operations.py',
    code: `"""Use file operation tools safely."""
import asyncio
from effgen.tools.builtin import FileOperations

async def main():
    files = FileOperations()

    # Write to a file
    result = await files.execute(
        operation="write",
        path="./output.txt",
        content="Hello, World!",
    )
    print(f"Write success: {result.success}")

    # Read a file
    result = await files.execute(
        operation="read",
        path="./output.txt",
    )
    print(f"File contents: {result.output}")

    # List files
    result = await files.execute(
        operation="list",
        path=".",
    )
    print(f"Files: {result.output}")

asyncio.run(main())`
  },

  // Memory
  {
    id: 'short-term-memory',
    title: 'Short-Term Memory',
    description: 'Manage conversation context efficiently',
    category: 'Memory',
    icon: <Brain size={20} />,
    filename: 'test_short_term_memory.py',
    code: `"""Use short-term memory for conversations."""
from effgen.memory import ShortTermMemory, MessageRole

# Create short-term memory
memory = ShortTermMemory(
    max_tokens=4096,
    max_messages=50
)

# Add messages
memory.add_message(MessageRole.USER, "My name is Alice")
memory.add_message(MessageRole.ASSISTANT, "Hello Alice! Nice to meet you.")

# Get formatted context for model input
messages = memory.get_conversation_context()
print(f"Messages: {len(messages)}")

# Check token usage
stats = memory.get_statistics()
print(f"Token count: {stats['current_tokens']}")
print(f"Message count: {stats['current_messages']}")

# Search recent messages
matches = memory.search_messages("Alice")
print(f"Matches: {len(matches)}")`
  },
  {
    id: 'long-term-memory',
    title: 'Long-Term Memory',
    description: 'Persistent storage with search capabilities',
    category: 'Memory',
    icon: <Database size={20} />,
    filename: 'test_long_term_memory.py',
    code: `"""Use long-term memory with persistence."""
from effgen.memory import (
    LongTermMemory,
    MemoryType,
    ImportanceLevel,
    SQLiteStorageBackend
)

# Create with SQLite backend
backend = SQLiteStorageBackend(db_path="./memory.db")
memory = LongTermMemory(backend=backend)

# Start a session
session = memory.start_session(name="user_session")

# Add memories with metadata
memory.add_memory(
    content="User prefers Python over JavaScript",
    memory_type=MemoryType.OBSERVATION,
    importance=ImportanceLevel.HIGH,
    tags=["preference", "programming"]
)

memory.add_memory(
    content="Completed ML project successfully",
    memory_type=MemoryType.TASK,
    importance=ImportanceLevel.MEDIUM,
    tags=["task", "ml"]
)

# Search memories
results = memory.search(
    query="programming",
    min_importance=ImportanceLevel.MEDIUM
)
print(f"Found {len(results)} relevant memories")

# Get statistics
stats = memory.get_statistics()
print(f"Total memories: {stats['total_memories']}")`
  },
  {
    id: 'memory-integration',
    title: 'Memory Integration',
    description: 'Combine memory types with agents',
    category: 'Memory',
    icon: <Layers size={20} />,
    filename: 'test_memory_integration.py',
    code: `"""Integrate memory systems with agents."""
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig

model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

# Create agent with memory
config = AgentConfig(
    name="memory_agent",
    model=model,
    enable_memory=True,
    memory_config={
        "short_term_max_tokens": 4096,
        "short_term_max_messages": 50,
        "long_term_backend": "sqlite",
        "long_term_persist_path": "./memory",
    },
    system_prompt="You are a helpful assistant with memory."
)

agent = Agent(config=config)

# Multi-turn conversation - agent remembers context
agent.run("My name is Alice and I'm a software engineer")
agent.run("I'm working on a machine learning project")
agent.run("I prefer Python and PyTorch")

# Agent uses context for personalized response
result = agent.run("What programming tools should I learn next?")
print(result.output)  # Personalized recommendation`
  },

  // Agents
  {
    id: 'react-loop',
    title: 'ReAct Loop',
    description: 'Reason-Act execution pattern',
    category: 'Agents',
    icon: <RefreshCw size={20} />,
    filename: 'test_react_loop.py',
    code: `"""Test the ReAct (Reason-Act) loop execution."""
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Load model and tools
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
calculator = Calculator()

# Create agent with ReAct pattern
config = AgentConfig(
    name="react_agent",
    model=model,
    tools=[calculator],
    max_iterations=5,
    temperature=0.1
)
agent = Agent(config=config)

# Run task requiring multi-step reasoning
task = "Calculate the square root of 144 and then add 8 to it."
response = agent.run(task)

# Analyze execution
print(f"Total iterations: {response.iterations}")
print(f"Answer: {response.output}")
print(f"Tool calls: {response.tool_calls}")`
  },
  {
    id: 'agent-streaming',
    title: 'Streaming Responses',
    description: 'Stream agent responses in real-time',
    category: 'Agents',
    icon: <Play size={20} />,
    filename: 'test_agent_streaming.py',
    code: `"""Stream agent responses in real-time."""
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

config = AgentConfig(
    name="streaming_agent",
    model=model,
    tools=[],
    enable_streaming=True
)

agent = Agent(config=config)

# Stream response
print("Streaming response:")
for chunk in agent.stream("Explain quantum computing in 3 sentences."):
    print(chunk, end="", flush=True)
print()

# Stream with execution-event callbacks
def on_thought(text: str):
    print(f"Thought: {text}")

def on_tool_call(tool_name: str, tool_input):
    print(f"Tool: {tool_name}({tool_input})")

def on_answer(answer: str):
    print(f"Answer: {answer}")

for chunk in agent.stream(
    "What is machine learning?",
    on_thought=on_thought,
    on_tool_call=on_tool_call,
    on_answer=on_answer,
):
    print(chunk, end="", flush=True)`
  },
  {
    id: 'multi-tool-task',
    title: 'Multi-Tool Tasks',
    description: 'Use multiple tools to solve complex tasks',
    category: 'Agents',
    icon: <Workflow size={20} />,
    filename: 'test_multi_tool_task.py',
    code: `"""Solve complex tasks using multiple tools."""
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL

model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

# Create multiple tools
tools = [Calculator(), PythonREPL()]

config = AgentConfig(
    name="multi_tool_agent",
    model=model,
    tools=tools,
    system_prompt="""You are a problem-solving assistant.
Use the appropriate tools to solve tasks efficiently.""",
    max_iterations=10
)

agent = Agent(config=config)

# Complex task requiring multiple tools
task = """
Analyze this data and provide insights:
Numbers: [12, 45, 23, 67, 89, 34, 56]

1. Calculate the mean
2. Find the max and min
3. Calculate the standard deviation
"""

result = agent.run(task)
print(f"Analysis: {result.output}")
print(f"Tools used: {result.tool_calls}")`
  },

  // Prompts
  {
    id: 'prompt-templates',
    title: 'Prompt Templates',
    description: 'Create reusable prompt templates',
    category: 'Prompts',
    icon: <FileText size={20} />,
    filename: 'test_template_manager.py',
    code: `"""Create and manage prompt templates."""
from effgen.prompts import TemplateManager
from effgen.prompts.template_manager import PromptTemplate

# Create template manager
manager = TemplateManager()

# Register a PromptTemplate instance
manager.add_template(PromptTemplate(
    name="analyze",
    template="""Analyze the following {{ content_type }}:

{{ content }}

Focus on:
{% for point in focus_points %}
- {{ point }}
{% endfor %}

Provide a {{ output_format }} response.""",
))

# Render template by name with variables
prompt = manager.render_template(
    "analyze",
    variables={
        "content_type": "code",
        "content": "def hello(): print('Hello')",
        "focus_points": ["readability", "efficiency", "best practices"],
        "output_format": "detailed",
    },
)
print(prompt)`
  },
  {
    id: 'prompt-chains',
    title: 'Prompt Chains',
    description: 'Create multi-step prompt workflows',
    category: 'Prompts',
    icon: <GitBranch size={20} />,
    filename: 'test_chain_manager.py',
    code: `"""Create and execute prompt chains."""
import asyncio
from effgen import Agent, AgentConfig, load_model
from effgen.prompts import ChainManager
from effgen.prompts.chain_manager import (
    PromptChain,
    ChainStep,
    ChainType
)

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = Agent(config=AgentConfig(name="chain_agent", model=model))

async def main():
    manager = ChainManager(executor=agent.run)

    chain = PromptChain(
        name="research_chain",
        chain_type=ChainType.SEQUENTIAL,
        steps=[
            ChainStep(
                name="research",
                type="prompt",
                prompt="Research the topic: {topic}. List 5 key facts.",
                output_var="facts"
            ),
            ChainStep(
                name="analyze",
                type="prompt",
                prompt="Analyze these facts: {facts}. Identify patterns.",
                output_var="analysis"
            ),
            ChainStep(
                name="summarize",
                type="prompt",
                prompt="Create executive summary from: {analysis}",
                output_var="summary"
            ),
        ]
    )

    result = await manager.execute_chain(
        chain,
        initial_state={"topic": "renewable energy trends"}
    )
    print(f"Summary: {result.get_variable('summary')}")

asyncio.run(main())`
  },

  // Multi-Agent
  {
    id: 'orchestrator',
    title: 'Multi-Agent Orchestrator',
    description: 'Coordinate multiple agents with different patterns',
    category: 'Multi-Agent',
    icon: <Network size={20} />,
    filename: 'test_orchestrator.py',
    code: `"""Orchestrate multiple agents."""
from effgen.core.orchestrator import (
    MultiAgentOrchestrator,
    OrchestrationPattern
)
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig

model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

# Create specialized agents
researcher = Agent(config=AgentConfig(
    name="researcher",
    model=model,
    system_prompt="You research and gather information."
))

analyst = Agent(config=AgentConfig(
    name="analyst",
    model=model,
    system_prompt="You analyze data and find patterns."
))

writer = Agent(config=AgentConfig(
    name="writer",
    model=model,
    system_prompt="You write clear, engaging content."
))

# Create orchestrator
orchestrator = MultiAgentOrchestrator()
orchestrator.register_agent(researcher)
orchestrator.register_agent(analyst)
orchestrator.register_agent(writer)

# Create team with sequential pattern
team = orchestrator.create_team(
    name="research_team",
    agents=[researcher, analyst, writer],
    pattern=OrchestrationPattern.SEQUENTIAL
)

# Execute task
result = orchestrator.assign_task(
    task="Research AI trends, analyze findings, write a summary",
    team=team
)
print(f"Output: {result.output}")`
  },
  {
    id: 'sub-agent-routing',
    title: 'Sub-Agent Routing',
    description: 'Let Agent route complex tasks through built-in sub-agents',
    category: 'Multi-Agent',
    icon: <GitBranch size={20} />,
    filename: 'test_sub_agent_routing.py',
    code: `"""Route complex tasks through Agent sub-agent mode."""
from effgen import Agent, AgentConfig, load_model
from effgen.core.agent import AgentMode

model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

agent = Agent(config=AgentConfig(
    name="planner",
    model=model,
    enable_sub_agents=True,
    max_sub_agent_depth=2,
    sub_agent_config={"max_sub_agents": 4},
))

complex_task = """
Create a comprehensive report on climate change:
1. Research current statistics
2. Analyze trends over the past decade
3. Identify key factors
4. Propose solutions
5. Write executive summary
"""

result = agent.run(complex_task, mode=AgentMode.AUTO)
print(result.output)

if result.routing_decision:
    print(f"Strategy: {result.routing_decision.strategy.value}")`
  },

  // Configuration
  {
    id: 'config-loader',
    title: 'Configuration Loading',
    description: 'Load and manage configurations',
    category: 'Configuration',
    icon: <Settings size={20} />,
    filename: 'test_config_loader.py',
    code: `"""Load and manage configurations."""
from effgen.config import ConfigLoader, Config

# Initialize config loader
loader = ConfigLoader()

# Load configuration from YAML file
config = loader.load_config("config/config.yaml")

# Access configuration values
print(f"Config loaded: {config}")

# Get specific values
model_name = loader.get("model.default_model")
temperature = loader.get("agent.temperature")

# Set configuration values
loader.set("agent.temperature", 0.5)

# Save configuration
loader.save_config("config/config.yaml")

# Or create programmatically with the Config dataclass (single data dict)
config = Config(data={
    "model": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "engine": "transformers",
        "quantization": "4bit",
    },
    "agent": {
        "max_iterations": 10,
        "temperature": 0.3,
    },
    "memory": {
        "enable_long_term": True,
        "storage_path": "./memory",
    },
})

# Access config values; dot-path lookup is on ConfigLoader
print(f"Model: {loader.get('model.name')}")
print(f"Max iterations: {loader.get('agent.max_iterations')}")`
  },

  // Execution
  {
    id: 'sandbox-execution',
    title: 'Sandbox Execution',
    description: 'Execute code safely in a sandbox',
    category: 'Execution',
    icon: <Shield size={20} />,
    filename: 'test_sandbox.py',
    code: `"""Execute code safely in a sandbox."""
from effgen.execution import (
    CodeExecutor,
    SandboxConfig,
    ExecutionStatus
)

# Create executor with custom config
config = SandboxConfig(
    timeout=10,
    memory_limit="256M",
    allow_network=False,
    allow_file_ops=False
)
executor = CodeExecutor(sandbox_type="local", config=config)

# Execute Python code
code = '''
x = 10
y = 20
result = x + y
print(f"Result: {result}")
'''
result = executor.execute(code, language="python")
print(f"Status: {result.status.value}")
print(f"Output: {result.output}")

# Handle errors gracefully
result = executor.execute("x = 1 / 0", language="python")
print(f"Error handled: {result.error}")

# Execute with retry
result = executor.execute_with_retry(
    "print('Hello')",
    language="python",
    max_retries=3
)`
  },
];

const categories = ['All', 'Getting Started', 'Tools', 'Memory', 'Agents', 'Prompts', 'Multi-Agent', 'Configuration', 'Execution'];

export default function Examples() {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [selectedExample, setSelectedExample] = useState<Example | null>(examples[0]);
  const [viewMode, setViewMode] = useState<'split' | 'list'>('split');
  const contentRef = useRef<HTMLDivElement>(null);

  const filteredExamples = selectedCategory === 'All'
    ? examples
    : examples.filter(e => e.category === selectedCategory);

  const handleExampleClick = (example: Example) => {
    setSelectedExample(example);
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
    if (window.innerWidth <= 968) {
      const contentElement = document.querySelector('.examples-content');
      if (contentElement) {
        contentElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  };

  useEffect(() => {
    if (filteredExamples.length > 0 && !filteredExamples.find(e => e.id === selectedExample?.id)) {
      setSelectedExample(filteredExamples[0]);
    }
  }, [selectedCategory]);

  const navItems = categories.map(cat => ({
    label: cat,
    path: '#',
  }));

  return (
    <div className="examples-page">
      <PageNavbar
        title="Examples"
        items={[]}
        rightContent={
          <div className="examples-view-toggle">
            <button
              className={`view-btn ${viewMode === 'split' ? 'active' : ''}`}
              onClick={() => setViewMode('split')}
              title="Split View"
            >
              <Grid size={18} />
            </button>
            <button
              className={`view-btn ${viewMode === 'list' ? 'active' : ''}`}
              onClick={() => setViewMode('list')}
              title="List View"
            >
              <List size={18} />
            </button>
          </div>
        }
      />

      <div className="examples-category-bar">
        {categories.map(cat => (
          <button
            key={cat}
            className={`category-btn ${selectedCategory === cat ? 'active' : ''}`}
            onClick={() => setSelectedCategory(cat)}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className={`examples-layout ${viewMode}`}>
        <div className="examples-sidebar">
          <div className="examples-list">
            {filteredExamples.map(example => (
              <div
                key={example.id}
                className={`example-card ${selectedExample?.id === example.id ? 'active' : ''}`}
                onClick={() => handleExampleClick(example)}
              >
                <div className="example-icon">{example.icon}</div>
                <div className="example-info">
                  <h3>{example.title}</h3>
                  <p>{example.description}</p>
                  <span className="example-category">{example.category}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="examples-content" ref={contentRef}>
          {selectedExample ? (
            <>
              <div className="example-header">
                <div className="example-header-icon">{selectedExample.icon}</div>
                <div className="example-header-info">
                  <h2>{selectedExample.title}</h2>
                  <p>{selectedExample.description}</p>
                </div>
              </div>
              <CodeBlock
                code={selectedExample.code}
                language="python"
                filename={selectedExample.filename}
              />
            </>
          ) : (
            <div className="example-placeholder">
              <BookOpen size={64} />
              <h3>Select an Example</h3>
              <p>Choose an example from the list to view the code.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
