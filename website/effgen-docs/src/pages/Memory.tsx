import React from 'react';
import { Link } from 'react-router-dom';
import { Database } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';

export default function Memory() {
  const memoryDiagram = `
flowchart TB
    subgraph STM["Short-Term Memory"]
        C[Conversation History]
        S[Auto Summarization]
        T[Token Management]
    end

    subgraph LTM["Long-Term Memory"]
        F[Facts & Knowledge]
        O[Observations]
        R[Reflections]
        B[Storage Backend]
    end

    subgraph VMS["Vector Memory Store"]
        V[Semantic Search]
        E[Embeddings]
        VB[Vector Backend]
    end

    A[Agent] --> STM
    A --> LTM
    A --> VMS
    STM --> S
    S -->|"Overflow"| LTM
    B --> JSON[(JSON)]
    B --> SQLite[(SQLite)]
    VB --> FAISS[(FAISS)]
    VB --> Chroma[(ChromaDB)]
`;

  return (
    <DocPage
      title="Memory Systems"
      subtitle="effGen provides dual memory architecture: short-term for conversation context and long-term for persistent knowledge."
      icon={<Database size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Core Concepts', path: '/agents' },
        { label: 'Memory' },
      ]}
    >
      <h2>Overview</h2>
      <p>
        Memory systems allow agents to maintain context across conversations and recall important
        information. effGen implements a dual memory architecture:
      </p>

      <MermaidDiagram chart={memoryDiagram} title="Memory Architecture" />

      <FeatureList
        features={[
          { icon: '⚡', title: 'Short-Term Memory', description: 'Conversation history with automatic overflow handling and summarization. Always initialized, stores conversation turns.' },
          { icon: '💾', title: 'Long-Term Memory', description: 'Persistent storage with importance-based retrieval. JSON and SQLite backends.' },
          { icon: '🔍', title: 'Vector Memory Store', description: 'Semantic search via FAISS or ChromaDB backends with configurable embedding providers.' },
        ]}
      />

      <h2>Short-Term Memory</h2>
      <p>
        Short-term memory manages conversation context within a session. It automatically handles
        context window limits through truncation and summarization.
      </p>

      <CodeBlock
        code={`from effgen.memory.short_term import ShortTermMemory, MessageRole

# Create memory with limits
memory = ShortTermMemory(
    max_tokens=4096,              # Maximum context tokens
    max_messages=100,             # Maximum messages to keep
    summarization_threshold=0.8,  # Summarize at 80% capacity
    keep_recent_messages=10       # Always keep last 10 messages
)

# Add messages
memory.add_message(MessageRole.SYSTEM, "You are a helpful assistant.")
memory.add_message(MessageRole.USER, "Hello, how are you?")
memory.add_message(MessageRole.ASSISTANT, "I'm doing well! How can I help?")

# Convenience methods
memory.add_user_message("What's the weather like?")
memory.add_assistant_message("I don't have weather data.")
memory.add_system_message("Remember: be concise.")

# Get recent messages
recent = memory.get_recent_messages(n=10)
for msg in recent:
    print(f"[{msg.role.value}]: {msg.content[:50]}...")

# Check token count
token_count = memory.get_token_count()
print(f"Tokens used: {token_count}/{memory.max_tokens}")

# Clear memory
memory.clear()`}
        language="python"
        filename="short_term_memory.py"
      />

      <h3>Message Roles</h3>

      <ApiTable
        headers={['Role', 'Description', 'Usage']}
        rows={[
          [<code>SYSTEM</code>, 'System instructions', 'Set agent behavior and constraints'],
          [<code>USER</code>, 'User messages', 'Store user inputs'],
          [<code>ASSISTANT</code>, 'Agent responses', 'Store agent outputs'],
          [<code>TOOL</code>, 'Tool results', 'Store tool execution results'],
        ]}
      />

      <h3>Auto-Summarization</h3>
      <p>
        When memory approaches capacity, older messages are automatically summarized:
      </p>

      <CodeBlock
        code={`memory = ShortTermMemory(
    max_tokens=4096,
    summarization_threshold=0.8  # Trigger at 80% capacity
)

# After many messages, older content is summarized
# [Original]: 100 detailed messages
# [After summarization]: Summary block + recent messages

# Check how many summarisation passes ran
print(f"Summarisations: {memory.total_summarizations}")
print(f"Summary blocks: {len(memory.summaries)}")`}
        language="python"
        filename="summarization.py"
      />

      <h2>Long-Term Memory</h2>
      <p>
        Long-term memory provides persistent storage across sessions with multiple backend options.
      </p>

      <CodeBlock
        code={`from effgen.memory.long_term import (
    LongTermMemory,
    JSONStorageBackend,
    SQLiteStorageBackend,
    MemoryType,
    ImportanceLevel
)

# Create with JSON backend (simple, file-based)
json_backend = JSONStorageBackend(filepath="./memory.json")
memory = LongTermMemory(backend=json_backend, max_memories=10000)

# Or use SQLite backend (better for large datasets)
sqlite_backend = SQLiteStorageBackend(db_path="./memory.db")
memory = LongTermMemory(backend=sqlite_backend)

# Start a session
session = memory.start_session(name="research_session")

# Add memories with metadata
memory.add_memory(
    content="Python was created by Guido van Rossum in 1991",
    memory_type=MemoryType.FACT,
    importance=ImportanceLevel.HIGH,
    tags=["python", "programming", "history"],
    metadata={"source": "wikipedia", "verified": True}
)

memory.add_memory(
    content="User prefers detailed technical explanations",
    memory_type=MemoryType.OBSERVATION,
    importance=ImportanceLevel.MEDIUM,
    tags=["user_preference"]
)

# End session
memory.end_session()`}
        language="python"
        filename="long_term_memory.py"
      />

      <h3>Memory Types</h3>

      <ApiTable
        headers={['Type', 'Description', 'Example']}
        rows={[
          [<code>FACT</code>, 'Verified information', '"Python was created in 1991"'],
          [<code>OBSERVATION</code>, 'Observed patterns', '"User prefers code examples"'],
          [<code>CONVERSATION</code>, 'Important exchanges', 'Key discussion points'],
          [<code>TASK</code>, 'Task-related info', '"Current project: web scraper"'],
          [<code>TOOL_RESULT</code>, 'Tool outputs', 'Search results, calculations'],
          [<code>REFLECTION</code>, 'Agent insights', 'Learning from interactions'],
        ]}
      />

      <h3>Importance Levels</h3>

      <ApiTable
        headers={['Level', 'Retention', 'Use Case']}
        rows={[
          [<code>CRITICAL</code>, 'Never delete', 'Core facts, user identity'],
          [<code>HIGH</code>, 'Long retention', 'Important context'],
          [<code>MEDIUM</code>, 'Normal retention', 'General information'],
          [<code>LOW</code>, 'Short retention', 'Temporary data'],
        ]}
      />

      <h3>Searching Memory</h3>

      <CodeBlock
        code={`# Search by query
results = memory.search(
    query="python programming",
    limit=10,
)

# Search with filters
results = memory.search(
    query="user preferences",
    tags=["user_preference"],
    min_importance=ImportanceLevel.MEDIUM,
    memory_type=MemoryType.OBSERVATION,
    limit=5,
)

# Get a specific memory by id
specific = memory.get_memory(memory_id="abc123")

# Filter by type — search returns MemoryEntry objects
facts = memory.search(memory_type=MemoryType.FACT, limit=100)

# Process results
for mem in results:
    print(f"[{mem.memory_type.value}] {mem.content}")
    print(f"  Importance: {mem.importance.value}")
    print(f"  Tags: {mem.tags}")
    print(f"  Created: {mem.created_at}")`}
        language="python"
        filename="search_memory.py"
      />

      <h3>Memory Consolidation</h3>
      <p>
        Consolidate and optimize stored memories:
      </p>

      <CodeBlock
        code={`# Consolidate when over capacity — keeps top N memories by
# importance + access frequency - recency, drops the rest.
removed = memory.consolidate()
print(f"Consolidation removed {removed} memories")

# Filter via search() with min_importance / memory_type / tags
high_importance = memory.search(
    min_importance=ImportanceLevel.HIGH,
    limit=100,
)
print(f"High-importance memories: {len(high_importance)}")`}
        language="python"
        filename="consolidation.py"
      />

      <h2>Integration with Agents</h2>

      <CodeBlock
        code={`from effgen import Agent, load_model
from effgen.core.agent import AgentConfig

model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

# Memory is wired in via AgentConfig.memory_config — flat keys, not nested
config = AgentConfig(
    name="memory_agent",
    model=model,
    enable_memory=True,
    memory_config={
        "short_term_max_tokens": 4096,
        "short_term_max_messages": 100,
        "long_term_backend": "sqlite",                   # "json" | "sqlite"
        "long_term_persist_path": "./agent_memory",  # directory; backend appends long_term.db / long_term.json
        "auto_summarize": True,
    },
)
agent = Agent(config=config)

# Multi-turn conversation with memory
agent.run("My name is Alice and I'm a data scientist")
agent.run("I work primarily with Python and pandas")

# Agent remembers context
result = agent.run("What tools should I learn next?")
# Agent considers: name is Alice, data scientist, uses Python/pandas

# Direct access to the underlying memory buffers
print(f"STM messages: {len(agent.short_term_memory.messages)}")
if agent.long_term_memory:
    print(f"LTM total: {agent.long_term_memory.get_statistics()['total_memories']}")`}
        language="python"
        filename="agent_memory.py"
      />

      <h2>Vector Memory Store</h2>
      <p>
        The VectorMemoryStore provides semantic search capabilities using vector embeddings.
        It supports FAISS and ChromaDB backends:
      </p>

      <CodeBlock
        code={`from effgen.memory.vector_store import (
    VectorMemoryStore,
    SentenceTransformerEmbedding,
    SimpleEmbedding,
)

# FAISS backend (recommended for local use)
store = VectorMemoryStore(
    backend_type="faiss",                                      # "faiss" | "chroma"
    embedding_provider=SentenceTransformerEmbedding("all-MiniLM-L6-v2"),
    persist_directory="./vector_store/",
)

# Or use simple TF-IDF embeddings (no extra dependencies)
store = VectorMemoryStore(
    backend_type="faiss",
    embedding_provider=SimpleEmbedding(),
)

# Add entries
store.add("Python is great for data science", metadata={"topic": "python"})
store.add("JavaScript powers the web", metadata={"topic": "javascript"})

# Semantic search — k results above the optional similarity floor
results = store.search("best language for ML", k=3, min_similarity=0.2)
for result in results:
    print(f"Score: {result.similarity:.2f} | {result.entry.content}")`}
        language="python"
        filename="vector_memory.py"
      />

      <h2>Multi-Turn Conversation with Memory</h2>
      <CodeBlock
        code={`from effgen import load_model
from effgen.presets import create_agent

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = create_agent("general", model)

# Memory automatically tracks conversation turns
agent.run("My name is Alice and I'm a data scientist")
agent.run("I work primarily with Python and pandas")
agent.run("I'm interested in deep learning frameworks")

# Agent remembers all previous context
result = agent.run("Based on my background, what should I learn next?")
print(result.output)
# Agent considers: name is Alice, data scientist, Python/pandas, deep learning interest

# Access conversation history through the short-term memory buffer
print(f"Total messages: {len(agent.short_term_memory.messages)}")
for msg in agent.short_term_memory.messages:
    print(f"[{msg.role.value}]: {msg.content[:60]}...")`}
        language="python"
        filename="multi_turn_memory.py"
      />

      <h2>Memory Configuration</h2>
      <p>
        <code>AgentConfig.memory_config</code> is a flat dict with the following keys.
        Defaults shown are what the agent uses when you do not override them.
      </p>
      <CodeBlock
        code={`from effgen.core.agent import AgentConfig

config = AgentConfig(
    name="memory_agent",
    model=model,
    enable_memory=True,
    memory_config={
        "short_term_max_tokens": 4096,
        "short_term_max_messages": 100,
        "long_term_backend": "sqlite",          # "json" | "sqlite"
        "long_term_persist_path": None,         # path for backend (uses default if None)
        "auto_summarize": True,
    },
)`}
        language="python"
        filename="memory_config.py"
      />

      <h2>Best Practices</h2>

      <InfoBox type="info" title="Memory Guidelines">
        <ul>
          <li><strong>Set appropriate limits:</strong> Match max_tokens to your model's context window</li>
          <li><strong>Use importance levels:</strong> Mark critical information appropriately</li>
          <li><strong>Tag memories:</strong> Use tags for efficient retrieval</li>
          <li><strong>Consolidate regularly:</strong> Merge similar memories to reduce redundancy</li>
          <li><strong>Choose the right backend:</strong> JSON for simple use, SQLite for larger datasets</li>
        </ul>
      </InfoBox>

      <InfoBox type="success" title="Next Steps">
        <p>
          Learn about <Link to="/prompts">Prompt Management</Link> for optimizing agent instructions,
          or explore <Link to="/multi-agent">Multi-Agent Systems</Link> for complex workflows.
        </p>
      </InfoBox>
    </DocPage>
  );
}
