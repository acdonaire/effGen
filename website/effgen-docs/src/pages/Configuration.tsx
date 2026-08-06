import React from 'react';
import { Link } from 'react-router-dom';
import { Cog } from 'lucide-react';
import DocPage, { InfoBox, ApiTable } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Configuration() {
  return (
    <DocPage
      title="Configuration"
      subtitle="Configure effGen through YAML files, environment variables, and programmatic settings."
      icon={<Cog size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Advanced', path: '/multi-agent' },
        { label: 'Configuration' },
      ]}
    >
      <h2>Configuration Files</h2>
      <p>
        effGen uses YAML configuration files for flexible deployment settings.
      </p>

      <CodeBlock
        code={`# config.yaml
effgen:
  # Model configuration
  model:
    name: "Qwen/Qwen2.5-7B-Instruct"
    engine: "vllm"
    quantization: "4bit"
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.9

  # Agent defaults
  agent:
    max_iterations: 10
    temperature: 0.7
    enable_memory: true
    enable_sub_agents: true
    max_context_length: 8192

  # Tool configuration
  tools:
    enabled:
      - calculator
      - code_executor
      - python_repl
    code_executor:
      sandbox_type: "docker"
      timeout: 30
      memory_limit: "512m"

  # Memory settings
  memory:
    short_term:
      max_tokens: 4096
      max_messages: 100
    long_term:
      backend: "sqlite"
      db_path: "./data/memory.db"

  # API keys (use environment variables)
  api_keys:
    openai: \${OPENAI_API_KEY}
    anthropic: \${ANTHROPIC_API_KEY}

  # Logging
  logging:
    level: "INFO"
    file: "./logs/effgen.log"
    format: "json"`}
        language="yaml"
        filename="config.yaml"
      />

      <h2>Loading Configuration</h2>

      <CodeBlock
        code={`from effgen.config import ConfigLoader, ConfigValidator

# Initialize config loader
loader = ConfigLoader()

# Load from file
config = loader.load_config("config.yaml")

# Access settings using get method
model_name = loader.get("model.default_model")
max_iterations = loader.get("agent.max_iterations", default=10)

# Set configuration values
loader.set("agent.temperature", 0.7)

# Save configuration
loader.save_config("config.yaml")

# Validate configuration
validator = ConfigValidator()
result = validator.validate_all(config.data)

if not result.valid:
    for error in result.errors:
        print(f"Config error: {error}")
for warning in result.warnings:
    print(f"Config warning: {warning}")

# Reload configuration
loader.reload()

# v0.3.0: ConfigLoader.load is an additive alias for load_config
config = loader.load("config.yaml")`}
        language="python"
        filename="load_config.py"
      />

      <h2>Environment Variables</h2>

      <ApiTable
        headers={['Variable', 'Description', 'Default']}
        rows={[
          [<code>EFFGEN_MODEL</code>, 'Default model name', 'Qwen/Qwen2.5-3B-Instruct'],
          [<code>EFFGEN_ENGINE</code>, 'Model engine', 'transformers'],
          [<code>EFFGEN_LOG_LEVEL</code>, 'Logging level', 'INFO'],
          [<code>OPENAI_API_KEY</code>, 'OpenAI API key', '-'],
          [<code>OPENAI_ORG_ID</code>, 'Optional OpenAI organization ID', '-'],
          [<code>ANTHROPIC_API_KEY</code>, 'Anthropic API key', '-'],
          [<code>GOOGLE_API_KEY</code>, 'Google Gemini API key', '-'],
          [<code>CEREBRAS_API_KEY</code>, 'Cerebras API key', '-'],
          [<code>GROQ_API_KEY</code>, 'Groq API key', '-'],
          [<code>TOGETHER_API_KEY</code>, 'Together AI API key', '-'],
          [<code>FIREWORKS_API_KEY</code>, 'Fireworks API key', '-'],
          [<code>REPLICATE_API_TOKEN</code>, 'Replicate API token', '-'],
          [<code>HF_TOKEN</code>, 'HuggingFace Inference token', '-'],
        ]}
      />

      <CodeBlock
        code={`# .env file
EFFGEN_MODEL=Qwen/Qwen2.5-7B-Instruct
EFFGEN_ENGINE=vllm
EFFGEN_LOG_LEVEL=DEBUG

OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org_...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
CEREBRAS_API_KEY=...
GROQ_API_KEY=...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
REPLICATE_API_TOKEN=...
HF_TOKEN=hf_...

# v0.2.3+: verify provider auth
effgen doctor

# Load in Python
from dotenv import load_dotenv
load_dotenv()

from effgen import load_model
import os

model = load_model(os.environ["EFFGEN_MODEL"])`}
        language="bash"
        filename=".env"
      />

      <h2>Programmatic Configuration</h2>

      <CodeBlock
        code={`from effgen.config import Config, ConfigLoader

# Build configuration programmatically — Config wraps a plain dict
config = Config(data={
    "model": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "engine": "vllm",
        "tensor_parallel_size": 2,
    },
    "agent": {
        "max_iterations": 15,
        "temperature": 0.3,
    },
    "tools": {
        "code_executor": {"sandbox_type": "docker", "timeout": 60},
    },
})

# Read with dot-paths via ConfigLoader
loader = ConfigLoader()
loader.config = config
print(loader.get("model.name"))         # "Qwen/Qwen2.5-7B-Instruct"
print(loader.get("agent.max_iterations", 10))

# Use values to build the agent
from effgen import Agent, AgentConfig, load_model
model = load_model(loader.get("model.name"), engine=loader.get("model.engine"))
agent = Agent(AgentConfig(
    name="from-config",
    model=model,
    max_iterations=loader.get("agent.max_iterations", 10),
    temperature=loader.get("agent.temperature", 0.7),
))`}
        language="python"
        filename="programmatic_config.py"
      />

      <h2>Deployment Profiles</h2>

      <CodeBlock
        code={`# config/development.yaml
effgen:
  model:
    engine: "transformers"
    quantization: "4bit"
  logging:
    level: "DEBUG"

# config/production.yaml
effgen:
  model:
    engine: "vllm"
    tensor_parallel_size: 4
  logging:
    level: "INFO"
    file: "/var/log/effgen/app.log"

# Load based on environment
import os
from effgen.config import ConfigLoader

env = os.getenv("ENVIRONMENT", "development")
loader = ConfigLoader()
config = loader.load_config(f"config/{env}.yaml")`}
        language="yaml"
        filename="profiles.yaml"
      />

      <h2>Memory Configuration</h2>

      <CodeBlock
        code={`# config.yaml - Memory section
effgen:
  memory:
    short_term:
      max_tokens: 4096
      max_messages: 100
      summarization_threshold: 0.8
    long_term:
      backend: "sqlite"        # "json" or "sqlite"
      db_path: "./memory.db"
      max_memories: 10000
    vector_store:
      backend: "faiss"         # "faiss" or "chroma"
      embedding: "sentence-transformers"
      index_path: "./vectors.faiss"
    auto_summarize: true
    persistence_path: "./memory_data/"`}
        language="yaml"
        filename="memory_config.yaml"
      />

      <h2>API Server Configuration</h2>

      <ApiTable
        headers={['Variable', 'Description', 'Default']}
        rows={[
          [<code>EFFGEN_API_KEY</code>, 'API key for authenticating requests to the effGen server', '-'],
          [<code>EFFGEN_RATE_LIMIT</code>, 'Max requests per minute per client', '60'],
          [<code>EFFGEN_HOST</code>, 'API server bind address', '0.0.0.0'],
          [<code>EFFGEN_PORT</code>, 'API server port', '8000'],
        ]}
      />

      <CodeBlock
        code={`# Start the API server
effgen serve --port 8000

# Or configure in YAML:
# config.yaml
effgen:
  api:
    host: "0.0.0.0"
    port: 8000
    auth:
      api_key: "\${EFFGEN_API_KEY}"
    rate_limit: 60            # requests per minute
    cors_origins: ["*"]
    workers: 4`}
        language="yaml"
        filename="api_config.yaml"
      />

      <h2>Streaming Configuration</h2>

      <CodeBlock
        code={`# Enable streaming in config
effgen:
  agent:
    enable_streaming: true     # Enable token streaming
  api:
    streaming:
      enabled: true
      ws_endpoint: "/ws"       # WebSocket endpoint path
      heartbeat_interval: 15   # Seconds between keepalive pings`}
        language="yaml"
        filename="streaming_config.yaml"
      />

      <h2>Secret Management</h2>

      <InfoBox type="warning" title="Security">
        <p>Never commit API keys or secrets to version control. Use environment variables or secret managers.</p>
      </InfoBox>

      <CodeBlock
        code={`import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access secrets from environment variables
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
hf_token = os.getenv("HF_TOKEN")

# Use secrets with effgen
from effgen import load_model

# API keys are automatically picked up from environment
model = load_model(
    "gpt-5.4-nano",
    provider="openai",
    api_key=openai_key  # Or let it auto-detect from env
)

# Best practice: Never hardcode secrets
# ✓ Good: Use environment variables
# ✗ Bad: api_key = "sk-..."

# Validate configuration has no plaintext secrets
from effgen.config import ConfigValidator
validator = ConfigValidator()
# Add custom validation as needed`}
        language="python"
        filename="secrets.py"
      />

      <InfoBox type="success" title="Next Steps">
        <p>
          Check the <Link to="/api-reference">API Reference</Link> for complete documentation,
          or see <Link to="/examples">Examples</Link> for configuration patterns.
        </p>
      </InfoBox>
    </DocPage>
  );
}
