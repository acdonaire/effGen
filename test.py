from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL

# Load a small but mighty model
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# Create agent with tools
config = AgentConfig(
    name="math_agent",
    model=model,
    tools=[Calculator(), PythonREPL()]
)
agent = Agent(config=config)

# Run computation
result = agent.run("What is 24344 * 334?")
print(f"Answer: {result.output}")