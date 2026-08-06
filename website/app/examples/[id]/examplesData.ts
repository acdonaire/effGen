// Content for every example detail page, keyed by the id in the URL.
// Kept out of the view component so the route can enumerate the ids it
// generates without a second copy of the data.
export const examplesData: Record<string, any> = {
  "code-assistant": {
    icon: "💻", title: "AI Code Assistant", subtitle: "A powerful coding companion that helps you write better code faster.", badge: "Code Generation",
    description: "The AI Code Assistant is a production-ready example of how effGen can be used to create sophisticated coding tools. It leverages Small Language Models to provide intelligent code completion, bug detection, and refactoring suggestions across 50+ programming languages.",
    accent: "#00e5ff",
    features: [
      { icon: "⚡", title: "Real-time Code Completion", description: "Get intelligent code suggestions as you type with context-aware completions." },
      { icon: "🐛", title: "Smart Bug Detection", description: "Automatically identify potential bugs and security vulnerabilities in your code." },
      { icon: "🔄", title: "Code Refactoring", description: "Receive suggestions for improving code quality and maintainability." },
      { icon: "💬", title: "Natural Language to Code", description: "Convert plain English descriptions into working code snippets." },
    ],
    useCases: [
      { icon: "🎯", title: "IDE Integration", description: "Integrate directly into your development environment for seamless assistance." },
      { icon: "📚", title: "Learning Tool", description: "Help developers learn new languages and best practices." },
      { icon: "🏢", title: "Enterprise Development", description: "Accelerate development cycles and maintain code quality standards." },
    ],
    codeExample: `from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import CodeExecutor, WebSearch, FileOperations

# Load the code assistant model
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="vllm",
    tensor_parallel_size=2
)

# Create the code assistant agent
assistant = Agent(AgentConfig(
    model=model,
    name="code-assistant",
    system_prompt="""You are an expert coding assistant.
    Provide clear, efficient, and well-documented code.""",
    tools=[CodeExecutor(), WebSearch(), FileOperations()],
    enable_memory=True,
))

# Use the assistant
result = assistant.run("""
    Create a Python function that efficiently finds
    the longest palindromic substring in a given string.
    Include error handling and unit tests.
""")

print(result.output)`,
    stats: [{ value: "50+", label: "Languages Supported" }, { value: "95%", label: "Accuracy Rate" }, { value: "3x", label: "Faster Development" }, { value: "10K+", label: "Lines Analyzed/sec" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/tools/coding_agent.py",
  },
  "research-agent": {
    icon: "🔍", title: "Research Agent", subtitle: "An intelligent research assistant that gathers, analyzes, and synthesizes information.", badge: "Information Gathering",
    description: "The Research Agent demonstrates how effGen can autonomously conduct comprehensive research tasks. It combines web search, data extraction, and synthesis to provide well-researched, cited reports on any topic.",
    accent: "#a78bfa",
    features: [
      { icon: "🌐", title: "Multi-Source Research", description: "Gather information from multiple authoritative sources simultaneously." },
      { icon: "✅", title: "Fact Verification", description: "Cross-reference facts across sources to ensure accuracy." },
      { icon: "📎", title: "Citation Management", description: "Automatically generate and manage citations for all sources." },
      { icon: "📝", title: "Summary Generation", description: "Create comprehensive summaries with key insights highlighted." },
    ],
    useCases: [
      { icon: "🎓", title: "Academic Research", description: "Accelerate literature reviews and research paper preparation." },
      { icon: "💼", title: "Market Analysis", description: "Conduct competitive intelligence and market research." },
      { icon: "📰", title: "Journalism", description: "Research and verify facts for news articles and reports." },
    ],
    codeExample: `from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import WebSearch, WikipediaTool, Calculator

# Load the research model
model = load_model("Qwen/Qwen2.5-7B-Instruct")

# Create the research agent
researcher = Agent(AgentConfig(
    model=model,
    name="research-agent",
    system_prompt="""You are a thorough research assistant.
    Always verify facts and cite your sources.""",
    tools=[WebSearch(), WikipediaTool(), Calculator()],
    enable_memory=True,
    max_iterations=15,
))

# Conduct research
result = researcher.run("""
    Research the latest developments in quantum computing
    from 2024. Focus on:
    1. Major breakthroughs
    2. Key companies and institutions
    3. Practical applications
    4. Future outlook

    Provide a comprehensive report with citations.
""")

print(result.output)`,
    stats: [{ value: "10+", label: "Sources Per Query" }, { value: "90%", label: "Fact Accuracy" }, { value: "5x", label: "Faster Research" }, { value: "100+", label: "Topics Covered" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/web_retrieval/web_agent.py",
  },
  "data-analysis": {
    icon: "📊", title: "Data Analysis Agent", subtitle: "Automate complex data analysis workflows with intelligent processing.", badge: "Data Science",
    description: "The Data Analysis Agent showcases how effGen handles sophisticated data science tasks. It combines data processing, statistical analysis, and visualization capabilities to extract meaningful insights.",
    accent: "#00ff88",
    features: [
      { icon: "🧹", title: "Automated Data Cleaning", description: "Detect and fix data quality issues automatically." },
      { icon: "📈", title: "Statistical Analysis", description: "Perform comprehensive statistical tests and analyses." },
      { icon: "📉", title: "Interactive Visualizations", description: "Generate publication-quality charts and graphs." },
      { icon: "🔍", title: "Pattern Recognition", description: "Identify trends, anomalies, and patterns in data." },
    ],
    useCases: [
      { icon: "💹", title: "Financial Analysis", description: "Analyze market trends and financial performance metrics." },
      { icon: "🏥", title: "Healthcare Analytics", description: "Process and analyze patient data for insights." },
      { icon: "🛒", title: "E-commerce Insights", description: "Understand customer behavior and optimize sales." },
    ],
    codeExample: `from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import CodeExecutor, FileOperations, Calculator

# Load the data analysis model
model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm")

# Create the data analysis agent
analyst = Agent(AgentConfig(
    model=model,
    name="data-analyst",
    system_prompt="""You are an expert data scientist.
    Provide thorough analysis with visualizations.""",
    tools=[CodeExecutor(), FileOperations(), Calculator()],
    enable_memory=True,
))

# Analyze data
result = analyst.run("""
    Analyze the sales_data.csv file:
    1. Clean and validate the data
    2. Calculate key metrics (revenue, growth, trends)
    3. Identify top-performing products
    4. Create visualizations
    5. Provide actionable insights
""")

print(result.output)`,
    stats: [{ value: "1M+", label: "Rows/Second" }, { value: "20+", label: "File Formats" }, { value: "10x", label: "Faster Analysis" }, { value: "50+", label: "Chart Types" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/advanced/data_processing_agent.py",
  },
  "multi-agent": {
    icon: "🤖", title: "Multi-Agent System", subtitle: "Build complex workflows with multiple specialized agents collaborating.", badge: "Orchestration",
    description: "The Multi-Agent System demonstrates the power of agent collaboration in effGen. Multiple specialized agents work together, each handling specific aspects of complex tasks.",
    accent: "#ff6b6b",
    features: [
      { icon: "🎭", title: "Agent Orchestration", description: "Coordinate multiple agents with different specializations." },
      { icon: "📋", title: "Task Delegation", description: "Intelligently distribute tasks based on agent capabilities." },
      { icon: "⚡", title: "Parallel Execution", description: "Run multiple agents simultaneously for maximum efficiency." },
      { icon: "🔗", title: "Result Synthesis", description: "Combine outputs from multiple agents into cohesive results." },
    ],
    useCases: [
      { icon: "🏗️", title: "Complex Workflows", description: "Handle multi-step processes requiring different expertise." },
      { icon: "🔬", title: "Research Projects", description: "Coordinate research, analysis, and documentation tasks." },
      { icon: "🎯", title: "Business Automation", description: "Automate complex business processes end-to-end." },
    ],
    codeExample: `from effgen import load_model
from effgen.presets import create_agent
from effgen.core.workflow import WorkflowDAG, WorkflowNode

# Load the model
model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm")

# Create specialized agents via presets
researcher = create_agent("research", model)   # WebSearch + URLFetch + Wikipedia
analyst    = create_agent("math",     model)   # Calculator + PythonREPL
writer     = create_agent("general",  model)   # All-purpose tools

# Wire them into a DAG — independent levels run in parallel via asyncio.gather
dag = WorkflowDAG(name="research_to_report")
dag.add_node(WorkflowNode(id="research", agent=researcher, output_key="facts"))
dag.add_node(WorkflowNode(id="analyze",  agent=analyst,    input_keys=["facts"], output_key="analysis"))
dag.add_node(WorkflowNode(id="write",    agent=writer,
                          input_keys=["facts", "analysis"], output_key="report"))

dag.connect("research", "analyze", key="facts")
dag.connect("research", "write",   key="facts")
dag.connect("analyze",  "write",   key="analysis")

result = dag.run(initial_inputs={
    "research": "Research AI trends in 2026, analyze the data, and create a comprehensive report.",
})

print(result.outputs["write"])`,
    stats: [{ value: "5+", label: "Specialized Agents" }, { value: "100%", label: "Parallel Execution" }, { value: "15x", label: "Efficiency Gain" }, { value: "\u221E", label: "Scalability" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/advanced/multi_agent_pipeline.py",
  },
  "weather-json-pipeline": {
    icon: "🌤️", title: "Weather & JSON Pipeline", subtitle: "Fetch real-time weather data and process JSON responses automatically.", badge: "Data Pipeline",
    description: "The Weather & JSON Pipeline demonstrates how effGen's WeatherTool and JSONTool work together to create automated data pipelines. Fetch weather from the free Open-Meteo API and process with JSONPath — no API keys required.",
    accent: "#ffd700",
    features: [
      { icon: "🌡️", title: "Real-Time Weather Data", description: "Fetch current weather, forecasts, and historical data from Open-Meteo API (free, no key)." },
      { icon: "📋", title: "JSON Processing", description: "Parse, query with JSONPath, transform, and validate JSON data automatically." },
      { icon: "📊", title: "Report Generation", description: "Combine weather data with text processing to generate formatted reports." },
      { icon: "🔄", title: "Pipeline Automation", description: "Chain multiple tools together for end-to-end data pipeline workflows." },
    ],
    useCases: [
      { icon: "🏙️", title: "City Weather Dashboards", description: "Build automated weather monitoring for multiple cities." },
      { icon: "🌾", title: "Agriculture Planning", description: "Analyze weather patterns for crop planning and irrigation." },
      { icon: "✈️", title: "Travel Planning", description: "Generate weather-based travel recommendations." },
    ],
    codeExample: `from effgen import load_model
from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin import WeatherTool, JSONTool

# Load model with 4-bit quantization
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# Create a custom agent with WeatherTool + JSONTool
agent = Agent(config=AgentConfig(
    name="weather_agent", model=model,
    tools=[WeatherTool(), JSONTool()]
))

# Fetch and process weather data
result = agent.run("""
    Get the current weather for San Francisco and Tokyo.
    Compare temperatures, humidity, and conditions.
    Format the results as a clean comparison table.
""")

print(result.output)`,
    stats: [{ value: "Free", label: "No API Key" }, { value: "31", label: "Tools Available" }, { value: "Real-time", label: "Weather Data" }, { value: "JSONPath", label: "Query Support" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/web_retrieval/weather_agent.py",
  },
  "rag-knowledge-base": {
    icon: "📚", title: "RAG Knowledge Base", subtitle: "Build intelligent knowledge bases with hybrid search capabilities.", badge: "Knowledge Management",
    description: "The RAG Knowledge Base showcases effGen's Retrieval tool with hybrid search combining vector similarity and BM25 keyword matching. Load from multiple formats, chunk intelligently, and query with semantic understanding.",
    accent: "#ff9500",
    features: [
      { icon: "📄", title: "Multi-Format Document Loading", description: "Load and process documents from TXT, Markdown, PDF, CSV, and JSON formats." },
      { icon: "🔍", title: "Hybrid Search", description: "Combine vector similarity search with BM25 keyword matching for best results." },
      { icon: "✂️", title: "Smart Chunking", description: "Intelligent document chunking with configurable overlap and size parameters." },
      { icon: "🧠", title: "Semantic Understanding", description: "Vector embeddings enable semantic search beyond simple keyword matching." },
    ],
    useCases: [
      { icon: "📖", title: "Documentation Q&A", description: "Build chatbots that answer questions from your documentation." },
      { icon: "⚖️", title: "Legal Research", description: "Search and analyze legal documents with semantic understanding." },
      { icon: "🏥", title: "Medical Knowledge Base", description: "Build searchable medical reference systems from research papers." },
    ],
    codeExample: `from effgen import load_model
from effgen.presets import create_agent
from effgen.tools.builtin import Retrieval

# Load model
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# Create research agent with retrieval capabilities
agent = create_agent("research", model)

# Build knowledge base from documents
result = agent.run("""
    Load all markdown files from ./docs/ directory.
    Build a searchable knowledge base with hybrid search.
    Then answer: What are the main features of the system?
""")

print(result.output)`,
    stats: [{ value: "5+", label: "File Formats" }, { value: "Hybrid", label: "Search Mode" }, { value: "BM25", label: "Keyword Search" }, { value: "Vector", label: "Semantic Search" }],
    githubUrl: "https://github.com/ctrl-gaurav/effGen/blob/main/examples/web_retrieval/retrieval_agent.py",
  },
};
