import React from 'react';
import { Link } from 'react-router-dom';
import { FileText } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';

export default function Prompts() {
  const chainDiagram = `
flowchart LR
    I[Input] --> S1[Step 1: Research]
    S1 --> S2[Step 2: Analyze]
    S2 --> S3[Step 3: Synthesize]
    S3 --> O[Output]

    style S1 fill:#00c96e,color:#fff
    style S2 fill:#009950,color:#fff
    style S3 fill:#00b8d4,color:#fff
`;

  return (
    <DocPage
      title="Prompt Management"
      subtitle="effGen provides a comprehensive prompt engineering system: the Prompt Library (35 curated templates across 8 domains with golden + live evals, a CLI, and a playground), plus Jinja2 TemplateManager, ChainManager, PromptOptimizer, and few-shot helpers."
      icon={<FileText size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Core Concepts', path: '/agents' },
        { label: 'Prompts' },
      ]}
    >
      <h2>Overview</h2>
      <p>
        Effective prompts are crucial for getting the best results from Small Language Models.
        effGen provides tools specifically designed for SLM optimization:
      </p>

      <FeatureList
        features={[
          { icon: '📚', title: 'Prompt Library', description: '35 curated, domain-organized templates across 8 domains with golden + live evals, CLI, and playground' },
          { icon: '📝', title: 'Template Manager', description: 'Jinja2-based templates with variables and conditionals' },
          { icon: '🔗', title: 'Chain Manager', description: 'Multi-step prompt chains for complex workflows' },
          { icon: '⚡', title: 'Prompt Optimizer', description: 'SLM-specific optimization for smaller models' },
          { icon: '📦', title: 'Few-Shot Learning', description: 'Example-based learning for improved accuracy' },
        ]}
      />

      <h2>Prompt Library (v0.2.7)</h2>
      <p>
        <strong>v0.2.7</strong> introduced <code>effgen.prompts.library</code> — a curated catalog
        now totaling <strong>35 reusable prompt templates</strong> across eight domains (the{' '}
        <code>education</code> set was added in v0.3.1). Every template is a Python callable that
        renders deterministically for fixed inputs and ships with a fixture, a stored golden, and
        (optionally) an <code>expected_shape</code> for validating live model output.
      </p>

      <ApiTable
        headers={['Domain', 'Count', 'Templates']}
        rows={[
          ['Research', '5', 'literature_review.v1.zero_shot / .cot, paper_summary.v1, citation_extract.v1, methodology_critique.v1'],
          ['Coding', '5', 'code_review.v1, bug_diagnose.v1, refactor_plan.v1, test_generate.v1, docstring_fill.v1'],
          ['Data / SQL', '5', 'sql_from_nl.v1, sql_explain.v1, sql_optimize.v1, data_profile.v1, etl_plan.v1'],
          ['Legal', '3', 'contract_summarize.v1, clause_classify.v1, legal_research_brief.v1'],
          ['Medical', '3', 'symptom_triage.v1, drug_interaction_query.v1, medical_literature.v1'],
          ['Creative', '5', 'story_continuation.v1.zero_shot / .few_shot, poetry_forms.v1, character_bio.v1, world_building.v1'],
          ['Business', '5', 'meeting_summary.v1, email_draft.v1, okr_generate.v1, swot_analysis.v1, elevator_pitch.v1'],
          ['Education', '4', 'socratic_tutor.v1, lesson_plan.v1, quiz_generate.v1, explain_simply.v1 (v0.3.1)'],
        ]}
      />

      <h3>Python API</h3>
      <CodeBlock
        code={`from effgen.prompts.library import registry

# Browse every registered template
for p in registry.all():
    print(p.name, p.variant, p.domain)

# Get a specific template and render it
p = registry.get("data.sql_from_nl.v1")
prompt_text = p.template(
    schema_ddl="CREATE TABLE orders (id INT, total FLOAT)",
    question="Total orders this month",
    dialect="sqlite",
)
print(prompt_text)

# Search by domain + variant
sql = registry.search(domain="data", variant="structured")
coding_cot = registry.search(domain="coding", variant="cot")`}
        language="python"
        filename="prompt_library_python.py"
      />

      <p>
        Each <code>LibraryPrompt</code> declares:{' '}
        <code>name</code> (e.g. <code>"research.literature_review.v1.cot"</code>),{' '}
        <code>domain</code>, <code>variant</code> (one of <code>zero_shot</code>,{' '}
        <code>cot</code>, <code>few_shot</code>, <code>tool</code>, <code>structured</code> —
        validated by the registry), <code>description</code>, <code>template</code> (the
        deterministic render callable), <code>input_schema</code> (JSON Schema),{' '}
        <code>fixture</code>, optional <code>expected_shape</code>, and <code>tags</code>.
        Domain packages under <code>effgen/prompts/library/domains/</code> are auto-discovered
        on first access to <code>registry</code>.
      </p>

      <h3>CLI quick-start</h3>
      <CodeBlock
        code={`# Discover templates
effgen prompts list
effgen prompts list --domain research --variant cot
effgen prompts list --format markdown          # regenerate docs/prompts/gallery.md

# Inspect a specific template (description, input_schema, fixture, expected_shape)
effgen prompts show research.literature_review.v1.cot

# Run golden evaluations (no model needed) — first run writes goldens
effgen prompts eval

# Live eval against a model — validates expected_shape per template
effgen prompts eval --domain coding --live --model llama3.1-8b

# Non-interactive render / run
effgen prompts render data.sql_from_nl.v1 \\
    --input '{"schema_ddl": "CREATE TABLE orders (id INT, total FLOAT)", "question": "Total orders this month", "dialect": "sqlite"}'

effgen prompts run    data.sql_from_nl.v1 \\
    --input ... --model groq:llama-3.3-70b-versatile

# Interactive REPL: select / set / unset / render / run / save / load / reload / list / show / help / exit
effgen prompts playground`}
        language="bash"
        filename="terminal"
      />

      <h3>Golden + live eval harness</h3>
      <CodeBlock
        code={`from effgen.prompts.library import registry, PromptEval

# Optional kwargs: goldens_dir=Path(...), live_retries=1, live_retry_delay=65.0
evaluator = PromptEval()

# Compare every template's render(fixture) against its stored .txt golden.
# A missing golden is written on first run and counted as a pass.
report = evaluator.eval_all_golden(list(registry))
print(report.as_table())
for r in report.failed():
    print(r.name, r.message)

# Live eval: pass a LibraryPrompt instance (not a name string).
#   - SQL templates use sqlglot.parse() to confirm syntactic validity
#   - coding.test_generate.v1 uses ast.parse() on generated Python
#   - structured templates validate against the expected_shape JSON schema
prompt = registry.get("coding.test_generate.v1")
live = evaluator.eval_live(prompt, model="llama3.1-8b")
print(live.passed, live.message)
print(live.model_output[:200])`}
        language="python"
        filename="prompt_eval.py"
      />

      <InfoBox type="warning" title="Legal + medical safety">
        <p>
          Every legal and medical template renders a verbatim non-advice disclaimer in its system
          prompt — enforced by unit tests, not convention. Removing or paraphrasing the disclaimer
          will fail the test suite.
        </p>
      </InfoBox>

      <InfoBox type="info" title="Auto-generated gallery">
        <p>
          Regenerate <code>docs/prompts/gallery.md</code> (the canonical listing of every shipped
          template with name, variant, and description) any time with{' '}
          <code>effgen prompts list --format markdown</code>.
        </p>
      </InfoBox>

      <h2>Template Manager</h2>
      <p>
        Use Jinja2 templates for consistent, reusable prompts:
      </p>

      <CodeBlock
        code={`from effgen.prompts import TemplateManager, PromptTemplate

manager = TemplateManager()

# Create and register a PromptTemplate
manager.add_template(PromptTemplate(
    name="code_review",
    template="""You are a code reviewer. Review the following code:

Language: {{ language }}
Code:
\`\`\`{{ language }}
{{ code }}
\`\`\`

Review criteria:
{% for criterion in criteria %}
- {{ criterion }}
{% endfor %}

Provide specific, actionable feedback.""",
))

# Render via render_template(name, variables=...)
prompt = manager.render_template(
    name="code_review",
    variables={
        "language": "python",
        "code": "def add(a, b):\\n    return a + b",
        "criteria": ["Code quality", "Error handling", "Performance"],
    },
)

print(prompt)`}
        language="python"
        filename="template_manager.py"
      />

      <h3>Built-in Templates</h3>
      <p>
        Use <code>create_default_template_manager()</code> to load the templates shipped with
        effGen from <code>effgen/prompts/templates/*.yaml</code>:
      </p>

      <ApiTable
        headers={['Template', 'Use Case']}
        rows={[
          [<code>analysis</code>, 'Data / problem analysis'],
          [<code>coding</code>, 'Code generation and modification'],
          [<code>general</code>, 'General-purpose assistant'],
          [<code>reasoning</code>, 'Multi-step reasoning, chain-of-thought'],
        ]}
      />

      <CodeBlock
        code={`from effgen.prompts import create_default_template_manager

manager = create_default_template_manager()
print(manager.list_templates())   # ['analysis', 'coding', 'general', 'reasoning']`}
        language="python"
        filename="default_templates.py"
      />

      <h2>Prompt Chains</h2>
      <p>
        Chain multiple prompts for complex, multi-step workflows:
      </p>

      <MermaidDiagram chart={chainDiagram} title="Sequential Prompt Chain" />

      <CodeBlock
        code={`import asyncio
from effgen.prompts import ChainManager, ChainType, ChainStep, PromptChain
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig

model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")
agent = Agent(AgentConfig(name="chain_agent", model=model))

# Wire the agent in as the chain executor
def executor(prompt: str, **kwargs) -> str:
    return agent.run(prompt).output

manager = ChainManager(executor=executor)

# Build a sequential chain with three prompt steps
chain = PromptChain(name="research_to_summary", chain_type=ChainType.SEQUENTIAL)
chain.add_step(ChainStep(
    name="research",
    type="prompt",
    prompt="Research the topic: {{ topic }}. List 5 key facts.",
    output_var="research_data",
))
chain.add_step(ChainStep(
    name="analyze",
    type="prompt",
    prompt="Analyse this data: {{ research_data }}. Identify patterns and insights.",
    output_var="analysis",
))
chain.add_step(ChainStep(
    name="synthesize",
    type="prompt",
    prompt="Create a comprehensive summary from: {{ analysis }}",
    output_var="summary",
))
manager.add_chain(chain)

# execute_chain is async — pass initial state, get back the final ChainState
state = asyncio.run(manager.execute_chain(
    chain="research_to_summary",
    initial_state={"topic": "quantum computing advances in 2026"},
))
print(state.variables["summary"])`}
        language="python"
        filename="prompt_chains.py"
      />

      <h3>Chain Types</h3>

      <ApiTable
        headers={['Type', 'Description', 'Use Case']}
        rows={[
          [<code>SEQUENTIAL</code>, 'Steps run one after another', 'Multi-step analysis'],
          [<code>PARALLEL</code>, 'Steps run simultaneously', 'Independent sub-tasks'],
          [<code>CONDITIONAL</code>, 'Steps based on conditions', 'Branching workflows'],
        ]}
      />

      <h2>Prompt Optimizer</h2>
      <p>
        Optimize prompts specifically for Small Language Models:
      </p>

      <CodeBlock
        code={`from effgen.prompts import (
    PromptOptimizer, OptimizationConfig, ModelSize,
    create_optimizer_for_model,
)

# Configure for a target SLM size
config = OptimizationConfig.for_model_size(ModelSize.SMALL)   # 2B-3B class
optimizer = PromptOptimizer(config=config)

# Original verbose prompt
original = """
I would like you to please carefully and thoroughly analyze the following
complex dataset that I am providing to you. Please consider all possible
factors and variables when conducting your comprehensive analysis. Make sure
to provide detailed insights and recommendations based on your findings.
"""

# Optimize: returns OptimizationResult with original/optimised text + diff metadata
result = optimizer.optimize(prompt=original)
print(result.optimized_prompt)
print(f"Tokens saved: {result.get_savings()}  (compression: {result.get_compression_percentage():.1f}%)")
print(f"Techniques applied: {result.techniques_applied}")

# Or auto-pick a config from a model name
optimizer = create_optimizer_for_model("Qwen/Qwen2.5-3B-Instruct")
result = optimizer.optimize_for_task(original, task_type="analysis")`}
        language="python"
        filename="prompt_optimizer.py"
      />

      <h3>Optimization Targets</h3>

      <FeatureList
        features={[
          { icon: '✂️', title: 'Concise instructions', description: 'Strip verbose / hedging language and pleasantries' },
          { icon: '📋', title: 'Structured output', description: 'Inject explicit output schema / format hints' },
          { icon: '📝', title: 'Step-by-step', description: 'Break complex tasks into numbered steps for SLMs' },
          { icon: '🔄', title: 'Token budgeting', description: 'Trim prompt to OptimizationConfig.max_prompt_tokens' },
          { icon: '🎯', title: 'Few-shot ratio', description: 'Pick example count appropriate to model size' },
        ]}
      />

      <h2>Few-Shot Examples</h2>
      <p>
        <code>TemplateManager</code> can attach few-shot examples to any template and inject
        them automatically when rendering. Examples are <code>FewShotExample</code> dataclasses
        with <code>input</code>, <code>output</code>, optional <code>context</code>,
        <code>tags</code>, and <code>difficulty</code> ("easy" / "medium" / "hard").
      </p>

      <CodeBlock
        code={`from effgen.prompts import (
    TemplateManager, PromptTemplate, FewShotExample,
)

manager = TemplateManager()
manager.add_template(PromptTemplate(
    name="math_qa",
    template="Question: {{ question }}\\nAnswer:",
))

# Attach examples to the "math_qa" template (TemplateManager.examples is a
# dict[str, list[FewShotExample]] keyed by template name)
manager.examples["math_qa"] = [
    FewShotExample(input="What is 2 + 2?", output="4", tags=["arithmetic"]),
    FewShotExample(input="Calculate 15% of 80", output="12", tags=["percentage"]),
    FewShotExample(input="Convert 100 km to miles", output="62.14", tags=["conversion"]),
]

# Render with few-shot examples (selection by tag/diversity)
prompt = manager.render_template(
    name="math_qa",
    variables={"question": "What is 25% of 200?"},
    include_examples=True,
    num_examples=2,
    example_style="default",          # "default" | "chat" | "instruct"
    example_filter={"tags": ["arithmetic", "percentage"]},
)
print(prompt)`}
        language="python"
        filename="few_shot.py"
      />

      <h3>Selecting Examples</h3>

      <CodeBlock
        code={`# Select examples manually for any template
examples = manager.select_examples(
    template_name="math_qa",
    num_examples=3,
    filter_criteria={"tags": ["arithmetic"]},
    diversity_mode="random",          # "random" | "diverse" | "sequential"
)
for ex in examples:
    print(ex.format(style="default"))`}
        language="python"
        filename="select_examples.py"
      />

      <h2>SLM-Specific Tips</h2>

      <InfoBox type="info" title="Best Practices for SLMs">
        <ul>
          <li><strong>Be concise:</strong> Shorter prompts perform better with limited context</li>
          <li><strong>Use structure:</strong> Numbered lists and clear sections help comprehension</li>
          <li><strong>Explicit format:</strong> Tell the model exactly how to format the output</li>
          <li><strong>Step-by-step:</strong> Break complex tasks into sequential steps</li>
          <li><strong>Few examples:</strong> 2-3 examples often work better than many</li>
          <li><strong>Clear constraints:</strong> State what NOT to do as well as what to do</li>
        </ul>
      </InfoBox>

      <h2>System Prompt Templates</h2>

      <CodeBlock
        code={`# Effective system prompt for SLMs
system_prompt = """You are a helpful assistant with the following capabilities:
- Calculator: mathematical operations
- Code Executor: run Python code

Rules:
1. Use tools when needed, not for simple questions
2. Show your reasoning before acting
3. Be concise but complete
4. If unsure, ask for clarification

Output format: Think briefly, then act or respond."""

# Use with agent
config = AgentConfig(
    name="optimized_agent",
    model=model,
    system_prompt=system_prompt,
    tools=[Calculator(), CodeExecutor()]
)`}
        language="python"
        filename="system_prompt.py"
      />

      <InfoBox type="success" title="Next Steps">
        <p>
          Explore <Link to="/multi-agent">Multi-Agent Systems</Link> for complex orchestration,
          or check out <Link to="/examples">Examples</Link> for real-world prompt patterns.
        </p>
      </InfoBox>
    </DocPage>
  );
}
