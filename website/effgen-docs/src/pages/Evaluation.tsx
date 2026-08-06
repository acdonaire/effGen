import React from 'react';
import { Link } from 'react-router-dom';
import { CheckCircle } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Evaluation() {
  return (
    <DocPage
      title="Evaluation &amp; Regression"
      subtitle="Built-in test suites, multiple scoring modes, baseline regression tracking, and multi-model comparison — with CLI and nightly CI hooks."
      icon={<CheckCircle size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Advanced', path: '/multi-agent' },
        { label: 'Evaluation' },
      ]}
    >
      <h2>Overview</h2>
      <p>
        <code>effgen.eval</code> provides <code>AgentEvaluator</code>, five built-in test
        <code> TestSuite</code>s, multiple <code>ScoringMode</code>s, a
        <code> RegressionTracker</code> for baseline diffs, and <code>ModelComparison</code> for
        running a matrix of models × suites.
      </p>

      <h2>Quick Start</h2>
      <CodeBlock
        code={`from effgen import load_model
from effgen.presets import create_agent
from effgen.eval import AgentEvaluator, MathSuite

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = create_agent("math", model)

evaluator = AgentEvaluator(agent)
results = evaluator.run_suite(MathSuite())   # SuiteResults

print(results.accuracy)               # e.g. 0.87
print(results.avg_latency)            # seconds
print(results.total_tokens)
print(results.avg_tool_accuracy)
print([r for r in results.results if not r.passed][:5])`}
        language="python"
        filename="eval_quickstart.py"
      />

      <h2>Built-in Test Suites</h2>
      <ApiTable
        headers={['Suite', 'Cases', 'Focus']}
        rows={[
          [<code>MathSuite</code>, '77', 'Arithmetic, algebra, unit conversion, stats'],
          [<code>ToolUseSuite</code>, '93', 'Correct tool selection and argument extraction'],
          [<code>ReasoningSuite</code>, '40', 'Multi-step reasoning and planning'],
          [<code>SafetySuite</code>, '40', 'Refusals, injection resistance, PII handling'],
          [<code>ConversationSuite</code>, '20', 'Multi-turn conversation and memory'],
        ]}
      />

      <CodeBlock
        code={`from effgen.eval import (
    MathSuite, ToolUseSuite, ReasoningSuite,
    SafetySuite, ConversationSuite, list_suites, get_suite,
)

print(list_suites())                        # names of all suites
math = get_suite("math")                    # lookup by name
print(len(math.test_cases), "cases")`}
        language="python"
        filename="suites.py"
      />

      <h2>Scoring Modes</h2>
      <FeatureList
        features={[
          { icon: '🎯', title: 'EXACT_MATCH', description: 'String equality after whitespace normalisation.' },
          { icon: '🔍', title: 'CONTAINS', description: 'Expected substring appears in the agent output.' },
          { icon: '🧩', title: 'REGEX', description: 'Expected value treated as a regex pattern.' },
          { icon: '🧠', title: 'SEMANTIC_SIMILARITY', description: 'sentence-transformers cosine similarity over a threshold. Optional dep.' },
          { icon: '⚖️', title: 'LLM_JUDGE', description: 'An LLM scores each (expected, actual) pair — reuses your agent\'s model for free.' },
        ]}
      />

      <CodeBlock
        code={`from effgen.eval import AgentEvaluator, TestCase
from effgen.eval.evaluator import ScoringMode

class CustomSuite:
    name = "custom"
    test_cases = [
        TestCase(query="2+2", expected_output="4"),
        TestCase(query="Sum of first 10 primes", expected_output="129"),
    ]

evaluator = AgentEvaluator(
    agent,
    scoring=ScoringMode.LLM_JUDGE,          # or EXACT_MATCH / CONTAINS / REGEX / SEMANTIC_SIMILARITY
    pass_threshold=0.5,
)
results = evaluator.run_suite(CustomSuite())`}
        language="python"
        filename="custom_suite.py"
      />

      <InfoBox type="success" title="v0.3.0 — ergonomic TestCase aliases">
        <p>
          v0.3.0 adds additive aliases so the obvious call works:{' '}
          <code>TestCase(input=, expected=)</code> reads naturally alongside the existing{' '}
          <code>query=</code> / <code>expected_output=</code> fields, and test suites accept a{' '}
          <code>name=</code> per test. Existing code is unchanged — these are aliases, not
          replacements.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.1 — quick bake-offs on your own data">
        <p>
          <code>effgen compare</code> gained <code>--max-cases</code> and <code>--difficulty</code>,
          and both <code>compare</code> and <code>eval</code> accept a path to your own{' '}
          <code>.jsonl</code> / <code>.json</code> test cases — so a quick bake-off on a subsample
          or your own dataset is a one-liner. <code>eval --suite list</code> now shows each
          suite&apos;s real case count, and on an accuracy tie <code>compare</code> recommends the
          lower-latency (then fewer-token) model instead of whichever was listed first. Add{' '}
          <code>--json</code> to <code>eval</code> / <code>compare</code> / <code>workflow</code> /{' '}
          <code>sessions list</code> for machine-readable CI gating.
        </p>
      </InfoBox>

      <h2>Regression Tracking</h2>
      <p>
        <code>RegressionTracker</code> saves a run as a baseline and then compares future runs
        against it, tagging each diff with severity (<code>warning</code>, <code>high</code>,
        <code> critical</code>).
      </p>
      <ApiTable
        headers={['Metric', 'Trigger threshold (default)', 'Severity scaling']}
        rows={[
          ['accuracy drop', '> 5%', 'warning, then high above 2× threshold, critical above 3×'],
          ['avg_latency increase', '> 20%', 'warning, then high above 2× threshold, critical above 3×'],
          ['avg_tool_accuracy drop', '> 5%', 'warning, then high above 2× threshold, critical above 3×'],
        ]}
      />

      <CodeBlock
        code={`from effgen.eval import RegressionTracker

tracker = RegressionTracker(baselines_dir="./baselines")

# First run — save as baseline (one file per suite name)
tracker.save_baseline("math", results, version="0.2.0")

# Later — compare current vs. stored baseline
report = tracker.compare("math", new_results, version="0.2.1")
if report.has_regressions:
    for alert in report.alerts:
        print(alert.metric, alert.baseline_value, alert.current_value)`}
        language="python"
        filename="regression.py"
      />

      <h2>Model Comparison</h2>
      <CodeBlock
        code={`from effgen.eval import ModelComparison, MathSuite, ReasoningSuite
from effgen.eval.evaluator import ScoringMode

cmp = ModelComparison(scoring=ScoringMode.CONTAINS, pass_threshold=0.5)

matrix = cmp.run(
    agents={
        "qwen-3b":  agent_qwen,            # Agent instances, one per model under test
        "phi-3.5":  agent_phi,
    },
    suites=[MathSuite(), ReasoningSuite()],
)

print(matrix.to_markdown())                # human-readable matrix
print(matrix.recommendations)              # {suite_name: best_model_name}
import json; open("comparison.json","w").write(matrix.to_json())`}
        language="python"
        filename="model_comparison.py"
      />

      <h2>CLI</h2>
      <CodeBlock
        code={`# Single model, single suite
effgen eval --suite math --model Qwen/Qwen2.5-3B-Instruct

# Matrix of models × suites
effgen compare --models "Qwen/Qwen2.5-3B-Instruct,microsoft/Phi-3.5-mini-instruct" \\
                --suite reasoning

# Save current results as a baseline (under ./baselines/ by default)
effgen eval --suite math --model <m> --save-baseline

# Compare against the stored baseline (prints regression alerts; non-zero exit on regression)
effgen eval --suite math --model <m> --compare-baseline`}
        language="bash"
        filename="terminal"
      />

      <InfoBox type="info" title="Nightly CI">
        <p>
          effGen's own CI runs an <code>eval-regression</code> job nightly that compares against
          stored baselines and opens a GitHub issue on failure. Wire the same pattern into your
          CI with <code>effgen eval --compare-baseline</code>.
        </p>
      </InfoBox>

      <h2>See Also</h2>
      <p>
        <Link to="/guardrails">Guardrails</Link> · <Link to="/debug">Debugging</Link> ·
        {' '}<Link to="/models">Models</Link>
      </p>
    </DocPage>
  );
}
