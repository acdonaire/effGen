import React from 'react';
import { Link } from 'react-router-dom';
import { Layers } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Domains() {
  return (
    <DocPage
      title="Domains &amp; Keyword Expansion"
      subtitle="Domain-specific configurations bundling keywords, prompts, tools, and guardrails — with a KeywordExpander that grows seed keywords via WordNet, templates, and optional LLM expansion."
      icon={<Layers size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Advanced', path: '/multi-agent' },
        { label: 'Domains' },
      ]}
    >
      <h2>Overview</h2>
      <p>
        A <code>Domain</code> bundles everything an agent needs to operate in a specialised
        area — seed keywords, system prompt additions, recommended tools, and guardrails — and
        exposes a <code>KeywordExpander</code> that grows N seed keywords to 10N+ related terms.
      </p>

      <InfoBox type="success" title="New in v0.3.1 — a domain becomes a runnable agent in one call">
        <p>
          Every <code>Domain</code> now has an obvious on-ramp to a working agent.{' '}
          <code>Domain.to_agent(model, **overrides)</code> (and a <code>domain=</code> kwarg on{' '}
          <code>create_agent</code>) wires the domain&apos;s system prompt, registry-resolved
          recommended tools, and guardrails preset into the agent — giving the bundled{' '}
          <code>Domain.guardrails</code> its first real consumer. The non-tech domains (legal,
          finance, health, science) also expand seed keywords into field-appropriate query
          variants (<code>{'"{kw} clause"'}</code>, <code>{'"{kw} obligations"'}</code>,{' '}
          <code>{'"{kw} regulation"'}</code>) instead of borrowing the tech how-to templates.
        </p>
      </InfoBox>
      <CodeBlock
        code={`from effgen import LegalDomain, create_agent

# One call: system prompt + recommended tools + guardrails, wired together
agent = LegalDomain().to_agent("openai:gpt-5-nano")
print(agent.run("What should an NDA's confidentiality clause cover?").text)

# Equivalent via the factory
agent = create_agent(domain=LegalDomain(), model="groq:llama-3.1-8b-instant")

# Field-appropriate keyword expansion (non-tech domains)
terms = LegalDomain().expand_keywords(use_llm=True, model="openai:gpt-5-nano")`}
      />

      <h2>Built-in Domains</h2>
      <FeatureList
        features={[
          { icon: '💻', title: 'TechDomain', description: 'Software engineering, infrastructure, DevOps, ML.' },
          { icon: '🔬', title: 'ScienceDomain', description: 'Physics, chemistry, biology, research methods.' },
          { icon: '💰', title: 'FinanceDomain', description: 'Markets, accounting, investment. Ships with "not financial advice" guardrail copy.' },
          { icon: '🏥', title: 'HealthDomain', description: 'General health info. Ships with "not medical advice" disclaimer.' },
          { icon: '⚖️', title: 'LegalDomain', description: 'Legal concepts. Ships with "not legal advice" disclaimer.' },
        ]}
      />

      <h2>Usage</h2>
      <CodeBlock
        code={`from effgen.domains import TechDomain, KeywordExpander

domain = TechDomain(keywords=["Python", "machine learning"])

# Expand keywords 10×
expanded = domain.expand_keywords(factor=10)
print(expanded)
# → ['Python', 'machine learning', 'python3', 'ML', 'deep learning',
#    'neural networks', 'scikit-learn', 'pytorch', 'tensorflow', ...]

# Domain hints a good agent configuration
print(domain.system_prompt)       # tailored system prompt
print(domain.tool_names)          # ['web_search', 'code_executor', 'python_repl', ...]
print(domain.guardrails)          # optional guardrail preset name or chain`}
        language="python"
        filename="domain_usage.py"
      />

      <h3>Standalone KeywordExpander</h3>
      <CodeBlock
        code={`from effgen.domains import KeywordExpander

expander = KeywordExpander(
    use_wordnet=True,         # optional NLTK WordNet synonym expansion
    use_templates=True,       # fixed template patterns (e.g. "X tutorial", "X vs Y")
    use_llm=False,            # optional — uses the supplied model
)

expanded = expander.expand(["agent"], factor=10)`}
        language="python"
        filename="keyword_expander.py"
      />

      <InfoBox type="info" title="Why expand keywords?">
        <p>
          Keyword expansion is useful anywhere you build a corpus or shape a search: seed-based
          RAG ingestion, query rewriting for <Link to="/rag">HybridSearchEngine</Link>, or
          data-collection pipelines that start from a small list of target terms.
        </p>
      </InfoBox>

      <h2>Writing Your Own Domain</h2>
      <CodeBlock
        code={`from effgen.domains import Domain

# Domain is a dataclass — just instantiate it with the fields you want
domain = Domain(
    name="robots",
    keywords=["ROS2", "SLAM", "motion planning"],
    description="Robotics, ROS2, and SLAM",
    system_prompt=(
        "You are an expert in robotics, ROS2, and SLAM. "
        "Always cite sources when claiming performance numbers."
    ),
    tool_names=["web_search", "arxiv", "github"],
)
expanded = domain.expand_keywords(factor=5)`}
        language="python"
        filename="custom_domain.py"
      />

      <h2>See Also</h2>
      <p>
        <Link to="/rag">RAG</Link> · <Link to="/tools">Tools</Link> ·
        {' '}<Link to="/agents">Agents</Link>
      </p>
    </DocPage>
  );
}
