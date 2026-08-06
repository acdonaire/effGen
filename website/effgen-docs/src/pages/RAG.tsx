import React from 'react';
import { Link } from 'react-router-dom';
import { BookOpen } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';
import MermaidDiagram from '../components/MermaidDiagram';

export default function RAG() {
  const pipeline = `
flowchart LR
    D[Documents] --> I[DocumentIngester]
    I --> C[Chunkers]
    C --> S[HybridSearchEngine]
    S --> R[Reranker]
    R --> B[ContextBuilder]
    B --> A[Agent]
    A --> O[Answer + Citations]
`;

  return (
    <DocPage
      title="RAG Pipeline"
      subtitle="Hybrid dense + BM25 + keyword search with RRF fusion, semantic/code/table/hierarchical chunkers, rerankers, and inline [N] citations."
      icon={<BookOpen size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Advanced', path: '/multi-agent' },
        { label: 'RAG' },
      ]}
    >
      <h2>Overview</h2>
      <p>
        <code>effgen.rag</code> is a production-grade retrieval pipeline built on top of the
        <code> Retrieval</code> tool. Core formats (txt, md, json, jsonl, csv, html) work with no
        external dependencies; pdf/docx/epub and sentence-transformers embeddings are opt-in.
      </p>

      <MermaidDiagram chart={pipeline} title="RAG Pipeline" />

      <InfoBox type="success" title="New in v0.3.1 — connected memory, honest PDFs, traceable answers">
        <p>
          A RAG agent now accepts a pre-built <code>VectorMemoryStore</code> as its{' '}
          <code>knowledge_base</code> (single, or mixed in a list with paths/text), folding its
          stored entries into the retrieval index — so a vector knowledge base built with the
          memory tools connects straight to retrieval-augmented answering (an empty store still
          fails loudly). <strong>PDFs ingest out of the box</strong>: ingestion falls back to{' '}
          <code>pypdf</code> / <code>pdfplumber</code> when <code>pymupdf</code> is absent, and a
          skipped file&apos;s error now names <em>why</em> each was skipped instead of a bare
          &quot;0 documents to index&quot;. <code>response.sources</code> and{' '}
          <code>response.citations</code> are populated from the retrieved chunks.
        </p>
      </InfoBox>
      <CodeBlock
        code={`from effgen import create_agent
from effgen.memory import VectorMemoryStore

# Build (or persist + reopen) a vector knowledge base, then hand it to RAG
store = VectorMemoryStore(persist_directory="./kb")
agent = create_agent("rag", "openai:gpt-5-nano", knowledge_base=store)
print(agent.run("What is our breach-notification window?").text)`}
      />

      <h2>Quickest Path — the <code>rag</code> Preset</h2>
      <CodeBlock
        code={`from effgen import load_model
from effgen.presets import create_agent

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = create_agent("rag", model, knowledge_base="./docs/")

answer = agent.run("How do I configure guardrails?")
print(answer.output)
for c in answer.citations:      # list[Citation]
    print(f"  [{c.index}] {c.source}  (score={c.relevance_score:.3f})")
print(answer.sources)           # de-duplicated source list`}
        language="python"
        filename="rag_preset.py"
      />

      <InfoBox type="info" title="Why a preset?">
        <p>
          The <code>rag</code> preset auto-ingests the knowledge base on construction, populates
          the <code>Retrieval</code> tool, and sets a system prompt that instructs the model to
          cite inline with <code>[N]</code> markers and say "I don't know" when the KB has no
          match — rather than hallucinate.
        </p>
      </InfoBox>

      <h2>Document Ingestion</h2>
      <p>
        <code>DocumentIngester</code> walks a directory (or takes a list of paths), loads each
        file, normalises text, and emits <code>IngestedChunk</code> objects with a SHA-256
        content hash for deduplication.
      </p>

      <CodeBlock
        code={`from effgen.rag import DocumentIngester

ingester = DocumentIngester(show_progress=True)
chunks = ingester.ingest("./docs/")
print(f"Ingested {len(chunks)} chunks")
for c in chunks[:3]:
    print(c.id, c.source, c.metadata)`}
        language="python"
        filename="ingest.py"
      />

      <ApiTable
        headers={['Format', 'Built-in', 'Requires']}
        rows={[
          ['txt / md', '✓', 'stdlib'],
          ['json / jsonl', '✓', 'stdlib'],
          ['csv', '✓', 'stdlib'],
          ['html', '✓', 'stdlib html.parser'],
          ['pdf', 'optional', 'pymupdf'],
          ['docx', 'optional', 'python-docx'],
          ['epub', 'optional', 'ebooklib + bs4'],
        ]}
      />

      <h2>Chunking Strategies</h2>
      <FeatureList
        features={[
          { icon: '🧠', title: 'SemanticChunker', description: 'Splits at natural sentence / paragraph boundaries within a target size.' },
          { icon: '💻', title: 'CodeChunker', description: 'Respects function / class boundaries for py, js, ts, go, rust, java.' },
          { icon: '📊', title: 'TableChunker', description: 'Keeps markdown / CSV tables intact rather than splitting rows.' },
          { icon: '🌳', title: 'HierarchicalChunker', description: 'Preserves heading hierarchy (H1 → H2 → H3) as chunk context.' },
        ]}
      />

      <CodeBlock
        code={`from effgen.rag import SemanticChunker, CodeChunker

semantic = SemanticChunker(max_chunk_size=1000, similarity_threshold=0.6)
code = CodeChunker(language="python", max_chunk_size=1500)

# Each chunker exposes .chunk(text, doc_id) -> list[Document]
sem_chunks = semantic.chunk("long document text here ...", doc_id="doc1")
code_chunks = code.chunk(open("agent.py").read(), doc_id="agent_py")`}
        language="python"
        filename="chunking.py"
      />

      <h2>Hybrid Search</h2>
      <p>
        <code>HybridSearchEngine</code> combines multiple retrieval signals and fuses them via
        <strong> Reciprocal Rank Fusion (RRF)</strong>:
      </p>
      <ul>
        <li><strong>Dense</strong> — sentence-transformers embeddings + cosine similarity</li>
        <li><strong>Sparse</strong> — BM25 term-frequency scoring</li>
        <li><strong>Keyword</strong> — exact substring / phrase match</li>
        <li><strong>Metadata filter</strong> — restrict by source, type, date, etc.</li>
      </ul>

      <CodeBlock
        code={`from effgen.rag import HybridSearchEngine

engine = HybridSearchEngine(
    weights={"dense": 1.0, "sparse": 1.0, "keyword": 0.5},  # set any to 0 to disable
    rrf_k=60,
)
engine.index(chunks)

results = engine.search(
    "how to wire guardrails",
    top_k=10,
    filter_metadata={"type": "markdown"},
)
for r in results:
    print(r.relevance_score, r.source, r.chunk_id)`}
        language="python"
        filename="hybrid_search.py"
      />

      <h2>Rerankers</h2>
      <FeatureList
        features={[
          { icon: '🎯', title: 'CrossEncoderReranker', description: 'Highest quality — runs a cross-encoder over (query, chunk) pairs. Optional dep: sentence-transformers.' },
          { icon: '🤖', title: 'LLMReranker', description: 'Uses the agent\'s own LLM to score relevance. Free if you already have a model loaded.' },
          { icon: '⚙️', title: 'RuleBasedReranker', description: 'Deterministic: recency, authority (source weights), keyword boosts, title match.' },
        ]}
      />

      <CodeBlock
        code={`from effgen.rag import LLMReranker, RuleBasedReranker

# LLM rerank (free — reuses your agent's model)
llm_rr = LLMReranker(model=agent.config.model)
reranked = llm_rr.rerank(query, results, top_k=5)

# Rule-based (pure Python, zero deps)
rules_rr = RuleBasedReranker(
    recency_weight=0.2,
    keyword_weight=0.15,
    authority_weight=0.25,
    title_weight=0.1,
    authority_map={"docs/internal/": 1.5, "docs/archive/": 0.5},
)
reranked = rules_rr.rerank(query, results, top_k=5)`}
        language="python"
        filename="rerank.py"
      />

      <h2>Context Builder &amp; Citations</h2>
      <p>
        <code>ContextBuilder</code> assembles the final LLM context window: it fits sources into
        a token budget, deduplicates near-identical chunks, orders by relevance or chronology,
        and inserts <code>[N]</code> citation markers wired to the returned
        <code> Citation</code> list.
      </p>

      <CodeBlock
        code={`from effgen.rag import ContextBuilder

builder = ContextBuilder(
    max_tokens=3000,
    per_source_limit=1,        # max chunks per source (0 = unlimited)
    order="relevance",         # or "chronological"
    include_citations=True,    # inject [N] markers
)

built = builder.build(reranked)        # BuiltContext
context_text = built.text              # prompt-ready text
citations    = built.citations         # list[Citation] for AgentResponse.citations`}
        language="python"
        filename="context_builder.py"
      />

      <h2>Citation Tracking</h2>
      <CodeBlock
        code={`from effgen.rag import CitationTracker

tracker = CitationTracker(citations=citations)
# Or build incrementally: tracker = CitationTracker(); tracker.add(c1); tracker.add(c2)

# Parse [N] markers actually referenced in the answer
indices = tracker.extract_used_indices(agent_response.output)

# Filter to only the citations referenced in the answer
used = tracker.filter_used(agent_response.output)

# Heuristic claim verification — citations whose quote token-overlaps the claim
supporting = tracker.verify("Guardrails block injection by default", min_overlap=0.3)

# De-duplicated source list
print(tracker.sources())`}
        language="python"
        filename="citations.py"
      />

      <InfoBox type="success" title="Where citations surface">
        <p>
          When a RAG-enabled agent runs, its <code>AgentResponse.citations</code> is a list of
          <code> Citation</code> dataclasses ({'{'}index, source, chunk_id, relevance_score, quote, page, section{'}'})
          and <code> AgentResponse.sources</code> is a de-duplicated string list of sources backing
          the answer. Both are serialised by <code>.to_dict()</code>.
        </p>
      </InfoBox>

      <h2>Building a RAG Pipeline From Scratch</h2>
      <CodeBlock
        code={`from effgen import Agent, AgentConfig, load_model
from effgen.rag import (
    DocumentIngester, SemanticChunker, HybridSearchEngine,
    LLMReranker, ContextBuilder,
)
from effgen.tools.builtin import Retrieval

# 1. Ingest + chunk
chunks = DocumentIngester().ingest("./knowledge_base/")

# 2. Index
engine = HybridSearchEngine(
    weights={"dense": 1.0, "sparse": 1.0, "keyword": 0.5},
)
engine.index(chunks)

# 3. Wrap as a tool
retrieval = Retrieval()
retrieval.add_documents(
    [{"id": c.id, "content": c.content, "metadata": c.metadata} for c in chunks],
    chunk=False,
)

# 4. Build the agent
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = Agent(AgentConfig(
    name="custom_rag",
    model=model,
    tools=[retrieval],
    system_prompt="Answer ONLY from the retrieved context. Cite with [N] markers.",
))

answer = agent.run("Explain the guardrail presets")`}
        language="python"
        filename="custom_rag.py"
      />

      <h2>See Also</h2>
      <p>
        <Link to="/tools">Tools</Link> · <Link to="/memory">Memory</Link> ·
        {' '}<Link to="/guardrails">Guardrails</Link>
      </p>
    </DocPage>
  );
}
