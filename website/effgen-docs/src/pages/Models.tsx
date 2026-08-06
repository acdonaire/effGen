import React from 'react';
import { Link } from 'react-router-dom';
import { Brain } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Models() {
  return (
    <DocPage
      title="Models"
      subtitle="effGen supports 14 inference backends plus an opt-in policy-based ModelRouter (added in v0.2.4) over the 9 cloud providers. provider= cloud routing supports OpenAI/Anthropic/Gemini auto-detection plus explicit Cerebras/Groq/Together/Fireworks/Replicate/HF Inference routing. v0.2.8 adds multimodal image / audio / video input across the vision-capable providers plus a local MLX-VLM adapter; v0.2.9 wires per-provider circuit breakers and bulkheads into the registry. v0.3.0 adds a self-updating, drift-aware model catalog (effgen models refresh), real GPU support, and greedy temperature=0 decoding across local backends."
      icon={<Brain size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Core Concepts', path: '/agents' },
        { label: 'Models' },
      ]}
    >
      <h2>Model Loading</h2>
      <p>
        The <code>load_model</code> function provides a unified interface for loading models from various backends:
      </p>

      <CodeBlock
        code={`from effgen import load_model

# Basic loading with Transformers
model = load_model("Qwen/Qwen2.5-7B-Instruct")

# With specific engine
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="transformers"  # or "vllm" | "mlx" | "mlx_vlm"; GGUF/API are auto-detected
)

# With quantization (4-bit or 8-bit)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="transformers",
    quantization="4bit"
)

# With multi-GPU support (using Transformers)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="transformers",
    device_map="auto"  # Automatic multi-GPU
)`}
        language="python"
        filename="load_models.py"
      />

      <InfoBox type="info" title="v0.2.1 provider loading">
        <p>
          In v0.2.1, cloud loading supports explicit <code>provider=</code>.
          OpenAI, Anthropic, and Gemini model names are also auto-detected;
          Cerebras should use explicit <code>provider="cerebras"</code>.
          Provider-prefixed IDs such as <code>groq:...</code> and the
          ProviderRegistry are v0.2.3+.
        </p>
      </InfoBox>

      <h2>ProviderRegistry (v0.2.3+)</h2>
      <p>
        Cloud providers register themselves with <code>ProviderRegistry</code>. Use
        provider-prefixed model IDs to make routing explicit when the same model name is
        available from multiple providers.
      </p>
      <CodeBlock
        code={`from effgen import load_model
from effgen.models.registry import list_providers, list_models, lookup
from effgen.models.auth import check_keys

print(list_providers())
print(list_models("groq")[:3])

model = load_model("groq:llama-3.3-70b-versatile")
provider, adapter_cls, info = lookup("groq:llama-3.3-70b-versatile")
keys = check_keys()`}
        language="python"
        filename="provider_registry.py"
      />
      <CodeBlock
        code={`effgen doctor
effgen doctor --provider openai
effgen doctor --json
effgen doctor --live --cheap   # v0.3.0: a real, low-cost call per provider — "key present" vs "key usable"`}
        language="bash"
        filename="terminal"
      />

      <h2>Self-Updating Model Catalog (v0.3.0)</h2>
      <p>
        Every provider ships a local <strong>catalog snapshot</strong> — for each model: id, context
        window, max output, input/output price per 1M tokens, tool/vision/audio support, a free-tier
        flag, and rate limits — along with a model <strong>count</strong> and a{' '}
        <strong>"verified on" date</strong>. The catalog backs <code>effgen models list / info</code>,
        and the cost tracker reads pricing from it (no silent <code>$0</code>); token accounting
        prefers real provider usage over heuristics.
      </p>
      <CodeBlock
        code={`# Pull the live model list, update the snapshot, and report what changed
effgen models refresh                  # all configured providers
effgen models refresh --provider cerebras

# Catalog-backed views (cached / configured / live) with status, price, context,
# deprecation, and "verified on" date
effgen models list --json
effgen models info groq:llama-3.3-70b-versatile`}
        language="bash"
        filename="terminal"
      />
      <p>
        Refresh filters to chat / text-generation base models and never persists private{' '}
        <code>ft:</code> fine-tune ids, embeddings, audio, or image models. effGen calls{' '}
        <code>check_drift()</code> and shows a single, non-spammy warning when its catalog looks
        stale relative to what the live APIs actually serve. A wrong or 404 model id no longer
        crashes — it suggests the nearest live alternative
        (<code>Unknown Cerebras model 'llama3.1-8b'. Did you mean: gpt-oss-120b, zai-glm-4.7?</code>).
      </p>
      <InfoBox type="info" title="Pricing completeness (v0.3.0)">
        <p>
          Every catalog record is priced or explicitly flagged free / unpriced. The cost tracker
          reads pricing straight from the catalog, so a missing price surfaces as a warning rather
          than a silently-wrong <code>$0</code>.
        </p>
      </InfoBox>

      <InfoBox type="success" title="New in v0.3.1 — local truth & grammar-constrained output">
        <p>
          <code>effgen models status</code> now reports <strong>physical GPU memory</strong> (the
          driver&apos;s view across all processes, via <code>mem_get_info</code>) plus a utilization
          column — so it shows which card is actually free, not the calling process&apos;s
          reservations. <code>effgen models info</code> is <strong>cache-aware</strong>: a model in
          the local HuggingFace cache is described as locally-runnable (engines, on-disk size,
          context window) instead of &quot;not found&quot; or a cloud-only route, and incomplete
          downloads are flagged. For structured output, the optional <code>effgen[grammar]</code>{' '}
          extra (<code>outlines</code>) lets small local models that won&apos;t follow a JSON schema
          by prompting alone emit schema-valid output in one constrained pass; without it, the
          honest <code>success=False</code> ends with a one-line install hint.
        </p>
      </InfoBox>

      <h2>Real GPU Support &amp; Greedy Decoding (v0.3.0)</h2>
      <p>
        The documented GPU install now selects a <strong>driver-compatible torch</strong>, and a
        runtime guard warns once when NVML sees GPUs but <code>torch.cuda</code> cannot use them
        (CPU fallback). CPU / CUDA-12.4 / CUDA-13 / vLLM install matrices are documented, and a{' '}
        <code>constraints-cu1xx.txt</code> flow prevents an extras re-install from replacing a
        working torch. The GPU allocator no longer deadlocks on <code>reset()</code>, reads real
        free memory via NVML / <code>torch.cuda.mem_get_info</code> (respecting other processes),
        and no longer mutates the global CUDA device during discovery.
      </p>
      <CodeBlock
        code={`from effgen import load_model
from effgen.models.base import GenerationConfig

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# v0.3.0: temperature <= 0 means greedy decoding (do_sample=False) across local
# backends — it no longer raises "temperature must be strictly positive".
result = model.generate("Capital of France?", config=GenerationConfig(temperature=0))
print(result.text)`}
        language="python"
        filename="v030_gpu_greedy.py"
      />
      <p>
        Local tool answers also no longer leak chat-template special tokens (e.g. Gemma{' '}
        <code>&lt;end_of_turn&gt;</code> / <code>&lt;eos&gt;</code>, Qwen{' '}
        <code>&lt;|im_start|&gt;</code>); the strip set is derived from the tokenizer while
        preserving tool-call delimiters.
      </p>

      <h2>Supported Backends by Version</h2>

      <ApiTable
        headers={['Backend', 'Provider / engine', 'Install', 'Best for']}
        rows={[
          ['Transformers', <code>engine="transformers"</code>, 'built in', 'Development, quantization, broad HuggingFace support'],
          ['vLLM', <code>engine="vllm"</code>, <code>effgen[vllm]</code>, 'High-throughput CUDA serving'],
          ['MLX', <code>engine="mlx"</code>, <code>effgen[mlx]</code>, 'Apple Silicon text generation'],
          ['MLX-VLM', <code>engine="mlx_vlm"</code>, <code>effgen[mlx-vlm]</code>, 'Apple Silicon vision-language models'],
          ['GGUF', '.gguf path', <code>effgen[gguf]</code>, 'CPU/Metal quantized models through llama-cpp'],
          ['OpenAI', <code>provider="openai"</code>, 'built in', 'gpt-5/gpt-5.4-nano chat models, o-series reasoning controls, native tools'],
          ['Anthropic', <code>provider="anthropic"</code>, 'built in', 'v0.2.1 adapter; v0.2.2+ Claude 4.x thinking, prompt caching, streaming'],
          ['Gemini', <code>provider="gemini"</code>, 'built in', 'v0.2.1 adapter; v0.2.2+ Gemini thinking, grounding, Files API, native tools'],
          ['Cerebras', <code>provider="cerebras"</code>, <code>effgen[cerebras]</code>, '4 registered models; 2 reliably free-tier callable; streaming and model-dependent native tools'],
          ['Groq (v0.2.3+)', <code>provider="groq"</code>, <code>effgen[groq]</code>, 'Low-latency free-tier inference'],
          ['Together AI (v0.2.3+)', <code>provider="together"</code>, <code>effgen[together]</code>, 'Large model catalog and live refresh'],
          ['Fireworks (v0.2.3+)', <code>provider="fireworks"</code>, <code>effgen[fireworks]</code>, 'OpenAI-compatible hosted models; 54 tool-capable'],
          ['Replicate (v0.2.3+)', <code>provider="replicate"</code>, <code>effgen[replicate]</code>, 'Async run/poll predictions; 34 streaming-capable models'],
          ['HuggingFace Inference (v0.2.3+)', <code>provider="hf"</code>, <code>effgen[hf]</code>, 'HF Inference Router and custom endpoint URLs'],
        ]}
      />

      <h3>vLLM (Recommended for Production)</h3>
      <p>
        vLLM provides the fastest inference through continuous batching and PagedAttention:
      </p>

      <CodeBlock
        code={`from effgen import load_model

model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="vllm",
    tensor_parallel_size=2,    # Multi-GPU
    gpu_memory_utilization=0.9, # GPU memory fraction
    max_model_len=8192          # Max context length
)`}
        language="python"
        filename="vllm_backend.py"
      />

      <FeatureList
        features={[
          { icon: '⚡', title: '5-10x Faster', description: 'Compared to standard Transformers inference' },
          { icon: '🔀', title: 'Multi-GPU', description: 'Automatic tensor parallelism across GPUs' },
          { icon: '📊', title: 'PagedAttention', description: 'Efficient memory management for long contexts' },
          { icon: '🔄', title: 'Continuous Batching', description: 'High throughput for concurrent requests' },
        ]}
      />

      <h3>Transformers (Development)</h3>
      <p>
        HuggingFace Transformers is great for development and supports quantization:
      </p>

      <CodeBlock
        code={`from effgen import load_model

# 4-bit quantization (lowest memory)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="transformers",
    quantization="4bit",
    device_map="auto"
)

# 8-bit quantization (better quality)
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="transformers",
    quantization="8bit"
)

# Full precision (requires more VRAM)
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    engine="transformers",
    torch_dtype="float16"
)`}
        language="python"
        filename="transformers_backend.py"
      />

      <h3>MLX (Apple Silicon)</h3>
      <p>
        On <code>darwin/arm64</code> (M1/M2/M3/M4), <code>MLXEngine</code> runs text models natively.
        <code>MLXVLMEngine</code> handles vision-language models (LLaVA, Qwen-VL, Phi-Vision, etc. —
        30+ architectures). Install via <code>pip install effgen[mlx]</code> or <code>[mlx-vlm]</code>.
      </p>
      <CodeBlock
        code={`from effgen import load_model
from effgen.hardware import is_apple_silicon, get_best_local_backend

if is_apple_silicon():
    # Text
    model = load_model("mlx-community/Qwen2.5-3B-Instruct-4bit", engine="mlx")

    # Vision-language
    vlm = load_model("mlx-community/Qwen2-VL-2B-Instruct-4bit", engine="mlx_vlm")

# Ask the framework which backend is best for this machine:
print(get_best_local_backend())   # "mlx" | "vllm" | "transformers"`}
        language="python"
        filename="mlx_backend.py"
      />

      <h3>GGUF (llama-cpp)</h3>
      <p>
        Any <code>.gguf</code> path is auto-routed to <code>GGUFEngine</code> (backed by
        <code> llama-cpp-python</code>) — a good fit for quantized models on CPU or Metal.
      </p>
      <CodeBlock
        code={`model = load_model("./models/qwen2.5-3b-instruct-q4_k_m.gguf")`}
        language="python"
        filename="gguf_backend.py"
      />

      <h2>Local Model Router (v0.2.0)</h2>
      <p>
        effGen can route between multiple local models based on task complexity and per-model capability
        scores. <code>estimate_complexity()</code> runs a sub-millisecond keyword analysis
        (code / math / reasoning / multilingual), and <code>MODEL_CAPABILITIES</code> holds
        pre-populated profiles for 12+ SLMs (Qwen 0.5B-7B, Llama 1B-3B, Phi-3/3.5/4, Mistral 7B,
        Gemma 2B/9B).
      </p>
      <CodeBlock
        code={`from effgen import load_model, Agent, AgentConfig

fast   = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")
strong = load_model("Qwen/Qwen2.5-7B-Instruct",   quantization="4bit")

config = AgentConfig(
    name="routed",
    model=fast,                 # primary
    models=[strong],            # router can pick 'strong' for hard queries
    speculative_execution=False, # flip to True for "first success wins"
)`}
        language="python"
        filename="model_router.py"
      />

      <h2>Policy-Based Cloud ModelRouter (v0.2.4+)</h2>
      <p>
        v0.2.4 introduces an opt-in <code>PolicyBasedRouter</code> that sits between
        application code and the 9 cloud providers. Compose
        <code> FirstAvailablePolicy</code>, <code>CostBasedPolicy</code>, and
        <code> LatencyBasedPolicy</code> in any order; every <code>RouterDecision</code>
        lists the chosen <code>(provider, model_id)</code> plus eliminated
        candidates with reasons (<code>no_key</code>, <code>cost_exceeds_budget</code>,
        <code> latency_exceeds_sla</code>, <code>missing capabilities: tools</code>,
        <code> rate_limited</code>, <code>requires dedicated endpoint</code>).
      </p>
      <ApiTable
        headers={['Capability', 'Meaning']}
        rows={[
          [<code>chat</code>, 'Basic text completion / chat'],
          [<code>tools</code>, 'Function / tool calling'],
          [<code>streaming</code>, 'Token streaming via generate_stream()'],
          [<code>vision</code>, 'Image inputs'],
          [<code>grounding</code>, 'Web grounding / Google Search integration'],
          [<code>thinking</code>, 'Extended thinking / chain-of-thought reasoning'],
          [<code>json_schema</code>, 'Structured output with JSON schema constraint'],
        ]}
      />
      <CodeBlock
        code={`import effgen.models  # registers all 9 cloud adapters
from effgen import (
    PolicyBasedRouter, RoutingContext,
    CostBasedPolicy, LatencyBasedPolicy,
    load_model,
)
from effgen.models.capabilities import Capability

router = PolicyBasedRouter(
    policies=[LatencyBasedPolicy(), CostBasedPolicy()],
    failover_hops=3,
)
router.subscribe(lambda ev: print("failover:", ev.as_dict()))

context = RoutingContext(
    prompt_tokens_estimate=500,
    user_budget_usd=0.01,
    latency_budget_ms=3000,
    required_capabilities={Capability.chat, Capability.tools},
)

decision = router.route(context)
print(decision.chosen.provider, decision.chosen.model_id, decision.policy_name)

def call_model(pair):
    return load_model(f"{pair.provider}:{pair.model_id}").generate("Hello")

answer = router.route_and_execute(context, call_model)`}
        language="python"
        filename="v024_policy_router.py"
      />
      <p>
        <code>route_and_execute(context, fn)</code> classifies
        <code> RateLimitExceeded</code>, <code>ProviderTransientError</code>,
        <code> ModelTimeoutError</code>, and <code>BudgetExceededError</code> as
        retriable; it adds the failing provider to a temporary blocklist,
        re-routes, and emits a <code>RouterEvent</code> per hop until
        <code> failover_hops</code> is exhausted (then
        <code> AllCandidatesExhaustedError</code>). Auth, refusal, and invalid-request
        errors re-raise immediately.
      </p>

      <h3>Model Pool &amp; Hot-Swap</h3>
      <CodeBlock
        code={`# CLI
effgen models load Qwen/Qwen2.5-3B-Instruct
effgen models status
effgen models unload Qwen/Qwen2.5-3B-Instruct

# Python
from effgen.models.pool import ModelPool
pool = ModelPool(max_models=3, gpu_memory_limit_gb=40)
pool.load("Qwen/Qwen2.5-3B-Instruct")
pool.load("microsoft/Phi-3.5-mini-instruct")  # LRU eviction if full`}
        language="python"
        filename="model_pool.py"
      />

      <h3>LazyModel &amp; ContinuousBatcher</h3>
      <p>
        Two more optimizations ship in <code>effgen.models</code>: <code>LazyModel</code> defers
        weight loading until the first <code>generate()</code> / <code>count_tokens()</code>
        call and unloads after an idle timeout, and <code>ContinuousBatcher</code> coalesces
        concurrent <code>submit()</code> calls into a single batched inference for backends that
        expose a fast batched path.
      </p>
      <CodeBlock
        code={`from effgen import load_model
from effgen.models.lazy import LazyModel
from effgen.models.batching import ContinuousBatcher

# Lazy loading — weights stay on disk until first use, then evicted after 600s idle
inner  = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
lazy   = LazyModel(inner, idle_timeout=600.0)

# Continuous batching — coalesce concurrent generate() calls, max_wait_ms flushes early
batcher = ContinuousBatcher(inner, max_batch_size=8, max_wait_ms=20.0)
result  = batcher.submit("Hello, world")`}
        language="python"
        filename="lazy_and_batching.py"
      />

      <h2>Quantization — AWQ &amp; GPTQ (v0.2.0)</h2>
      <CodeBlock
        code={`# AWQ
model = load_model("TheBloke/Qwen2.5-7B-Instruct-AWQ", quantization="awq")

# GPTQ
model = load_model("TheBloke/Qwen2.5-7B-Instruct-GPTQ", quantization="gptq")

# Speculative decoding — faster generation via a cheap draft model
from effgen.models.base import GenerationConfig
gen = GenerationConfig(draft_model="Qwen/Qwen2.5-0.5B-Instruct")`}
        language="python"
        filename="advanced_quant.py"
      />

      <h3>OpenAI API</h3>
      <CodeBlock
        code={`import os
from effgen import load_model
from effgen.models.base import GenerationConfig

model = load_model(
    "gpt-5.4-nano",
    provider="openai",
    api_key=os.environ.get("OPENAI_API_KEY")
)

reasoning_model = load_model("o4-mini", provider="openai")

# o-series reasoning models accept reasoning controls.
# v0.2.1 validates membership and passes values through.
result = reasoning_model.generate(
    "Solve the task carefully.",
    config=GenerationConfig(reasoning_effort="medium", max_reasoning_tokens=4096),
)`}
        language="python"
        filename="openai_backend.py"
      />

      <h3>Anthropic API (v0.2.1-compatible)</h3>
      <p>
        v0.2.2 adds Claude 4.x registries, extended thinking, prompt caching,
        and native tools. Use <code>claude-sonnet-4-6</code> or
        <code> claude-haiku-4-5</code> for explicit <code>thinking</code>;
        <code> claude-opus-4-7</code> uses adaptive thinking and does not expose
        an explicit thinking trace. The snippet below stays runnable on v0.2.1.
      </p>
      <CodeBlock
        code={`from effgen import load_model

model = load_model("claude-3-5-sonnet-20241022", provider="anthropic")
result = model.generate("Think through this architecture tradeoff.")
print(result.text)`}
        language="python"
        filename="anthropic_backend.py"
      />

      <h3>Google Gemini API (v0.2.1-compatible)</h3>
      <p>
        v0.2.2 adds Gemini 2.5/3.x registries, thinking budgets, grounding,
        Files API, and native tools. The snippet below stays runnable on v0.2.1.
      </p>
      <CodeBlock
        code={`from effgen import load_model

model = load_model("gemini-2.0-flash-lite", provider="gemini")
result = model.generate("Answer with concise context.")
print(result.text)`}
        language="python"
        filename="google_backend.py"
      />

      <h3>Cerebras and v0.2.3 Providers</h3>
      <CodeBlock
        code={`from effgen import load_model

cerebras = load_model("llama3.1-8b", provider="cerebras")

# v0.2.3+
groq = load_model("groq:llama-3.3-70b-versatile")
together = load_model("meta-llama/Meta-Llama-3-8B-Instruct-Lite", provider="together")
fireworks = load_model("accounts/fireworks/models/qwen3-8b", provider="fireworks")
replicate = load_model("meta/meta-llama-3-8b-instruct", provider="replicate")
hf = load_model("Qwen/Qwen2.5-72B-Instruct", provider="hf")`}
        language="python"
        filename="cloud_providers.py"
      />

      <h2>Recommended Models</h2>

      <ApiTable
        headers={['Model', 'Size', 'VRAM (4-bit)', 'Compatibility']}
        rows={[
          ['Qwen/Qwen2.5-1.5B-Instruct', '1.5B', '~2GB', '10/10 agents pass'],
          ['Qwen/Qwen2.5-3B-Instruct', '3B', '~3GB', '10/10 agents pass (recommended)'],
          ['microsoft/Phi-4-mini-instruct', '3.8B', '~3GB', '10/10 agents pass'],
          ['Qwen/Qwen3-1.7B', '1.7B', '~2GB', '9.5/10 agents pass'],
          ['Qwen/Qwen2.5-7B-Instruct', '7B', '~5GB', '9/10 agents pass'],
          ['meta-llama/Llama-3.2-3B-Instruct', '3B', '~3GB', '8.5/10 agents pass'],
          ['google/gemma-3-4b-it', '4B', '~3GB', '8/10 agents pass'],
          ['HuggingFaceTB/SmolLM2-1.7B-Instruct', '1.7B', '~2GB', 'Lightweight tasks'],
          ['HuggingFaceTB/SmolLM3-3B', '3B', '~3GB', 'General purpose'],
          ['meta-llama/Llama-3.1-8B-Instruct', '8B', '~6GB', 'Complex reasoning'],
        ]}
      />

      <InfoBox type="info" title="Model Selection Tips">
        <ul>
          <li><strong>11 model families tested</strong> with 73% overall pass rate across 10 example agents</li>
          <li><strong>Qwen 2.5 series:</strong> Top performers, specifically trained for tool use and agentic workflows</li>
          <li><strong>3B models:</strong> Good balance of speed and capability for most tasks</li>
          <li><strong>7B models:</strong> Better reasoning, recommended for complex agentic tasks</li>
          <li><strong>14B+ models:</strong> Best quality, but require more resources</li>
        </ul>
      </InfoBox>

      <h2>Generation Configuration</h2>

      <p>
        The example below is v0.2.1-compatible. Gemini and Anthropic thinking
        fields were added in v0.2.2 and are listed in the API reference.
      </p>
      <CodeBlock
        code={`from effgen import GenerationConfig, load_model

model = load_model("gpt-5.4-nano", provider="openai")

config = GenerationConfig(
    max_tokens=512,           # Maximum tokens to generate
    temperature=0.7,          # Randomness (0=deterministic, 1=creative)
    top_p=0.9,               # Nucleus sampling threshold
    top_k=50,                # Top-k sampling
    repetition_penalty=1.1,   # Penalize repetition
    stop_sequences=["Human:", "###"], # Stop tokens

    # v0.2.1 OpenAI reasoning controls.
    # Valid values: none, minimal, low, medium, high, xhigh.
    reasoning_effort="medium",
    max_reasoning_tokens=4096,        # OpenAI reasoning budget
)

# Use with model
result = model.generate(
    prompt="Explain quantum computing",
    config=config
)

print(result.text)
print(f"Tokens generated: {result.tokens_used}")`}
        language="python"
        filename="generation_config.py"
      />

      <h2>Quantization</h2>
      <p>
        Quantization reduces model size and memory usage with minimal quality loss:
      </p>

      <ApiTable
        headers={['Quantization', 'Memory Reduction', 'Quality Impact', 'Use Case']}
        rows={[
          ['None (fp16)', 'Baseline', 'Best', 'Production with ample VRAM'],
          ['8-bit', '~50%', 'Minimal', 'Balanced quality/memory'],
          ['4-bit', '~75%', 'Small', 'Memory-constrained environments'],
        ]}
      />

      <CodeBlock
        code={`# 4-bit quantization with BitsAndBytes
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="transformers",
    quantization="4bit",
    bnb_config={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True
    }
)`}
        language="python"
        filename="quantization.py"
      />

      <h2>Multi-GPU Setup</h2>

      <CodeBlock
        code={`# vLLM with tensor parallelism
model = load_model(
    "Qwen/Qwen2.5-14B-Instruct",
    engine="vllm",
    tensor_parallel_size=2  # Split across 2 GPUs
)

# Transformers with device_map
model = load_model(
    "Qwen/Qwen2.5-14B-Instruct",
    engine="transformers",
    device_map="auto"  # Automatically distribute across available GPUs
)`}
        language="python"
        filename="multi_gpu.py"
      />

      <h2>Model-Specific Prompt Formatting</h2>
      <p>
        effGen includes a <code>ToolPromptGenerator</code> that creates model-specific system prompts
        optimized for each model family's tool-use capabilities:
      </p>

      <CodeBlock
        code={`from effgen.prompts import ToolPromptGenerator

# Automatically generates optimized prompts for Qwen, Llama, Phi, etc.
generator = ToolPromptGenerator(
    tools=[Calculator(), WebSearch()],
    model_name="qwen"
)

# Generate system prompt with tool descriptions
system_prompt = generator.generate_tool_prompt(
    tools=[Calculator(), WebSearch()]
)

# Model families with specific optimizations:
# - "qwen"   → Optimized for Qwen2.5/Qwen3 tool format
# - "llama"  → Optimized for Llama 3.x function calling
# - "phi"    → Optimized for Phi-4 instruction format
# - "smollm" → Optimized for SmolLM compact format
# - "gemma"  → Optimized for Gemma instruction format
# - "auto"   → Auto-detect from model name`}
        language="python"
        filename="tool_prompt_generator.py"
      />

      <InfoBox type="success" title="Next Steps">
        <p>
          Learn about <Link to="/tools">Tools</Link> to extend your model's capabilities,
          or explore <Link to="/memory">Memory Systems</Link> for context management.
        </p>
      </InfoBox>
    </DocPage>
  );
}
