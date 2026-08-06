import React from 'react';
import { Link } from 'react-router-dom';
import { Cpu } from 'lucide-react';
import DocPage, { InfoBox, ApiTable, FeatureList } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Hardware() {
  return (
    <DocPage
      title="Hardware &amp; Backends"
      subtitle="Detect Apple Silicon, CUDA, and MLX; pick the best backend for the current machine; run vision-language models on Apple Silicon."
      icon={<Cpu size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Core Concepts', path: '/agents' },
        { label: 'Hardware' },
      ]}
    >
      <h2>Fourteen Inference Backends</h2>
      <ApiTable
        headers={['Backend', 'Hardware', 'Install', 'Best for']}
        rows={[
          [<code>transformers</code>, 'CPU / CUDA', 'default', 'Development, quantization (4bit/8bit/AWQ/GPTQ)'],
          [<code>vllm</code>, 'CUDA', <code>pip install effgen[vllm]</code>, 'Production — continuous batching, PagedAttention, 5-10× speedup'],
          [<code>mlx</code>, 'Apple Silicon', <code>pip install effgen[mlx]</code>, 'Native M1/M2/M3/M4 text generation'],
          [<code>mlx_vlm</code>, 'Apple Silicon', <code>pip install effgen[mlx-vlm]</code>, 'Vision-language models on Apple Silicon (30+ architectures)'],
          [<code>gguf</code>, 'CPU / Metal', <code>pip install effgen[gguf]</code>, 'Quantized GGUF models via llama-cpp-python'],
          [<code>openai</code>, 'API', 'built in', 'gpt-5/gpt-5.4-nano chat models, o-series reasoning controls, native tools'],
          [<code>anthropic</code>, 'API', 'built in', 'v0.2.1 adapter; v0.2.2+ Claude 4.x thinking, prompt caching, streaming'],
          [<code>gemini</code>, 'API', 'built in', 'v0.2.1 adapter; v0.2.2+ Gemini thinking, grounding, Files API, native tools'],
          [<code>cerebras</code>, 'API', <code>pip install effgen[cerebras]</code>, '4 registered models, 2 reliably free-tier callable, streaming, model-dependent native tools'],
          [<code>groq</code>, 'API', <code>pip install effgen[groq]</code>, '16 chat models, streaming, native tools, rate-limit windows'],
          [<code>together</code>, 'API', <code>pip install effgen[together]</code>, '163-model catalog, streaming, refresh/drift checks'],
          [<code>fireworks</code>, 'API', <code>pip install effgen[fireworks]</code>, '80 hosted chat models, 54 tool-capable, streaming'],
          [<code>replicate</code>, 'API', <code>pip install effgen[replicate]</code>, '38 models, 34 streaming-capable, async predictions, SSE streaming'],
          [<code>hf</code>, 'API', <code>pip install effgen[hf]</code>, '124-model HF Inference Router plus custom endpoints'],
        ]}
      />
      <InfoBox type="info" title="Local engines plus ProviderRegistry (v0.2.3+)">
        <p>
          <code>mlx_vlm</code> is technically the same MLX runtime as <code>mlx</code> but
          configured for vision-language models. We list them separately because they have
          different install extras and entry points. In v0.2.3+, cloud providers are registered through
          <Link to="/providers"> ProviderRegistry</Link> and can be checked with
          <code> effgen doctor</code>.
        </p>
      </InfoBox>

      <h2>Hardware Detection</h2>
      <p>
        <code>effgen.hardware</code> provides helpers to detect the current platform and
        recommend the best backend:
      </p>

      <CodeBlock
        code={`from effgen.hardware import (
    is_apple_silicon,
    is_cuda_available,
    is_mlx_available,
    is_mlx_vlm_available,
    get_unified_memory_gb,
    get_best_local_backend,
    HardwarePlatform,
)
from effgen.hardware.platform import detect_platform

print(detect_platform())
# HardwarePlatform.APPLE_SILICON / CUDA / CPU_ONLY

print(get_best_local_backend())
# "mlx" | "vllm" | "transformers"

if is_apple_silicon():
    print(f"Unified memory: {get_unified_memory_gb()} GB")`}
        language="python"
        filename="hardware_detect.py"
      />

      <h2>Apple Silicon (MLX)</h2>
      <FeatureList
        features={[
          { icon: '🍎', title: 'MLXEngine', description: 'Text generation via mlx-lm. Streaming and batch supported.' },
          { icon: '👁️', title: 'MLXVLMEngine', description: 'Vision-language models across 30+ architectures (LLaVA, Qwen-VL, Phi-Vision, Idefics, ...).' },
          { icon: '⚡', title: 'Unified memory', description: 'Model weights live in unified memory — no host ↔ device copies.' },
          { icon: '📦', title: 'Optional install', description: 'pip install effgen[mlx] or [mlx-vlm] — only installs on darwin/arm64.' },
        ]}
      />

      <CodeBlock
        code={`from effgen import load_model

# Text
model = load_model("mlx-community/Qwen2.5-3B-Instruct-4bit", engine="mlx")

# Vision-language
vlm = load_model("mlx-community/Qwen2-VL-2B-Instruct-4bit", engine="mlx_vlm")

# Vision-capable adapters pass images through:
#   OpenAIAdapter, AnthropicAdapter  → image_url / image blocks
#   MLXVLMEngine                     → native PIL.Image / bytes`}
        language="python"
        filename="mlx_usage.py"
      />

      <h2>CUDA (vLLM &amp; Transformers)</h2>
      <CodeBlock
        code={`# vLLM — production, multi-GPU
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="vllm",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
)

# Transformers — development, quantization
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="transformers",
    quantization="4bit",         # or "8bit", "awq", "gptq"
    device_map="auto",
)`}
        language="python"
        filename="cuda_usage.py"
      />

      <h2>GGUF (CPU / Metal)</h2>
      <p>
        Any <code>.gguf</code> model path is auto-routed to <code>GGUFEngine</code>
        (backed by <code>llama-cpp-python</code>) — a good fit for laptops without a GPU.
        The file extension is the trigger; you do not need to pass <code>engine="gguf"</code>.
      </p>
      <CodeBlock
        code={`model = load_model("./models/qwen2.5-3b-instruct-q4_k_m.gguf")`}
        language="python"
        filename="gguf_usage.py"
      />

      <h2>GPU Management</h2>
      <p>
        The <code>effgen.gpu</code> module provides <code>GPUAllocator</code> (pick a GPU by
        memory / utilisation), <code>GPUMonitor</code> (live usage, temperature, power), and
        helpers for multi-GPU distribution.
      </p>
      <InfoBox type="success" title="Real GPU support (v0.3.0)">
        <p>
          v0.3.0 makes local GPUs work out of the box. The documented GPU install selects a{' '}
          <strong>driver-compatible torch</strong>, and a runtime guard warns once when NVML sees
          GPUs but <code>torch.cuda</code> cannot use them (falling back to CPU) — so a broken
          ABI/CUDA mismatch is loud instead of silent. The allocator no longer deadlocks on{' '}
          <code>reset()</code>, reads real free memory via NVML / <code>torch.cuda.mem_get_info</code>{' '}
          (respecting other processes), and no longer mutates the global CUDA device during
          discovery. CPU / CUDA-12.4 / CUDA-13 / vLLM install matrices are documented, and a{' '}
          <code>constraints-cu1xx.txt</code> flow prevents an extras re-install from replacing a
          working torch. <code>temperature=0</code> now decodes greedily
          (<code>do_sample=False</code>) across local backends instead of raising.
        </p>
      </InfoBox>

      <InfoBox type="success" title="v0.3.1 — physical GPU memory in models status">
        <p>
          v0.3.1 makes <code>effgen models status</code> report <strong>physical GPU memory</strong>{' '}
          — the driver's view across all processes (via <code>mem_get_info</code>) plus a
          utilization column — so it shows which GPU is actually free, not just the calling
          process's reservations. <code>effgen models info</code> is now local-aware too: a model in
          the local HuggingFace cache is described as locally-runnable (engines, on-disk size,
          context window) instead of "not found", and incomplete downloads are flagged rather than
          counted as ready.
        </p>
      </InfoBox>

      <CodeBlock
        code={`from effgen.gpu import GPUAllocator, GPUMonitor
from effgen.gpu.allocator import AllocationRequest

allocator = GPUAllocator()
allocation = allocator.allocate(AllocationRequest(
    requester_id="agent_1",
    memory_required=24 * 1024**3,   # bytes (24 GB)
    num_gpus=1,
))
if allocation:
    print("Allocated GPUs:", allocation.device_ids)
allocator.deallocate("agent_1")

monitor = GPUMonitor()
metrics = monitor.get_metrics()           # {device_id: GPUMetrics}
for device_id, m in metrics.items():
    print(device_id, m.gpu_utilization, m.vram_used_gb, m.temperature)`}
        language="python"
        filename="gpu_management.py"
      />

      <h2>See Also</h2>
      <p>
        <Link to="/models">Models</Link> · <Link to="/installation">Installation</Link> ·
        {' '}<Link to="/configuration">Configuration</Link>
      </p>
    </DocPage>
  );
}
