import React from 'react';
import { Link } from 'react-router-dom';
import { Settings } from 'lucide-react';
import DocPage, { InfoBox, ApiTable } from '../components/DocPage';
import CodeBlock from '../components/CodeBlock';

export default function Installation() {
  return (
    <DocPage
      title="Installation"
      subtitle="Get effGen up and running in your environment. Supports pip, conda, and development installations."
      icon={<Settings size={48} />}
      breadcrumbs={[
        { label: 'Docs', path: '/introduction' },
        { label: 'Installation' },
      ]}
    >
      <h2>Quick Installation</h2>
      <p>
        The fastest way to get started with effGen is via pip:
      </p>

      <CodeBlock
        code="pip install effgen"
        language="bash"
        filename="terminal"
      />

      <h3>Conda-Forge</h3>
      <CodeBlock
        code="conda install -c conda-forge effgen"
        language="bash"
        filename="terminal"
      />

      <h2>Automated Installation (Recommended for Development)</h2>
      <p>
        We provide a comprehensive installation script that automates the entire setup process.
        This script handles conda environment creation, dependency installation, configuration,
        and verification tests with a beautiful CLI interface.
      </p>

      <CodeBlock
        code={`# Clone the repository
git clone https://github.com/ctrl-gaurav/effGen.git
cd effGen

# Run the automated installer
bash scripts/install_effgen.sh

# Or with options:
bash scripts/install_effgen.sh --install-vllm --download-models --dev`}
        language="bash"
        filename="terminal"
      />

      <InfoBox type="success" title="What the Installer Does">
        <ul>
          <li><strong>Environment Management:</strong> Creates and activates a conda environment named 'effgen'</li>
          <li><strong>Dependency Installation:</strong> Installs all required packages and dependencies</li>
          <li><strong>Framework Setup:</strong> Runs pip install -e . in development mode</li>
          <li><strong>Configuration:</strong> Creates default config files and .env for API keys</li>
          <li><strong>Verification:</strong> Runs tests to ensure everything is working correctly</li>
          <li><strong>Beautiful CLI:</strong> Provides colored, step-by-step progress output</li>
        </ul>
      </InfoBox>

      <h3>Installation Script Options</h3>
      <ApiTable
        headers={['Option', 'Description']}
        rows={[
          [<code>--skip-conda</code>, 'Use system Python instead of creating conda environment'],
          [<code>--install-vllm</code>, 'Install vLLM for 5-10x faster inference'],
          [<code>--download-models</code>, 'Download recommended models during installation'],
          [<code>--dev</code>, 'Install development dependencies (pytest, black, etc.)'],
          [<code>--minimal</code>, 'Install only core dependencies'],
          [<code>--skip-verification</code>, 'Skip verification tests'],
          [<code>--help</code>, 'Show all available options'],
        ]}
      />

      <h2>Installation with vLLM (Recommended for Production)</h2>
      <p>
        For maximum performance in production environments, install with vLLM support.
        vLLM provides 5-10x faster inference through continuous batching and PagedAttention:
      </p>

      <CodeBlock
        code="pip install effgen[vllm]"
        language="bash"
        filename="terminal"
      />

      <InfoBox type="success" title="Real GPU support &amp; the torch constraints flow (v0.3.0)">
        <p>
          v0.3.0 documents CPU / CUDA-12.4 / CUDA-13 / vLLM install matrices and selects a{' '}
          <strong>driver-compatible torch</strong>. A runtime guard warns once when NVML sees GPUs
          but <code>torch.cuda</code> cannot use them (so an ABI / CUDA mismatch is loud, not
          silent). Use the matching <code>constraints-cu1xx.txt</code> when installing extras so a
          re-install never replaces a working torch:
        </p>
        <CodeBlock
          code={`# Pin a working CUDA torch while installing extras
pip install -e ".[all]" -c constraints-cu124.txt   # or constraints-cu130.txt
# CPU-only
pip install -e ".[all]" -c constraints-cpu.txt`}
          language="bash"
          filename="terminal"
        />
      </InfoBox>

      <h2>Apple Silicon (MLX)</h2>
      <p>
        On Apple Silicon (M1/M2/M3/M4) effGen can use the MLX backend directly — no CUDA required.
        <code>MLXEngine</code> handles text models; <code>MLXVLMEngine</code> handles vision-language models
        across 30+ architectures. Both are optional deps and only install on <code>darwin/arm64</code>.
      </p>
      <CodeBlock
        code={`pip install effgen[mlx]        # text generation on Apple Silicon
pip install effgen[mlx-vlm]    # vision-language models`}
        language="bash"
        filename="terminal"
      />

      <h2>GGUF (llama-cpp)</h2>
      <p>
        For <code>.gguf</code> quantized models the <code>GGUFEngine</code> wraps <code>llama-cpp-python</code>.
        The model loader auto-routes any <code>.gguf</code> path to this engine.
      </p>
      <CodeBlock
        code="pip install effgen[gguf]"
        language="bash"
        filename="terminal"
      />

      <h2>Other Optional Extras</h2>
      <ApiTable
        headers={['Extra', 'Purpose']}
        rows={[
          [<code>effgen[vllm]</code>, '5-10× faster CUDA inference (PagedAttention, continuous batching)'],
          [<code>effgen[cerebras]</code>, 'v0.2.1: Cerebras Cloud SDK backend with 4 registered models, 2 reliably free-tier callable models, streaming, model-dependent native tools, and rate limits'],
          [<code>effgen[groq]</code>, 'v0.2.3+: Groq backend with 16 chat models, streaming, native tool-calling support, and rate-limit windows'],
          [<code>effgen[together]</code>, 'v0.2.3+: Together AI backend with a 163-model catalog, streaming, refresh_models(), and drift detection'],
          [<code>effgen[fireworks]</code>, 'v0.2.3+: Fireworks backend with 80 OpenAI-compatible chat models, 54 tool-capable models, and pricing metadata'],
          [<code>effgen[replicate]</code>, 'v0.2.3+: Replicate backend with 38 models, 34 streaming-capable models, async prediction polling, SSE streaming, and timeout cancellation'],
          [<code>effgen[hf]</code>, 'v0.2.3+: HuggingFace Inference Router backend with 124 bundled models and custom Inference Endpoint URLs'],
          [<code>effgen[dev]</code>, 'pytest, pytest-asyncio, coverage, pytest-timeout, black, isort, flake8, mypy, bitsandbytes'],
          [<code>effgen[mlx]</code>, 'MLX backend for Apple Silicon text generation'],
          [<code>effgen[mlx-vlm]</code>, 'MLX-VLM for vision-language models on Apple Silicon'],
          [<code>effgen[gguf]</code>, 'GGUF quantized models via llama-cpp-python'],
          [<code>effgen[vector-db]</code>, 'faiss-cpu, chromadb, and qdrant-client; VectorMemoryStore currently supports faiss and chroma backends'],
          [<code>effgen[rag]</code>, 'sentence-transformers + faiss-cpu for embeddings & hybrid search'],
          [<code>effgen[finance]</code>, 'yfinance for StockPriceTool'],
          [<code>effgen[data]</code>, 'matplotlib + plotly for PlotTool / DataFrame visualisations'],
          [<code>effgen[documents]</code>, 'v0.2.6: pypdf + pdfplumber + python-docx + openpyxl + pandas (PDFTool, DOCXTool, ExcelTool)'],
          [<code>effgen[audio]</code>, 'v0.2.6: faster-whisper + huggingface_hub (AudioTranscribeTool — auto-detects CPU/GPU and falls back to HF Inference)'],
          [<code>effgen[tools]</code>, 'v0.2.6: pytesseract + Pillow (OCRTool, ImageInfoTool) — install Tesseract via apt/brew for the local primary path'],
          [<code>effgen[geo]</code>, 'v0.2.6: staticmap + Pillow for MapsTool (OSM static PNG renderer). WeatherTool + GeocodeTool need no extras (stdlib + httpx).'],
          [<code>effgen[rss]</code>, 'v0.2.5: feedparser for RSSFeedTool and NewsTool'],
          [<code>effgen[youtube]</code>, 'v0.2.5: youtube-transcript-api + yt-dlp (YouTubeTranscriptTool, YouTubeMetadataTool)'],
          [<code>effgen[translate]</code>, 'v0.2.5: argostranslate offline fallback + langdetect (TranslateTool, LanguageDetectTool)'],
          [<code>effgen[qr]</code>, 'v0.2.5: qrcode[pil] + pyzbar + opencv-python-headless (QRGenerateTool, QRReadTool — fully local)'],
          [<code>effgen[eval]</code>, 'rouge-score + nltk for evaluation scoring'],
          [<code>effgen[monitoring]</code>, 'wandb + tensorboard experiment tracking'],
          [<code>effgen[flash-attn]</code>, 'Flash Attention for faster transformer inference'],
          [<code>effgen[search]</code>, 'Google / SerpAPI web-search backends'],
          [<code>effgen[cloud-secrets]</code>, 'AWS Secrets Manager / Vault / Azure Key Vault'],
          [<code>effgen[all]</code>, 'Most feature and provider extras bundled (excludes dev, flash-attn, and Apple Silicon MLX extras)'],
        ]}
      />

      <h2>Cloud Provider Extras (v0.2.1-v0.2.3)</h2>
      <p>
        Cloud adapters are optional where they need provider SDKs. Install only the
        providers you use, or install <code>effgen[all]</code> for most extras.
      </p>
      <CodeBlock
        code={`pip install "effgen[cerebras]"
pip install "effgen[groq]"
pip install "effgen[together]"
pip install "effgen[fireworks]"
pip install "effgen[replicate]"
pip install "effgen[hf]"`}
        language="bash"
        filename="terminal"
      />

      <InfoBox type="info" title="vLLM Benefits">
        <ul>
          <li><strong>5-10x faster inference</strong> compared to standard Transformers</li>
          <li><strong>Automatic multi-GPU support</strong> with tensor parallelism</li>
          <li><strong>PagedAttention</strong> for efficient memory management</li>
          <li><strong>Continuous batching</strong> for high throughput</li>
        </ul>
      </InfoBox>

      <h2>Full Installation with All Features</h2>
      <p>
        To install effGen with the bundled feature extras:
      </p>

      <CodeBlock
        code="pip install effgen[all]"
        language="bash"
        filename="terminal"
      />

      <p>This includes most optional feature dependencies:</p>
      <ul>
        <li>vLLM for fast inference</li>
        <li>Vector stores for long-term memory and RAG</li>
        <li>Search, cloud secrets, monitoring, finance, data, eval, GGUF, and cloud provider extras</li>
        <li>Not included: <code>dev</code>, <code>flash-attn</code>, <code>mlx</code>, and <code>mlx-vlm</code></li>
      </ul>

      <h2>Development Installation</h2>
      <p>
        For contributors or those who want the latest features from the main branch:
      </p>

      <CodeBlock
        code={`git clone https://github.com/ctrl-gaurav/effGen.git
cd effGen
pip install -e ".[dev]"`}
        language="bash"
        filename="terminal"
      />

      <h2>System Requirements</h2>

      <ApiTable
        headers={['Requirement', 'Minimum', 'Recommended']}
        rows={[
          ['Python', '3.10+', '3.11+'],
          ['RAM', '8GB', '16GB+'],
          ['GPU VRAM', '4GB (for 3B models)', '16GB+ (for 7B+ models)'],
          ['CUDA', '11.8+ (optional)', '12.1+'],
          ['Disk Space', '10GB', '50GB+'],
        ]}
      />

      <InfoBox type="warning" title="GPU Requirements">
        <ul>
          <li><strong>No GPU:</strong> Use CPU mode with smaller models (0.5B-1.5B) - slower but functional</li>
          <li><strong>4GB VRAM:</strong> Run 3B models with 4-bit quantization</li>
          <li><strong>8GB VRAM:</strong> Run 7B models with 4-bit quantization</li>
          <li><strong>16GB+ VRAM:</strong> Run larger models with full precision</li>
        </ul>
      </InfoBox>

      <h2>Verify Installation</h2>
      <p>
        After installation, verify everything is working correctly:
      </p>

      <CodeBlock
        code={`python -c "import effgen; print(f'effGen v{effgen.__version__}')"

# Test model loading
python -c "
from effgen import load_model
model = load_model('Qwen/Qwen2.5-0.5B-Instruct', engine='transformers')
result = model.generate('Hello, world!')
print(result.text)
"`}
        language="bash"
        filename="terminal"
      />

      <h2>Environment Setup</h2>

      <h3>API Keys (Optional)</h3>
      <p>
        If you plan to use external API providers, set up your environment variables.
        On v0.2.3 and later, verify them with <code>effgen doctor</code>:
      </p>

      <CodeBlock
code={`# OpenAI
export OPENAI_API_KEY="your-openai-key"
export OPENAI_ORG_ID="optional-openai-org-id"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google (Gemini)
export GOOGLE_API_KEY="your-google-key"

# Cerebras
export CEREBRAS_API_KEY="your-cerebras-key"

# Groq
export GROQ_API_KEY="your-groq-key"

# Together AI
export TOGETHER_API_KEY="your-together-key"

# Fireworks
export FIREWORKS_API_KEY="your-fireworks-key"

# Replicate
export REPLICATE_API_TOKEN="your-replicate-token"

# HuggingFace Inference
export HF_TOKEN="your-hf-token"

# v0.2.3+: check ProviderRegistry auth readiness
effgen doctor

# v0.2.4+: persistent cost dashboard + budget guardrails
effgen cost today
effgen cost set-budget 1.0

# Or use a .env file
echo 'OPENAI_API_KEY=your-openai-key' >> .env
echo 'ANTHROPIC_API_KEY=your-anthropic-key' >> .env
echo 'GROQ_API_KEY=your-groq-key' >> .env`}
        language="bash"
        filename=".bashrc or .env"
      />

      <h3>State Directory (v0.2.4+)</h3>
      <p>
        v0.2.4 writes persistent cost + rate-limit state under
        <code> ~/.effgen/</code>:
      </p>
      <ul>
        <li><code>~/.effgen/costs.sqlite</code> — every paid API call (SQLiteCostStore)</li>
        <li><code>~/.effgen/rate_limits.sqlite</code> — cross-process rate-limit budgets (SQLiteRateLimitStore, WAL mode)</li>
        <li><code>~/.effgen/budget.json</code> — daily / monthly budget caps used by <code>effgen cost</code></li>
        <li><code>~/.effgen/.env</code> — optional shared API keys; loaded by <code>effgen doctor</code> and the CLI before any command runs</li>
      </ul>

      <h3>HuggingFace Token (Optional)</h3>
      <p>
        For gated models on HuggingFace:
      </p>

      <CodeBlock
        code={`# Login to HuggingFace
huggingface-cli login

# Or set the token directly
export HF_TOKEN="your-huggingface-token"`}
        language="bash"
        filename="terminal"
      />

      <h2>Docker Installation</h2>
      <p>
        For containerized deployments:
      </p>

      <CodeBlock
        code={`# Build the production server image (multi-stage; non-root; HEALTHCHECK on /health)
docker build -f deploy/docker/Dockerfile --build-arg EXTRAS=server -t effgen:0.3.1 .

# Run (dev mode — auth disabled, for local testing only)
docker run --rm -p 8080:8080 -e EFFGEN_DEV_MODE=1 effgen:0.3.1
curl http://localhost:8080/health   # {"status":"ok"}

# Or build the base image from the repo root Dockerfile
docker build -t effgen .
docker run --gpus all -it effgen`}
        language="bash"
        filename="terminal"
      />
      <p>
        See the <a href="/docs/deployment">Deployment</a> guide for Helm, AWS Lambda, and the
        Cloudflare Worker edge proxy, plus production OIDC-auth configuration.
      </p>

      <h2>Troubleshooting</h2>

      <h3>CUDA Not Found</h3>
      <CodeBlock
        code={`# Check CUDA installation
nvidia-smi

# If not found, install CUDA toolkit
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit

# Or use conda:
conda install -c conda-forge cudatoolkit=11.8`}
        language="bash"
        filename="terminal"
      />

      <h3>Out of Memory Errors</h3>
      <CodeBlock
        code={`# Use quantization to reduce memory usage
from effgen import load_model

model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    engine="transformers",
    quantization="4bit"  # or "8bit"
)`}
        language="python"
        filename="reduce_memory.py"
      />

      <h3>Slow Model Loading</h3>
      <CodeBlock
        code={`# Pre-download models
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen2.5-7B-Instruct")

# Models are cached in ~/.cache/huggingface/`}
        language="python"
        filename="predownload.py"
      />

      <InfoBox type="success" title="Next Steps">
        <p>
          Installation complete! Head to the <Link to="/quickstart">Quick Start guide</Link> to build your first agent.
        </p>
      </InfoBox>
    </DocPage>
  );
}
