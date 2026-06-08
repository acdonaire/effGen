# Installation Guide

## Standard install

```bash
pip install effgen                    # base install (all core features)
pip install effgen[dev]                # dev tools (pytest, ruff, mypy, pytest-forked)
pip install effgen[rag]                # sentence-transformers + faiss-cpu
pip install effgen[vector-db]          # faiss + chromadb + qdrant
pip install effgen[vllm]               # vLLM backend (NVIDIA GPUs)
pip install effgen[mlx]                # MLX backend (Apple Silicon)
```

> **On an NVIDIA GPU box?** The default PyTorch wheel on PyPI may be built for a
> newer CUDA runtime than your driver supports, which silently disables the GPU
> (`torch.cuda.is_available()` is `False` even though `nvidia-smi` lists your
> GPUs). Pick a torch wheel that matches your driver — see
> [GPU / CUDA compatibility](#gpu--cuda-compatibility) below. The bundled
> `./install.sh` does this automatically.

### Use-case extras

Install only the slice you need instead of the everything-extra `[all]`. The
base install already leaves the heavy optional stacks (vLLM, vector DBs, OCR,
OpenCV, spaCy/Stanza) out, so these pull just their use case on top:

```bash
pip install effgen[api]                # hosted-inference SDKs (groq/together/fireworks/replicate/cerebras/hf)
pip install effgen[local]              # local-model quantization + GGUF backends (bitsandbytes, llama-cpp)
pip install effgen[server]             # OIDC/JWT auth + Prometheus metrics for the API server
pip install effgen[tools-web]          # web-search / browsing tools
pip install effgen[tools-docs]         # document parsing (PDF / DOCX / XLSX)
```

> `[all]` is the **CI / everything** extra — it installs every optional
> dependency so the full test suite runs. It is large and slow to resolve;
> prefer a use-case extra above unless you specifically need everything. See
> [Installing the `[all]` extra](#installing-the-all-extra) for the constraints
> lock it requires.

### Installing the `[all]` extra

`[all]` pulls vLLM plus every provider SDK and the full Google client stack.
Under the `protobuf>=5.29.5` security floor this dependency graph is too deep
for pip to resolve on its own (`resolution-too-deep`). Install it with the
committed constraints lock, which pins a single consistent, CVE-safe solution:

```bash
# from a clone of the repo:
pip install -e ".[all]" -c requirements-all-lock.txt

# or against the published package:
pip install "effgen[all]" -c https://raw.githubusercontent.com/ctrl-gaurav/effGen/main/requirements-all-lock.txt
```

Regenerate the lock after changing dependencies (requires
[uv](https://github.com/astral-sh/uv)):

```bash
uv pip compile pyproject.toml --extra all --output-file requirements-all-lock.txt
```

> **Why isn't `flash-attn` in `[all]`?**
> `flash-attn`'s own `setup.py` imports `torch` during wheel-metadata generation,
> but pip's isolated build environment does not have `torch` available at that
> moment. This is a well-known upstream bug in `flash-attn` — any package that
> lists `flash-attn` as a dependency will cause `pip install` to fail. To keep
> `pip install effgen[all]` working for everyone, `flash-attn` is kept out of
> `[all]` and installed separately (see below).

## GPU / CUDA compatibility

PyTorch wheels are built against a specific CUDA runtime (`torch 2.x+cu124`,
`+cu130`, …). An NVIDIA driver is **forward-compatible only**: a driver that
reports `CUDA Version: 12.4` (`nvidia-smi`) can run torch built for CUDA ≤ 12.x,
but **not** a torch built for CUDA 13. Installing a CUDA-13 wheel on a CUDA-12.4
driver leaves you with `torch.cuda.is_available() == False` and everything runs
(slowly) on the CPU.

Check your driver's maximum CUDA version first:

```bash
nvidia-smi    # top-right: "CUDA Version: 12.4"
```

Then install a matching torch wheel **before** installing effgen (or let
`./install.sh` detect the driver and do it for you):

| Your environment | torch index URL | Install command |
|---|---|---|
| **CPU only** (no NVIDIA GPU) | `cpu` | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| **CUDA 12.1–12.7 driver** (e.g. 12.4) | `cu124` | `pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu124` |
| **CUDA 12.8+ driver** | `cu128` | `pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu128` |
| **CUDA 13.x driver** | `cu130` | `pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu130` |
| **vLLM** (NVIDIA GPU) | matches vLLM's pinned torch | see [Installing vLLM](#installing-vllm-optional) |

```bash
# Example: a host with a CUDA 12.4 driver
pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu124
pip install effgen
python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

effGen detects a torch-CUDA / driver mismatch at runtime: when it sees physical
NVIDIA GPUs but `torch.cuda` cannot use them, it prints **one** warning naming the
torch CUDA build vs the driver's CUDA version and pointing back here, instead of
silently running on the CPU. Set `EFFGEN_NO_GPU_WARN=1` to silence it if you are
deliberately running CPU-only on a GPU box.

## Installing flash-attn (optional, NVIDIA GPUs only)

**Step 1 — install effgen first (gets torch and everything else):**

```bash
pip install -e ".[all]" -c requirements-all-lock.txt
```

**Step 2 — install flash-attn with build isolation disabled:**

```bash
pip install flash-attn --no-build-isolation
```

`--no-build-isolation` lets flash-attn's setup.py reuse the torch already
installed in your environment, bypassing the bug.

### Requirements for flash-attn

- NVIDIA GPU with compute capability ≥ 7.5 (Turing or newer)
- CUDA toolkit (`nvcc`) matching your torch CUDA version
- GCC 9+ and a few GB of RAM for compilation (can take 10–30 minutes)

If the build fails, prefer the official pre-built wheel from
<https://github.com/Dao-AILab/flash-attention/releases> matching your
exact `(python, torch, cuda)` triple.

## Installing vLLM (optional)

vLLM ships pre-built wheels on PyPI for common
`(python, torch, cuda)` combinations:

```bash
pip install effgen[vllm]
```

If the resolver cannot find a matching wheel, use vLLM's extra index:

```bash
pip install effgen
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124
```

## Supported Python versions

`effgen` officially supports Python **3.10, 3.11, 3.12, 3.13**. Python 3.14
is best-effort — several upstream packages (torch, bitsandbytes) do not yet
ship cp314 wheels.

## Verifying your install

```bash
python -c "import effgen; print(effgen.__version__)"
python -c "from effgen import Agent; print(Agent)"
```
