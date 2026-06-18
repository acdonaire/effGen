"""
Together AI model registry for effGen.

Full model catalog fetched from Together API on 2026-04-28 (244 models total:
149 chat, 13 language, 11 audio, 31 image, 30 video, 5 transcribe, 2 rerank, 1 embedding, 1 code, 1 moderation).

Pricing from Together AI pricing page (https://www.together.ai/pricing).
Rate limits: Together AI applies per-account limits; serverless tier ~100 RPM typical.

serverless=True  — accessible via standard API without a dedicated endpoint.
serverless=False — requires a dedicated endpoint started in Together console.

Tool-calling support verified by live API tests on 2026-04-28.
Together uses an OpenAI-compatible function-calling API.

REGISTRY_FETCH_DATE: str = "2026-04-28"
Use refresh_models() to fetch the live catalog and detect drift.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Date this bundled registry was last fetched from the Together API
REGISTRY_FETCH_DATE = "2026-06-17"

# ---------------------------------------------------------------------------
# Chat models (149 total as of 2026-04-28)
# Sorted by input price (ascending). Free/\$0 models listed first.
# ---------------------------------------------------------------------------
TOGETHER_MODELS: dict[str, dict] = {
    # Google — Gemma 4 E4B-it
    "google/gemma-4-E4B-it": {
        "family": "gemma-4",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Hcompany — Holo3 35B A3b
    "Hcompany/Holo3-35B-A3B": {
        "family": "holo",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Hcompany",
    },
    # Google — Gemma 4 26B A4b It
    "google/gemma-4-26B-A4B-it": {
        "family": "gemma-4",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Qwen — Qwen3 Coder 30B A3b Instruct
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen2.5 32B
    "Qwen/Qwen2.5-32B": {
        "family": "qwen",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Google — Gemma 2 9B It
    "google/gemma-2-9b-it": {
        "family": "gemma",
        "context": 8192,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Llama — nim/meta/llama-3.1-70b-instruct
    "nim/meta/llama-3.1-70b-instruct": {
        "family": "llama",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Llama",
    },
    # Meta — nim/meta/llama-3.1-8b-instruct
    "nim/meta/llama-3.1-8b-instruct": {
        "family": "llama",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # NVIDIA — nim/nv-mistralai/mistral-nemo-12b-instruct
    "nim/nv-mistralai/mistral-nemo-12b-instruct": {
        "family": "mistral",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "NVIDIA",
    },
    # NVIDIA — nim/nvidia/llama-3.1-nemotron-70b-instruct
    "nim/nvidia/llama-3.1-nemotron-70b-instruct": {
        "family": "llama",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "NVIDIA",
    },
    # deepcogito — Cogito V1 Preview Llama 70B
    "deepcogito/cogito-v1-preview-llama-70B": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "deepcogito",
    },
    # deepcogito — Cogito V1 Preview Llama 70B Turbo
    "deepcogito/cogito-v1-preview-llama-70B-Turbo": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "deepcogito",
    },
    # Zai Org — GLM 5 Fp4
    "zai-org/GLM-5-FP4": {
        "family": "glm-5",
        "context": 202752,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Zai Org",
    },
    # Nvidia — Nemotron 3 Nano Omni 30B A3b Reasoning Fp8
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-fp8": {
        "family": "nemotron",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Nvidia",
    },
    # deepcogito — Cogito V1 Preview Llama 8B
    "deepcogito/cogito-v1-preview-llama-8B": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "deepcogito",
    },
    # MiniMaxAI — MiniMax M2.5 FP4
    "MiniMaxAI/MiniMax-M2.5-FP4": {
        "family": "minimax",
        "context": 8192,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "MiniMaxAI",
    },
    # deepcogito — Cogito V1 Preview Qwen 14B
    "deepcogito/cogito-v1-preview-qwen-14B": {
        "family": "cogito",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "deepcogito",
    },
    # deepcogito — Cogito V1 Preview Qwen 32B
    "deepcogito/cogito-v1-preview-qwen-32B": {
        "family": "qwen-3",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "deepcogito",
    },
    # Deepseek — Deepseek OCR 2
    "deepseek-ai/DeepSeek-OCR-2": {
        "family": "deepseek",
        "context": 8192,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Deepseek",
    },
    # Google — Gemma 3 4b it
    "google/gemma-3-4b-it": {
        "family": "gemma-3",
        "context": 65536,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Deepseek — DeepSeek R1 Distill Qwen 7B
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "family": "deepseek-r1",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Deepseek",
    },
    # Deepseek — Deepseek V3
    # Deepseek — Deepseek V3.1 Base
    # Deepseek — Deepseek V3.2 Exp
    # Meta — Llama 3.2 1B
    "meta-llama/Llama-3.2-1B": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Nvidia — Nvidia Nemotron 3 Super 120B A12b Fp8
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": {
        "family": "nemotron",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Nvidia",
    },
    # Togethercomputer — Deepcoder 14B Preview
    "agentica-org/DeepCoder-14B-Preview": {
        "family": "deepcoder",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Togethercomputer",
    },
    # MiniMax — Minimax M1 40K
    "MiniMaxAI/MiniMax-M1-40k": {
        "family": "minimax",
        "context": 1048576,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "MiniMax",
    },
    # MiniMax — Minimax M1 80K
    "MiniMaxAI/MiniMax-M1-80k": {
        "family": "minimax",
        "context": 1048576,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "MiniMax",
    },
    # Qwen — Qwen2.5 1.5B
    "Qwen/Qwen2.5-1.5B": {
        "family": "qwen",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen2.5 1.5B Instruct
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "family": "qwen",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen2.5 14B
    "Qwen/Qwen2.5-14B": {
        "family": "qwen",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen2.5 3B Instruct
    "Qwen/Qwen2.5-3B-Instruct": {
        "family": "qwen",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen2.5 72B
    "Qwen/Qwen2.5-72B": {
        "family": "qwen",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen2.5 7B
    "Qwen/Qwen2.5-7B": {
        "family": "qwen",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen2.5 7B Instruct
    "Qwen/Qwen2.5-7B-Instruct": {
        "family": "qwen",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3 0.6B
    "Qwen/Qwen3-0.6B": {
        "family": "qwen-3",
        "context": 40960,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3 1.7B
    "Qwen/Qwen3-1.7B": {
        "family": "qwen-3",
        "context": 40960,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3 235B A22B FP8
    # Qwen — Qwen3 8B
    "Qwen/Qwen3-8B": {
        "family": "qwen-3",
        "context": 40960,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3 Next 80B A3b Instruct Fp8
    "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": {
        "family": "qwen-3",
        "context": 0,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3-VL-235B-A22B-Instruct-FP8
    "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Google — Gemma 3 27B Pt
    "google/gemma-3-27b-pt": {
        "family": "gemma-3",
        "context": 0,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Meta — meta-llama/Llama-2-7b-chat-hf
    "meta-llama/Llama-2-7b-chat-hf": {
        "family": "llama-2",
        "context": 4096,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Meta — nim/meta/llama-3.2-90b-vision-instruct
    "nim/meta/llama-3.2-90b-vision-instruct": {
        "family": "llama",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "supports_vision": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Qwen — Qwen3.5 397B A17b Fp8
    # Google — Gemma 2B It
    "google/gemma-2b-it": {
        "family": "gemma",
        "context": 8192,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Mistralai — Magistral Small 2506
    "mistralai/Magistral-Small-2506": {
        "family": "mistral",
        "context": 40960,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Mistralai",
    },
    # Mistralai — Mistral 7B v0.1
    "mistralai/Mistral-7B-v0.1": {
        "family": "mistral",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Mistralai",
    },
    # DeepSeek — DeepSeek R1 (Original)
    # Deepseek — Deepseek V3.1 Terminus
    # Qwen — Qwen3 30B A3B Instruct 2507 Lora
    "Qwen/Qwen3-30B-A3B-Instruct-2507-Lora": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3 4B Instruct 2507
    "Qwen/Qwen3-4B-Instruct-2507": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3 8B Lora
    "Qwen/Qwen3-8B-Lora": {
        "family": "qwen-3",
        "context": 40960,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Meta — Llama 3.1 405B
    "meta-llama/Llama-3.1-405B": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Meta — Llama 3.1 70B
    "meta-llama/Meta-Llama-3.1-70B": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Deepseek — Deepseek V3.2
    # Meta — Llama 4 Maverick 17B 128E
    # Mistral — nim/mistralai/mixtral-8x22b-instruct-v01
    "nim/mistralai/mixtral-8x22b-instruct-v01": {
        "family": "mistral",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Mistral",
    },
    # Meta — Meta Llama 3.1 8B Instruct Awq Int4
    "togethercomputer/meta-llama-3.1-8B-Instruct-AWQ-INT4": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Zai Org — GLM 4.5V
    "zai-org/GLM-4.5V": {
        "family": "glm-4",
        "context": 65536,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Zai Org",
    },
    # Zai Org — GLM OCR
    "zai-org/GLM-OCR": {
        "family": "glm",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Zai Org",
    },
    # MiniMaxAI — MiniMax M2
    "MiniMaxAI/MiniMax-M2": {
        "family": "minimax",
        "context": 196608,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "MiniMaxAI",
    },
    # Google — Gemma 4 E2B-it
    "google/gemma-4-E2B-it": {
        "family": "gemma-4",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Qwen — Qwen3.6 35B A3b Fp8
    "Qwen/Qwen3.6-35B-A3B-FP8": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Nvidia — Nvidia Nemotron 3 Super 120B A12b Bf16
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": {
        "family": "nemotron",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Nvidia",
    },
    # Qwen — Qwen3.5 122B A10b Fp8
    "Qwen/Qwen3.5-122B-A10B-FP8": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3 30B A3b
    "Qwen/Qwen3-30B-A3B": {
        "family": "qwen-3",
        "context": 40960,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Google — Gemma 3 1b it
    "google/gemma-3-1b-it": {
        "family": "gemma-3",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Google — Gemma 3 270M It
    "google/gemma-3-270m-it": {
        "family": "gemma-3",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Meta — Llama 4 Scout (17Bx16E)
    "meta-llama/Llama-4-Scout-17B-16E": {
        "family": "llama-4",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Mistralai — Devstral Small 2505
    "mistralai/Devstral-Small-2505": {
        "family": "mistral",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Mistralai",
    },
    # Mistralai — Mixtral 8X22b Instruct V0.1
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {
        "family": "mistral",
        "context": 65536,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Mistralai",
    },
    # Nvidia — nim/meta/llama-3.2-11b-vision-instruct
    "nim/meta/llama-3.2-11b-vision-instruct": {
        "family": "llama",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "supports_vision": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Nvidia",
    },
    # Meta — nim/meta/llama-3.3-70b-instruct
    "nim/meta/llama-3.3-70b-instruct": {
        "family": "llama",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # mistralai — nim/mistralai/mixtral-8x7b-instruct-v01
    "nim/mistralai/mixtral-8x7b-instruct-v01": {
        "family": "mistral",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "mistralai",
    },
    # Sarvamai — Sarvam M
    "sarvamai/sarvam-m": {
        "family": "sarvam",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Sarvamai",
    },
    # Essential AI — EssentialAI Rnj-1 Instruct
    "togethercomputer/EssentialAI-RNJ-1-Instruct": {
        "family": "essentialai",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Essential AI",
    },
    # Nvidia — Nvidia Nemotron 3 Nano 30B A3b Bf16
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {
        "family": "nemotron",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Nvidia",
    },
    # Nvidia — nim/nvidia/llama-3.3-nemotron-super-49b-v1
    "nim/nvidia/llama-3.3-nemotron-super-49b-v1": {
        "family": "llama",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Nvidia",
    },
    # Qwen — Qwen3.5 9B Fp8
    "Qwen/Qwen3.5-9B-FP8": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3.5 35B A3b
    "Qwen/Qwen3.5-35B-A3B": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen 2 Instruct (1.5B)
    "Qwen/Qwen2-1.5B-Instruct": {
        "family": "qwen",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.02,
        "pricing_per_1m_output": 0.02,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Togethercomputer — LFM2-24B-A2B
    "LiquidAI/LFM2-24B-A2B": {
        "family": "lfm",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.030000000000000002,
        "pricing_per_1m_output": 0.12000000000000001,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Togethercomputer",
    },
    # OpenAI — OpenAI GPT-OSS 20B
    "openai/gpt-oss-20b": {
        "family": "gpt-oss",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.05,
        "pricing_per_1m_output": 0.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "OpenAI",
    },
    # Google — Gemma 3N E4B Instruct
    "google/gemma-3n-E4B-it": {
        "family": "gemma-3",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.060000000000000005,
        "pricing_per_1m_output": 0.12000000000000001,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Meta — Meta Llama 3.2 1B Instruct
    "meta-llama/Llama-3.2-1B-Instruct": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.060000000000000005,
        "pricing_per_1m_output": 0.060000000000000005,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Nvidia — Nvidia Nemotron Nano 9B V2
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": {
        "family": "nemotron",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.060000000000000005,
        "pricing_per_1m_output": 0.25,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Nvidia",
    },
    # Qwen — Qwen3.5 9B FP8
    "Qwen/Qwen3.5-9B": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.1,
        "pricing_per_1m_output": 0.15,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Meta — Meta Llama 3 8B Instruct Lite
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite": {
        "family": "llama",
        "context": 8192,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.1,
        "pricing_per_1m_output": 0.1,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Togethercomputer — Arize AI Qwen 2 1.5B Instruct
    "arize-ai/qwen-2-1.5b-instruct": {
        "family": "arize",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.1,
        "pricing_per_1m_output": 0.1,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Togethercomputer",
    },
    # mistralai — Mistral Small (24B) Instruct 25.01
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "family": "mistral",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.1,
        "pricing_per_1m_output": 0.3,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "mistralai",
    },
    # OpenAI — OpenAI GPT-OSS 120B
    "openai/gpt-oss-120b": {
        "family": "gpt-oss",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.15,
        "pricing_per_1m_output": 0.6,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "OpenAI",
    },
    # Essential AI — EssentialAI Rnj-1 Instruct
    "essentialai/rnj-1-instruct": {
        "family": "essentialai",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.15,
        "pricing_per_1m_output": 0.15,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Essential AI",
    },
    # Qwen — Qwen3 Next 80B A3b Thinking
    "Qwen/Qwen3-Next-80B-A3B-Thinking": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.15,
        "pricing_per_1m_output": 1.5,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3 Next 80B A3b Instruct
    "Qwen/Qwen3-Next-80B-A3B-Instruct": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.15,
        "pricing_per_1m_output": 1.5,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3-VL-8B-Instruct
    "Qwen/Qwen3-VL-8B-Instruct": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.18000000000000002,
        "pricing_per_1m_output": 0.68,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # DeepSeek — DeepSeek R1 Distill Qwen 1.5B
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "family": "deepseek-r1",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.18000000000000002,
        "pricing_per_1m_output": 0.18000000000000002,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "DeepSeek",
    },
    # Meta — Meta Llama 3.1 8B Instruct Turbo
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.18000000000000002,
        "pricing_per_1m_output": 0.18000000000000002,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Meta — Llama 4 Scout Instruct (17Bx16E)
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "family": "llama-4",
        "context": 1048576,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.18000000000000002,
        "pricing_per_1m_output": 0.5900000000000001,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Google — Gemma 4 31B-it FP8
    "google/gemma-4-31B-it": {
        "family": "gemma-4",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.2,
        "pricing_per_1m_output": 0.5,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Qwen — Qwen3 235B A22B Instruct 2507 FP8 Throughput
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.2,
        "pricing_per_1m_output": 0.6,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Zai Org — Glm 4.5 Air Fp8
    "zai-org/GLM-4.5-Air-FP8": {
        "family": "glm-4",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.2,
        "pricing_per_1m_output": 1.1,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Zai Org",
    },
    # mistralai — Mistral (7B) Instruct v0.1
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "family": "mistral",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.2,
        "pricing_per_1m_output": 0.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "mistralai",
    },
    # Meta — Meta Llama 3 8B Instruct
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "family": "llama",
        "context": 8192,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.2,
        "pricing_per_1m_output": 0.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Mistralai — Ministral 3 14B Instruct 2512
    "mistralai/Ministral-3-14B-Instruct-2512": {
        "family": "mistral",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.2,
        "pricing_per_1m_output": 0.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Mistralai",
    },
    # mistralai — Mistral (7B) Instruct v0.3
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "family": "mistral",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.2,
        "pricing_per_1m_output": 0.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "mistralai",
    },
    # Meta — Meta Llama 3 8B Instruct Reference
    "meta-llama/Llama-3-8b-chat-hf": {
        "family": "llama",
        "context": 8192,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.2,
        "pricing_per_1m_output": 0.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Meta — Llama 4 Maverick Instruct (17Bx128E) FP8
    # MiniMaxAI — MiniMax M2.7 FP4
    "MiniMaxAI/MiniMax-M2.7": {
        "family": "minimax",
        "context": 196608,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.3,
        "pricing_per_1m_output": 1.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "MiniMaxAI",
    },
    # MiniMaxAI — MiniMax M2.5 FP4
    # Qwen — Qwen2.5 7B Instruct Turbo
    "Qwen/Qwen2.5-7B-Instruct-Turbo": {
        "family": "qwen",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.3,
        "pricing_per_1m_output": 0.3,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Zai Org — GLM 4.7 Fp8
    "zai-org/GLM-4.7": {
        "family": "glm-4",
        "context": 202752,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.45,
        "pricing_per_1m_output": 2.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Zai Org",
    },
    # Moonshot AI — Kimi K2.5
    # Qwen — Qwen3 Coder Next Fp8
    "Qwen/Qwen3-Coder-Next-FP8": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.5,
        "pricing_per_1m_output": 1.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3.6 Plus
    "Qwen/Qwen3.6-Plus": {
        "family": "qwen-3",
        "context": 1000000,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.5,
        "pricing_per_1m_output": 3.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3-VL-32B-Instruct
    "Qwen/Qwen3-VL-32B-Instruct": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.5,
        "pricing_per_1m_output": 1.5,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3.5 397B A17b
    "Qwen/Qwen3.5-397B-A17B": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.6,
        "pricing_per_1m_output": 3.6,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # DeepSeek — Deepseek V3.1
    "deepseek-ai/DeepSeek-V3.1": {
        "family": "deepseek-v3",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.6,
        "pricing_per_1m_output": 1.7,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "DeepSeek",
    },
    # Nousresearch — Nous Hermes 2 Mixtral 8X7B Dpo
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {
        "family": "mistral",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.6,
        "pricing_per_1m_output": 0.6,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Nousresearch",
    },
    # Zai Org — GLM 4.6 Fp8
    "zai-org/GLM-4.6": {
        "family": "glm-4",
        "context": 202752,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.6,
        "pricing_per_1m_output": 2.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Zai Org",
    },
    # mistralai — Mixtral-8x7B Instruct v0.1
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "family": "mistral",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.6,
        "pricing_per_1m_output": 0.6,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "mistralai",
    },
    # Qwen — Qwen3 235B A22B Thinking 2507 FP8
    # Deepseek — Deepseek Coder 33B Instruct
    "deepseek-ai/deepseek-coder-33b-instruct": {
        "family": "deepseek",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.8,
        "pricing_per_1m_output": 0.8,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Deepseek",
    },
    # Qwen — Qwen 2.5 Coder 32B Instruct
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "family": "qwen",
        "context": 16384,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.8,
        "pricing_per_1m_output": 0.8,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen 2.5 14B Instruct
    "Qwen/Qwen2.5-14B-Instruct": {
        "family": "qwen",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.8,
        "pricing_per_1m_output": 0.8,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Google — Gemma-2 Instruct (27B)
    "google/gemma-2-27b-it": {
        "family": "gemma",
        "context": 8192,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.8,
        "pricing_per_1m_output": 0.8,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Google",
    },
    # Meta — Meta Llama 3.3 70B Instruct Turbo
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 0.88,
        "pricing_per_1m_output": 0.88,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Meta — Meta Llama 3.1 70B Instruct Turbo
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.88,
        "pricing_per_1m_output": 0.88,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # Meta — Meta Llama 3 70B Instruct Turbo
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": {
        "family": "llama",
        "context": 8192,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.88,
        "pricing_per_1m_output": 0.88,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
    # nvidia — Llama 3.1 Nemotron 70B Instruct HF
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": {
        "family": "llama",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 0.88,
        "pricing_per_1m_output": 0.88,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "nvidia",
    },
    # Zai Org — GLM 5 Fp4
    "zai-org/GLM-5": {
        "family": "glm-5",
        "context": 202752,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 1.0,
        "pricing_per_1m_output": 3.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Zai Org",
    },
    # Moonshot AI — Kimi K2.6 Fp4
    "moonshotai/Kimi-K2.6": {
        "family": "kimi",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 1.2,
        "pricing_per_1m_output": 4.5,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Moonshot AI",
    },
    # Moonshot AI — Kimi K2 Thinking
    # Qwen — Qwen QwQ-32B
    "Qwen/QwQ-32B": {
        "family": "qwen",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 1.2,
        "pricing_per_1m_output": 1.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen2.5 72B Instruct
    "Qwen/Qwen2.5-72B-Instruct": {
        "family": "qwen",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 1.2,
        "pricing_per_1m_output": 1.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen2.5 72B Instruct Turbo
    "Qwen/Qwen2.5-72B-Instruct-Turbo": {
        "family": "qwen",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 1.2,
        "pricing_per_1m_output": 1.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen2-VL (72B) Instruct
    "Qwen/Qwen2-VL-72B-Instruct": {
        "family": "qwen-vl",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 1.2,
        "pricing_per_1m_output": 1.2,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Deepcogito — Cogito v2.1 671B
    "deepcogito/cogito-v2-1-671b": {
        "family": "cogito",
        "context": 163840,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 1.25,
        "pricing_per_1m_output": 1.25,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Deepcogito",
    },
    # Deepseek — Deepseek V3 0324
    # Zai Org — GLM 5.1 FP4
    "zai-org/GLM-5.1": {
        "family": "glm-5",
        "context": 202752,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 1.4,
        "pricing_per_1m_output": 4.4,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Zai Org",
    },
    # DeepSeek — DeepSeek R1 Distill Qwen 14B
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "family": "deepseek-r1",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 1.6,
        "pricing_per_1m_output": 1.6,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "DeepSeek",
    },
    # Qwen — Qwen2.5-VL (72B) Instruct
    "Qwen/Qwen2.5-VL-72B-Instruct": {
        "family": "qwen-vl",
        "context": 32768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 1.95,
        "pricing_per_1m_output": 8.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # Qwen — Qwen3 Coder 480B A35B Instruct Fp8
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": {
        "family": "qwen-3",
        "context": 262144,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 2.0,
        "pricing_per_1m_output": 2.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Qwen",
    },
    # DeepSeek — DeepSeek R1 Distill Llama 70B
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
        "family": "llama",
        "context": 131072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 2.0,
        "pricing_per_1m_output": 2.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "DeepSeek",
    },
    # Deepseek AI — Deepseek V4 Pro
    "deepseek-ai/DeepSeek-V4-Pro": {
        "family": "deepseek-v4",
        "context": 512000,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 2.1,
        "pricing_per_1m_output": 4.4,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Deepseek AI",
    },
    # DeepSeek — DeepSeek R1-0528
    # Deepseek — DeepSeek R1 0528
    "deepseek-ai/DeepSeek-R1-0528": {
        "family": "deepseek-r1",
        "context": 163840,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "serverless": True,
        "pricing_per_1m_input": 3.0,
        "pricing_per_1m_output": 7.0,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Deepseek",
    },
    # Meta — Meta Llama 3.1 405B Instruct
    "meta-llama/Llama-3.1-405B-Instruct": {
        "family": "llama",
        "context": 4096,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "serverless": False,
        "pricing_per_1m_input": 3.5,
        "pricing_per_1m_output": 3.5,
        "rpm": 100,
        "tpm": 100_000,
        "active": True,
        "modality": "chat",
        "organization": "Meta",
    },
}

# ---------------------------------------------------------------------------
# Language models (base / completion, 13 total as of 2026-04-28)
# These are NOT instruction-tuned and don't support chat completions.
# ---------------------------------------------------------------------------
TOGETHER_LANGUAGE_MODELS: dict[str, dict] = {
    "Qwen/Qwen3-4B-Base": {"family": "qwen-3", "context": 32768, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Qwen"},
    "Qwen/Qwen2-1.5B": {"family": "qwen", "context": 32768, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Qwen"},
    "deepseek-ai/DeepSeek-V3-Base": {"family": "deepseek-v3", "context": 0, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Deepseek"},
    "Qwen/Qwen2-72B": {"family": "qwen", "context": 32768, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Qwen"},
    "Qwen/Qwen2-7B": {"family": "qwen", "context": 32768, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Qwen"},
    "Qwen/Qwen3-0.6B-Base": {"family": "qwen-3", "context": 32768, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Qwen"},
    "Qwen/Qwen3-1.7B-Base": {"family": "qwen-3", "context": 32768, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Qwen"},
    "Qwen/Qwen3-14B-Base": {"family": "qwen-3", "context": 32768, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Qwen"},
    "Qwen/Qwen3-30B-A3B-Base": {"family": "qwen-3", "context": 32768, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Qwen"},
    "Qwen/Qwen3-8B-Base": {"family": "qwen-3", "context": 32768, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Qwen"},
    "meta-llama/Llama-3.2-3B": {"family": "llama", "context": 131072, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "Meta"},
    "mistralai/Mixtral-8x7B-v0.1": {"family": "mistral", "context": 32768, "pricing_per_1m_input": 0.0, "pricing_per_1m_output": 0.0, "modality": "language", "organization": "mistralai"},
    "meta-llama/Meta-Llama-3.1-8B": {"family": "llama", "context": 16384, "pricing_per_1m_input": 0.2, "pricing_per_1m_output": 0.2, "modality": "language", "organization": "Meta"},
}

# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------
TOGETHER_EMBEDDING_MODELS: dict[str, dict] = {
    "intfloat/multilingual-e5-large-instruct": {"context": 514, "pricing_per_1m_input": 0.02, "modality": "embedding", "organization": "Intfloat"},
}

# ---------------------------------------------------------------------------
# Defaults and convenience helpers
# ---------------------------------------------------------------------------

# Default model: cheapest confirmed-serverless model with tool support
TOGETHER_DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

# Serverless-accessible (no dedicated endpoint needed)
TOGETHER_SERVERLESS_MODELS = {
    k: v for k, v in TOGETHER_MODELS.items() if v.get("serverless", False)
}

# Chat-capable models (all in TOGETHER_MODELS)
TOGETHER_CHAT_MODELS = TOGETHER_MODELS


def available_models() -> list[str]:
    """Return all registered Together chat model IDs."""
    return list(TOGETHER_MODELS.keys())


def serverless_models() -> list[str]:
    """Return chat model IDs accessible without a dedicated endpoint."""
    return list(TOGETHER_SERVERLESS_MODELS.keys())


def chat_models() -> list[str]:
    """Return model IDs usable via chat completions."""
    return list(TOGETHER_CHAT_MODELS.keys())


def tool_capable_models() -> list[str]:
    """Return chat models that support native function calling."""
    return [k for k, v in TOGETHER_CHAT_MODELS.items() if v.get("supports_native_tools")]


def model_info(model_name: str) -> dict:
    """Return registry entry for *model_name*, or raise KeyError."""
    if model_name in TOGETHER_MODELS:
        return TOGETHER_MODELS[model_name]
    if model_name in TOGETHER_LANGUAGE_MODELS:
        return TOGETHER_LANGUAGE_MODELS[model_name]
    raise KeyError(
        f"Unknown Together model {model_name!r}. "
        f"Available chat models: {available_models()}"
    )


def pricing_table() -> list[dict]:
    """Return pricing details for all registered chat models, sorted by input price."""
    rows = []
    for mid, info in TOGETHER_MODELS.items():
        rows.append({
            "model_id": mid,
            "organization": info.get("organization", ""),
            "family": info.get("family", ""),
            "context_length": info.get("context", 0),
            "input_per_1m_usd": info.get("pricing_per_1m_input", 0),
            "output_per_1m_usd": info.get("pricing_per_1m_output", 0),
            "supports_tools": info.get("supports_native_tools", False),
            "serverless": info.get("serverless", False),
        })
    rows.sort(key=lambda r: r["input_per_1m_usd"])
    return rows


def refresh_models(api_key: str | None = None) -> dict[str, Any]:
    """Fetch the live Together AI model catalog and compare against the bundled registry.

    Prints a drift summary so users know if the bundled registry is stale.
    Does NOT mutate :data:`TOGETHER_MODELS` — the bundled registry is the
    offline fallback.  Returns a dict with the comparison results.

    Args:
        api_key: Together API key. Falls back to ``TOGETHER_API_KEY`` env var.

    Returns:
        Dict with keys:
            ``live_count``   — total models returned by the live API.
            ``new_models``   — list of chat model IDs present live but not in bundle.
            ``removed_models`` — list of chat model IDs in bundle but not live.
            ``pricing_changes`` — list of dicts describing pricing drift.
            ``registry_fetch_date`` — date of the bundled snapshot.

    Example::

        from effgen.models.together_models import refresh_models
        drift = refresh_models()
        if drift["new_models"]:
            print("New models available:", drift["new_models"])
    """
    key = api_key or os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise ValueError(
            "Together API key not found. Set TOGETHER_API_KEY or pass api_key=."
        )

    try:
        from together import Together
    except ImportError as exc:
        raise RuntimeError(
            "together SDK not installed. Install with: pip install 'effgen[together]'"
        ) from exc

    client = Together(api_key=key)
    live_all = list(client.models.list())
    live_chat = {m.id: m for m in live_all if getattr(m, "type", "") == "chat"}

    bundled_ids = set(TOGETHER_MODELS.keys())
    live_ids = set(live_chat.keys())

    new_models = sorted(live_ids - bundled_ids)
    removed_models = sorted(bundled_ids - live_ids)

    pricing_changes: list[dict] = []
    for mid in bundled_ids & live_ids:
        lm = live_chat[mid]
        lp = getattr(lm, "pricing", None)
        if lp is None:
            continue
        bundled_in = TOGETHER_MODELS[mid].get("pricing_per_1m_input", 0)
        bundled_out = TOGETHER_MODELS[mid].get("pricing_per_1m_output", 0)
        live_in = getattr(lp, "input", 0) or 0
        live_out = getattr(lp, "output", 0) or 0
        if abs(bundled_in - live_in) > 0.0001 or abs(bundled_out - live_out) > 0.0001:
            pricing_changes.append({
                "model_id": mid,
                "bundled_input": bundled_in,
                "live_input": live_in,
                "bundled_output": bundled_out,
                "live_output": live_out,
            })

    # Print human-readable drift summary
    total_live = len(live_all)
    if new_models or removed_models or pricing_changes:
        logger.warning(
            "Together registry may be outdated (snapshot: %s). "
            "new=%d, removed=%d, pricing_changes=%d",
            REGISTRY_FETCH_DATE, len(new_models), len(removed_models), len(pricing_changes),
        )
        print(
            f"\n[effgen] Together model registry drift detected "
            f"(bundled snapshot: {REGISTRY_FETCH_DATE}):"
        )
        if new_models:
            print(f"  + {len(new_models)} new model(s) not in bundled registry:")
            for m in new_models[:10]:
                lm = live_chat[m]
                lp = getattr(lm, "pricing", None)
                inp = getattr(lp, "input", 0) if lp else 0
                ctx = getattr(lm, "context_length", 0)
                print(f"      {m}  (ctx={ctx}, in=${inp:.4f}/1M)")
            if len(new_models) > 10:
                print(f"      ... and {len(new_models) - 10} more")
        if removed_models:
            print(f"  - {len(removed_models)} model(s) in registry but no longer live:")
            for m in removed_models[:10]:
                print(f"      {m}")
            if len(removed_models) > 10:
                print(f"      ... and {len(removed_models) - 10} more")
        if pricing_changes:
            print(f"  ~ {len(pricing_changes)} model(s) with pricing changes:")
            for pc in pricing_changes[:5]:
                print(
                    f"      {pc['model_id']}: "
                    f"input ${pc['bundled_input']:.4f} → ${pc['live_input']:.4f}/1M, "
                    f"output ${pc['bundled_output']:.4f} → ${pc['live_output']:.4f}/1M"
                )
        print(
            "  Run effgen.models.together_models.refresh_models() to see full diff.\n"
            "  The bundled registry is still used as the offline fallback.\n"
        )
    else:
        print(
            f"[effgen] Together registry is up-to-date "
            f"({len(bundled_ids)} chat models match live API, snapshot: {REGISTRY_FETCH_DATE})"
        )

    return {
        "live_count": total_live,
        "live_chat_count": len(live_chat),
        "bundled_chat_count": len(bundled_ids),
        "new_models": new_models,
        "removed_models": removed_models,
        "pricing_changes": pricing_changes,
        "registry_fetch_date": REGISTRY_FETCH_DATE,
    }

