"""
Fireworks AI model registry for effGen.

Full model catalog fetched from the Fireworks API on 2026-04-28 (206 active
non-deprecated models with context > 0; 30 deprecated excluded).

Rate limits: free tier ~ 1 request / 6s per model; paid tiers higher.
Pricing from https://fireworks.ai/pricing (2026-04-28):
  - <4B params:        $0.10/1M tokens (in+out blended)
  - 4B-16B params:     $0.20/1M tokens
  - >16B params:       $0.90/1M tokens
  - MoE 0-56B:         $0.50/1M tokens
  - MoE 56-176B:       $1.20/1M tokens
  - DeepSeek-V4-Pro:   $1.74/$3.48 per 1M in/out
  - DeepSeek V3:       $0.56/$1.68 per 1M in/out
  - Kimi K2.6:         $0.95/$4.00 per 1M in/out
  - GLM-5:             $1.00/$3.20 per 1M in/out
  - GLM-5.1:           $1.40/$4.40 per 1M in/out

Model IDs must be given in full format: accounts/fireworks/models/<id>

REGISTRY_FETCH_DATE: str = "2026-04-28"
Use refresh_models() to fetch the live catalog and detect drift vs this snapshot.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

REGISTRY_FETCH_DATE = "2026-04-28"
_FIREWORKS_PREFIX = "accounts/fireworks/models/"

# ---------------------------------------------------------------------------
# Pricing helpers: tiered by parameter count
# ---------------------------------------------------------------------------
def _p(params_b: float, moe: bool = False) -> tuple[float, float]:
    """Return (input_per_1m, output_per_1m) based on parameter count."""
    if moe:
        return (1.20, 1.20) if params_b > 56 else (0.50, 0.50)
    if params_b < 4:
        return (0.10, 0.10)
    if params_b <= 16:
        return (0.20, 0.20)
    return (0.90, 0.90)


# ---------------------------------------------------------------------------
# Main registry — keyed by SHORT model ID (without accounts/fireworks/models/)
# ---------------------------------------------------------------------------
_REGISTRY_SHORT: dict[str, dict] = {
    # -------------------------------------------------------------------------
    # DeepSeek family
    # -------------------------------------------------------------------------
    "deepseek-v4-pro": {
        "display_name": "DeepSeek-V4-Pro",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 1_048_576,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 1.74,
        "pricing_per_1m_output": 3.48,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-v3p1": {
        "display_name": "DeepSeek V3.1",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 163_840,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.56,
        "pricing_per_1m_output": 1.68,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-v3p2": {
        "display_name": "DeepSeek v3.2",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 163_840,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.56,
        "pricing_per_1m_output": 1.68,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-v2-lite-chat": {
        "display_name": "DeepSeek V2 Lite Chat",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 163_840,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-v2p5": {
        "display_name": "DeepSeek V2.5",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.50,
        "pricing_per_1m_output": 0.50,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-r1-0528-distill-qwen3-8b": {
        "display_name": "DeepSeek R1 0528 Distill Qwen3 8B",
        "family": "deepseek-r1",
        "organization": "DeepSeek",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-r1-distill-llama-70b": {
        "display_name": "DeepSeek R1 Distill Llama 70B",
        "family": "deepseek-r1",
        "organization": "DeepSeek",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-r1-distill-llama-8b": {
        "display_name": "DeepSeek R1 Distill Llama 8B",
        "family": "deepseek-r1",
        "organization": "DeepSeek",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-r1-distill-qwen-32b": {
        "display_name": "DeepSeek R1 Distill Qwen 32B",
        "family": "deepseek-r1",
        "organization": "DeepSeek",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-r1-distill-qwen-14b": {
        "display_name": "DeepSeek R1 Distill Qwen 14B",
        "family": "deepseek-r1",
        "organization": "DeepSeek",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-r1-distill-qwen-7b": {
        "display_name": "DeepSeek R1 Distill Qwen 7B",
        "family": "deepseek-r1",
        "organization": "DeepSeek",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-r1-distill-qwen-1p5b": {
        "display_name": "DeepSeek R1 Distill Qwen 1.5B",
        "family": "deepseek-r1",
        "organization": "DeepSeek",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-prover-v2": {
        "display_name": "DeepSeek Prover V2",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 163_840,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-coder-v2-lite-instruct": {
        "display_name": "DeepSeek Coder V2 Lite Instruct",
        "family": "deepseek-coder",
        "organization": "DeepSeek",
        "context": 163_840,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "deepseek-coder-33b-instruct": {
        "display_name": "DeepSeek Coder 33B Instruct",
        "family": "deepseek-coder",
        "organization": "DeepSeek",
        "context": 16_384,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # Llama family
    # -------------------------------------------------------------------------
    "llama-v3p3-70b-instruct": {
        "display_name": "Llama 3.3 70B Instruct",
        "family": "llama",
        "organization": "Meta",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,  # returns JSON text not structured tool_calls
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "llama-v3p2-3b-instruct": {
        "display_name": "Llama 3.2 3B Instruct",
        "family": "llama",
        "organization": "Meta",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": False,  # Llama-on-Fireworks uses JSON text format
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "llama-v3p2-1b-instruct": {
        "display_name": "Llama 3.2 1B Instruct",
        "family": "llama",
        "organization": "Meta",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": False,  # Llama-on-Fireworks uses JSON text format
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "llama-v3p2-11b-vision-instruct": {
        "display_name": "Llama 3.2 11B Vision Instruct",
        "family": "llama",
        "organization": "Meta",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": False,  # Llama-on-Fireworks uses JSON text format
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "vision",
    },
    "llama-v3p1-nemotron-70b-instruct": {
        "display_name": "Llama 3.1 Nemotron 70B",
        "family": "llama",
        "organization": "NVIDIA",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,  # Llama-on-Fireworks uses JSON text format
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "llama-v3-70b-instruct": {
        "display_name": "Llama 3 70B Instruct",
        "family": "llama",
        "organization": "Meta",
        "context": 8_192,
        "max_output": 4_096,
        "supports_native_tools": False,  # Llama-on-Fireworks uses JSON text format
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "llama-v3-8b-instruct": {
        "display_name": "Llama 3 8B Instruct",
        "family": "llama",
        "organization": "Meta",
        "context": 8_192,
        "max_output": 4_096,
        "supports_native_tools": False,  # Llama-on-Fireworks uses JSON text format
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "cogito-v1-preview-llama-70b": {
        "display_name": "Cogito v1 Preview Llama 70B",
        "family": "llama",
        "organization": "DeepCogito",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,  # Llama-based; Fireworks uses JSON text format
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "cogito-v1-preview-llama-8b": {
        "display_name": "Cogito v1 Preview Llama 8B",
        "family": "llama",
        "organization": "DeepCogito",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,  # Llama-based; Fireworks uses JSON text format
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "cogito-v1-preview-llama-3b": {
        "display_name": "Cogito v1 Preview Llama 3B",
        "family": "llama",
        "organization": "DeepCogito",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": False,  # Llama-based; Fireworks uses JSON text format
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # Qwen3 family
    # -------------------------------------------------------------------------
    "qwen3-8b": {
        "display_name": "Qwen3 8B",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 40_960,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen3-4b": {
        "display_name": "Qwen3 4B",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 40_960,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen3-14b": {
        "display_name": "Qwen3 14B",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 40_960,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen3-32b": {
        "display_name": "Qwen3 32B",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen3-1p7b": {
        "display_name": "Qwen3 1.7B",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen3-0p6b": {
        "display_name": "Qwen3 0.6B",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 40_960,
        "max_output": 2_048,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen3-4b-instruct-2507": {
        "display_name": "Qwen3 4B Instruct 2507",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 262_144,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "cogito-v1-preview-qwen-32b": {
        "display_name": "Cogito v1 Preview Qwen 32B",
        "family": "qwen3",
        "organization": "DeepCogito",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # Qwen2.5 family
    # -------------------------------------------------------------------------
    "qwen2p5-72b-instruct": {
        "display_name": "Qwen2.5 72B Instruct",
        "family": "qwen2",
        "organization": "Alibaba",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen2p5-32b-instruct": {
        "display_name": "Qwen2.5 32B Instruct",
        "family": "qwen2",
        "organization": "Alibaba",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen2p5-14b-instruct": {
        "display_name": "Qwen2.5 14B Instruct",
        "family": "qwen2",
        "organization": "Alibaba",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen2p5-7b-instruct": {
        "display_name": "Qwen2.5 7B Instruct",
        "family": "qwen2",
        "organization": "Alibaba",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen2p5-1p5b-instruct": {
        "display_name": "Qwen2.5 1.5B Instruct",
        "family": "qwen2",
        "organization": "Alibaba",
        "context": 32_768,
        "max_output": 2_048,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen2p5-72b": {
        "display_name": "Qwen2.5 72B (base)",
        "family": "qwen2",
        "organization": "Alibaba",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen2p5-coder-32b-instruct": {
        "display_name": "Qwen2.5-Coder 32B Instruct",
        "family": "qwen2-coder",
        "organization": "Alibaba",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen2p5-coder-32b-instruct-128k": {
        "display_name": "Qwen2.5-Coder 32B Instruct 128K",
        "family": "qwen2-coder",
        "organization": "Alibaba",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen2p5-coder-7b-instruct": {
        "display_name": "Qwen2.5-Coder 7B Instruct",
        "family": "qwen2-coder",
        "organization": "Alibaba",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwq-32b": {
        "display_name": "QWQ 32B",
        "family": "qwen2",
        "organization": "Alibaba",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen2-72b-instruct": {
        "display_name": "Qwen2 72B Instruct",
        "family": "qwen2",
        "organization": "Alibaba",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen2-7b-instruct": {
        "display_name": "Qwen2 7B Instruct",
        "family": "qwen2",
        "organization": "Alibaba",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # Qwen3.5 family
    # -------------------------------------------------------------------------
    "qwen3p5-122b-a10b": {
        "display_name": "Qwen3.5 122B A10B",
        "family": "qwen3.5",
        "organization": "Alibaba",
        "context": 262_144,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 1.20,
        "pricing_per_1m_output": 1.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen3p5-27b": {
        "display_name": "Qwen3.5 27B",
        "family": "qwen3.5",
        "organization": "Alibaba",
        "context": 262_144,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen3p5-9b": {
        "display_name": "Qwen3.5 9B",
        "family": "qwen3.5",
        "organization": "Alibaba",
        "context": 262_144,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen3p5-35b-a3b": {
        "display_name": "Qwen3.5 35B A3B",
        "family": "qwen3.5",
        "organization": "Alibaba",
        "context": 262_144,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.50,
        "pricing_per_1m_output": 0.50,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "qwen3p5-397b-a17b": {
        "display_name": "Qwen3.5 397B A17B",
        "family": "qwen3.5",
        "organization": "Alibaba",
        "context": 262_144,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 1.20,
        "pricing_per_1m_output": 1.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # Kimi family
    # -------------------------------------------------------------------------
    "kimi-k2p5": {
        "display_name": "Kimi K2.5",
        "family": "kimi",
        "organization": "Moonshot AI",
        "context": 262_144,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 4.00,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "kimi-k2p6": {
        "display_name": "Kimi K2.6",
        "family": "kimi",
        "organization": "Moonshot AI",
        "context": 262_144,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.95,
        "pricing_per_1m_output": 4.00,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # GLM family
    # -------------------------------------------------------------------------
    "glm-5": {
        "display_name": "GLM-5",
        "family": "glm",
        "organization": "Zhipu AI",
        "context": 202_752,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 1.00,
        "pricing_per_1m_output": 3.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "glm-5p1": {
        "display_name": "GLM 5.1",
        "family": "glm",
        "organization": "Zhipu AI",
        "context": 202_752,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 1.40,
        "pricing_per_1m_output": 4.40,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "glm-4p7": {
        "display_name": "GLM-4.7",
        "family": "glm",
        "organization": "Zhipu AI",
        "context": 202_752,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.50,
        "pricing_per_1m_output": 0.50,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "glm-4p7-flash": {
        "display_name": "GLM-4.7 Flash",
        "family": "glm",
        "organization": "Zhipu AI",
        "context": 202_752,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # MiniMax family
    # -------------------------------------------------------------------------
    "minimax-m2p7": {
        "display_name": "MiniMax M2.7",
        "family": "minimax",
        "organization": "MiniMax",
        "context": 196_608,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.30,
        "pricing_per_1m_output": 0.30,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "minimax-m2p5": {
        "display_name": "MiniMax-M2.5",
        "family": "minimax",
        "organization": "MiniMax",
        "context": 196_608,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # Mixtral / Mistral family
    # -------------------------------------------------------------------------
    "mixtral-8x7b-instruct": {
        "display_name": "Mixtral MoE 8x7B Instruct",
        "family": "mixtral",
        "organization": "Mistral AI",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.50,
        "pricing_per_1m_output": 0.50,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "mixtral-8x7b-instruct-hf": {
        "display_name": "Mixtral MoE 8x7B Instruct (HF version)",
        "family": "mixtral",
        "organization": "Mistral AI",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.50,
        "pricing_per_1m_output": 0.50,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "mixtral-8x22b": {
        "display_name": "Mixtral MoE 8x22B",
        "family": "mixtral",
        "organization": "Mistral AI",
        "context": 65_536,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 1.20,
        "pricing_per_1m_output": 1.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "mistral-7b-instruct-v3": {
        "display_name": "Mistral 7B Instruct v0.3",
        "family": "mistral",
        "organization": "Mistral AI",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "mistral-nemo-instruct-2407": {
        "display_name": "Mistral Nemo Instruct 2407",
        "family": "mistral",
        "organization": "Mistral AI",
        "context": 128_000,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "mistral-small-24b-instruct-2501": {
        "display_name": "Mistral Small 24B Instruct 2501",
        "family": "mistral",
        "organization": "Mistral AI",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "ministral-3-8b-instruct-2512": {
        "display_name": "Ministral 3 8B Instruct 2512",
        "family": "mistral",
        "organization": "Mistral AI",
        "context": 256_000,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "ministral-3-3b-instruct-2512": {
        "display_name": "Ministral 3 3B Instruct 2512",
        "family": "mistral",
        "organization": "Mistral AI",
        "context": 256_000,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "devstral-small-2505": {
        "display_name": "Devstral Small 2505",
        "family": "mistral",
        "organization": "Mistral AI",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # Gemma family
    # -------------------------------------------------------------------------
    "gemma2-9b-it": {
        "display_name": "Gemma 2 9B Instruct",
        "family": "gemma",
        "organization": "Google",
        "context": 8_192,
        "max_output": 4_096,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "gemma-3-27b-it": {
        "display_name": "Gemma 3 27B Instruct",
        "family": "gemma",
        "organization": "Google",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "gemma-3-12b-it": {
        "display_name": "Gemma 3 12B Instruct",
        "family": "gemma",
        "organization": "Google",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "gemma-3-4b-it": {
        "display_name": "Gemma 3 4B Instruct",
        "family": "gemma",
        "organization": "Google",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.10,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "gemma-4-26b-a4b-it": {
        "display_name": "Gemma 4 26B A4B IT",
        "family": "gemma",
        "organization": "Google",
        "context": 262_144,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.50,
        "pricing_per_1m_output": 0.50,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "gemma-4-31b-it": {
        "display_name": "Gemma 4 31B IT",
        "family": "gemma",
        "organization": "Google",
        "context": 262_144,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # NVIDIA Nemotron family
    # -------------------------------------------------------------------------
    "nvidia-nemotron-nano-12b-v2": {
        "display_name": "NVIDIA Nemotron Nano 12B v2",
        "family": "nemotron",
        "organization": "NVIDIA",
        "context": 128_000,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.20,
        "pricing_per_1m_output": 0.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "nvidia-nemotron-3-nano-omni-30b-a3b": {
        "display_name": "NVIDIA Nemotron Nano Omni 30B A3B",
        "family": "nemotron",
        "organization": "NVIDIA",
        "context": 262_144,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.50,
        "pricing_per_1m_output": 0.50,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # OpenAI OSS (gpt-oss) family
    # -------------------------------------------------------------------------
    "gpt-oss-120b": {
        "display_name": "OpenAI gpt-oss-120b",
        "family": "gpt-oss",
        "organization": "OpenAI",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "gpt-oss-20b": {
        "display_name": "OpenAI gpt-oss-20b",
        "family": "gpt-oss",
        "organization": "OpenAI",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.07,
        "pricing_per_1m_output": 0.07,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    # -------------------------------------------------------------------------
    # Seed / misc families
    # -------------------------------------------------------------------------
    "seed-oss-36b-instruct": {
        "display_name": "Seed OSS 36B Instruct",
        "family": "seed",
        "organization": "ByteDance",
        "context": 524_288,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "firefunction-v1": {
        "display_name": "FireFunction V1",
        "family": "firefunction",
        "organization": "Fireworks",
        "context": 32_768,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
    "dolphin-2-9-2-qwen2-72b": {
        "display_name": "Dolphin 2.9.2 Qwen2 72B",
        "family": "dolphin",
        "organization": "Cognitive Computations",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.90,
        "pricing_per_1m_output": 0.90,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
    },
}

# Build the full registry with accounts/fireworks/models/ prefix
FIREWORKS_MODELS: dict[str, dict] = {
    f"{_FIREWORKS_PREFIX}{k}": v for k, v in _REGISTRY_SHORT.items()
}

FIREWORKS_DEFAULT_MODEL = f"{_FIREWORKS_PREFIX}llama-v3p3-70b-instruct"

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def available_models() -> list[str]:
    """Return all model IDs in the registry (full path format)."""
    return sorted(FIREWORKS_MODELS.keys())


def chat_models() -> list[str]:
    """Return IDs of chat/text models (excludes vision-only etc.)."""
    return [
        mid for mid, info in FIREWORKS_MODELS.items()
        if info.get("modality", "chat") in ("chat", "vision")
    ]


def tool_capable_models() -> list[str]:
    """Return IDs of models that support native function calling."""
    return [
        mid for mid, info in FIREWORKS_MODELS.items()
        if info.get("supports_native_tools", False)
    ]


def streaming_models() -> list[str]:
    """Return IDs of models that support streaming."""
    return [
        mid for mid, info in FIREWORKS_MODELS.items()
        if info.get("supports_streaming", True)
    ]


def pricing_table() -> list[dict]:
    """Return a list of dicts with model pricing info, sorted by input cost."""
    rows = []
    for mid, info in FIREWORKS_MODELS.items():
        rows.append({
            "model": mid,
            "display_name": info.get("display_name", ""),
            "family": info.get("family", ""),
            "context": info.get("context", 0),
            "input_per_1m_usd": info.get("pricing_per_1m_input", 0),
            "output_per_1m_usd": info.get("pricing_per_1m_output", 0),
            "supports_tools": info.get("supports_native_tools", False),
        })
    return sorted(rows, key=lambda r: r["input_per_1m_usd"])


# ---------------------------------------------------------------------------
# Live catalog refresh + drift detection
# ---------------------------------------------------------------------------

def refresh_models(
    api_key: str | None = None,
    *,
    warn_on_drift: bool = True,
) -> dict[str, Any]:
    """Fetch the live Fireworks model catalog and compare against the bundled registry.

    Uses the ``/v1/accounts/fireworks/models`` API endpoint (no cost, no tokens).

    Args:
        api_key: Fireworks API key. Reads ``FIREWORKS_API_KEY`` env var if omitted.
        warn_on_drift: Emit :mod:`logging` warnings when the live catalog differs
            from the bundled snapshot.

    Returns:
        A summary dict::

            {
                "fetch_date": "2026-04-28",
                "live_total": int,
                "bundled_total": int,
                "new_models": [str, ...],      # in live but not in bundled
                "removed_models": [str, ...],  # in bundled but not in live
                "live_models": {<id>: {...}},  # full live catalog
            }
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for refresh_models(). "
            "Install with: pip install requests"
        ) from exc

    key = api_key or os.getenv("FIREWORKS_API_KEY")
    if not key:
        raise ValueError("FIREWORKS_API_KEY not set; cannot refresh model catalog.")

    # Paginate through all models
    live_raw: list[dict] = []
    page_token: str | None = None
    while True:
        url = "https://api.fireworks.ai/v1/accounts/fireworks/models?pageSize=500"
        if page_token:
            url += f"&pageToken={page_token}"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Fireworks catalog fetch failed: HTTP {resp.status_code} — {resp.text[:200]}"
            )
        data = resp.json()
        live_raw.extend(data.get("models", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    # Build live dict (only active text models with context > 0)
    live_models: dict[str, dict] = {}
    for m in live_raw:
        if m.get("deprecationDate"):
            continue
        ctx = m.get("contextLength", 0)
        if ctx <= 0:
            continue
        name = m.get("name", "")
        if not name.startswith(_FIREWORKS_PREFIX):
            continue
        live_models[name] = {
            "display_name": m.get("displayName", ""),
            "context": ctx,
            "model_type": (m.get("baseModelDetails") or {}).get("modelType", ""),
        }

    bundled_ids = set(FIREWORKS_MODELS.keys())
    live_ids = set(live_models.keys())

    new_models = sorted(live_ids - bundled_ids)
    removed_models = sorted(bundled_ids - live_ids)

    if warn_on_drift:
        if new_models:
            logger.warning(
                "Fireworks catalog drift: %d new model(s) not in bundled registry:\n  %s\n"
                "Consider upgrading effGen or calling refresh_models() to get the latest.",
                len(new_models), "\n  ".join(new_models),
            )
        if removed_models:
            logger.warning(
                "Fireworks catalog drift: %d model(s) in bundled registry no longer listed:\n  %s",
                len(removed_models), "\n  ".join(removed_models),
            )

    return {
        "fetch_date": REGISTRY_FETCH_DATE,
        "live_total": len(live_models),
        "bundled_total": len(FIREWORKS_MODELS),
        "new_models": new_models,
        "removed_models": removed_models,
        "live_models": live_models,
    }
