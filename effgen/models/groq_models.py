"""
Groq model registry for effGen.

Rate limits from Groq dashboard (Developer plan free tier):
  https://console.groq.com/settings/limits
Fetch date: 2026-04-28

Context windows from Groq Models API (/openai/v1/models).
"""

from __future__ import annotations

GROQ_MODELS: dict[str, dict] = {
    # -----------------------------------------------------------------------
    # Chat Completions — text models
    # -----------------------------------------------------------------------
    "llama-3.3-70b-versatile": {
        "family": "llama",
        "context": 131_072,
        "max_output": 32_768,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 12_000,
        "tpd": 100_000,
        "active": True,
        "modality": "chat",
        "notes": "Llama 3.3 70B — best quality on free tier",
    },
    "llama-3.1-8b-instant": {
        "family": "llama",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 14_400,
        "tpm": 6_000,
        "tpd": 500_000,
        "active": True,
        "modality": "chat",
        "notes": "Llama 3.1 8B — fastest free-tier model",
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "family": "llama-4",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 30_000,
        "tpd": 500_000,
        "active": True,
        "modality": "chat",
        "notes": "Llama 4 Scout 17B-16E — vision capable, MoE",
    },
    "qwen/qwen3-32b": {
        "family": "qwen",
        "context": 131_072,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 60,
        "rpd": 1_000,
        "tpm": 6_000,
        "tpd": 500_000,
        "active": True,
        "modality": "chat",
        "notes": "Qwen3 32B — highest RPM on free tier (60 RPM)",
    },
    "openai/gpt-oss-120b": {
        "family": "gpt-oss",
        "context": 131_072,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 8_000,
        "tpd": 200_000,
        "active": True,
        "modality": "chat",
        "notes": "OpenAI GPT-OSS 120B open weights",
    },
    "openai/gpt-oss-20b": {
        "family": "gpt-oss",
        "context": 131_072,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 8_000,
        "tpd": 200_000,
        "active": True,
        "modality": "chat",
        "notes": "OpenAI GPT-OSS 20B open weights",
    },
    "openai/gpt-oss-safeguard-20b": {
        "family": "gpt-oss",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 8_000,
        "tpd": 200_000,
        "active": True,
        "modality": "chat",
        "notes": "GPT-OSS 20B safety/guardrail variant",
    },
    "groq/compound": {
        "family": "compound",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 250,
        "tpm": 70_000,
        "tpd": None,
        "active": True,
        "modality": "chat",
        "notes": "Groq Compound (agentic system) — no daily token limit",
    },
    "groq/compound-mini": {
        "family": "compound",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 250,
        "tpm": 70_000,
        "tpd": None,
        "active": True,
        "modality": "chat",
        "notes": "Groq Compound Mini (smaller agentic system)",
    },
    "allam-2-7b": {
        "family": "allam",
        "context": 4_096,
        "max_output": 2_048,
        "supports_native_tools": False,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 7_000,
        "tpm": 6_000,
        "tpd": 500_000,
        "active": True,
        "modality": "chat",
        "notes": "ALLaM 2 7B — Arabic/English bilingual",
    },
    "meta-llama/llama-prompt-guard-2-86m": {
        "family": "llama-guard",
        "context": 512,
        "max_output": 256,
        "supports_native_tools": False,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 14_400,
        "tpm": 15_000,
        "tpd": 500_000,
        "active": True,
        "modality": "chat",
        "notes": "Llama Prompt Guard 2 86M — safety classifier",
    },
    "meta-llama/llama-prompt-guard-2-22m": {
        "family": "llama-guard",
        "context": 512,
        "max_output": 256,
        "supports_native_tools": False,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 14_400,
        "tpm": 15_000,
        "tpd": 500_000,
        "active": True,
        "modality": "chat",
        "notes": "Llama Prompt Guard 2 22M — lightweight safety classifier",
    },
    # -----------------------------------------------------------------------
    # Speech-to-Text models (not usable via chat completions)
    # -----------------------------------------------------------------------
    "whisper-large-v3": {
        "family": "whisper",
        "context": 448,
        "max_output": 448,
        "supports_native_tools": False,
        "supports_streaming": False,
        "rpm": 20,
        "rpd": 2_000,
        "tpm": None,
        "tpd": None,
        "active": True,
        "modality": "stt",
        "notes": "Whisper Large v3 — speech-to-text only, not a chat model",
    },
    "whisper-large-v3-turbo": {
        "family": "whisper",
        "context": 448,
        "max_output": 448,
        "supports_native_tools": False,
        "supports_streaming": False,
        "rpm": 20,
        "rpd": 2_000,
        "tpm": None,
        "tpd": None,
        "active": True,
        "modality": "stt",
        "notes": "Whisper Large v3 Turbo — faster speech-to-text",
    },
    # -----------------------------------------------------------------------
    # Text-to-Speech models
    # -----------------------------------------------------------------------
    "canopylabs/orpheus-v1-english": {
        "family": "orpheus",
        "context": 4_000,
        "max_output": 4_000,
        "supports_native_tools": False,
        "supports_streaming": False,
        "rpm": 10,
        "rpd": 100,
        "tpm": 1_200,
        "tpd": 3_600,
        "active": True,
        "modality": "tts",
        "notes": "Orpheus v1 English — text-to-speech",
    },
    "canopylabs/orpheus-arabic-saudi": {
        "family": "orpheus",
        "context": 4_000,
        "max_output": 4_000,
        "supports_native_tools": False,
        "supports_streaming": False,
        "rpm": 10,
        "rpd": 100,
        "tpm": 1_200,
        "tpd": 3_600,
        "active": True,
        "modality": "tts",
        "notes": "Orpheus Arabic (Saudi) — text-to-speech",
    },
}

# Default model for new GroqAdapter instances
GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"

# Chat-capable model IDs (usable via chat.completions.create)
GROQ_CHAT_MODELS = {
    k: v for k, v in GROQ_MODELS.items()
    if v.get("modality") == "chat"
}


def available_models() -> list[str]:
    """Return all registered Groq model IDs."""
    return list(GROQ_MODELS.keys())


def chat_models() -> list[str]:
    """Return model IDs usable via chat completions."""
    return list(GROQ_CHAT_MODELS.keys())


def tool_capable_models() -> list[str]:
    """Return chat models that support native function calling."""
    return [k for k, v in GROQ_CHAT_MODELS.items() if v.get("supports_native_tools")]


def model_info(model_name: str) -> dict:
    """Return registry entry for *model_name*, or raise ``KeyError``."""
    if model_name not in GROQ_MODELS:
        raise KeyError(
            f"Unknown Groq model '{model_name}'. "
            f"Available: {available_models()}"
        )
    return GROQ_MODELS[model_name]
