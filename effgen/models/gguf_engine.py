"""GGUF model engine via llama-cpp-python (optional).

Loads GGUF-quantized models (Q2_K, Q4_K_M, Q5_K_M, Q8_0, ...) on CPU or GPU.
``llama-cpp-python`` is an optional dependency; importing this module without
it raises a clear ImportError on engine instantiation only.

Install:
    pip install llama-cpp-python                # CPU
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python  # CUDA
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

from ._adapter_utils import provider_runtime_error
from .base import BaseModel, GenerationConfig, GenerationResult, ModelType, TokenCount

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "GGUF support requires the optional 'llama-cpp-python' package.\n"
    "Install with:\n"
    "    pip install llama-cpp-python\n"
    "  or with CUDA:\n"
    "    CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python"
)


def is_gguf_path(path: str) -> bool:
    """Return True if *path* looks like a GGUF model file."""
    return isinstance(path, str) and path.lower().endswith(".gguf")


class GGUFEngine(BaseModel):
    """llama.cpp-backed engine for GGUF-quantized models."""

    def __init__(
        self,
        model_name: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type=ModelType.TRANSFORMERS,  # closest existing enum value
            context_length=n_ctx,
        )
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.verbose = verbose
        self._extra = kwargs
        self._llm: Any = None

    # ------------------------------------------------------------------ load/unload
    def load(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        if not os.path.exists(self.model_name):
            raise FileNotFoundError(f"GGUF model file not found: {self.model_name}")

        logger.info("Loading GGUF model: %s", self.model_name)
        try:
            self._llm = Llama(
                model_path=self.model_name,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose,
                **self._extra,
            )
        except Exception as exc:
            raise provider_runtime_error(
                "gguf", self.model_name, "load", exc,
                message="GGUF model loading failed",
            ) from exc
        self._is_loaded = True
        self._metadata = {
            "engine": "llama-cpp-python",
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
        }

    def unload(self) -> None:
        self._llm = None
        self._is_loaded = False

    # ------------------------------------------------------------------ inference
    def _to_kwargs(self, config: GenerationConfig | None) -> dict[str, Any]:
        cfg = config or GenerationConfig()
        # Normalize deterministic generation: temperature<=0 means greedy. llama.cpp
        # treats temperature=0 as greedy; clamp negatives to 0 so the effGen config
        # behaves consistently with the other backends.
        greedy = cfg.temperature is None or cfg.temperature <= 0
        return {
            "temperature": 0.0 if greedy else cfg.temperature,
            "top_p": 1.0 if greedy else cfg.top_p,
            "top_k": -1 if greedy else cfg.top_k,
            "max_tokens": cfg.max_tokens or 256,
            "stop": cfg.stop_sequences or None,
            "repeat_penalty": cfg.repetition_penalty,
            "seed": cfg.seed if cfg.seed is not None else -1,
        }

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        if not self._is_loaded:
            self.load()
        params = self._to_kwargs(config)
        params.update(kwargs)
        try:
            # Clear residual context/KV state from any prior call so each
            # completion starts from a clean state. Without this the sampler's
            # RNG continues from the previous call and a fixed seed does not
            # reproduce.
            self._llm.reset()
            out = self._llm(prompt, **params)
        except Exception as exc:
            raise provider_runtime_error(
                "gguf", self.model_name, "generate", exc,
                message="GGUF generation failed",
            ) from exc
        choice = out["choices"][0]
        text = choice.get("text", "")
        usage = out.get("usage", {}) or {}
        return GenerationResult(
            text=text,
            tokens_used=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason", "stop") or "stop",
            model_name=self.model_name,
            metadata={"prompt_tokens": usage.get("prompt_tokens", 0)},
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if not self._is_loaded:
            self.load()
        params = self._to_kwargs(config)
        params.update(kwargs)
        params["stream"] = True
        try:
            self._llm.reset()
            for chunk in self._llm(prompt, **params):
                yield chunk["choices"][0].get("text", "")
        except Exception as exc:
            raise provider_runtime_error(
                "gguf", self.model_name, "generate_stream", exc,
                message="GGUF streaming generation failed",
            ) from exc

    def count_tokens(self, text: str) -> TokenCount:
        if not self._is_loaded:
            self.load()
        try:
            tokens = self._llm.tokenize(text.encode("utf-8"))
        except Exception as exc:
            raise provider_runtime_error(
                "gguf", self.model_name, "count_tokens", exc,
                message="GGUF token counting failed",
            ) from exc
        return TokenCount(count=len(tokens), model_name=self.model_name)

    def get_context_length(self) -> int:
        return self.n_ctx
