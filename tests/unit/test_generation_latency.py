"""Every engine's raw ``generate``/``generate_batch`` result carries call latency.

``BaseModel.__init_subclass__`` auto-instruments concrete engines so a benchmarker
calling ``model.generate(...)`` directly can read ``latency_ms``/``duration_s`` off
``GenerationResult.metadata`` without a manual timer — mirroring what the agent layer
already surfaces. These tests use a tiny in-process engine (no network/model) to
exercise the wrapping mechanism itself.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)


class _TinyEngine(BaseModel):
    """Minimal concrete engine that just echoes — to test latency instrumentation."""

    def __init__(self):
        super().__init__("tiny", ModelType.TRANSFORMERS, context_length=128)
        self._is_loaded = True

    def load(self) -> None:  # pragma: no cover - trivial
        self._is_loaded = True

    def unload(self) -> None:  # pragma: no cover - trivial
        self._is_loaded = False

    def get_context_length(self) -> int:  # pragma: no cover - trivial
        return self._context_length or 128

    def generate(self, prompt: str, config: GenerationConfig | None = None, **kwargs):
        time.sleep(0.005)
        return GenerationResult(
            text=f"echo:{prompt}", tokens_used=1, finish_reason="stop",
            model_name=self.model_name, metadata={"prompt_tokens": 1},
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        yield prompt

    def generate_batch(self, prompts, config=None, **kwargs):
        time.sleep(0.005)
        return [
            GenerationResult(text=f"echo:{p}", tokens_used=1, finish_reason="stop",
                             model_name=self.model_name, metadata={})
            for p in prompts
        ]

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover - trivial
        return TokenCount(count=len(text.split()), model_name=self.model_name)


class _SelfTimedEngine(_TinyEngine):
    """An engine that already records its own latency — must not be overwritten."""

    def generate(self, prompt, config=None, **kwargs):
        return GenerationResult(
            text="x", tokens_used=1, finish_reason="stop", model_name=self.model_name,
            metadata={"latency_ms": 1.0, "duration_s": 0.001},
        )


def test_generate_stamps_latency():
    r = _TinyEngine().generate("hi")
    assert "latency_ms" in r.metadata and "duration_s" in r.metadata
    assert r.metadata["latency_ms"] >= 0
    assert r.metadata["duration_s"] >= 0
    # original metadata is preserved alongside the new keys.
    assert r.metadata["prompt_tokens"] == 1


def test_generate_batch_stamps_latency_on_every_item():
    results = _TinyEngine().generate_batch(["a", "b", "c"])
    assert len(results) == 3
    for r in results:
        assert "latency_ms" in r.metadata and "duration_s" in r.metadata


def test_engine_self_timing_is_not_overwritten():
    # setdefault semantics: an engine that measures its own latency keeps it.
    r = _SelfTimedEngine().generate("hi")
    assert r.metadata["latency_ms"] == 1.0
    assert r.metadata["duration_s"] == 0.001


def test_wrapping_is_idempotent_across_subclassing():
    # _SelfTimedEngine subclasses _TinyEngine; neither generate should be double
    # wrapped (the marker prevents re-wrapping an already-timed method).
    assert getattr(_TinyEngine.generate, "__effgen_timed__", False) is True
    assert getattr(_SelfTimedEngine.generate, "__effgen_timed__", False) is True
