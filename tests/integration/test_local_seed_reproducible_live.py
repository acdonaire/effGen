"""Local Transformers engine honors ``seed`` for reproducible sampling.

With a fixed ``seed`` and ``temperature>0``, two on-device generations of the
same prompt must return identical text; changing the seed must change it. Runs
a real small model on GPU when one is available (skipped otherwise). No mocking
of generation behavior.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch always present in this env
    _HAS_CUDA = False

from effgen.models import load_model
from effgen.models.base import GenerationConfig

MODEL_NAME = "transformers:Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Write one imaginative sentence about the ocean."


@pytest.mark.skipif(not _HAS_CUDA, reason="SKIPPED: no CUDA device for local model")
def test_local_engine_seed_makes_sampling_reproducible():
    model = load_model(MODEL_NAME)

    def _cfg(seed: int) -> GenerationConfig:
        return GenerationConfig(temperature=0.8, top_p=0.95, max_tokens=40, seed=seed)

    first = model.generate(PROMPT, _cfg(1234)).text
    again = model.generate(PROMPT, _cfg(1234)).text
    other = model.generate(PROMPT, _cfg(9999)).text

    assert first.strip()
    assert first == again, "same seed must reproduce identical sampled text"
    assert first != other, "a different seed must change the sampled text"
