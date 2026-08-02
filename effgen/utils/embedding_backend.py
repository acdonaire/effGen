"""Loading a sentence-transformers model without a working network route.

The embedding model behind RAG retrieval, semantic chunking, the vector store
and semantic eval scoring is downloaded from the model hub on first use and then
served from the local cache. When the hub cannot be reached, ``huggingface_hub``
closes its HTTP client after the first failed request, and a load makes several,
so the request that follows raises ``RuntimeError: Cannot send a request, as the
client has been closed`` instead of reading the copy already on disk.

:func:`load_sentence_transformer` retries the load against the local cache in
that case, so a machine with a warm cache and no network still works, and reports
a model that is genuinely absent by name with the way to obtain it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "sentence-transformers is not installed. Install with: pip install sentence-transformers"
)

# Substrings that identify a failure to reach the model hub, as opposed to a bad
# model id or a corrupt download.
_OFFLINE_MARKERS = (
    "client has been closed",
    "name resolution",
    "temporary failure",
    "connection error",
    "connectionerror",
    "max retries exceeded",
    "network is unreachable",
    "failed to establish a new connection",
    "couldn't connect to",
    "connection aborted",
    "timed out",
    "offlinemodeisenabled",
)


def looks_offline(exc: BaseException) -> bool:
    """True when *exc*, or anything it chains to, reports an unreachable host."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".lower()
        if any(marker in text for marker in _OFFLINE_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def load_sentence_transformer(model_name: str, **kwargs: Any) -> Any:
    """Return a loaded ``SentenceTransformer``, reading the cache when offline.

    Args:
        model_name: Model id or path passed to ``SentenceTransformer``.
        **kwargs: Forwarded to the constructor unchanged.

    Returns:
        The loaded model.

    Raises:
        ImportError: ``sentence-transformers`` is not installed.
        RuntimeError: The hub is unreachable and the model is not cached.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc

    try:
        return SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        if not looks_offline(exc):
            raise
        logger.info(
            "Model hub unreachable while loading '%s'; reading the local cache instead.",
            model_name,
        )
        try:
            return SentenceTransformer(model_name, local_files_only=True, **kwargs)
        except Exception as cached_exc:
            raise RuntimeError(
                f"Could not load the embedding model '{model_name}': the model hub is "
                f"unreachable and no copy is in the local cache. Download it once with a "
                f"network connection, or point HF_HOME at a cache that already has it."
            ) from cached_exc
