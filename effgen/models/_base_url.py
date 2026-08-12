"""Resolution of the endpoint URL for OpenAI-protocol adapters.

One place decides where an OpenAI-protocol call is sent, so the adapter, the
model loader and the CLI all agree. An explicit ``base_url=`` argument always
wins; with none given the environment is consulted in the order below.
"""

from __future__ import annotations

import os

#: Environment variables consulted for the endpoint, in precedence order.
#: ``EFFGEN_BASE_URL`` is effGen's own and comes first so it can point effGen at
#: a server without redirecting every other OpenAI client on the machine. The
#: two ``OPENAI_*`` names are the conventional ones the OpenAI SDK and most
#: tooling already read.
BASE_URL_ENV_VARS: tuple[str, ...] = (
    "EFFGEN_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
)

#: The api_key sent when a server needs the field populated but checks nothing.
#: vLLM, SGLang, TGI, llama.cpp and Ollama all accept any non-empty string.
PLACEHOLDER_API_KEY = "EMPTY"


def resolve_base_url(explicit: str | None = None) -> str | None:
    """Return the endpoint URL to call, or None to use the provider's own.

    Args:
        explicit: A URL passed by the caller. Used as-is when non-empty.

    Returns:
        The URL with any trailing slash removed, or None when neither an
        argument nor any of :data:`BASE_URL_ENV_VARS` supplies one.
    """
    candidate = explicit
    if not candidate:
        for var in BASE_URL_ENV_VARS:
            value = os.getenv(var)
            if value and value.strip():
                candidate = value
                break
    if not candidate or not candidate.strip():
        return None
    return candidate.strip().rstrip("/")


def describe_endpoint(base_url: str | None) -> str:
    """Return a short label for the endpoint, for logs and error messages."""
    return base_url if base_url else "the OpenAI API"
