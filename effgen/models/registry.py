"""
Unified provider registry for effGen.

ProviderRegistry is a singleton that consolidates all provider/model
metadata so that presets, the router, and CLI commands can introspect
available providers and models without scattered per-provider imports.

Usage::

    from effgen.models.registry import ProviderRegistry

    ProviderRegistry.register(
        "groq",
        GroqAdapter,
        GROQ_MODELS,
        env_keys=["GROQ_API_KEY"],
    )

    providers = ProviderRegistry.list_providers()
    models    = ProviderRegistry.list_models("groq")
    provider, adapter_cls, info = ProviderRegistry.lookup("llama-3.3-70b-versatile")
"""

from __future__ import annotations

import logging
from typing import Any

from effgen.models.errors import AmbiguousModelError

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Singleton registry mapping providers → adapters + models."""

    # {provider_name: {"adapter_cls": cls, "models": dict, "env_keys": list[str]}}
    _providers: dict[str, dict[str, Any]] = {}
    # {model_id: [provider_name, ...]}  — tracks which providers expose a model id
    _model_index: dict[str, list[str]] = {}

    @classmethod
    def register(
        cls,
        provider_name: str,
        adapter_cls: type,
        models: dict[str, dict],
        env_keys: list[str] | None = None,
    ) -> None:
        """Register a provider.

        Idempotent: repeated calls with the same *provider_name* overwrite
        silently (the last registration wins) so that adapter modules can
        call this at import time without double-registration issues.

        Args:
            provider_name: Short lowercase provider label (``"groq"``, ``"openai"`` …).
            adapter_cls:   The adapter class (subclass of BaseModel).
            models:        Model registry dict ``{model_id: info_dict}``.
            env_keys:      Environment variable names that carry the API key,
                           e.g. ``["GROQ_API_KEY"]``.  Checked in order; the
                           first one present counts as "available".
        """
        if provider_name in cls._providers:
            # Remove old model-index entries for this provider
            for mid in cls._providers[provider_name].get("models", {}):
                if mid in cls._model_index:
                    cls._model_index[mid] = [
                        p for p in cls._model_index[mid] if p != provider_name
                    ]
                    if not cls._model_index[mid]:
                        del cls._model_index[mid]

        cls._providers[provider_name] = {
            "adapter_cls": adapter_cls,
            "models": models,
            "env_keys": list(env_keys or []),
        }

        # Rebuild model index for this provider
        for model_id in models:
            cls._model_index.setdefault(model_id, [])
            if provider_name not in cls._model_index[model_id]:
                cls._model_index[model_id].append(provider_name)

        logger.debug("ProviderRegistry: registered provider %r (%d models)", provider_name, len(models))

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return sorted list of all registered provider names."""
        return sorted(cls._providers)

    @classmethod
    def list_models(cls, provider: str) -> list[dict[str, Any]]:
        """Return list of model info dicts for *provider*.

        Each dict includes the ``model_id`` key so callers don't need to
        re-index.

        Raises:
            KeyError: If *provider* is not registered.
        """
        if provider not in cls._providers:
            raise KeyError(f"Provider {provider!r} is not registered. "
                           f"Available: {cls.list_providers()}")
        models = cls._providers[provider]["models"]
        return [{"model_id": mid, **info} for mid, info in models.items()]

    @classmethod
    def get_provider_info(cls, provider: str) -> dict[str, Any]:
        """Return the full registration record for *provider*."""
        if provider not in cls._providers:
            raise KeyError(f"Provider {provider!r} is not registered.")
        return cls._providers[provider]

    @classmethod
    def lookup(
        cls,
        model_id: str,
        provider: str | None = None,
    ) -> tuple[str, type, dict[str, Any]]:
        """Resolve *model_id* to ``(provider_name, adapter_cls, model_info)``.

        Args:
            model_id: The model identifier, optionally prefixed with
                      ``"provider:model_id"`` (e.g. ``"groq:llama-3.3-70b-versatile"``).
            provider: Explicit provider override (takes precedence over prefix
                      in *model_id*).

        Returns:
            A 3-tuple ``(provider_name, adapter_cls, model_info_dict)``.

        Raises:
            AmbiguousModelError: If *model_id* matches multiple providers and
                                 no provider is specified.
            KeyError: If *model_id* is not found in any registered provider.
        """
        # Parse "provider:model_id" prefix
        if provider is None and ":" in model_id:
            parts = model_id.split(":", 1)
            # Heuristic: if the prefix part is a known provider, use it
            if parts[0] in cls._providers:
                provider = parts[0]
                model_id = parts[1]

        if provider is not None:
            if provider not in cls._providers:
                raise KeyError(f"Provider {provider!r} is not registered.")
            rec = cls._providers[provider]
            if model_id not in rec["models"]:
                raise KeyError(
                    f"Model {model_id!r} not found in provider {provider!r}. "
                    f"Call list_models({provider!r}) to see available models."
                )
            return provider, rec["adapter_cls"], rec["models"][model_id]

        # No provider specified — look in index
        providers_for_model = cls._model_index.get(model_id, [])

        if not providers_for_model:
            raise KeyError(
                f"Model {model_id!r} not found in any registered provider. "
                f"Use 'provider:model_id' syntax or call list_models() to browse."
            )

        if len(providers_for_model) > 1:
            raise AmbiguousModelError(model_id, providers_for_model)

        prov = providers_for_model[0]
        rec = cls._providers[prov]
        return prov, rec["adapter_cls"], rec["models"][model_id]

    @classmethod
    def reset(cls) -> None:
        """Clear the registry (useful for testing)."""
        cls._providers.clear()
        cls._model_index.clear()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def list_providers() -> list[str]:
    """Return sorted list of all registered provider names."""
    return ProviderRegistry.list_providers()


def list_models(provider: str) -> list[dict[str, Any]]:
    """Return model info dicts for *provider*."""
    return ProviderRegistry.list_models(provider)


def lookup(
    model_id: str,
    provider: str | None = None,
) -> tuple[str, type, dict[str, Any]]:
    """Resolve *model_id* → ``(provider_name, adapter_cls, model_info)``."""
    return ProviderRegistry.lookup(model_id, provider)
