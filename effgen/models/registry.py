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
from typing import TYPE_CHECKING, Any

from effgen.models.errors import AmbiguousModelError

if TYPE_CHECKING:
    from effgen.models.capabilities import Capability
    from effgen.reliability.bulkhead import Bulkhead
    from effgen.reliability.circuit import CircuitBreaker

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Singleton registry mapping providers → adapters + models."""

    # {provider_name: {"adapter_cls": cls, "models": dict, "env_keys": list[str], "capabilities": set}}
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
        capabilities: "set[Capability] | None" = None,
        pricing: "dict[str, float | bool] | None" = None,
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
            capabilities:  Set of ``Capability`` flags this provider supports
                           (e.g. ``{Capability.chat, Capability.tools}``).
                           Used by the policy-based ModelRouter to filter candidates.
            pricing:       Provider-level default pricing dict with keys:
                           ``"input_per_1m"`` (float, USD per 1M input tokens),
                           ``"output_per_1m"`` (float, USD per 1M output tokens),
                           ``"free_tier"`` (bool, True if provider has a free tier).
                           Per-model pricing in ``models[id]["pricing_per_1m_input"]``
                           takes precedence over these provider defaults when available.
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
            "capabilities": set(capabilities) if capabilities else set(),
            "pricing": dict(pricing) if pricing else {
                "input_per_1m": 0.0,
                "output_per_1m": 0.0,
                "free_tier": False,
            },
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
    def is_available(cls, provider: str) -> bool:
        """Return True if *provider* has at least one configured API key."""
        import os
        rec = cls._providers.get(provider, {})
        return any(os.environ.get(k) for k in rec.get("env_keys", []))

    @classmethod
    def get_capabilities(cls, provider: str) -> "set[Capability]":
        """Return the capability set registered for *provider*."""
        return set(cls._providers.get(provider, {}).get("capabilities", set()))

    @classmethod
    def get_pricing(cls, provider: str) -> "dict[str, float | bool]":
        """Return the pricing dict for *provider*.

        Returns a dict with keys ``input_per_1m``, ``output_per_1m``,
        ``free_tier``.  Falls back to zeros/False if not registered.
        """
        return dict(cls._providers.get(provider, {}).get("pricing", {
            "input_per_1m": 0.0,
            "output_per_1m": 0.0,
            "free_tier": False,
        }))

    @classmethod
    def reset(cls) -> None:
        """Clear the registry (useful for testing)."""
        cls._providers.clear()
        cls._model_index.clear()

    # ------------------------------------------------------------------
    # Reliability middleware — circuit breakers + bulkheads per provider
    # ------------------------------------------------------------------

    @classmethod
    def get_circuit_breaker(
        cls,
        provider: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_probes: int = 1,
    ) -> "CircuitBreaker":
        """Return (or lazily create) the :class:`~effgen.reliability.circuit.CircuitBreaker`
        for *provider*.

        Circuit breakers are kept on the registry record so they survive across
        call sites.  Parameters are only applied on first creation.

        Args:
            provider:          Registered provider name (e.g. ``"openai"``).
            failure_threshold: Consecutive failures before OPEN.
            recovery_timeout:  Seconds OPEN before HALF_OPEN.
            half_open_probes:  Successes needed in HALF_OPEN before CLOSED.

        Returns:
            The :class:`~effgen.reliability.circuit.CircuitBreaker` instance.

        Raises:
            KeyError: If *provider* is not registered.
        """
        from effgen.reliability.circuit import CircuitBreaker as _CB

        if provider not in cls._providers:
            raise KeyError(f"Provider {provider!r} is not registered.")
        rec = cls._providers[provider]
        if "_circuit_breaker" not in rec:
            rec["_circuit_breaker"] = _CB(
                name=provider,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_probes=half_open_probes,
            )
        return rec["_circuit_breaker"]

    @classmethod
    def get_bulkhead(
        cls,
        provider: str,
        *,
        max_concurrency: int = 20,
        queue_size: int = 100,
        queue_timeout: float = 5.0,
    ) -> "Bulkhead":
        """Return (or lazily create) the :class:`~effgen.reliability.bulkhead.Bulkhead`
        for *provider*.

        Args:
            provider:        Registered provider name.
            max_concurrency: Maximum simultaneous active calls.
            queue_size:      Maximum calls that may wait for a permit.
            queue_timeout:   Seconds to wait before :class:`~effgen.reliability.bulkhead.BulkheadFull`.

        Returns:
            The :class:`~effgen.reliability.bulkhead.Bulkhead` instance.

        Raises:
            KeyError: If *provider* is not registered.
        """
        from effgen.reliability.bulkhead import Bulkhead as _BH

        if provider not in cls._providers:
            raise KeyError(f"Provider {provider!r} is not registered.")
        rec = cls._providers[provider]
        if "_bulkhead" not in rec:
            rec["_bulkhead"] = _BH(
                name=provider,
                max_concurrency=max_concurrency,
                queue_size=queue_size,
                queue_timeout=queue_timeout,
            )
        return rec["_bulkhead"]

    @classmethod
    def reliability_stats(cls) -> dict[str, Any]:
        """Return circuit-breaker and bulkhead stats for all providers.

        Returns a dict keyed by provider name, each with ``circuit_breaker``
        and ``bulkhead`` sub-dicts (or ``None`` if not yet created).
        """
        result: dict[str, Any] = {}
        for name, rec in cls._providers.items():
            cb = rec.get("_circuit_breaker")
            bh = rec.get("_bulkhead")
            result[name] = {
                "circuit_breaker": cb.stats() if cb is not None else None,
                "bulkhead": bh.stats() if bh is not None else None,
            }
        return result

    @classmethod
    def with_chaos(cls, chaos: Any) -> "ChaosMiddlewareRegistry":
        """Return a :class:`ChaosMiddlewareRegistry` that wraps this registry
        with the given *chaos* instance.

        Usage::

            from effgen.reliability.chaos import Chaos, Http5xx
            from effgen.models.registry import ProviderRegistry

            chaos = Chaos(seed=42)
            chaos.add_rule("primary", Http5xx, every_nth=3)
            registry = ProviderRegistry.with_chaos(chaos)
            # registry.call("primary", adapter.generate, prompt="Hello")

        Args:
            chaos: A :class:`~effgen.reliability.chaos.Chaos` instance with
                   rules already configured.

        Returns:
            A :class:`ChaosMiddlewareRegistry` that intercepts provider calls.
        """
        return ChaosMiddlewareRegistry(chaos=chaos)


class ChaosMiddlewareRegistry:
    """Thin registry wrapper that injects chaos faults before every provider call.

    Construct via :meth:`ProviderRegistry.with_chaos`.

    Args:
        chaos: The :class:`~effgen.reliability.chaos.Chaos` instance.
    """

    def __init__(self, chaos: Any) -> None:
        self._chaos = chaos

    def call(self, provider: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run chaos injection then forward *fn* if no fault fires.

        Args:
            provider: Provider name (used for rule matching).
            fn:       The real adapter callable.
            *args:    Positional args forwarded to *fn*.
            **kwargs: Keyword args forwarded to *fn*.
        """
        self._chaos.maybe_inject(provider)
        return fn(*args, **kwargs)

    async def async_call(
        self, provider: str, fn: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Async version of :meth:`call`."""
        await self._chaos.async_maybe_inject(provider)
        return await fn(*args, **kwargs)

    @property
    def chaos(self) -> Any:
        """The underlying :class:`~effgen.reliability.chaos.Chaos` instance."""
        return self._chaos


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
