"""FirstAvailablePolicy — returns the first provider with a configured key that meets capabilities."""

from __future__ import annotations

import os

from effgen.models.router import (
    NoCandidateError,
    ProviderModelPair,
    RouterDecision,
    RoutingContext,
    RoutingPolicy,
)


class FirstAvailablePolicy(RoutingPolicy):
    """Select the first provider whose API key is set and supports all required capabilities.

    Candidates are evaluated in the order returned by ProviderRegistry (alphabetical).
    The first model for each qualifying provider is chosen.

    This is a no-op fallback policy — it makes no attempt to optimise for cost or latency.
    """

    @property
    def name(self) -> str:
        return "first_available"

    def select(
        self,
        candidates: list[ProviderModelPair],
        context: RoutingContext,
    ) -> RouterDecision:
        from effgen.models.registry import ProviderRegistry

        eliminated: list[tuple[ProviderModelPair, str]] = []
        seen_providers: dict[str, str] = {}
        chosen: ProviderModelPair | None = None

        for pair in candidates:
            provider = pair.provider

            # One decision per provider — pick first model only
            if provider in seen_providers:
                eliminated.append((pair, f"provider already evaluated: {seen_providers[provider]}"))
                continue

            # Check API key availability
            rec = ProviderRegistry.get_provider_info(provider)
            env_keys = rec.get("env_keys", [])
            if env_keys and not any(os.environ.get(k) for k in env_keys):
                reason = f"no API key configured ({', '.join(env_keys)})"
                eliminated.append((pair, reason))
                seen_providers[provider] = reason
                continue

            # Check capabilities
            provider_caps = ProviderRegistry.get_capabilities(provider)
            missing = context.required_capabilities - provider_caps
            if missing:
                names = ", ".join(c.value for c in missing)
                reason = f"missing capabilities: {names}"
                eliminated.append((pair, reason))
                seen_providers[provider] = reason
                continue

            if chosen is None:
                chosen = pair
                seen_providers[provider] = f"selected {pair.provider}/{pair.model_id}"
                continue

            reason = f"not selected: first available was {chosen.provider}/{chosen.model_id}"
            eliminated.append((pair, reason))
            seen_providers[provider] = reason

        if chosen is not None:
            return RouterDecision(
                chosen=chosen,
                eliminated=eliminated,
                policy_name=self.name,
                score=1.0,
            )

        raise NoCandidateError(
            f"FirstAvailablePolicy: no provider has a configured key and supports "
            f"capabilities {[c.value for c in context.required_capabilities]}. "
            f"Eliminated {len(eliminated)} candidates."
        )
