"""
Model Router for effGen framework.

Two routing layers:

1. **Policy-based router** (new in v0.2.4): ``ModelRouter(policies=[...])`` /
   ``PolicyBasedRouter`` / ``RoutingPolicy`` / ``RouterDecision`` route across
   cloud *providers* based on declared capabilities, cost, and latency.  Use this
   when you want to let effGen pick the best provider.

2. **Complexity-based router** (v0.2.3): legacy ``ModelRouter`` for local model
   pools that scores candidates by task complexity and GPU availability.
   Preserved for back-compat — existing code using ``ModelRouter(models=[...])``
   continues to work unchanged.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

from effgen.models.base import BaseModel
from effgen.models.capabilities import (
    Capability,
    ModelCapability,
    estimate_capability,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy-based routing (v0.2.4+)
# ---------------------------------------------------------------------------

class ProviderModelPair(NamedTuple):
    """A (provider, model_id) pair that the router can route to."""
    provider: str
    model_id: str


@dataclass
class RoutingContext:
    """Input to the policy-based router.

    Attributes:
        prompt_tokens_estimate: Estimated input token count.
        user_budget_usd:        Max spend per call in USD (None = unlimited).
        latency_budget_ms:      Max acceptable latency in ms (None = unlimited).
        required_capabilities:  Set of Capability flags the chosen provider must support.
    """
    prompt_tokens_estimate: int = 0
    user_budget_usd: float | None = None
    latency_budget_ms: int | None = None
    required_capabilities: set[Capability] = field(default_factory=set)


@dataclass
class RouterDecision:
    """Result of a policy-based routing call.

    Attributes:
        chosen:      The selected (provider, model_id) pair.
        eliminated:  List of (pair, reason) for every rejected candidate.
        policy_name: Name of the policy that made the selection.
        score:       Numeric score for the chosen candidate (policy-defined).
    """
    chosen: ProviderModelPair
    eliminated: list[tuple[ProviderModelPair, str]] = field(default_factory=list)
    policy_name: str = ""
    score: float = 0.0


class RoutingPolicy(ABC):
    """Abstract base for all routing policies."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def select(
        self,
        candidates: list[ProviderModelPair],
        context: RoutingContext,
    ) -> RouterDecision:
        """Select the best candidate given *context*.

        Args:
            candidates: All (provider, model_id) pairs to consider.
            context:    Routing constraints and requirements.

        Returns:
            RouterDecision with the chosen pair and elimination log.

        Raises:
            NoCandidateError: If no candidate satisfies the requirements.
        """


class NoCandidateError(RuntimeError):
    """Raised when no provider/model satisfies the routing constraints."""


class PolicyBasedRouter:
    """Routes to the best cloud provider by running a chain of policies.

    Usage::

        from effgen.models.router import PolicyBasedRouter, RoutingContext
        from effgen.models.capabilities import Capability
        from effgen.models.routing.first_available import FirstAvailablePolicy

        router = PolicyBasedRouter(policies=[FirstAvailablePolicy()])
        decision = router.route(RoutingContext(required_capabilities={Capability.chat}))
        print(decision.chosen.provider, decision.chosen.model_id)

    Args:
        policies:  Ordered list of policies to try.  The first policy that
                   returns a decision wins.
        fallback:  Policy used if all others raise NoCandidateError.
                   Defaults to FirstAvailablePolicy.
    """

    def __init__(
        self,
        policies: list[RoutingPolicy],
        fallback: RoutingPolicy | None = None,
    ) -> None:
        if fallback is None:
            from effgen.models.routing.first_available import FirstAvailablePolicy
            fallback = FirstAvailablePolicy()
        self._policies = list(policies)
        self._fallback = fallback

    def _get_candidates(self) -> list[ProviderModelPair]:
        """Return all registered (provider, model_id) pairs."""
        from effgen.models.registry import ProviderRegistry
        pairs: list[ProviderModelPair] = []
        for provider in ProviderRegistry.list_providers():
            for model_info in ProviderRegistry.list_models(provider):
                pairs.append(ProviderModelPair(provider, model_info["model_id"]))
        return pairs

    def route(self, context: RoutingContext) -> RouterDecision:
        """Run policies in order and return the first successful decision."""
        candidates = self._get_candidates()
        last_error: NoCandidateError | None = None
        for policy in self._policies:
            try:
                return policy.select(candidates, context)
            except NoCandidateError as exc:
                last_error = exc
                logger.debug("Policy %r found no candidate: %s", policy.name, exc)
        try:
            return self._fallback.select(candidates, context)
        except NoCandidateError:
            raise last_error or NoCandidateError("No provider available for the given context")


# ---------------------------------------------------------------------------
# Task complexity estimation (v0.2.3, preserved for back-compat)
# ---------------------------------------------------------------------------

class ComplexityLevel(Enum):
    """Discrete complexity levels for routing decisions."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ComplexityEstimate:
    """Result of complexity estimation."""
    level: ComplexityLevel
    score: float  # 0.0 (trivial) to 1.0 (very complex)
    required_capabilities: list[str] = field(default_factory=list)
    reasoning: str = ""


# Keyword sets for heuristic complexity detection.
# Kept as module-level frozensets for speed (no allocation per call).
_SIMPLE_KEYWORDS = frozenset([
    "what", "who", "when", "where", "define", "meaning", "name",
    "capital", "color", "colour", "list", "true", "false", "yes", "no",
    "hello", "hi", "hey", "thanks", "thank",
])

_CODE_KEYWORDS = frozenset([
    "code", "program", "function", "class", "implement", "debug", "fix",
    "refactor", "algorithm", "api", "endpoint", "sql", "query", "script",
    "python", "javascript", "typescript", "java", "rust", "golang", "cpp",
    "html", "css", "regex", "compile", "runtime", "error", "exception",
    "test", "unittest", "pytest", "recursive", "recursion", "sort",
    "sorting", "binary", "tree", "stack", "queue", "hash", "database",
    "write", "build", "create", "deploy", "docker", "kubernetes",
])

_MATH_KEYWORDS = frozenset([
    "calculate", "compute", "solve", "equation", "integral", "derivative",
    "matrix", "vector", "probability", "statistics", "sum", "product",
    "factorial", "fibonacci", "prime", "graph", "optimization", "linear",
    "algebra", "geometry", "trigonometry", "calculus",
])

_REASONING_KEYWORDS = frozenset([
    "explain", "why", "how", "compare", "contrast", "analyse", "analyze",
    "evaluate", "argue", "reason", "logic", "proof", "infer", "deduce",
    "hypothesis", "theory", "because", "therefore", "strategy", "plan",
    "design", "architecture", "trade-off", "tradeoff",
])

_MULTILINGUAL_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff"
    r"\u0400-\u04ff\u0600-\u06ff\u0900-\u097f"
    r"\uac00-\ud7af]"
)

_COMPLEX_STRUCTURE_PATTERNS = [
    re.compile(r"step[- ]?by[- ]?step", re.IGNORECASE),
    re.compile(r"multi[- ]?step", re.IGNORECASE),
    re.compile(r"with error handling", re.IGNORECASE),
    re.compile(r"edge cases?", re.IGNORECASE),
    re.compile(r"comprehensive", re.IGNORECASE),
    re.compile(r"full[- ]?(stack|implementation)", re.IGNORECASE),
]


def estimate_complexity(
    query: str,
    tools: list[Any] | None = None,
) -> ComplexityEstimate:
    """Estimate task complexity from the query text.

    This is a pure-heuristic function optimised for speed (< 1ms).
    It analyses keyword density, query length, and structural patterns.

    Args:
        query: The user query / task description.
        tools: Optional list of tools available (more tools = potentially
               more complex orchestration).

    Returns:
        ComplexityEstimate with level, score, and detected capabilities.
    """
    if not query:
        return ComplexityEstimate(
            level=ComplexityLevel.SIMPLE, score=0.0, reasoning="empty query"
        )

    words = set(query.lower().split())
    score = 0.0
    capabilities: list[str] = []

    # --- Length signal ---
    n_chars = len(query)
    n_words = len(words)
    if n_chars > 500:
        score += 0.2
    elif n_chars > 200:
        score += 0.1
    elif n_chars < 30:
        score -= 0.1

    # --- Simple-query signal ---
    simple_hits = words & _SIMPLE_KEYWORDS
    if simple_hits and n_words < 12:
        score -= 0.15

    # --- Domain keyword signals ---
    code_hits = words & _CODE_KEYWORDS
    if code_hits:
        score += min(len(code_hits) * 0.08, 0.3)
        capabilities.append("code")

    math_hits = words & _MATH_KEYWORDS
    if math_hits:
        score += min(len(math_hits) * 0.08, 0.3)
        capabilities.append("math")

    reasoning_hits = words & _REASONING_KEYWORDS
    if reasoning_hits:
        score += min(len(reasoning_hits) * 0.06, 0.25)
        capabilities.append("reasoning")

    # --- Multilingual signal ---
    if _MULTILINGUAL_PATTERN.search(query):
        score += 0.1
        capabilities.append("multilingual")

    # --- Structural complexity ---
    for pat in _COMPLEX_STRUCTURE_PATTERNS:
        if pat.search(query):
            score += 0.1

    # --- Tool signal ---
    if tools:
        n_tools = len(tools)
        if n_tools > 5:
            score += 0.15
            capabilities.append("tool_calling")
        elif n_tools > 2:
            score += 0.08
            capabilities.append("tool_calling")

    # Clamp
    score = max(0.0, min(1.0, score))

    # Map to discrete level
    if score < 0.25:
        level = ComplexityLevel.SIMPLE
    elif score < 0.50:
        level = ComplexityLevel.MODERATE
    else:
        level = ComplexityLevel.COMPLEX

    if not capabilities:
        capabilities.append("instruction_following")

    return ComplexityEstimate(
        level=level,
        score=round(score, 3),
        required_capabilities=capabilities,
        reasoning=f"len={n_chars}, words={n_words}, "
                  f"code={len(code_hits)}, math={len(math_hits)}, "
                  f"reasoning={len(reasoning_hits)}",
    )


# ---------------------------------------------------------------------------
# Routing configuration
# ---------------------------------------------------------------------------

@dataclass
class RoutingConfig:
    """Configuration for the model router.

    Attributes:
        complexity_threshold_simple: Max score to still use the smallest model.
        complexity_threshold_moderate: Max score for a mid-sized model.
        gpu_memory_headroom_gb: GB of GPU memory to keep free as headroom.
        prefer_loaded: If True, prefer models already loaded in memory.
        latency_budget_ms: Optional max latency in ms (unused for now).
        cost_budget: Optional cost ceiling (unused for now, reserved for API models).
    """
    complexity_threshold_simple: float = 0.25
    complexity_threshold_moderate: float = 0.50
    gpu_memory_headroom_gb: float = 1.0
    prefer_loaded: bool = True
    latency_budget_ms: float | None = None
    cost_budget: float | None = None


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    model: BaseModel
    model_name: str
    complexity: ComplexityEstimate
    reason: str
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    """Routes either across providers by policy or across local model pools.

    Policy-based routing is enabled by passing ``policies`` and using
    :meth:`route`.  The original complexity-based local pool behavior is
    preserved when passing ``models`` and using :meth:`select`.

    The router scores each candidate model against the estimated task
    complexity and selects the best fit.  It prefers smaller models for
    simple tasks and larger/more-capable models for complex ones.

    Args:
        models: List of BaseModel instances or model-name strings.
        config: Optional RoutingConfig.
        policies: Optional ordered list of provider routing policies.
        fallback: Optional fallback provider routing policy.
    """

    def __init__(
        self,
        models: list[BaseModel] | None = None,
        config: RoutingConfig | None = None,
        *,
        policies: list[RoutingPolicy] | None = None,
        fallback: RoutingPolicy | None = None,
    ):
        self.config = config or RoutingConfig()
        self._models: list[BaseModel] = list(models) if models else []
        self._capabilities: dict[str, ModelCapability] = {}
        self._policy_router: PolicyBasedRouter | None = (
            PolicyBasedRouter(policies=policies, fallback=fallback)
            if policies is not None
            else None
        )

        # Pre-fetch capability profiles
        for m in self._models:
            name = getattr(m, "model_name", str(m))
            self._capabilities[name] = estimate_capability(name)

    # -- public API --

    def add_model(self, model: BaseModel) -> None:
        """Add a model to the router's candidate pool."""
        self._models.append(model)
        name = getattr(model, "model_name", str(model))
        self._capabilities[name] = estimate_capability(name)
        logger.info("Router: added model '%s'", name)

    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the router by name."""
        for i, m in enumerate(self._models):
            if getattr(m, "model_name", "") == model_name:
                self._models.pop(i)
                self._capabilities.pop(model_name, None)
                logger.info("Router: removed model '%s'", model_name)
                return True
        return False

    @property
    def models(self) -> list[BaseModel]:
        """Return the list of candidate models."""
        return list(self._models)

    def route(self, context: RoutingContext) -> RouterDecision:
        """Route across registered providers using the configured policies."""
        if self._policy_router is None:
            raise TypeError(
                "ModelRouter.route() requires policies=[...]. "
                "Use select() for complexity-based model-pool routing."
            )
        return self._policy_router.route(context)

    def select(
        self,
        query: str,
        tools: list[Any] | None = None,
    ) -> RoutingDecision:
        """Select the best model for a query.

        Args:
            query: The user query / task description.
            tools: Optional list of tools the agent has.

        Returns:
            RoutingDecision with the chosen model and reasoning.

        Raises:
            ValueError: If no models are available.
        """
        if not self._models:
            raise ValueError("ModelRouter has no candidate models")

        t0 = time.monotonic()
        complexity = estimate_complexity(query, tools)

        best_model = None
        best_score = -1.0
        best_reason = ""

        for model in self._models:
            name = getattr(model, "model_name", str(model))
            cap = self._capabilities.get(name)
            if cap is None:
                cap = estimate_capability(name)
                self._capabilities[name] = cap

            score = self._score_model(model, cap, complexity)

            if score > best_score:
                best_score = score
                best_model = model
                best_reason = (
                    f"score={score:.3f}, size={cap.size_billions:.1f}B, "
                    f"complexity={complexity.level.value}"
                )

        elapsed = (time.monotonic() - t0) * 1000

        decision = RoutingDecision(
            model=best_model,
            model_name=getattr(best_model, "model_name", str(best_model)),
            complexity=complexity,
            reason=best_reason,
            elapsed_ms=round(elapsed, 3),
        )
        logger.info(
            "Router selected '%s' for query (complexity=%s, %.2fms)",
            decision.model_name, complexity.level.value, elapsed,
        )
        return decision

    # -- scoring --

    def _score_model(
        self,
        model: BaseModel,
        cap: ModelCapability,
        complexity: ComplexityEstimate,
    ) -> float:
        """Score a model for a given complexity estimate.

        Higher is better.  The scoring balances:
        - Capability match (does the model have the required skills?)
        - Size efficiency (prefer smaller models for simple tasks)
        - Loaded-preference (avoid load latency)
        """
        score = 0.0

        # 1. Capability match — average score on required dimensions
        if complexity.required_capabilities:
            cap_scores = [
                cap.get_score(dim) for dim in complexity.required_capabilities
            ]
            avg_cap = sum(cap_scores) / len(cap_scores)
        else:
            avg_cap = cap.overall_score()

        # Weight capability match by complexity
        score += avg_cap * (0.5 + complexity.score * 0.5)

        # 2. Size efficiency — penalise oversized models for simple tasks
        if complexity.level == ComplexityLevel.SIMPLE:
            # Strongly prefer the smallest available model to save resources.
            # Inverse-size bonus: 0.5B -> +0.45, 1.5B -> +0.35, 3B -> +0.2, 7B -> -0.2
            if cap.size_billions > 0:
                size_bonus = max(-0.2, 0.5 - cap.size_billions * 0.1)
            else:
                size_bonus = 0.3
            score += size_bonus
        elif complexity.level == ComplexityLevel.COMPLEX:
            # Prefer larger models
            if cap.size_billions >= 7.0:
                score += 0.25
            elif cap.size_billions >= 3.0:
                score += 0.15
            else:
                score -= 0.05

        # 3. Prefer loaded models
        if self.config.prefer_loaded and hasattr(model, "is_loaded"):
            if model.is_loaded():
                score += 0.15

        return round(score, 4)
