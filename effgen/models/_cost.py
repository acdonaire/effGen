"""
Cost tracker for effGen model adapters.

Accumulates prompt and completion token counts per (provider, model) pair
and converts them to USD using per-provider rate tables.  Cerebras free-tier
cost is $0.  The process-global tracker persists events to SQLite so the
``effgen cost`` CLI can summarize spend across restarts.

Usage::

    from effgen.models._cost import CostTracker

    tracker = CostTracker.get()
    tracker.record("cerebras", "llama3.1-8b", prompt_tokens=50, completion_tokens=20)
    print(tracker.total_cost("cerebras", "llama3.1-8b"))
    print(tracker.summary())
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

_BUDGET_CONFIG_PATH = Path.home() / ".effgen" / "budget.json"


def _budget_config_path() -> Path:
    """Resolve the budget-config file, honoring an ``EFFGEN_BUDGET_CONFIG``
    override (mirrors ``EFFGEN_COST_DB``) so a sandbox or test run can point the
    budget away from the user's real ``~/.effgen/budget.json``."""
    env_path = os.environ.get("EFFGEN_BUDGET_CONFIG")
    return Path(env_path) if env_path else _BUDGET_CONFIG_PATH


def _load_budget() -> dict:
    """Load budget config (returns empty dict if absent)."""
    try:
        path = _budget_config_path()
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        logger.debug("Failed to load budget config; treating as empty", exc_info=True)
    return {}


def _configured_budgets(budget_cfg: dict) -> list[tuple[str, float]]:
    """Return the ``(period, budget_usd)`` pairs with a valid positive cap.

    Skips a period that is absent, non-numeric, or <= 0 (logging a warning for
    the non-numeric case) so callers can iterate only the budgets that apply.
    """
    result = []
    for period in ("daily", "monthly"):
        raw_budget = budget_cfg.get(period)
        if raw_budget is None:
            continue
        try:
            budget_usd = float(raw_budget)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid %s budget value: %r", period, raw_budget)
            continue
        if budget_usd <= 0:
            continue
        result.append((period, budget_usd))
    return result


def format_usd(amount: float) -> str:
    """Format a USD amount with enough significant digits for sub-cent values.

    Amounts of ``$0.0001`` or more use the familiar 4-decimal form
    (``$1.2300``); smaller amounts switch to enough decimal places to keep at
    least two significant digits, then drop trailing zeros, so a cap like
    ``$0.00005`` prints as ``$0.00005`` rather than rounding to ``$0.0001``
    (or vanishing to ``$0.0000``) under a fixed ``:.4f``.
    """
    if amount == 0:
        return "$0.0000"
    magnitude = abs(amount)
    if magnitude >= 0.0001:
        return f"${amount:.4f}"
    exponent = math.floor(math.log10(magnitude))
    decimals = -exponent + 1
    text = f"{amount:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"${text}"


# Internal alias kept short for call sites in this module.
_format_usd = format_usd

# ---------------------------------------------------------------------------
# Pricing resolution
#
# The normalized model catalog (``effgen/models/_catalog.py``) is the single
# source of truth for per-token prices: it reads the same in-package ``*_MODELS``
# dicts the adapters call, so the cost a user is charged can never disagree with
# the catalog the CLI shows.  ``_rate()`` consults the catalog first and only
# falls back to the small legacy table below for ids no catalog knows (e.g. the
# deprecated Cerebras test ids or a bare local model name).
#
# A model is treated as:
#   * "priced"   - the catalog publishes a per-token price (use it),
#   * "free"     - the catalog flags a genuine free tier ($0, labeled "free"),
#   * "metered"  - non-token billing (e.g. Replicate per-second; cost arrives
#                  pre-computed via ``record(cost_usd=...)``),
#   * "unpriced" - no published price (we never invent one; shown as "unpriced"
#                  rather than a misleading "$0.000000").
#
# The process-global tracker persists events to SQLite. Constructing
# ``CostTracker(storage=None)`` keeps the old in-memory behavior for callers
# that need isolated accounting.
# ---------------------------------------------------------------------------

# Providers whose legacy fallback rate of $0 means "genuinely free tier" rather
# than "price unknown" (used only for ids absent from the catalog).
_FREE_TIER_PROVIDERS = {"cerebras"}

# Legacy fallback rates (USD per 1M tokens) for ids the catalog does not carry.
# Kept intentionally small — the catalog is authoritative for everything else.
_RATES: dict[str, dict[str, tuple[float, float]]] = {
    "cerebras": {
        # OFFICIAL: Cerebras free tier pricing = $0 for all models.
        "*": (0.0, 0.0),
    },
    "openai": {
        # PLACEHOLDER (verify against OpenAI pricing page before billing use)
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "*": (1.00, 3.00),
    },
    "anthropic": {
        # PLACEHOLDER
        "claude-3-opus": (15.00, 75.00),
        "claude-3-sonnet": (3.00, 15.00),
        "claude-3-haiku": (0.25, 1.25),
        "*": (3.00, 15.00),
    },
    "gemini": {
        # PLACEHOLDER
        "gemini-pro": (0.125, 0.375),
        "gemini-1.5-pro": (3.50, 10.50),
        "*": (1.00, 3.00),
    },
    "groq": {
        # OFFICIAL: Groq free tier = $0 for all models (2026-04-28)
        "*": (0.0, 0.0),
    },
    "together": {
        # OFFICIAL rates from Together AI pricing page (2026-04-28).
        # Per million tokens: (input, output)
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
        "meta-llama/Meta-Llama-3-8B-Instruct": (0.20, 0.20),
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85),
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": (0.18, 0.18),
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": (0.88, 0.88),
        "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": (0.88, 0.88),
        "meta-llama/Meta-Llama-3.1-405B-Instruct": (3.50, 3.50),
        "meta-llama/Llama-3.2-1B-Instruct": (0.06, 0.06),
        "meta-llama/Llama-3-8b-chat-hf": (0.20, 0.20),
        "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
        "Qwen/Qwen2.5-72B-Instruct-Turbo": (1.20, 1.20),
        "Qwen/Qwen2.5-72B-Instruct": (1.20, 1.20),
        "Qwen/Qwen2.5-14B-Instruct": (0.80, 0.80),
        "Qwen/Qwen2.5-Coder-32B-Instruct": (0.80, 0.80),
        "Qwen/QwQ-32B": (1.20, 1.20),
        "Qwen/Qwen3.5-9B": (0.10, 0.15),
        "Qwen/Qwen3.5-397B-A17B": (0.60, 3.60),
        "Qwen/Qwen3-Coder-Next-FP8": (0.50, 1.20),
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": (2.00, 2.00),
        "Qwen/Qwen3-235B-A22B-Thinking-2507": (0.65, 3.00),
        "Qwen/Qwen3-Next-80B-A3B-Instruct": (0.15, 1.50),
        "Qwen/Qwen3-Next-80B-A3B-Thinking": (0.15, 1.50),
        "Qwen/Qwen3-VL-8B-Instruct": (0.18, 0.68),
        "Qwen/Qwen3-VL-32B-Instruct": (0.50, 1.50),
        "Qwen/Qwen2-VL-72B-Instruct": (1.20, 1.20),
        "Qwen/Qwen2.5-VL-72B-Instruct": (1.95, 8.00),
        "deepseek-ai/DeepSeek-V3.1": (0.60, 1.70),
        "deepseek-ai/DeepSeek-V3-0324": (1.25, 1.25),
        "deepseek-ai/DeepSeek-V4-Pro": (2.10, 4.40),
        "deepseek-ai/DeepSeek-R1": (3.00, 7.00),
        "deepseek-ai/DeepSeek-R1-0528": (3.00, 7.00),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (0.18, 0.18),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": (0.00, 0.00),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": (1.60, 1.60),
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": (2.00, 2.00),
        "mistralai/Mixtral-8x7B-Instruct-v0.1": (0.60, 0.60),
        "mistralai/Mistral-7B-Instruct-v0.3": (0.20, 0.20),
        "mistralai/Mistral-7B-Instruct-v0.1": (0.20, 0.20),
        "mistralai/Mistral-Small-24B-Instruct-2501": (0.10, 0.30),
        "mistralai/Ministral-3-14B-Instruct-2512": (0.20, 0.20),
        "openai/gpt-oss-20b": (0.05, 0.20),
        "openai/gpt-oss-120b": (0.15, 0.60),
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": (0.88, 0.88),
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2": (0.06, 0.25),
        "moonshotai/Kimi-K2.5": (0.50, 2.80),
        "moonshotai/Kimi-K2.6": (1.20, 4.50),
        "moonshotai/Kimi-K2-Thinking": (1.20, 4.00),
        "MiniMaxAI/MiniMax-M2.5": (0.30, 1.20),
        "MiniMaxAI/MiniMax-M2.7": (0.30, 1.20),
        "MiniMaxAI/MiniMax-M2": (0.00, 0.00),
        "zai-org/GLM-4.5-Air-FP8": (0.20, 1.10),
        "zai-org/GLM-4.6": (0.60, 2.20),
        "zai-org/GLM-4.7": (0.45, 2.00),
        "zai-org/GLM-5": (1.00, 3.20),
        "zai-org/GLM-5.1": (1.40, 4.40),
        "google/gemma-4-31B-it": (0.20, 0.50),
        "google/gemma-3n-E4B-it": (0.06, 0.12),
        "arize-ai/qwen-2-1.5b-instruct": (0.10, 0.10),
        "deepcogito/cogito-v2-1-671b": (1.25, 1.25),
        "Qwen/Qwen2-1.5B-Instruct": (0.02, 0.02),
        # Free / dedicated-endpoint models → $0 in tracker
        "*": (0.0, 0.0),
    },
}


def _catalog_pricing(provider: str, model: str) -> tuple[float, float, str, str] | None:
    """Resolve pricing for *provider*/*model* from the normalized catalog.

    Returns ``(price_in_per_1m, price_out_per_1m, status, note)`` where *status*
    is one of ``priced``/``free``/``metered``/``unpriced``, or ``None`` when the
    catalog has no record for this id (so the caller can fall back).
    """
    try:
        from effgen.models import _catalog as _cat

        rec = _cat.lookup(model, provider.lower())
    except Exception:  # pragma: no cover - catalog import/lookup is best-effort
        logger.debug("Catalog pricing lookup failed for %s/%s", provider, model, exc_info=True)
        return None
    if rec is None:
        return None

    pin = rec.price_in_per_1m
    pout = rec.price_out_per_1m
    # A genuinely nonzero published price wins over free_tier: some providers
    # (e.g. Gemini) flag a model free_tier=True to mean "usable within a free
    # quota" while still publishing the real per-token rate that applies
    # beyond that quota, so a nonzero price is the more accurate label there.
    # A catalog entry with an explicit 0/0 price is a different situation —
    # some Together entries carry this without knowing the real rate — so it
    # is never treated as "priced" on its own; it falls through to the
    # free_tier / price_note / unpriced checks below instead.
    if (pin is not None and pin > 0) or (pout is not None and pout > 0):
        return (float(pin or 0.0), float(pout or 0.0), "priced", rec.price_note or "")
    if rec.free_tier:
        return (0.0, 0.0, "free", rec.price_note or "")
    if rec.price_note:
        # Non-token billing (e.g. Replicate per-second); cost comes in via cost_usd.
        return (0.0, 0.0, "metered", rec.price_note)
    return (0.0, 0.0, "unpriced", "")


def _legacy_rate(provider: str, model: str) -> tuple[float, float] | None:
    """Per-1M rate from the small fallback table, or None if the provider/id is unknown."""
    provider_rates = _RATES.get(provider.lower())
    if not provider_rates:
        return None
    if model in provider_rates:
        return provider_rates[model]
    for key in provider_rates:
        if key != "*" and model.startswith(key):
            return provider_rates[key]
    if "*" in provider_rates:
        return provider_rates["*"]
    return None


def _rate(provider: str, model: str) -> tuple[float, float]:
    """Lookup (input_per_M, output_per_M) rate for provider/model.

    Catalog-first (the single source of truth shared with the adapters); falls
    back to the small legacy table for ids the catalog does not carry.  Returns
    ``(0.0, 0.0)`` for genuinely free/unpriced/unknown ids — pricing is never
    invented.
    """
    catalog = _catalog_pricing(provider, model)
    if catalog is not None:
        # priced -> real numbers; free/metered/unpriced -> 0 token rate.
        return (catalog[0], catalog[1])
    legacy = _legacy_rate(provider, model)
    return legacy if legacy is not None else (0.0, 0.0)


def pricing_status(provider: str, model: str) -> str:
    """Return how *provider*/*model* is priced: ``priced``/``free``/``metered``/``unpriced``.

    Used by ``effgen cost`` to label a $0 row accurately — a genuine free tier
    reads "free" while a model with no published price reads "unpriced" instead
    of a misleading "$0.000000".
    """
    catalog = _catalog_pricing(provider, model)
    if catalog is not None:
        return catalog[2]
    legacy = _legacy_rate(provider, model)
    if legacy is not None and (legacy[0] > 0 or legacy[1] > 0):
        return "priced"
    if provider.lower() in _FREE_TIER_PROVIDERS:
        return "free"
    return "unpriced"


@dataclass
class _ModelStats:
    """Per-(provider, model) accumulated stats."""
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    total_cost_usd: float = 0.0


class CostTracker:
    """Thread-safe cost tracker with optional SQLite persistence and budget alerts.

    Use :meth:`get` for the process-global singleton (default: SQLite-backed).
    Pass ``storage=None`` to get a pure in-memory instance (back-compat).

    Budget alerts
    -------------
    Set budgets via ``effgen config set budget.daily 1.0`` (writes
    ``~/.effgen/budget.json``).  On each :meth:`record` call:

    - At 80% of daily or monthly budget → :class:`UserWarning` is emitted.
    - At 100% → :class:`~effgen.models.errors.BudgetExceededError` is raised
      for paid calls. Zero-cost calls are still allowed so router failover can
      land on free-tier providers.

    The router catches ``BudgetExceededError`` and fails over to a free-tier
    provider when available.

    Example::

        tracker = CostTracker.get()
        cost = tracker.record("cerebras", "llama3.1-8b", 50, 20)
        print(f"Cost: ${cost:.6f}")  # 0.000000 for Cerebras free tier
    """

    _instance: "CostTracker | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        storage: "SQLiteCostStore | None" = None,
    ) -> None:
        self._data: dict[tuple[str, str], _ModelStats] = {}
        self._lock = threading.Lock()
        self._storage = storage

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "CostTracker":
        """Return the process-global CostTracker instance (SQLite-backed)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    from effgen.models._cost_store import SQLiteCostStore
                    cls._instance = cls(storage=SQLiteCostStore())
        return cls._instance

    @classmethod
    def instance(cls) -> "CostTracker":
        """Backward-compatible alias for :meth:`get`."""
        return cls.get()

    @classmethod
    def get_instance(cls) -> "CostTracker":
        """Backward-compatible alias for :meth:`get`."""
        return cls.get()

    @classmethod
    def reset(cls) -> None:
        """Reset the global tracker (useful in tests)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> float:
        """Record a completed API call and return the USD cost.

        Args:
            provider: Provider name, e.g. ``"cerebras"``.
            model: Model ID, e.g. ``"llama3.1-8b"``.
            prompt_tokens: Tokens in the prompt (from API usage).
            completion_tokens: Tokens in the completion (from API usage).
            input_tokens: Alias for ``prompt_tokens`` used by some adapters.
            output_tokens: Alias for ``completion_tokens`` used by some adapters.
            cost_usd: Optional precomputed USD cost override for providers that
                do not bill by prompt/completion token price.

        Returns:
            USD cost for this call (0.0 for Cerebras free tier).

        Raises:
            BudgetExceededError: If daily budget is configured and exceeded.
        """
        import time

        if input_tokens is not None:
            prompt_tokens = input_tokens
        if output_tokens is not None:
            completion_tokens = output_tokens

        input_rate, output_rate = _rate(provider, model)
        cost = (
            float(cost_usd)
            if cost_usd is not None
            else (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
        )

        key = (provider.lower(), model)
        with self._lock:
            if key not in self._data:
                self._data[key] = _ModelStats(provider=provider, model=model)
            stats = self._data[key]
            stats.prompt_tokens += prompt_tokens
            stats.completion_tokens += completion_tokens
            stats.requests += 1
            stats.total_cost_usd += cost

        if self._storage is not None:
            try:
                self._storage.insert(
                    provider=provider,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    timestamp=time.time(),
                )
            except Exception as exc:
                logger.warning("CostStore insert failed: %s", exc)

        logger.debug(
            "CostTracker.record %s/%s: prompt=%d completion=%d cost=$%.6f",
            provider, model, prompt_tokens, completion_tokens, cost,
        )

        self._check_budget(provider=provider, model=model, cost=cost)
        return cost

    def check_preflight(self, provider: str, model: str) -> None:
        """Refuse to start a call when a configured budget is already at/over its cap.

        Called before the provider call is made (see
        :func:`effgen.models.base._preflight_budget_check`), so a call refused
        here is never billed. This is a pre-existing-spend check only: it
        cannot foresee the cost of the call about to be made, so a call that
        pushes spend past the cap for the first time is still allowed through
        here and is caught after the fact by :meth:`_check_budget` inside
        :meth:`record`.
        """
        budget_cfg = _load_budget()
        for period, budget_usd in _configured_budgets(budget_cfg):
            spend = self._period_spend(period)
            if spend >= budget_usd:
                from effgen.models.errors import BudgetExceededError
                raise BudgetExceededError(
                    budget_usd=budget_usd,
                    actual_usd=spend,
                    period=period,
                    provider=provider,
                    model=model,
                )

    def _check_budget(self, provider: str, model: str, cost: float) -> None:
        """Emit a warning or raise BudgetExceededError for configured budgets."""
        budget_cfg = _load_budget()
        for period, budget_usd in _configured_budgets(budget_cfg):
            spend = self._period_spend(period)
            previous_spend = max(0.0, spend - cost)
            ratio = spend / budget_usd

            if ratio >= 1.0:
                if cost <= 0.0:
                    continue
                from effgen.models.errors import BudgetExceededError
                raise BudgetExceededError(
                    budget_usd=budget_usd,
                    actual_usd=spend,
                    period=period,
                    provider=provider,
                    model=model,
                )

            if previous_spend < budget_usd * 0.8 <= spend:
                warnings.warn(
                    f"effGen {period} budget warning: {_format_usd(spend)} / "
                    f"{_format_usd(budget_usd)} ({ratio * 100:.0f}%) spent.",
                    UserWarning,
                    stacklevel=4,
                )

    def _period_spend(self, period: str) -> float:
        """Return spend for *period* using storage when available."""
        if self._storage is not None:
            try:
                if period == "daily":
                    return sum(e.cost_usd for e in self._storage.query_today())
                if period == "monthly":
                    query_month = getattr(self._storage, "query_month", None)
                    if query_month is not None:
                        return sum(e.cost_usd for e in query_month())
                    import time

                    return sum(e.cost_usd for e in self._storage.query_since(
                        time.time() - 30 * 86400.0
                    ))
            except Exception:
                logger.warning("CostStore budget query failed; falling back to memory")
        return self.total_cost()

    def total_cost(self, provider: str | None = None, model: str | None = None) -> float:
        """Return total USD cost accumulated in memory, optionally filtered.

        Args:
            provider: Filter to this provider (None = all providers).
            model: Filter to this model (None = all models).

        Returns:
            Cumulative USD cost.
        """
        with self._lock:
            total = 0.0
            for (prov, mod), stats in self._data.items():
                if provider and prov != provider.lower():
                    continue
                if model and mod != model:
                    continue
                total += stats.total_cost_usd
        return total

    def total_tokens(self, provider: str | None = None, model: str | None = None) -> dict[str, int]:
        """Return total token counts, optionally filtered.

        Returns:
            Dict with keys ``prompt``, ``completion``, ``total``.
        """
        prompt = completion = 0
        with self._lock:
            for (prov, mod), stats in self._data.items():
                if provider and prov != provider.lower():
                    continue
                if model and mod != model:
                    continue
                prompt += stats.prompt_tokens
                completion += stats.completion_tokens
        return {"prompt": prompt, "completion": completion, "total": prompt + completion}

    def summary(self) -> list[dict]:
        """Return a list of per-(provider, model) usage summaries (in-memory).

        Returns:
            List of dicts with keys: ``provider``, ``model``, ``requests``,
            ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
            ``cost_usd``.
        """
        with self._lock:
            rows = []
            for stats in self._data.values():
                rows.append({
                    "provider": stats.provider,
                    "model": stats.model,
                    "requests": stats.requests,
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "total_tokens": stats.prompt_tokens + stats.completion_tokens,
                    "cost_usd": round(stats.total_cost_usd, 8),
                    "pricing": pricing_status(stats.provider, stats.model),
                })
        return rows

    def reset_stats(self) -> None:
        """Clear all accumulated in-memory stats (does not reset singleton or DB)."""
        with self._lock:
            self._data.clear()


if TYPE_CHECKING:
    from effgen.models._cost_store import SQLiteCostStore
