"""
Model comparison utilities.

Run the same evaluation suite across multiple models and generate
comparison reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .evaluator import AgentEvaluator, ScoringMode, SuiteResults

logger = logging.getLogger(__name__)


@dataclass
class ModelScore:
    """Per-model results for a single suite."""
    model_name: str
    suite_name: str
    accuracy: float = 0.0
    avg_latency: float = 0.0
    total_tokens: int = 0
    avg_tool_accuracy: float = 0.0
    error: str | None = None


def _recommendation_key(score: "ModelScore") -> tuple[float, float, float]:
    """Sort key for "best model" — higher is better.

    Primary: accuracy (higher wins). Tie-breaks: lower ``avg_latency`` then lower
    ``total_tokens`` (negated so "higher is better" still holds). When latency and
    tokens are also equal, the first-seen model is kept (strict ``>`` comparison),
    preserving the prior stable ordering for genuine ties.
    """
    return (score.accuracy, -float(score.avg_latency), -float(score.total_tokens))


@dataclass
class ComparisonMatrix:
    """Comparison of multiple models across one or more suites.

    Attributes:
        scores: List of per-model-per-suite scores.
        recommendations: ``{suite: model_name}`` best model per suite.
    """
    scores: list[ModelScore] = field(default_factory=list)
    recommendations: dict[str, str] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the matrix as a Markdown table."""
        if not self.scores:
            return "_No scores recorded._"

        suites = sorted({s.suite_name for s in self.scores})
        models = sorted({s.model_name for s in self.scores})

        # Build lookup
        lookup: dict[tuple[str, str], ModelScore] = {}
        for s in self.scores:
            lookup[(s.model_name, s.suite_name)] = s

        lines: list[str] = ["# Model Comparison", ""]

        # Accuracy table
        lines.append("## Accuracy")
        lines.append("")
        header = "| Model | " + " | ".join(suites) + " |"
        sep = "|-------|" + "|".join(["-------"] * len(suites)) + "|"
        lines.extend([header, sep])
        for m in models:
            cells = []
            for su in suites:
                sc = lookup.get((m, su))
                if sc and sc.error:
                    cells.append("ERROR")
                elif sc:
                    cells.append(f"{sc.accuracy:.1%}")
                else:
                    cells.append("—")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")

        # Latency table
        lines.extend(["", "## Avg Latency (s)", ""])
        lines.extend([header.replace("Accuracy", "Latency"), sep])
        for m in models:
            cells = []
            for su in suites:
                sc = lookup.get((m, su))
                if sc and not sc.error:
                    cells.append(f"{sc.avg_latency:.3f}")
                else:
                    cells.append("—")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")

        # Recommendations
        if self.recommendations:
            lines.extend(["", "## Recommendations", ""])
            for su, model in sorted(self.recommendations.items()):
                lines.append(f"- **{su}**: {model}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": [
                {
                    "model": s.model_name,
                    "suite": s.suite_name,
                    "accuracy": s.accuracy,
                    "avg_latency": s.avg_latency,
                    "total_tokens": s.total_tokens,
                    "avg_tool_accuracy": s.avg_tool_accuracy,
                    "error": s.error,
                }
                for s in self.scores
            ],
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ModelComparison:
    """Run the same suite(s) across multiple models and compare.

    Usage::

        comparison = ModelComparison()
        matrix = comparison.run(
            agents={"qwen-3b": agent_a, "llama-3b": agent_b},
            suites=[MathSuite()],
        )
        print(matrix.to_markdown())
    """

    def __init__(
        self,
        scoring: ScoringMode = ScoringMode.CONTAINS,
        pass_threshold: float = 0.5,
    ) -> None:
        self.scoring = scoring
        self.pass_threshold = pass_threshold

    def run(
        self,
        agents: dict[str, Any],
        suites: list[Any],
    ) -> ComparisonMatrix:
        """Evaluate all *agents* on all *suites*.

        Args:
            agents: ``{model_name: agent_instance}``
            suites: List of ``TestSuite`` instances.

        Returns:
            A :class:`ComparisonMatrix` with scores and recommendations.
        """
        matrix = ComparisonMatrix()

        for model_name, agent in agents.items():
            evaluator = AgentEvaluator(
                agent,
                scoring=self.scoring,
                pass_threshold=self.pass_threshold,
            )
            for suite in suites:
                suite_name = suite.name if hasattr(suite, "name") else "custom"
                logger.info("Evaluating %s on %s ...", model_name, suite_name)
                try:
                    results: SuiteResults = evaluator.run_suite(suite)
                    matrix.scores.append(ModelScore(
                        model_name=model_name,
                        suite_name=suite_name,
                        accuracy=results.accuracy,
                        avg_latency=results.avg_latency,
                        total_tokens=results.total_tokens,
                        avg_tool_accuracy=results.avg_tool_accuracy,
                    ))
                except Exception as exc:
                    logger.error("Error evaluating %s on %s: %s", model_name, suite_name, exc)
                    matrix.scores.append(ModelScore(
                        model_name=model_name,
                        suite_name=suite_name,
                        error=str(exc),
                    ))

        # Generate recommendations: highest accuracy per suite, breaking ties on
        # lower latency, then fewer tokens — so a "good enough" tie recommends the
        # cheaper/faster model instead of whichever happened to be listed first.
        suite_best: dict[str, ModelScore] = {}
        for s in matrix.scores:
            if s.error:
                continue
            cur = suite_best.get(s.suite_name)
            if cur is None or _recommendation_key(s) > _recommendation_key(cur):
                suite_best[s.suite_name] = s
        matrix.recommendations = {k: v.model_name for k, v in suite_best.items()}

        return matrix
