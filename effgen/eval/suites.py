"""
Built-in evaluation test suites.

Each suite loads test cases from JSONL files shipped under ``effgen/eval/data/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .evaluator import TestCase

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SUITES: dict[str, type[TestSuite]] = {}


def _register(cls: type[TestSuite]) -> type[TestSuite]:
    _SUITES[cls.name] = cls
    return cls


def list_suites() -> dict[str, str]:
    """Return ``{name: description}`` for every registered suite."""
    return {name: cls.description for name, cls in _SUITES.items()}


def get_suite(name: str) -> TestSuite:
    """Instantiate a suite by name. Raises ``KeyError`` if unknown."""
    if name not in _SUITES:
        raise KeyError(
            f"Unknown suite {name!r}. Available: {', '.join(_SUITES)}"
        )
    return _SUITES[name]()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TestSuite:
    """Base class for evaluation suites."""
    __test__ = False  # not a pytest test class
    name: str = "base"
    description: str = ""
    filename: str = ""  # JSONL file under data/

    def __init__(self, test_cases: list[TestCase] | None = None) -> None:
        """Create a suite.

        Args:
            test_cases: Explicit test cases to use. When omitted, the suite's
                bundled cases are loaded from its JSONL data file. Supplying
                ``test_cases`` lets you build an ad-hoc suite without a data file.
        """
        self.test_cases: list[TestCase] = (
            list(test_cases) if test_cases is not None else self._load()
        )

    def _load(self) -> list[TestCase]:
        path = _DATA_DIR / self.filename
        if not path.exists():
            logger.warning("Suite data file not found: %s", path)
            return []
        cases: list[TestCase] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                cases.append(TestCase.from_dict(json.loads(line)))
        return cases

    def __iter__(self):
        return iter(self.test_cases)

    def __len__(self):
        return len(self.test_cases)

    def filter(self, **kwargs: Any) -> list[TestCase]:
        """Filter test cases by arbitrary field values.

        Example: ``suite.filter(difficulty="easy", tags=["arithmetic"])``
        """
        out: list[TestCase] = []
        for tc in self.test_cases:
            match = True
            for k, v in kwargs.items():
                val = getattr(tc, k, None)
                if isinstance(v, list) and isinstance(val, list):
                    if not set(v) & set(val):
                        match = False
                elif val != v:
                    match = False
            if match:
                out.append(tc)
        return out


# ---------------------------------------------------------------------------
# Concrete suites
# ---------------------------------------------------------------------------

@_register
class MathSuite(TestSuite):
    name = "math"
    description = "50 math problems — basic arithmetic to calculus"
    filename = "math.jsonl"


@_register
class ToolUseSuite(TestSuite):
    name = "tool_use"
    description = "30 tool-use scenarios across built-in tools"
    filename = "tool_use.jsonl"


@_register
class ReasoningSuite(TestSuite):
    name = "reasoning"
    description = "20 multi-step reasoning problems"
    filename = "reasoning.jsonl"


@_register
class SafetySuite(TestSuite):
    name = "safety"
    description = "20 prompt injection and safety tests"
    filename = "safety.jsonl"


@_register
class ConversationSuite(TestSuite):
    name = "conversation"
    description = "10 multi-turn conversation evaluations"
    filename = "conversation.jsonl"
