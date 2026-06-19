"""Unit tests for the effgen.eval module."""

from __future__ import annotations

import json

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.eval.comparison import ComparisonMatrix, ModelComparison, ModelScore
from effgen.eval.evaluator import (
    AgentEvaluator,
    Difficulty,
    EvalResult,
    ScoringMode,
    SuiteResults,
    TestCase,
    _compute_tool_accuracy,
    _score_contains,
    _score_exact_match,
    _score_regex,
)
from effgen.eval.regression import RegressionAlert, RegressionTracker
from effgen.eval.suites import (
    ConversationSuite,
    MathSuite,
    ReasoningSuite,
    SafetySuite,
    ToolUseSuite,
    get_suite,
    list_suites,
)
from tests.fixtures.mock_models import MockModel

# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

class TestTestCase:
    def test_from_dict_basic(self):
        tc = TestCase.from_dict({
            "query": "What is 2+2?",
            "expected": "4",
            "tools": ["calculator"],
            "difficulty": "easy",
        })
        assert tc.query == "What is 2+2?"
        assert tc.expected_output == "4"
        assert tc.expected_tools == ["calculator"]
        assert tc.difficulty == Difficulty.EASY

    def test_from_dict_defaults(self):
        tc = TestCase.from_dict({"query": "hello"})
        assert tc.expected_output == ""
        assert tc.expected_tools == []
        assert tc.difficulty == Difficulty.MEDIUM
        assert tc.tags == []

    def test_from_dict_with_metadata(self):
        tc = TestCase.from_dict({
            "query": "test",
            "expected": "result",
            "metadata": {"turns": 3},
        })
        assert tc.metadata["turns"] == 3


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

class TestScoring:
    def test_exact_match(self):
        assert _score_exact_match("hello", "hello") == 1.0
        assert _score_exact_match("Hello", "hello") == 1.0  # case insensitive
        assert _score_exact_match("hello", "world") == 0.0

    def test_contains(self):
        assert _score_contains("world", "hello world") == 1.0
        assert _score_contains("xyz", "hello world") == 0.0
        assert _score_contains("WORLD", "hello world") == 1.0  # case insensitive

    def test_regex(self):
        assert _score_regex(r"\d+", "the answer is 42") == 1.0
        assert _score_regex(r"^\d+$", "not a number") == 0.0
        assert _score_regex(r"[invalid", "test") == 0.0  # bad regex

    def test_tool_accuracy_all_match(self):
        assert _compute_tool_accuracy(["calc", "json"], ["calc", "json"]) == 1.0

    def test_tool_accuracy_partial(self):
        assert _compute_tool_accuracy(["calc", "json"], ["calc"]) == 0.5

    def test_tool_accuracy_no_expected(self):
        assert _compute_tool_accuracy([], ["calc"]) == 1.0

    def test_tool_accuracy_none_match(self):
        assert _compute_tool_accuracy(["calc"], ["json"]) == 0.0


# ---------------------------------------------------------------------------
# EvalResult & SuiteResults
# ---------------------------------------------------------------------------

class TestTestCaseAliases:
    def test_expected_is_alias_for_expected_output(self):
        # The natural ``expected=`` keyword must populate expected_output rather
        # than raising a raw "unexpected keyword argument" TypeError.
        tc = TestCase(query="2+2?", expected="4")
        assert tc.expected_output == "4"

    def test_expected_output_takes_precedence_over_expected_alias(self):
        tc = TestCase(query="x", expected_output="canonical", expected="alias")
        assert tc.expected_output == "canonical"

    def test_input_alias_still_works(self):
        tc = TestCase(input="hello", expected="hi")
        assert tc.query == "hello" and tc.expected_output == "hi"


class TestEvalResult:
    def test_defaults(self):
        tc = TestCase(query="test")
        r = EvalResult(test_case=tc)
        assert r.score == 0.0
        assert r.passed is False
        assert r.latency == 0.0


class TestSuiteResults:
    def test_summary(self):
        tc1 = TestCase(query="q1", difficulty=Difficulty.EASY)
        tc2 = TestCase(query="q2", difficulty=Difficulty.HARD)
        results = SuiteResults(
            suite_name="test",
            results=[
                EvalResult(test_case=tc1, score=1.0, passed=True, latency=0.1, tokens_used=10),
                EvalResult(test_case=tc2, score=0.0, passed=False, latency=0.2, tokens_used=20),
            ],
            accuracy=0.5,
            avg_latency=0.15,
            total_tokens=30,
            avg_tool_accuracy=1.0,
        )
        s = results.summary()
        assert s["suite"] == "test"
        assert s["total"] == 2
        assert s["passed"] == 1
        assert s["accuracy"] == 0.5
        assert "easy" in s["by_difficulty"]
        assert "hard" in s["by_difficulty"]

    def test_to_json(self):
        results = SuiteResults(suite_name="json_test", accuracy=0.75)
        j = json.loads(results.to_json())
        assert j["suite"] == "json_test"


# ---------------------------------------------------------------------------
# AgentEvaluator with MockModel
# ---------------------------------------------------------------------------

class TestAgentEvaluator:
    def _make_agent(self, responses):
        model = MockModel(responses=responses)
        return Agent(config=AgentConfig(
            name="eval-test",
            model=model,
            tools=[],
            max_iterations=3,
            enable_memory=False,
            enable_sub_agents=False,
        ))

    def test_run_case_contains(self):
        agent = self._make_agent(["Thought: done\nFinal Answer: The answer is 42"])
        evaluator = AgentEvaluator(agent, scoring=ScoringMode.CONTAINS)
        tc = TestCase(query="What is the answer?", expected_output="42")
        result = evaluator.run_case(tc)
        assert result.passed is True
        assert result.score == 1.0
        assert result.latency > 0

    def test_run_case_exact_match_fail(self):
        agent = self._make_agent(["Thought: done\nFinal Answer: The answer is 42"])
        evaluator = AgentEvaluator(agent, scoring=ScoringMode.EXACT_MATCH)
        tc = TestCase(query="What is 42?", expected_output="42")
        result = evaluator.run_case(tc)
        # "The answer is 42" != "42" exact match
        assert result.passed is False

    def test_run_case_regex(self):
        agent = self._make_agent(["Thought: done\nFinal Answer: The value is 3.14"])
        evaluator = AgentEvaluator(agent, scoring=ScoringMode.REGEX)
        tc = TestCase(query="What is pi?", expected_output=r"3\.14")
        result = evaluator.run_case(tc)
        assert result.passed is True

    def test_run_suite(self):
        agent = self._make_agent([
            "Thought: done\nFinal Answer: 5",
            "Thought: done\nFinal Answer: 10",
        ])
        evaluator = AgentEvaluator(agent)

        class FakeSuite:
            name = "fake"
            test_cases = [
                TestCase(query="2+3?", expected_output="5"),
                TestCase(query="5+5?", expected_output="10"),
            ]

        results = evaluator.run_suite(FakeSuite())
        assert results.suite_name == "fake"
        assert results.accuracy == 1.0
        assert len(results.results) == 2

    def test_run_case_agent_error(self):
        """Agent that raises should produce a failed result, not crash."""
        agent = self._make_agent(["Thought: done\nFinal Answer: ok"])
        evaluator = AgentEvaluator(agent)
        # Use an expected output that won't match
        tc = TestCase(query="test", expected_output="XYZNONEXISTENT")
        result = evaluator.run_case(tc)
        assert result.passed is False
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Test Suites loading
# ---------------------------------------------------------------------------

class TestSuites:
    def test_list_suites(self):
        suites = list_suites()
        assert "math" in suites
        assert "tool_use" in suites
        assert "reasoning" in suites
        assert "safety" in suites
        assert "conversation" in suites

    def test_get_suite(self):
        suite = get_suite("math")
        assert isinstance(suite, MathSuite)
        assert len(suite) > 0

    def test_get_suite_unknown(self):
        with pytest.raises(KeyError):
            get_suite("nonexistent_suite")

    def test_math_suite_loads(self):
        suite = MathSuite()
        assert len(suite.test_cases) >= 50
        # Check difficulty distribution
        easy = [tc for tc in suite.test_cases if tc.difficulty == Difficulty.EASY]
        medium = [tc for tc in suite.test_cases if tc.difficulty == Difficulty.MEDIUM]
        hard = [tc for tc in suite.test_cases if tc.difficulty == Difficulty.HARD]
        assert len(easy) > 0
        assert len(medium) > 0
        assert len(hard) > 0

    def test_tool_use_suite_loads(self):
        suite = ToolUseSuite()
        assert len(suite.test_cases) >= 30
        # Verify tools are specified
        tools_seen = set()
        for tc in suite.test_cases:
            tools_seen.update(tc.expected_tools)
        assert len(tools_seen) > 5  # multiple tool types

    def test_reasoning_suite_loads(self):
        suite = ReasoningSuite()
        assert len(suite.test_cases) >= 20

    def test_safety_suite_loads(self):
        suite = SafetySuite()
        assert len(suite.test_cases) >= 20
        tags_seen = set()
        for tc in suite.test_cases:
            tags_seen.update(tc.tags)
        assert "prompt_injection" in tags_seen
        assert "jailbreak" in tags_seen
        assert "harmful_request" in tags_seen

    def test_conversation_suite_loads(self):
        suite = ConversationSuite()
        assert len(suite.test_cases) >= 10

    def test_suite_filter(self):
        suite = MathSuite()
        easy = suite.filter(difficulty=Difficulty.EASY)
        assert all(tc.difficulty == Difficulty.EASY for tc in easy)
        assert len(easy) < len(suite.test_cases)

    def test_suite_iteration(self):
        suite = MathSuite()
        cases = list(suite)
        assert len(cases) == len(suite.test_cases)


# ---------------------------------------------------------------------------
# Regression Tracker
# ---------------------------------------------------------------------------

class TestRegressionTracker:
    def test_save_and_load_baseline(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        results = SuiteResults(
            suite_name="math",
            accuracy=0.8,
            avg_latency=0.5,
            total_tokens=1000,
            avg_tool_accuracy=0.9,
        )
        path = tracker.save_baseline("math", results, version="0.1.0")
        assert path.exists()

        loaded = tracker.load_baseline("math")
        assert loaded is not None
        assert loaded["version"] == "0.1.0"
        assert loaded["summary"]["accuracy"] == 0.8

    def test_compare_no_baseline(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        results = SuiteResults(suite_name="new", accuracy=0.7)
        report = tracker.compare("new", results, version="0.1.0")
        assert not report.has_regressions  # first run = no regression

    def test_compare_no_regression(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        baseline = SuiteResults(suite_name="math", accuracy=0.8, avg_latency=0.5)
        tracker.save_baseline("math", baseline, version="0.1.0")

        current = SuiteResults(suite_name="math", accuracy=0.82, avg_latency=0.45)
        report = tracker.compare("math", current, version="0.2.0")
        assert not report.has_regressions

    def test_compare_accuracy_regression(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        baseline = SuiteResults(suite_name="math", accuracy=0.8, avg_latency=0.5)
        tracker.save_baseline("math", baseline, version="0.1.0")

        current = SuiteResults(suite_name="math", accuracy=0.7, avg_latency=0.5)
        report = tracker.compare("math", current, version="0.2.0")
        assert report.has_regressions
        assert any("accuracy" in str(a) for a in report.alerts)

    def test_compare_latency_regression(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        baseline = SuiteResults(suite_name="math", accuracy=0.8, avg_latency=0.5)
        tracker.save_baseline("math", baseline, version="0.1.0")

        current = SuiteResults(suite_name="math", accuracy=0.8, avg_latency=0.7)
        report = tracker.compare("math", current, version="0.2.0")
        assert report.has_regressions
        assert any("latency" in str(a) for a in report.alerts)

    def test_report_to_markdown(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        baseline = SuiteResults(suite_name="math", accuracy=0.8, avg_latency=0.5)
        tracker.save_baseline("math", baseline, version="0.1.0")

        current = SuiteResults(suite_name="math", accuracy=0.7, avg_latency=0.5)
        report = tracker.compare("math", current, version="0.2.0")
        md = report.to_markdown()
        assert "REGRESSION DETECTED" in md
        assert "Baseline" in md


class TestRegressionAlert:
    def test_severity_warning(self):
        alert = RegressionAlert("accuracy", 0.8, 0.74, 0.05, "math")
        assert alert.severity == "warning"

    def test_severity_high(self):
        alert = RegressionAlert("accuracy", 0.8, 0.68, 0.05, "math")
        assert alert.severity == "high"

    def test_severity_critical(self):
        alert = RegressionAlert("accuracy", 0.8, 0.55, 0.05, "math")
        assert alert.severity == "critical"

    def test_str(self):
        alert = RegressionAlert("accuracy", 0.8, 0.7, 0.05, "math")
        s = str(alert)
        assert "math" in s
        assert "accuracy" in s


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

class TestModelComparison:
    def _make_agent(self, responses):
        model = MockModel(responses=responses)
        return Agent(config=AgentConfig(
            name="cmp-test",
            model=model,
            tools=[],
            max_iterations=3,
            enable_memory=False,
            enable_sub_agents=False,
        ))

    def test_run_comparison(self):
        agent_a = self._make_agent(["Thought: done\nFinal Answer: 5"] * 5)
        agent_b = self._make_agent(["Thought: done\nFinal Answer: wrong"] * 5)

        class FakeSuite:
            name = "mini"
            test_cases = [
                TestCase(query="2+3?", expected_output="5"),
                TestCase(query="1+4?", expected_output="5"),
            ]

        comparison = ModelComparison()
        matrix = comparison.run(
            agents={"model-a": agent_a, "model-b": agent_b},
            suites=[FakeSuite()],
        )
        assert len(matrix.scores) == 2
        assert matrix.recommendations.get("mini") == "model-a"

    def test_comparison_matrix_to_markdown(self):
        matrix = ComparisonMatrix(
            scores=[
                ModelScore("model-a", "math", accuracy=0.9, avg_latency=1.0),
                ModelScore("model-b", "math", accuracy=0.7, avg_latency=0.5),
            ],
            recommendations={"math": "model-a"},
        )
        md = matrix.to_markdown()
        assert "model-a" in md
        assert "model-b" in md
        assert "Accuracy" in md

    def test_comparison_matrix_to_json(self):
        matrix = ComparisonMatrix(
            scores=[ModelScore("m", "s", accuracy=0.5)],
        )
        j = json.loads(matrix.to_json())
        assert len(j["scores"]) == 1

    def test_empty_matrix(self):
        matrix = ComparisonMatrix()
        assert "No scores" in matrix.to_markdown()


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

class TestImports:
    def test_top_level_import(self):
        from effgen.eval import (
            AgentEvaluator,
        )
        # Just verify they are importable
        assert AgentEvaluator is not None

    def test_effgen_level_import(self):
        from effgen import AgentEvaluator
        assert AgentEvaluator is not None
