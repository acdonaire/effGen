"""
Shared test fixtures for the effGen test suite.
"""

import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import pytest

# Suppress ImportWarning from optional dependencies before importing effgen
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Secret scrubbing for traceback locals (defense in depth: even if a future
# dev re-enables --showlocals, real API key values must not land in test logs).
# ---------------------------------------------------------------------------

_SECRET_ENV_PATTERNS = (
    "API_KEY", "API_TOKEN", "SECRET", "TOKEN", "PASSWORD",
)


def _collect_secret_values() -> tuple[str, ...]:
    values = []
    for name, val in os.environ.items():
        if not val or len(val) < 8:
            continue
        if any(p in name.upper() for p in _SECRET_ENV_PATTERNS):
            values.append(val)
    # Sort longest-first so substrings don't shadow longer matches
    return tuple(sorted(set(values), key=len, reverse=True))


def pytest_exception_interact(node, call, report):
    """Redact known secret values from rendered traceback text."""
    secrets = _collect_secret_values()
    if not secrets:
        return
    longrepr = getattr(report, "longrepr", None)
    if longrepr is None:
        return
    text = str(longrepr)
    redacted = text
    for s in secrets:
        redacted = redacted.replace(s, "***REDACTED***")
    if redacted != text:
        # Replace longrepr with the scrubbed string so terminal + xml reporters
        # both see the redacted version.
        report.longrepr = redacted

# Ensure effgen package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load test credentials without printing or inspecting values. Project-local
# values are useful for live integration tests; user-level values remain a
# fallback for developer machines.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
    load_dotenv(Path.home() / ".effgen" / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Isolate the cost-tracker SQLite DB from the user's real ~/.effgen/costs.sqlite
# during tests. Many unit tests record large fixture values (1M tokens etc.)
# and those must not pollute the user's actual cost history.
# ---------------------------------------------------------------------------
_TEST_COST_DB_DIR: str | None = None
if "EFFGEN_COST_DB" not in os.environ:
    _TEST_COST_DB_DIR = tempfile.mkdtemp(prefix="effgen_test_costs_")
    os.environ["EFFGEN_COST_DB"] = str(Path(_TEST_COST_DB_DIR) / "costs.sqlite")


def pytest_sessionfinish(session, exitstatus):
    """Remove the isolated cost DB created for this pytest session."""
    _ = (session, exitstatus)
    if _TEST_COST_DB_DIR:
        shutil.rmtree(_TEST_COST_DB_DIR, ignore_errors=True)

from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin import Calculator, DateTimeTool, JSONTool, TextProcessingTool
from tests.fixtures.mock_models import MockModel, MockStreamingModel, MockToolCallingModel

# ---------------------------------------------------------------------------
# Mock Model Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """A simple mock model that returns 'Final Answer: 42'."""
    return MockModel(responses=[
        "Thought: I know the answer.\nFinal Answer: 42"
    ])


@pytest.fixture
def mock_tool_model():
    """Mock model that calls calculator then gives final answer."""
    return MockToolCallingModel(tool_sequence=[
        {
            "thought": "I need to calculate this.",
            "action": "calculator",
            "action_input": '{"expression": "2 + 2"}',
        },
        {
            "thought": "I now know the final answer.",
            "action": "Final Answer",
            "action_input": "The answer is 4.",
        },
    ])


@pytest.fixture
def mock_streaming_model():
    """Mock model for streaming tests."""
    return MockStreamingModel(responses=[
        "Thought: Let me think.\nFinal Answer: Hello, world!"
    ])


# ---------------------------------------------------------------------------
# Tool Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calculator():
    return Calculator()


@pytest.fixture
def datetime_tool():
    return DateTimeTool()


@pytest.fixture
def json_tool():
    return JSONTool()


@pytest.fixture
def text_tool():
    return TextProcessingTool()


# ---------------------------------------------------------------------------
# Agent Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_agent(mock_model):
    """Agent with no tools."""
    config = AgentConfig(
        name="test-basic",
        model=mock_model,
        tools=[],
        max_iterations=3,
        enable_memory=False,
        enable_sub_agents=False,
    )
    return Agent(config=config)


@pytest.fixture
def tool_agent(mock_tool_model, calculator):
    """Agent with calculator tool."""
    config = AgentConfig(
        name="test-tool",
        model=mock_tool_model,
        tools=[calculator],
        max_iterations=5,
        enable_memory=False,
        enable_sub_agents=False,
    )
    return Agent(config=config)


@pytest.fixture
def multi_tool_agent(mock_model, calculator, datetime_tool, json_tool):
    """Agent with multiple tools."""
    config = AgentConfig(
        name="test-multi",
        model=mock_model,
        tools=[calculator, datetime_tool, json_tool],
        max_iterations=5,
        enable_memory=False,
        enable_sub_agents=False,
    )
    return Agent(config=config)


# ---------------------------------------------------------------------------
# Temp Directory Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp(prefix="effgen_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# GPU Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def free_gpu_id():
    """Find a free GPU with minimal memory usage."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        # Find GPU with least memory usage
        min_mem = float("inf")
        best_gpu = 0
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i)
            if mem_used < min_mem:
                min_mem = mem_used
                best_gpu = i
        return best_gpu
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Fixtures directory path
# ---------------------------------------------------------------------------

@pytest.fixture
def fixtures_dir():
    """Path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def knowledge_base_dir(fixtures_dir):
    """Path to the knowledge base fixtures."""
    return fixtures_dir / "knowledge_base"


# ---------------------------------------------------------------------------
# GPU test isolation
# ---------------------------------------------------------------------------
#
# e2e tests load models with bitsandbytes 4-bit quantization. The bnb kernels
# leave CUDA state that historically caused the integration streaming tests
# (TextIteratorStreamer) to deadlock when they ran AFTER e2e in the same
# pytest session. Two-part defense:
#
#   1. Reorder collection so all tests/e2e/* items run LAST, after unit +
#      integration. This way any CUDA-state corruption happens after
#      everything downstream has already completed.
#   2. Between each module we force a CUDA cleanup via an autouse
#      module-scoped fixture. Cheap when CUDA is absent; essential when
#      GPU tests and streaming tests coexist.


# Directory -> marker auto-application. Keeps the marker surface authoritative
# (so `-m security` / `-m deployment` actually select the right suites and map to
# CI jobs) without sprinkling `pytestmark` into dozens of files. Explicit markers
# on individual tests still apply and stack on top of these.
_DIR_MARKERS = {
    "/tests/security/": "security",
    "/tests/deploy/": "deployment",
}


def _norm_path(item) -> str:
    return str(item.fspath).replace("\\", "/")


def pytest_collection_modifyitems(config, items):
    """Auto-mark by directory and run tests/e2e/* last.

    e2e tests load models with bitsandbytes 4-bit quantization; their CUDA state
    historically deadlocked the integration streaming tests when they ran first,
    so e2e is sorted to the end of the session.
    """
    for item in items:
        p = _norm_path(item)
        for needle, marker in _DIR_MARKERS.items():
            if needle in p and marker not in {m.name for m in item.iter_markers()}:
                item.add_marker(getattr(pytest.mark, marker))

    def _bucket(item):
        p = _norm_path(item)
        if "/tests/e2e/" in p:
            return 2
        if "/tests/integration/" in p:
            return 1
        return 0

    items.sort(key=_bucket)


@pytest.fixture(autouse=True, scope="module")
def _cuda_state_hygiene():
    """Flush CUDA caches between modules to reduce state leakage."""
    yield
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
