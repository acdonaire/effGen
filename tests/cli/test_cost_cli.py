"""Tests for the 'effgen cost' CLI subcommand."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from effgen.models import _cost as cost_mod

if TYPE_CHECKING:
    from effgen.cli import CLIInterface


@pytest.fixture(autouse=True)
def isolated_budget_path(tmp_path, monkeypatch):
    """Keep CLI budget tests away from the real user config."""
    monkeypatch.setattr(cost_mod, "_BUDGET_CONFIG_PATH", tmp_path / "budget.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cli() -> "CLIInterface":
    """Return a CLIInterface with a no-op console so tests don't print to stdout."""
    from effgen.cli import CLIInterface
    cli = CLIInterface()
    cli.console = None  # suppress Rich output
    return cli


def _args(**kwargs):
    """Build a minimal argparse-like namespace."""
    ns = MagicMock()
    # argparse always provides these flags as real booleans; default them so the
    # MagicMock doesn't auto-vivify them as truthy mocks.
    ns.output_json = False
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def _clear_budget():
    try:
        cost_mod._BUDGET_CONFIG_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _run_cost_cmd(subcmd: str | None, store):
    """Call _handle_cost_command, injecting *store* via patching the module import."""
    from effgen.cli import _handle_cost_command
    from effgen.models import _cost_store as cost_store_mod
    cli = _make_cli()
    args = _args(cost_command=subcmd)
    with patch.object(cost_store_mod, "SQLiteCostStore", return_value=store):
        with patch("effgen.models._cost_store.SQLiteCostStore", return_value=store):
            return _handle_cost_command(args, cli)


# ---------------------------------------------------------------------------
# set-budget / clear-budget
# ---------------------------------------------------------------------------

class TestBudgetManagementCLI:
    def teardown_method(self):
        _clear_budget()

    def test_set_budget_via_cost_subcommand(self, tmp_path):
        """'effgen cost set-budget 2.5' writes budget.json."""
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="set-budget", amount=2.5)
        budget_path = tmp_path / "budget.json"
        with patch("effgen.models._cost._BUDGET_CONFIG_PATH", budget_path):
            code = _handle_cost_command(args, cli)
        assert code == 0
        cfg = json.loads(budget_path.read_text())
        assert cfg.get("daily") == 2.5

    def test_config_set_budget_daily(self, tmp_path):
        """'effgen config set budget.daily 1.5' writes the budget file."""
        cli = _make_cli()
        args = _args(config_command="set", key="budget.daily", value="1.5")
        budget_path = tmp_path / "budget.json"
        with patch("effgen.models._cost._BUDGET_CONFIG_PATH", budget_path):
            cli._config_set(args)
        cfg = json.loads(budget_path.read_text())
        assert cfg.get("daily") == 1.5

    def test_config_set_budget_monthly(self, tmp_path):
        cli = _make_cli()
        args = _args(config_command="set", key="budget.monthly", value="10")
        budget_path = tmp_path / "budget.json"
        with patch("effgen.models._cost._BUDGET_CONFIG_PATH", budget_path):
            cli._config_set(args)
        cfg = json.loads(budget_path.read_text())
        assert cfg.get("monthly") == 10.0

    def test_config_set_unknown_key_no_crash(self):
        cli = _make_cli()
        args = _args(config_command="set", key="unknown.key", value="xyz")
        cli._config_set(args)  # Should not raise

    def test_clear_budget(self):
        """'effgen cost clear-budget' removes the budget limits."""
        cost_mod._BUDGET_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cost_mod._BUDGET_CONFIG_PATH.write_text(json.dumps({"daily": 5.0}))

        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="clear-budget")
        code = _handle_cost_command(args, cli)
        assert code == 0
        if cost_mod._BUDGET_CONFIG_PATH.exists():
            cfg = json.loads(cost_mod._BUDGET_CONFIG_PATH.read_text())
            assert "daily" not in cfg


# ---------------------------------------------------------------------------
# Cost report subcommands
# ---------------------------------------------------------------------------

class TestCostReportCLI:
    """Test effgen cost today/week/by-provider output generation."""

    def setup_method(self):
        _clear_budget()

    def teardown_method(self):
        _clear_budget()

    def test_today_empty_store(self):
        """'effgen cost today' with no data returns exit code 0."""
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="today")
        with patch("effgen.models._cost_store.SQLiteCostStore.__init__", return_value=None):
            with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
                code = _handle_cost_command(args, cli)
        assert code == 0

    def test_week_empty_store(self):
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="week")
        with patch("effgen.models._cost_store.SQLiteCostStore.query_week", return_value=[]):
            code = _handle_cost_command(args, cli)
        assert code == 0

    def test_by_provider_empty_store(self):
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="by-provider")
        with patch("effgen.models._cost_store.SQLiteCostStore.query_all", return_value=[]):
            code = _handle_cost_command(args, cli)
        assert code == 0

    def test_by_provider_groups_models(self, capsys):
        from effgen.cli import _handle_cost_command
        from effgen.models._cost_store import CostEvent
        cli = _make_cli()
        args = _args(cost_command="by-provider")
        events = [
            CostEvent("openai", "gpt-4o-mini", 10, 5, 0.001, time.time()),
            CostEvent("openai", "gpt-4o", 20, 5, 0.002, time.time()),
            CostEvent("groq", "llama-3.1-8b-instant", 5, 5, 0.0, time.time()),
        ]
        with patch("effgen.models._cost_store.SQLiteCostStore.query_all", return_value=events):
            code = _handle_cost_command(args, cli)
        out = capsys.readouterr().out
        assert code == 0
        assert out.count("openai") == 1
        assert "all models" in out
        assert "$   0.003000" in out

    def test_today_with_real_inmemory_store(self):
        """Actually use an in-memory store and verify exit code 0."""
        from effgen.cli import _handle_cost_command
        from effgen.models._cost_store import SQLiteCostStore
        cli = _make_cli()
        # Use real in-memory store and patch the constructor in the handler
        store = SQLiteCostStore(":memory:")
        store.insert("groq", "llama-3.1-8b-instant", 100, 50, 0.0, time.time())
        store.insert("openai", "gpt-4o-mini", 200, 80, 0.00005, time.time())
        args = _args(cost_command="today")
        with patch("effgen.models._cost_store.SQLiteCostStore") as mock_cls:
            mock_cls.return_value = store
            code = _handle_cost_command(args, cli)
        assert code == 0

    def test_default_subcommand_shows_today(self):
        """effgen cost with no subcommand (None) defaults to today."""
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command=None)
        with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
            code = _handle_cost_command(args, cli)
        assert code == 0

    def test_invalid_subcommand_returns_error(self):
        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="nonexistent")
        with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
            code = _handle_cost_command(args, cli)
        assert code == 1

    def test_budget_display_with_configured_budget(self, tmp_path):
        """Budget bar appears in output when daily budget is set."""
        budget_cfg = tmp_path / "budget.json"
        budget_cfg.write_text(json.dumps({"daily": 5.0}))

        from effgen.cli import _handle_cost_command
        cli = _make_cli()
        args = _args(cost_command="today")
        with patch("effgen.models._cost._BUDGET_CONFIG_PATH", budget_cfg):
            with patch("effgen.models._cost_store.SQLiteCostStore.query_today", return_value=[]):
                code = _handle_cost_command(args, cli)
        assert code == 0


# ---------------------------------------------------------------------------
# BudgetExceededError integration
# ---------------------------------------------------------------------------

class TestBudgetExceededIntegration:
    """Tests that BudgetExceededError is properly classified by retry policy."""

    def test_budget_exceeded_error_is_retriable(self):
        from effgen.models.errors import BudgetExceededError
        from effgen.models.routing.retry import RetryPolicy
        policy = RetryPolicy()
        err = BudgetExceededError(1.0, 2.0, "daily", "openai", "gpt-4o")
        assert policy.is_retriable(err)

    def test_budget_exceeded_not_in_non_retriable(self):
        from effgen.models.errors import BudgetExceededError
        from effgen.models.routing.retry import _NON_RETRIABLE
        assert not isinstance(BudgetExceededError(1.0, 2.0, "daily"), _NON_RETRIABLE)

    def test_budget_exceeded_failover_reason_daily(self):
        from effgen.models.errors import BudgetExceededError
        from effgen.models.router import _exc_to_reason
        err = BudgetExceededError(1.0, 2.0, "daily", "openai", "gpt-4o")
        assert _exc_to_reason(err) == "budget_exceeded_daily"

    def test_budget_exceeded_failover_reason_monthly(self):
        from effgen.models.errors import BudgetExceededError
        from effgen.models.router import _exc_to_reason
        err = BudgetExceededError(1.0, 2.0, "monthly", "together", "llama-3")
        assert _exc_to_reason(err) == "budget_exceeded_monthly"

    def test_budget_exceeded_exported_from_models(self):
        from effgen.models import BudgetExceededError
        assert BudgetExceededError is not None

    def test_budget_exceeded_error_message_format(self):
        from effgen.models.errors import BudgetExceededError
        err = BudgetExceededError(1.0, 1.5, "daily", "openai", "gpt-4o")
        msg = str(err)
        assert "daily" in msg.lower()
        assert "1.5" in msg or "1.50" in msg
        assert "failover" in msg.lower()
