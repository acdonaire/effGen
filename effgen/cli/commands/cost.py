"""The ``effgen cost`` command: spend dashboard and budget management.

``_main`` parses arguments and dispatches; it imports this at module scope and
re-exports it (``effgen.cli._handle_cost_command`` resolves through the chain).
Holds the budget set/clear round-trip, the per-provider/model spend aggregation,
the priced/free/unpriced cost labels (never a misleading ``$0`` for an unpriced
model), and the budget-burn bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from effgen.cli.commands.report import _write_html_report_arg, _write_result_artifact
from effgen.ui.tables import console_is_interactive, empty_state

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface


def _handle_cost_command(args, cli: "CLIInterface") -> int:
    """Handle the 'effgen cost' subcommand: spend dashboard and budget management."""
    import json as _json

    try:
        from effgen.models._cost_store import SQLiteCostStore
    except ImportError:
        cli.print_error("Cost store not available. Please reinstall effGen.")
        return 1

    cost_cmd = getattr(args, 'cost_command', None)

    # Budget management subcommands
    from effgen.models._cost import _budget_config_path, format_usd
    budget_path = _budget_config_path()

    if cost_cmd == 'set-budget':
        amount = float(args.amount)
        budget_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if budget_path.exists():
            try:
                existing = _json.loads(budget_path.read_text())
            except Exception:
                pass
        existing['daily'] = amount
        budget_path.write_text(_json.dumps(existing, indent=2))
        cli.print_success(f"Daily budget set to {format_usd(amount)} USD")
        return 0

    if cost_cmd == 'clear-budget':
        if budget_path.exists():
            try:
                cfg = _json.loads(budget_path.read_text())
                cfg.pop('daily', None)
                cfg.pop('monthly', None)
                budget_path.write_text(_json.dumps(cfg, indent=2))
                cli.print_success("Budget limits cleared.")
            except Exception as e:
                cli.print_error(f"Failed to clear budget: {e}")
                return 1
        else:
            cli.print("No budget configured.")
        return 0

    # Spend-report subcommands
    store = SQLiteCostStore()

    # period_days is the window the spend covers, so a budget comparison can be
    # scaled to it. Lifetime spans no fixed window, so it carries None.
    if cost_cmd == 'today' or cost_cmd is None:
        events = store.query_today()
        period_label = "Last 24 hours"
        period_days: int | None = 1
    elif cost_cmd == 'week':
        events = store.query_week()
        period_label = "Last 7 days"
        period_days = 7
    elif cost_cmd == 'by-provider':
        events = store.query_all()
        period_label = "Lifetime"
        period_days = None
    else:
        cli.print_error(f"Unknown cost command: {cost_cmd}")
        cli.print("Usage: effgen cost [today|week|by-provider|set-budget|clear-budget]")
        return 1

    # Aggregate events by (provider, model), except by-provider which intentionally
    # collapses all models for each provider into one lifetime row.
    group_by_provider = cost_cmd == 'by-provider'
    agg: dict[tuple[str, str], dict] = {}
    for ev in events:
        model_label = "all models" if group_by_provider else ev.model
        key = (ev.provider, model_label)
        if key not in agg:
            agg[key] = {
                'provider': ev.provider,
                'model': model_label,
                'requests': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'cost_usd': 0.0,
            }
        agg[key]['requests'] += 1
        agg[key]['prompt_tokens'] += ev.prompt_tokens
        agg[key]['completion_tokens'] += ev.completion_tokens
        agg[key]['cost_usd'] += ev.cost_usd

    rows = sorted(agg.values(), key=lambda r: r['cost_usd'], reverse=True)
    total_cost = sum(r['cost_usd'] for r in rows)
    total_requests = sum(r['requests'] for r in rows)

    # Cost label: a genuine free tier reads "free" and a model with no
    # published price reads "unpriced", instead of a misleading "$0.000000".
    from effgen.models._cost import pricing_status as _pricing_status

    def _cost_label(row: dict) -> str:
        cost = row['cost_usd']
        if cost > 0 or row['model'] == 'all models':
            return f"${cost:.6f}"
        status = _pricing_status(row['provider'], row['model'])
        if status == 'free':
            return 'free'
        if status == 'unpriced':
            return 'unpriced'
        return f"${cost:.6f}"

    def _row_cost(row: dict) -> float | None:
        """The row's spend, or ``None`` when the model publishes no price.

        The ledger stores a non-null number for every call, so a row on an
        unpriced model reads ``0.0`` there. Reporting that as the spend would
        state a price nobody published; a reader of the JSON document gets the
        same answer the table's label gives.
        """
        if _cost_label(row) == 'unpriced':
            return None
        return round(row['cost_usd'], 8)

    # Load budget for display
    budget_cfg = {}
    if budget_path.exists():
        try:
            budget_cfg = _json.loads(budget_path.read_text())
        except Exception:
            pass
    daily_budget = budget_cfg.get('daily')

    spend_document = {
        "period": period_label,
        "period_days": period_days,
        "total_requests": total_requests,
        "total_cost_usd": round(total_cost, 8),
        "daily_budget_usd": daily_budget,
        "rows": [
            {
                "provider": r["provider"],
                "model": r["model"],
                "requests": r["requests"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "cost_usd": _row_cost(r),
                "cost_label": _cost_label(r),
            }
            for r in rows
        ],
    }

    # JSON output — machine-readable spend report. Keep stdout to the JSON
    # document alone, so any file-written notice goes to stderr.
    json_mode = getattr(args, 'output_json', False)
    if json_mode:
        print(_json.dumps(spend_document, indent=2))
        cli._human_to_stderr = True

    # File output — the extension chooses the format; --report always writes HTML.
    if getattr(args, 'output', None):
        _write_result_artifact(
            args.output,
            cli=cli,
            data=spend_document,
            kind="cost",
            json_text=_json.dumps(spend_document, indent=2),
        )
    _write_html_report_arg(args, cli=cli, data=spend_document, kind="cost")

    if json_mode:
        return 0

    # Empty state: a next step instead of a blank table. The interactive terminal
    # gets the shared empty-state block; a piped/redirected stream keeps the exact
    # plain lines so its bytes are unchanged.
    if not rows:
        if console_is_interactive(cli.console):
            empty_state(
                cli.console,
                title=f"effGen Cost Summary — {period_label}",
                message="No spend recorded yet.",
                hints=[
                    (
                        "Run an agent to start tracking — e.g. "
                        "effgen run \"What is 2+2?\" -m gpt-5-nano --provider openai"
                    ),
                    "Set a daily cap with effgen cost set-budget 1.00",
                ],
            )
        else:
            cli.print_header(f"effGen Cost Summary — {period_label}")
            cli.print("No spend recorded yet. 🎉")
            cli.print("Run an agent to start tracking — e.g. effgen run \"What is 2+2?\" "
                      "-m gpt-5-nano --provider openai")
            cli.print("Then set a cap with: effgen cost set-budget 1.00")
        return 0

    if console_is_interactive(cli.console):
        from rich.table import Table
        table = Table(title=f"effGen Cost Summary — {period_label}", show_footer=True)
        table.add_column("Provider", style="effgen.accent", no_wrap=True)
        # Wrap (fold) long model ids instead of truncating with an ellipsis.
        table.add_column("Model", style="effgen.model", overflow="fold")
        table.add_column("Requests", justify="right")
        table.add_column("Prompt Tokens", justify="right")
        table.add_column("Completion Tokens", justify="right")
        table.add_column("Cost (USD)", style="effgen.cost", justify="right",
                         footer=f"${total_cost:.6f}")

        for r in rows:
            table.add_row(
                r['provider'],
                r['model'],
                str(r['requests']),
                f"{r['prompt_tokens']:,}",
                f"{r['completion_tokens']:,}",
                _cost_label(r),
            )

        cli.console.print(table)
        cli.console.print(f"\n[effgen.label]Total:[/effgen.label] {total_requests} requests  "
                          f"[effgen.cost]${total_cost:.6f} USD[/effgen.cost]")
        if daily_budget is not None and cost_cmd in (None, 'today'):
            ratio = total_cost / daily_budget if daily_budget > 0 else 0
            filled = min(20, max(0, int(ratio * 20)))
            bar = "█" * filled + "░" * (20 - filled)
            role = ("effgen.error" if ratio >= 1.0
                    else "effgen.warning" if ratio >= 0.8 else "effgen.success")
            cli.console.print(
                f"[effgen.label]Daily budget:[/effgen.label] [{role}]{bar}[/{role}] "
                f"{format_usd(total_cost)} / {format_usd(daily_budget)} ({ratio*100:.0f}%)"
            )
    else:
        print(f"\neffGen Cost Summary — {period_label}")
        print("-" * 80)
        print(f"{'Provider':<12} {'Model':<48} {'Reqs':>5} {'Cost (USD)':>12}")
        print("-" * 80)
        for r in rows:
            # Show the full model id (wrap rather than truncate).
            model = r['model']
            cost_label = _cost_label(r)
            if len(model) > 48:
                print(f"{r['provider']:<12} {model}")
                print(f"{'':<12} {'':<48} {r['requests']:>5} {cost_label:>12}")
            else:
                print(f"{r['provider']:<12} {model:<48} {r['requests']:>5} {cost_label:>12}")
        print("-" * 80)
        print(f"{'TOTAL':<12} {'':<48} {total_requests:>5} ${total_cost:>11.6f}")
        if daily_budget is not None and cost_cmd in (None, 'today'):
            ratio = total_cost / daily_budget if daily_budget > 0 else 0
            print(f"\nDaily budget: {format_usd(total_cost)} / {format_usd(daily_budget)} ({ratio*100:.0f}%)")

    return 0
