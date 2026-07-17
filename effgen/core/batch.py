"""
Batch Execution Engine for effGen.

Execute multiple queries in parallel with concurrency control,
retry logic, progress tracking, and file-based I/O.

Usage:
    from effgen.core.batch import BatchRunner, BatchConfig

    runner = BatchRunner(agent)
    results = runner.run(["What is X?", "What is Y?"], config=BatchConfig(max_concurrency=10))

    # Or from files:
    results = runner.run_from_file("queries.jsonl")
    runner.write_results(results, "results.jsonl")
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Field names accepted for the query text of a JSONL/CSV/JSON row, in priority
# order — mirrors the eval/compare loaders so a batch file keyed on any of these
# works without ``--query-field``.
_QUERY_ALIASES = ("query", "input", "prompt", "question", "text")

# File extensions ``write_results`` can serialize a batch to. The CLI checks a
# requested ``--output`` against this set up front so an unsupported extension
# fails before any billed call rather than after the whole run.
SUPPORTED_OUTPUT_FORMATS = (".jsonl", ".json", ".csv")


def _resolve_row_query(obj: dict[str, Any], query_field: str) -> str:
    """Return a row's query text, trying *query_field* then the common aliases.

    The explicitly-requested field wins; when it is absent or empty the aliases
    ``input``/``prompt``/``question``/``text`` are tried in order. Returns an
    empty string when none carry text.
    """
    order = [query_field] + [k for k in _QUERY_ALIASES if k != query_field]
    for key in order:
        val = obj.get(key)
        if val not in (None, ""):
            return str(val)
    return ""


@dataclass
class BatchConfig:
    """Configuration for batch execution.

    Attributes:
        max_concurrency: Maximum number of concurrent agent runs.
        batch_size: Process queries in batches of this size (0 = all at once).
        retry_failed: Number of retries for failed queries.
        timeout_per_item: Timeout in seconds per individual query (0 = no timeout).
        progress_callback: Called with (completed, total) after each query finishes.
        on_result: Called with (index, query, AgentResponse) for each result.
    """

    max_concurrency: int = 5
    batch_size: int = 0
    retry_failed: int = 1
    timeout_per_item: float = 120.0
    progress_callback: Callable[[int, int], None] | None = None
    on_result: Callable[[int, str, Any], None] | None = None


def _row_metrics(resp: Any) -> dict[str, Any]:
    """Pull per-row cost/token figures and a failure reason off a response.

    Reads the same ``metadata`` keys the agent populates after each run
    (``cost_usd``, ``prompt_tokens``/``completion_tokens``/``total_tokens``) and
    the parsed structured object (``metadata["parsed"]``). Returns only the keys
    that are actually present, so an unpriced model contributes no ``cost_usd``.
    """
    out: dict[str, Any] = {}
    if resp is None:
        return out
    meta = getattr(resp, "metadata", None) or {}
    cost = meta.get("cost_usd", meta.get("cost"))
    if isinstance(cost, int | float):
        out["cost_usd"] = float(cost)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = meta.get(key)
        if isinstance(val, int):
            out[key] = val
    if "total_tokens" not in out:
        used = getattr(resp, "tokens_used", 0)
        if isinstance(used, int) and used > 0:
            out["total_tokens"] = used
    return out


def _row_parsed(resp: Any) -> Any:
    """Return the validated structured object for a row as a plain dict/JSON value.

    ``metadata["parsed"]`` is the Pydantic instance produced when a schema was
    requested; serialize it to a dict so it round-trips through JSON/CSV.
    """
    if resp is None:
        return None
    parsed = (getattr(resp, "metadata", None) or {}).get("parsed")
    if parsed is None:
        return None
    if hasattr(parsed, "model_dump"):
        try:
            return parsed.model_dump(mode="json")
        except Exception:  # noqa: BLE001 - fall through to other shapes
            pass
    if hasattr(parsed, "dict"):
        try:
            return parsed.dict()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(parsed, dict | list | str | int | float | bool):
        return parsed
    return str(parsed)


def _row_error(resp: Any) -> str | None:
    """Return a failure reason for a row, or ``None`` when it succeeded.

    Prefers the structured ``metadata["error"]`` (a dict carrying ``message`` /
    ``type``) the agent attaches on failure, then a top-level ``reason``, then
    the response text — so a downstream consumer can always tell *why* a row
    failed instead of guessing from an empty ``output``.
    """
    if resp is None:
        return "no response produced"
    if getattr(resp, "success", False):
        return None
    meta = getattr(resp, "metadata", None) or {}
    err = meta.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("type")
        if msg:
            return str(msg)
    if isinstance(err, str) and err:
        return err
    reason = meta.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    text = getattr(resp, "output", "") or ""
    return text.strip() or "failed"


@dataclass
class BatchResult:
    """Container for batch execution results.

    Attributes:
        results: List of AgentResponse objects (one per query, in order).
        total: Total number of queries.
        succeeded: Number of successful queries.
        failed: Number of failed queries.
        total_time: Wall-clock time for the entire batch.
        per_query_times: Execution time per query (seconds).
        total_cost_usd: Summed ``cost_usd`` across priced rows (``None`` when no
            row reported a cost, e.g. an unpriced/local model).
        total_tokens: Summed total tokens across all rows.
        total_prompt_tokens: Summed prompt tokens across all rows.
        total_completion_tokens: Summed completion tokens across all rows.
    """

    results: list[Any] = field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    total_time: float = 0.0
    per_query_times: list[float] = field(default_factory=list)
    total_cost_usd: float | None = None
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def success_rate(self) -> float:
        """Return fraction of successful queries."""
        return self.succeeded / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate": round(self.success_rate(), 4),
            "total_time": round(self.total_time, 2),
            "total_cost_usd": (
                round(self.total_cost_usd, 8)
                if self.total_cost_usd is not None else None
            ),
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }


class BatchRunner:
    """Execute multiple queries through an Agent in parallel.

    Uses asyncio.Semaphore for concurrency control and supports
    retry, timeout, progress tracking, and file I/O.
    """

    def __init__(self, agent: Any) -> None:
        """
        Args:
            agent: An effgen Agent instance with a .run() method.
        """
        self.agent = agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        queries: list[str],
        config: BatchConfig | None = None,
        **run_kwargs: Any,
    ) -> BatchResult:
        """Run a list of queries through the agent.

        Args:
            queries: List of query strings.
            config: Batch configuration (defaults applied if None).
            **run_kwargs: Extra keyword arguments forwarded to agent.run().

        Returns:
            BatchResult with all responses in input order.
        """
        config = config or BatchConfig()
        return self._run_sync(queries, config, run_kwargs)

    def run_from_file(
        self,
        path: str | Path,
        config: BatchConfig | None = None,
        query_field: str = "query",
        strict: bool = False,
        on_skip: Callable[[int, str], None] | None = None,
        **run_kwargs: Any,
    ) -> BatchResult:
        """Load queries from a JSONL or CSV file and run them.

        For JSONL, each line must be a JSON object with *query_field* key.
        For CSV, the column named *query_field* is used.

        Args:
            path: Path to JSONL or CSV file.
            config: Batch configuration.
            query_field: Field/column name containing the query text.
            strict: Hard-fail on the first malformed input line instead of
                skipping it (see :meth:`_read_queries`).
            on_skip: Called with ``(line_number, message)`` for each skipped
                malformed line.
            **run_kwargs: Extra keyword arguments forwarded to agent.run().
        """
        queries = self._read_queries(
            Path(path), query_field, strict=strict, on_skip=on_skip,
        )
        return self.run(queries, config=config, **run_kwargs)

    @staticmethod
    def write_results(
        batch_result: BatchResult,
        path: str | Path,
        query_list: list[str] | None = None,
        excel_bom: bool = False,
    ) -> None:
        """Write batch results to a JSONL or CSV file.

        Format is inferred from the file extension (.jsonl, .csv, .json).

        Args:
            excel_bom: When the target is ``.csv``, prepend a UTF-8 BOM
                (``utf-8-sig``) so Excel on Windows opens non-Latin scripts
                (Arabic, CJK, Devanagari, ...) correctly on double-click
                instead of misreading them as another encoding. The file
                stays valid UTF-8 either way; this only affects the CSV path.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        rows = [
            BatchRunner._result_row(
                i, resp, query_list[i] if query_list and i < len(query_list) else None,
            )
            for i, resp in enumerate(batch_result.results)
        ]

        if suffix == ".jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        elif suffix == ".csv":
            if rows:
                # A stable, union column set so every row lines up even when
                # some carry a cost/parsed value and others don't.
                fieldnames: list[str] = []
                for row in rows:
                    for key in row:
                        if key not in fieldnames:
                            fieldnames.append(key)
                csv_encoding = "utf-8-sig" if excel_bom else "utf-8"
                with open(path, "w", encoding=csv_encoding, newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(BatchRunner._csv_flatten(row))
        else:
            raise ValueError(
                f"Unsupported output format: {suffix}. "
                f"Use one of: {', '.join(SUPPORTED_OUTPUT_FORMATS)}."
            )

        logger.info("Wrote %d results to %s", len(rows), path)

    @staticmethod
    def _result_row(
        index: int, resp: Any, query: str | None = None,
    ) -> dict[str, Any]:
        """Build one output row, carrying cost/tokens/parsed/error when present.

        The core columns (``index``/``query``/``output``/``success``/
        ``execution_time``) are always written; ``cost_usd``, token counts, the
        validated ``parsed`` object, and a failure ``error`` are added only when
        the row actually has them, so a downstream job never loses spend, token,
        or failure-reason data that the run produced.
        """
        row: dict[str, Any] = {"index": index}
        if query is not None:
            row["query"] = query
        if resp is not None:
            row["output"] = resp.output
            row["success"] = bool(resp.success)
            row["execution_time"] = round(resp.execution_time, 3)
        else:
            row["output"] = ""
            row["success"] = False
            row["execution_time"] = 0.0
        for key, val in _row_metrics(resp).items():
            row[key] = round(val, 8) if key == "cost_usd" else val
        parsed = _row_parsed(resp)
        if parsed is not None:
            row["parsed"] = parsed
        if not row["success"]:
            row["error"] = _row_error(resp)
        return row

    @staticmethod
    def _csv_flatten(row: dict[str, Any]) -> dict[str, Any]:
        """Render nested values (``parsed``) as compact JSON for a CSV cell."""
        flat: dict[str, Any] = {}
        for key, val in row.items():
            if isinstance(val, dict | list):
                flat[key] = json.dumps(val, ensure_ascii=False)
            else:
                flat[key] = val
        return flat

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_sync(
        self,
        queries: list[str],
        config: BatchConfig,
        run_kwargs: dict[str, Any],
    ) -> BatchResult:
        """Bridge sync -> async execution via the centralized helper so the
        behaviour (direct run vs. worker-thread when a loop is already running)
        is identical everywhere and never silently skips the work."""
        from ..utils.async_bridge import run_coroutine_sync

        # Batch jobs can legitimately run long; give the worker-thread bridge a
        # generous ceiling rather than the 120s default.
        return run_coroutine_sync(
            self._run_async(queries, config, run_kwargs), timeout=86400.0
        )

    async def _run_async(
        self,
        queries: list[str],
        config: BatchConfig,
        run_kwargs: dict[str, Any],
    ) -> BatchResult:
        start = time.time()
        total = len(queries)
        results: list[Any] = [None] * total
        per_query_times: list[float] = [0.0] * total
        completed = 0
        lock = asyncio.Lock()

        semaphore = asyncio.Semaphore(config.max_concurrency)

        async def _process_one(idx: int, query: str) -> None:
            nonlocal completed
            async with semaphore:
                resp = await self._run_single_with_retry(
                    query, config, run_kwargs,
                )
                results[idx] = resp
                per_query_times[idx] = resp.execution_time if resp else 0.0
                async with lock:
                    completed += 1
                    if config.progress_callback:
                        config.progress_callback(completed, total)
                    if config.on_result:
                        config.on_result(idx, query, resp)

        # Batch processing
        if config.batch_size > 0:
            for batch_start in range(0, total, config.batch_size):
                batch_end = min(batch_start + config.batch_size, total)
                tasks = [
                    asyncio.create_task(_process_one(i, queries[i]))
                    for i in range(batch_start, batch_end)
                ]
                await asyncio.gather(*tasks)
        else:
            tasks = [
                asyncio.create_task(_process_one(i, q))
                for i, q in enumerate(queries)
            ]
            await asyncio.gather(*tasks)

        elapsed = time.time() - start
        succeeded = sum(1 for r in results if r is not None and r.success)
        failed = total - succeeded

        # Aggregate spend + tokens across every row so a job reports a single
        # figure without post-processing the output file. Cost stays None until
        # at least one row is priced (unpriced/local models contribute tokens
        # only).
        cost_seen = False
        total_cost = 0.0
        tok_total = tok_prompt = tok_completion = 0
        for r in results:
            m = _row_metrics(r)
            if "cost_usd" in m:
                cost_seen = True
                total_cost += m["cost_usd"]
            tok_total += m.get("total_tokens", 0)
            tok_prompt += m.get("prompt_tokens", 0)
            tok_completion += m.get("completion_tokens", 0)

        return BatchResult(
            results=results,
            total=total,
            succeeded=succeeded,
            failed=failed,
            total_time=elapsed,
            per_query_times=per_query_times,
            total_cost_usd=(total_cost if cost_seen else None),
            total_tokens=tok_total,
            total_prompt_tokens=tok_prompt,
            total_completion_tokens=tok_completion,
        )

    async def _run_single_with_retry(
        self,
        query: str,
        config: BatchConfig,
        run_kwargs: dict[str, Any],
    ) -> Any:
        """Run a single query with retry and timeout."""
        from .agent import AgentResponse

        attempts = 1 + config.retry_failed
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                if config.timeout_per_item > 0:
                    resp = await asyncio.wait_for(
                        asyncio.to_thread(self.agent.run, query, **run_kwargs),
                        timeout=config.timeout_per_item,
                    )
                else:
                    resp = await asyncio.to_thread(
                        self.agent.run, query, **run_kwargs,
                    )
                return resp
            except asyncio.TimeoutError:  # noqa: UP041 - distinct class on py3.10
                last_exc = TimeoutError(
                    f"Query timed out after {config.timeout_per_item}s: {query[:80]}"
                )
                logger.warning(
                    "Timeout on attempt %d/%d for query: %s",
                    attempt + 1, attempts, query[:80],
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Error on attempt %d/%d for query '%s': %s",
                    attempt + 1, attempts, query[:80], exc,
                )

        # All retries exhausted — return a failed AgentResponse
        return AgentResponse(
            output=f"Failed after {attempts} attempts: {last_exc}",
            success=False,
            metadata={"error": str(last_exc)},
        )

    @staticmethod
    def _read_queries(
        path: Path,
        query_field: str,
        strict: bool = False,
        on_skip: Callable[[int, str], None] | None = None,
        on_empty: Callable[[int, list[str]], None] | None = None,
    ) -> list[str]:
        """Read queries from a JSONL, CSV, JSON, or plain-text file.

        A dict row's query text is read from *query_field*, falling back to the
        common aliases ``input``/``prompt``/``question``/``text`` when that field
        is absent — so a file keyed on any of those works without
        ``--query-field``.

        A malformed JSONL/JSON line does not abort the whole job. By default the
        bad line is skipped and reported (through ``on_skip`` if given, else a
        logged warning) with the **file path and line number** — not the JSON
        parser's internal character offset. Pass ``strict=True`` to hard-fail on
        the first bad line instead (still naming the file and line).

        Args:
            path: Path to the input file.
            query_field: Field/column name holding the query text.
            strict: Hard-fail on the first malformed line instead of skipping.
            on_skip: Called with ``(line_number, message)`` for each skipped
                line; when ``None``, a warning is logged instead.
            on_empty: Called with ``(line_number, available_keys)`` for each dict
                row that carries no recognized query text, so the caller can name
                the fields it saw.
        """
        suffix = path.suffix.lower()
        queries: list[str] = []

        def _report_bad(lineno: int, exc: Exception) -> None:
            location = f"{path}:{lineno}"
            if strict:
                raise ValueError(f"{location}: malformed line: {exc}") from exc
            if on_skip is not None:
                on_skip(lineno, str(exc))
            else:
                logger.warning("%s: skipping malformed line: %s", location, exc)

        def _resolve(lineno: int, row: dict) -> str:
            text = _resolve_row_query(row, query_field)
            if not text and on_empty is not None:
                on_empty(lineno, sorted(str(k) for k in row.keys()))
            return text

        if suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        _report_bad(lineno, exc)
                        continue
                    if isinstance(obj, str):
                        queries.append(obj)
                    elif isinstance(obj, dict):
                        queries.append(_resolve(lineno, obj))
                    else:
                        queries.append(str(obj))
        elif suffix == ".csv":
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for lineno, row in enumerate(reader, start=1):
                    queries.append(_resolve(lineno, row))
        elif suffix == ".json":
            with open(path, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}: not valid JSON: {exc}") from exc
                if isinstance(data, list):
                    for pos, item in enumerate(data, start=1):
                        if isinstance(item, str):
                            queries.append(item)
                        elif isinstance(item, dict):
                            queries.append(_resolve(pos, item))
                        else:
                            _report_bad(pos, TypeError(
                                f"expected string or object, got {type(item).__name__}"
                            ))
        else:
            # Treat as plain text — one query per line
            with open(path, encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip()]

        if not queries:
            raise ValueError(f"No queries found in {path}")

        logger.info("Loaded %d queries from %s", len(queries), path)
        return queries
