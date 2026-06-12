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
    """

    results: list[Any] = field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    total_time: float = 0.0
    per_query_times: list[float] = field(default_factory=list)

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
        **run_kwargs: Any,
    ) -> BatchResult:
        """Load queries from a JSONL or CSV file and run them.

        For JSONL, each line must be a JSON object with *query_field* key.
        For CSV, the column named *query_field* is used.

        Args:
            path: Path to JSONL or CSV file.
            config: Batch configuration.
            query_field: Field/column name containing the query text.
            **run_kwargs: Extra keyword arguments forwarded to agent.run().
        """
        queries = self._read_queries(Path(path), query_field)
        return self.run(queries, config=config, **run_kwargs)

    @staticmethod
    def write_results(
        batch_result: BatchResult,
        path: str | Path,
        query_list: list[str] | None = None,
    ) -> None:
        """Write batch results to a JSONL or CSV file.

        Format is inferred from the file extension (.jsonl, .csv, .json).
        """
        path = Path(path)
        suffix = path.suffix.lower()

        rows: list[dict[str, Any]] = []
        for i, resp in enumerate(batch_result.results):
            row: dict[str, Any] = {"index": i}
            if query_list and i < len(query_list):
                row["query"] = query_list[i]
            if resp is not None:
                row["output"] = resp.output
                row["success"] = resp.success
                row["execution_time"] = round(resp.execution_time, 3)
            else:
                row["output"] = ""
                row["success"] = False
                row["execution_time"] = 0.0
            rows.append(row)

        if suffix == ".jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        elif suffix == ".csv":
            if rows:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
        else:
            raise ValueError(f"Unsupported output format: {suffix}")

        logger.info("Wrote %d results to %s", len(rows), path)

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

        return BatchResult(
            results=results,
            total=total,
            succeeded=succeeded,
            failed=failed,
            total_time=elapsed,
            per_query_times=per_query_times,
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
            except asyncio.TimeoutError:
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
    def _read_queries(path: Path, query_field: str) -> list[str]:
        """Read queries from a JSONL or CSV file."""
        suffix = path.suffix.lower()
        queries: list[str] = []

        if suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        queries.append(obj)
                    elif isinstance(obj, dict):
                        queries.append(str(obj.get(query_field, "")))
                    else:
                        queries.append(str(obj))
        elif suffix == ".csv":
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    queries.append(row.get(query_field, ""))
        elif suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            queries.append(item)
                        elif isinstance(item, dict):
                            queries.append(str(item.get(query_field, "")))
        else:
            # Treat as plain text — one query per line
            with open(path, encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip()]

        if not queries:
            raise ValueError(f"No queries found in {path}")

        logger.info("Loaded %d queries from %s", len(queries), path)
        return queries
