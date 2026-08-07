"""
SQLite-backed persistence for CostTracker.

Stores every cost event so that spend can be queried across restarts and
processes.  The schema is append-only; no row is ever updated or deleted
during normal operation (cleanup removes old rows).

Schema
------
    cost_events(
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        provider          TEXT    NOT NULL,
        model             TEXT    NOT NULL,
        prompt_tokens     INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd          REAL    NOT NULL DEFAULT 0.0,
        timestamp         REAL    NOT NULL   -- UNIX epoch (time.time())
    )

Concurrency
-----------
WAL journal mode + BEGIN IMMEDIATE give multi-reader / single-writer
semantics safe for concurrent processes without external locking, so a
file-backed store hands each thread its own connection. A ``:memory:``
database exists only inside the connection that opened it, so that store keeps
one shared connection and serializes statements on it — otherwise every thread
but the first would be writing to a database of its own.

Usage::

    from effgen.models._cost_store import SQLiteCostStore
    store = SQLiteCostStore()            # ~/.effgen/costs.sqlite
    store = SQLiteCostStore(":memory:")  # in-process tests

    store.insert(provider="cerebras", model="llama3.1-8b",
                 prompt_tokens=50, completion_tokens=20,
                 cost_usd=0.0, timestamp=time.time())

    rows = store.query_today()
    rows = store.query_since(since_timestamp)
    rows = store.query_all()
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

_DEFAULT_DB_PATH = Path.home() / ".effgen" / "costs.sqlite"


def _default_db_path() -> Path:
    """Where cost events are stored when no path is given.

    ``EFFGEN_HOME`` relocates the whole effGen state directory, so the cost
    database follows it to ``$EFFGEN_HOME/costs.sqlite``; otherwise it lives at
    ``~/.effgen/costs.sqlite``.
    """
    home = os.environ.get("EFFGEN_HOME")
    if home:
        return Path(os.path.expanduser(home)).absolute() / "costs.sqlite"
    return _DEFAULT_DB_PATH

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cost_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    provider          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL    NOT NULL DEFAULT 0.0,
    timestamp         REAL    NOT NULL
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_cost_events_lookup
    ON cost_events (provider, model, timestamp);
"""

_INSERT = """
INSERT INTO cost_events (provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp)
VALUES (?, ?, ?, ?, ?, ?);
"""

_QUERY_SINCE = """
SELECT provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp
FROM cost_events
WHERE timestamp >= ?
ORDER BY timestamp ASC;
"""

_QUERY_ALL = """
SELECT provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp
FROM cost_events
ORDER BY timestamp ASC;
"""

_DELETE_OLD = """
DELETE FROM cost_events WHERE timestamp < ?;
"""


@dataclass
class CostEvent:
    """One recorded cost event row."""
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: float


class SQLiteCostStore:
    """Thread-safe, cross-process SQLite store for cost events.

    Args:
        db_path: Path to the SQLite file.  Use ``":memory:"`` for tests.
                 Defaults to ``$EFFGEN_HOME/costs.sqlite`` when ``EFFGEN_HOME``
                 is set, else ``~/.effgen/costs.sqlite``. ``EFFGEN_COST_DB``
                 overrides both.
    """

    def __init__(self, db_path: str | os.PathLike | None = None) -> None:
        if db_path is None:
            # Allow tests / sandboxes to redirect persistence away from the
            # user's real ~/.effgen/costs.sqlite via EFFGEN_COST_DB.
            env_path = os.environ.get("EFFGEN_COST_DB")
            db_path = env_path if env_path else _default_db_path()
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # An in-memory database belongs to the connection that opened it, so a
        # per-thread connection would give every thread its own empty database
        # and every call recorded off the creating thread would be lost. One
        # connection, shared, with a lock around each statement.
        self._shared_lock = threading.Lock() if self._path == ":memory:" else None
        self._shared_conn: sqlite3.Connection | None = None
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        if self._path == ":memory:":
            # An in-memory database is created empty along with its connection,
            # so the schema belongs to opening one rather than to constructing
            # the store: a store reopened after close() would otherwise have
            # nothing to write to and drop every call recorded on it.
            with conn:
                conn.execute(_CREATE_TABLE)
                conn.execute(_CREATE_INDEX)
        return conn

    def _conn(self) -> sqlite3.Connection:
        if self._shared_lock is not None:
            with self._shared_lock:
                if self._shared_conn is None:
                    self._shared_conn = self._open()
                return self._shared_conn
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._open()
        return cast(sqlite3.Connection, self._local.conn)

    @contextmanager
    def _exclusive(self) -> Iterator[sqlite3.Connection]:
        """Yield the connection, serialized when it is shared across threads.

        A file-backed store gives each thread its own connection and SQLite's
        own locking keeps writers apart, so there is nothing to serialize. The
        single in-memory connection is shared, and two threads issuing
        statements on one connection interleave, so that case takes a lock.
        """
        conn = self._conn()
        if self._shared_lock is None:
            yield conn
            return
        with self._shared_lock:
            yield conn

    def _init_schema(self) -> None:
        with self._exclusive() as conn, conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        timestamp: float | None = None,
    ) -> None:
        """Insert one cost event atomically.

        Args:
            provider: The provider that served the call.
            model: The model id the call used.
            prompt_tokens: Input tokens the call consumed.
            completion_tokens: Output tokens the call produced.
            cost_usd: What the call cost in US dollars.
            timestamp: Unix time of the call, defaulting to now.
        """
        ts = timestamp if timestamp is not None else time.time()
        with self._exclusive() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute(
                    _INSERT,
                    (provider, model, prompt_tokens, completion_tokens, cost_usd, ts),
                )
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def query_since(self, since: float) -> list[CostEvent]:
        """Return all events with timestamp >= *since*."""
        with self._exclusive() as conn:
            rows = conn.execute(_QUERY_SINCE, (since,)).fetchall()
        return [CostEvent(*row) for row in rows]

    def query_today(self) -> list[CostEvent]:
        """Return events from the last 24 hours (rolling day)."""
        since = time.time() - 86400.0
        return self.query_since(since)

    def query_week(self) -> list[CostEvent]:
        """Return events from the last 7 days."""
        since = time.time() - 7 * 86400.0
        return self.query_since(since)

    def query_month(self) -> list[CostEvent]:
        """Return events from the last 30 days."""
        since = time.time() - 30 * 86400.0
        return self.query_since(since)

    def query_all(self) -> list[CostEvent]:
        """Return all stored events (lifetime)."""
        with self._exclusive() as conn:
            rows = conn.execute(_QUERY_ALL).fetchall()
        return [CostEvent(*row) for row in rows]

    def cleanup(self, max_age_seconds: float) -> int:
        """Delete events older than *max_age_seconds*.  Returns rows deleted."""
        cutoff = time.time() - max_age_seconds
        with self._exclusive() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cursor = conn.execute(_DELETE_OLD, (cutoff,))
                count = cursor.rowcount
                conn.execute("COMMIT;")
                return count
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def close(self) -> None:
        """Close this thread's connection, or the shared in-memory one."""
        if self._shared_lock is not None:
            with self._shared_lock:
                if self._shared_conn is not None:
                    self._shared_conn.close()
                    self._shared_conn = None
            return
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
