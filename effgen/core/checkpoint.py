"""
Agent checkpointing for effGen framework.

Provides CheckpointManager which can serialize agent execution state
(scratchpad, iterations, tool history, partial results, memory) to disk
as human-readable JSON. Supports filesystem (default) and SQLite backends.

Checkpoints are JSON-serializable only (no pickle) for security.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _read_checkpoint_json(path: str) -> dict:
    """Read+parse a checkpoint JSON file, raising a clear error if corrupt."""
    with open(path) as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        from ..errors import CorruptStateError
        raise CorruptStateError("checkpoint", path, str(e)) from e


@dataclass
class Checkpoint:
    """A single checkpoint snapshot."""

    checkpoint_id: str
    agent_name: str
    task: str
    iteration: int
    model: str = ""
    scratchpad: str = ""
    partial_output: str | None = None
    tool_calls: int = 0
    tokens_used: int = 0
    memory: dict[str, Any] = field(default_factory=dict)
    tool_states: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return the checkpoint as a JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Rebuild a checkpoint from ``to_dict()`` output (unknown fields ignored)."""
        # Load forgivingly so checkpoints written by a different effGen build (which
        # may carry since-renamed fields) still resume instead of raising a cryptic
        # TypeError. See effgen.core._compat.load_from_dict.
        from ._compat import load_from_dict

        return load_from_dict(cls, data, label="Checkpoint")


class CheckpointManager:
    """
    Save and restore agent state to filesystem (JSON) or SQLite.

    Usage:
        mgr = CheckpointManager("./checkpoints")
        cp_id = mgr.save(checkpoint)
        cp = mgr.load(cp_id)         # by id
        cp = mgr.load_latest()       # most recent
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        backend: str = "filesystem",
    ):
        self.backend = backend
        self.checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if backend == "sqlite":
            self.db_path = os.path.join(self.checkpoint_dir, "checkpoints.db")
            self._init_sqlite()
        elif backend != "filesystem":
            raise ValueError(f"Unsupported backend: {backend}")

    # ------------------------------------------------------------------ sqlite
    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    agent_name TEXT,
                    task TEXT,
                    iteration INTEGER,
                    created_at TEXT,
                    data TEXT
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------ save
    def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint and return its id."""
        if not checkpoint.checkpoint_id:
            checkpoint.checkpoint_id = self._new_id(checkpoint.agent_name)

        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        checkpoint.checkpoint_id,
                        checkpoint.agent_name,
                        checkpoint.task,
                        checkpoint.iteration,
                        checkpoint.created_at,
                        json.dumps(checkpoint.to_dict()),
                    ),
                )
                conn.commit()
        else:
            path = os.path.join(self.checkpoint_dir, f"{checkpoint.checkpoint_id}.json")
            with open(path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)
            # Maintain a "latest.json" pointer for convenience
            latest = os.path.join(self.checkpoint_dir, "latest.json")
            with open(latest, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)

        return checkpoint.checkpoint_id

    # ------------------------------------------------------------------ load
    def load(self, checkpoint_id: str) -> Checkpoint:
        """Load a checkpoint by id, or by file path."""
        # Allow passing a path directly
        if os.path.sep in checkpoint_id or checkpoint_id.endswith(".json"):
            if not os.path.exists(checkpoint_id):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
            return Checkpoint.from_dict(_read_checkpoint_json(checkpoint_id))

        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT data FROM checkpoints WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                ).fetchone()
                if row is None:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
                try:
                    data = json.loads(row[0])
                except (json.JSONDecodeError, ValueError) as e:
                    from ..errors import CorruptStateError
                    raise CorruptStateError("checkpoint", self.db_path, str(e)) from e
                return Checkpoint.from_dict(data)

        path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return Checkpoint.from_dict(_read_checkpoint_json(path))

    def load_latest(self) -> Checkpoint:
        """Return the most recently created checkpoint."""
        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT data FROM checkpoints ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                if row is None:
                    raise FileNotFoundError("No checkpoints found")
                try:
                    data = json.loads(row[0])
                except (json.JSONDecodeError, ValueError) as e:
                    from ..errors import CorruptStateError
                    raise CorruptStateError("checkpoint", self.db_path, str(e)) from e
                return Checkpoint.from_dict(data)

        latest = os.path.join(self.checkpoint_dir, "latest.json")
        if os.path.exists(latest):
            return Checkpoint.from_dict(_read_checkpoint_json(latest))

        files = sorted(
            (f for f in os.listdir(self.checkpoint_dir) if f.endswith(".json")),
            key=lambda f: os.path.getmtime(os.path.join(self.checkpoint_dir, f)),
            reverse=True,
        )
        if not files:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        return self.load(files[0].replace(".json", ""))

    # ------------------------------------------------------------------ list/delete
    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Return a list of checkpoint summaries."""
        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT checkpoint_id, agent_name, task, iteration, created_at "
                    "FROM checkpoints ORDER BY created_at DESC"
                ).fetchall()
            return [
                {
                    "checkpoint_id": r[0],
                    "agent_name": r[1],
                    "task": r[2],
                    "iteration": r[3],
                    "created_at": r[4],
                }
                for r in rows
            ]

        results: list[dict[str, Any]] = []
        for fname in os.listdir(self.checkpoint_dir):
            if not fname.endswith(".json") or fname == "latest.json":
                continue
            try:
                with open(os.path.join(self.checkpoint_dir, fname)) as f:
                    data = json.load(f)
                results.append({
                    "checkpoint_id": data.get("checkpoint_id", fname[:-5]),
                    "agent_name": data.get("agent_name"),
                    "task": data.get("task"),
                    "iteration": data.get("iteration"),
                    "created_at": data.get("created_at"),
                })
            except (OSError, json.JSONDecodeError) as e:
                logger.debug("Skipping unreadable checkpoint file %s: %s", fname, e)
                continue
        results.sort(key=lambda d: d.get("created_at") or "", reverse=True)
        return results

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint by id. Returns True if removed."""
        if self.backend == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                )
                conn.commit()
                return cur.rowcount > 0
        path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _new_id(agent_name: str) -> str:
        ts = int(time.time() * 1000)
        return f"{agent_name}-{ts}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def snapshot_agent(
        agent: Any,
        task: str,
        iteration: int,
        scratchpad: str = "",
        partial_output: str | None = None,
        tool_calls: int = 0,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """
        Build a Checkpoint by snapshotting an Agent's serializable state.

        Tool *instances* are stored only by name + class (config, not
        instances), keeping the checkpoint JSON-safe.
        """
        memory_dict: dict[str, Any] = {}
        try:
            stm = getattr(agent, "short_term_memory", None)
            if stm is not None and hasattr(stm, "to_dict"):
                memory_dict["short_term"] = stm.to_dict()
        except Exception:
            logger.debug("Failed to snapshot short-term memory for checkpoint", exc_info=True)

        # Record the model id so `resume` can reuse it (and warn on a mismatch)
        # rather than silently resuming on a different model.
        model_id = getattr(agent, "model_name", None) or ""

        tool_states: dict[str, Any] = {}
        for tname, tool in getattr(agent, "tools", {}).items():
            tool_states[tname] = {
                "name": tname,
                "class": type(tool).__name__,
                "module": type(tool).__module__,
            }

        return Checkpoint(
            checkpoint_id="",
            agent_name=getattr(agent, "name", "agent"),
            task=task,
            iteration=iteration,
            model=model_id,
            scratchpad=scratchpad,
            partial_output=partial_output,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            memory=memory_dict,
            tool_states=tool_states,
            metadata=metadata or {},
        )

    @staticmethod
    def restore_to_agent(agent: Any, checkpoint: Checkpoint) -> None:
        """
        Restore checkpoint state into an existing agent (memory only).

        Tool instances are *not* recreated — the agent must already be
        constructed with the same tools. Scratchpad / iteration are
        consumed by Agent.resume() to seed the next run.
        """
        stm_data = checkpoint.memory.get("short_term") if checkpoint.memory else None
        if stm_data:
            try:
                from ..memory.short_term import ShortTermMemory
                agent.short_term_memory = ShortTermMemory.from_dict(stm_data)
            except Exception:
                logger.debug("Failed to restore short-term memory from checkpoint", exc_info=True)
