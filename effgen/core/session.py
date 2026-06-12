"""
Persistent conversation sessions for effGen.

A Session stores a conversation history (and optional memory snapshot)
in ~/.effgen/sessions/<session_id>.json so it can be reloaded by any
agent process. Sessions are keyed by UUID by default, but callers may
provide their own session_id (e.g. "user-123").
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

DEFAULT_SESSION_DIR = os.path.expanduser("~/.effgen/sessions")


@dataclass
class Session:
    """
    A persistent conversation session.

    Attributes:
        session_id: Stable identifier (UUID by default).
        agent_name: Name of the owning agent (informational).
        messages: Conversation history as list of {role, content, timestamp}.
        memory: Optional memory snapshot (e.g. ShortTermMemory.to_dict()).
        metadata: Free-form metadata.
        created_at / updated_at: ISO timestamps.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ------------------------------------------------------------------ messages
    def add_message(self, role: str, content: str, **meta: Any) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": meta or {},
        })
        self.updated_at = datetime.now().isoformat()

    def add_user_message(self, content: str) -> None:
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.add_message("assistant", content)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        # Load forgivingly so sessions saved by older/newer effGen builds (which may
        # carry since-renamed fields) still reload instead of raising a cryptic
        # TypeError. See effgen.core._compat.load_from_dict.
        from ._compat import load_from_dict

        return load_from_dict(cls, data, label="Session")

    def save(self, sessions_dir: str = DEFAULT_SESSION_DIR) -> str:
        os.makedirs(sessions_dir, exist_ok=True)
        path = os.path.join(sessions_dir, f"{self.session_id}.json")
        self.updated_at = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(
        cls,
        session_id: str,
        sessions_dir: str = DEFAULT_SESSION_DIR,
    ) -> "Session":
        path = os.path.join(sessions_dir, f"{session_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session not found: {session_id}")
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_or_create(
        cls,
        session_id: str | None,
        agent_name: str = "",
        sessions_dir: str = DEFAULT_SESSION_DIR,
    ) -> "Session":
        if session_id:
            try:
                return cls.load(session_id, sessions_dir)
            except FileNotFoundError:
                return cls(session_id=session_id, agent_name=agent_name)
        return cls(agent_name=agent_name)


class SessionManager:
    """
    Filesystem-backed session store living in ~/.effgen/sessions/.

    Provides list / get / delete / cleanup operations used by the CLI.
    """

    def __init__(self, sessions_dir: str = DEFAULT_SESSION_DIR):
        self.sessions_dir = os.path.abspath(os.path.expanduser(sessions_dir))
        os.makedirs(self.sessions_dir, exist_ok=True)

    def list_sessions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for fname in os.listdir(self.sessions_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.sessions_dir, fname)) as f:
                    data = json.load(f)
                out.append({
                    "session_id": data.get("session_id", fname[:-5]),
                    "agent_name": data.get("agent_name", ""),
                    "messages": len(data.get("messages", [])),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                })
            except (OSError, json.JSONDecodeError):
                continue
        out.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
        return out

    def get(self, session_id: str) -> Session:
        return Session.load(session_id, self.sessions_dir)

    def delete(self, session_id: str) -> bool:
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def export(self, session_id: str, format: str = "json") -> str:
        session = self.get(session_id)
        if format == "json":
            return json.dumps(session.to_dict(), indent=2, default=str)
        if format == "text":
            lines = [f"Session: {session.session_id}", f"Agent: {session.agent_name}", ""]
            for m in session.messages:
                lines.append(f"[{m.get('role')}] {m.get('content')}")
            return "\n".join(lines)
        raise ValueError(f"Unsupported export format: {format}")

    def cleanup(self, older_than_days: int = 30) -> int:
        """Delete sessions whose updated_at is older than the cutoff."""
        cutoff = datetime.now() - timedelta(days=older_than_days)
        removed = 0
        for entry in self.list_sessions():
            updated = entry.get("updated_at")
            if not updated:
                continue
            try:
                ts = datetime.fromisoformat(updated)
            except ValueError:
                continue
            if ts < cutoff:
                if self.delete(entry["session_id"]):
                    removed += 1
        return removed
