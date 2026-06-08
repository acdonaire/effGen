"""
Background task runner for long-running effGen agent executions.

Uses threading.Thread (not multiprocessing) so tasks share the in-memory
agent / model and avoid serialization overhead. A simple priority queue
schedules pending tasks; running tasks expose status, progress, result,
and basic cancel/pause/resume controls.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(order=True)
class _PrioritizedItem:
    priority: int
    seq: int
    task_id: str = field(compare=False)


@dataclass
class BackgroundTask:
    task_id: str
    task_text: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5
    progress: float = 0.0
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BackgroundTaskRunner:
    """
    Execute agent tasks in background worker threads.

    Usage:
        runner = BackgroundTaskRunner(agent, max_workers=2)
        task_id = runner.submit("Long task...", priority=1)
        status = runner.get_status(task_id)
        result = runner.get_result(task_id)  # blocks until done if wait=True
        runner.shutdown()
    """

    def __init__(self, agent: Any, max_workers: int = 1):
        self.agent = agent
        self.max_workers = max_workers

        self._tasks: dict[str, BackgroundTask] = {}
        self._queue: list[_PrioritizedItem] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pause_events: dict[str, threading.Event] = {}
        self._cancel_flags: dict[str, threading.Event] = {}
        self._progress_callbacks: dict[str, Callable[[BackgroundTask], None]] = {}
        self._seq = 0

        self._shutdown = False
        self._workers: list[threading.Thread] = []
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"effgen-bg-worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    # ------------------------------------------------------------------ submit
    def submit(
        self,
        task_text: str,
        priority: int = 5,
        progress_callback: Callable[[BackgroundTask], None] | None = None,
        **run_kwargs: Any,
    ) -> str:
        """Enqueue a task and return its id. Lower priority value = higher priority."""
        task_id = str(uuid.uuid4())
        task = BackgroundTask(
            task_id=task_id,
            task_text=task_text,
            priority=priority,
            metadata={"run_kwargs": run_kwargs},
        )
        with self._cv:
            self._tasks[task_id] = task
            self._pause_events[task_id] = threading.Event()
            self._pause_events[task_id].set()  # not paused
            self._cancel_flags[task_id] = threading.Event()
            if progress_callback:
                self._progress_callbacks[task_id] = progress_callback
            self._seq += 1
            heapq.heappush(self._queue, _PrioritizedItem(priority, self._seq, task_id))
            self._cv.notify()
        return task_id

    # ------------------------------------------------------------------ status / result
    def get_status(self, task_id: str) -> TaskStatus:
        with self._lock:
            return self._tasks[task_id].status

    def get_task(self, task_id: str) -> BackgroundTask:
        with self._lock:
            return self._tasks[task_id]

    def get_result(self, task_id: str, wait: bool = False, timeout: float | None = None) -> Any:
        if wait:
            deadline = None if timeout is None else time.time() + timeout
            while True:
                with self._lock:
                    task = self._tasks[task_id]
                    if task.status in (
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                    ):
                        return task.result
                if deadline is not None and time.time() > deadline:
                    raise TimeoutError(f"Task {task_id} did not finish in time")
                time.sleep(0.05)
        with self._lock:
            return self._tasks[task_id].result

    def list_tasks(self) -> list[BackgroundTask]:
        with self._lock:
            return list(self._tasks.values())

    # ------------------------------------------------------------------ control
    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.finished_at = time.time()
                return True
            if task.status in (TaskStatus.RUNNING, TaskStatus.PAUSED):
                self._cancel_flags[task_id].set()
                # Unblock any pause wait so the worker can observe the cancel
                self._pause_events[task_id].set()
                return True
            return False

    def pause(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.RUNNING:
                return False
            self._pause_events[task_id].clear()
            task.status = TaskStatus.PAUSED
            return True

    def resume(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.PAUSED:
                return False
            self._pause_events[task_id].set()
            task.status = TaskStatus.RUNNING
            return True

    def shutdown(self, wait: bool = False) -> None:
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        if wait:
            for t in self._workers:
                t.join(timeout=5)

    # ------------------------------------------------------------------ worker
    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._shutdown:
                    self._cv.wait()
                if self._shutdown and not self._queue:
                    return
                item = heapq.heappop(self._queue)
                task = self._tasks.get(item.task_id)
                if task is None or task.status == TaskStatus.CANCELLED:
                    continue
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()

            self._execute(task)

    def _execute(self, task: BackgroundTask) -> None:
        try:
            # Honor pause / cancel before starting heavy work
            self._pause_events[task.task_id].wait()
            if self._cancel_flags[task.task_id].is_set():
                with self._lock:
                    task.status = TaskStatus.CANCELLED
                    task.finished_at = time.time()
                return

            run_kwargs = task.metadata.get("run_kwargs", {}) or {}
            result = self.agent.run(task.task_text, **run_kwargs)

            with self._lock:
                if self._cancel_flags[task.task_id].is_set():
                    task.status = TaskStatus.CANCELLED
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.progress = 1.0
                task.finished_at = time.time()
        except Exception as e:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.finished_at = time.time()
        finally:
            cb = self._progress_callbacks.get(task.task_id)
            if cb:
                try:
                    cb(task)
                except Exception:
                    logger.debug(
                        "Progress callback for task %s raised", task.task_id, exc_info=True
                    )
