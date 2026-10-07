"""Background job runner: one thread, bounded queue, progress and cancellation between work units."""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable


class QueueFull(RuntimeError):
    pass


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    result: dict | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class JobContext:
    def __init__(self, job: Job, on_update: Callable[[Job], None] | None):
        self.job = job
        self._on_update = on_update

    def progress(self, done: int, total: int | None = None, message: str | None = None) -> None:
        self.job.done = done
        if total is not None:
            self.job.total = total
        if message is not None:
            self.job.message = message
        if self._on_update:
            self._on_update(self.job)

    @property
    def cancelled(self) -> bool:
        return self.job._cancel.is_set()

    def check_cancelled(self) -> None:
        if self.job._cancel.is_set():
            raise JobCancelled()


JobFn = Callable[[JobContext], dict | None]


class JobRunner:
    def __init__(self, maxsize: int = 8, on_update: Callable[[Job], None] | None = None):
        self._q: queue.Queue[tuple[Job, JobFn] | None] = queue.Queue(maxsize=maxsize)
        self._jobs: dict[str, Job] = {}
        self._on_update = on_update
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="mediaindex-jobs", daemon=True)
        self._thread.start()

    def submit(self, kind: str, fn: JobFn, job_id: str | None = None) -> Job:
        job = Job(id=job_id or uuid.uuid4().hex, kind=kind)
        try:
            self._q.put_nowait((job, fn))
        except queue.Full:
            raise QueueFull("job queue is full; wait for running jobs to finish") from None
        with self._lock:
            self._jobs[job.id] = job
        self._notify(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job and job.status in ("queued", "running"):
            job._cancel.set()
            if job.status == "queued":
                job.status = "cancelled"
                job.finished = time.time()
                self._notify(job)
        return job

    def _notify(self, job: Job) -> None:
        if self._on_update:
            try:
                self._on_update(job)
            except Exception:
                traceback.print_exc()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            job, fn = item
            if job._cancel.is_set():
                continue
            job.status = "running"
            self._notify(job)
            try:
                job.result = fn(JobContext(job, self._on_update)) or {}
                job.status = "cancelled" if job._cancel.is_set() else "done"
            except JobCancelled:
                job.status = "cancelled"
            except Exception as e:
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
            job.finished = time.time()
            self._notify(job)

    def shutdown(self, timeout: float = 10.0) -> None:
        for job in self.list():
            job._cancel.set()
        try:
            self._q.put(None, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
