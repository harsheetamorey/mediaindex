"""Background job runner: one thread, bounded queue, progress and cancellation between work units."""

from __future__ import annotations

import os
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
    library_id: str | None = None
    status: str = "queued"  # queued | running | done | failed | cancelled
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    result: dict | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _work_started: float | None = field(default=None, repr=False)  # first unit of real work (skips excluded)
    _work_units: int = field(default=0, repr=False)

    def eta_seconds(self) -> float | None:
        """Rough remaining time from the rate of real work so far (unchanged files are not counted as work)."""
        if self.status != "running" or not self._work_started or self._work_units < 3 or not self.total:
            return None
        elapsed = time.time() - self._work_started
        return max(0.0, elapsed / self._work_units * max(0, self.total - self.done))

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        eta = self.eta_seconds()
        d["eta_seconds"] = round(eta) if eta is not None else None
        return d


class JobContext:
    def __init__(self, job: Job, on_update: Callable[[Job], None] | None, before_unit: Callable[[], None] | None = None):
        self.job = job
        self._on_update = on_update
        self._before_unit = before_unit

    def progress(self, done: int, total: int | None = None, message: str | None = None, work: bool = True) -> None:
        """Report progress. work=False marks a cheap step (e.g. an unchanged file) that the ETA should ignore."""
        if self._before_unit:
            self._before_unit()
        if work and done > self.job.done:
            if self.job._work_started is None:
                self.job._work_started = time.time()
            else:
                self.job._work_units += done - self.job.done
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


def _set_thread_background(background: bool) -> bool:
    """macOS: run the calling thread at background QoS (lower CPU and I/O priority). No-op elsewhere."""
    if not hasattr(os, "PRIO_DARWIN_THREAD"):
        return False
    try:
        os.setpriority(os.PRIO_DARWIN_THREAD, 0, os.PRIO_DARWIN_BG if background else 0)
        return True
    except OSError:
        return False


class JobRunner:
    def __init__(self, maxsize: int = 8, on_update: Callable[[Job], None] | None = None, low_priority: bool = False):
        self._q: queue.Queue[tuple[Job, JobFn] | None] = queue.Queue(maxsize=maxsize)
        self._jobs: dict[str, Job] = {}
        self._on_update = on_update
        self.low_priority = low_priority  # read by the worker thread before each unit of work
        self._applied_priority: bool | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="mediaindex-jobs", daemon=True)
        self._thread.start()

    def submit(self, kind: str, fn: JobFn, job_id: str | None = None, library_id: str | None = None) -> Job:
        job = Job(id=job_id or uuid.uuid4().hex, kind=kind, library_id=library_id)
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
            self._sync_priority()
            self._notify(job)
            try:
                job.result = fn(JobContext(job, self._on_update, before_unit=self._sync_priority)) or {}
                job.status = "cancelled" if job._cancel.is_set() else "done"
            except JobCancelled:
                job.status = "cancelled"
            except Exception as e:
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
            job.finished = time.time()
            self._notify(job)

    def _sync_priority(self) -> None:
        """Called on the worker thread: apply a changed low-priority setting between work units."""
        if self._applied_priority != self.low_priority:
            _set_thread_background(self.low_priority)
            self._applied_priority = self.low_priority

    def shutdown(self, timeout: float = 10.0) -> None:
        for job in self.list():
            job._cancel.set()
        try:
            self._q.put(None, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
