"""ModelHost: exactly one backend instance per process, inference serialized by a lock."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable

from .backend import EmbeddingBackend
from .profiles import IndexProfile


class ModelBusy(RuntimeError):
    """The model is occupied by other work and the caller's wait budget expired."""


class ModelHost:
    def __init__(self, profile: IndexProfile, factory: Callable[[IndexProfile], EmbeddingBackend]):
        self.profile = profile
        self._factory = factory
        self._backend: EmbeddingBackend | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.load_count = 0
        self.load_seconds: float | None = None
        self.load_error: str | None = None
        self._closed = False

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def _ensure_loaded(self) -> EmbeddingBackend:
        if self._closed:
            raise RuntimeError("model host is shut down")
        if self._backend is None:
            with self._load_lock:
                if self._backend is None:
                    t0 = time.perf_counter()
                    try:
                        self._backend = self._factory(self.profile)
                    except Exception as e:  # surfaced through /api/model/status
                        self.load_error = f"{type(e).__name__}: {e}"
                        raise
                    self.load_error = None
                    self.load_count += 1
                    self.load_seconds = time.perf_counter() - t0
        return self._backend

    @contextmanager
    def use(self, timeout: float | None = None):
        """Exclusive access to the backend. Raises ModelBusy if not acquired within timeout."""
        acquired = self._infer_lock.acquire(timeout=-1 if timeout is None else timeout)
        if not acquired:
            raise ModelBusy("model is busy with other work; try again shortly")
        try:
            yield self._ensure_loaded()
        finally:
            self._infer_lock.release()

    def capabilities(self) -> dict[str, bool] | None:
        return self._backend.capabilities() if self._backend else None

    def memory(self) -> dict:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out = {"process_peak_rss_mb": round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)}
        if getattr(self._backend, "device", None) == "mps":
            import torch

            out["mps_current_allocated_mb"] = round(torch.mps.current_allocated_memory() / 2**20, 1)
            out["mps_driver_allocated_mb"] = round(torch.mps.driver_allocated_memory() / 2**20, 1)
        return out

    def status(self) -> dict:
        return {
            "memory": self.memory(),
            "loaded": self.loaded,
            "load_count": self.load_count,
            "load_seconds": self.load_seconds,
            "load_error": self.load_error,
            "busy": self._infer_lock.locked(),
            "device": getattr(self._backend, "device", None),
            "capabilities": self.capabilities(),
            "profile": self.profile.to_dict(),
            "profile_key": self.profile.key,
        }

    def shutdown(self) -> None:
        with self._infer_lock:
            self._closed = True
            if self._backend is not None:
                self._backend.close()
                self._backend = None
