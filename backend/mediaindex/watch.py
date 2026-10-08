"""Watched folders: poll library roots cheaply (stat only) and queue a re-scan when their contents change.

Polling instead of OS file events keeps this portable and dependency-free. A change must be seen on two
consecutive polls (files stop changing) before a re-scan is queued, so a large copy in progress triggers one
import, not dozens. Originals are only stat'ed here; the import itself is the normal read-only import job.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from .media.audio import AUDIO_EXTENSIONS
from .media.images import IMAGE_EXTENSIONS
from .media.video import VIDEO_EXTENSIONS
from .paths import walk_files
from .store import Store

MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def folder_signature(root: Path) -> tuple[int, int, int] | None:
    """(file count, sum of sizes, sum of mtimes) over importable files; None if the folder is unavailable."""
    if not root.is_dir():
        return None
    n = size = mtime = 0
    for p, _ in walk_files(root, MEDIA_EXTENSIONS):
        try:
            st = p.stat()
        except OSError:
            continue
        n += 1
        size += st.st_size
        mtime += st.st_mtime_ns
    return n, size, mtime


class FolderWatcher:
    def __init__(self, store: Store, submit_import: Callable[[str], object], interval: float = 20.0):
        self.store = store
        self.submit_import = submit_import
        self.interval = interval
        self._baseline: dict[str, tuple] = {}  # library id -> signature the index reflects
        self._pending: dict[str, tuple] = {}  # library id -> changed signature seen once (waiting to settle)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="mediaindex-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - a bad poll must never kill the watcher
                import traceback

                traceback.print_exc()

    def poll(self) -> list[str]:
        """One pass over watched libraries. Returns the ids for which a re-scan was queued."""
        queued = []
        watched = {lib["id"]: lib for lib in self.store.list_libraries() if lib.get("watch")}
        for lid in list(self._baseline):
            if lid not in watched:  # unwatched or removed: forget it, so re-enabling starts fresh
                self._baseline.pop(lid, None)
                self._pending.pop(lid, None)
        for lid, lib in watched.items():
            sig = folder_signature(Path(lib["root_path"]))
            if sig is None:
                continue
            if lid not in self._baseline:
                self._baseline[lid] = sig
                continue
            if sig == self._baseline[lid]:
                self._pending.pop(lid, None)
                continue
            if self._pending.get(lid) != sig:  # changed, or still changing: wait for it to settle
                self._pending[lid] = sig
                continue
            try:
                self.submit_import(lid)
            except Exception:  # noqa: BLE001 - e.g. queue full: retry on the next poll
                continue
            self._baseline[lid] = sig
            self._pending.pop(lid, None)
            queued.append(lid)
        return queued
