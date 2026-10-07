"""One MediaIndex process per data directory (prevents two model copies and conflicting job recovery)."""

from __future__ import annotations

import os
from pathlib import Path


class AlreadyRunning(RuntimeError):
    pass


def acquire_instance_lock(data_dir: Path):
    """Hold an exclusive advisory lock on <data_dir>/.mediaindex.lock for the life of the process.

    The OS releases it automatically if the process dies (including SIGKILL).
    """
    import fcntl

    path = Path(data_dir) / ".mediaindex.lock"
    f = open(path, "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise AlreadyRunning(f"another MediaIndex process is already using {data_dir}; stop it first") from None
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f
