"""Entry point: `uv run python -m mediaindex` (loopback only, single worker, one instance per data dir)."""

from __future__ import annotations

import sys

import uvicorn

from .app import create_app
from .config import Settings
from .instance import AlreadyRunning, acquire_instance_lock


def main() -> None:
    settings = Settings()
    settings.ensure_dirs()
    try:
        lock = acquire_instance_lock(settings.data_dir)
    except AlreadyRunning as e:
        print(f"MediaIndex: {e}", file=sys.stderr)
        raise SystemExit(1)
    try:
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
