"""Entry point: `uv run python -m mediaindex` (loopback only, single worker, one instance per data dir).

`uv run python -m mediaindex setup` downloads the models once (setup.py)."""

from __future__ import annotations

import os
import sys

# Normal use is offline: model files come from the local cache only, and library telemetry is off.
# (First-time setup downloads the weights explicitly: `python -m mediaindex setup`.)
SETUP = sys.argv[1:2] == ["setup"]
if os.environ.get("MEDIAINDEX_ALLOW_DOWNLOAD") != "1" and not SETUP:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import uvicorn  # noqa: E402

from .app import create_app  # noqa: E402
from .config import Settings  # noqa: E402
from .instance import AlreadyRunning, acquire_instance_lock  # noqa: E402


def main() -> None:
    if SETUP:
        from .setup import main as setup_main

        raise SystemExit(setup_main(sys.argv[2:]))
    settings = Settings()
    settings.ensure_dirs()
    try:
        lock = acquire_instance_lock(settings.data_dir)
    except AlreadyRunning as e:
        print(f"MediaIndex: {e}", file=sys.stderr)
        raise SystemExit(1)
    # Multipart parts are spooled by Starlette into the process temp dir; keep them in a private
    # app directory (emptied at every start) instead of the system temp folder.
    import tempfile

    from .uploads import cleanup_stale_uploads

    spool = settings.data_dir / "tmp_spool"
    spool.mkdir(parents=True, exist_ok=True)
    cleanup_stale_uploads(spool, max_age=0)
    tempfile.tempdir = str(spool)
    try:
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
