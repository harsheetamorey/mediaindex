"""Runtime settings. All state lives under a single local data directory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    env = os.environ.get("MEDIAINDEX_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "data").resolve()


@dataclass
class Settings:
    data_dir: Path = field(default_factory=_default_data_dir)
    host: str = "127.0.0.1"
    port: int = int(os.environ.get("MEDIAINDEX_PORT", "8765"))
    device: str = os.environ.get("MEDIAINDEX_DEVICE", "auto")  # auto | mps | cpu
    precision: str = os.environ.get("MEDIAINDEX_PRECISION", "auto")  # auto | bfloat16 | float32
    job_queue_size: int = 8
    query_wait_seconds: float = float(os.environ.get("MEDIAINDEX_QUERY_WAIT", "30"))
    max_request_bytes: int = 60 * 1024 * 1024  # largest accepted upload (50 MB audio) + multipart overhead
    allowed_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.allowed_origins:  # the UI served by this server, plus the Vite dev server
            self.allowed_origins = tuple(f"http://{h}:{p}" for p in (self.port, 5173) for h in ("127.0.0.1", "localhost"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "mediaindex.sqlite3"

    @property
    def vectors_dir(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "tmp_uploads"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.vectors_dir, self.thumbs_dir, self.uploads_dir):
            p.mkdir(parents=True, exist_ok=True)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def resolved_precision(self) -> str:
        if self.precision != "auto":
            return self.precision
        return "bfloat16" if self.resolved_device() == "mps" else "float32"
