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
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    )

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
