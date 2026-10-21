"""Runtime settings. All state lives under a single local data directory."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def setup_step(command: str) -> str:
    """How to download optional models: a menu item in the Mac app, a terminal command otherwise."""
    if getattr(sys, "frozen", False):
        return "In the menu bar, choose Models → Download photo editing models…"
    return f"Run this once in the MediaIndex folder: {command}"


def auto_device(torch) -> str:
    """Apple GPU (MPS) on a Mac, an NVIDIA GPU (CUDA) on Windows or Linux when torch was installed with CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def use_all_cores(torch) -> None:
    """On the CPU, run on every logical core. torch's default is the physical cores; on a 4-thread Windows runner every
    thread made indexing 1.6x faster (evaluation/bench_cpu.py). MEDIAINDEX_CPU_THREADS overrides."""
    torch.set_num_threads(int(os.environ.get("MEDIAINDEX_CPU_THREADS") or os.cpu_count() or 1))


def free_device_memory(torch, device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


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
    device: str = os.environ.get("MEDIAINDEX_DEVICE", "auto")  # auto | mps | cuda | cpu
    precision: str = os.environ.get("MEDIAINDEX_PRECISION", "auto")  # auto | bfloat16 | float32
    image_tokens: int = int(os.environ.get("MEDIAINDEX_IMAGE_TOKENS", "0"))  # 0 = auto (resolved_image_tokens)
    job_queue_size: int = 8
    query_wait_seconds: float = float(os.environ.get("MEDIAINDEX_QUERY_WAIT", "30"))
    watch_interval: float = float(os.environ.get("MEDIAINDEX_WATCH_INTERVAL", "20"))  # seconds; 0 disables watching
    ollama_url: str = os.environ.get("MEDIAINDEX_OLLAMA_URL", "http://127.0.0.1:11434")  # loopback only
    chat_model: str = os.environ.get("MEDIAINDEX_CHAT_MODEL", "gemma4:e2b-it-qat")  # "" turns the chat model off
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

            return auto_device(torch)
        except ImportError:
            return "cpu"

    def resolved_precision(self) -> str:
        if self.precision != "auto":
            return self.precision
        return "bfloat16" if self.resolved_device() == "mps" else "float32"

    def resolved_image_tokens(self) -> int:
        """Vision soft tokens per photo: the model's default 280 on a GPU; 140 on a CPU (a PC without an Apple or
        NVIDIA GPU), where it indexes photos 2.4x faster (evaluation/bench_cpu.py, docs/progress.md)."""
        if self.image_tokens:
            return self.image_tokens
        return 140 if self.resolved_device() == "cpu" else 280

    def index_profile(self):
        from .model.profiles import IndexProfile

        return IndexProfile(precision=self.resolved_precision(), image_max_soft_tokens=self.resolved_image_tokens())
