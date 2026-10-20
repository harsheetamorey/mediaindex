"""One-step setup: `uv run python -m mediaindex setup` (add `--editing` for the photo editing models).

Running it is the request to download, so no environment variables are needed, on macOS, Windows or Linux. It checks
FFmpeg, builds the UI if it isn't built, downloads the pinned models that aren't in the local cache yet, and loads the
search model once to check it works. Running it again only fills in what is missing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FFMPEG_HINT = {"darwin": "brew install ffmpeg", "win32": "winget install Gyan.FFmpeg (then open a new terminal)"}


def _say(step: str, text: str) -> None:
    print(f"[{step}] {text}", flush=True)


def check_ffmpeg() -> bool:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        hint = FFMPEG_HINT.get(sys.platform, "install FFmpeg with your package manager")
        _say("ffmpeg", f"not found ({', '.join(missing)}). Photos work without it; for sounds and videos: {hint}")
        return False
    _say("ffmpeg", "found")
    return True


def build_ui(force: bool = False) -> bool:
    from .app import FRONTEND_DIST

    if (FRONTEND_DIST / "index.html").exists() and not force:
        _say("ui", "already built")
        return True
    frontend = ROOT / "frontend"
    npm = shutil.which("npm")
    if npm is None or not (frontend / "package.json").exists():
        _say("ui", "not built, and npm wasn't found: install Node 20+, then run this again")
        return False
    _say("ui", "building (npm ci, npm run build)…")
    for args in (["ci"], ["run", "build"]):
        if subprocess.run([npm, *args], cwd=frontend).returncode != 0:
            _say("ui", f"`npm {' '.join(args)}` failed (see above)")
            return False
    _say("ui", "built")
    return True


def download_models(editing: bool) -> bool:
    import huggingface_hub.constants as hub
    from huggingface_hub import snapshot_download

    from . import creative, detect, edit
    from .desktop import model_cached
    from .model.profiles import MODEL_ID, MODEL_REVISION

    hub.HF_HUB_OFFLINE = False  # this command is the user's request to download
    ok = True
    if model_cached():
        _say("models", "search model and object detector already downloaded")
    else:
        _say("models", "downloading the search model (EmbeddingGemma 2, ~1.5 GB) and the object detector (~170 MB)…")
        snapshot_download(MODEL_ID, revision=MODEL_REVISION)
        detect.download()
        _say("models", "downloaded")
    if editing:
        if edit.models_cached():
            _say("editing", "SAM and LaMa already downloaded")
        else:
            _say("editing", "downloading SAM and LaMa (~550 MB) for removing objects…")
            edit.download()
        try:
            import diffusers  # noqa: F401
            import peft  # noqa: F401
        except ImportError:
            _say("editing", "replacing objects and making images also need: uv sync --extra create (then run this "
                            "again with --editing)")
            ok = False
        else:
            if creative.available()[0]:
                _say("editing", "Stable Diffusion already downloaded")
            else:
                _say("editing", "downloading Stable Diffusion inpainting and LCM-LoRA (~2.1 GB)…")
                creative.download()
        _say("editing", "done" if ok else "removing objects is ready")
    return ok


def check_search_model() -> bool:
    """Load the search model from the local cache and embed one sentence (the same path the app uses)."""
    from .config import Settings
    from .model.backend import make_backend
    from .model.profiles import IndexProfile

    settings = Settings()
    _say("check", f"loading the search model on {settings.resolved_device()}…")
    os.environ["HF_HUB_OFFLINE"] = "1"
    backend = make_backend(IndexProfile(precision=settings.resolved_precision()), device=settings.resolved_device())
    try:
        v = backend.embed_query_texts(["a dog on a beach"])[0]
    finally:
        backend.close()
    _say("check", f"works ({len(v)}-dimensional vectors)")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mediaindex setup", description=__doc__.split("\n\n")[0])
    ap.add_argument("--editing", action="store_true", help="also download the photo editing models (~2.7 GB)")
    ap.add_argument("--no-check", action="store_true", help="skip loading the search model at the end")
    ap.add_argument("--rebuild-ui", action="store_true", help="build the UI even if it is already built")
    a = ap.parse_args(argv)
    check_ffmpeg()
    ui = build_ui(a.rebuild_ui)
    try:
        models = download_models(a.editing)
    except Exception as e:  # noqa: BLE001 (network errors come in many types)
        _say("models", f"download failed: {e}. Check your internet connection and run this again; it resumes.")
        return 1
    if not a.no_check:
        check_search_model()
    print("\nSetup finished." if ui and models else "\nSetup finished with notes above.")
    print("Start MediaIndex with: uv run python -m mediaindex   then open http://127.0.0.1:8765")
    return 0 if ui else 1
