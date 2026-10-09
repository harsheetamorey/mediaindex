"""FastAPI application factory."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import __version__
from .config import Settings
from .db import Database
from . import api_ask, api_search, api_selections
from .indexer import make_audio_embedder, make_image_embedder, make_video_embedder
from .ingest import run_image_import, thumb_path, window_thumb_path
from .search import MatrixCache
from .uploads import cleanup_stale_uploads
from .jobs import JobRunner, QueueFull
from .paths import PathRejected, resolve_in_root, validate_root
from .store import Store
from .model.backend import make_backend
from .model.host import ModelHost
from .model.profiles import IndexProfile
from .security import BodySizeLimitMiddleware, LocalGuardMiddleware
from .watch import FolderWatcher
from .detect import DetectorHost, default_factory as default_detector_factory
from .llm import OllamaChat


class LibraryCreate(BaseModel):
    path: str
    name: str | None = None


class LibraryPatch(BaseModel):
    watch: bool | None = None


class AppSettingsPatch(BaseModel):
    low_priority_indexing: bool | None = None


PICK_FOLDER_SCRIPT = ['tell me to activate',
                      'POSIX path of (choose folder with prompt "Choose a media folder for MediaIndex")']


def create_app(settings: Settings | None = None, backend_factory=None, detector_factory=None, chat=...) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    profile = IndexProfile(precision=settings.resolved_precision())
    factory = backend_factory or (lambda p: make_backend(p, device=settings.resolved_device()))
    host = ModelHost(profile, factory)
    db = Database(settings.db_path)
    store = Store(db)
    store.register_profile(profile)
    store.rekey_compatible_profiles(profile)
    store.recover_interrupted_jobs()
    cleanup_stale_uploads(settings.uploads_dir, max_age=0)
    def persist_job(job) -> None:
        store.save_job(job.to_dict(), job.library_id)

    runner = JobRunner(maxsize=settings.job_queue_size, on_update=persist_job,
                       low_priority=store.get_setting("low_priority_indexing", "0") == "1")
    watcher = FolderWatcher(store, lambda lid: submit_import(lid), interval=settings.watch_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.watch_interval > 0:
            watcher.start()
        yield
        watcher.stop()
        runner.shutdown()
        host.shutdown()
        db.close()

    app = FastAPI(title="MediaIndex", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.host = host
    app.state.runner = runner
    app.state.watcher = watcher
    app.state.db = db
    app.state.store = store
    app.state.profile = profile
    app.state.matrix_cache = MatrixCache(store)
    app.state.embed_batch = make_image_embedder(store, host)
    app.state.embed_audio = make_audio_embedder(store, host)
    app.state.embed_video = make_video_embedder(store, host, settings.thumbs_dir)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["content-type"],
    )
    app.add_middleware(LocalGuardMiddleware, allowed_origins=settings.allowed_origins)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }

    @app.get("/api/model/status")
    def model_status() -> dict:
        return host.status()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [j.to_dict() for j in runner.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = runner.get(job_id)
        if job is not None:
            return job.to_dict()
        stored = store.get_job(job_id)  # jobs from earlier processes (e.g. 'interrupted')
        if stored is None:
            raise HTTPException(404, "job not found")
        return stored

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        job = runner.cancel(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job.to_dict()

    # ---- libraries & import --------------------------------------------------
    @app.get("/api/libraries")
    def list_libraries() -> list[dict]:
        return store.list_libraries()

    @app.post("/api/libraries")
    def create_library(body: LibraryCreate) -> dict:
        try:
            root = validate_root(body.path)
        except PathRejected as e:
            raise HTTPException(400, str(e))
        return store.create_library(body.name or root.name, root)

    @app.patch("/api/libraries/{library_id}")
    def patch_library(library_id: str, body: LibraryPatch) -> dict:
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        if body.watch is not None:
            store.set_library_watch(library_id, body.watch)
        return store.get_library(library_id)

    @app.post("/api/pick-folder")
    def pick_folder() -> dict:
        """Show the native macOS folder chooser on this computer and return the chosen absolute path."""
        if sys.platform != "darwin":
            raise HTTPException(501, "the folder chooser is only available on macOS; paste the folder path instead")
        cmd = ["osascript"] + [a for line in PICK_FOLDER_SCRIPT for a in ("-e", line)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return {"cancelled": True}
        if r.returncode != 0:  # -128 = user cancelled
            return {"cancelled": True}
        path = r.stdout.strip()
        return {"cancelled": False, "path": path.rstrip("/") or path}

    @app.get("/api/settings")
    def get_settings() -> dict:
        return {"low_priority_indexing": runner.low_priority, "watch_interval_s": settings.watch_interval}

    @app.patch("/api/settings")
    def patch_settings(body: AppSettingsPatch) -> dict:
        if body.low_priority_indexing is not None:
            store.set_setting("low_priority_indexing", "1" if body.low_priority_indexing else "0")
            runner.low_priority = body.low_priority_indexing
        return get_settings()

    @app.delete("/api/libraries/{library_id}")
    def delete_library(library_id: str) -> dict:
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        store.delete_library(library_id)  # index records only; files untouched
        return {"deleted": library_id}

    def _embed_batch_factory():
        return app.state.embed_batch if hasattr(app.state, "embed_batch") else None

    def submit_import(library_id: str):
        """Queue an import for a library unless one is already queued or running (shared by API and watcher)."""
        for j in runner.list():
            if j.kind == "import" and j.library_id == library_id and j.status in ("queued", "running"):
                return j.to_dict()

        def fn(ctx):
            return run_image_import(ctx, store, settings.thumbs_dir, library_id, _embed_batch_factory(),
                                    embed_audio=app.state.embed_audio, embed_video=app.state.embed_video)

        return runner.submit("import", fn, library_id=library_id).to_dict()

    @app.post("/api/libraries/{library_id}/import")
    def import_library(library_id: str) -> dict:
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        try:
            return submit_import(library_id)
        except QueueFull as e:
            raise HTTPException(503, str(e))

    @app.get("/api/libraries/{library_id}/assets")
    def library_assets(library_id: str, status: str | None = None, media_type: str | None = None) -> list[dict]:
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        return [_public_asset(a) for a in store.list_assets(library_id, status, media_type)]

    # ---- asset serving (by immutable ID only) ----------------------------------
    def _public_asset(a: dict) -> dict:
        out = {k: a[k] for k in ("id", "library_id", "rel_path", "media_type", "size", "content_hash", "width",
                                 "height", "duration", "status", "error")}
        import json as _json

        meta = _json.loads(a["meta_json"]) if a.get("meta_json") else {}
        out["source"] = meta.get("source")
        out["thumbnail_url"] = f"/api/assets/{a['id']}/thumbnail"
        out["file_url"] = f"/api/assets/{a['id']}/file"
        return out

    def _asset_path(asset_id: str) -> tuple[dict, Path]:
        a = store.get_asset(asset_id)
        if a is None:
            raise HTTPException(404, "asset not found")
        lib = store.get_library(a["library_id"])
        try:
            p = resolve_in_root(lib["root_path"], a["rel_path"])
        except PathRejected:
            raise HTTPException(403, "asset path rejected")
        if not p.is_file():
            raise HTTPException(410, "original file is missing")
        return a, p

    @app.get("/api/assets/{asset_id}")
    def get_asset(asset_id: str) -> dict:
        a = store.get_asset(asset_id)
        if a is None:
            raise HTTPException(404, "asset not found")
        return _public_asset(a)

    @app.get("/api/assets/{asset_id}/thumbnail")
    def asset_thumbnail(asset_id: str):
        a = store.get_asset(asset_id)
        if a is None or not a["content_hash"]:
            raise HTTPException(404, "thumbnail not found")
        tp = thumb_path(settings.thumbs_dir, a["content_hash"])
        if not tp.is_file():
            raise HTTPException(404, "thumbnail not found")
        return FileResponse(tp, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/assets/{asset_id}/window-thumbnail")
    def window_thumbnail(asset_id: str, start: float):
        a = store.get_asset(asset_id)
        if a is None or not a["content_hash"] or a["media_type"] != "video":
            raise HTTPException(404, "thumbnail not found")
        tp = window_thumb_path(settings.thumbs_dir, a["content_hash"], float(start))
        if not tp.is_file():
            raise HTTPException(404, "thumbnail not found")
        return FileResponse(tp, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/assets/{asset_id}/file")
    def asset_file(asset_id: str):
        a, p = _asset_path(asset_id)
        media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
                 ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".mp4": "video/mp4"}
        return FileResponse(p, media_type=media.get(p.suffix.lower(), "application/octet-stream"))

    app.state.public_asset = _public_asset
    app.state.asset_path = _asset_path
    api_search.register(app)
    api_selections.register(app)
    api_selections.register_clips(app)
    if chat is ...:
        chat = OllamaChat(settings.ollama_url, settings.chat_model) if settings.chat_model else None
    api_ask.register(app, DetectorHost(detector_factory or default_detector_factory(settings.resolved_device())), chat)
    mount_frontend(app)
    return app


FRONTEND_DIST = Path(os.environ.get("MEDIAINDEX_UI_DIR") or Path(__file__).resolve().parents[2] / "frontend" / "dist")
CSP = ("default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'; frame-ancestors 'none'; "
       "base-uri 'none'; form-action 'self'")


def mount_frontend(app: FastAPI) -> None:
    """Serve the built UI from the same loopback origin (no CDNs, no external fonts)."""
    from fastapi.staticfiles import StaticFiles

    @app.middleware("http")
    async def security_headers(request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("Content-Security-Policy", CSP)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        return resp

    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
