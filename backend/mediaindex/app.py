"""FastAPI application factory."""

from __future__ import annotations

import platform
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import __version__
from .config import Settings
from .db import Database
from .ingest import run_image_import, thumb_path
from .jobs import JobRunner, QueueFull
from .paths import PathRejected, resolve_in_root, validate_root
from .store import Store
from .model.backend import make_backend
from .model.host import ModelHost
from .model.profiles import IndexProfile
from .security import LocalGuardMiddleware


class LibraryCreate(BaseModel):
    path: str
    name: str | None = None


def create_app(settings: Settings | None = None, backend_factory=None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    profile = IndexProfile(precision=settings.resolved_precision())
    factory = backend_factory or (lambda p: make_backend(p, device=settings.resolved_device()))
    host = ModelHost(profile, factory)
    db = Database(settings.db_path)
    store = Store(db)
    store.register_profile(profile)
    store.recover_interrupted_jobs()
    job_libraries: dict[str, str | None] = {}

    def persist_job(job) -> None:
        store.save_job(job.to_dict(), job_libraries.get(job.id))

    runner = JobRunner(maxsize=settings.job_queue_size, on_update=persist_job)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        runner.shutdown()
        host.shutdown()
        db.close()

    app = FastAPI(title="MediaIndex", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.host = host
    app.state.runner = runner
    app.state.db = db
    app.state.store = store
    app.state.profile = profile
    app.state.job_libraries = job_libraries

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["content-type"],
    )
    app.add_middleware(LocalGuardMiddleware, allowed_origins=settings.allowed_origins)

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

    @app.delete("/api/libraries/{library_id}")
    def delete_library(library_id: str) -> dict:
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        store.delete_library(library_id)  # index records only; files untouched
        return {"deleted": library_id}

    def _embed_batch_factory():
        return app.state.embed_batch if hasattr(app.state, "embed_batch") else None

    @app.post("/api/libraries/{library_id}/import")
    def import_library(library_id: str) -> dict:
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        for j in runner.list():
            if j.kind == "import" and job_libraries.get(j.id) == library_id and j.status in ("queued", "running"):
                return j.to_dict()

        def fn(ctx):
            return run_image_import(ctx, store, settings.thumbs_dir, library_id, _embed_batch_factory())

        import uuid as _uuid

        jid = _uuid.uuid4().hex
        job_libraries[jid] = library_id
        try:
            job = runner.submit("import", fn, job_id=jid)
        except QueueFull as e:
            raise HTTPException(503, str(e))
        return job.to_dict()

    @app.get("/api/libraries/{library_id}/assets")
    def library_assets(library_id: str, status: str | None = None, media_type: str | None = None) -> list[dict]:
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        return [_public_asset(a) for a in store.list_assets(library_id, status, media_type)]

    # ---- asset serving (by immutable ID only) ----------------------------------
    def _public_asset(a: dict) -> dict:
        out = {k: a[k] for k in ("id", "library_id", "rel_path", "media_type", "size", "content_hash", "width",
                                 "height", "duration", "status", "error")}
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

    @app.get("/api/assets/{asset_id}/file")
    def asset_file(asset_id: str):
        a, p = _asset_path(asset_id)
        media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        return FileResponse(p, media_type=media.get(p.suffix.lower(), "application/octet-stream"))

    app.state.public_asset = _public_asset
    app.state.asset_path = _asset_path
    return app
