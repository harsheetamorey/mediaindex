"""FastAPI application factory."""

from __future__ import annotations

import platform
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import Settings
from .db import Database
from .jobs import JobRunner
from .store import Store
from .model.backend import make_backend
from .model.host import ModelHost
from .model.profiles import IndexProfile
from .security import LocalGuardMiddleware


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

    return app
