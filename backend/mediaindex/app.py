"""FastAPI application factory."""

from __future__ import annotations

import platform

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import Settings
from .security import LocalGuardMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    app = FastAPI(title="MediaIndex", version=__version__)
    app.state.settings = settings

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

    return app
