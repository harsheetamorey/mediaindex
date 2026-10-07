"""Loopback-only request guards: Host and Origin validation."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]", "testserver"}
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


class LocalGuardMiddleware(BaseHTTPMiddleware):
    """Reject non-loopback Host headers (DNS rebinding) and foreign Origins on mutations."""

    def __init__(self, app, allowed_origins: tuple[str, ...]):
        super().__init__(app)
        self.allowed_origins = set(allowed_origins)

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "")
        hostname = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        if hostname not in LOOPBACK_HOSTS:
            return JSONResponse({"detail": "Host not allowed"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method in MUTATING and origin is not None and origin not in self.allowed_origins:
            return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
        return await call_next(request)
