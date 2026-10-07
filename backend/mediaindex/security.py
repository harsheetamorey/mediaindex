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


class BodySizeLimitMiddleware:
    """Reject request bodies above `max_bytes` before they are parsed or spooled (pure ASGI)."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    if int(v) > self.max_bytes:
                        return await self._reject(send)
                except ValueError:
                    return await self._reject(send, 400, b"invalid content-length")
        seen = 0

        async def counting_receive():
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.max_bytes:
                    raise _TooLarge()
            return msg

        try:
            await self.app(scope, counting_receive, send)
        except _TooLarge:
            await self._reject(send)

    async def _reject(self, send, status=413, body=b'{"detail":"request body too large"}'):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


class _TooLarge(Exception):
    pass
