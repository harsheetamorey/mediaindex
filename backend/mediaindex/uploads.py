"""Temporary query uploads: size-limited, validated, always deleted after use."""

from __future__ import annotations

import hashlib
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
STALE_SECONDS = 3600


@contextmanager
def temp_upload(upload: UploadFile, uploads_dir: Path, allowed_ext: set[str], max_bytes: int = MAX_UPLOAD_BYTES):
    """Stream an upload to a private temp file; yields (path, sha256). The file is removed afterwards."""
    name = upload.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(415, f"unsupported file type {ext or '(none)'}; allowed: {', '.join(sorted(allowed_ext))}")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / f"{uuid.uuid4().hex}{ext}"  # never use the client-supplied name as a path
    h = hashlib.sha256()
    size = 0
    try:
        with open(dest, "wb") as f:
            while chunk := upload.file.read(1 << 20):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"upload too large (limit {max_bytes // (1024 * 1024)} MB)")
                h.update(chunk)
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, "empty upload")
        yield dest, h.hexdigest()
    finally:
        dest.unlink(missing_ok=True)


def cleanup_stale_uploads(uploads_dir: Path, max_age: float = STALE_SECONDS) -> int:
    """Remove leftovers from crashed requests (called at startup)."""
    n = 0
    if not uploads_dir.is_dir():
        return 0
    cutoff = time.time() - max_age
    for p in uploads_dir.iterdir():
        if p.is_file() and (max_age == 0 or p.stat().st_mtime < cutoff):
            p.unlink(missing_ok=True)
            n += 1
    return n
