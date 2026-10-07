"""Folder import pipeline for images: discover -> validate -> hash -> thumbnail -> (embed).

Originals are only ever opened for reading. Re-import is idempotent: unchanged files
(same size, mtime and an existing record) are skipped, changed content is re-processed under
the same asset ID, and files that disappeared are marked 'missing' (never deleted).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from .jobs import JobContext
from .media.images import IMAGE_EXTENSIONS, ImageRejected, probe_image, sha256_file, write_thumbnail
from .paths import walk_files
from .store import FAILED, INDEXED, MISSING, PENDING, Store


@dataclass
class PendingImage:
    asset_id: str
    content_hash: str
    image: Image.Image
    rel_path: str


# embed_batch(items) -> {"embedded": n, "reused": n, "failed": [(rel, err)]}
EmbedBatch = Callable[[list[PendingImage]], dict]


@dataclass
class ImportSummary:
    discovered: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    embedded: int = 0
    reused_vectors: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    cancelled: bool = False

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["failed_count"] = len(self.failed)
        d["skipped_count"] = len(self.skipped)
        d["missing_count"] = len(self.missing)
        return d


def thumb_path(thumbs_dir: Path, content_hash: str) -> Path:
    return thumbs_dir / content_hash[:2] / f"{content_hash}.jpg"


def run_image_import(ctx: JobContext, store: Store, thumbs_dir: Path, library_id: str,
                     embed_batch: EmbedBatch | None = None, batch_size: int = 4) -> dict:
    lib = store.get_library(library_id)
    if lib is None:
        raise ValueError("library not found")
    root = Path(lib["root_path"])
    if not root.is_dir():
        raise FileNotFoundError(f"library folder is not available: {root}")

    s = ImportSummary()
    files = list(walk_files(root, IMAGE_EXTENSIONS, s.skipped))
    s.discovered = len(files)
    ctx.progress(0, len(files), "scanning")
    existing = {a["rel_path"]: a for a in store.list_assets(library_id, media_type="image")}
    seen: set[str] = set()
    batch: list[PendingImage] = []

    def flush():
        if not batch or embed_batch is None:
            batch.clear()
            return
        r = embed_batch(list(batch))
        s.embedded += r.get("embedded", 0)
        s.reused_vectors += r.get("reused", 0)
        s.failed.extend(r.get("failed", []))
        batch.clear()

    try:
        for i, (path, rel) in enumerate(files):
            ctx.check_cancelled()
            seen.add(rel)
            st = path.stat()
            ex = existing.get(rel)
            done_status = INDEXED if embed_batch else PENDING
            if (ex and ex["size"] == st.st_size and ex["mtime_ns"] == st.st_mtime_ns and ex["content_hash"]
                    and (ex["status"] in (done_status, INDEXED, FAILED))
                    and (ex["status"] == FAILED or thumb_path(thumbs_dir, ex["content_hash"]).exists())):
                s.unchanged += 1
                ctx.progress(i + 1, message=f"unchanged {rel}")
                continue
            h = None
            try:
                h = sha256_file(path)
                info, im = probe_image(path)
            except (ImageRejected, OSError) as e:
                aid, _ = store.upsert_asset(library_id, rel, "image", size=st.st_size, mtime_ns=st.st_mtime_ns,
                                            content_hash=h)
                store.set_asset_status(aid, FAILED, str(e))
                s.failed.append((rel, str(e)))
                ctx.progress(i + 1, message=f"failed {rel}")
                continue
            tp = thumb_path(thumbs_dir, h)
            if not tp.exists():
                write_thumbnail(im, tp)
            aid, changed = store.upsert_asset(
                library_id, rel, "image", size=st.st_size, mtime_ns=st.st_mtime_ns, content_hash=h,
                width=info.width, height=info.height,
                meta={"format": info.format, "exif_orientation": info.exif_orientation})
            if ex is None:
                s.new += 1
            elif changed:
                s.changed += 1
            else:
                s.unchanged += 1
            asset = store.get_asset(aid)
            if embed_batch is not None and asset["status"] != INDEXED:
                batch.append(PendingImage(aid, h, im, rel))
                if len(batch) >= batch_size:
                    flush()
            ctx.progress(i + 1, message=f"processed {rel}")
        flush()
    except Exception as e:
        from .jobs import JobCancelled

        if isinstance(e, JobCancelled):
            s.cancelled = True
            ctx.job.result = s.to_dict()
        raise

    for rel, a in existing.items():
        if rel not in seen and a["status"] != MISSING:
            store.set_asset_status(a["id"], MISSING, "file not found at last import")
            s.missing.append(rel)
    return s.to_dict()
