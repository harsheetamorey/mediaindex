"""Image validation, orientation and thumbnails (Pillow; JPEG/PNG/WebP verified in Phase 4)."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}  # MPO = multi-picture JPEG from some cameras
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_PIXELS = 80_000_000
THUMB_SIZE = 384

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageRejected(ValueError):
    pass


@dataclass
class ImageInfo:
    format: str
    width: int  # after EXIF orientation
    height: int
    exif_orientation: int | None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _open_checked(path: Path) -> Image.Image:
    size = path.stat().st_size
    if size == 0:
        raise ImageRejected("empty file")
    if size > MAX_FILE_BYTES:
        raise ImageRejected(f"file too large ({size} bytes > {MAX_FILE_BYTES})")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(path)
            if im.format not in ALLOWED_FORMATS:
                raise ImageRejected(f"unsupported image format {im.format}")
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise ImageRejected(f"image too large ({w}x{h} pixels)")
            return im
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ImageRejected(f"cannot decode image: {e}") from e


def load_image(path: Path) -> Image.Image:
    """Fully decode, apply EXIF orientation, return RGB. Raises ImageRejected for corrupt/oversized files."""
    im = _open_checked(path)
    try:
        im.load()
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
    except (OSError, SyntaxError, ValueError) as e:
        raise ImageRejected(f"corrupt image: {e}") from e


def probe_image(path: Path) -> tuple[ImageInfo, Image.Image]:
    im0 = _open_checked(path)
    fmt = im0.format
    orientation = None
    try:
        orientation = im0.getexif().get(0x0112)
    except Exception:
        pass
    im = load_image(path)
    return ImageInfo(fmt, im.width, im.height, orientation), im


def write_thumbnail(im: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t = im.copy()
    t.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
    tmp = dest.with_suffix(".tmp")
    t.save(tmp, "JPEG", quality=85)
    tmp.replace(dest)
