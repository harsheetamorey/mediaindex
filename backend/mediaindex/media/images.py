"""Image validation, orientation and thumbnails (Pillow; JPEG/PNG/WebP verified in Phase 4, HEIC/HEIF via pillow-heif)."""

from __future__ import annotations

import hashlib
import io
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
BROWSER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}  # others are shown as a JPEG copy
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "MPO", "HEIF"}  # MPO = multi-picture JPEG from some cameras
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
    taken_at: str | None = None  # EXIF DateTimeOriginal, camera-local time: "2024-07-14T18:03:22"
    lat: float | None = None  # EXIF GPS, degrees
    lon: float | None = None


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


EXIF_IFD, GPS_IFD = 0x8769, 0x8825
DATE_TAGS = (0x9003, 0x9004)  # DateTimeOriginal, then DateTimeDigitized
EXIF_DATE_RE = re.compile(r"^\s*(\d{4})[:-](\d{2})[:-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})")


def exif_date(exif: Image.Exif) -> str | None:
    """When the photo was taken, as ISO text in the camera's local time; None if missing or invalid."""
    ifd = exif.get_ifd(EXIF_IFD)
    for tag in DATE_TAGS:
        v = ifd.get(tag)
        if isinstance(v, bytes):
            v = v.decode("ascii", "ignore")
        m = EXIF_DATE_RE.match(v) if isinstance(v, str) else None
        if m is None:
            continue
        try:
            return datetime(*map(int, m.groups())).isoformat()
        except ValueError:  # "0000:00:00 00:00:00" and other placeholders
            continue
    return None


def _degrees(v) -> float:
    d, m, s = (float(x) for x in v)
    return d + m / 60 + s / 3600


def exif_gps(exif: Image.Exif) -> tuple[float, float] | None:
    """(latitude, longitude) in degrees; None if missing, invalid or exactly 0,0 (cameras without a fix write that)."""
    g = exif.get_ifd(GPS_IFD)
    try:
        lat, lon = _degrees(g[2]), _degrees(g[4])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if str(g.get(1, "N")).strip().upper().startswith("S"):
        lat = -lat
    if str(g.get(3, "E")).strip().upper().startswith("W"):
        lon = -lon
    if not (math.isfinite(lat) and math.isfinite(lon)) or abs(lat) > 90 or abs(lon) > 180 or (lat == 0 and lon == 0):
        return None
    return round(lat, 6), round(lon, 6)


def photo_meta(im: Image.Image) -> tuple[str | None, tuple[float, float] | None]:
    """Date taken and GPS position from an opened (not necessarily decoded) image."""
    try:
        exif = im.getexif()
    except Exception:
        return None, None
    try:
        taken = exif_date(exif)
    except Exception:
        taken = None
    try:
        gps = exif_gps(exif)
    except Exception:
        gps = None
    return taken, gps


def read_photo_meta(path: Path) -> tuple[str | None, tuple[float, float] | None]:
    """Only reads the header: for photos imported before dates were stored."""
    return photo_meta(_open_checked(path))


def probe_image(path: Path) -> tuple[ImageInfo, Image.Image]:
    im0 = _open_checked(path)
    fmt = im0.format
    orientation = None
    try:
        orientation = im0.getexif().get(0x0112)
    except Exception:
        pass
    taken, gps = photo_meta(im0)
    im = load_image(path)
    return ImageInfo(fmt, im.width, im.height, orientation, taken, *(gps or (None, None))), im


def jpeg_bytes(im: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def write_thumbnail(im: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t = im.copy()
    t.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
    tmp = dest.with_suffix(".tmp")
    t.save(tmp, "JPEG", quality=85)
    tmp.replace(dest)
