import hashlib
import io
import os
import time
from pathlib import Path

import pytest
from PIL import Image


def make_image(path: Path, color=(200, 30, 30), size=(64, 48), fmt="JPEG", exif_orientation=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    kw = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kw["exif"] = exif.tobytes()
    im.save(path, fmt, **kw)
    return path


def tree_digest(root: Path) -> dict[str, tuple[str, int]]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            out[str(p.relative_to(root))] = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
    return out


@pytest.fixture
def media_folder(tmp_path):
    root = tmp_path / "media folder"  # path with a space
    make_image(root / "red.jpg", (220, 20, 20))
    make_image(root / "sub/green.png", (20, 200, 20), fmt="PNG")
    make_image(root / "sub/blue.webp", (20, 20, 220), fmt="WEBP")
    make_image(root / "rotated.jpg", (120, 120, 0), size=(80, 40), exif_orientation=6)
    make_image(root / "dupe/red-copy.jpg", (220, 20, 20))
    (root / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0not really a jpeg")
    good = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(good, "JPEG")
    (root / "truncated.jpg").write_bytes(good.getvalue()[:200])
    (root / "notes.txt").write_text("ignored")
    (root / ".hidden.jpg").write_bytes(good.getvalue())
    outside = tmp_path / "outside"
    make_image(outside / "secret.jpg", (1, 2, 3))
    os.symlink(outside / "secret.jpg", root / "escape.jpg")
    os.symlink(outside, root / "linkdir")
    return root
