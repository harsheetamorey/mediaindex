"""Make a video from photos ("make a video of those", "turn my mountain photos into a reel").

The video shows exactly the photos asked for, in order: no generated content. Each photo gets a slow zoom
("Ken Burns") and a crossfade into the next. Photos that don't match the video's shape are shown whole over a blurred,
darkened copy of themselves instead of being cropped. Frames are resampled by Pillow (bilinear, sub-pixel zoom, so no
jitter) on several threads, and encoded by FFmpeg (H.264, yuv420p: plays everywhere, including phones and social apps).
Pillow on 4 threads draws a 1080p frame in ~11 ms on an M1; torch grid_sample took ~24 ms on the GPU for the same frame.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PIL import Image, ImageEnhance, ImageFilter

from .media.audio import ffmpeg_bin

LANDSCAPE = (1920, 1080)
VERTICAL = (1080, 1920)  # reels, shorts, stories
SQUARE = (1080, 1080)
FPS = 30
SECONDS = 3.0  # each photo, including its fade
FADE = 0.6
ZOOM = 0.08  # how far each photo zooms over its time on screen
MAX_PHOTOS = 40
THREADS = max(1, min(4, (os.cpu_count() or 2) - 1))
CHUNK = 16  # frames drawn at once (bounded memory: ~6 MB per 1080p frame)


def _plate(im: Image.Image, size: tuple[int, int], over: float = 1.0 + ZOOM) -> Image.Image:
    """The photo whole over a blurred fill, drawn at `over` x the output size so zooming in keeps full detail."""
    W, H = round(size[0] * over), round(size[1] * over)
    im = im.convert("RGB")
    cover = max(W / im.width, H / im.height)
    bg = im.resize((max(1, round(im.width * cover)), max(1, round(im.height * cover))), Image.Resampling.BILINEAR)
    bg = bg.crop(((bg.width - W) // 2, (bg.height - H) // 2, (bg.width - W) // 2 + W, (bg.height - H) // 2 + H))
    bg = ImageEnhance.Brightness(bg.resize((W // 8, H // 8)).filter(ImageFilter.GaussianBlur(6)).resize((W, H))).enhance(0.55)
    fit = min(W / im.width, H / im.height)
    fg = im.resize((max(1, round(im.width * fit)), max(1, round(im.height * fit))), Image.Resampling.LANCZOS)
    bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
    return bg


def _zoomed(plate: Image.Image, size: tuple[int, int], zoom: float) -> Image.Image:
    """The centre of `plate`, zoomed: zoom 1 shows the whole plate, larger shows less of it."""
    (W, H), (w, h) = plate.size, size
    a, e = W / w / zoom, H / h / zoom
    return plate.transform(size, Image.Transform.AFFINE, (a, 0, W / 2 - a * w / 2, 0, e, H / 2 - e * h / 2),
                           resample=Image.Resampling.BILINEAR)


def make_slideshow(photos: list[Callable[[], Image.Image]], out: Path, size: tuple[int, int] = LANDSCAPE,
                   seconds: float = SECONDS, fade: float = FADE, fps: int = FPS,
                   progress: Callable[[int, int], None] | None = None,
                   cancelled: Callable[[], bool] = lambda: False, device: str = "cpu") -> dict:
    """Write an MP4 of `photos` (loaders, in order) to `out`. Returns its duration and frame count.
    `device` is accepted for compatibility; frames are drawn on the CPU."""
    if not photos:
        raise ValueError("no photos")
    photos = photos[:MAX_PHOTOS]
    per = round(seconds * fps)
    overlap = round(fade * fps) if len(photos) > 1 else 0
    total = per * len(photos) - overlap * (len(photos) - 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.mp4")
    cmd = [ffmpeg_bin("ffmpeg"), "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    plates: dict = {}

    def plate(i: int) -> Image.Image:
        if i not in plates:
            for old in [j for j in plates if j < i - 1]:  # keep at most the current and the next photo
                del plates[old]
            plates[i] = _plate(photos[i](), size)
        return plates[i]

    def at(i: int, k: int) -> Image.Image:
        """Frame k (0..per-1) of photo i; even photos zoom in, odd ones zoom out."""
        p = k / max(1, per - 1)
        return _zoomed(plate(i), size, 1.0 + ZOOM * (p if i % 2 == 0 else 1 - p))

    def draw(job) -> bytes:
        i, k = job
        frame = at(i, k)
        if overlap and i + 1 < len(photos) and k >= per - overlap:
            j = k - (per - overlap)
            frame = Image.blend(frame, at(i + 1, j), (j + 1) / (overlap + 1))
        return frame.tobytes()

    written = 0
    try:
        with ThreadPoolExecutor(THREADS) as pool:
            for i in range(len(photos)):
                plate(i)
                if i + 1 < len(photos):
                    plate(i + 1)  # loaded here, not in the threads
                start = 0 if i == 0 else overlap  # the first `overlap` frames were drawn as the previous fade
                for c in range(start, per, CHUNK):
                    if cancelled():
                        raise InterruptedError("cancelled")
                    for data in pool.map(draw, [(i, k) for k in range(c, min(per, c + CHUNK))]):
                        proc.stdin.write(data)
                        written += 1
                if progress:
                    progress(i + 1, len(photos))
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.read().decode(errors='replace')[-300:]}")
    except BaseException:
        proc.kill()
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)
    return {"frames": written, "expected_frames": total, "seconds": round(written / fps, 2), "width": size[0],
            "height": size[1], "photos": len(photos)}
