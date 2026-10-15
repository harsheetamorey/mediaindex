"""Make a video from photos ("make a video of those", "turn my mountain photos into a reel").

The video shows exactly the photos asked for, in order: no generated content. Each photo gets a slow zoom
("Ken Burns") and a crossfade into the next. Photos that don't match the video's shape are shown whole over a blurred,
darkened copy of themselves instead of being cropped. Frames are resampled with torch (sub-pixel zoom, so no jitter; on
the GPU when there is one) and encoded by FFmpeg (H.264, yuv420p: plays everywhere, including phones and social apps).
"""

from __future__ import annotations

import subprocess
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


class _Zoomer:
    """Zoomed frames of a plate, resampled on `device` (bilinear; the plate is only ~8% larger than the frame)."""

    def __init__(self, size: tuple[int, int], device: str):
        import torch
        import torch.nn.functional as F

        self.torch, self.F, self.size, self.device = torch, F, size, device

    def tensor(self, plate: Image.Image):
        import numpy as np

        a = self.torch.from_numpy(np.asarray(plate, dtype=np.uint8).copy()).to(self.device)
        return a.permute(2, 0, 1)[None].float()

    def frame(self, plate, zoom: float):
        """The centre of `plate`, zoomed: zoom 1 shows the whole plate, larger shows less of it."""
        w, h = self.size
        theta = self.torch.tensor([[[1 / zoom, 0, 0], [0, 1 / zoom, 0]]], dtype=plate.dtype, device=self.device)
        grid = self.F.affine_grid(theta, [1, 3, h, w], align_corners=False)
        return self.F.grid_sample(plate, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def to_bytes(self, frame) -> bytes:
        return frame[0].clamp(0, 255).round().to(self.torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy().tobytes()


def make_slideshow(photos: list[Callable[[], Image.Image]], out: Path, size: tuple[int, int] = LANDSCAPE,
                   seconds: float = SECONDS, fade: float = FADE, fps: int = FPS,
                   progress: Callable[[int, int], None] | None = None,
                   cancelled: Callable[[], bool] = lambda: False, device: str = "cpu") -> dict:
    """Write an MP4 of `photos` (loaders, in order) to `out`. Returns its duration and frame count."""
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
    zoomer = _Zoomer(size, device)
    plates: dict = {}

    def plate(i: int):
        if i not in plates:
            for old in [j for j in plates if j < i - 1]:  # keep at most the current and the next photo
                del plates[old]
            plates[i] = zoomer.tensor(_plate(photos[i](), size))
        return plates[i]

    def at(i: int, k: int):
        """Frame k (0..per-1) of photo i; even photos zoom in, odd ones zoom out."""
        p = k / max(1, per - 1)
        z = 1.0 + ZOOM * (p if i % 2 == 0 else 1 - p)
        return zoomer.frame(plate(i), z)

    written = 0
    try:
        for i in range(len(photos)):
            start = 0 if i == 0 else overlap  # the first `overlap` frames were drawn as the previous fade
            for k in range(start, per):
                if cancelled():
                    raise InterruptedError("cancelled")
                frame = at(i, k)
                if overlap and i + 1 < len(photos) and k >= per - overlap:
                    j = k - (per - overlap)
                    a = (j + 1) / (overlap + 1)
                    frame = frame * (1 - a) + at(i + 1, j) * a
                proc.stdin.write(zoomer.to_bytes(frame))
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
