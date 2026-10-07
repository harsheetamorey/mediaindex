"""Video probing, frame sampling and thumbnails via FFmpeg/ffprobe (argument lists only).

Video embeddings use the model's native video input: frames sampled at a fixed rate are passed as
one sequence (vision encoder, 140 soft tokens per frame by default). The soundtrack is handled
separately by the audio encoder; passing frames does not include audio (verified in Phase 14).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .audio import SAFE_INPUT, ffmpeg_bin

VIDEO_EXTENSIONS = {".mp4"}
SUPPORTED_VIDEO_CODECS = {"h264", "hevc", "mpeg4", "av1", "vp9"}
MAX_FILE_BYTES = 8 * 1024 * 1024 * 1024
MAX_DURATION_S = 6 * 3600
MAX_FRAME_SIDE = 768


class VideoRejected(ValueError):
    pass


@dataclass
class VideoInfo:
    codec: str
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    audio_codec: str | None


def probe_video(path: Path) -> VideoInfo:
    size = path.stat().st_size
    if size == 0:
        raise VideoRejected("empty file")
    if size > MAX_FILE_BYTES:
        raise VideoRejected(f"file too large ({size} bytes)")
    cmd = [ffmpeg_bin("ffprobe"), *SAFE_INPUT, "-v", "error", "-show_entries",
           "stream=codec_type,codec_name,width,height,avg_frame_rate:format=duration,format_name",
           "-of", "json", "--", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    except subprocess.TimeoutExpired as e:
        raise VideoRejected("ffprobe timed out") from e
    if out.returncode != 0:
        raise VideoRejected(f"cannot read video: {out.stderr.decode(errors='replace').strip()[:200]}")
    data = json.loads(out.stdout or b"{}")
    fmt = data.get("format") or {}
    if "mp4" not in (fmt.get("format_name") or ""):
        raise VideoRejected(f"unsupported container {fmt.get('format_name')!r}; MP4 is supported")
    streams = data.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise VideoRejected("no video stream")
    if v.get("codec_name") not in SUPPORTED_VIDEO_CODECS:
        raise VideoRejected(f"unsupported video codec {v.get('codec_name')!r}; supported: "
                            f"{', '.join(sorted(SUPPORTED_VIDEO_CODECS))}")
    try:
        num, den = (v.get("avg_frame_rate") or "0/1").split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        duration = float(fmt.get("duration") or 0)
    except ValueError:
        fps, duration = 0.0, 0.0
    if duration <= 0:
        raise VideoRejected("unknown or zero duration")
    if duration > MAX_DURATION_S:
        raise VideoRejected(f"video too long ({duration:.0f}s)")
    return VideoInfo(v["codec_name"], int(v.get("width") or 0), int(v.get("height") or 0), fps, duration,
                     a is not None, a.get("codec_name") if a else None)


def _scaled(info: VideoInfo, max_side: int) -> tuple[int, int]:
    w, h = info.width, info.height
    s = min(1.0, max_side / max(w, h))
    return max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)


def _decode_frame(path: Path, t: float, w: int, h: int) -> np.ndarray | None:
    cmd = [ffmpeg_bin("ffmpeg"), "-nostdin", *SAFE_INPUT, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-map", "0:v:0",
           "-an", "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    except subprocess.TimeoutExpired as e:
        raise VideoRejected("frame decoding timed out") from e
    if out.returncode != 0:
        raise VideoRejected(f"cannot decode video frames: {out.stderr.decode(errors='replace').strip()[:200]}")
    if len(out.stdout) < w * h * 3:
        return None
    return np.frombuffer(out.stdout[: w * h * 3], dtype=np.uint8).reshape(h, w, 3)


def sample_frames(path: Path, info: VideoInfo, start: float, duration: float, fps: float = 1.0,
                  max_side: int = MAX_FRAME_SIDE) -> tuple[np.ndarray, list[float]]:
    """Frames at `fps` within [start, start+duration), each taken at the centre of its 1/fps slot
    (start + (k + 0.5)/fps) with an accurate seek. Returns (T,H,W,3) uint8 and source timestamps."""
    w, h = _scaled(info, max_side)
    n = max(1, int(round(duration * fps)))
    frames, times = [], []
    for k in range(n):
        t = start + (k + 0.5) / fps
        if t >= info.duration:
            break
        f = _decode_frame(path, t, w, h)
        if f is not None:
            frames.append(f)
            times.append(round(t, 3))
    if not frames:
        raise VideoRejected("no frames decoded in window")
    return np.stack(frames), times


def frame_at(path: Path, t: float, max_side: int = 480) -> Image.Image:
    cmd = [ffmpeg_bin("ffmpeg"), "-nostdin", *SAFE_INPUT, "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(path),
           "-frames:v", "1", "-vf", f"scale='min({max_side},iw)':-2", "-f", "image2pipe", "-vcodec", "png", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    if out.returncode != 0 or not out.stdout:
        raise VideoRejected("cannot extract frame")
    import io

    return Image.open(io.BytesIO(out.stdout)).convert("RGB")
