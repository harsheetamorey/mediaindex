"""Audio probing/decoding via FFmpeg (argument lists, never a shell) into mono 16 kHz float32.

The model card specifies mono 16 kHz input. Multichannel audio is down-mixed by FFmpeg (`-ac 1`)
and resampled (`-ar 16000`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
SAMPLE_RATE = 16000
MAX_FILE_BYTES = 1024 * 1024 * 1024
MAX_DURATION_S = 4 * 3600
PROBE_TIMEOUT = 30


# Only local files and pipes: a crafted media file (e.g. playlist or external data reference) must not
# make FFmpeg open network URLs or other paths.
SAFE_INPUT = ("-protocol_whitelist", "file,pipe")


class AudioRejected(ValueError):
    pass


@dataclass
class AudioInfo:
    codec: str
    duration: float
    channels: int
    sample_rate: int


def ffmpeg_bin(name: str) -> str:
    p = shutil.which(name)
    if p is None:
        raise AudioRejected(f"{name} not found; install FFmpeg (e.g. `brew install ffmpeg`) to index audio/video")
    return p


def probe_audio(path: Path) -> AudioInfo:
    size = path.stat().st_size
    if size == 0:
        raise AudioRejected("empty file")
    if size > MAX_FILE_BYTES:
        raise AudioRejected(f"file too large ({size} bytes)")
    cmd = [ffmpeg_bin("ffprobe"), *SAFE_INPUT, "-v", "error", "-select_streams", "a:0", "-show_entries",
           "stream=codec_name,channels,sample_rate:format=duration", "-of", "json", "--", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=PROBE_TIMEOUT, check=False)
    except subprocess.TimeoutExpired as e:
        raise AudioRejected("ffprobe timed out") from e
    if out.returncode != 0:
        raise AudioRejected(f"cannot decode audio: {out.stderr.decode(errors='replace').strip()[:200]}")
    data = json.loads(out.stdout or b"{}")
    streams = data.get("streams") or []
    if not streams:
        raise AudioRejected("no audio stream")
    s = streams[0]
    try:
        duration = float((data.get("format") or {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    if duration <= 0:
        raise AudioRejected("unknown or zero duration")
    if duration > MAX_DURATION_S:
        raise AudioRejected(f"audio too long ({duration:.0f}s > {MAX_DURATION_S}s)")
    return AudioInfo(s.get("codec_name", "?"), duration, int(s.get("channels") or 0), int(s.get("sample_rate") or 0))


def decode_audio(path: Path, start: float | None = None, duration: float | None = None,
                 timeout: float = 600) -> np.ndarray:
    """Decode (a span of) a file's first audio stream to mono 16 kHz float32 in [-1, 1]."""
    cmd = [ffmpeg_bin("ffmpeg"), "-nostdin", *SAFE_INPUT, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise AudioRejected("audio decoding timed out") from e
    if out.returncode != 0:
        raise AudioRejected(f"cannot decode audio: {out.stderr.decode(errors='replace').strip()[:200]}")
    x = np.frombuffer(out.stdout, dtype="<f4").astype(np.float32)
    if x.size == 0:
        raise AudioRejected("decoded audio is empty")
    if not np.isfinite(x).all():
        raise AudioRejected("decoded audio contains non-finite samples")
    return x


def plan_windows(duration: float, window: float, stride: float, min_tail: float = 1.0) -> list[tuple[float, float]]:
    """Fixed windows covering the file. Short files get one window. A remaining tail of at least
    `min_tail` seconds gets an extra end-aligned window; a shorter tail extends the last window."""
    if duration <= window:
        return [(0.0, round(duration, 3))]
    out: list[tuple[float, float]] = []
    t = 0.0
    while t + window <= duration + 1e-9:
        out.append((round(t, 3), round(t + window, 3)))
        t += stride
    tail = duration - out[-1][1]
    if tail > 1e-6:
        if tail >= min_tail:
            out.append((round(duration - window, 3), round(duration, 3)))
        else:
            out[-1] = (out[-1][0], round(duration, 3))
    return out


def write_waveform(path: Path, dest: Path, width: int = 384, height: int = 216) -> None:
    """Render a simple waveform thumbnail (decoded at a low rate; it's only a picture)."""
    from PIL import Image, ImageDraw

    cmd = [ffmpeg_bin("ffmpeg"), "-nostdin", *SAFE_INPUT, "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1",
           "-ar", "2000", "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
    x = np.frombuffer(out.stdout, dtype="<f4") if out.returncode == 0 else np.zeros(0, np.float32)
    im = Image.new("RGB", (width, height), (32, 36, 48))
    d = ImageDraw.Draw(im)
    mid = height // 2
    if x.size:
        cols = np.array_split(np.abs(x), width)
        peak = max(float(np.abs(x).max()), 1e-6)
        for i, c in enumerate(cols):
            v = float(c.max()) / peak if c.size else 0.0
            h = max(1, int(v * (height * 0.42)))
            d.line([(i, mid - h), (i, mid + h)], fill=(124, 156, 255))
    else:
        d.line([(0, mid), (width, mid)], fill=(124, 156, 255))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    im.save(tmp, "JPEG", quality=85)
    tmp.replace(dest)
