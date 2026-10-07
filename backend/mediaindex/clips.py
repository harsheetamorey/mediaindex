"""Video clip export: preview -> explicit confirm -> FFmpeg render to a partial file -> validate -> place
without overwriting, plus a provenance manifest. The source file is only ever read.

Modes:
- "accurate": re-encode (H.264/AAC). The output starts at the requested time (frame-accurate seek).
- "fast": stream copy. No quality loss and very fast, but the cut can only start on a keyframe, so the clip
  begins at the keyframe at or before the requested start (reported as `effective_start_s`).
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .jobs import JobCancelled, JobContext
from .media.audio import ffmpeg_bin
from .media.video import VideoRejected, probe_video
from .selections import PARTIAL_PREFIX, SelectionError, _free_name, _place_exclusive, validate_destination
from .store import Store

MAX_CLIP_S = 600.0
MIN_CLIP_S = 0.5
PLAN_TTL_SECONDS = 15 * 60


@dataclass
class ClipPlan:
    id: str
    asset: dict
    source: Path
    start_s: float
    end_s: float
    mode: str
    destination: Path
    dest_name: str
    effective_start_s: float
    query_context: dict | None = None
    created: float = field(default_factory=time.time)


def fmt_ts(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}m{s:04.1f}s".replace(".", "_")


def keyframe_at_or_before(path: Path, t: float) -> float:
    """Timestamp of the last video keyframe at or before t (what a stream-copy cut actually starts on)."""
    if t <= 0:
        return 0.0
    cmd = [ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
           "-show_entries", "frame=pts_time", "-of", "csv=p=0", "-read_intervals", f"{max(0.0, t - 20):.3f}%{t + 0.05:.3f}",
           "--", str(path)]
    out = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    times = []
    for line in out.stdout.decode().split():
        try:
            times.append(float(line.strip(",")))
        except ValueError:
            pass
    before = [x for x in times if x <= t + 1e-3]
    return round(max(before), 3) if before else 0.0


def plan_clip(store: Store, asset: dict, source: Path, start_s: float, end_s: float, mode: str, destination: str,
              query_context: dict | None = None) -> ClipPlan:
    if asset["media_type"] != "video":
        raise SelectionError("clips can only be exported from video assets")
    dur = float(asset["duration"] or 0)
    if not (0 <= start_s < end_s):
        raise SelectionError("clip start must be ≥ 0 and before the end")
    end_s = min(end_s, dur)
    if end_s - start_s < MIN_CLIP_S:
        raise SelectionError(f"clip must be at least {MIN_CLIP_S}s long and inside the video ({dur:.1f}s)")
    if end_s - start_s > MAX_CLIP_S:
        raise SelectionError(f"clip longer than {MAX_CLIP_S:.0f}s; export a shorter range")
    if mode not in ("accurate", "fast"):
        raise SelectionError("mode must be 'accurate' or 'fast'")
    dest = validate_destination(store, destination)
    name = _free_name(f"{Path(asset['rel_path']).stem}_{fmt_ts(start_s)}-{fmt_ts(end_s)}.mp4", set(), dest)
    eff = start_s if mode == "accurate" else keyframe_at_or_before(source, start_s)
    return ClipPlan(uuid.uuid4().hex, asset, source, round(start_s, 3), round(end_s, 3), mode, dest, name, eff,
                    query_context)


def plan_dict(p: ClipPlan) -> dict:
    return {"plan_id": p.id, "asset_id": p.asset["id"], "source": str(p.source), "start_s": p.start_s,
            "end_s": p.end_s, "duration_s": round(p.end_s - p.start_s, 3), "mode": p.mode,
            "effective_start_s": p.effective_start_s, "destination": str(p.destination), "dest_name": p.dest_name,
            "manifest_name": p.dest_name.rsplit(".", 1)[0] + ".mediaindex.json", "overwrites": 0,
            "note": ("accurate: re-encoded, starts exactly at start_s" if p.mode == "accurate" else
                     "fast: stream copy, starts at the keyframe at effective_start_s (≤ start_s); no re-encode")}


def ffmpeg_cmd(source: Path, start: float, end: float, mode: str, out: Path) -> list[str]:
    cmd = [ffmpeg_bin("ffmpeg"), "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(source),
           "-t", f"{end - start:.3f}"]
    if mode == "accurate":
        cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-avoid_negative_ts", "make_zero"]
    return cmd + ["-movflags", "+faststart", "-f", "mp4", str(out)]


def render_clip(ctx: JobContext | None, source: Path, start: float, end: float, mode: str, out: Path,
                effective_start: float) -> dict:
    """Run FFmpeg into `out` (a partial path); cancellable; validates the result. Raises on failure."""
    proc = subprocess.Popen(ffmpeg_cmd(source, start, end, mode, out), stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    try:
        while proc.poll() is None:
            if ctx is not None and ctx.cancelled:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise JobCancelled()
            time.sleep(0.1)
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        if proc.returncode != 0:
            raise OSError(f"ffmpeg failed: {err.strip()[:300]}")
        try:
            info = probe_video(out)
        except VideoRejected as e:
            raise OSError(f"exported clip is not a valid video: {e}") from e
        want = end - start
        got = info.duration
        tol = 0.15 if mode == "accurate" else max(0.6, start - effective_start + 0.3)
        if abs(got - want) > tol and not (mode == "fast" and got >= want - 0.6):
            raise OSError(f"exported duration {got:.2f}s differs from requested {want:.2f}s")
        return {"output_duration_s": round(got, 3), "video_codec": info.codec, "has_audio": info.has_audio,
                "width": info.width, "height": info.height}
    finally:
        if proc.poll() is None:
            proc.kill()


def run_clip_export(ctx: JobContext, plan: ClipPlan) -> dict:
    plan.destination.mkdir(parents=False, exist_ok=True)
    tmp = plan.destination / f"{PARTIAL_PREFIX}{uuid.uuid4().hex}.mp4"
    ctx.progress(0, 1, "rendering clip")
    try:
        check = render_clip(ctx, plan.source, plan.start_s, plan.end_s, plan.mode, tmp, plan.effective_start_s)
        taken: set[str] = set()
        final = _place_exclusive(tmp, plan.destination, plan.dest_name, taken)
    finally:
        tmp.unlink(missing_ok=True)
    meta = json.loads(plan.asset.get("meta_json") or "{}")
    manifest = {
        "format": "mediaindex-clip-manifest", "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clip": final, "asset_id": plan.asset["id"], "source_path": str(plan.source),
        "source_sha256": plan.asset["content_hash"], "requested_range_s": [plan.start_s, plan.end_s],
        "mode": plan.mode, "effective_start_s": plan.effective_start_s, "output": check,
        "source_record": meta.get("source"), "query_context": plan.query_context,
        "note": ("Timestamps come from fixed index windows and sampling; they are not exact event boundaries. "
                 "source_record reproduces publisher declarations and is not a rights audit."),
    }
    mtmp = plan.destination / f"{PARTIAL_PREFIX}{uuid.uuid4().hex}.json"
    mtmp.write_text(json.dumps(manifest, indent=2))
    mname = _place_exclusive(mtmp, plan.destination, final.rsplit(".", 1)[0] + ".mediaindex.json", set())
    ctx.progress(1, 1, "done")
    return {"clip": final, "manifest": mname, "destination": str(plan.destination), **check}
