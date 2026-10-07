"""REAL-MODEL video checks: does a video input include its soundtrack? native video vs frame average;
joint video+audio; text ranking over windows of the generated demo video.

Usage: uv run python scripts/video_smoke.py --video data/demo/videos/scenes.mp4 --silent data/demo/videos/scenes-silent.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from mediaindex.media.audio import decode_audio  # noqa: E402
from mediaindex.media.video import probe_video, sample_frames  # noqa: E402
from mediaindex.model.backend import GemmaBackend  # noqa: E402
from mediaindex.model.profiles import IndexProfile, QUERY_PROMPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--silent", type=Path, required=True)
    a = ap.parse_args()
    b = GemmaBackend(IndexProfile(), device="mps")
    enc = lambda inputs, **kw: b._encode(inputs, batch_size=1, **kw)  # noqa: E731
    rep = {}
    info = probe_video(a.video)
    rep["probe"] = info.__dict__

    # 1. path input: with soundtrack vs without (first 8 s via processing_kwargs max_frames)
    t0 = time.perf_counter()
    v_path = enc([{"video": str(a.video)}], processing_kwargs={"video": {"max_frames": 8}})[0]
    v_silent = enc([{"video": str(a.silent)}], processing_kwargs={"video": {"max_frames": 8}})[0]
    rep["path_video_s"] = round((time.perf_counter() - t0) / 2, 2)
    rep["video_with_vs_without_audio_max_abs_diff"] = float(np.abs(v_path - v_silent).max())

    # 2. windows 8 s / stride 4 s, frames at 1 fps passed as native video input
    windows = [(s, min(s + 8, info.duration)) for s in np.arange(0, info.duration - 4, 4.0)]
    vis, aud, joint, avg = [], [], [], []
    t0 = time.perf_counter()
    for s, e in windows:
        frames, times = sample_frames(a.video, info, s, e - s, fps=1.0)
        meta = {"fps": 1.0, "total_num_frames": len(frames), "duration": float(len(frames))}
        vis.append(enc([{"video": {"array": frames, "video_metadata": meta}}])[0])
        wav = decode_audio(a.video, start=s, duration=e - s)
        aud.append(b.embed_audio([wav])[0])
        joint.append(enc([{"video": {"array": frames, "video_metadata": meta},
                           "audio": {"array": wav, "sampling_rate": 16000}}])[0])
        fv = b.embed_images([__import__("PIL.Image", fromlist=["Image"]).fromarray(f) for f in frames])
        m = fv.mean(0)
        avg.append(m / np.linalg.norm(m))
    rep["window_seconds_each_all_modes"] = round((time.perf_counter() - t0) / len(windows), 2)
    vis, aud, joint, avg = map(np.vstack, (vis, aud, joint, avg))
    rep["finite"] = bool(all(np.isfinite(x).all() for x in (vis, aud, joint)))
    rep["cos_native_video_vs_frame_average_mean"] = round(float((vis * avg).sum(1).mean()), 4)
    rep["cos_joint_vs_visual_mean"] = round(float((joint * vis).sum(1).mean()), 4)
    rep["cos_joint_vs_audio_mean"] = round(float((joint * aud).sum(1).mean()), 4)
    qs = ["a train on railway tracks", "a dog", "an insect in a meadow", "a city skyline at night",
          "a dog barking", "crickets chirping", "traffic noise"]
    Q = b.embed_query_texts(qs)
    rep["ranking"] = {}
    for q, qv in zip(qs, Q):
        row = {}
        for name, M in (("visual", vis), ("audio", aud), ("joint", joint)):
            i = int(np.argmax(M @ qv))
            row[name] = f"[{windows[i][0]:.0f},{windows[i][1]:.0f})"
        rep["ranking"][q] = row
        print(f"  {q!r:<28} best window visual {row['visual']:<8} audio {row['audio']:<8} joint {row['joint']}")
    print(json.dumps({k: v for k, v in rep.items() if k != "ranking"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
