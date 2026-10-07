"""REAL-MODEL check that stored video windows refer to the correct source intervals.

For an indexed MP4, re-extracts each stored window's frames with an independent decoder (torchcodec,
not the FFmpeg CLI used at indexing), embeds them with the native video input, and compares with the
stored `video-visual` vectors; a +2 s shifted control should match less. Then ranks windows for text
queries describing the known scenes (from the generated video's provenance). Run with the server stopped.

Usage: uv run python scripts/video_offsets_check.py --rel-path scenes.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from mediaindex.config import Settings  # noqa: E402
from mediaindex.db import Database  # noqa: E402
from mediaindex.media.video import MAX_FRAME_SIDE  # noqa: E402
from mediaindex.model.backend import GemmaBackend  # noqa: E402
from mediaindex.model.profiles import IndexProfile  # noqa: E402
from mediaindex.paths import resolve_in_root  # noqa: E402
from mediaindex.store import Store, blob_to_vec  # noqa: E402


def frames_torchcodec(path: Path, times: list[float], max_side: int) -> np.ndarray:
    import torch
    from torchcodec.decoders import VideoDecoder

    d = VideoDecoder(str(path))
    batch = d.get_frames_played_at(seconds=times).data  # (T, C, H, W) uint8
    t, c, h, w = batch.shape
    s = min(1.0, max_side / max(w, h))
    nh, nw = max(2, int(h * s) // 2 * 2), max(2, int(w * s) // 2 * 2)
    batch = torch.nn.functional.interpolate(batch.float(), size=(nh, nw), mode="bilinear", antialias=True)
    return batch.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel-path", default="scenes.mp4")
    a = ap.parse_args()
    s = Settings()
    st = Store(Database(s.db_path))
    prof = IndexProfile(precision=s.resolved_precision())
    asset = st.db.one("SELECT * FROM assets WHERE rel_path=? AND media_type='video'", (a.rel_path,))
    lib = st.get_library(asset["library_id"])
    path = resolve_in_root(lib["root_path"], asset["rel_path"])
    meta = json.loads(asset["meta_json"])
    rows = st.db.query("""SELECT start_s, end_s, dim, vector FROM embeddings WHERE asset_id=? AND profile_key=?
                          AND modality='video-visual' ORDER BY segment_index""", (asset["id"], prof.key))
    stored = np.vstack([blob_to_vec(r["vector"], r["dim"]) for r in rows])
    print(f"{a.rel_path}: {asset['duration']:.2f}s, {len(rows)} visual windows; has_audio={meta['has_audio']}")
    b = GemmaBackend(prof, device=s.resolved_device(), offline=True)
    fps = prof.video_fps

    def emb(start, end):
        n = int(round((end - start) * fps))
        times = [t for t in (start + (k + 0.5) / fps for k in range(n)) if t < asset["duration"]]
        return b.embed_video([frames_torchcodec(path, times, MAX_FRAME_SIDE)], fps)[0]

    fresh = np.vstack([emb(r["start_s"], r["end_s"]) for r in rows])
    cos = (fresh * stored).sum(1)
    shifted = np.vstack([emb(r["start_s"] + 2, r["end_s"] + 2) for r in rows[:-1]])
    for r, c in zip(rows, cos):
        print(f"  [{r['start_s']:6.2f}, {r['end_s']:6.2f})  cos(stored, independent re-decode) = {c:.4f}")
    print(f"  control (+2 s shift): mean cos {float((shifted * stored[:-1]).sum(1).mean()):.4f}")
    scenes = (meta.get("source") or {}).get("scenes") or []
    if not scenes:
        prov = json.loads((Path(lib["root_path"]) / "mediaindex-provenance.json").read_text())
        scenes = prov["items"][a.rel_path]["scenes"]
    queries = {"train / railway": "a train on railway tracks", "dog / barking": "a dog",
               "insect, meadow / crickets": "an insect on a flower in a meadow", "city at night / traffic": "a city skyline at night"}
    ok_scenes = 0
    for sc in scenes:
        q = queries.get(sc["label"], sc["label"])
        qv = b.embed_query_texts([q])[0]
        sims = stored @ qv
        i = int(np.argmax(sims))
        w0, w1 = rows[i]["start_s"], rows[i]["end_s"]
        overlap = max(0.0, min(w1, sc["end_s"]) - max(w0, sc["start_s"]))
        good = overlap >= 4
        ok_scenes += good
        print(f"  scene [{sc['start_s']},{sc['end_s']}) {q!r}: best window [{w0}, {w1}) overlap {overlap:.1f}s "
              f"{'OK' if good else 'MISS'}")
    ok = bool(np.isfinite(stored).all() and (cos > 0.97).all())
    print(f"OFFSETS {'OK' if ok else 'MISMATCH'}; scene text queries landing on their scene: {ok_scenes}/{len(scenes)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
