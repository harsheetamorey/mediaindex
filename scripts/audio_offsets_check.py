"""REAL-MODEL check that stored audio window vectors correspond to the right source offsets.

For one indexed audio asset, independently re-decodes each stored [start, end) span from the original
file (FFmpeg seek), embeds it, and compares with the stored vector. Then ranks the windows for a few
text queries. Run with the server stopped (avoids loading a second model copy).

Usage: uv run python scripts/audio_offsets_check.py --rel-path medley.wav [--query "glass shattering" ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from mediaindex.config import Settings  # noqa: E402
from mediaindex.db import Database  # noqa: E402
from mediaindex.media.audio import decode_audio  # noqa: E402
from mediaindex.model.backend import GemmaBackend  # noqa: E402
from mediaindex.model.profiles import IndexProfile  # noqa: E402
from mediaindex.paths import resolve_in_root  # noqa: E402
from mediaindex.store import Store, blob_to_vec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel-path", required=True)
    ap.add_argument("--query", action="append", default=[])
    a = ap.parse_args()
    s = Settings()
    st = Store(Database(s.db_path))
    prof = IndexProfile(precision=s.resolved_precision())
    asset = st.db.one("SELECT * FROM assets WHERE rel_path=? AND media_type='audio'", (a.rel_path,))
    lib = st.get_library(asset["library_id"])
    path = resolve_in_root(lib["root_path"], asset["rel_path"])
    rows = st.db.query("""SELECT start_s, end_s, dim, vector FROM embeddings WHERE asset_id=? AND profile_key=?
                          AND modality='audio' ORDER BY segment_index""", (asset["id"], prof.key))
    print(f"{a.rel_path}: duration {asset['duration']:.2f}s, {len(rows)} windows")
    b = GemmaBackend(prof, device=s.resolved_device(), offline=True)
    spans = [decode_audio(path, start=r["start_s"], duration=r["end_s"] - r["start_s"]) for r in rows]
    fresh = b.embed_audio(spans)
    stored = np.vstack([blob_to_vec(r["vector"], r["dim"]) for r in rows])
    cos = (fresh * stored).sum(1)
    for r, c, x in zip(rows, cos, spans):
        print(f"  [{r['start_s']:6.2f}, {r['end_s']:6.2f})  samples={len(x):7d}  cos(stored, re-decoded span)={c:.4f}")
    shifted = b.embed_audio([decode_audio(path, start=r["start_s"] + 2.5, duration=r["end_s"] - r["start_s"])
                             for r in rows[:-1]])
    print(f"  control: same windows shifted by +2.5s -> mean cos {float((shifted * stored[:-1]).sum(1).mean()):.4f}")
    for q in a.query:
        qv = b.embed_query_texts([q])[0]
        sims = stored @ qv
        best = int(np.argmax(sims))
        print(f"  query {q!r}: best window [{rows[best]['start_s']}, {rows[best]['end_s']}) sim {sims[best]:.4f}; "
              f"all: {[round(float(x), 3) for x in sims]}")
    ok = bool(np.isfinite(stored).all() and (cos > 0.98).all())
    print("OFFSETS OK" if ok else "OFFSET MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
