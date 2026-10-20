"""Pick the look-alike threshold (Search → Find look-alikes) with the real model. REAL MODEL, run by hand.

Takes N photos of an indexed library, makes look-alike copies of each (smaller re-save, crop, brighter, slightly
tilted: what burst shots and edited copies look like), embeds only the copies, and compares each copy's similarity to
its original with the closest *other* photo of the library (stored vectors, read-only). Copies are written to a temp
folder, never next to the originals.

    uv run python evaluation/bench_lookalikes.py --library stockimages-cc0 --n 30
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "resave": lambda im: im.resize((im.width // 2, im.height // 2)),
    "crop": lambda im: im.crop((int(im.width * .08), int(im.height * .08), int(im.width * .92), int(im.height * .92))),
    "brighter": lambda im: ImageEnhance.Brightness(im).enhance(1.25),
    "tilt": lambda im: im.rotate(3, resample=Image.Resampling.BICUBIC, expand=False).crop(
        (int(im.width * .04), int(im.height * .04), int(im.width * .96), int(im.height * .96))),
}


def main() -> int:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from mediaindex.config import Settings
    from mediaindex.media.images import load_image
    from mediaindex.model.backend import make_backend
    from mediaindex.model.profiles import IndexProfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--db", default=str(ROOT / "data" / "mediaindex.sqlite3"))
    a = ap.parse_args()
    settings = Settings()
    profile = IndexProfile(precision=settings.resolved_precision())
    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    rows = db.execute("""SELECT a.id, a.rel_path, l.root_path, e.vector FROM assets a JOIN libraries l ON l.id=a.library_id
                         JOIN embeddings e ON e.asset_id=a.id AND e.modality='image' AND e.profile_key=?
                         WHERE l.name=? AND a.status='indexed' ORDER BY a.rel_path""", (profile.key, a.library)).fetchall()
    if not rows:
        raise SystemExit(f"no vectors for library {a.library!r} under profile {profile.key}")
    ids = [r[0] for r in rows]
    M = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    picks = list(range(0, len(rows), max(1, len(rows) // a.n)))[:a.n]
    # how close different photos of the library get (the false-alarm side), excluding identical files
    sims = M @ M.T
    np.fill_diagonal(sims, -1)
    sims[sims > 0.9999] = -1
    nearest_other = sims.max(1)
    backend = make_backend(profile, device=settings.resolved_device())
    out = {"library": a.library, "photos": len(rows), "profile": profile.key, "variants": {}}
    copies = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in VARIANTS.items():
            vals = []
            for i in picks:
                im = load_image(resolve(rows[i][2], rows[i][1]))
                p = Path(tmp) / f"{name}-{i}.jpg"
                fn(im).convert("RGB").save(p, "JPEG", quality=70)
                v = backend.embed_images([load_image(p)])[0]
                vals.append(float(v @ M[i]))
            copies += vals
            out["variants"][name] = {"min": round(min(vals), 4), "median": round(float(np.median(vals)), 4)}
            print(name, out["variants"][name], flush=True)
    q = np.quantile(nearest_other, [0.5, 0.9, 0.99, 1.0])
    out["nearest_other_photo"] = {"median": round(float(q[0]), 4), "p90": round(float(q[1]), 4),
                                  "p99": round(float(q[2]), 4), "max": round(float(q[3]), 4)}
    # per threshold: copies found, and photos that would be grouped with a different photo (false alarms)
    out["thresholds"] = {f"{t:.2f}": {"copies_found": round(float(np.mean(np.array(copies) >= t)), 3),
                                      "photos_with_a_false_match": round(float(np.mean(nearest_other >= t)), 3)}
                         for t in (0.88, 0.90, 0.92, 0.93, 0.94, 0.95)}
    j = int(np.argmax(nearest_other))
    out["closest_different_pair"] = [rows[j][1], rows[int(np.argmax(sims[j]))][1]]
    print(json.dumps(out, indent=2))
    dest = ROOT / "evaluation" / "bench" / f"lookalikes_{a.library}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", dest)
    return 0


def resolve(root: str, rel: str) -> Path:
    from mediaindex.paths import resolve_in_root

    return resolve_in_root(root, rel)


if __name__ == "__main__":
    raise SystemExit(main())
