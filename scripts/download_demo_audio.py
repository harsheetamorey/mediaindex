"""Optional, deterministic demo sound pack from quinnlue/FSD50K-16k: individually declared CC0 clips only.

Selection: pinned revision; the row groups (64 rows) of `validation-00000-of-00002.parquet` are shuffled with
a fixed seed and consumed in order; only rows whose per-clip `license` is the CC0 1.0 URL are kept, until
TARGET clips are collected. Only the needed row groups are fetched (HTTP range reads).

Writes FLAC files + `mediaindex-provenance.json` (freesound id/url, uploader, title, per-clip license) to
data/demo/fsd50k-cc0 and verifies the committed selection manifest manifests/fsd50k-cc0.json.
Labels/tags/titles are inspection metadata only and are never embedded.

Usage: uv run --extra demo python scripts/download_demo_audio.py [--out DIR] [--write-manifest]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

DATASET = "quinnlue/FSD50K-16k"
REVISION = "2a60d475f4e2f0db624a902881af2df79d656145"
SHARD = "data/validation-00000-of-00002.parquet"
CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
SEED = 20261007
TARGET = 60
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "manifests" / "fsd50k-cc0.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "data" / "demo" / "fsd50k-cc0")
    ap.add_argument("--write-manifest", action="store_true")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    sys.path.insert(0, str(Path(__file__).parent))
    from download_demo import CountingFile

    fs = HfFileSystem()
    path = f"datasets/{DATASET}@{REVISION}/{SHARD}"
    with fs.open(path, "rb") as f:
        md = pq.ParquetFile(f).metadata
    groups = list(range(md.num_row_groups))
    starts, acc = [], 0
    for g in groups:
        starts.append(acc)
        acc += md.row_group(g).num_rows
    random.Random(SEED).shuffle(groups)

    a.out.mkdir(parents=True, exist_ok=True)
    entries, used_groups, transferred, non_cc0 = [], [], 0, 0
    cols = ["audio", "freesound_id", "freesound_url", "license", "uploader", "title", "labels", "tags",
            "duration_seconds"]
    for g in groups:
        if len(entries) >= TARGET:
            break
        with fs.open(path, "rb", block_size=1 << 20) as raw:
            cf = CountingFile(raw)
            t = pq.ParquetFile(cf).read_row_group(g, columns=cols)
            transferred += cf.bytes_read
        used_groups.append(g)
        rows = t.to_pylist()
        for k, r in enumerate(rows):
            if len(entries) >= TARGET:
                break
            if r["license"] != CC0:
                non_cc0 += 1
                continue
            data = r["audio"]["bytes"]
            sha = hashlib.sha256(data).hexdigest()
            name = f"fsd-{r['freesound_id']}.flac"
            dest = a.out / name
            if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != sha:
                dest.write_bytes(data)
            entries.append({
                "file": name, "row": starts[g] + k, "row_group": g, "sha256": sha, "bytes": len(data),
                "freesound_id": r["freesound_id"], "freesound_url": r["freesound_url"], "uploader": r["uploader"],
                "title": r["title"], "license_declared": f"CC0-1.0 (per-clip license field: {r['license']})",
                "source": f"{r['freesound_url']} via https://huggingface.co/datasets/{DATASET} @ {REVISION}",
                "duration_seconds": r["duration_seconds"],
                "tags": ", ".join((r["labels"] or [])),  # FSD50K labels; inspection only
            })
        print(f"  row group {g}: cumulative {len(entries)} CC0 clips, transfer {transferred / 1e6:.1f} MB")

    keep = {e["file"] for e in entries}
    for stale in a.out.glob("fsd-*.flac"):
        if stale.name not in keep:
            stale.unlink()
    provenance = {
        "dataset": DATASET, "revision": REVISION, "dataset_url": f"https://huggingface.co/datasets/{DATASET}",
        "license_declared": "per clip (only CC0-1.0 clips selected)",
        "note": ("Only clips whose own license field is CC0 1.0 were selected; FSD50K as a collection is CC BY 4.0 "
                 "(cite Fonseca et al. 2020). Declarations come from Freesound uploaders via the dataset; "
                 "no independent rights audit has been performed."),
        "tags_policy": "Labels/titles are inspection metadata only and are never embedded.",
        "items": {e["file"]: e for e in entries},
    }
    (a.out / "mediaindex-provenance.json").write_text(json.dumps(provenance, indent=1))
    selection = {"dataset": DATASET, "revision": REVISION, "shard": SHARD, "seed": SEED,
                 "row_groups": used_groups, "count": len(entries),
                 "items": [{"row": e["row"], "file": e["file"], "sha256": e["sha256"],
                            "freesound_id": e["freesound_id"]} for e in entries]}
    if a.write_manifest:
        MANIFEST.write_text(json.dumps(selection, indent=1) + "\n")
        print(f"wrote {MANIFEST}")
    elif MANIFEST.exists():
        if json.loads(MANIFEST.read_text())["items"] != selection["items"]:
            print("ERROR: selection differs from committed manifest", file=sys.stderr)
            return 1
        print("selection matches committed manifest")
    print(f"{len(entries)} CC0 clips ({non_cc0} non-CC0 rows skipped) in {a.out}; transferred ~{transferred / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
