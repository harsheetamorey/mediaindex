"""Optional, deterministic 500-image demo pack from KoalaAI/StockImages-CC0.

Selection: dataset revision pinned; the 40 parquet row groups (100 rows each) across both shards
are shuffled with a fixed seed and consumed in that order. Rows whose bytes are not a decodable
JPEG/PNG/WebP (the dataset contains some HTML error pages) are excluded and recorded; the first
500 valid rows (in group order) form the pack.
Only the selected row groups are fetched via HTTP range requests (not the full ~889 MB).

Writes images + `mediaindex-provenance.json` to the output folder (default: data/demo/stockimages-cc0),
then verifies against the committed selection manifest `manifests/stockimages-cc0-500.json`.

Tags are kept as inspection metadata only; they are never used in embeddings.

Usage: uv run --extra demo python scripts/download_demo.py [--out DIR] [--write-manifest]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

DATASET = "KoalaAI/StockImages-CC0"
REVISION = "206f3575579f1187548c6f47042ae9174c0a51fc"
SHARDS = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
SEED = 20261006
TARGET = 500
LICENSE_DECLARED = "CC0-1.0 (publisher-declared for the whole dataset; no per-image provenance in the dataset)"
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "manifests" / "stockimages-cc0-500.json"


class CountingFile:
    def __init__(self, f):
        self.f = f
        self.bytes_read = 0

    def read(self, n=-1):
        b = self.f.read(n)
        self.bytes_read += len(b)
        return b

    def __getattr__(self, name):
        return getattr(self.f, name)


def sniff_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    raise ValueError(f"unknown image format header {data[:16]!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "data" / "demo" / "stockimages-cc0")
    ap.add_argument("--write-manifest", action="store_true", help="(maintainers) regenerate committed manifest")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    groups = []  # (shard_idx, row_group_idx, first_global_row, n_rows)
    metas = []
    base = 0
    for si, shard in enumerate(SHARDS):
        with fs.open(f"datasets/{DATASET}@{REVISION}/{shard}", "rb") as f:
            md = pq.ParquetFile(f).metadata
        metas.append(md)
        for gi in range(md.num_row_groups):
            n = md.row_group(gi).num_rows
            groups.append((si, gi, base, n))
            base += n
    order = list(range(len(groups)))
    random.Random(SEED).shuffle(order)

    from PIL import Image

    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    excluded = []
    selected = []
    transferred = 0
    for idx in order:
        if len(entries) >= TARGET:
            break
        si, gi, first, n = groups[idx]
        selected.append((si, gi, first, n))
        with fs.open(f"datasets/{DATASET}@{REVISION}/{SHARDS[si]}", "rb", block_size=1 << 20) as raw:
            cf = CountingFile(raw)
            table = pq.ParquetFile(cf).read_row_group(gi, columns=["image", "tags"])
            transferred += cf.bytes_read
        images = table.column("image").to_pylist()
        tags = table.column("tags").to_pylist()
        for k, (img, tag) in enumerate(zip(images, tags)):
            if len(entries) >= TARGET:
                break
            data = img["bytes"]
            row = first + k
            sha = hashlib.sha256(data).hexdigest()
            try:
                ext = sniff_ext(data)
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
            except Exception as e:
                excluded.append({"row": row, "sha256": sha, "reason": str(e)[:120]})
                continue
            name = f"stock-{row:05d}{ext}"
            dest = args.out / name
            if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != sha:
                dest.write_bytes(data)
            entries.append({
                "file": name,
                "row": row,
                "shard": SHARDS[si],
                "row_group": gi,
                "sha256": sha,
                "bytes": len(data),
                "license_declared": LICENSE_DECLARED,
                "source": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION} (train split, row {row})",
                "image_provenance": None,
                "tags": (tag or "").strip().strip("'").strip(),
            })
        print(f"  shard {si} group {gi}: {n} rows, cumulative transfer {transferred / 1e6:.1f} MB")

    selection = {
        "dataset": DATASET,
        "revision": REVISION,
        "seed": SEED,
        "row_groups": [[SHARDS[si], gi] for si, gi, _, _ in selected],
        "count": len(entries),
        "excluded_rows": excluded,
        "items": [{"row": e["row"], "file": e["file"], "sha256": e["sha256"]} for e in entries],
    }
    provenance = {
        "dataset": DATASET,
        "revision": REVISION,
        "dataset_url": f"https://huggingface.co/datasets/{DATASET}",
        "license_declared": LICENSE_DECLARED,
        "note": ("Publisher declares CC0 for the dataset. Image-level provenance (original photographer/source URL) "
                 "is not included in the dataset. No independent rights audit has been performed."),
        "tags_policy": "Tags are inspection metadata only and are never embedded.",
        "items": {e["file"]: e for e in entries},
    }
    (args.out / "mediaindex-provenance.json").write_text(json.dumps(provenance, indent=1))

    keep = {e["file"] for e in entries}
    for stale in args.out.glob("stock-*"):
        if stale.name not in keep:  # only files this script created; never other user files
            stale.unlink()
    print(f"excluded {len(excluded)} invalid rows")
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(selection, indent=1) + "\n")
        print(f"wrote {MANIFEST}")
    elif MANIFEST.exists():
        expected = json.loads(MANIFEST.read_text())
        if expected["items"] != selection["items"] or expected["revision"] != REVISION:
            print("ERROR: selection differs from committed manifest", file=sys.stderr)
            return 1
        print("selection matches committed manifest")
    print(f"{len(entries)} images in {args.out}; transferred ~{transferred / 1e6:.1f} MB "
          f"(full dataset download would be ~889 MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
