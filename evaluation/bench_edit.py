"""Measure "remove the <object>" (Ask tab edits, backend/mediaindex/edit.py) on COCO val2017 photos.

Uses the 300 photos sampled by scripts/coco_check.py (data/bench/coco, human-drawn object outlines).
One task per photo: the largest object the detector is sure of (score >= 0.75) that matches a human-labelled object
(box IoU >= 0.5), whose human outline covers 0.5-25% of the photo. The edit is run exactly as in the app.

Metrics (per task, then averaged):
- outline IoU: SAM's mask vs the human outline of that object (how exactly the right pixels are picked);
- removed: after the edit, the detector (threshold lowered to 0.3, stricter than the app's 0.5) finds no object of
  that type overlapping the removed one (box IoU >= 0.3). The same detector picks and judges, so this is a check that
  the object is gone, not an independent one; a human contact sheet is written for a visual check as well;
- others kept: of the other objects the detector was sure of that lie outside the changed area, the share still
  found after the edit (same type, box IoU >= 0.5);
- untouched: pixels outside the changed area are identical to the original (by construction; verified here).

Usage: uv run python evaluation/bench_edit.py --n 100 [--sheet]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "backend"))
from mediaindex.detect import RTDetrDetector  # noqa: E402
from mediaindex.edit import EditRefused, default_editor  # noqa: E402

COCO = REPO / "data" / "bench" / "coco"
OUT = HERE / "bench"
# COCO category names -> the detector's label names, where they differ
TO_DET = {"airplane": "aeroplane", "couch": "sofa", "dining table": "diningtable", "motorcycle": "motorbike",
          "potted plant": "pottedplant", "tv": "tvmonitor"}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gt_mask(ann, w, h) -> np.ndarray | None:
    if ann["iscrowd"] or not isinstance(ann["segmentation"], list):
        return None
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for poly in ann["segmentation"]:
        if len(poly) >= 6:
            d.polygon(poly, fill=255)
    return np.asarray(im) > 0


def tasks(n: int, seed: int):
    d = json.load(open(COCO / "annotations" / "instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["file_name"]: i for i in d["images"]}
    anns: dict[int, list] = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    files = sorted(p.name for p in (COCO / "val2017").glob("*.jpg"))
    random.Random(seed).shuffle(files)
    return files, imgs, anns, cats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sheet", action="store_true", help="also write before/after pairs for a visual check")
    ap.add_argument("--tag", default="remove")
    ap.add_argument("--no-retry", action="store_true", help="skip the app's check-and-retry step")
    a = ap.parse_args()
    files, imgs, anns, cats = tasks(a.n, a.seed)
    det = RTDetrDetector(a.device)
    editor = default_editor(a.device)
    sheet_dir = OUT / f"edit_{a.tag}_sheet"
    if a.sheet:
        sheet_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in files:
        if len(rows) >= a.n:  # includes declined tasks
            break
        info = imgs[name]
        im = Image.open(COCO / "val2017" / name).convert("RGB")
        w, h = im.size
        found = det.detect(im)
        best = None
        for ann in anns.get(info["id"], []):
            m = gt_mask(ann, w, h)
            if m is None or not 0.005 <= m.mean() <= 0.25:
                continue
            label = TO_DET.get(cats[ann["category_id"]], cats[ann["category_id"]])
            x, y, bw, bh = ann["bbox"]
            gt_box = (x / w, y / h, (x + bw) / w, (y + bh) / h)
            for score, box in found.get(label, []):
                if score >= 0.75 and iou(box, gt_box) >= 0.5 and (best is None or m.mean() > best[2].mean()):
                    best = (label, box, m)
        if best is None:
            continue
        label, box, truth = best
        px = (box[0] * w, box[1] * h, box[2] * w, box[3] * h)
        t = time.perf_counter()
        raw = editor.segment(im, [px])
        seg_s = time.perf_counter() - t
        def still_there(edited, label=label):  # the app's check: the same type detected again (threshold 0.5)
            return [(b[0] * w, b[1] * h, b[2] * w, b[3] * h) for _, b in det.detect(edited).get(label, [])]

        try:
            r = editor.remove(im, [px], still_there=None if a.no_retry else still_there)
        except EditRefused as e:  # declined, photo unchanged: counted separately, not as a removal
            rows.append({"file": name, "label": label, "area": round(float(truth.mean()), 4), "refused": str(e)})
            print(json.dumps(rows[-1]), flush=True)
            continue
        inter, union = (raw & truth).sum(), (raw | truth).sum()
        after = det.detect(r.image, threshold=0.3)
        removed = not any(iou(b, box) >= 0.3 for _, b in after.get(label, []))
        others = [(lab, b) for lab, bs in found.items() for s, b in bs if s >= 0.75 and not (lab == label and b == box)]
        outside = []
        for lab, b in others:
            x0, y0, x1, y1 = int(b[0] * w), int(b[1] * h), int(np.ceil(b[2] * w)), int(np.ceil(b[3] * h))
            region = r.mask[y0:y1, x0:x1]
            if region.size and region.mean() < 0.1:
                outside.append((lab, b))
        after5 = det.detect(r.image)
        kept = sum(any(iou(b, b2) >= 0.5 for _, b2 in after5.get(lab, [])) for lab, b in outside)
        diff = np.abs(np.asarray(r.image, dtype=np.int16) - np.asarray(im, dtype=np.int16)).max(axis=2)
        untouched = bool((diff[~r.mask] == 0).all())
        row = {"file": name, "label": label, "area": round(float(truth.mean()), 4), "outline_iou": round(inter / union, 4),
               "removed": removed, "others": len(outside), "others_kept": kept, "untouched": untouched,
               "changed_area": round(float(r.mask.mean()), 4), "seconds": round(r.seconds, 2),
               "outline_seconds": round(seg_s, 2), "attempts": r.attempts, "clean": r.clean}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if a.sheet:
            both = Image.new("RGB", (w * 2 + 8, h), "white")
            both.paste(im, (0, 0))
            both.paste(r.image, (w + 8, 0))
            both.save(sheet_dir / f"{len(rows):03d}_{'ok' if removed else 'LEFT'}_{label.replace(' ', '-')}_{name}",
                      quality=85)
    refused = [r for r in rows if "refused" in r]
    all_rows, rows = rows, [r for r in rows if "refused" not in r]
    n = len(rows)
    others = sum(r["others"] for r in rows)
    summary = {
        "tasks": len(all_rows),
        "refused": len(refused),
        "edited": n,
        "outline_iou_mean": round(float(np.mean([r["outline_iou"] for r in rows])), 3),
        "outline_iou_median": round(float(np.median([r["outline_iou"] for r in rows])), 3),
        "removed_rate": round(sum(r["removed"] for r in rows) / n, 3),
        "others_kept_rate": round(sum(r["others_kept"] for r in rows) / others, 3) if others else None,
        "others_checked": others,
        "untouched_rate": round(sum(r["untouched"] for r in rows) / n, 3),
        "seconds_median": round(float(np.median([r["seconds"] for r in rows])), 2),
        "retried": sum(r["attempts"] > 1 for r in rows),
        "flagged_not_clean": sum(r["clean"] is False for r in rows),
        "removed_when_not_flagged": round(sum(r["removed"] for r in rows if r["clean"] is not False) /
                                          max(1, sum(r["clean"] is not False for r in rows)), 3),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"edit_{a.tag}.json").write_text(json.dumps({"summary": summary, "rows": all_rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
