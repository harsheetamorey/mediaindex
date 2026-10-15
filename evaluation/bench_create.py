"""Measure the Ask tab's image model (backend/mediaindex/creative.py): "replace the X with a Y" and "make an image of Y".

replace: the same COCO val2017 tasks as bench_edit.py (one sure, human-confirmed object per photo, 0.5-25% of it).
  The object is replaced with a fixed, plausible other kind of thing (dog -> cat, bus -> truck, ...; see SWAP).
  - new found: the detector (threshold 0.3) finds the new kind of thing mostly inside the changed area;
  - old gone: it no longer finds the old kind there (box IoU >= 0.3);
  - success: both. Pixels outside the changed area must be unchanged.
generate: a new image for "a <thing>" for 24 of the detector's object types, and 12 scenes ("a dog on a beach at
  sunset"), 2 seeds each.
  - found: the detector finds that thing (score >= 0.5).
The detector is a proxy judge (it can be fooled by an object-shaped smudge, and can miss a real but odd-looking one),
so a contact sheet is written for a visual check too.

Usage: uv run --extra create python evaluation/bench_create.py replace --n 40 [--sheet]
       uv run --extra create python evaluation/bench_create.py generate [--sheet]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(HERE))
from bench_edit import COCO, OUT, TO_DET, gt_mask, iou, tasks  # noqa: E402
from mediaindex.creative import Painter, painter_factory  # noqa: E402
from mediaindex.detect import RTDetrDetector  # noqa: E402
from mediaindex.edit import EditRefused, Editor, LamaFiller, SamSegmenter  # noqa: E402

SWAP = {"dog": "cat", "cat": "dog", "horse": "cow", "cow": "horse", "sheep": "dog", "bear": "dog", "zebra": "horse",
        "giraffe": "horse", "elephant": "horse", "bird": "cat", "car": "truck", "truck": "bus", "bus": "truck",
        "motorbike": "bicycle", "bicycle": "motorbike", "aeroplane": "bird", "boat": "car", "train": "bus",
        "cup": "bowl", "bowl": "cup", "bottle": "vase", "vase": "bottle", "wine glass": "cup", "pizza": "cake",
        "cake": "pizza", "donut": "cake", "sandwich": "pizza", "banana": "apple", "apple": "orange",
        "orange": "apple", "chair": "pottedplant", "sofa": "bed", "bed": "sofa", "tvmonitor": "laptop",
        "laptop": "book", "book": "laptop", "clock": "vase", "umbrella": "kite", "kite": "bird", "suitcase": "backpack",
        "backpack": "suitcase", "handbag": "backpack", "teddy bear": "dog", "person": "dog"}
WORDS = {"pottedplant": "potted plant", "tvmonitor": "TV", "aeroplane": "airplane", "motorbike": "motorcycle",
         "sofa": "sofa"}
SCENES = [("dog", "a dog on a beach at sunset"), ("cat", "a cat sleeping on a sofa"),
          ("car", "a red car on a city street at night"), ("bird", "a bird on a tree branch"),
          ("horse", "a horse in a snowy field"), ("boat", "a boat on a lake with mountains behind"),
          ("cup", "a cup of coffee on a wooden table"), ("bicycle", "a bicycle leaning against a brick wall"),
          ("pizza", "a pizza on a picnic table in a park"), ("umbrella", "a person with an umbrella on a rainy street"),
          ("sheep", "sheep on a green hill"), ("bus", "a bus in the rain")]
GENERATE = ["dog", "cat", "horse", "bird", "cow", "sheep", "elephant", "car", "bus", "bicycle", "motorbike", "boat",
            "aeroplane", "train", "pizza", "cake", "banana", "apple", "cup", "bottle", "chair", "laptop", "clock",
            "umbrella"]


def word(label: str) -> str:
    w = WORDS.get(label, label)
    return ("an " if w[0] in "aeiou" else "a ") + w


def replace(a) -> None:
    files, imgs, anns, cats = tasks(a.n, a.seed)
    det = RTDetrDetector(a.device)
    editor = Editor(lambda: SamSegmenter(a.device), lambda: LamaFiller(a.device), painter_factory(a.device))
    sheet = OUT / f"create_{a.tag}_sheet"
    if a.sheet:
        sheet.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in files:
        if len(rows) >= a.n:
            break
        info = imgs[name]
        im = Image.open(COCO / "val2017" / name).convert("RGB")
        w, h = im.size
        found = det.detect(im)
        best = None
        for ann in anns.get(info["id"], []):
            m = gt_mask(ann, w, h)
            label = TO_DET.get(cats[ann["category_id"]], cats[ann["category_id"]])
            if m is None or not 0.005 <= m.mean() <= 0.25 or label not in SWAP:
                continue
            x, y, bw, bh = ann["bbox"]
            gt_box = (x / w, y / h, (x + bw) / w, (y + bh) / h)
            for score, box in found.get(label, []):
                if score >= 0.75 and iou(box, gt_box) >= 0.5 and (best is None or m.mean() > best[2].mean()):
                    best = (label, box, m)
        if best is None:
            continue
        label, box, _ = best
        new = SWAP[label]
        px = (box[0] * w, box[1] * h, box[2] * w, box[3] * h)
        try:
            r = editor.replace(im, [px], word(new))
        except EditRefused as e:
            rows.append({"file": name, "label": label, "new": new, "refused": str(e)})
            continue
        after = det.detect(r.image, threshold=0.3)
        old_gone = not any(iou(b, box) >= 0.3 for _, b in after.get(label, []))

        def inside(b) -> float:  # share of a detected box that lies in the changed area
            x0, y0, x1, y1 = int(b[0] * w), int(b[1] * h), int(np.ceil(b[2] * w)), int(np.ceil(b[3] * h))
            region = r.mask[y0:y1, x0:x1]
            return float(region.mean()) if region.size else 0.0

        new_found = any(inside(b) >= 0.5 for _, b in after.get(new, []))
        diff = np.abs(np.asarray(r.image, dtype=np.int16) - np.asarray(im, dtype=np.int16)).max(axis=2)
        row = {"file": name, "label": label, "new": new, "new_found": new_found, "old_gone": old_gone,
               "success": new_found and old_gone, "untouched": bool((diff[~r.mask] == 0).all()),
               "seconds": round(r.seconds, 2)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if a.sheet:
            both = Image.new("RGB", (w * 2 + 8, h), "white")
            both.paste(im, (0, 0))
            both.paste(r.image, (w + 8, 0))
            both.save(sheet / f"{len(rows):03d}_{'ok' if row['success'] else 'MISS'}_{label}-to-{new}_{name}", quality=85)
    done = [r for r in rows if "refused" not in r]
    summary = {"tasks": len(rows), "refused": len(rows) - len(done), "edited": len(done),
               "success_rate": round(np.mean([r["success"] for r in done]), 3),
               "new_found_rate": round(np.mean([r["new_found"] for r in done]), 3),
               "old_gone_rate": round(np.mean([r["old_gone"] for r in done]), 3),
               "untouched_rate": round(np.mean([r["untouched"] for r in done]), 3),
               "seconds_median": round(float(np.median([r["seconds"] for r in done])), 2)}
    (OUT / f"create_{a.tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


def generate(a) -> None:
    det = RTDetrDetector(a.device)
    t = time.perf_counter()
    painter = Painter(a.device, mode=a.model)
    load_s = time.perf_counter() - t
    sheet = OUT / f"create_{a.tag}_sheet"
    if a.sheet:
        sheet.mkdir(parents=True, exist_ok=True)
    rows = []
    prompts = [(lab, word(lab)) for lab in GENERATE] + SCENES
    for label, prompt in prompts:
        for seed in range(a.seeds):
            painter.seed = seed
            t = time.perf_counter()
            im = painter.generate(prompt)
            s = time.perf_counter() - t
            got = [sc for sc, _ in det.detect(im).get(label, [])]
            row = {"label": label, "prompt": prompt, "scene": (label, prompt) in SCENES, "seed": seed, "found": bool(got), "score": max(got, default=0.0), "seconds": round(s, 2)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if a.sheet:
                im.save(sheet / f"{prompt.replace(' ', '-')}_{seed}_{'ok' if got else 'MISS'}.jpg", quality=85)
    summary = {"model": a.model, "images": len(rows), "found_rate": round(np.mean([r["found"] for r in rows]), 3),
               "single_found_rate": round(np.mean([r["found"] for r in rows if not r["scene"]]), 3),
               "scene_found_rate": round(np.mean([r["found"] for r in rows if r["scene"]]), 3),
               "seconds_median": round(float(np.median([r["seconds"] for r in rows])), 2), "load_seconds": round(load_s, 1)}
    (OUT / f"create_{a.tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["replace", "generate"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--model", choices=["generate", "paint"], default="paint",
                    help="paint: the inpainting model on a blank canvas (the app's choice); generate: text-to-image UNet "
                         "(download it with: python -m mediaindex.creative --download --base)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    a.tag = a.tag or a.mode
    (replace if a.mode == "replace" else generate)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
