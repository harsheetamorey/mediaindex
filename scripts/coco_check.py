"""Measure the Ask counting rules on COCO val2017 (human-labelled boxes for the detector's 80 object types).

RT-DETR was trained on COCO *train*2017 (and Objects365); val2017 is held out. Photos and labels are downloaded to
data/bench/coco (git-ignored). Steps are resumable:

  python scripts/coco_check.py detect --n 300   # sample photos (seed 0), download them, run the detector (all boxes >= 0.3)
  python scripts/coco_check.py confirm          # whole-photo yes/no from the local chat model for each detected photo+type
  python scripts/coco_check.py report           # photo-level precision/recall of each rule, overall, by size and by type

"Photo level" is what Ask answers: does this photo contain a dog, and how many.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "bench" / "coco"
ANN = ROOT / "annotations" / "instances_val2017.json"
IMG = ROOT / "val2017"
DET = ROOT / "detections.json"
CONF = ROOT / "confirmations.json"


def load_gt():
    d = json.load(open(ANN))
    cats = sorted(d["categories"], key=lambda c: c["id"])
    imgs = {i["id"]: i for i in d["images"]}
    gt = defaultdict(lambda: defaultdict(int))  # image id -> category index -> count
    for a in d["annotations"]:
        gt[a["image_id"]][[c["id"] for c in cats].index(a["category_id"])] += 1
    return cats, imgs, gt


def cmd_detect(n: int) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from PIL import Image

    from mediaindex.detect import RTDetrDetector

    cats, imgs, gt = load_gt()
    ids = sorted(imgs)
    random.Random(0).shuffle(ids)
    ids = ids[:n]
    IMG.mkdir(parents=True, exist_ok=True)
    det = RTDetrDetector("mps")
    names = [det.model.config.id2label[i] for i in range(80)]
    print("label order check:", list(zip([c["name"] for c in cats], names))[:6], flush=True)
    out = json.load(open(DET)) if DET.exists() else {}
    t = time.time()
    for k, iid in enumerate(ids, 1):
        if str(iid) in out:
            continue
        f = IMG / imgs[iid]["file_name"]
        if not f.exists():
            urllib.request.urlretrieve(imgs[iid]["coco_url"], f)
        im = Image.open(f).convert("RGB")
        out[str(iid)] = {lab: [[s, b] for s, b in boxes] for lab, boxes in det.detect(im, threshold=0.3).items()}
        if k % 25 == 0:
            json.dump(out, open(DET, "w"))
            print(f"{k}/{n} photos, {time.time() - t:.0f}s", flush=True)
    json.dump(out, open(DET, "w"))
    print("done:", len(out), "photos", flush=True)


def cmd_confirm(min_score: float, max_score: float) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from PIL import Image

    from mediaindex.api_ask import encode_image
    from mediaindex.assistant import display
    from mediaindex.llm import ChatUnavailable, OllamaChat

    schema = {"type": "object", "properties": {"present": {"type": "boolean"}}, "required": ["present"]}

    def ask_present(chat, image, label):  # the whole-photo yes/no check that was measured (not used by the app)
        raw = chat.chat([{"role": "user", "content": f"Is there a {display(label, 1)} in this photo? Only say yes if "
                          "you can clearly see one.", "images": [encode_image(image)]}], schema=schema, max_tokens=20)
        try:
            return bool(json.loads(raw).get("present"))
        except ValueError as e:
            raise ChatUnavailable(raw) from e

    cats, imgs, gt = load_gt()
    chat = OllamaChat()
    if not chat.status()["available"]:
        sys.exit(chat.status()["reason"])
    dets = json.load(open(DET))
    conf = json.load(open(CONF)) if CONF.exists() else {}
    todo = [(iid, lab) for iid, d in dets.items() for lab, boxes in d.items()
            if min_score <= max(s for s, _ in boxes) < max_score and f"{iid}|{lab}" not in conf]
    print(len(todo), "photo+type pairs to ask about", flush=True)
    t = time.time()
    for k, (iid, lab) in enumerate(todo, 1):
        im = Image.open(IMG / imgs[int(iid)]["file_name"]).convert("RGB")
        try:
            conf[f"{iid}|{lab}"] = ask_present(chat, im, lab)
        except ChatUnavailable as e:
            print("skip", iid, lab, e, flush=True)
        if k % 20 == 0:
            json.dump(conf, open(CONF, "w"))
            print(f"{k}/{len(todo)} pairs, {time.time() - t:.0f}s", flush=True)
    json.dump(conf, open(CONF, "w"))


def cmd_report() -> None:
    cats, imgs, gt = load_gt()
    dets = json.load(open(DET))
    conf = json.load(open(CONF)) if CONF.exists() else {}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from transformers import AutoConfig

    from mediaindex.detect import DETECTOR_ID, DETECTOR_REVISION

    id2label = AutoConfig.from_pretrained(DETECTOR_ID, revision=DETECTOR_REVISION).id2label
    labels = [id2label[i] for i in range(80)]  # index i = the i-th COCO category by id (checked in `detect`)
    idx = {lab: i for i, lab in enumerate(labels)}

    def feats(iid, lab):
        boxes = dets[iid].get(lab, [])
        if not boxes:
            return None
        top = max(boxes, key=lambda b: b[0])
        area = (top[1][2] - top[1][0]) * (top[1][3] - top[1][1])
        return {"score": top[0], "area": area, "n": {t: sum(s >= t for s, _ in boxes) for t in (0.5, 0.75)},
                "gemma": conf.get(f"{iid}|{lab}")}

    rules = {
        "score >= 0.5": lambda f: f["score"] >= 0.5,
        "score >= 0.75": lambda f: f["score"] >= 0.75,
        ">= 0.5 and chat model yes": lambda f: f["score"] >= 0.5 and f["gemma"] is True,
        ">= 0.75, or 0.5-0.75 and chat yes": lambda f: f["score"] >= 0.75 or (f["score"] >= 0.5 and f["gemma"] is True),
        ">= 0.5; chat no removes it only if box >= 2% of photo": lambda f: f["score"] >= 0.5 and not (f["gemma"] is False and f["area"] >= 0.02),
        ">= 0.5; chat no removes it only if box >= 5% of photo": lambda f: f["score"] >= 0.5 and not (f["gemma"] is False and f["area"] >= 0.05),
    }

    def evaluate(rule, keep_label=lambda lab: True, size=None):
        tp = fp = fn = 0
        for iid in dets:
            g = gt.get(int(iid), {})
            for lab in labels:
                if not keep_label(lab):
                    continue
                f = feats(iid, lab)
                if size and f and not size(f["area"]):
                    continue
                pred = bool(f) and rule(f)
                truth = g.get(idx[lab], 0) > 0
                if size and not f:  # size buckets only describe detections
                    continue
                tp += pred and truth
                fp += pred and not truth
                fn += (not pred) and truth
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        return p, r, tp, fp, fn

    n_pairs = sum(1 for iid in dets for lab in dets[iid] if max(s for s, _ in dets[iid][lab]) >= 0.5)
    print(f"COCO val2017: {len(dets)} photos; {n_pairs} photo+type detections at score >= 0.5; "
          f"{sum(v is not None for v in conf.values())} chat-model answers\n")
    print("Photo level, all 80 types (precision = share of reported photos that really contain the thing; "
          "recall = share of photos with the thing that were found):")
    for name, rule in rules.items():
        p, r, tp, fp, fn = evaluate(rule)
        print(f"  {name:55s} precision {p:.3f}  recall {r:.3f}   (tp {tp}, fp {fp}, fn {fn})")
    print("\nChat model 'no' answers, by size of the detector's best box (among detections with score >= 0.5):")
    for lo, hi in ((0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 1.01)):
        right = wrong = 0
        for iid in dets:
            for lab in dets[iid]:
                f = feats(iid, lab)
                if f["score"] < 0.5 or f["gemma"] is not False or not lo <= f["area"] < hi:
                    continue
                truth = gt.get(int(iid), {}).get(idx[lab], 0) > 0
                right += not truth
                wrong += truth
        tot = right + wrong
        print(f"  box {lo * 100:4.1f}-{min(hi, 1) * 100:5.1f}% of photo: {tot:4d} 'no' answers, "
              f"{right / tot if tot else float('nan'):.2f} of them correct")
    print("\nBy type (score >= 0.5 | score >= 0.75), types with >= 5 photos true or reported:")
    final = rules["score >= 0.75"]
    for lab in labels:
        p0, r0, tp0, fp0, fn0 = evaluate(rules["score >= 0.5"], lambda x, lab=lab: x == lab)
        p1, r1, *_ = evaluate(final, lambda x, lab=lab: x == lab)
        if tp0 + fp0 + fn0 >= 5:
            print(f"  {lab:14s} p {p0:.2f} r {r0:.2f}  |  p {p1:.2f} r {r1:.2f}   (true photos {tp0 + fn0})")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--n", type=int, default=300)
    c = sub.add_parser("confirm")
    c.add_argument("--min-score", type=float, default=0.5)
    c.add_argument("--max-score", type=float, default=1.01)
    sub.add_parser("report")
    a = ap.parse_args()
    {"detect": lambda: cmd_detect(a.n), "confirm": lambda: cmd_confirm(a.min_score, a.max_score), "report": cmd_report}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
