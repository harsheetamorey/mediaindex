"""Fixed set of cross-modal examples against the demo packs (REAL MODEL, live API). Records successes and failures.

"Expected" items are chosen from publisher tags/labels purely to judge the examples by eye; tags are not ground
truth and are never embedded. Usage: uv run python scripts/crossmodal_check.py [--k 10] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from search_check import BASE, post_form  # noqa: E402

# (query kind, reference file, text, target, expected files)
CASES = [
    ("audio→image", "fsd-144028.flac", None, "image", ["stock-01642.jpg"]),                      # dog bark → dog photo
    ("audio→image", "fsd-150978.flac", None, "image", ["stock-00105.jpg", "stock-00158.jpg", "stock-02000.jpg", "stock-02043.jpg"]),  # train
    ("audio→image", "fsd-141471.flac", None, "image", ["stock-01683.jpg"]),                      # crickets → insect photo
    ("audio→image", "fsd-151373.flac", None, "image", ["stock-01609.jpg"]),                      # organ → keyboard photo
    ("audio→image", "fsd-144321.flac", None, "image", ["stock-00108.jpg", "stock-00115.jpg", "stock-00134.jpg", "stock-01672.jpg"]),  # glass
    ("image→audio", "stock-01642.jpg", None, "audio", ["fsd-144028.flac", "fsd-146343.flac"]),    # dog photo → barks
    ("image→audio", "stock-02000.jpg", None, "audio", ["fsd-150978.flac"]),                      # train photo → train
    ("image→audio", "stock-01683.jpg", None, "audio", ["fsd-141471.flac", "fsd-151776.flac"]),    # insect → crickets
    ("image→audio", "stock-01609.jpg", None, "audio", ["fsd-151373.flac"]),                      # keyboard → organ
    ("image→audio", "stock-03414.jpg", None, "audio", ["fsd-151359.flac"]),                      # car → traffic
    ("image+text→audio", "stock-01642.jpg", "barking loudly", "audio", ["fsd-144028.flac", "fsd-146343.flac"]),
    ("audio+text→image", "fsd-150978.flac", "at night", "image", ["stock-00105.jpg", "stock-00158.jpg", "stock-02000.jpg", "stock-02043.jpg"]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    assets = {}
    for lib in json.load(urllib.request.urlopen(f"{BASE}/api/libraries")):
        for x in json.load(urllib.request.urlopen(f"{BASE}/api/libraries/{lib['id']}/assets")):
            assets.setdefault(x["rel_path"], x)
    rows = []
    for kind, ref, text, target, expected in CASES:
        fields = {"asset_id": assets[ref]["id"], "target": target, "limit": a.k}
        if text:
            fields["text"] = text
        if ref.endswith(".flac"):
            path = "/api/search/audio"
        else:
            path = "/api/search/image-text" if text else "/api/search/image"
        r = post_form(path, fields, None)
        got = [x["asset"]["rel_path"] for x in r["results"]]
        ranks = [got.index(e) + 1 for e in expected if e in got]
        best = min(ranks) if ranks else None
        verdict = "hit@1" if best == 1 else f"hit@{best}" if best else f"miss (not in top {a.k})"
        print(f"{kind:<17} {ref:<16} {text or '':<15} → {verdict:<22} top3: {got[:3]}  [{r['query'].get('mode_status')}]")
        rows.append({"kind": kind, "reference": ref, "text": text, "target": target, "expected": expected,
                     "best_rank": best, "top": got, "mode_status": r["query"].get("mode_status")})
    n = len(rows)
    print(f"\nexpected item in top {a.k}: {sum(1 for x in rows if x['best_rank'])}/{n}; "
          f"at rank 1: {sum(1 for x in rows if x['best_rank'] == 1)}/{n}  (tiny hand-picked set; not an accuracy estimate)")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
