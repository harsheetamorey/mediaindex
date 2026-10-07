"""Compare image-only, text-only and native image+text queries on a few documented cases (REAL MODEL, live API).

Publisher tags are printed for human inspection only; they are not ground truth.
Usage: uv run python scripts/refinement_check.py --library ID [--k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from search_check import BASE, post_form, post_json  # noqa: E402

CASES = [
    ("stock-00102.jpg", "yellow flowers", "colour attribute"),
    ("stock-00150.jpg", "at night", "time-of-day setting"),
    ("stock-00171.jpg", "at sunset", "lighting/setting"),
    ("stock-02096.jpg", "black and white photo", "style attribute"),
    ("stock-03482.jpg", "a car", "object (fixing the image-only failure)"),
    ("stock-00102.jpg", "without any flowers", "negation (expected to be unreliable)"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    with urllib.request.urlopen(f"{BASE}/api/libraries/{a.library}/assets") as r:
        assets = {x["rel_path"]: x for x in json.load(r)}
    out = []
    for ref, text, kind in CASES:
        aid = assets[ref]["id"]
        img = post_form("/api/search/image", {"asset_id": aid, "library_ids": a.library, "limit": a.k}, None)
        txt = post_json("/api/search/text", {"text": text, "library_ids": [a.library], "limit": a.k})
        mix = post_form("/api/search/image-text", {"asset_id": aid, "text": text, "library_ids": a.library,
                                                   "limit": a.k}, None)
        print(f"\n### {ref} + {text!r}  ({kind})")
        print(f"    reference tags: {assets[ref]['source']['inspection_tags'][:90]}")
        row = {"reference": ref, "text": text, "kind": kind}
        for label, r in (("image only", img), ("text only", txt), ("image+text", mix)):
            print(f"  {label}:")
            row[label] = []
            for x in r["results"]:
                tags = (x["asset"]["source"] or {}).get("inspection_tags", "")
                print(f"    {x['similarity']:.3f} {x['asset']['rel_path']}  [{tags[:70]}]")
                row[label].append([x["asset"]["rel_path"], round(x["similarity"], 4), tags])
        overlap_img = len({p for p, *_ in row["image+text"]} & {p for p, *_ in row["image only"]})
        overlap_txt = len({p for p, *_ in row["image+text"]} & {p for p, *_ in row["text only"]})
        print(f"  overlap of image+text top-{a.k} with image-only: {overlap_img}, with text-only: {overlap_txt}")
        row["overlap"] = {"image_only": overlap_img, "text_only": overlap_txt}
        out.append(row)
    print("\nJSON:", json.dumps(out)[:200], "...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
