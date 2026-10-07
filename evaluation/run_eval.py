"""Run the frozen image evaluation queries against a running MediaIndex server and store ranked results.

Writes evaluation/runs/<UTC time>_<profile key>.json and updates the judgement pool
evaluation/pool-v1.json (union of top-`pool_depth` candidates across runs). No relevance is inferred here.

Usage: uv run python evaluation/run_eval.py --library <demo library id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from search_check import BASE, post_form, post_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    a = ap.parse_args()
    spec = json.loads((HERE / "queries.json").read_text())
    depth = spec["protocol"]["pool_depth"]
    assets = {x["rel_path"]: x for x in json.load(urllib.request.urlopen(f"{BASE}/api/libraries/{a.library}/assets"))}
    status = json.load(urllib.request.urlopen(f"{BASE}/api/model/status"))
    runs = []
    for q in spec["queries"]:
        t0 = time.perf_counter()
        if q["mode"] == "text":
            r = post_json("/api/search/text", {"text": q["text"], "library_ids": [a.library], "limit": depth})
        elif q["mode"] == "image":
            r = post_form("/api/search/image", {"asset_id": assets[q["reference"]]["id"], "library_ids": a.library,
                                                "limit": depth}, None)
        else:
            r = post_form("/api/search/image-text", {"asset_id": assets[q["reference"]]["id"], "text": q["text"],
                                                     "library_ids": a.library, "limit": depth}, None)
        runs.append({"id": q["id"], "ranked": [x["asset"]["rel_path"] for x in r["results"]],
                     "similarity": [x["similarity"] for x in r["results"]],
                     "excluded_identical": r["query"].get("excluded_identical", 0),
                     "client_latency_ms": round((time.perf_counter() - t0) * 1000, 1), "timing_ms": r["timing_ms"]})
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = {"spec_version": spec["version"], "created": stamp, "profile_key": status["profile_key"],
           "profile": status["profile"], "device": status["device"], "results": runs}
    path = HERE / "runs" / f"{stamp}_{status['profile_key']}.json"
    path.write_text(json.dumps(out, indent=1))
    pool_path = HERE / "pool-v1.json"
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else {}
    for r in runs:
        merged = pool.get(r["id"], [])
        for c in r["ranked"][:depth]:
            if c not in merged:
                merged.append(c)
        pool[r["id"]] = merged
    pool_path.write_text(json.dumps(pool, indent=1))
    print(f"wrote {path.relative_to(HERE.parent)}; pool has {sum(len(v) for v in pool.values())} candidates "
          f"over {len(pool)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
