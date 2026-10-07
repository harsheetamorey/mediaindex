"""Compute retrieval metrics from human labels (if any) and write docs/evaluation-report.md.

Metrics stay PENDING for queries without labels; nothing is inferred from tags or model output.
Usage: uv run python evaluation/report.py [--run evaluation/runs/<file>.json] [--perf evaluation/perf-latest.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "backend"))
from mediaindex.eval_metrics import aggregate, hit_at_k, ndcg_at_k, recall_at_k  # noqa: E402


def fmt(m: dict) -> str:
    return "PENDING (no labelled queries)" if m["value"] is None else f"{m['value']:.3f} (n={m['n']}, excluded {m['excluded']})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--perf", type=Path, default=HERE / "perf-latest.json")
    a = ap.parse_args()
    spec = json.loads((HERE / "queries.json").read_text())
    run_path = a.run or sorted((HERE / "runs").glob("*.json"))[-1]
    run = json.loads(run_path.read_text())
    pool = json.loads((HERE / "pool-v1.json").read_text())
    labels_path = HERE / "labels" / "labels-v1.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    by_id = {r["id"]: r for r in run["results"]}
    modes = {"text": [], "image": [], "image+text": []}
    for q in spec["queries"]:
        modes[q["mode"]].append(q["id"])
    lines = ["# Image retrieval evaluation report", "",
             f"Run: `{run_path.relative_to(REPO)}` · profile `{run['profile_key']}` · device `{run['device']}` · "
             f"precision `{run['profile']['precision']}` · spec v{spec['version']} (frozen {spec['frozen']})", "",
             "## Relevance metrics (human labels only)", ""]
    judged_any = sum(1 for q in spec["queries"] if labels.get(q["id"]))
    judged_full = sum(1 for q in spec["queries"] if pool.get(q["id"]) and all(c in labels.get(q["id"], {}) for c in pool[q["id"]]))
    lines += [f"Label coverage: **{judged_any}/50 queries have any labels, {judged_full}/50 have a fully judged pool** "
              f"(pool = union of top-{spec['protocol']['pool_depth']} results across runs; "
              f"{sum(len(v) for v in pool.values())} candidates).", ""]
    lines += ["| Mode | Queries | Hit@1 | Hit@5 | Recall@5 (pool-relative) | nDCG@10 |", "|---|---|---|---|---|---|"]
    for mode, ids in modes.items():
        h1, h5, r5, nd = [], [], [], []
        for qid in ids:
            ranked = by_id[qid]["ranked"]
            lab = labels.get(qid, {})
            h1.append(hit_at_k(ranked, lab, 1))
            h5.append(hit_at_k(ranked, lab, 5))
            r5.append(recall_at_k(ranked, lab, pool.get(qid, []), 5))
            nd.append(ndcg_at_k(ranked, lab, pool.get(qid, []), 10))
        lines.append(f"| {mode} | {len(ids)} | {fmt(aggregate(h1))} | {fmt(aggregate(h5))} | {fmt(aggregate(r5))} | {fmt(aggregate(nd))} |")
    lines += ["", "Rules: a candidate without a human label is *unknown*, not irrelevant. Hit@k is computed only when decidable "
              "(a labelled relevant item is in the top k, or the whole top k is labelled). Recall@5 and nDCG@10 are computed only for "
              "queries whose whole pool is judged, and recall is relative to that pool, so it overstates recall when relevant items exist "
              "outside the pool. Relevant means grade ≥ 1; nDCG uses graded gains 2^g − 1. Similarity scores are never used as relevance.", ""]
    lines += ["## End-to-end query latency (HTTP, this run)", "",
              "Client-measured per query on the warm server: embedding + ranking + JSON. Image references to library assets reuse the stored "
              "vector, so they need no embedding.", "", "| Mode | Median | p95 | Queries |", "|---|---|---|---|"]
    for mode, ids in modes.items():
        xs = sorted(by_id[q]["client_latency_ms"] for q in ids)
        lines.append(f"| {mode} | {xs[len(xs) // 2]:.0f} ms | {xs[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))]:.0f} ms | {len(xs)} |")
    lines.append("")
    if a.perf.exists():
        perf = json.loads(a.perf.read_text())
        lines += ["## Performance", "", f"Measured {perf['created']} on {perf['machine']} · device `{perf['device']}` · precision "
                  f"`{perf['precision']}` · profile `{perf['profile_key']}` · model cache warm on disk.", "",
                  "| Measure | Median | p95 | Runs |", "|---|---|---|---|"]
        for row in perf["latency"]:
            lines.append(f"| {row['name']} | {row['median_ms']:.1f} ms | {row['p95_ms']:.1f} ms | {row['n']} |")
        lines += ["", "| Resource | Value |", "|---|---|"]
        for k, v in perf["resources"].items():
            lines.append(f"| {k} | {v} |")
        lines += ["", perf.get("notes", "")]
    out = REPO / "docs" / "evaluation-report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.relative_to(REPO)}; labelled queries: {judged_any}/50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
