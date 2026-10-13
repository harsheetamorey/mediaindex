"""OPTIONAL baseline: open image–text models on the frozen image evaluation (evaluation/queries.json, spec v1).

Evaluation only. The app uses only EmbeddingGemma 2; these vectors never enter MediaIndex.

Configurations (each run in its own process):
  clip-l14     openai/clip-vit-large-patch14 (pinned rev), MPS fp32, 224 px
  siglip2-so400m  google/siglip2-so400m-patch14-384 (pinned rev), MPS bf16, 384 px, text padded to 64 tokens (model card)

Same library (the 500-image CC0 pack, pinned manifest), same queries, same top-10 depth as run_eval.py.
Query text is passed as written, with no prompt template, so nothing is tuned on the test queries.
Image references: the reference and identical-hash copies are excluded, as in the app.
image+text queries are skipped: neither model has a native joint image+text query.

Fair pooling: --add-to-pool merges each saved baseline's top 10 into evaluation/pool-v1.json, so the human labeller
judges every system's results blind (label_server.py shuffles and hides the source). Until those new candidates are
labelled, unlabelled results count as unknown, never as irrelevant.

Usage: uv run python evaluation/bench_images.py --config clip-l14 --images data/demo/stockimages-cc0
       uv run python evaluation/bench_images.py --add-to-pool
       uv run python evaluation/bench_images.py --summarize
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "bench"
sys.path.insert(0, str(REPO / "backend"))
from mediaindex.eval_metrics import aggregate, hit_at_k, ndcg_at_k  # noqa: E402

MODELS = {
    "clip-l14": ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41", "float32"),
    "siglip2-so400m": ("google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f", "bfloat16"),
}
GEMMA_RUN = HERE / "runs" / "20261007T090550Z_3e76b03f5fdbe69e.json"


def l2(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def embed(config: str, files: list[Path], texts: list[str]):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    mid, rev, dtype = MODELS[config]
    model = AutoModel.from_pretrained(mid, revision=rev, dtype=getattr(torch, dtype)).to("mps").eval()
    proc = AutoProcessor.from_pretrained(mid, revision=rev)
    text_kw = {"padding": "max_length", "max_length": 64} if config.startswith("siglip") else {"padding": True}

    def feats(out):  # transformers 5 returns a model output; older versions a tensor
        return (out if isinstance(out, torch.Tensor) else out.pooler_output).float().cpu().numpy()

    with torch.inference_mode():
        t = proc(text=texts, return_tensors="pt", truncation=True, **text_kw).to("mps")
        T = feats(model.get_text_features(**t))
        V = []
        for i in range(0, len(files), 16):
            ims = [Image.open(f).convert("RGB") for f in files[i:i + 16]]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to("mps", getattr(torch, dtype))
            V.append(feats(model.get_image_features(pixel_values=px)))
            print(f"  images {min(i + 16, len(files))}/{len(files)}", flush=True)
    torch.mps.synchronize()
    return l2(T), l2(np.vstack(V)), round(torch.mps.driver_allocated_memory() / 2**20, 1)


def run(config: str, images: Path):
    spec = json.loads((HERE / "queries.json").read_text())
    depth = spec["protocol"]["pool_depth"]
    items = json.loads((REPO / spec["library"]["manifest"]).read_text())["items"]
    names = [i["file"] for i in items]
    sha = {i["file"]: i["sha256"] for i in items}
    text_qs = [q for q in spec["queries"] if q["mode"] == "text"]
    t0 = time.perf_counter()
    T, V, mps_mb = embed(config, [images / n for n in names], [q["text"] for q in text_qs])
    secs = time.perf_counter() - t0
    tvec = {q["id"]: T[i] for i, q in enumerate(text_qs)}
    results = []
    for q in spec["queries"]:
        if q["mode"] == "text":
            s, banned = V @ tvec[q["id"]], set()
        elif q["mode"] == "image":
            s = V @ V[names.index(q["reference"])]
            banned = {n for n in names if sha[n] == sha[q["reference"]]}
        else:
            continue
        order = [names[j] for j in np.argsort(-s, kind="stable") if names[j] not in banned]
        results.append({"id": q["id"], "ranked": order[:depth], "excluded_identical": len(banned) - 1})
    mid, rev, dtype = MODELS[config]
    res = {"config": config, "model": {"id": mid, "revision": rev}, "device": "mps", "precision": dtype,
           "spec_version": spec["version"], "images": len(names), "query_text": "as written, no prompt template",
           "embed_seconds_total": round(secs, 1), "images_per_s": round(len(names) / secs, 2),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
           "mps_driver_mb": mps_mb, "results": results}
    OUT.mkdir(exist_ok=True)
    (OUT / f"images_{config}.json").write_text(json.dumps(res, indent=1))
    print(f"wrote evaluation/bench/images_{config}.json ({secs:.0f} s)")


def add_to_pool():
    """Merge every saved baseline's top 10 into the judgement pool (as run_eval.py does for app runs)."""
    pool_path = HERE / "pool-v1.json"
    pool = json.loads(pool_path.read_text())
    added = 0
    for p in sorted(OUT.glob("images_*.json")):
        for r in json.loads(p.read_text())["results"]:
            merged = pool.setdefault(r["id"], [])
            for c in r["ranked"]:
                if c not in merged:
                    merged.append(c)
                    added += 1
    pool_path.write_text(json.dumps(pool, indent=1))
    print(f"pool: +{added} candidates to label, {sum(len(v) for v in pool.values())} in total")


def summarize():
    spec = json.loads((HERE / "queries.json").read_text())
    pool = json.loads((HERE / "pool-v1.json").read_text())
    labels = json.loads((HERE / "labels" / "labels-v1.json").read_text())
    systems = [("EmbeddingGemma 2 (app run)", {r["id"]: r["ranked"] for r in json.loads(GEMMA_RUN.read_text())["results"]})]
    for config in MODELS:
        p = OUT / f"images_{config}.json"
        if p.exists():
            systems.append((MODELS[config][0], {r["id"]: r["ranked"] for r in json.loads(p.read_text())["results"]}))
    fmt = lambda m: "PENDING" if m["value"] is None else f"{m['value']:.3f} (n={m['n']})"  # noqa: E731
    lines = ["| Mode | System | Hit@1 | Hit@5 | nDCG@10 | Unlabelled in top 10 |", "|---|---|---|---|---|---|"]
    for mode in ("text", "image"):
        ids = [q["id"] for q in spec["queries"] if q["mode"] == mode]
        for name, ranked in systems:
            h1 = [hit_at_k(ranked[i], labels.get(i, {}), 1) for i in ids]
            h5 = [hit_at_k(ranked[i], labels.get(i, {}), 5) for i in ids]
            # nDCG only when this system's own top 10 is fully labelled (unknown is never irrelevant)
            nd = [ndcg_at_k(ranked[i], labels.get(i, {}), pool.get(i, []), 10)
                  if all(c in labels.get(i, {}) for c in ranked[i][:10]) else None for i in ids]
            unl = sum(c not in labels.get(i, {}) for i in ids for c in ranked[i][:10])
            lines.append(f"| {mode} | {name} | {fmt(aggregate(h1))} | {fmt(aggregate(h5))} | {fmt(aggregate(nd))} | "
                         f"{unl}/{10 * len(ids)} |")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(MODELS))
    ap.add_argument("--images", type=Path, default=REPO / "data" / "demo" / "stockimages-cc0")
    ap.add_argument("--add-to-pool", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.add_to_pool:
        add_to_pool()
    elif a.summarize:
        summarize()
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        run(a.config, a.images)
