"""CPU speed of EmbeddingGemma 2 image indexing (Windows has no Apple GPU), and what each speed-up costs in search quality.

Same model and weights in every variant; only how it runs on the CPU changes:
  fp32       float32, batch 2 (the Windows app in 1.1.0)
  fp32-b8    float32, batch 8
  bf16       bfloat16 weights and activations (what the Mac app uses on its GPU)
  int8       dynamic int8 quantization of the Linear layers (weights int8, activations quantized per batch)
  tok140     float32 with 140 vision soft tokens per image instead of the default 280

Quality: the frozen image evaluation (evaluation/queries.json, human labels only) on the 500-image CC0 pack, plus how far
each variant's vectors and top 10s move from fp32. Unlabelled results count as unknown, never as irrelevant.

Speed alone can be measured on generated photos (--synthetic N: phone-camera sizes, same amount of model work per photo),
for machines without the pack, such as GitHub's Windows runners.

Usage: uv run python evaluation/bench_cpu.py --variant fp32 [--images DIR] [--limit N] [--synthetic N] [--out DIR]
       uv run python evaluation/bench_cpu.py --summarize DIR
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "backend"))
from mediaindex.eval_metrics import aggregate, hit_at_k  # noqa: E402

VARIANTS = ("fp32", "fp32-b8", "bf16", "int8", "tok140")


def load(variant: str):
    import torch
    from sentence_transformers import SentenceTransformer

    from mediaindex.model.profiles import MODEL_ID, MODEL_REVISION

    dtype = torch.bfloat16 if variant == "bf16" else torch.float32
    m = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu", model_kwargs={"dtype": dtype},
                            config_kwargs={"audio_config": None})
    m.eval()
    if variant == "int8":
        if "fbgemm" not in torch.backends.quantized.supported_engines:  # Arm CPUs
            torch.backends.quantized.engine = "qnnpack"
        q = torch.ao.quantization
        names = {n: q.default_dynamic_qconfig for n, mod in m.named_modules()  # the patch embedder reads its weight dtype
                 if isinstance(mod, torch.nn.Linear) and "patch_embedder" not in n}
        m = q.quantize_dynamic(m, names, dtype=torch.qint8)
    return m


def synthetic(n: int) -> list:
    from PIL import Image

    rng = np.random.default_rng(0)
    sizes = [(4032, 3024), (3024, 4032), (1920, 1080), (2048, 1536)]
    return [Image.fromarray(rng.integers(0, 256, (h // 8, w // 8, 3), np.uint8)).resize((w, h))
            for w, h in (sizes[i % len(sizes)] for i in range(n))]


def run(variant: str, images: Path, limit: int, out: Path, n_synthetic: int = 0, threads: int = 0, tag: str = "") -> None:
    import torch
    from PIL import Image

    if threads:
        torch.set_num_threads(threads)

    spec = json.loads((HERE / "queries.json").read_text())
    items = json.loads((REPO / spec["library"]["manifest"]).read_text())["items"][:limit]
    names = [i["file"] for i in items]
    texts = [q["text"] for q in spec["queries"] if q["mode"] == "text"]
    batch = 8 if variant == "fp32-b8" else 2
    kw = {"processing_kwargs": {"image": {"max_soft_tokens": 140}}} if variant == "tok140" else {}

    t0 = time.perf_counter()
    m = load(variant)
    load_s = time.perf_counter() - t0

    def enc(inputs, bs):
        with torch.inference_mode():
            return m.encode(inputs, batch_size=bs, convert_to_numpy=True, normalize_embeddings=True, **kw)

    if n_synthetic:
        ims, names = synthetic(n_synthetic), []
    else:
        ims = [Image.open(images / n).convert("RGB") for n in names]
    enc([{"image": ims[0]}], 1)  # warm-up
    vecs, per_batch = [], []
    t0 = time.perf_counter()
    for i in range(0, len(ims), batch):
        tb = time.perf_counter()
        vecs.append(enc([{"image": im} for im in ims[i:i + batch]], batch))
        per_batch.append((time.perf_counter() - tb) / len(ims[i:i + batch]))
        print(f"  {variant}: {min(i + batch, len(ims))}/{len(ims)}  {per_batch[-1]:.2f} s/image", flush=True)
    img_s = time.perf_counter() - t0
    kw.clear()  # queries are text only
    tq = []
    T = []
    for t in texts:
        tb = time.perf_counter()
        T.append(enc(["task: search result | query: " + t], 1)[0])
        tq.append(time.perf_counter() - tb)

    out.mkdir(parents=True, exist_ok=True)
    if names:
        np.savez(out / f"{variant}{tag}.npz", images=np.vstack(vecs).astype(np.float32),
                 texts=np.vstack(T).astype(np.float32), names=np.array(names))
    res = {"variant": variant + tag, "batch": batch, "images": len(ims), "synthetic": bool(n_synthetic),
           "platform": platform.platform(),
           "cpu": platform.processor(), "cores": os.cpu_count(), "torch_threads": torch.get_num_threads(),
           "cpu_capability": torch.backends.cpu.get_cpu_capability(), "load_s": round(load_s, 1),
           "s_per_image": round(img_s / len(ims), 3), "s_per_image_median": round(statistics.median(per_batch), 3),
           "query_s_median": round(statistics.median(tq), 3)}
    try:
        import psutil
        res["rss_mb"] = round(psutil.Process().memory_info().rss / 2**20)
    except ImportError:
        pass
    (out / f"{variant}{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def summarize(d: Path) -> None:
    spec = json.loads((HERE / "queries.json").read_text())
    labels = json.loads((HERE / "labels" / "labels-v1.json").read_text())
    sha = {i["file"]: i["sha256"] for i in json.loads((REPO / spec["library"]["manifest"]).read_text())["items"]}
    text_qs = [q for q in spec["queries"] if q["mode"] == "text"]
    image_qs = [q for q in spec["queries"] if q["mode"] == "image"]

    def ranked(z) -> dict[str, list[str]]:
        V, T, names = z["images"], z["texts"], list(z["names"])
        r = {}
        for k, q in enumerate(text_qs):
            r[q["id"]] = [names[j] for j in np.argsort(-(V @ T[k]), kind="stable")][:10]
        for q in image_qs:
            if q["reference"] not in names:
                continue
            banned = {n for n in names if sha[n] == sha[q["reference"]]}
            s = V @ V[names.index(q["reference"])]
            r[q["id"]] = [names[j] for j in np.argsort(-s, kind="stable") if names[j] not in banned][:10]
        return r

    base = np.load(d / "fp32.npz") if (d / "fp32.npz").exists() else None
    base_r = ranked(base) if base is not None else None
    fmt = lambda m: "–" if m["value"] is None else f"{m['value']:.3f} (n={m['n']})"  # noqa: E731
    lines = ["| Variant | s/image | vs fp32 | query s | Text Hit@1 | Text Hit@5 | Image Hit@5 | cos to fp32 (min) "
             "| top-10 overlap with fp32 |", "|---|---|---|---|---|---|---|---|---|"]
    base_s = json.loads((d / "fp32.json").read_text())["s_per_image"] if (d / "fp32.json").exists() else None
    for v in sorted((p.stem for p in d.glob("*.json")), key=lambda v: (VARIANTS.index(v.split("-t")[0]), v)):
        res = json.loads((d / f"{v}.json").read_text())
        speed = f"{base_s / res['s_per_image']:.2f}×" if base_s else "–"
        if not (d / f"{v}.npz").exists():  # speed only (generated photos)
            lines.append(f"| {v} | {res['s_per_image']} | {speed} | {res['query_s_median']} | – | – | – | – | – |")
            continue
        z = np.load(d / f"{v}.npz")
        r = ranked(z)
        h = lambda qs, k: aggregate([hit_at_k(r[q["id"]], labels.get(q["id"], {}), k) for q in qs if q["id"] in r])  # noqa: E731
        cos = overlap = "–"
        if base is not None and v != "fp32":
            c = np.sum(z["images"] * base["images"], axis=1)
            cos = f"{c.mean():.4f} ({c.min():.4f})"
            overlap = f"{np.mean([len(set(r[k]) & set(base_r[k])) / 10 for k in r]):.2f}"
        lines.append(f"| {v} | {res['s_per_image']} | {speed} | {res['query_s_median']} | {fmt(h(text_qs, 1))} | "
                     f"{fmt(h(text_qs, 5))} | {fmt(h(image_qs, 5))} | {cos} | {overlap} |")
    first = json.loads(next(d.glob("*.json")).read_text())
    print(f"{first['platform']}, {first['cores']} cores, torch threads {first['torch_threads']}, "
          f"CPU {first['cpu_capability']}, {first['images']} images\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--images", type=Path, default=REPO / "data" / "demo" / "stockimages-cc0")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (default: torch's own choice)")
    ap.add_argument("--tag", default="", help="suffix for the result files")
    ap.add_argument("--out", type=Path, default=HERE / "bench" / "cpu")
    ap.add_argument("--summarize", type=Path)
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
    else:
        run(a.variant, a.images, a.limit, a.out, a.synthetic, a.threads, a.tag)
