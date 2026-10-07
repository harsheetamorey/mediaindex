"""Performance measurements on the actual machine: cold load, warm query embedding per mode, ranking,
indexing throughput, memory and disk. Accelerator work is synchronized (torch.mps.synchronize) around
every timed call; repeated runs give median/p95. Run with the MediaIndex server STOPPED (one model copy).

Writes evaluation/perf-latest.json. Usage: uv run python evaluation/benchmark.py
"""

from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "backend"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from mediaindex.config import Settings  # noqa: E402
from mediaindex.db import Database  # noqa: E402
from mediaindex.media.audio import decode_audio  # noqa: E402
from mediaindex.media.images import load_image  # noqa: E402
from mediaindex.media.video import probe_video, sample_frames  # noqa: E402
from mediaindex.model.backend import GemmaBackend  # noqa: E402
from mediaindex.model.profiles import IndexProfile  # noqa: E402
from mediaindex.search import rank  # noqa: E402
from mediaindex.store import Store  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def du_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    out = subprocess.run(["du", "-sk", "-L", str(p)], capture_output=True, text=True).stdout.split()
    return round(int(out[0]) / 1024, 1) if out else 0.0


def main() -> int:
    s = Settings()
    device, precision = s.resolved_device(), s.resolved_precision()
    prof = IndexProfile(precision=precision)
    import torch

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    lat = []

    def timed(name, fn, n, warmup=2):
        for _ in range(warmup):
            fn()
        xs = []
        for _ in range(n):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            xs.append((time.perf_counter() - t0) * 1000)
        lat.append({"name": name, "median_ms": statistics.median(xs), "p95_ms": pct(xs, 95), "n": n})
        print(f"  {name:<44} median {statistics.median(xs):8.1f} ms  p95 {pct(xs, 95):8.1f} ms  (n={n})")

    # cold model load in fresh processes (OS file cache warm after the first)
    code = ("import time,sys;sys.path.insert(0,'backend');t=time.perf_counter();"
            "from mediaindex.model.backend import GemmaBackend;from mediaindex.model.profiles import IndexProfile;"
            f"b=GemmaBackend(IndexProfile(precision='{precision}'),device='{device}');"
            "b.embed_query_texts(['warm']);print(time.perf_counter()-t)")
    cold = []
    for _ in range(3):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                             env={**os.environ, "HF_HUB_OFFLINE": "1"})
        cold.append(float(out.stdout.strip().splitlines()[-1]) * 1000)
    lat.append({"name": "cold start: import + model load + first query (new process)",
                "median_ms": statistics.median(cold), "p95_ms": max(cold), "n": 3})
    print(f"  cold start runs (ms): {[round(c) for c in cold]}")

    b = GemmaBackend(prof, device=device, offline=True)
    imgs = [load_image(p) for p in sorted((REPO / "data/demo/stockimages-cc0").glob("*.jpg"))[:24]]
    wav = decode_audio(REPO / "data/demo/fsd50k-cc0/fsd-144028.flac")[:160000]
    vpath = REPO / "data/demo/videos/scenes.mp4"
    vinfo = probe_video(vpath)
    frames, _ = sample_frames(vpath, vinfo, 12, 8)
    timed("warm query embedding: text", lambda: b.embed_query_texts(["a bridge over a river"]), 30)
    timed("warm query embedding: image (reference)", lambda: b.embed_images([imgs[0]]), 15)
    timed("warm query embedding: image + text", lambda: b.embed_image_text(imgs[1], "at night"), 15)
    timed("warm query embedding: audio 10 s", lambda: b.embed_audio([wav]), 10)
    timed("warm embedding: video window (8 frames, 1,120 tokens)", lambda: b.embed_video([frames], 1.0), 5, warmup=1)

    st = Store(Database(s.db_path))
    vm = st.load_matrix(prof.key, None, ["image"])
    q = b.embed_query_texts(["x"])[0]
    timed(f"ranking: exact dot product over {len(vm.asset_ids)} image vectors", lambda: rank(q, vm, 48), 200)
    vm_all = st.load_matrix(prof.key)
    timed(f"ranking: exact over all {len(vm_all.asset_ids)} vectors (all media)", lambda: rank(q, vm_all, 48), 200)

    # indexing throughput (model part only; decode + thumbnail excluded)
    sync()
    t0 = time.perf_counter()
    for i in range(0, 24, 2):
        b.embed_images(imgs[i:i + 2])
    sync()
    img_rate = 24 / (time.perf_counter() - t0)
    print(f"  image indexing throughput (embedding only, batch 2): {img_rate:.2f} images/s")

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)
    fp = subprocess.run(["footprint", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    phys = next((ln.split(":")[1].strip() for ln in fp.splitlines() if "phys_footprint_peak" in ln), "n/a")
    hf = Path.home() / ".cache/huggingface/hub/models--google--embeddinggemma-2" / "snapshots"
    resources = {
        "process peak RSS (CPU side)": f"{peak_rss:.0f} MB",
        "process physical footprint peak (incl. Metal)": phys,
        "MPS driver allocated": f"{torch.mps.driver_allocated_memory() / 2**20:.0f} MB" if device == "mps" else "n/a",
        "image indexing throughput (embedding only)": f"{img_rate:.2f} images/s",
        "model cache on disk": f"{du_mb(hf)} MB",
        "SQLite database": f"{du_mb(s.db_path)} MB",
        "thumbnails": f"{du_mb(s.thumbs_dir)} MB",
        "indexed vectors (all media)": str(len(vm_all.asset_ids)),
    }
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "machine": f"{platform.machine()} {platform.platform()} (Apple M1, 8 GB)", "device": device,
           "precision": precision, "profile_key": prof.key, "latency": lat, "resources": resources,
           "notes": ("Warm timings exclude decoding and HTTP. Ranking uses the in-memory matrix (cache hit). End-to-end HTTP latency per "
                     "query mode is in the evaluation run file (client_latency_ms). Library indexing is precomputed; query-time cost is "
                     "one query embedding plus ranking.")}
    (HERE / "perf-latest.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(resources, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
