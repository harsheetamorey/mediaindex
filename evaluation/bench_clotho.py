"""OPTIONAL Phase 18 benchmark: text→audio retrieval on the official Clotho v2.1 *evaluation* split.

Clotho captions are licensed for experimental, non-commercial use only (Tampere University licence); audio
clips carry per-file Freesound licences. Data stays local under data/bench/clotho and is never bundled.

Configurations (each run in its own process; vectors from different models are never compared):
  gemma-best   EmbeddingGemma 2 (pinned rev), MPS bf16, whole clip in ONE native pass (16 kHz mono), captions with
               the documented SearchQuery prompt.
  gemma-app    as deployed in MediaIndex: 10 s windows / 5 s stride, clip score = max over windows.
  clap-best    LAION CLAP laion/clap-htsat-fused (pinned rev), MPS fp32, 48 kHz, documented "fusion" truncation
               (numpy seed 0 for its random chunk choice).
  gemma-ctrl / clap-ctrl  controlled runtime: MPS fp32 for both, identical first-10-s excerpt of every clip.

Candidate pool: all 1,045 evaluation clips; queries: all 5,225 captions (5 per clip). Each caption has exactly
one relevant clip. Metrics per caption are averaged per clip (the clip is the statistical unit), then over
clips, with 95% bootstrap CIs over clips.

Usage: uv run --extra bench python evaluation/bench_clotho.py --config gemma-best [--limit N]
       uv run --extra bench python evaluation/bench_clotho.py --summarize
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data" / "bench" / "clotho"
OUT = HERE / "bench"
CLAP_ID, CLAP_REV = "laion/clap-htsat-fused", "365dea6ef167def6676140ed93bbc43f84dabb28"
sys.path.insert(0, str(REPO / "backend"))


def decode(path: Path, sr: int, seconds: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-protocol_whitelist", "file,pipe", "-v", "error", "-i", str(path)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype="<f4").astype(np.float32)


def load_split(limit: int | None):
    rows = list(csv.DictReader(open(DATA / "clotho_captions_evaluation.csv", newline="", encoding="utf-8")))
    rows.sort(key=lambda r: r["file_name"])
    if limit:
        rows = rows[:limit]
    audio_dir = next(p for p in DATA.rglob("*") if p.is_dir() and any(p.glob("*.wav")))
    files = [audio_dir / r["file_name"] for r in rows]
    caps = [[r[f"caption_{k}"] for k in range(1, 6)] for r in rows]
    return files, caps


def l2(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def embed_gemma(files, caps, mode: str, dtype: str):
    import torch

    from mediaindex.media.audio import plan_windows
    from mediaindex.model.backend import GemmaBackend
    from mediaindex.model.profiles import IndexProfile

    b = GemmaBackend(IndexProfile(precision=dtype), device="mps", offline=True)
    texts = [c for cs in caps for c in cs]
    T = np.vstack([b.embed_query_texts(texts[i:i + 64]) for i in range(0, len(texts), 64)])
    A, owner = [], []
    for ci, f in enumerate(files):
        if mode == "ctrl":
            x = decode(f, 16000, 10.0)
            A.append(b.embed_audio([x])[0]); owner.append(ci)
        elif mode == "best":
            A.append(b.embed_audio([decode(f, 16000)])[0]); owner.append(ci)
        else:  # app windows
            x = decode(f, 16000)
            for s0, e0 in plan_windows(len(x) / 16000, 10.0, 5.0):
                A.append(b.embed_audio([x[int(s0 * 16000):int(e0 * 16000)]])[0]); owner.append(ci)
        if ci % 100 == 0:
            print(f"  audio {ci}/{len(files)}", flush=True)
    torch.mps.synchronize()
    return T, np.vstack(A), np.array(owner)


def embed_clap(files, caps, mode: str):
    import torch
    from transformers import ClapModel, ClapProcessor

    np.random.seed(0)
    dev = "mps"
    model = ClapModel.from_pretrained(CLAP_ID, revision=CLAP_REV).to(dev).eval()
    proc = ClapProcessor.from_pretrained(CLAP_ID, revision=CLAP_REV)
    texts = [c for cs in caps for c in cs]
    T = []
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            t = proc(text=texts[i:i + 64], return_tensors="pt", padding=True).to(dev)
            T.append(model.get_text_features(**t).pooler_output.float().cpu().numpy())  # projected + normalized
        A = []
        for ci, f in enumerate(files):
            x = decode(f, 48000, 10.0 if mode == "ctrl" else None)
            a = proc(audio=[x], sampling_rate=48000, return_tensors="pt").to(dev)
            A.append(model.get_audio_features(**a).pooler_output.float().cpu().numpy()[0])
            if ci % 100 == 0:
                print(f"  audio {ci}/{len(files)}", flush=True)
    return l2(np.vstack(T)), l2(np.vstack(A)), np.arange(len(files))


def metrics(T, A, owner, n_clips, seed=0):
    """Per caption: rank of its clip (clip score = max over that clip's vectors)."""
    S = l2(T) @ l2(A).T
    clip_scores = np.full((S.shape[0], n_clips), -np.inf, np.float32)
    for j in range(n_clips):
        clip_scores[:, j] = S[:, owner == j].max(axis=1)
    truth = np.repeat(np.arange(n_clips), 5)
    true_score = clip_scores[np.arange(len(truth)), truth]
    rank = (clip_scores > true_score[:, None]).sum(axis=1) + 1  # 1 = best
    per_caption = {"R@1": (rank <= 1).astype(float), "R@5": (rank <= 5).astype(float),
                   "R@10": (rank <= 10).astype(float), "mAP@10": np.where(rank <= 10, 1.0 / rank, 0.0)}
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in per_caption.items():
        per_clip = v.reshape(n_clips, 5).mean(axis=1)
        boots = [per_clip[rng.integers(0, n_clips, n_clips)].mean() for _ in range(1000)]
        out[k] = {"mean": round(float(per_clip.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                                                    round(float(np.percentile(boots, 97.5)), 4)]}
    out["median_rank"] = float(np.median(rank))
    return out


def run(config: str, limit: int | None):
    files, caps = load_split(limit)
    t0 = time.perf_counter()
    if config.startswith("gemma"):
        mode = config.split("-")[1]
        T, A, owner = embed_gemma(files, caps, "app" if mode == "app" else mode, "float32" if mode == "ctrl" else "bfloat16")
        model = {"id": "google/embeddinggemma-2", "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b"}
    else:
        mode = config.split("-")[1]
        T, A, owner = embed_clap(files, caps, mode)
        model = {"id": CLAP_ID, "revision": CLAP_REV}
    secs = time.perf_counter() - t0
    import torch

    res = {
        "config": config, "model": model, "clips": len(files), "captions": len(files) * 5, "subset": bool(limit),
        "device": "mps", "precision": "float32" if (mode == "ctrl" or config.startswith("clap")) else "bfloat16",
        "audio_input": {"gemma-best": "whole clip, 16 kHz mono, one pass", "gemma-app": "10 s windows / 5 s stride, max",
                        "clap-best": "whole clip, 48 kHz mono, fusion truncation", "gemma-ctrl": "first 10 s, 16 kHz",
                        "clap-ctrl": "first 10 s, 48 kHz"}[config],
        "audio_vectors": int(len(A)), "embed_seconds_total": round(secs, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
        "mps_driver_mb": round(torch.mps.driver_allocated_memory() / 2**20, 1),
        "metrics": metrics(T, A, owner, len(files)),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"clotho_{config}{'_subset' if limit else ''}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "model"}, indent=1))


def summarize():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("clotho_*.json")) if "subset" not in p.name]
    lines = ["| Config | Model | Precision | Audio input | R@1 [95% CI] | R@5 | R@10 | mAP@10 | Embed time | Peak MPS |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["metrics"]
        f = lambda k: f"{m[k]['mean']:.3f} [{m[k]['ci95'][0]:.3f}, {m[k]['ci95'][1]:.3f}]"  # noqa: E731
        lines.append(f"| {r['config']} | {r['model']['id']} | {r['precision']} | {r['audio_input']} | {f('R@1')} | {f('R@5')} | "
                     f"{f('R@10')} | {f('mAP@10')} | {r['embed_seconds_total']:.0f} s | {r['mps_driver_mb']:.0f} MB |")
    print("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["gemma-best", "gemma-app", "clap-best", "gemma-ctrl", "clap-ctrl"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        run(a.config, a.limit)
