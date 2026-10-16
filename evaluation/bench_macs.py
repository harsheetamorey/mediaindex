"""Text→audio retrieval on MACS (Multi-Annotator Captioned Soundscapes): a fairer CLAP vs EmbeddingGemma comparison.

Why MACS: CLAP (laion/clap-htsat-fused) was trained on LAION-Audio-630K (mostly Freesound) plus AudioCaps and
Clotho, so Clotho, ESC-50, UrbanSound8K and FSD50K all overlap its training data. MACS audio is from TAU Urban
Acoustic Scenes 2019 (recorded by Tampere University in airports, parks and public squares; not Freesound), with
one-sentence descriptions written by several annotators per clip. Neither appears in CLAP's documented training
data. EmbeddingGemma 2's training data isn't documented in enough detail to rule overlap in or out.

Licences: MACS annotations and TAU 2019 audio are non-commercial research licences (see their LICENSE files). Data
stays local under data/bench/macs and is never bundled.

Candidate pool: all 3,930 MACS clips (10 s each); queries: every caption. Each caption has exactly one relevant clip.
Metrics per caption are averaged per clip (the clip is the statistical unit), then over clips, with 95% bootstrap CIs.
Same models, revisions and settings as evaluation/bench_clotho.py ("best" configs: whole 10 s clip).

Usage: uv run --extra bench python evaluation/bench_macs.py --download     # ~36 GB streamed, ~4 GB kept
       uv run --extra bench python evaluation/bench_macs.py --config gemma-best [--limit N]
       uv run --extra bench python evaluation/bench_macs.py --config clap-best [--limit N]
       uv run --extra bench python evaluation/bench_macs.py --summarize
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_clotho import CLAP_ID, CLAP_REV, embed_clap, embed_gemma, l2  # noqa: E402

REPO = HERE.parent
DATA = REPO / "data" / "bench" / "macs"
AUDIO = DATA / "audio"
OUT = HERE / "bench"
MACS_RECORD, TAU_RECORD = "5114771", "2589280"


def _fetch(url: str, dest: Path, size: int | None = None, md5: str | None = None, tries: int = 8) -> None:
    """Download with resume (HTTP Range) and retries; checks the size and MD5 that Zenodo lists."""
    import hashlib

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(tries):
        have = tmp.stat().st_size if tmp.exists() else 0
        if size is not None and have >= size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab" if have and r.status == 206 else "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
        except OSError as e:
            print(f"  {dest.name}: interrupted at {have / 1e9:.2f} GB ({type(e).__name__}), resuming", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        if size is None or tmp.stat().st_size >= size:
            break
    if size is not None and tmp.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: got {tmp.stat().st_size} bytes, expected {size}")
    if md5:
        h = hashlib.md5()
        with open(tmp, "rb") as f:
            while chunk := f.read(1 << 22):
                h.update(chunk)
        if h.hexdigest() != md5:
            tmp.unlink()
            raise RuntimeError(f"{dest.name}: checksum mismatch; deleted, run --download again")
    tmp.rename(dest)


def download() -> None:
    """MACS annotations, then each TAU 2019 audio zip in turn: keep only MACS clips (as 48 kHz mono FLAC), delete
    the zip. Resumable: clips already converted are skipped."""
    import yaml

    DATA.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(exist_ok=True)
    files = {f["key"]: f for f in json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{MACS_RECORD}"))["files"]}
    for name in ("MACS.yaml", "LICENSE.txt", "MACS_competence.csv"):
        if not (DATA / name).exists():
            _fetch(files[name]["links"]["self"], DATA / name)
    wanted = {Path(f["filename"]).name for f in yaml.safe_load(open(DATA / "MACS.yaml"))["files"]}
    print(f"MACS: {len(wanted)} clips", flush=True)
    tau = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{TAU_RECORD}"))["files"]
    zips = sorted((f for f in tau if ".audio." in f["key"]), key=lambda f: int(f["key"].split(".")[-2]))
    done_file = DATA / "zips_done.txt"  # zips already converted: never downloaded again
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    for z in zips:
        have = {p.stem for p in AUDIO.glob("*.flac")}
        if len(have) >= len(wanted):
            break
        if z["key"] in done:
            continue
        dest = DATA / z["key"]
        if dest.exists() and dest.stat().st_size != z["size"]:  # a truncated zip from an earlier run
            dest.rename(dest.with_suffix(dest.suffix + ".part"))
        if not dest.exists():
            print(f"{z['key']}: downloading {z['size'] / 1e9:.1f} GB", flush=True)
            md5 = z.get("checksum", "").removeprefix("md5:") or None
            _fetch(z["links"]["self"], dest, z["size"], md5)
        n = 0
        with zipfile.ZipFile(dest) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name
                if name in wanted and Path(name).stem not in have:
                    with zf.open(info) as src:
                        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "48000",
                                        str(AUDIO / f"{Path(name).stem}.flac")], input=src.read(), check=True)
                    n += 1
        dest.unlink()
        with open(done_file, "a") as f:
            f.write(z["key"] + "\n")
        print(f"{z['key']}: kept {n} clips ({len(have) + n}/{len(wanted)})", flush=True)
    print(f"done: {len(list(AUDIO.glob('*.flac')))} clips in {AUDIO}")


def load(limit: int | None):
    import yaml

    rows = sorted(yaml.safe_load(open(DATA / "MACS.yaml"))["files"], key=lambda f: f["filename"])
    files, caps = [], []
    for r in rows:
        p = AUDIO / f"{Path(r['filename']).stem}.flac"
        sents = [a["sentence"].strip() for a in r.get("annotations", []) if a.get("sentence", "").strip()]
        if p.exists() and sents:
            files.append(p)
            caps.append(sents)
    if limit:  # every n-th clip, so all three scenes stay in a subset
        step = max(1, len(files) // limit)
        files, caps = files[::step][:limit], caps[::step][:limit]
    return files, caps


def metrics(T, A, caps, seed=0):
    """Per caption: rank of its clip among all clips (one vector per clip)."""
    S = l2(T) @ l2(A).T
    truth = np.concatenate([[i] * len(c) for i, c in enumerate(caps)])
    true_score = S[np.arange(len(truth)), truth]
    rank = (S > true_score[:, None]).sum(axis=1) + 1
    per_caption = {"R@1": (rank <= 1).astype(float), "R@5": (rank <= 5).astype(float),
                   "R@10": (rank <= 10).astype(float), "mAP@10": np.where(rank <= 10, 1.0 / rank, 0.0)}
    n = len(caps)
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in per_caption.items():
        per_clip = np.array([v[truth == i].mean() for i in range(n)])
        boots = [per_clip[rng.integers(0, n, n)].mean() for _ in range(1000)]
        out[k] = {"mean": round(float(per_clip.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                                                    round(float(np.percentile(boots, 97.5)), 4)]}
    out["median_rank"] = float(np.median(rank))
    out["chance_R@10"] = round(10 / n, 4)
    return out


def run(config: str, limit: int | None):
    files, caps = load(limit)
    t0 = time.perf_counter()
    if config == "gemma-best":
        T, A, _ = embed_gemma(files, caps, "best", "bfloat16")
        model = {"id": "google/embeddinggemma-2", "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b"}
    else:
        T, A, _ = embed_clap(files, caps, "best")
        model = {"id": CLAP_ID, "revision": CLAP_REV}
    secs = time.perf_counter() - t0
    res = {"config": config, "dataset": "MACS (TAU Urban Acoustic Scenes 2019 audio)", "model": model,
           "clips": len(files), "captions": int(sum(len(c) for c in caps)), "subset": bool(limit), "device": "mps",
           "precision": "bfloat16" if config.startswith("gemma") else "float32",
           "audio_input": "whole 10 s clip, " + ("16 kHz mono" if config.startswith("gemma") else "48 kHz mono"),
           "embed_seconds_total": round(secs, 1),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
           "metrics": metrics(T, A, caps)}
    OUT.mkdir(exist_ok=True)
    (OUT / f"macs_{config}{'_subset' if limit else ''}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "model"}, indent=1))


def summarize():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("macs_*.json")) if "subset" not in p.name]
    lines = ["| Config | Model | Clips | Captions | R@1 [95% CI] | R@5 | R@10 [95% CI] | median rank | Embed time |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["metrics"]
        f = lambda k: f"{m[k]['mean']:.3f} [{m[k]['ci95'][0]:.3f}, {m[k]['ci95'][1]:.3f}]"  # noqa: E731
        lines.append(f"| {r['config']} | {r['model']['id']} | {r['clips']} | {r['captions']} | {f('R@1')} | "
                     f"{m['R@5']['mean']:.3f} | {f('R@10')} | {m['median_rank']:.0f} | {r['embed_seconds_total']:.0f} s |")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--config", choices=["gemma-best", "clap-best"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.download:
        download()
    elif a.summarize:
        summarize()
    else:
        run(a.config, a.limit)
