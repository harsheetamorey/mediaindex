"""REAL-MODEL audio smoke check + encoder-configuration compatibility check for image vectors.

1. Loads the full model (text+vision+audio), embeds >=3 real audio clips decoded by FFmpeg to mono 16 kHz,
   and ranks them against text queries.
2. Embeds the same images with (a) the full model and (b) the text+image-only model and reports the
   maximum difference, to decide whether image vectors can be reused across encoder configurations.

Usage: uv run python scripts/audio_smoke.py --audio data/demo/fsd50k-cc0 --images data/samples/smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from mediaindex.media.audio import decode_audio, probe_audio  # noqa: E402
from mediaindex.model.profiles import MODEL_ID, MODEL_REVISION, QUERY_PROMPT  # noqa: E402


def load(device, dtype, audio: bool):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=device, model_kwargs={"dtype": dtype},
                               config_kwargs={} if audio else {"audio_config": None}, local_files_only=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    import torch
    from PIL import Image

    dtype = getattr(torch, a.dtype)
    prov = json.loads((a.audio / "mediaindex-provenance.json").read_text())["items"]
    files = sorted(a.audio.glob("*.flac"))[: a.n]
    report = {"device": a.device, "dtype": a.dtype}

    t0 = time.perf_counter()
    m = load(a.device, dtype, audio=True)
    report["load_full_s"] = round(time.perf_counter() - t0, 2)
    clips = []
    for f in files:
        info = probe_audio(f)
        x = decode_audio(f)
        clips.append((f.name, x, info))
    t0 = time.perf_counter()
    A = m.encode([{"audio": {"array": x, "sampling_rate": 16000}} for _, x, _ in clips], batch_size=2,
                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    report["audio_embed_s"] = round(time.perf_counter() - t0, 2)
    norms = np.linalg.norm(A, axis=1)
    report["audio_shape"] = list(A.shape)
    report["audio_finite"] = bool(np.isfinite(A).all())
    report["audio_norms"] = [round(float(v), 4) for v in norms]
    queries = ["a dog barking", "glass shattering", "a snare drum", "footsteps", "insects chirping at night"]
    Q = m.encode([QUERY_PROMPT + q for q in queries], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    S = Q @ (A / norms[:, None]).T
    report["text_to_audio"] = {}
    for qi, q in enumerate(queries):
        order = np.argsort(-S[qi])[:3]
        ranked = [(clips[j][0], prov[clips[j][0]]["tags"][:40], round(float(S[qi, j]), 4)) for j in order]
        report["text_to_audio"][q] = ranked
        print(f"  {q!r}: {ranked}")
    imgs = [Image.open(p).convert("RGB") for p in sorted(a.images.glob("*.jpg"))]
    I_full = m.encode([{"image": im} for im in imgs], batch_size=2, normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float32)
    T_full = Q
    del m
    gc.collect()
    if a.device == "mps":
        torch.mps.empty_cache()

    m2 = load(a.device, dtype, audio=False)
    I_img = m2.encode([{"image": im} for im in imgs], batch_size=2, normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float32)
    T_img = m2.encode([QUERY_PROMPT + q for q in queries], normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float32)
    report["image_vec_max_abs_diff_full_vs_textimage"] = float(np.abs(I_full - I_img).max())
    report["image_vec_min_cos_full_vs_textimage"] = float((I_full * I_img).sum(1).min())
    report["text_vec_max_abs_diff_full_vs_textimage"] = float(np.abs(T_full - T_img).max())
    report["ok"] = report["audio_finite"] and A.shape[1] == 768 and bool(np.allclose(norms, 1, atol=1e-2))
    print(json.dumps({k: v for k, v in report.items() if k != "text_to_audio"}, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
