"""Real-model smoke check for google/embeddinggemma-2 (text, image, native image+text).

This is a smoke check of real inference, NOT an accuracy evaluation.

Usage:
    uv run python scripts/model_smoke.py --images data/samples/smoke [--device mps|cpu] [--dtype bfloat16|float32] [--offline]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float32"])
    ap.add_argument("--revision", default=None)
    ap.add_argument("--offline", action="store_true", help="local_files_only loading; no network")
    ap.add_argument("--no-audio-encoder", action="store_true", help="skip loading the audio encoder")
    ap.add_argument("--mixed-ref", default="zebra.jpg", help="filename of the reference image for the mixed query")
    ap.add_argument("--mixed-text", default="a musical instrument")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import numpy as np
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device == "mps" else "float32"
    dtype = getattr(torch, dtype_name)
    assert dtype is not torch.float16, "float16 is prohibited by the model card"

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if len(paths) < 3:
        print("need at least 3 images", file=sys.stderr)
        return 2

    config_kwargs = {"audio_config": None} if args.no_audio_encoder else {}
    t0 = time.perf_counter()
    model = SentenceTransformer(
        MODEL_ID,
        revision=args.revision,
        device=device,
        model_kwargs={"dtype": dtype},
        config_kwargs=config_kwargs,
        local_files_only=args.offline,
    )
    load_s = time.perf_counter() - t0
    report: dict = {
        "model": MODEL_ID,
        "revision": args.revision,
        "device": str(model.device),
        "dtype": dtype_name,
        "audio_encoder_loaded": not args.no_audio_encoder,
        "offline": args.offline,
        "load_seconds": round(load_s, 2),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }

    def check(name: str, emb) -> np.ndarray:
        arr = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(arr.reshape(-1, arr.shape[-1]), axis=1)
        ok = bool(np.isfinite(arr).all()) and arr.shape[-1] == 768 and bool(np.allclose(norms, 1.0, atol=1e-2))
        report[f"{name}_shape"] = list(arr.shape)
        report[f"{name}_norms"] = [round(float(n), 4) for n in norms]
        report[f"{name}_ok"] = ok
        print(f"[{name}] shape={arr.shape} finite={np.isfinite(arr).all()} norms={norms.round(4).tolist()} ok={ok}")
        return arr

    # Text (asymmetric search: query prompt for queries)
    queries = ["a zebra with black and white stripes", "a musical instrument", "a bright yellow flower"]
    t0 = time.perf_counter()
    q = check("text", model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True))
    report["text_seconds"] = round(time.perf_counter() - t0, 3)

    # Images (no prefix)
    t0 = time.perf_counter()
    imgs = [Image.open(p).convert("RGB") for p in paths]
    d = check("image", model.encode([{"image": im} for im in imgs], normalize_embeddings=True, batch_size=1))
    report["image_seconds"] = round(time.perf_counter() - t0, 3)
    names = [p.name for p in paths]

    sims = q @ d.T
    report["text_to_image"] = {}
    for qi, qtext in enumerate(queries):
        order = np.argsort(-sims[qi])
        ranked = [(names[j], round(float(sims[qi, j]), 4)) for j in order]
        report["text_to_image"][qtext] = ranked
        print(f"  '{qtext}': {ranked[:3]}")

    # Native mixed image + text in ONE input (verified dict interface)
    ref_idx = names.index(args.mixed_ref) if args.mixed_ref in names else 0
    ref = imgs[ref_idx]
    mixed_text = args.mixed_text
    t0 = time.perf_counter()
    m = check("mixed", model.encode([{"text": "task: search result | query: " + mixed_text, "image": ref}],
                                    normalize_embeddings=True))
    report["mixed_seconds"] = round(time.perf_counter() - t0, 3)
    msims = (m @ d.T)[0]
    order = np.argsort(-msims)
    report["mixed_query"] = {"reference": names[ref_idx], "text": mixed_text,
                             "ranked": [(names[j], round(float(msims[j]), 4)) for j in order]}
    print(f"  mixed ({names[ref_idx]} + '{mixed_text}'): {report['mixed_query']['ranked'][:3]}")

    report["all_ok"] = all(report[k] for k in ("text_ok", "image_ok", "mixed_ok"))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (dict, list))}, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
