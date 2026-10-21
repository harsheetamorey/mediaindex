"""REAL-MODEL check of photo indexing speed through the app, and of the upgrade from 1.1.0 on a PC.

1. An app set up like 1.1.0 on a PC (280 vision tokens per photo) imports the photos.
2. The app with today's settings starts on the same data: it re-indexes that library by itself, then a text search
   finds the red photo first.
Phone-size generated photos (solid colours to search for, plus noise). Prints seconds per photo for each step.

Usage: MEDIAINDEX_ALLOW_DOWNLOAD=1 uv run python scripts/index_speed_check.py [--photos 24] [--work DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from fastapi.testclient import TestClient  # noqa: E402

from mediaindex.app import create_app  # noqa: E402
from mediaindex.config import Settings  # noqa: E402

COLOURS = {"red": (220, 20, 20), "green": (20, 180, 40), "blue": (20, 40, 220), "yellow": (240, 220, 20)}


def photos(root: Path, n: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, rgb in COLOURS.items():
        Image.new("RGB", (4032, 3024), rgb).save(root / f"{name}.jpg", quality=90)
    for i in range(n - len(COLOURS)):
        noise = rng.integers(0, 256, (378, 504, 3), np.uint8)
        Image.fromarray(noise).resize((4032, 3024)).save(root / f"noise{i:02d}.jpg", quality=90)


def wait(c: TestClient, job_id: str) -> dict:
    while (j := c.get(f"/api/jobs/{job_id}").json())["status"] in ("queued", "running"):
        time.sleep(1)
    assert j["status"] == "done", j
    return j


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=int, default=24)
    ap.add_argument("--work", type=Path)
    a = ap.parse_args()
    work = a.work or Path(tempfile.mkdtemp())
    root, data = work / "photos", work / "data"
    photos(root, a.photos)
    out = {}

    old = create_app(Settings(data_dir=data, image_tokens=280, watch_interval=0))
    with TestClient(old) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        t0 = time.perf_counter()
        j = wait(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        out["1.1.0 (280 tokens)"] = {"s_per_photo": round((time.perf_counter() - t0) / a.photos, 2),
                                     "embedded": j["result"]["embedded"]}

    new = create_app(Settings(data_dir=data, watch_interval=0))
    with TestClient(new) as c:
        t0 = time.perf_counter()
        job = next(x for x in c.get("/api/jobs").json() if x["kind"] == "import")  # queued at startup, by itself
        j = wait(c, job["id"])
        st = c.get("/api/model/status").json()
        import torch

        out["now"] = {"s_per_photo": round((time.perf_counter() - t0) / a.photos, 2), "embedded": j["result"]["embedded"],
                      "device": st["device"], "image_tokens": st["profile"]["image_max_soft_tokens"],
                      "torch_threads": torch.get_num_threads()}
        top = {}
        for name in COLOURS:
            r = c.post("/api/search/text", json={"text": f"a plain {name} picture", "limit": 3}).json()["results"]
            top[name] = r[0]["asset"]["rel_path"]
        out["top_result"] = top
    print(json.dumps(out, indent=1))
    ok = (out["now"]["embedded"] == a.photos and all(top[n] == f"{n}.jpg" for n in COLOURS))
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
