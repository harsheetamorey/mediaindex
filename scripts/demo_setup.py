"""Add the three demo-pack folders to a running MediaIndex and wait until they are indexed.

Idempotent: existing libraries are re-scanned (unchanged files are skipped). Talks only to the local server.
Requires the demo packs (scripts/download_demo.py, download_demo_audio.py, make_demo_video.py).

Usage: uv run python scripts/demo_setup.py [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FOLDERS = [REPO / "data" / "demo" / d for d in ("stockimages-cc0", "fsd50k-cc0", "videos")]


def call(base: str, method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    missing = [str(f) for f in FOLDERS if not f.is_dir()]
    if missing:
        print("missing demo folders (run the demo download scripts first):", *missing, sep="\n  ")
        return 1
    libs = {Path(l["root_path"]).resolve(): l for l in call(base, "GET", "/api/libraries")}
    jobs = []
    for folder in FOLDERS:
        lib = libs.get(folder.resolve()) or call(base, "POST", "/api/libraries", {"path": str(folder)})
        jobs.append((folder.name, call(base, "POST", f"/api/libraries/{lib['id']}/import")["id"]))
    t0 = time.time()
    for name, jid in jobs:
        while (job := call(base, "GET", f"/api/jobs/{jid}"))["status"] in ("queued", "running"):
            print(f"\r{name}: {job.get('done', 0)}/{job.get('total') or '?'}  ({time.time() - t0:.0f} s)", end="", flush=True)
            time.sleep(3)
        r = job.get("result") or {}
        counts = ", ".join(f"{k} {r[k]}" for k in ("discovered", "new", "unchanged", "failed_count") if k in r)
        print(f"\r{name}: {job['status']} ({counts})  ({time.time() - t0:.0f} s)")
    return 0 if all(call(base, "GET", f"/api/jobs/{j}")["status"] == "done" for _, j in jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
