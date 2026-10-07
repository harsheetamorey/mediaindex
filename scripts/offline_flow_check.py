"""Exercise every implemented media flow against a running server (intended to run while the server is
network-sandboxed; see docs/offline-and-robustness.md). Prints one line per step; exits non-zero on failure.

Usage: uv run python scripts/offline_flow_check.py --folder /abs/mixed/folder --dest /abs/export/dir
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search_check import BASE, post_form, post_json  # noqa: E402


def get(path):
    return json.load(urllib.request.urlopen(BASE + path))


def wait(job):
    while job["status"] in ("queued", "running"):
        time.sleep(1)
        job = get(f"/api/jobs/{job['id']}")
    return job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    ok = True

    def step(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    t0 = time.time()
    lib = post_json("/api/libraries", {"path": a.folder})
    j = wait(post_json(f"/api/libraries/{lib['id']}/import", {}))
    r = j["result"] or {}
    step("import images+audio+video", j["status"] == "done" and r.get("failed_count") == 0,
         f"({r.get('new')} new, {r.get('audio_files')} audio, {r.get('video_files')} video, "
         f"{r.get('video_windows')} video windows, {time.time() - t0:.0f}s)")
    assets = {x["rel_path"]: x for x in get(f"/api/libraries/{lib['id']}/assets")}
    st = get("/api/model/status")
    step("model loaded from local cache", st["loaded"] and st["load_error"] is None, f"(device {st['device']}, load {st['load_seconds']:.1f}s)")
    r = post_json("/api/search/text", {"text": "a zebra", "limit": 1})
    step("text → image", r["results"][0]["asset"]["rel_path"] == "zebra.jpg", r["results"][0]["asset"]["rel_path"])
    r = post_json("/api/search/text", {"text": "a dog barking", "media_types": ["audio"], "limit": 1})
    step("text → sound", r["results"][0]["asset"]["rel_path"] == "fsd-144028.flac", r["results"][0]["asset"]["rel_path"])
    r = post_json("/api/search/text", {"text": "a dog", "media_types": ["video"], "limit": 1})
    step("text → video moment", r["results"][0]["asset"]["media_type"] == "video",
         f"{r['results'][0]['asset']['rel_path']} [{r['results'][0]['start_s']}-{r['results'][0]['end_s']}]")
    r = post_form("/api/search/image", {"limit": 3}, Path(a.folder) / "piano.jpg")
    step("upload image reference", r["results"] and r["query"]["reference"] == "upload", r["results"][0]["asset"]["rel_path"])
    r = post_form("/api/search/image-text", {"asset_id": assets["zebra.jpg"]["id"], "text": "a musical instrument", "limit": 2}, None)
    step("image + text refinement", len(r["results"]) == 2, [x["asset"]["rel_path"] for x in r["results"]])
    r = post_form("/api/search/audio", {"limit": 2}, Path(a.folder) / "fsd-150978.flac")
    step("upload sound reference", r["results"] and r["query"]["reference"] == "upload", r["results"][0]["asset"]["rel_path"])
    r = post_form("/api/search/image", {"asset_id": assets["guitar.jpg"]["id"], "target": "audio", "limit": 1}, None)
    step("image → sound (experimental)", r["results"] and r["query"]["experimental"], r["results"][0]["asset"]["rel_path"])
    v = assets["clip-dog.mp4"]
    r = post_json("/api/search/video-window", {"asset_id": v["id"], "start_s": 4, "target": "audio", "limit": 1})
    step("video moment → sound", bool(r["results"]), r["results"][0]["asset"]["rel_path"])
    p = post_json(f"/api/assets/{v['id']}/clip/preview", {"start_s": 2, "end_s": 6, "destination": a.dest})
    j = wait(post_json(f"/api/assets/{v['id']}/clip", {"plan_id": p["plan_id"], "confirm": True}))
    step("clip export", j["status"] == "done", f"{j['result'] and j['result'].get('clip')} {j['result'] and j['result'].get('output_duration_s')}s")
    s = post_json("/api/selections", {"name": "offline"})
    post_json(f"/api/selections/{s['id']}/items", {"asset_id": assets["zebra.jpg"]["id"]})
    plan = post_json(f"/api/selections/{s['id']}/export/preview", {"destination": a.dest})
    j = wait(post_json(f"/api/selections/{s['id']}/export", {"plan_id": plan["plan_id"], "confirm": True}))
    step("selection copy export + manifest", j["status"] == "done" and j["result"]["copied"] == 1, j["result"]["manifest"])
    html = urllib.request.urlopen(BASE + "/").read().decode()
    step("UI served locally", "<div id=\"root\">" in html)
    print("ALL PASS" if ok else "SOME STEPS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
