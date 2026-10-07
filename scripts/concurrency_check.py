"""Fire N concurrent searches (optionally while an import keeps the model busy); report status codes,
latencies, model load count and server RSS. Usage: uv run python scripts/concurrency_check.py [--n 20] [--busy-folder DIR]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8765"


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()
            return r.status, time.perf_counter() - t0, ""
    except urllib.error.HTTPError as e:
        return e.code, time.perf_counter() - t0, json.loads(e.read() or b"{}").get("detail", "")


def server_pid() -> str:
    out = subprocess.run(["lsof", "-nP", "-t", "-iTCP:8765", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
    return out.split()[0]


def rss_mb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", server_pid()], capture_output=True, text=True).stdout.strip()
    return int(out or 0) / 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--busy-folder")
    a = ap.parse_args()
    if a.busy_folder:
        lib = json.load(urllib.request.urlopen(urllib.request.Request(
            BASE + "/api/libraries", data=json.dumps({"path": a.busy_folder}).encode(), method="POST",
            headers={"content-type": "application/json"})))
        urllib.request.urlopen(urllib.request.Request(BASE + f"/api/libraries/{lib['id']}/import", data=b"", method="POST"))
        time.sleep(8)  # let the job start and hold the model
    queries = ["a zebra", "a dog barking", "a city at night", "a musical instrument", "a flower"]
    results = [None] * a.n

    def worker(i):
        mt = ["image"] if i % 3 == 0 else ["audio"] if i % 3 == 1 else ["image", "audio", "video"]
        results[i] = post("/api/search/text", {"text": queries[i % len(queries)], "media_types": mt, "limit": 5})

    peak = [rss_mb()]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak.append(rss_mb())
            time.sleep(0.5)

    st = threading.Thread(target=sampler)
    st.start()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(a.n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.perf_counter() - t0
    stop.set()
    st.join()
    codes = {}
    for c, _, d in results:
        codes.setdefault(c, []).append(d)
    lat = sorted(x[1] for x in results if x[0] == 200)
    status = json.load(urllib.request.urlopen(BASE + "/api/model/status"))
    print(f"{a.n} concurrent searches in {wall:.1f}s; status codes: { {k: len(v) for k, v in codes.items()} }")
    for k, v in codes.items():
        if k != 200:
            print(f"  {k}: {v[0]}")
    if lat:
        print(f"  latency ok-requests: median {statistics.median(lat):.2f}s, max {lat[-1]:.2f}s")
    print(f"  model load_count={status['load_count']} busy={status['busy']}; server RSS peak {max(peak):.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
