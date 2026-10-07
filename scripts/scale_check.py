"""Measure exact-search scaling with synthetic unit vectors in a temporary database (no model needed).

Reports DB size, cold matrix load, warm ranking latency (median/p95), and matrix memory for N vectors,
to decide whether ANN/quantization is needed. Usage: uv run python scripts/scale_check.py --n 20000 100000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from mediaindex.db import Database, now  # noqa: E402
from mediaindex.model.profiles import IndexProfile  # noqa: E402
from mediaindex.search import MatrixCache, rank  # noqa: E402
from mediaindex.store import Store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[20000, 100000])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for n in a.n:
        with tempfile.TemporaryDirectory() as d:
            db = Database(Path(d) / "s.sqlite3")
            st = Store(db)
            prof = IndexProfile()
            st.register_profile(prof)
            lib = st.create_library("scale", Path(d))
            t0 = time.perf_counter()
            with db.tx() as c:
                t = now()
                for i in range(n):
                    aid = f"a{i:07d}"
                    v = rng.standard_normal(768).astype(np.float32)
                    v /= np.linalg.norm(v)
                    c.execute("""INSERT INTO assets(id, library_id, rel_path, media_type, size, mtime_ns, content_hash,
                                 status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                              (aid, lib["id"], f"{i}.jpg", "image", 1, 1, f"h{i}", "indexed", t, t))
                    c.execute("""INSERT INTO embeddings(id, asset_id, profile_key, modality, segment_index, dim, vector,
                                 created_at) VALUES (?,?,?,?,?,?,?,?)""", (f"e{i:07d}", aid, prof.key, "image", 0, 768,
                                                                          v.tobytes(), t))
            insert_s = time.perf_counter() - t0
            size_mb = (Path(d) / "s.sqlite3").stat().st_size / 2**20
            cache = MatrixCache(st)
            t0 = time.perf_counter()
            vm = cache.get(prof.key, None, ["image"])
            load_s = time.perf_counter() - t0
            q = rng.standard_normal(768).astype(np.float32)
            q /= np.linalg.norm(q)
            lat = []
            for _ in range(50):
                t0 = time.perf_counter()
                cache.get(prof.key, None, ["image"])  # cache hit
                rank(q, vm, 48)
                lat.append((time.perf_counter() - t0) * 1000)
            lat.sort()
            print(f"N={n:>7}: insert {insert_s:5.1f}s, DB {size_mb:6.1f} MB, cold matrix load {load_s:5.2f}s, "
                  f"matrix {vm.matrix.nbytes / 2**20:6.1f} MB, warm rank median {statistics.median(lat):6.2f} ms, "
                  f"p95 {lat[int(0.95 * len(lat)) - 1]:6.2f} ms")
            db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
