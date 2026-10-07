import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_image
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.db import Database
from mediaindex.model.backend import FakeBackend
from mediaindex.model.profiles import IndexProfile
from mediaindex.search import MatrixCache, rank
from mediaindex.store import Store, VectorMatrix


def vm_from(vectors, ids):
    m = np.asarray(vectors, np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return VectorMatrix([f"e{i}" for i in ids], list(ids), m, [None] * len(ids), [None] * len(ids),
                        ["image"] * len(ids), 0)


def test_rank_known_vectors_and_exclusion():
    vm = vm_from([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [-1, 0, 0]], ["a", "b", "c", "d"])
    hits = rank(np.array([1, 0, 0], np.float32), vm, 3)
    assert [h.asset_id for h in hits] == ["a", "b", "c"]
    assert hits[0].similarity == pytest.approx(1.0)
    hits = rank(np.array([1, 0, 0], np.float32), vm, 10, exclude_assets={"a"})
    assert [h.asset_id for h in hits] == ["b", "c", "d"]
    assert rank(np.array([1, 0, 0], np.float32), vm_from(np.zeros((0, 3)) + 1, [])[0:0] if False else
                VectorMatrix([], [], np.zeros((0, 0), np.float32), [], [], [], 0), 5) == []
    with pytest.raises(ValueError):
        rank(np.ones(4, np.float32), vm, 1)


def test_cache_invalidates_on_writes_and_deletes(tmp_path):
    st = Store(Database(tmp_path / "d.sqlite3"))
    p = IndexProfile()
    st.register_profile(p)
    lib = st.create_library("l", tmp_path)
    cache = MatrixCache(st)
    aid, _ = st.upsert_asset(lib["id"], "a.jpg", "image", size=1, mtime_ns=1, content_hash="h")
    assert cache.get(p.key, None, ["image"]).asset_ids == []
    v = np.zeros(768, np.float32)
    v[0] = 1
    st.write_embeddings(aid, p.key, "image", v)
    assert cache.get(p.key, None, ["image"]).asset_ids == [aid]
    st.remove_asset(aid)  # stale ID must disappear
    assert cache.get(p.key, None, ["image"]).asset_ids == []


def wait_job(c, jid):
    for _ in range(500):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.01)
    raise AssertionError("job timeout")


def test_text_search_api_states_and_filters(tmp_path):
    a_root, b_root = tmp_path / "A", tmp_path / "B"
    for i in range(4):
        make_image(a_root / f"a{i}.jpg", (i * 50, 10, 10))
        make_image(b_root / f"b{i}.jpg", (10, i * 50, 10))
    (tmp_path / "E").mkdir()
    app = create_app(Settings(data_dir=tmp_path / "data", precision="float32"), backend_factory=FakeBackend)
    with TestClient(app) as c:
        la = c.post("/api/libraries", json={"path": str(a_root)}).json()
        lb = c.post("/api/libraries", json={"path": str(b_root)}).json()
        le = c.post("/api/libraries", json={"path": str(tmp_path / "E")}).json()
        r = c.post("/api/search/text", json={"text": "red", "library_ids": [la["id"]]}).json()
        assert r["index_state"]["state"] == "empty" and r["results"] == []
        for lib in (la, lb):
            assert wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])["status"] == "done"
        r = c.post("/api/search/text", json={"text": "red", "library_ids": [la["id"]], "limit": 3}).json()
        assert len(r["results"]) == 3 and r["index_state"]["state"] == "ready"
        assert all(x["asset"]["library_id"] == la["id"] for x in r["results"])
        sims = [x["similarity"] for x in r["results"]]
        assert sims == sorted(sims, reverse=True)
        assert set(r["timing_ms"]) == {"query_embedding", "ranking"}
        assert "not a probability" in r["note"]
        r = c.post("/api/search/text", json={"text": "red", "limit": 50}).json()
        assert len(r["results"]) == 8
        r = c.post("/api/search/text", json={"text": "x", "library_ids": [le["id"]]}).json()
        assert r["index_state"]["state"] == "empty" and r["results"] == []
        assert c.post("/api/search/text", json={"text": "x", "library_ids": ["nope"]}).status_code == 404
        assert c.post("/api/search/text", json={"text": ""}).status_code == 422
        assert c.post("/api/search/text", json={"text": "x", "limit": 1000}).status_code == 422


def test_incompatible_profile_rejected(tmp_path):
    root = tmp_path / "A"
    make_image(root / "a.jpg")
    data = tmp_path / "data"
    app = create_app(Settings(data_dir=data, precision="float32"), backend_factory=FakeBackend)
    with TestClient(app) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
    app2 = create_app(Settings(data_dir=data, precision="bfloat16"), backend_factory=FakeBackend)
    with TestClient(app2) as c:
        r = c.post("/api/search/text", json={"text": "x", "library_ids": [lib["id"]]})
        assert r.status_code == 409 and "different model profile" in r.json()["detail"]
        # re-import reindexes unchanged files under the new profile
        assert wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])["result"]["embedded"] == 1
        r = c.post("/api/search/text", json={"text": "x", "library_ids": [lib["id"]]})
        assert r.status_code == 200 and len(r.json()["results"]) == 1
