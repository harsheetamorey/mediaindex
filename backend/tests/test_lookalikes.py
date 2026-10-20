"""Look-alike groups (mocked embeddings for the API; the threshold was set with the real model, see lookalikes.py)."""

import numpy as np
from fastapi.testclient import TestClient

from conftest import make_image, tree_digest
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.lookalikes import group_lookalikes
from mediaindex.model.backend import FakeBackend
from test_search import wait_job


def unit(*v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_groups_are_stars_not_chains():
    # a~b (0.99), b~c (0.95) but a and c are further apart: c must not join through b; d stands alone
    a, b = unit(1, 0, 0), unit(0.99, 0.141, 0)
    c = unit(0.89, 0.456, 0)
    d = unit(0, 0, 1)
    M = np.stack([a, b, c, d])
    assert float(a @ c) < 0.92 <= float(b @ c)
    groups = group_lookalikes(["a", "b", "c", "d"], M, 0.92)
    assert groups == [["b", "a", "c"]] or groups == [["b", "c", "a"]]  # b leads (2 neighbours): one star
    assert group_lookalikes(["a", "c", "d"], np.stack([a, c, d]), 0.92) == []
    assert group_lookalikes(["a"], M[:1], 0.5) == []


def test_api_groups_identical_copies_and_never_touches_files(tmp_path):
    root = tmp_path / "Shoot"
    make_image(root / "IMG_1.jpg", (220, 20, 20))
    make_image(root / "copy/IMG_1.jpg", (220, 20, 20))  # the same file twice
    make_image(root / "blue.jpg", (20, 20, 220))
    before = tree_digest(root)
    app = create_app(Settings(data_dir=tmp_path / "data", precision="float32", watch_interval=0), backend_factory=FakeBackend)
    with TestClient(app) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        r = c.get("/api/lookalikes").json()
        assert r["mode"] == "lookalikes" and r["query"]["groups"] >= 1
        first = r["lookalike_groups"][0]
        members = [x["asset"]["rel_path"] for x in r["results"] if x["group"] == first["group"]]
        assert sorted(members) == ["IMG_1.jpg", "copy/IMG_1.jpg"] and first["identical"] is True
        assert all(x["similarity"] is None for x in r["results"])
        assert c.get("/api/lookalikes?level=bogus").status_code == 422
    assert tree_digest(root) == before
