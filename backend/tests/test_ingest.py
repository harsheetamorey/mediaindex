import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

from conftest import make_image, tree_digest
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.db import Database
from mediaindex.ingest import run_image_import
from mediaindex.jobs import Job, JobCancelled, JobContext
from mediaindex.model.backend import FakeBackend
from mediaindex.paths import PathRejected, resolve_in_root, validate_root
from mediaindex.store import Store


class Ctx(JobContext):
    def __init__(self, cancel_after=None):
        super().__init__(Job(id="t", kind="import"), None)
        self.cancel_after = cancel_after

    def progress(self, done, total=None, message=None):
        super().progress(done, total, message)
        if self.cancel_after is not None and done >= self.cancel_after:
            self.job._cancel.set()


def setup(tmp_path, root):
    st = Store(Database(tmp_path / "db.sqlite3"))
    lib = st.create_library("m", root)
    return st, lib, tmp_path / "thumbs"


def test_import_handles_formats_corrupt_symlinks_and_preserves_originals(tmp_path, media_folder):
    before = tree_digest(media_folder)
    st, lib, thumbs = setup(tmp_path, media_folder)
    r = run_image_import(Ctx(), st, thumbs, lib["id"])
    assert r["discovered"] == 7  # red, green, blue, rotated, red-copy, corrupt, truncated
    assert r["new"] == 5
    assert {f for f, _ in r["failed"]} == {"corrupt.jpg", "truncated.jpg"}
    assert ("escape.jpg", "symlink points outside library root") in [tuple(x) for x in r["skipped"]]
    rels = {a["rel_path"]: a for a in st.list_assets(lib["id"])}
    assert "linkdir/secret.jpg" not in rels and ".hidden.jpg" not in rels
    rot = rels["rotated.jpg"]
    assert (rot["width"], rot["height"]) == (40, 80)  # EXIF orientation 6 applied
    assert rels["red.jpg"]["content_hash"] == rels["dupe/red-copy.jpg"]["content_hash"]
    assert rels["red.jpg"]["id"] != rels["dupe/red-copy.jpg"]["id"]
    assert len(list(thumbs.rglob("*.jpg"))) == 4  # duplicate content shares one thumbnail
    assert tree_digest(media_folder) == before


def test_reimport_is_idempotent_and_tracks_changes_and_missing(tmp_path, media_folder):
    st, lib, thumbs = setup(tmp_path, media_folder)
    run_image_import(Ctx(), st, thumbs, lib["id"])
    ids1 = {a["rel_path"]: a["id"] for a in st.list_assets(lib["id"])}
    r2 = run_image_import(Ctx(), st, thumbs, lib["id"])
    assert r2["new"] == 0 and r2["changed"] == 0 and r2["unchanged"] == 7
    assert {a["rel_path"]: a["id"] for a in st.list_assets(lib["id"])} == ids1

    time.sleep(0.01)
    make_image(media_folder / "red.jpg", (0, 0, 0))
    (media_folder / "sub/green.png").unlink()
    r3 = run_image_import(Ctx(), st, thumbs, lib["id"])
    assert r3["changed"] == 1 and r3["missing"] == ["sub/green.png"]
    assets = {a["rel_path"]: a for a in st.list_assets(lib["id"])}
    assert assets["red.jpg"]["id"] == ids1["red.jpg"]
    assert assets["sub/green.png"]["status"] == "missing"
    make_image(media_folder / "sub/green.png", (20, 200, 20), fmt="PNG")
    run_image_import(Ctx(), st, thumbs, lib["id"])
    assert st.get_asset(ids1["sub/green.png"])["status"] == "pending"


def test_cancel_then_resume(tmp_path, media_folder):
    st, lib, thumbs = setup(tmp_path, media_folder)
    with pytest.raises(JobCancelled):
        run_image_import(Ctx(cancel_after=2), st, thumbs, lib["id"])
    assert len(st.list_assets(lib["id"])) < 7
    r = run_image_import(Ctx(), st, thumbs, lib["id"])
    assert len(st.list_assets(lib["id"])) == 7
    assert r["new"] + r["unchanged"] + len(r["failed"]) >= 7 - 2


def test_import_with_embedder_reuses_and_marks_indexed(tmp_path, media_folder):
    st, lib, thumbs = setup(tmp_path, media_folder)
    calls = []

    def embed_batch(items):
        calls.append(len(items))
        for it in items:
            st.set_asset_status(it.asset_id, "indexed")
        return {"embedded": len(items)}

    r = run_image_import(Ctx(), st, thumbs, lib["id"], embed_batch, batch_size=2)
    assert r["embedded"] == 5 and sum(calls) == 5 and max(calls) <= 2
    r2 = run_image_import(Ctx(), st, thumbs, lib["id"], embed_batch)
    assert r2["embedded"] == 0  # unchanged vectors are not recomputed


def test_path_guards(tmp_path):
    with pytest.raises(PathRejected):
        validate_root("relative/path")
    with pytest.raises(PathRejected):
        validate_root("/")
    with pytest.raises(PathRejected):
        validate_root(os.path.expanduser("~"))
    with pytest.raises(PathRejected):
        validate_root(str(tmp_path / "nope"))
    with pytest.raises(PathRejected):
        resolve_in_root(tmp_path, "../etc/passwd")
    with pytest.raises(PathRejected):
        resolve_in_root(tmp_path, "/etc/passwd")


def test_api_import_and_serving(tmp_path, media_folder):
    app = create_app(Settings(data_dir=tmp_path / "data", precision="float32"), backend_factory=FakeBackend)
    with TestClient(app) as c:
        assert c.post("/api/libraries", json={"path": "/"}).status_code == 400
        lib = c.post("/api/libraries", json={"path": str(media_folder)}).json()
        job = c.post(f"/api/libraries/{lib['id']}/import").json()
        for _ in range(200):
            j = c.get(f"/api/jobs/{job['id']}").json()
            if j["status"] in ("done", "failed"):
                break
            time.sleep(0.02)
        assert j["status"] == "done", j
        assert j["result"]["new"] == 5
        assets = c.get(f"/api/libraries/{lib['id']}/assets").json()
        red = next(a for a in assets if a["rel_path"] == "red.jpg")
        r = c.get(red["thumbnail_url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        r = c.get(red["file_url"])
        assert r.status_code == 200 and r.content == (media_folder / "red.jpg").read_bytes()
        assert c.get("/api/assets/../../etc/passwd/file").status_code == 404
        assert c.get("/api/assets/doesnotexist/file").status_code == 404
        (media_folder / "red.jpg").rename(media_folder / "moved.jpg")
        assert c.get(red["file_url"]).status_code == 410
        # library deletion removes index rows only
        assert c.delete(f"/api/libraries/{lib['id']}").status_code == 200
        assert (media_folder / "moved.jpg").exists()


def test_provenance_sidecar_attached_as_metadata(tmp_path):
    import json

    root = tmp_path / "pack"
    make_image(root / "stock-00001.jpg")
    (root / "mediaindex-provenance.json").write_text(json.dumps({
        "dataset": "X/Y", "revision": "abc", "license_declared": "CC0-1.0",
        "items": {"stock-00001.jpg": {"row": 1, "source": "https://example/row1", "tags": "cat, sofa"}}}))
    st, lib, thumbs = setup(tmp_path, root)
    r = run_image_import(Ctx(), st, thumbs, lib["id"])
    assert r["discovered"] == 1
    a = st.list_assets(lib["id"])[0]
    meta = json.loads(a["meta_json"])
    assert meta["source"]["license_declared"] == "CC0-1.0" and meta["source"]["row"] == 1
    assert meta["source"]["inspection_tags"] == "cat, sofa"
