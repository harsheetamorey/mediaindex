import json
import os
import threading

import pytest
from fastapi.testclient import TestClient

from conftest import make_image, tree_digest
from mediaindex import selections as sel
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.jobs import Job, JobCancelled, JobContext
from mediaindex.model.backend import FakeBackend
from test_search import wait_job


@pytest.fixture
def env(tmp_path):
    a = tmp_path / "Lib A"  # spaces in paths
    b = tmp_path / "Lib B"
    make_image(a / "photo one.jpg", (200, 0, 0))
    make_image(a / "shared.jpg", (0, 200, 0))
    make_image(b / "shared.jpg", (0, 0, 200))  # same filename, different content
    make_image(b / "nested/deep.png", (9, 9, 9), fmt="PNG")
    data = tmp_path / "data"
    app = create_app(Settings(data_dir=data, precision="float32"), backend_factory=FakeBackend)
    c = TestClient(app)
    c.__enter__()
    libs = []
    for root in (a, b):
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        libs.append(lib)
    assets = {}
    for lib in libs:
        for x in c.get(f"/api/libraries/{lib['id']}/assets").json():
            assets[f"{lib['name']}/{x['rel_path']}"] = x
    yield c, app, tmp_path, assets, (tree_digest(a), tree_digest(b)), (a, b), data
    c.__exit__(None, None, None)


def make_selection(c, assets, names):
    s = c.post("/api/selections", json={"name": "Pick 1"}).json()
    for n in names:
        r = c.post(f"/api/selections/{s['id']}/items",
                   json={"asset_id": assets[n]["id"], "query_context": {"mode": "text", "text": "test"}})
        assert r.status_code == 200
    return s


def test_crud_reorder_idempotent_and_persistence(env):
    c, app, tmp, assets, *_ , data = env
    s = make_selection(c, assets, ["Lib A/photo one.jpg", "Lib A/shared.jpg", "Lib B/shared.jpg"])
    again = c.post(f"/api/selections/{s['id']}/items", json={"asset_id": assets["Lib A/shared.jpg"]["id"]}).json()
    assert again["added"] is False
    items = c.get(f"/api/selections/{s['id']}").json()["items"]
    assert [i["snapshot"]["rel_path"] for i in items] == ["photo one.jpg", "shared.jpg", "shared.jpg"]
    ids = [i["id"] for i in items]
    assert c.put(f"/api/selections/{s['id']}/order", json={"item_ids": ids[::-1]}).status_code == 200
    assert c.put(f"/api/selections/{s['id']}/order", json={"item_ids": ids[:1]}).status_code == 422
    c.delete(f"/api/selections/{s['id']}/items/{ids[1]}")
    # restart: new app on same data dir
    app2 = create_app(Settings(data_dir=data, precision="float32"), backend_factory=FakeBackend)
    with TestClient(app2) as c2:
        items = c2.get(f"/api/selections/{s['id']}").json()["items"]
        assert [i["id"] for i in items] == [ids[2], ids[0]]
        assert all(i["status"] == "ok" for i in items)


def test_missing_and_removed_status_and_manifest(env):
    c, app, tmp, assets, digests, (a, b), _ = env
    s = make_selection(c, assets, ["Lib A/photo one.jpg", "Lib B/shared.jpg", "Lib B/nested/deep.png"])
    os.rename(a / "photo one.jpg", tmp / "elsewhere.jpg")
    libb = assets["Lib B/shared.jpg"]["library_id"]
    c.delete(f"/api/libraries/{libb}")  # removes index rows only
    items = c.get(f"/api/selections/{s['id']}").json()["items"]
    assert [i["status"] for i in items] == ["missing", "removed", "removed"]
    assert (b / "shared.jpg").exists()
    m = c.get(f"/api/selections/{s['id']}/manifest")
    assert "attachment" in m.headers["content-disposition"]
    man = m.json()
    assert man["format"] == "mediaindex-selection-manifest" and len(man["items"]) == 3
    first = man["items"][0]
    assert first["source_path"].endswith("Lib A/photo one.jpg") and first["sha256"]
    assert first["query_context"] == {"mode": "text", "text": "test"}


def test_export_preview_confirm_collisions_and_no_overwrite(env):
    c, app, tmp, assets, digests, (a, b), _ = env
    s = make_selection(c, assets, ["Lib A/photo one.jpg", "Lib A/shared.jpg", "Lib B/shared.jpg", "Lib B/nested/deep.png"])
    dest = tmp / "My Exports"
    dest.mkdir()
    (dest / "shared.jpg").write_bytes(b"pre-existing user file")
    # destination restrictions
    for bad in ("relative/dir", "/", str(a), str(a / "sub"), str(tmp), str(dest / "shared.jpg")):
        assert c.post(f"/api/selections/{s['id']}/export/preview", json={"destination": bad}).status_code == 400, bad
    plan = c.post(f"/api/selections/{s['id']}/export/preview", json={"destination": str(dest)}).json()
    names = [f["dest_name"] for f in plan["files"]]
    assert names == ["photo one.jpg", "shared (2).jpg", "shared (3).jpg", "deep.png"]
    assert plan["overwrites"] == 0 and plan["manifest_name"] == "mediaindex-manifest.json"
    assert c.post(f"/api/selections/{s['id']}/export", json={"plan_id": plan["plan_id"], "confirm": False}).status_code == 400
    job = c.post(f"/api/selections/{s['id']}/export", json={"plan_id": plan["plan_id"], "confirm": True}).json()
    j = wait_job(c, job["id"])
    assert j["status"] == "done" and j["result"]["copied"] == 4, j
    assert (dest / "shared.jpg").read_bytes() == b"pre-existing user file"
    assert (dest / "photo one.jpg").read_bytes() == (a / "photo one.jpg").read_bytes()
    assert (dest / "shared (3).jpg").read_bytes() == (b / "shared.jpg").read_bytes()
    man = json.loads((dest / "mediaindex-manifest.json").read_text())
    assert [i["exported_as"] for i in man["items"]] == names
    assert not [p for p in dest.iterdir() if p.name.startswith(sel.PARTIAL_PREFIX)]
    # plan is single-use
    assert c.post(f"/api/selections/{s['id']}/export", json={"plan_id": plan["plan_id"], "confirm": True}).status_code == 409
    # second export to same folder never overwrites, including the manifest
    plan2 = c.post(f"/api/selections/{s['id']}/export/preview", json={"destination": str(dest)}).json()
    assert plan2["files"][0]["dest_name"] == "photo one (2).jpg"
    assert plan2["manifest_name"] == "mediaindex-manifest (2).json"
    assert (tree_digest(a), tree_digest(b)) == digests


def test_interrupted_copy_leaves_no_partials(env, monkeypatch):
    c, app, tmp, assets, digests, (a, b), _ = env
    s = make_selection(c, assets, ["Lib A/photo one.jpg", "Lib A/shared.jpg", "Lib B/shared.jpg"])
    dest = tmp / "out"
    plan = sel.plan_export(app.state.store, s["id"], str(dest))
    ctx = JobContext(Job(id="x", kind="export"), None)
    real_copy = sel.shutil.copyfile
    calls = []

    def copy_then_cancel(src, dst):
        calls.append(src)
        real_copy(src, dst)
        if len(calls) == 2:
            ctx.job._cancel.set()  # cancellation arrives during the 2nd file

    monkeypatch.setattr(sel.shutil, "copyfile", copy_then_cancel)
    with pytest.raises(JobCancelled):
        sel.run_export(ctx, app.state.store, plan)
    files = sorted(p.name for p in dest.iterdir())
    assert not [f for f in files if f.startswith(sel.PARTIAL_PREFIX)]
    assert "photo one.jpg" in files and ctx.job.result["cancelled"] is True
    assert ctx.job.result["copied"] == 2  # file in flight finished atomically; 3rd never started
    # failure mid-copy (e.g. disk full) also leaves no partial file
    monkeypatch.setattr(sel.shutil, "copyfile", lambda s_, d: (open(d, "wb").write(b"x"), (_ for _ in ()).throw(OSError("No space left on device"))))
    plan2 = sel.plan_export(app.state.store, s["id"], str(tmp / "out2"))
    res = sel.run_export(JobContext(Job(id="y", kind="export"), None), app.state.store, plan2)
    assert len(res["failed"]) == 3 and res["copied"] == 0
    assert not list((tmp / "out2").iterdir())
    assert (tree_digest(a), tree_digest(b)) == digests


def test_copy_path_and_reveal_use_ids_only(env, monkeypatch):
    c, app, tmp, assets, *_ = env
    aid = assets["Lib A/photo one.jpg"]["id"]
    assert c.get(f"/api/assets/{aid}/path").json()["path"].endswith("Lib A/photo one.jpg")
    calls = []
    import mediaindex.api_selections as api

    monkeypatch.setattr(api.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    assert c.post(f"/api/assets/{aid}/reveal").status_code == 200
    cmd, kw = calls[0]
    assert isinstance(cmd, list) and kw["shell"] is False and cmd[-1].endswith("photo one.jpg")
    assert c.post("/api/assets/..%2F..%2Fetc/reveal").status_code in (404, 405)
    assert c.post("/api/assets/nope/reveal").status_code == 404
