"""Watched folders, low-priority indexing, ETA, native folder picker and Finder tags on exports (mocked model)."""

import os
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from conftest import make_image, tree_digest
from mediaindex import app as app_module
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.db import Database
from mediaindex.finder_tags import read_finder_tags, set_finder_tags
from mediaindex.jobs import Job, JobRunner
from mediaindex.model.backend import FakeBackend
from mediaindex.store import Store
from mediaindex.watch import FolderWatcher, folder_signature
from test_search import wait_job

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS feature")


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "Shoot"
    make_image(root / "a.jpg", (200, 0, 0))
    make_image(root / "b.jpg", (0, 200, 0))
    app = create_app(Settings(data_dir=tmp_path / "data", precision="float32", watch_interval=0),
                     backend_factory=FakeBackend)
    with TestClient(app) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        yield c, app, root, lib


def test_signature_counts_only_media(tmp_path):
    make_image(tmp_path / "x.jpg")
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / ".hidden.jpg").write_bytes(b"ignored")
    n, size, _ = folder_signature(tmp_path)
    assert n == 1 and size == (tmp_path / "x.jpg").stat().st_size
    assert folder_signature(tmp_path / "missing") is None


def test_watcher_waits_for_changes_to_settle(tmp_path):
    store = Store(Database(tmp_path / "db.sqlite3"))
    root = tmp_path / "lib"
    make_image(root / "one.jpg")
    lib = store.create_library("lib", root)
    queued = []
    w = FolderWatcher(store, queued.append, interval=0)
    assert w.poll() == []  # not watched yet
    store.set_library_watch(lib["id"], True)
    assert w.poll() == []  # first sight: baseline only
    make_image(root / "two.jpg", (1, 2, 3))
    assert w.poll() == []  # changed once: wait (a copy may still be in progress)
    assert w.poll() == [lib["id"]] and queued == [lib["id"]]  # stable for one more poll: re-scan
    assert w.poll() == []  # nothing new
    store.set_library_watch(lib["id"], False)
    make_image(root / "three.jpg", (3, 2, 1))
    assert w.poll() == [] and w.poll() == []  # unwatched folders are ignored


def test_watched_folder_reindexes_new_files(client):
    c, app, root, lib = client
    assert c.patch(f"/api/libraries/{lib['id']}", json={"watch": True}).json()["watch"] == 1
    assert c.get("/api/libraries").json()[0]["watch"] == 1
    w = app.state.watcher
    before = tree_digest(root)
    assert w.poll() == []
    make_image(root / "new" / "c.jpg", (0, 0, 200))
    w.poll()
    assert w.poll() == [lib["id"]]
    job = max((j for j in c.get("/api/jobs").json() if j["kind"] == "import" and j["library_id"] == lib["id"]),
              key=lambda j: j["created"])  # the re-scan the watcher just queued, not the first import
    wait_job(c, job["id"])
    rels = {a["rel_path"]: a["status"] for a in c.get(f"/api/libraries/{lib['id']}/assets").json()}
    assert rels.get("new/c.jpg") == "indexed"
    after = tree_digest(root)
    assert {k: after[k] for k in before} == before  # originals untouched
    assert c.patch("/api/libraries/nope", json={"watch": True}).status_code == 404


def test_low_priority_setting_persists_and_applies(tmp_path, client):
    c, app, root, lib = client
    assert c.get("/api/settings").json()["low_priority_indexing"] is False
    assert c.patch("/api/settings", json={"low_priority_indexing": True}).json()["low_priority_indexing"] is True
    assert app.state.runner.low_priority is True
    app2 = create_app(Settings(data_dir=app.state.settings.data_dir, precision="float32", watch_interval=0),
                      backend_factory=FakeBackend)
    assert app2.state.runner.low_priority is True  # stored in the database
    app2.state.runner.shutdown()
    app2.state.db.close()


@darwin_only
def test_low_priority_sets_background_qos_on_worker_thread():
    runner = JobRunner(low_priority=True)
    seen = {}
    try:
        def fn(ctx):
            seen["low"] = os.getpriority(os.PRIO_DARWIN_THREAD, 0)
            runner.low_priority = False
            ctx.progress(1, 1)  # setting changes apply between work units
            seen["normal"] = os.getpriority(os.PRIO_DARWIN_THREAD, 0)
            return {}

        job = runner.submit("t", fn)
        for _ in range(100):
            if job.status == "done":
                break
            time.sleep(0.05)
        assert job.status == "done", job.error
        assert seen["low"] != 0 and seen["normal"] == 0
    finally:
        runner.shutdown()


def test_eta_ignores_unchanged_files():
    job = Job(id="j", kind="import", status="running", total=100)
    from mediaindex.jobs import JobContext

    ctx = JobContext(job, None)
    for i in range(1, 51):
        ctx.progress(i, work=False)  # fast unchanged files
    assert job.to_dict()["eta_seconds"] is None
    job._work_started = time.time() - 10  # pretend: 10 s of real work for 5 files
    job._work_units = 5
    eta = job.to_dict()["eta_seconds"]
    assert 90 <= eta <= 110  # 50 left at 2 s each
    job.status = "done"
    assert job.to_dict()["eta_seconds"] is None


def test_pick_folder(client, monkeypatch):
    c, *_ = client
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="/Users/me/Shoot Day/\n", stderr="")

    monkeypatch.setattr(app_module.sys, "platform", "darwin")
    monkeypatch.setattr(app_module.subprocess, "run", fake_run)
    assert c.post("/api/pick-folder").json() == {"cancelled": False, "path": "/Users/me/Shoot Day"}
    assert calls[0][0] == "osascript" and "choose folder" in " ".join(calls[0])
    monkeypatch.setattr(app_module.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="User canceled. (-128)"))
    assert c.post("/api/pick-folder").json() == {"cancelled": True}
    monkeypatch.setattr(app_module.sys, "platform", "linux")
    assert c.post("/api/pick-folder").status_code == 501
    # cross-site pages cannot trigger the dialog
    assert c.post("/api/pick-folder", headers={"Origin": "https://evil.example"}).status_code == 403


@darwin_only
def test_finder_tags_roundtrip(tmp_path):
    f = tmp_path / "x.jpg"
    make_image(f)
    assert read_finder_tags(f) == []
    assert set_finder_tags(f, ["MediaIndex", "Client: Acme"])
    assert read_finder_tags(f) == ["MediaIndex", "Client: Acme"]


@darwin_only
def test_export_tags_copies_but_never_originals(client, tmp_path):
    c, app, root, lib = client
    assets = c.get(f"/api/libraries/{lib['id']}/assets").json()
    s = c.post("/api/selections", json={"name": "Hero shots"}).json()
    for a in assets:
        c.post(f"/api/selections/{s['id']}/items", json={"asset_id": a["id"]})
    dest = tmp_path / "out"
    dest.mkdir()
    plan = c.post(f"/api/selections/{s['id']}/export/preview", json={"destination": str(dest)}).json()
    j = wait_job(c, c.post(f"/api/selections/{s['id']}/export", json={"plan_id": plan["plan_id"], "confirm": True}).json()["id"])
    assert j["result"]["tagged"] == 2 and j["result"]["finder_tags"] == ["MediaIndex", "Hero shots"]
    for name in ("a.jpg", "b.jpg"):
        assert read_finder_tags(dest / name) == ["MediaIndex", "Hero shots"]
        assert read_finder_tags(root / name) == []  # originals carry no tags
    assert read_finder_tags(dest / "mediaindex-manifest.json") == []
