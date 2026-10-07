import json
import shutil
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_image, tree_digest
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.db import Database, now
from mediaindex.ingest import run_image_import
from mediaindex.indexer import make_audio_embedder
from mediaindex.jobs import Job, JobCancelled, JobContext
from mediaindex.media.audio import AudioRejected, decode_audio, plan_windows, probe_audio
from mediaindex.model.backend import FakeBackend
from mediaindex.model.host import ModelHost
from mediaindex.model.profiles import IndexProfile
from mediaindex.store import Store
from test_search import wait_job

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(path, seconds, freq=440, fmt_args=(), channels=1, rate=44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency={freq}:sample_rate={rate}:duration={seconds}", "-ac", str(channels),
                    *fmt_args, str(path)], check=True)
    return path


def test_plan_windows():
    assert plan_windows(3.2, 10, 5) == [(0.0, 3.2)]
    assert plan_windows(10, 10, 5) == [(0.0, 10)]
    assert plan_windows(23, 10, 5) == [(0.0, 10.0), (5.0, 15.0), (10.0, 20.0), (13.0, 23.0)]
    w = plan_windows(3600, 10, 10)
    assert w[0] == (0.0, 10.0) and w[-1][1] == 3600 and len(w) == 360


def test_decode_formats_mono_16k(tmp_path):
    for name, args in (("a.wav", ()), ("a.flac", ()), ("a.mp3", ("-c:a", "libmp3lame"))):
        p = tone(tmp_path / name, 2, channels=2, fmt_args=args)
        info = probe_audio(p)
        assert info.duration == pytest.approx(2, abs=0.1) and info.channels == 2
        x = decode_audio(p)
        assert x.dtype == np.float32 and abs(len(x) - 32000) < 2000  # 16 kHz mono
        span = decode_audio(p, start=1.0, duration=0.5)
        assert abs(len(span) - 8000) < 400
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"ID3" + b"\x00" * 100)
    with pytest.raises(AudioRejected):
        probe_audio(bad)


def _setup(tmp_path, root):
    st = Store(Database(tmp_path / "db.sqlite3"))
    prof = IndexProfile(precision="float32")
    st.register_profile(prof)
    host = ModelHost(prof, FakeBackend)
    lib = st.create_library("m", root)
    return st, prof, host, lib


def test_audio_import_segments_offsets_reuse_and_failures(tmp_path):
    root = tmp_path / "sounds"
    tone(root / "long.wav", 23)
    tone(root / "short.flac", 3, freq=880)
    shutil.copy(root / "short.flac", root / "copy-of-short.flac")
    (root / "broken.wav").write_bytes(b"RIFF....WAVEfmt garbage")
    make_image(root / "pic.jpg")
    before = tree_digest(root)
    st, prof, host, lib = _setup(tmp_path, root)
    from mediaindex.indexer import make_image_embedder

    ea = make_audio_embedder(st, host)
    r = run_image_import(JobContext(Job(id="j", kind="import"), None), st, tmp_path / "thumbs", lib["id"],
                         make_image_embedder(st, host), embed_audio=ea)
    assert r["audio_files"] == 4 and r["failed"][0][0] == "broken.wav"
    assets = {a["rel_path"]: a for a in st.list_assets(lib["id"])}
    assert assets["long.wav"]["media_type"] == "audio" and assets["long.wav"]["duration"] == pytest.approx(23, abs=0.1)
    rows = st.db.query("SELECT start_s, end_s, segment_index FROM embeddings WHERE asset_id=? ORDER BY segment_index",
                       (assets["long.wav"]["id"],))
    assert [(x["start_s"], x["end_s"]) for x in rows] == [(0.0, 10.0), (5.0, 15.0), (10.0, 20.0), (13.0, 23.0)]
    assert r["reused_vectors"] == 1  # identical short clip
    vm = st.load_matrix(prof.key, None, ["audio"])
    assert len(vm.asset_ids) == 4 + 1 + 1 and np.isfinite(vm.matrix).all()
    assert len(st.load_matrix(prof.key, None, ["image"]).asset_ids) == 1  # images unaffected
    assert tree_digest(root) == before
    r2 = run_image_import(JobContext(Job(id="j2", kind="import"), None), st, tmp_path / "thumbs", lib["id"],
                          make_image_embedder(st, host), embed_audio=ea)
    assert r2["embedded"] == 0 and r2["new"] == 0


def test_cancel_during_audio_file_leaves_it_unsearchable(tmp_path):
    root = tmp_path / "s"
    tone(root / "long.wav", 60)
    st, prof, host, lib = _setup(tmp_path, root)
    ea = make_audio_embedder(st, host, batch_size=1)
    ctx = JobContext(Job(id="j", kind="import"), None)
    orig = host.use
    calls = []

    def counting_use(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            ctx.job._cancel.set()
        return orig(*a, **k)

    host.use = counting_use
    with pytest.raises(JobCancelled):
        run_image_import(ctx, st, tmp_path / "t", lib["id"], None, embed_audio=ea)
    a = st.list_assets(lib["id"])[0]
    assert a["status"] == "pending" and st.get_asset_vectors(a["id"], prof.key) is None
    assert st.load_matrix(prof.key).asset_ids == []


def test_compatible_profile_rekey_preserves_image_vectors(tmp_path):
    db = Database(Settings(data_dir=tmp_path).db_path)
    st = Store(db)
    old = {**IndexProfile(precision="float32").to_dict(), "encoders": ["image", "text"]}
    old.pop("audio_window_s")
    old.pop("audio_stride_s")
    with db.tx() as c:
        c.execute("INSERT INTO index_profiles(key, json, created_at) VALUES (?,?,?)", ("oldkey", json.dumps(old), now()))
    lib = st.create_library("l", tmp_path)
    aid, _ = st.upsert_asset(lib["id"], "a.jpg", "image", size=1, mtime_ns=1, content_hash="h")
    v = np.zeros(768, np.float32)
    v[3] = 1
    st.write_embeddings(aid, "oldkey", "image", v)
    incompatible = {**old, "revision": "other"}
    with db.tx() as c:
        c.execute("INSERT INTO index_profiles(key, json, created_at) VALUES (?,?,?)", ("otherkey", json.dumps(incompatible), now()))
    db.close()
    app = create_app(Settings(data_dir=tmp_path, precision="float32"), backend_factory=FakeBackend)
    with TestClient(app):
        cur = app.state.profile.key
        m = app.state.store.load_matrix(cur)
        assert m.asset_ids == [aid] and np.allclose(m.matrix[0], v)
        assert app.state.store.load_matrix("otherkey").asset_ids == []


def test_api_import_mixed_library(tmp_path):
    root = tmp_path / "mix"
    tone(root / "beep.mp3", 4, fmt_args=("-c:a", "libmp3lame"))
    make_image(root / "pic.png", fmt="PNG")
    app = create_app(Settings(data_dir=tmp_path / "data", precision="float32"), backend_factory=FakeBackend)
    with TestClient(app) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        j = wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        assert j["status"] == "done" and j["result"]["audio_files"] == 1, j
        assets = {a["rel_path"]: a for a in c.get(f"/api/libraries/{lib['id']}/assets").json()}
        assert assets["beep.mp3"]["status"] == "indexed"
        assert c.get(assets["beep.mp3"]["thumbnail_url"]).status_code == 200
        assert c.get(assets["beep.mp3"]["file_url"]).headers["content-type"] == "audio/mpeg"
        r = c.post("/api/search/text", json={"text": "x", "library_ids": [lib["id"]]}).json()
        assert [x["asset"]["rel_path"] for x in r["results"]] == ["pic.png"]  # image search unaffected
