import shutil

import pytest
from fastapi.testclient import TestClient

from conftest import make_image
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.model.backend import FakeBackend
from test_audio import tone
from test_search import wait_job

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "snd"
    tone(root / "long.wav", 23, freq=300)
    tone(root / "a.flac", 3, freq=500)
    tone(root / "b.mp3", 4, freq=700, fmt_args=("-c:a", "libmp3lame"))
    shutil.copy(root / "a.flac", root / "a-copy.flac")
    make_image(root / "pic.jpg")
    settings = Settings(data_dir=tmp_path / "data", precision="float32")
    app = create_app(settings, backend_factory=FakeBackend)
    with TestClient(app) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        assets = {a["rel_path"]: a for a in c.get(f"/api/libraries/{lib['id']}/assets").json()}
        yield c, root, settings, assets, lib


def test_text_to_audio_grouped_segments(client):
    c, root, _, assets, lib = client
    r = c.post("/api/search/text", json={"text": "beep", "media_types": ["audio"], "limit": 10}).json()
    ids = [x["asset"]["id"] for x in r["results"]]
    assert len(ids) == len(set(ids)) == 4  # one result per asset
    long = next(x for x in r["results"] if x["asset"]["rel_path"] == "long.wav")
    assert long["modality"] == "audio" and long["end_s"] - long["start_s"] == pytest.approx(10)
    # 4 windows (0,10),(5,15),(10,20),(13,23): windows overlapping a kept one by >50% are merged
    assert len(long["other_segments"]) + long["overlapping_windows_merged"] == 3
    kept = [(long["start_s"], long["end_s"])] + [(o["start_s"], o["end_s"]) for o in long["other_segments"]]
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            assert max(0, min(a[1], b[1]) - max(a[0], b[0])) <= 5
    assert long["asset"]["duration"] == pytest.approx(23, abs=0.1)
    ungrouped = c.post("/api/search/text", json={"text": "beep", "media_types": ["audio"], "limit": 10,
                                                 "group_segments": False}).json()
    assert len(ungrouped["results"]) == 7  # 4 windows + 3 single-window files
    imgs = c.post("/api/search/text", json={"text": "beep", "media_types": ["image"]}).json()
    assert [x["asset"]["rel_path"] for x in imgs["results"]] == ["pic.jpg"]


def test_audio_reference_by_asset_window_and_upload(client):
    c, root, settings, assets, _ = client
    long = assets["long.wav"]
    r = c.post("/api/search/audio", data={"asset_id": long["id"], "start_s": "12"}).json()
    assert r["query"]["reference_span"] == [10.0, 20.0]
    assert long["id"] not in [x["asset"]["id"] for x in r["results"]]
    r = c.post("/api/search/audio", files={"file": ("q.flac", (root / "a.flac").read_bytes(), "audio/flac")}).json()
    names = [x["asset"]["rel_path"] for x in r["results"]]
    assert "a.flac" not in names and "a-copy.flac" not in names and r["query"]["excluded_identical"] == 2
    assert r["query"]["embedding"].startswith("computed")
    r = c.post("/api/search/audio", files={"file": ("q.flac", (root / "a.flac").read_bytes(), "audio/flac")},
               data={"include_identical": "true"}).json()
    assert {x["asset"]["rel_path"] for x in r["results"][:2]} == {"a.flac", "a-copy.flac"}
    assert list(settings.uploads_dir.iterdir()) == []


def test_audio_reference_errors(client):
    c, root, settings, assets, _ = client
    assert c.post("/api/search/audio", files={"file": ("q.ogg", b"OggS", "audio/ogg")}).status_code == 415
    assert c.post("/api/search/audio", files={"file": ("q.wav", b"RIFF0000WAVEjunk", "audio/wav")}).status_code == 422
    assert c.post("/api/search/audio", data={"asset_id": assets["pic.jpg"]["id"]}).status_code == 422
    assert c.post("/api/search/audio", data={}).status_code == 422
    assert c.post("/api/search/audio", data={"asset_id": "nope"}).status_code == 404
    assert list(settings.uploads_dir.iterdir()) == []


def test_audio_file_served_with_ranges(client):
    c, root, _, assets, _ = client
    r = c.get(assets["a.flac"]["file_url"], headers={"Range": "bytes=0-9"})
    assert r.status_code == 206 and len(r.content) == 10 and r.headers["content-type"] == "audio/flac"
