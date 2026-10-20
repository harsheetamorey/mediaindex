"""iPhone Live Photos: IMG_1234.HEIC + a ~3 s IMG_1234.MOV are one item (mocked embeddings)."""

import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from conftest import make_image, tree_digest
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.model.backend import FakeBackend
from test_search import wait_job

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def clip(path, seconds, colour="blue"):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=160x90:r=10:d={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def test_live_photo_video_is_shown_with_its_photo(tmp_path):
    root = tmp_path / "iPhone"
    make_image(root / "IMG_0001.HEIC", (200, 0, 0), fmt="HEIF")
    clip(root / "IMG_0001.MOV", 3)  # the Live Photo part
    make_image(root / "IMG_0002.JPG", (0, 200, 0))
    clip(root / "IMG_0002.MOV", 8)  # same name, but too long: a real video
    clip(root / "IMG_0003.MOV", 2)  # no photo next to it: a real video
    before = tree_digest(root)
    app = create_app(Settings(data_dir=tmp_path / "data", precision="float32", watch_interval=0), backend_factory=FakeBackend)
    with TestClient(app) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        job = wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        assert job["result"]["live_photos"] == 1 and job["result"]["failed"] == []
        assets = {a["rel_path"]: a for a in c.get(f"/api/libraries/{lib['id']}/assets").json()}
        live = assets["IMG_0001.MOV"]
        assert live["live_photo"] == "IMG_0001.HEIC" and live["status"] == "indexed"
        assert assets["IMG_0001.HEIC"]["live_video_url"] == live["file_url"]
        assert "live_video_url" not in assets["IMG_0002.JPG"] and assets["IMG_0002.MOV"]["live_photo"] is None
        videos = c.post("/api/search/text", json={"text": "blue", "media_types": ["video"]}).json()["results"]
        assert sorted(x["asset"]["rel_path"] for x in videos) == ["IMG_0002.MOV", "IMG_0003.MOV"]
        # re-import: nothing changes; remove the photo and the clip becomes a normal video again
        job = wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        assert job["result"]["embedded"] == 0 and job["result"]["live_photos"] == 1
        after = tree_digest(root)
        (root / "IMG_0001.HEIC").unlink()
        wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        videos = c.post("/api/search/text", json={"text": "blue", "media_types": ["video"]}).json()["results"]
        assert "IMG_0001.MOV" in [x["asset"]["rel_path"] for x in videos]
    assert after == before  # originals untouched
