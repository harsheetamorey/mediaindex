import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import tree_digest
from mediaindex import clips
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.jobs import Job, JobCancelled, JobContext
from mediaindex.media.video import probe_video, sample_frames
from mediaindex.model.backend import FakeBackend
from test_audio import tone
from test_search import wait_job
from test_video import RGB, color_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "My Videos"
    color_video(root / "colors.mp4")  # 20 s: red, green, blue, white, black (4 s each)
    tone(tmp_path / "sfx" / "beep.wav", 3)
    settings = Settings(data_dir=tmp_path / "data", precision="float32")
    app = create_app(settings, backend_factory=FakeBackend)
    with TestClient(app) as c:
        libs = []
        for r in (root, tmp_path / "sfx"):
            lib = c.post("/api/libraries", json={"path": str(r)}).json()
            j = wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
            assert j["status"] == "done", j
            libs.append(lib)
        assets = {a["rel_path"]: a for lib in libs for a in c.get(f"/api/libraries/{lib['id']}/assets").json()}
        yield c, tmp_path, root, assets, tree_digest(root)


def test_text_to_video_moments_grouped_and_deduped(env):
    c, tmp, root, assets, _ = env
    r = c.post("/api/search/text", json={"text": "green", "media_types": ["video"]}).json()
    assert len(r["results"]) == 1
    x = r["results"][0]
    assert x["modality"] == "video-visual" and x["asset"]["media_type"] == "video"
    kept = [(x["start_s"], x["end_s"])] + [(o["start_s"], o["end_s"]) for o in x["other_segments"]]
    assert len(kept) + x["overlapping_windows_merged"] == 4  # (0,8),(4,12),(8,16),(12,20)
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            assert max(0, min(a[1], b[1]) - max(a[0], b[0])) <= 4
    assert c.get(x["window_thumbnail_url"]).headers["content-type"] == "image/jpeg"
    ra = c.post("/api/search/text", json={"text": "beep", "media_types": ["video"], "video_signal": "audio"}).json()
    assert ra["results"][0]["modality"] == "video-audio" and ra["query"]["video_signal"] == "audio"


def test_video_window_to_audio_suggestions_and_image_text_to_video(env):
    c, tmp, root, assets, _ = env
    v = assets["colors.mp4"]
    r = c.post("/api/search/video-window", json={"asset_id": v["id"], "start_s": 5, "target": "audio"}).json()
    assert r["query"]["reference_span"] == [4.0, 12.0] and r["query"]["mode_status"] == "experimental"
    assert "not generated, timed or synchronized" in r["query"]["caveat"]
    assert [x["asset"]["rel_path"] for x in r["results"]] == ["beep.wav"]
    r = c.post("/api/search/video-window", json={"asset_id": v["id"], "start_s": 0, "target": "video"}).json()
    assert r["results"] == []  # same video excluded by default
    assert c.post("/api/search/video-window", json={"asset_id": assets["beep.wav"]["id"], "start_s": 0}).status_code == 404


def test_clip_preview_validation(env):
    c, tmp, root, assets, _ = env
    v = assets["colors.mp4"]
    dest = tmp / "exports"
    bad = [
        {"start_s": 5, "end_s": 4, "destination": str(dest)},
        {"start_s": 5, "end_s": 5.2, "destination": str(dest)},
        {"start_s": 0, "end_s": 4, "destination": str(root)},  # inside library
        {"start_s": 0, "end_s": 4, "destination": "relative"},
        {"start_s": 0, "end_s": 4, "destination": str(dest), "mode": "lossless"},
    ]
    for b in bad:
        assert c.post(f"/api/assets/{v['id']}/clip/preview", json=b).status_code == 400, b
    p = c.post(f"/api/assets/{v['id']}/clip/preview", json={"start_s": 18, "end_s": 99, "destination": str(dest)}).json()
    assert p["end_s"] == pytest.approx(20, abs=0.1)  # clamped to duration
    assert c.post(f"/api/assets/{assets['beep.wav']['id']}/clip/preview",
                  json={"start_s": 0, "end_s": 2, "destination": str(dest)}).status_code == 400


def test_accurate_clip_export_content_duration_manifest_no_overwrite(env):
    c, tmp, root, assets, before = env
    v = assets["colors.mp4"]
    dest = tmp / "exports"
    dest.mkdir()
    name = "colors_00m04_0s-00m08_0s.mp4"
    (dest / name).write_bytes(b"user file")
    p = c.post(f"/api/assets/{v['id']}/clip/preview",
               json={"start_s": 4, "end_s": 8, "destination": str(dest), "query_context": {"text": "green"}}).json()
    assert p["dest_name"] == "colors_00m04_0s-00m08_0s (2).mp4" and p["overwrites"] == 0
    assert c.post(f"/api/assets/{v['id']}/clip", json={"plan_id": p["plan_id"], "confirm": False}).status_code == 400
    j = wait_job(c, c.post(f"/api/assets/{v['id']}/clip", json={"plan_id": p["plan_id"], "confirm": True}).json()["id"])
    assert j["status"] == "done", j
    out = dest / j["result"]["clip"]
    info = probe_video(out)
    assert info.duration == pytest.approx(4, abs=0.15) and info.has_audio
    frames, _ = sample_frames(out, info, 0, 4)
    assert all(np.allclose(f.reshape(-1, 3).mean(0), RGB["green"], atol=40) for f in frames)  # exactly the green segment
    man = json.loads((dest / j["result"]["manifest"]).read_text())
    assert man["requested_range_s"] == [4.0, 8.0] and man["query_context"] == {"text": "green"}
    assert man["source_sha256"] == v["content_hash"]
    assert (dest / name).read_bytes() == b"user file"
    assert not [x for x in dest.iterdir() if x.name.startswith(".mediaindex-partial")]
    assert tree_digest(root) == before


def test_fast_clip_reports_keyframe_start(env):
    c, tmp, root, assets, _ = env
    v = assets["colors.mp4"]
    p = c.post(f"/api/assets/{v['id']}/clip/preview",
               json={"start_s": 6.3, "end_s": 9, "mode": "fast", "destination": str(tmp / "fast")}).json()
    assert p["effective_start_s"] <= 6.3
    j = wait_job(c, c.post(f"/api/assets/{v['id']}/clip", json={"plan_id": p["plan_id"], "confirm": True}).json()["id"])
    assert j["status"] == "done", j
    assert j["result"]["output_duration_s"] >= 9 - 6.3 - 0.6


def test_clip_cancel_and_codec_failure_leave_no_partials(env, tmp_path):
    c, tmp, root, assets, _ = env
    v = assets["colors.mp4"]
    ctx = JobContext(Job(id="x", kind="clip"), None)
    ctx.job._cancel.set()
    out = tmp / "cx" / ".mediaindex-partial-test.mp4"
    out.parent.mkdir()
    with pytest.raises(JobCancelled):
        clips.render_clip(ctx, root / "colors.mp4", 0, 10, "accurate", out, 0)
    broken = tmp / "broken.mp4"
    broken.write_bytes(b"\x00" * 1000)
    with pytest.raises(OSError, match="ffmpeg failed"):
        clips.render_clip(None, broken, 0, 2, "accurate", tmp / "cx" / "o.mp4", 0)


def test_selection_export_turns_video_moments_into_clips(env):
    c, tmp, root, assets, before = env
    v = assets["colors.mp4"]
    s = c.post("/api/selections", json={"name": "moments"}).json()
    c.post(f"/api/selections/{s['id']}/items", json={"asset_id": v["id"], "start_s": 8, "end_s": 12})
    c.post(f"/api/selections/{s['id']}/items", json={"asset_id": assets["beep.wav"]["id"]})
    plan = c.post(f"/api/selections/{s['id']}/export/preview", json={"destination": str(tmp / "sel")}).json()
    assert plan["files"][0]["clip"] == [8.0, 12.0] and plan["files"][1]["dest_name"] == "beep.wav"
    j = wait_job(c, c.post(f"/api/selections/{s['id']}/export", json={"plan_id": plan["plan_id"], "confirm": True}).json()["id"])
    assert j["status"] == "done" and j["result"]["copied"] == 2, j
    clip = tmp / "sel" / plan["files"][0]["dest_name"]
    assert probe_video(clip).duration == pytest.approx(4, abs=0.15)
    assert tree_digest(root) == before
