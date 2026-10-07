import shutil
import subprocess

import numpy as np
import pytest

from mediaindex.db import Database
from mediaindex.indexer import make_video_embedder
from mediaindex.ingest import run_image_import, window_thumb_path
from mediaindex.jobs import Job, JobCancelled, JobContext
from mediaindex.media.video import VideoRejected, probe_video, sample_frames
from mediaindex.model.backend import FakeBackend
from mediaindex.model.host import ModelHost
from mediaindex.model.profiles import IndexProfile
from mediaindex.store import Store

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
COLORS = ["red", "green", "blue", "white", "black"]
RGB = {"red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255), "white": (255, 255, 255), "black": (0, 0, 0)}


def color_video(path, seg_s=4, colors=COLORS, audio=True, vcodec=("-c:v", "libx264", "-pix_fmt", "yuv420p")):
    """Solid-colour segments of seg_s seconds each (known content per interval)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for c in colors:
        cmd += ["-f", "lavfi", "-i", f"color=c={c}:s=160x90:r=10:d={seg_s}"]
    n = len(colors)
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seg_s * n}"]
    filt = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    cmd += ["-filter_complex", filt, "-map", "[v]"]
    if audio:
        cmd += ["-map", f"{n}:a", "-c:a", "aac"]
    cmd += [*vcodec, str(path)]
    subprocess.run(cmd, check=True)
    return path


def test_probe_and_frame_sampling_offsets(tmp_path):
    v = color_video(tmp_path / "c.mp4")
    info = probe_video(v)
    assert info.codec == "h264" and info.has_audio and info.duration == pytest.approx(20, abs=0.2)
    for start, expect in ((0, "red"), (4, "green"), (8, "blue"), (12, "white"), (16, "black")):
        frames, times = sample_frames(v, info, start, 4, fps=1.0)
        assert len(frames) == 4 and times[0] == pytest.approx(start + 0.5)
        mean = frames.reshape(-1, 3).mean(0)
        assert np.allclose(mean, RGB[expect], atol=40), (start, expect, mean)
    frames, _ = sample_frames(v, info, 2, 4, fps=1.0)  # straddles red->green
    means = [f.reshape(-1, 3).mean(0) for f in frames]
    assert np.allclose(means[0], RGB["red"], atol=40) and np.allclose(means[-1], RGB["green"], atol=40)


def test_rejects_bad_containers_and_codecs(tmp_path):
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100)
    with pytest.raises(VideoRejected):
        probe_video(fake)
    mj = color_video(tmp_path / "mj.mp4", colors=["red"], audio=False, vcodec=("-c:v", "mjpeg"))
    with pytest.raises(VideoRejected, match="unsupported video codec"):
        probe_video(mj)


def _env(tmp_path, root, modalities=("video-visual", "video-audio", "video-joint")):
    st = Store(Database(tmp_path / "db.sqlite3"))
    prof = IndexProfile(precision="float32", video_modalities=modalities)
    st.register_profile(prof)
    host = ModelHost(prof, FakeBackend)
    lib = st.create_library("v", root)
    return st, prof, host, lib, make_video_embedder(st, host, tmp_path / "thumbs")


def test_video_import_windows_modalities_thumbs(tmp_path):
    root = tmp_path / "vids"
    color_video(root / "with audio.mp4")
    color_video(root / "silent.mp4", audio=False)
    shutil.copy(root / "silent.mp4", root / "silent-copy.mp4")
    st, prof, host, lib, ev = _env(tmp_path, root)
    r = run_image_import(JobContext(Job(id="j", kind="import"), None), st, tmp_path / "thumbs", lib["id"],
                         None, embed_video=ev)
    assert r["video_files"] == 3 and r["video_windows"] == 12 and r["reused_vectors"] == 1, r
    assets = {a["rel_path"]: a for a in st.list_assets(lib["id"])}
    a = assets["with audio.mp4"]
    assert a["media_type"] == "video" and a["status"] == "indexed" and (a["width"], a["height"]) == (160, 90)
    def segs(aid, m):
        return [(x["start_s"], x["end_s"]) for x in st.db.query(
            "SELECT start_s, end_s FROM embeddings WHERE asset_id=? AND modality=? ORDER BY segment_index", (aid, m))]
    want = [(0.0, 8.0), (4.0, 12.0), (8.0, 16.0), (12.0, 20.0)]
    for m in ("video-visual", "video-audio", "video-joint"):
        assert segs(a["id"], m) == want, m
    s = assets["silent.mp4"]
    assert segs(s["id"], "video-visual") == want and segs(s["id"], "video-audio") == [] and segs(s["id"], "video-joint") == []
    for st0, _ in want:
        assert window_thumb_path(tmp_path / "thumbs", a["content_hash"], st0).exists()
    vm = st.load_matrix(prof.key, None, ["video-visual"])
    assert len(vm.asset_ids) == 12 and np.isfinite(vm.matrix).all()
    r2 = run_image_import(JobContext(Job(id="j2", kind="import"), None), st, tmp_path / "thumbs", lib["id"],
                          None, embed_video=ev)
    assert r2["embedded"] == 0 and r2["unchanged"] == 3


def test_interrupted_video_is_never_partially_searchable_and_resumes(tmp_path):
    root = tmp_path / "vids"
    color_video(root / "v.mp4", colors=COLORS * 2)  # 40 s -> 9 windows
    st, prof, host, lib, ev = _env(tmp_path, root, ("video-visual", "video-audio"))
    ctx = JobContext(Job(id="j", kind="import"), None)
    orig, calls = host.use, []

    def use(*a, **k):
        calls.append(1)
        if len(calls) == 4:
            ctx.job._cancel.set()
        return orig(*a, **k)

    host.use = use
    with pytest.raises(JobCancelled):
        run_image_import(ctx, st, tmp_path / "thumbs", lib["id"], None, embed_video=ev)
    a = st.list_assets(lib["id"])[0]
    assert a["status"] == "pending" and st.get_asset_vectors(a["id"], prof.key) is None
    assert st.load_matrix(prof.key).asset_ids == []
    host.use = orig
    r = run_image_import(JobContext(Job(id="j2", kind="import"), None), st, tmp_path / "thumbs", lib["id"],
                         None, embed_video=ev)
    assert r["video_windows"] == 9 and st.get_asset(a["id"])["status"] == "indexed"


def test_broken_video_marked_failed(tmp_path):
    root = tmp_path / "vids"
    root.mkdir()
    (root / "broken.mp4").write_bytes(b"not a video at all")
    st, prof, host, lib, ev = _env(tmp_path, root)
    r = run_image_import(JobContext(Job(id="j", kind="import"), None), st, tmp_path / "t", lib["id"], None, embed_video=ev)
    assert r["failed"][0][0] == "broken.mp4" and st.list_assets(lib["id"])[0]["status"] == "failed"
