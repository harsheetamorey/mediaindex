"""Worker/host/profile behaviour with a mocked backend (no real model)."""

import threading
import time

import numpy as np
import pytest
from PIL import Image

from mediaindex.jobs import JobRunner, QueueFull
from mediaindex.model.backend import FakeBackend
from mediaindex.model.host import ModelBusy, ModelHost
from mediaindex.model.profiles import IndexProfile, ProfileMismatch, ensure_compatible


def wait_for(cond, timeout=5.0):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise AssertionError("timeout")
        time.sleep(0.01)


def test_host_loads_once_across_repeated_and_concurrent_requests():
    loads = []

    def factory(p):
        loads.append(1)
        time.sleep(0.05)
        return FakeBackend(p)

    host = ModelHost(IndexProfile(), factory)
    results = []

    def worker():
        with host.use() as b:
            results.append(b.embed_query_texts(["hello"]))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(loads) == 1 and host.load_count == 1
    assert len(results) == 8
    assert all(np.allclose(r, results[0]) for r in results)


def test_inference_is_serialized_and_busy_errors():
    host = ModelHost(IndexProfile(), FakeBackend)
    entered = threading.Event()
    release = threading.Event()
    active = []

    def long_user():
        with host.use():
            active.append(1)
            entered.set()
            release.wait(2)
            active.pop()

    t = threading.Thread(target=long_user)
    t.start()
    entered.wait(2)
    with pytest.raises(ModelBusy):
        with host.use(timeout=0.05):
            pass
    release.set()
    t.join()
    with host.use(timeout=1) as b:
        assert b.embed_images([Image.new("RGB", (8, 8), "red")]).shape == (1, 768)


def test_profile_key_is_stable_and_mismatch_rejected():
    a = IndexProfile()
    assert a.key == IndexProfile().key
    assert IndexProfile.from_dict(a.to_dict()).key == a.key
    ensure_compatible(a, IndexProfile())
    for other in (IndexProfile(precision="float32"), IndexProfile(revision="deadbeef"), IndexProfile(dim=256),
                  IndexProfile(image_max_soft_tokens=560), IndexProfile(query_prompt="x")):
        with pytest.raises(ProfileMismatch):
            ensure_compatible(a, other)


def test_float16_rejected():
    with pytest.raises(ValueError):
        IndexProfile(precision="float16")


def test_job_runner_progress_and_cancel():
    runner = JobRunner(maxsize=2)
    gate = threading.Event()

    def slow(ctx):
        for i in range(100):
            ctx.check_cancelled()
            ctx.progress(i + 1, 100)
            gate.wait(0.01)
        return {"n": 100}

    job = runner.submit("slow", slow)
    wait_for(lambda: job.done > 2)
    runner.cancel(job.id)
    wait_for(lambda: job.status == "cancelled")
    assert job.done < 100

    ok = runner.submit("ok", lambda ctx: {"x": 1})
    wait_for(lambda: ok.status == "done")
    assert ok.result == {"x": 1}

    bad = runner.submit("bad", lambda ctx: 1 / 0)
    wait_for(lambda: bad.status == "failed")
    assert "ZeroDivisionError" in bad.error
    runner.shutdown()


def test_job_queue_bounded():
    runner = JobRunner(maxsize=1)
    hold = threading.Event()
    runner.submit("hold", lambda ctx: hold.wait(2))
    time.sleep(0.05)  # first job now running; queue empty
    runner.submit("queued", lambda ctx: None)
    with pytest.raises(QueueFull):
        runner.submit("overflow", lambda ctx: None)
    hold.set()
    runner.shutdown()


def test_model_status_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from mediaindex.app import create_app
    from mediaindex.config import Settings

    app = create_app(Settings(data_dir=tmp_path, precision="float32"), backend_factory=FakeBackend)
    with TestClient(app) as c:
        s = c.get("/api/model/status").json()
        assert s["loaded"] is False and s["profile"]["precision"] == "float32"
        assert c.get("/api/jobs/nope").status_code == 404
