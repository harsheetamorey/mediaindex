"""Photos on a CPU (a PC without a GPU) use 140 vision tokens; libraries indexed the old way are redone on their own
(mocked embeddings)."""

from fastapi.testclient import TestClient

from conftest import make_image
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.model.backend import FakeBackend
from test_search import wait_job


def test_image_tokens_follow_the_device():
    assert Settings(device="cpu").resolved_image_tokens() == 140
    assert Settings(device="mps").resolved_image_tokens() == 280
    assert Settings(device="cuda").resolved_image_tokens() == 280
    assert Settings(device="cpu", image_tokens=280).resolved_image_tokens() == 280
    cpu, gpu = Settings(device="cpu", precision="float32"), Settings(device="cuda", precision="float32")
    assert cpu.index_profile().image_max_soft_tokens == 140
    assert cpu.index_profile().key != gpu.index_profile().key  # different vectors: never ranked together


def test_library_from_the_old_profile_is_reindexed_at_startup(tmp_path):
    root = tmp_path / "photos"
    for i, colour in enumerate([(200, 0, 0), (0, 200, 0), (0, 0, 200)]):
        make_image(root / f"p{i}.jpg", colour)
    data = tmp_path / "data"

    def app(tokens):
        return create_app(Settings(data_dir=data, device="cpu", precision="float32", image_tokens=tokens,
                                   watch_interval=0), backend_factory=FakeBackend)

    with TestClient(app(280)) as c:  # 1.1.0 on a PC
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        assert wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])["result"]["embedded"] == 3
        assert c.app.state.store.libraries_to_reindex(c.app.state.profile.key) == []

    new = app(140)
    assert new.state.store.libraries_to_reindex(new.state.profile.key) == [lib["id"]]
    with TestClient(new) as c:
        job = next(j for j in c.get("/api/jobs").json() if j["kind"] == "import")
        assert wait_job(c, job["id"])["result"]["embedded"] == 3
        assert c.app.state.store.libraries_to_reindex(c.app.state.profile.key) == []
        hits = c.post("/api/search/text", json={"text": "red"}).json()["results"]
        assert len(hits) == 3
