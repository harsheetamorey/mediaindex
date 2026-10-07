import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import make_image, tree_digest
from mediaindex import uploads
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.model.backend import FakeBackend
from test_search import wait_job


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "lib"
    for i in range(5):
        make_image(root / f"img{i}.jpg", (i * 40, 100, 200 - i * 30))
    make_image(root / "copy/img0-dup.jpg", (0, 100, 200))
    settings = Settings(data_dir=tmp_path / "data", precision="float32")
    app = create_app(settings, backend_factory=FakeBackend)
    with TestClient(app) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        assets = {a["rel_path"]: a for a in c.get(f"/api/libraries/{lib['id']}/assets").json()}
        yield c, root, settings, assets, tree_digest(root)


def jpeg_bytes(color):
    b = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(b, "JPEG")
    return b.getvalue()


def test_asset_reference_excludes_self_and_duplicates_by_default(client):
    c, root, _, assets, before = client
    ref = assets["img0.jpg"]
    r = c.post("/api/search/image", data={"asset_id": ref["id"]}).json()
    ids = [x["asset"]["id"] for x in r["results"]]
    assert ref["id"] not in ids and assets["copy/img0-dup.jpg"]["id"] not in ids
    assert len(ids) == 4 and r["query"]["embedding"] == "stored library vector"
    r = c.post("/api/search/image", data={"asset_id": ref["id"], "include_identical": "true"}).json()
    ids = [x["asset"]["id"] for x in r["results"]]
    assert ref["id"] in ids[:2] and assets["copy/img0-dup.jpg"]["id"] in ids[:2]
    assert tree_digest(root) == before


def test_upload_reference_identical_content_excluded_and_cleaned(client):
    c, root, settings, assets, before = client
    data = (root / "img2.jpg").read_bytes()
    r = c.post("/api/search/image", files={"file": ("q.jpg", data, "image/jpeg")}).json()
    assert assets["img2.jpg"]["id"] not in [x["asset"]["id"] for x in r["results"]]
    assert r["query"]["reference"] == "upload" and r["query"]["excluded_identical"] == 1
    r = c.post("/api/search/image", files={"file": ("q.jpg", jpeg_bytes((5, 5, 5)), "image/jpeg")}).json()
    assert len(r["results"]) == 6
    assert list(settings.uploads_dir.iterdir()) == []
    assert tree_digest(root) == before


def test_reference_errors(client):
    c, _, settings, assets, _ = client
    assert c.post("/api/search/image", data={}).status_code == 422
    assert c.post("/api/search/image", data={"asset_id": assets["img0.jpg"]["id"]},
                  files={"file": ("q.jpg", jpeg_bytes("red"), "image/jpeg")}).status_code == 422
    assert c.post("/api/search/image", files={"file": ("q.gif", b"GIF89a", "image/gif")}).status_code == 415
    assert c.post("/api/search/image", files={"file": ("q.jpg", b"not an image", "image/jpeg")}).status_code == 422
    assert c.post("/api/search/image", files={"file": ("q.jpg", b"", "image/jpeg")}).status_code == 400
    assert c.post("/api/search/image", data={"asset_id": "nope"}).status_code == 404
    big = b"\xff\xd8" + b"0" * (26 * 1024 * 1024)
    assert c.post("/api/search/image", files={"file": ("q.jpg", big, "image/jpeg")}).status_code == 413
    # path-like filenames are never used as paths
    r = c.post("/api/search/image", files={"file": ("../../etc/evil.jpg", jpeg_bytes("blue"), "image/jpeg")})
    assert r.status_code == 200
    assert list(settings.uploads_dir.iterdir()) == []


def test_stale_upload_cleanup(tmp_path):
    d = tmp_path / "u"
    d.mkdir()
    (d / "x.jpg").write_bytes(b"1")
    assert uploads.cleanup_stale_uploads(d, max_age=0) == 1
    assert not any(d.iterdir())


def test_mixed_query_native_and_validated(client):
    c, root, settings, assets, before = client
    ref = assets["img1.jpg"]
    r = c.post("/api/search/image-text", data={"asset_id": ref["id"], "text": "at night"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "image+text" and "native" in body["query"]["embedding"]
    assert "not an average" in body["query"]["interface"]
    assert ref["id"] not in [x["asset"]["id"] for x in body["results"]]
    # differs from image-only query (FakeBackend hashes image+text jointly)
    img_only = c.post("/api/search/image", data={"asset_id": ref["id"]}).json()
    assert [x["similarity"] for x in body["results"]] != [x["similarity"] for x in img_only["results"]]
    assert c.post("/api/search/image-text", data={"asset_id": ref["id"], "text": "   "}).status_code == 422
    assert c.post("/api/search/image-text", data={"asset_id": ref["id"]}).status_code == 422
    assert c.post("/api/search/image-text", data={"text": "x"}).status_code == 422
    r = c.post("/api/search/image-text", data={"text": "red"}, files={"file": ("q.jpg", jpeg_bytes("red"), "image/jpeg")})
    assert r.status_code == 200 and r.json()["query"]["reference"] == "upload"
    assert list(settings.uploads_dir.iterdir()) == []
