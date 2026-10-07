from fastapi.testclient import TestClient

from mediaindex.app import create_app
from mediaindex.config import Settings


def make_client(tmp_path):
    return TestClient(create_app(Settings(data_dir=tmp_path)))


def test_health(tmp_path):
    r = make_client(tmp_path).get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_rejects_foreign_host(tmp_path):
    r = make_client(tmp_path).get("/api/health", headers={"host": "evil.example.com"})
    assert r.status_code == 403


def test_rejects_foreign_origin_on_mutation(tmp_path):
    r = make_client(tmp_path).post("/api/health", headers={"origin": "https://evil.example.com"})
    assert r.status_code == 403


def test_single_instance_lock(tmp_path):
    import pytest

    from mediaindex.instance import AlreadyRunning, acquire_instance_lock

    first = acquire_instance_lock(tmp_path)
    with pytest.raises(AlreadyRunning):
        acquire_instance_lock(tmp_path)
    first.close()
    acquire_instance_lock(tmp_path).close()  # released -> can be re-acquired


def test_body_size_limit_rejects_before_parsing(tmp_path):
    from mediaindex.app import create_app as _create
    from mediaindex.config import Settings as _S

    s = _S(data_dir=tmp_path)
    s.max_request_bytes = 1000
    c = TestClient(_create(s))
    r = c.post("/api/search/image", files={"file": ("q.jpg", b"x" * 5000, "image/jpeg")})
    assert r.status_code == 413
    r = c.post("/api/libraries", content=b"x" * 5000, headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert c.get("/api/health").status_code == 200


def test_cors_only_allows_local_ui_origins(tmp_path):
    c = make_client(tmp_path)
    r = c.options("/api/libraries", headers={"origin": "https://evil.example.com", "access-control-request-method": "POST"})
    assert "access-control-allow-origin" not in r.headers
    r = c.options("/api/libraries", headers={"origin": "http://127.0.0.1:5173", "access-control-request-method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
    r = c.post("/api/libraries", json={"path": "/tmp"}, headers={"origin": "null"})
    assert r.status_code == 403
