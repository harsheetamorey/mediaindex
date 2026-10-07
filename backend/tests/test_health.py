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
