import shutil

import pytest

from test_audio_search import client  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_text_across_media_grouped_by_modality(client):
    c, root, _, assets, lib = client
    r = c.post("/api/search/text", json={"text": "x", "media_types": ["image", "audio"], "limit": 10}).json()
    assert r["grouping"] == "by_modality" and r["groups"] == {"image": 1, "audio": 4}
    assert [x["modality"] for x in r["results"]] == ["image"] + ["audio"] * 4
    assert set(r["index_state"]) == {"image", "audio"}
    g = c.post("/api/search/text", json={"text": "x", "media_types": ["image", "audio"], "limit": 3,
                                         "global_rank": True}).json()
    assert g["grouping"] == "global (experimental)" and "EXPERIMENTAL" in g["note"] and len(g["results"]) == 3
    sims = [x["similarity"] for x in g["results"]]
    assert sims == sorted(sims, reverse=True)


def test_image_to_audio_and_audio_to_image(client):
    c, root, _, assets, lib = client
    r = c.post("/api/search/image", data={"asset_id": assets["pic.jpg"]["id"], "target": "audio"}).json()
    assert r["mode"] == "image→audio" and r["query"]["experimental"] is True
    assert {x["modality"] for x in r["results"]} == {"audio"}
    r = c.post("/api/search/audio", data={"asset_id": assets["a.flac"]["id"], "target": "image"}).json()
    assert r["mode"] == "audio→image" and [x["asset"]["rel_path"] for x in r["results"]] == ["pic.jpg"]
    r = c.post("/api/search/image-text", data={"asset_id": assets["pic.jpg"]["id"], "text": "loud", "target": "audio"}).json()
    assert r["mode"] == "image+text→audio" and r["query"]["mode_status"] == "experimental"


def test_audio_plus_text_native(client):
    c, root, _, assets, lib = client
    r = c.post("/api/search/audio", data={"asset_id": assets["long.wav"]["id"], "start_s": "11", "text": "in a cave"}).json()
    assert r["mode"] == "audio+text" and "native audio+text" in r["query"]["embedding"]
    assert r["query"]["reference_span"] == [10.0, 20.0]
    r2 = c.post("/api/search/audio", data={"asset_id": assets["long.wav"]["id"], "start_s": "11"}).json()
    assert [x["similarity"] for x in r["results"]] != [x["similarity"] for x in r2["results"]]
    r = c.post("/api/search/audio", files={"file": ("q.flac", (root / "b.mp3").read_bytes(), "audio/flac")},
               data={"text": "x", "target": "image"}).json()
    assert r["mode"] == "audio+text→image"


def test_capabilities_lists_modes_with_status(client):
    c, *_ = client
    caps = c.get("/api/capabilities").json()
    modes = {(m["query"], m["target"]): m for m in caps["modes"]}
    assert modes[("image", "audio")]["status"] == "experimental" and modes[("image", "audio")]["available"]
    assert modes[("text", "image")]["status"] == "verified"
    assert any("not calibrated" in n for n in caps["notes"])
