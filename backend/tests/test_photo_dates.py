"""When and where photos were taken: EXIF dates and GPS, offline place names, Search date filter, Ask time/place
phrases (mocked embeddings and detector)."""

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import tree_digest
from mediaindex.app import create_app
from mediaindex.config import Settings
from mediaindex.db import Database
from mediaindex.model.backend import FakeBackend
from mediaindex.photofilter import parse_filter
from mediaindex.places import place_name
from mediaindex.store import Store
from test_ask import FakeDetector, ask
from test_search import wait_job

GOA = (15.5439, 73.7553)  # Calangute beach
PARIS = (48.8584, 2.2945)  # Eiffel Tower


def photo(path: Path, colour, taken: str | None = None, gps: tuple[float, float] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if taken:
        exif.get_ifd(0x8769)[0x9003] = taken
    if gps:
        g = exif.get_ifd(0x8825)
        for i, v in ((1, gps[0]), (3, gps[1])):
            g[i] = ("N" if v >= 0 else "S") if i == 1 else ("E" if v >= 0 else "W")
            a = abs(v)
            g[i + 1] = (int(a), int(a * 60) % 60, round(a * 3600 % 60, 4))
    Image.new("RGB", (64, 48), colour).save(path, "JPEG", exif=exif)
    return path


@pytest.fixture
def trip(tmp_path):
    root = tmp_path / "Trips"
    photo(root / "goa-dog.jpg", (200, 0, 0), "2024:07:14 18:03:22", GOA)
    photo(root / "goa-cat.jpg", (0, 200, 0), "2024:12:24 09:00:00", GOA)
    photo(root / "paris-dog.jpg", (200, 0, 0), "2023:12:02 12:00:00", PARIS)
    photo(root / "scan.jpg", (0, 0, 200))  # no EXIF at all
    app = create_app(Settings(data_dir=tmp_path / "data", precision="float32", watch_interval=0),
                     backend_factory=FakeBackend, detector_factory=FakeDetector, chat=None)
    app.state.assistant.today = lambda: date(2026, 10, 20)
    with TestClient(app) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        wait_job(c, c.post("/api/ask/prepare", json={}).json()["id"])
        yield c, root, lib


def by_name(c, lib) -> dict:
    return {a["rel_path"]: a for a in c.get(f"/api/libraries/{lib['id']}/assets").json()}


def names(results) -> list[str]:
    return [r["rel_path"] for r in results]


def test_exif_date_and_offline_place_are_stored(trip):
    c, root, lib = trip
    a = by_name(c, lib)
    assert a["goa-dog.jpg"]["taken_at"] == "2024-07-14T18:03:22"
    assert a["goa-dog.jpg"]["place"] == "Calangute, Goa, India"
    assert a["paris-dog.jpg"]["place"] == "Paris, Île-de-France, France"
    assert a["scan.jpg"]["taken_at"] is None and a["scan.jpg"]["place"] is None
    assert place_name(0.0, -30.0) is None  # mid-Atlantic: no town within reach


def test_photos_imported_before_dates_were_stored_get_them(tmp_path):
    root = tmp_path / "Old"
    photo(root / "a.jpg", (200, 0, 0), "2021:05:01 10:00:00", PARIS)
    settings = Settings(data_dir=tmp_path / "data", precision="float32", watch_interval=0)
    with TestClient(create_app(settings, backend_factory=FakeBackend)) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
    db = Database(settings.db_path)  # as if imported by an older version
    with db.tx() as x:
        x.execute("UPDATE assets SET taken_at=NULL, place=NULL, photo_meta_read=0")
    assert len(Store(db).photos_without_meta()) == 1
    db.close()
    before = tree_digest(root)
    with TestClient(create_app(settings, backend_factory=FakeBackend)) as c:  # read once at startup
        job = next(j for j in c.get("/api/jobs").json() if j["kind"] == "read-photo-dates")
        assert wait_job(c, job["id"])["result"] == {"photos": 1}
        a = c.get(f"/api/libraries/{lib['id']}/assets").json()[0]
        assert a["taken_at"] == "2021-05-01T10:00:00" and a["place"].startswith("Paris")
    assert tree_digest(root) == before


def test_search_date_range(trip):
    c, root, lib = trip
    r = c.post("/api/search/text", json={"text": "a photo", "date_from": "2024-01-01", "date_to": "2024-12-31"}).json()
    assert sorted(x["asset"]["rel_path"] for x in r["results"]) == ["goa-cat.jpg", "goa-dog.jpg"]
    assert r["query"]["taken"]["photos"] == 2 and r["query"]["taken"]["undated_photos"] == 1
    r = c.post("/api/search/text", json={"text": "a photo", "date_to": "2023-12-02"}).json()
    assert [x["asset"]["rel_path"] for x in r["results"]] == ["paris-dog.jpg"]  # the end day is included
    r = c.post("/api/search/text", json={"text": "a photo"}).json()
    assert len(r["results"]) == 4 and "taken" not in r["query"]
    assert c.post("/api/search/text", json={"text": "x", "date_from": "2025-01-02", "date_to": "2025-01-01"}).status_code == 422
    assert c.post("/api/search/text", json={"text": "x", "date_from": "last year"}).status_code == 422
    paris = by_name(c, lib)["paris-dog.jpg"]["id"]
    r = c.post("/api/search/image", data={"asset_id": paris, "date_from": "2024-08-01"}).json()
    assert [x["asset"]["rel_path"] for x in r["results"]] == ["goa-cat.jpg"]


def test_ask_photos_from_a_time(trip):
    c, *_ = trip
    r = ask(c, "photos from 2024")
    assert r["action"] == "search" and names(r["results"]) == ["goa-dog.jpg", "goa-cat.jpg"]  # oldest first
    assert r["answer"] == "2 photos from 2024, oldest first. 1 photo has no date, so it isn’t included."
    r = ask(c, "photos from last December")
    assert r["results"] == [] and r["answer"].startswith("I found no photos from December 2025.")
    r = ask(c, "show me everything from December")  # any year
    assert names(r["results"]) == ["paris-dog.jpg", "goa-cat.jpg"] and "December (any year)" in r["answer"]
    r = ask(c, "photos from 2019")
    assert r["results"] == [] and r["answer"].startswith("I found no photos from 2019.")


def test_ask_counts_and_searches_within_a_time_or_place(trip):
    c, *_ = trip
    r = ask(c, "how many dogs in 2024?")
    assert r["action"] == "count" and r["answer"].startswith("Dogs appear in 1 of 2 photos from 2024 (2 dogs counted).")
    assert "1 photo has no date" in r["answer"]
    r = ask(c, "how many photos from Goa?")
    assert r["answer"].startswith("You have 2 photos in Goa.") and "1 photo has no location" in r["answer"]
    r = ask(c, "everything from Paris")
    assert names(r["results"]) == ["paris-dog.jpg"]
    r = ask(c, "dog photos in Goa")
    assert r["action"] == "search" and names(r["results"])[0] == "goa-dog.jpg"
    assert set(names(r["results"])) <= {"goa-dog.jpg", "goa-cat.jpg"} and "I only looked at the 2 photos in Goa." in r["answer"]
    r = ask(c, "what's in my photos from 2023?")
    assert names(r["results"]) == ["paris-dog.jpg"]
    hist = [{"role": "user", "text": "show me some photos"},
            {"role": "assistant", "text": "…", "action": "search",
             "asset_ids": [a["id"] for a in ask(c, "show me some photos")["results"]]}]
    r = ask(c, "which of those are from 2024?", hist)
    assert r["action"] == "refine" and sorted(names(r["results"])) == ["goa-cat.jpg", "goa-dog.jpg"]
    assert r["answer"].startswith("2 of those 4 photos are from 2024.")


@pytest.mark.parametrize("text, rest, when, ranges, months", [
    ("beach photos from 2024", "beach photos", "from 2024", [("2024-01-01", "2025-01-01")], ()),
    ("photos from last December", "photos", "from December 2025", [("2025-12-01", "2026-01-01")], ()),
    ("last summer", "", "from summer 2026 (June–August)", [("2026-06-01", "2026-09-01")], ()),
    ("beach in the summer", "beach", "from summer (any year)", [], (6, 7, 8)),
    ("cats from 2019-2021", "cats", "from 2019 to 2021", [("2019-01-01", "2022-01-01")], ()),
    ("dogs in may 2024", "dogs", "from May 2024", [("2024-05-01", "2024-06-01")], ()),
    ("since March 2023", "", "since March 2023", [("2023-03-01", "2026-10-21")], ()),
    ("what did I shoot yesterday?", "what did I shoot?", "from yesterday", [("2026-10-19", "2026-10-20")], ()),
    ("cats from last month", "cats", "from last month (September 2026)", [("2026-09-01", "2026-10-01")], ()),
    ("winter 2024", "", "from winter 2024–2025 (December–February)", [("2024-12-01", "2025-03-01")], ()),
])
def test_time_phrases(text, rest, when, ranges, months):
    f, r = parse_filter(text, date(2026, 10, 20), [])
    assert (r, f.when, f.ranges, f.months) == (rest, when, ranges, months)


@pytest.mark.parametrize("text", ["may I see dogs", "the march of penguins", "photos of a forest",
                                  "remove the car from photo 2", "show me the sea in the evening"])
def test_not_time_or_place(text):
    assert parse_filter(text, date(2026, 10, 20), ["Calangute, Goa, India"]) == (None, text)


def test_place_phrases():
    known = ["Calangute, Goa, India", "Panjim, Goa, India", "São Paulo, Brazil", "New York City, New York, United States"]
    f, rest = parse_filter("everything from Goa", date(2026, 10, 20), known)
    assert rest == "everything" and f.where == "Goa" and f.places == known[:2]
    f, rest = parse_filter("photos in sao paulo from 2024", date(2026, 10, 20), known)
    assert rest == "photos" and f.places == ["São Paulo, Brazil"] and f.describe() == "from 2024 in São Paulo"
    f, rest = parse_filter("New York sunsets", date(2026, 10, 20), known)
    assert rest == "sunsets" and f.where == "New York"
    assert parse_filter("photos in Lisbon", date(2026, 10, 20), known)[0] is None  # not where any photo was taken


def test_overview_of_a_place_shows_its_photos(tmp_path, trip):
    c, *_ = trip
    from test_ask import FakeChat
    c.app.state.assistant.chat = FakeChat(['{"action": "summary", "subject": "", "item": 0}'])
    r = ask(c, "what did I take in Goa?")
    assert r["action"] == "summary" and r["answer"].startswith("You have 2 photos in Goa. Most common in them:")
    assert names(r["results"]) == ["goa-dog.jpg", "goa-cat.jpg"] and "1 photo has no location" in r["answer"]


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")
def test_videos_get_dates_and_places_too(tmp_path):
    import subprocess

    from mediaindex.media.video import video_date

    root = tmp_path / "Phone"
    photo(root / "IMG_0001.jpg", (200, 0, 0), "2024:07:14 18:00:00", GOA)
    root.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=10:d=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "use_metadata_tags",
                    "-metadata", "com.apple.quicktime.creationdate=2024-07-15T09:30:00+0530",
                    "-metadata", "com.apple.quicktime.location.ISO6709=+15.5439+073.7553+010.000/",
                    str(root / "IMG_0002.MOV")], check=True)
    app = create_app(Settings(data_dir=tmp_path / "data", precision="float32", watch_interval=0),
                     backend_factory=FakeBackend, detector_factory=FakeDetector, chat=None)
    with TestClient(app) as c:
        lib = c.post("/api/libraries", json={"path": str(root)}).json()
        wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
        v = by_name(c, lib)["IMG_0002.MOV"]
        assert v["taken_at"] == "2024-07-15T09:30:00" and v["place"] == "Calangute, Goa, India"
        r = c.post("/api/search/text", json={"text": "blue", "media_types": ["video"], "date_from": "2024-07-15"}).json()
        assert [x["asset"]["rel_path"] for x in r["results"]] == ["IMG_0002.MOV"]
        r = c.post("/api/search/text", json={"text": "blue", "media_types": ["video"], "date_to": "2024-07-14"}).json()
        assert r["results"] == []
        r = ask(c, "everything from July 2024")
        assert names(r["results"]) == ["IMG_0001.jpg", "IMG_0002.MOV"]
        assert r["answer"].startswith("1 photo and 1 video from July 2024, oldest first.")
    assert video_date({"creation_time": "1904-01-01T00:00:00.000000Z"}) is None  # the container's zero date
    assert video_date({"creation_time": "2024-07-14T12:33:22.000000Z"})[:4] == "2024"  # UTC, shown in local time


def test_search_by_place(trip):
    c, root, lib = trip
    levels = c.get("/api/places").json()
    assert [(p["name"], p["count"], p["depth"]) for p in levels] == [
        ("France", 1, 0), ("Île-de-France, France", 1, 1), ("Paris, Île-de-France, France", 1, 2),
        ("India", 2, 0), ("Goa, India", 2, 1), ("Calangute, Goa, India", 2, 2)]
    r = c.post("/api/search/text", json={"text": "a photo", "place": "Goa, India"}).json()
    assert sorted(x["asset"]["rel_path"] for x in r["results"]) == ["goa-cat.jpg", "goa-dog.jpg"]
    assert r["query"]["taken"]["unplaced_photos"] == 1 and r["query"]["taken"]["undated_photos"] == 0
    r = c.post("/api/search/text", json={"text": "a photo", "place": "India", "date_from": "2024-12-01"}).json()
    assert [x["asset"]["rel_path"] for x in r["results"]] == ["goa-cat.jpg"]
    assert c.post("/api/search/text", json={"text": "a photo", "place": "Lisbon"}).json()["results"] == []
    assert c.post("/api/search/text", json={"text": "a photo", "place": "India, Goa"}).json()["results"] == []
    paris = by_name(c, lib)["paris-dog.jpg"]["id"]
    r = c.post("/api/search/image", data={"asset_id": paris, "place": "France", "include_identical": "true"}).json()
    assert [x["asset"]["rel_path"] for x in r["results"]] == ["paris-dog.jpg"]
