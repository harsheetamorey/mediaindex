"""Ask assistant: object counts, routing, follow-ups, chat-model guard rails (mocked detector, embeddings and chat)."""

import json

import pytest
from fastapi.testclient import TestClient

from conftest import make_image, tree_digest
from mediaindex.app import create_app
from mediaindex.assistant import resolve_labels, rule_plan
from mediaindex.config import Settings
from mediaindex.detect import decide_counts
from mediaindex.llm import ChatUnavailable, OllamaChat, require_loopback
from mediaindex.model.backend import FakeBackend
from test_search import wait_job

# The fake detector "sees" objects by colour (box scores): red = 2 dogs, green = 1 cat + 1 person,
# blue = only an unsure dog box (dropped unless the chat model confirms it).
B = [0.1, 0.1, 0.5, 0.5]
COLOURS = {(200, 0, 0): {"dog": [(0.95, B), (0.9, [0.6, 0.6, 0.9, 0.9])]},
           (0, 200, 0): {"cat": [(0.9, B)], "person": [(0.8, B), (0.4, B)]},
           (0, 0, 200): {"dog": [(0.6, [0.9, 0.9, 0.95, 0.95])]}}


class FakeDetector:
    calls = 0

    def detect(self, image):
        FakeDetector.calls += 1
        px = image.convert("RGB").getpixel((1, 1))
        return next(v for c, v in COLOURS.items() if all(abs(a - b) < 40 for a, b in zip(c, px)))

    def close(self):
        pass


class FakeChat:
    """Records requests; replies from a queue (router JSON or text)."""

    def __init__(self, replies=(), available=True):
        self.replies = list(replies)
        self.requests = []
        self.available = available

    def status(self):
        return {"available": self.available, "model": "fake"}

    def chat(self, messages, *, schema=None, max_tokens=200):
        self.requests.append({"messages": messages, "schema": schema})
        if not self.replies:
            raise ChatUnavailable("no reply queued")
        return self.replies.pop(0)


def make_client(tmp_path, chat=None):
    root = tmp_path / "Pets"
    make_image(root / "red1.jpg", (200, 0, 0))
    make_image(root / "red2.jpg", (200, 0, 0), size=(60, 40))
    make_image(root / "green.jpg", (0, 200, 0))
    make_image(root / "blue.jpg", (0, 0, 200))
    app = create_app(Settings(data_dir=tmp_path / "data", precision="float32", watch_interval=0),
                     backend_factory=FakeBackend, detector_factory=FakeDetector, chat=chat)
    c = TestClient(app)
    c.__enter__()
    lib = c.post("/api/libraries", json={"path": str(root)}).json()
    wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
    return c, root, lib


@pytest.fixture
def client(tmp_path):
    c, root, lib = make_client(tmp_path)
    yield c, root, lib
    c.__exit__(None, None, None)


def ask(c, message, history=(), **kw):
    r = c.post("/api/ask", json={"message": message, "history": list(history), **kw})
    assert r.status_code == 200, r.text
    return r.json()


def turn(r):
    return {"role": "assistant", "text": r["answer"], "asset_ids": r["asset_ids"], "action": r["action"]}


def test_labels_and_rules():
    assert resolve_labels("puppies") == ["dog"]
    assert resolve_labels("cats or dogs") == ["cat", "dog"]
    assert resolve_labels("traffic lights and people") == ["traffic light", "person"]
    assert resolve_labels("sunsets") == []
    assert rule_plan("How many dogs do I have?", False).action == "count"
    assert rule_plan("do I have more cats or dogs?", False).action == "count"
    assert rule_plan("show me the beach", False).action == "search"
    assert rule_plan("which of those have people?", True).action == "refine"
    assert rule_plan("which of those have people?", False) is None  # nothing to narrow down
    assert rule_plan("describe the second one", True).item == 2
    assert rule_plan("what's in my library?", True) is None  # not a photo reference: left to the chat model
    assert rule_plan("tell me a joke", False) is None


def test_count_before_and_after_checking(client):
    c, root, lib = client
    before = tree_digest(root)
    r = ask(c, "how many dogs?")
    assert r["action"] == "count" and "haven't been checked" in r["answer"] and r["results"] == []
    st = c.get("/api/ask/status").json()
    assert st["detector"]["coverage"] == {"photos": 4, "checked": 0}
    job = wait_job(c, c.post("/api/ask/prepare", json={}).json()["id"])
    assert job["status"] == "done" and job["result"]["checked"] == 4
    assert c.get("/api/ask/status").json()["detector"]["coverage"] == {"photos": 4, "checked": 4}
    r = ask(c, "How many puppies do I have?")
    assert r["answer"].startswith("Dogs appear in 2 of 4 photos (4 dogs counted).")
    assert sorted(a["rel_path"] for a in r["results"]) == ["red1.jpg", "red2.jpg"]
    r = ask(c, "do I have more cats or dogs?")
    assert "Cats appear in 1 of 4 photos (1 cat counted)." in r["answer"] and "Dogs appear in 2" in r["answer"]
    assert "didn't find any cars" in ask(c, "how many cars")["answer"]
    r = ask(c, "how many sunsets?")  # not something the detector knows: falls back to search, says so
    assert "not “sunsets”" in r["answer"] and len(r["results"]) == 4
    assert tree_digest(root) == before  # originals untouched


def test_counting_skips_checked_photos(client):
    c, *_ = client
    wait_job(c, c.post("/api/ask/prepare", json={}).json()["id"])
    n = FakeDetector.calls
    job = wait_job(c, c.post("/api/ask/prepare", json={}).json()["id"])
    assert FakeDetector.calls == n and job["result"]["checked"] == 0 and job["result"]["already_checked"] == 4


def test_refine_and_describe_without_chat_model(client):
    c, *_ = client
    wait_job(c, c.post("/api/ask/prepare", json={}).json()["id"])
    hist = []
    r = ask(c, "show me some photos", hist)
    assert r["action"] == "search" and len(r["results"]) == 4
    hist += [{"role": "user", "text": "show me some photos"}, turn(r)]
    r2 = ask(c, "which of those have people?", hist)
    assert r2["action"] == "refine" and r2["answer"] == "1 of those 4 photos has people."
    assert [a["rel_path"] for a in r2["results"]] == ["green.jpg"]
    r3 = ask(c, "describe the first one", hist)
    assert r3["action"] == "describe" and r3["asset_ids"] == [r["asset_ids"][0]]
    assert "Turn on the local chat model" in r3["answer"]
    hist += [{"role": "user", "text": "describe the first one"}, turn(r3)]
    r4 = ask(c, "which of those have dogs?", hist)  # narrows the last list, not the described photo
    assert r4["answer"] == "2 of those 4 photos have dogs."
    assert "pick a number from 1 to 4" in ask(c, "describe photo 9", hist)["answer"]
    help_ = ask(c, "hi")
    assert help_["action"] == "chat" and "how many dogs" in help_["answer"] and "Dogs (1 photos)" not in help_["answer"]
    assert c.post("/api/ask", json={"message": "x", "library_ids": ["nope"]}).status_code == 404
    assert c.post("/api/ask", json={"message": "x", "asset_id": "nope"}).status_code == 404


def test_chat_model_routes_and_describes(tmp_path):
    chat = FakeChat([json.dumps({"action": "search", "subject": "red things", "item": 0}),
                     "A plain red picture."])
    c, root, lib = make_client(tmp_path, chat)
    try:
        r = ask(c, "got anything reddish?")
        assert r["action"] == "search" and r["routed_by"] == "chat model" and "“red things”" in r["answer"]
        assert chat.requests[0]["schema"]["properties"]["action"]["enum"]
        r2 = ask(c, "what is going on here?", asset_id=r["asset_ids"][0])
        assert r2["answer"] == "A plain red picture." and r2["described_by"] == "chat model"
        img = chat.requests[1]["messages"][-1]["images"][0]
        assert isinstance(img, str) and len(img) > 100  # the photo itself, base64, sent to the local model only
    finally:
        c.__exit__(None, None, None)


def test_chat_reply_with_invented_numbers_is_replaced(tmp_path):
    chat = FakeChat([json.dumps({"present": False}),  # counting: the unsure dog box is not confirmed
                     json.dumps({"action": "chat", "subject": "", "item": 0}), "You have 7 dogs and 99 cats!",
                     json.dumps({"action": "chat", "subject": "", "item": 0}), "You have 4 photos."])
    c, *_ = make_client(tmp_path, chat)
    try:
        wait_job(c, c.post("/api/ask/prepare", json={}).json()["id"])
        r = ask(c, "tell me about my library")
        assert "99" not in r["answer"] and "numbers not in the facts" in r["note"]
        assert ask(c, "tell me about my library again")["answer"] == "You have 4 photos."
    finally:
        c.__exit__(None, None, None)


def test_bad_router_output_falls_back_to_search(tmp_path):
    chat = FakeChat(["not json at all"])
    c, *_ = make_client(tmp_path, chat)
    try:
        r = ask(c, "something odd")
        assert r["action"] == "search" and r["routed_by"] == "rules"
    finally:
        c.__exit__(None, None, None)


def test_chat_model_must_be_local():
    assert require_loopback("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"
    assert require_loopback("http://localhost:11434")
    for bad in ("http://example.com:11434", "https://127.0.0.1:11434", "http://10.0.0.5:11434"):
        with pytest.raises(ValueError):
            OllamaChat(bad)
    st = OllamaChat("http://127.0.0.1:9").status()  # nothing listens there
    assert st["available"] is False and "not running" in st["reason"]


def test_counting_rule_for_animals():
    def n(kept):
        return {k: len(v) for k, v in kept[0].items()}, kept[1]
    found = {"dog": [(0.9, B), (0.6, B), (0.55, B)], "person": [(0.6, B), (0.45, B)], "bird": [(0.55, B)]}
    assert n(decide_counts(found)) == ({"dog": 1, "person": 1, "bird": 1}, [])  # unsure dogs dropped, people/birds kept
    assert n(decide_counts(found, None, lambda im, label: True)) == ({"dog": 3, "person": 1, "bird": 1}, ["dog"])
    assert n(decide_counts({"cat": [(0.52, B)]}, None, lambda im, label: False)) == ({}, ["cat"])


def test_unsure_animal_confirmed_by_chat_model(tmp_path):
    chat = FakeChat([json.dumps({"present": True})])
    c, *_ = make_client(tmp_path, chat)
    try:
        job = wait_job(c, c.post("/api/ask/prepare", json={}).json()["id"])
        assert job["result"]["chat_model_checks"] == 1  # only the blue photo's unsure dog was asked about
        assert "Is there a dog in this photo?" in chat.requests[0]["messages"][0]["content"]
        assert chat.requests[0]["messages"][0]["images"]
        assert ask(c, "how many dogs")["answer"].startswith("Dogs appear in 3 of 4 photos (5 dogs counted).")
    finally:
        c.__exit__(None, None, None)


def test_boxes_and_explaining_an_answer(client):
    c, *_ = client
    wait_job(c, c.post("/api/ask/prepare", json={}).json()["id"])
    hist = [{"role": "user", "text": "how many dogs?"}]
    r = ask(c, "how many dogs?")
    assert {len(v) for v in r["boxes"].values()} == {2} and set(r["boxes"]) == set(r["asset_ids"])
    assert all(b["label"] == "dog" and len(b["box"]) == 4 for v in r["boxes"].values() for b in v)
    hist.append(turn(r))
    r2 = ask(c, "describe the first one", hist)  # no chat model: lists objects, all boxes shown
    assert r2["boxes"][r2["asset_ids"][0]][0]["label"] == "dog"
    hist += [{"role": "user", "text": "describe the first one"}, turn(r2)]
    r3 = ask(c, "but that's not a dog, right?", hist)
    assert r3["action"] == "explain" and "found 2 dogs here, outlined" in r3["answer"] and "very sure" in r3["answer"]
    assert r3["asset_ids"] == r2["asset_ids"] and len(r3["boxes"][r3["asset_ids"][0]]) == 2
    r4 = ask(c, "where is the cat?", hist)  # no cat in that photo
    assert "didn't find a cat in this photo" in r4["answer"] and r4["boxes"] == {}
    hist2 = [{"role": "user", "text": "how many dogs?"}, turn(r)]
    r5 = ask(c, "why do you say those are dogs?", hist2)
    assert r5["action"] == "explain" and "outlined each dog" in r5["answer"] and len(r5["asset_ids"]) == 2


def test_describe_gets_the_detector_findings(tmp_path):
    chat = FakeChat([json.dumps({"present": False}), "A red picture with two dogs."])
    c, *_ = make_client(tmp_path, chat)
    try:
        wait_job(c, c.post("/api/ask/prepare", json={}).json()["id"])
        dog_photo = ask(c, "how many dogs")["asset_ids"][0]
        r = ask(c, "what is this?", asset_id=dog_photo)
        assert "the photo contains 2 dogs." in chat.requests[-1]["messages"][0]["content"]
        assert r["answer"] == "A red picture with two dogs." and len(r["boxes"][dog_photo]) == 2
    finally:
        c.__exit__(None, None, None)
