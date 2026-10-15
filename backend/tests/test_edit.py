"""Ask tab edits and creations: "remove the dog from photo 2", "replace the dog with a cat", "make an image of…",
"make a video of those". Mocked detector, outline, fill and image models; the video is really encoded (FFmpeg)."""

import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import make_image, tree_digest
from mediaindex.app import create_app
from mediaindex import api_ask
from mediaindex.assistant import create_plan, edit_plan, is_plural, rule_plan
from mediaindex.config import Settings
from mediaindex.edit import Editor, fill_region, grow
from mediaindex.model.backend import FakeBackend
from test_search import wait_job

RED, BLUE = (200, 0, 0), (0, 0, 220)
S = [0.25, 0.25, 0.5, 0.5]  # the big dog
T = [0.7, 0.6, 0.8, 0.75]  # a small dog
FULL = [0.02, 0.02, 0.98, 0.98]  # a dog that fills the photo


def is_blue(px) -> bool:
    return px[2] > 150 and px[0] < 100


class FakeDetector:
    """Sees a dog wherever the photo is blue at a known spot."""

    def detect(self, image, threshold=0.5):
        im = image.convert("RGB")
        w, h = im.size
        at = lambda b: im.getpixel((int((b[0] + b[2]) / 2 * w), int((b[1] + b[3]) / 2 * h)))  # noqa: E731
        if is_blue(im.getpixel((5, 5))):
            return {"dog": [(0.95, FULL)]}
        return {"dog": [(s, b) for s, b in ((0.9, S), (0.85, T)) if is_blue(at(b))]}

    def close(self):
        pass


class FakeSegmenter:
    def masks(self, image, boxes):
        out = []
        for x0, y0, x1, y1 in boxes:
            m = np.zeros((image.height, image.width), dtype=bool)
            m[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
            out.append(m)
        return out

    def close(self):
        pass


class FakeFiller:
    """Fills from the surroundings: the background here is red."""

    def fill(self, image, mask):
        a = np.asarray(image).copy()
        a[mask] = RED
        return Image.fromarray(a)

    def close(self):
        pass


class FakePainter:
    """Paints by colour: "a cat" is green; a made-up image is all yellow."""
    max_side = 64

    def paint(self, image, mask, prompt):
        return Image.new("RGB", image.size, (0, 200, 0) if "cat" in prompt else (90, 90, 90))

    def generate(self, prompt, width, height):
        return Image.new("RGB", (width, height), (230, 220, 0))

    def close(self):
        pass


def fake_painter(mode="paint"):
    return FakePainter()


fake_painter.model_name = "fake painter"


def dog_photo(path, boxes, size=(160, 120)):
    im = Image.new("RGB", size, RED)
    w, h = size
    for b in boxes:
        im.paste(BLUE, (int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "PNG")  # lossless, so colours stay exact


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "Photos"
    dog_photo(root / "park.png", [S, T])
    dog_photo(root / "closeup.png", [FULL])
    make_image(root / "plain.jpg", RED)
    editor = Editor(FakeSegmenter, FakeFiller, fake_painter)
    app = create_app(Settings(data_dir=tmp_path / "data", precision="float32", watch_interval=0, device="cpu"),
                     backend_factory=FakeBackend, detector_factory=FakeDetector, chat=None, editor=editor,
                     painter=fake_painter)
    c = TestClient(app)
    c.__enter__()
    lib = c.post("/api/libraries", json={"path": str(root)}).json()
    wait_job(c, c.post(f"/api/libraries/{lib['id']}/import").json()["id"])
    yield c, root, tmp_path
    c.__exit__(None, None, None)


def ask(c, message, history=(), **kw):
    r = c.post("/api/ask", json={"message": message, "history": list(history), **kw})
    assert r.status_code == 200, r.text
    return r.json()


def turn(r, edit=None):
    return {"role": "assistant", "text": r["answer"], "asset_ids": r["asset_ids"], "action": r["action"],
            "edit_id": edit["id"] if edit else None}


def photos(c):
    r = ask(c, "show me my photos")
    return r, {a["rel_path"]: i for i, a in enumerate(r["results"], 1)}


def finish(c, r):
    assert r["action"] in ("edit", "generate", "video") and "job" in r, r
    job = wait_job(c, r["job"]["id"])
    assert job["status"] == "done", job.get("error")
    return job["result"]


def fetch(c, edit) -> Image.Image:
    from io import BytesIO

    r = c.get(edit["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    return Image.open(BytesIO(r.content)).convert("RGB")


def test_edit_phrasing():
    assert edit_plan("remove the car from photo 2").subject == "car"
    assert edit_plan("remove the car from photo 2").item == 2
    p = edit_plan("Replace the dog with a cat in the second one.")
    assert (p.subject, p.new, p.item) == ("dog", "a cat", 2)
    assert edit_plan("swap the cat for a teddy bear").new == "a teddy bear"
    assert edit_plan("get rid of all the people in it").subject == "people"
    assert edit_plan("now remove the person too").subject == "person"
    assert edit_plan("show me dogs") is None
    assert rule_plan("remove the dog", False).action == "edit"  # clear wording: an edit even before any photos
    assert rule_plan("change the dog to a cat", False) is None  # "change" only with photos in the conversation
    assert rule_plan("photo 3 isn't a dog", True, True).action == "explain"  # corrections are not edits
    assert is_plural("cars") and is_plural("people") and not is_plural("bus") and not is_plural("wine glass")


def test_remove_biggest_then_chain_then_all(client):
    c, root, tmp = client
    before = tree_digest(root)
    r, pos = photos(c)
    hist = [{"role": "user", "text": "show me my photos"}, turn(r)]
    r2 = ask(c, f"remove the dog from photo {pos['park.png']}", hist)
    assert "new copy" in r2["answer"] and len(r2["results"]) == 1
    res = finish(c, r2)
    assert res["answer"].startswith("Done: I removed the dog. There is 1 more; I changed the biggest.")
    e = res["edit"]
    assert e["op"] == "remove" and e["clean"] is True and e["objects"] == 1
    im = fetch(c, e)
    w, h = im.size
    assert not is_blue(im.getpixel((int(0.375 * w), int(0.375 * h))))  # the big dog is gone
    assert is_blue(im.getpixel((int(0.75 * w), int(0.675 * h))))  # the small one is still there

    hist += [{"role": "user", "text": "remove the dog"}, turn(r2, e)]
    r3 = ask(c, "now remove the dog too", hist)  # keeps editing the copy just made
    e3 = finish(c, r3)["edit"]
    assert e3["parent_edit_id"] == e["id"] and e3["source_asset_id"] == e["source_asset_id"]
    assert not is_blue(fetch(c, e3).getpixel((int(0.75 * w), int(0.675 * h))))

    r4 = ask(c, f"remove all the dogs from photo {pos['park.png']}", hist)
    res4 = finish(c, r4)
    assert res4["answer"].startswith("Done: I removed the 2 dogs.") and res4["edit"]["objects"] == 2
    assert tree_digest(root) == before  # originals untouched


def test_decline_and_explain(client):
    c, root, _ = client
    r, pos = photos(c)
    hist = [{"role": "user", "text": "show me my photos"}, turn(r)]
    res = finish(c, ask(c, f"remove the dog from photo {pos['closeup.png']}", hist))
    assert res["edit"] is None and "too big to remove" in res["answer"]
    res = finish(c, ask(c, f"remove the dog from photo {pos['plain.jpg']}", hist))
    assert res["edit"] is None and "couldn't find a dog" in res["answer"]
    r = ask(c, "remove the tree from photo 1", hist)
    assert "job" not in r and "not “tree”" in r["answer"]
    r = ask(c, "delete photo 2", hist)
    assert "job" not in r and "never deletes" in r["answer"]
    r = ask(c, "remove the dog", hist)  # several photos and none named
    assert "job" not in r and r["answer"].startswith("Which photo?")
    r = ask(c, "remove the dog")  # no photos yet
    assert "job" not in r and r["answer"].startswith("First find the photo")


def test_edit_clicked_photo_and_save_copy(client):
    c, root, tmp = client
    r, pos = photos(c)
    park = r["results"][pos["park.png"] - 1]
    res = finish(c, ask(c, "remove the dog", asset_id=park["id"]))
    e = res["edit"]
    assert c.get(f"/api/edits/{e['id']}").json()["source_asset_id"] == park["id"]
    out = tmp / "Edited"
    s1 = c.post(f"/api/edits/{e['id']}/save", json={"destination": str(out)})
    s2 = c.post(f"/api/edits/{e['id']}/save", json={"destination": str(out)})
    assert s1.json()["saved"] == "park (edited).jpg" and s2.json()["saved"] == "park (edited) (2).jpg"
    assert sorted(p.name for p in out.iterdir()) == ["park (edited) (2).jpg", "park (edited).jpg"]
    assert c.post(f"/api/edits/{e['id']}/save", json={"destination": str(root / "x")}).status_code == 400
    assert c.get("/api/edits/../../etc/passwd/image").status_code == 404
    assert c.get("/api/edits/0123456789abcdef01234567/image").status_code == 404


def test_fill_changes_only_masked_pixels():
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    mask = np.zeros((300, 400), dtype=bool)
    mask[100:160, 150:230] = True
    out = fill_region(im, mask, lambda crop, m: Image.new("RGB", crop.size, (0, 0, 0)), max_side=64)
    a, b = np.asarray(im), np.asarray(out)
    assert (a[~mask] == b[~mask]).all() and (b[mask] == 0).all()
    g = grow(mask, 10)
    assert g[95, 190] and g[165, 190] and not g[50, 50] and g.sum() > mask.sum()


def test_create_phrasing():
    assert (create_plan("make an image of a red bicycle").subject, create_plan("make an image of a red bicycle").new) == \
        ("a red bicycle", "square")
    assert create_plan("Generate a wide picture of a castle at sunset").new == "wide"
    p = create_plan("turn my mountain photos into a reel")
    assert (p.action, p.subject, p.new) == ("video", "mountain", "vertical")
    assert create_plan("make a video of those").subject == ""
    assert create_plan("make a square video of the dog photos").new == "square"
    assert create_plan("show me videos of dogs") is None
    assert rule_plan("make an image of a cat", False).action == "generate"


def test_replace_generate_and_video(client, monkeypatch):
    c, root, tmp = client
    before = tree_digest(root)
    r, pos = photos(c)
    hist = [{"role": "user", "text": "show me my photos"}, turn(r)]
    res = finish(c, ask(c, f"replace the dog with a cat in photo {pos['park.png']}", hist))
    e = res["edit"]
    assert res["answer"].startswith("Done: I replaced the dog with a cat.") and e["op"] == "replace" and e["clean"] is True
    im = fetch(c, e)
    w, h = im.size
    px = im.getpixel((int(0.375 * w), int(0.375 * h)))
    assert px[1] > 150 and px[0] < 100  # painted green where the big dog was
    assert is_blue(im.getpixel((int(0.75 * w), int(0.675 * h))))  # the small dog is untouched

    r = ask(c, "make a wide image of a yellow square")
    assert r["action"] == "generate" and "not one of your photos" in r["answer"]
    res = finish(c, r)
    g = res["edit"]
    assert g["op"] == "generate" and (g["width"], g["height"]) == (768, 512) and "isn't a real photo" in res["answer"]
    assert fetch(c, g).getpixel((10, 10))[0] > 200
    saved = c.post(f"/api/edits/{g['id']}/save", json={"destination": str(tmp / "out")}).json()
    assert saved["saved"] == "MediaIndex image.jpg"

    monkeypatch.setitem(api_ask.VIDEO_SHAPES, "vertical", (90, 160))
    r = ask(c, "turn those into a reel", hist)
    assert r["action"] == "video" and len(r["results"]) == 3
    res = finish(c, r)
    v = res["edit"]
    assert v["kind"] == "video" and v["url"].endswith("/video") and (v["width"], v["height"]) == (90, 160)
    assert "3 photos" in res["answer"] and "nothing made up" in res["answer"]
    body = c.get(v["url"])
    assert body.status_code == 200 and body.headers["content-type"] == "video/mp4"
    out = tmp / "clip.mp4"
    out.write_bytes(body.content)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0",
                            str(out)], capture_output=True, text=True).stdout.strip()
    assert probe == "90,160,234"  # 3 photos x 90 frames, minus 2 crossfades of 18 frames
    assert c.get(f"/api/edits/{v['id']}/image").status_code == 404
    assert c.post(f"/api/edits/{v['id']}/save", json={"destination": str(tmp / "out")}).json()["saved"].endswith(
        "video.mp4")
    assert tree_digest(root) == before
