"""Ask API: the chat-style assistant over the library (see assistant.py and docs/ask.md)."""

from __future__ import annotations

import base64
import io
import shutil
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from .api_search import embed_query, run_search
from .assistant import Assistant, Turn, display, resolve_label
from .detect import DETECTOR_ID, DETECTOR_KEY, SURE, DetectorHost, run_detection
from .edit import DOWNLOAD_HINT as EDIT_DOWNLOAD_HINT
from .edit import EditRefused, Editor, EditStore
from .finder_tags import set_finder_tags
from .jobs import JobContext, QueueFull
from .media.images import ImageRejected, load_image
from .slideshow import LANDSCAPE, SQUARE, VERTICAL, make_slideshow
from .selections import PARTIAL_PREFIX, SelectionError, _free_name, _place_exclusive, validate_destination

BIG_FILL = 0.12  # edits filling more than this share of the photo more often leave smudges (bench: docs/edit.md)
SHAPES = {"square": (512, 512), "wide": (768, 512), "tall": (512, 768)}
VIDEO_SHAPES = {"landscape": LANDSCAPE, "vertical": VERTICAL, "square": SQUARE}
DESCRIBE_MAX_SIDE = 768  # enough detail for a description, and several times faster than full-size photos


class TurnIn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)
    asset_ids: list[str] = Field(default_factory=list, max_length=200)
    action: str | None = None
    edit_id: str | None = Field(None, max_length=64)


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[TurnIn] = Field(default_factory=list, max_length=40)
    library_ids: list[str] | None = None
    asset_id: str | None = None


class PrepareRequest(BaseModel):
    library_ids: list[str] | None = None


class SaveEditRequest(BaseModel):
    destination: str = Field(min_length=1, max_length=4096)


def public_edit(meta: dict) -> dict:
    keys = ("id", "op", "label", "new", "objects", "clean", "source_asset_id", "parent_edit_id", "width", "height",
            "seconds", "models")
    kind = "video" if meta.get("op") == "video" else "image"
    return {**{k: meta.get(k) for k in keys}, "kind": kind, "url": f"/api/edits/{meta['id']}/{kind}"}


def encode_image(im) -> str:
    im = im.copy()
    im.thumbnail((DESCRIBE_MAX_SIDE, DESCRIBE_MAX_SIDE))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def register(app: FastAPI, detector_host: DetectorHost, chat, editor: Editor | None = None, painter_factory=None) -> None:
    store = app.state.store
    edits = EditStore(app.state.settings.data_dir / "edits")

    def search(text: str, library_ids, limit: int) -> list[str]:
        vec, ms = embed_query(app, lambda b: b.embed_query_texts([text])[0])
        r = run_search(app, mode="text", query_info={"text": text}, vec=vec, embed_ms=ms, library_ids=library_ids,
                       media_types=["image"], limit=limit)
        return [x["asset"]["id"] for x in r["results"]]

    def rank_within(text: str, asset_ids: list[str]) -> list[str]:
        vec, _ = embed_query(app, lambda b: b.embed_query_texts([text])[0])
        scored = []
        for aid in asset_ids:
            v = store.get_asset_vectors(aid, app.state.profile.key, "image")
            scored.append((float(np.dot(v[0], vec)) if v is not None else -2.0, aid))
        return [aid for _, aid in sorted(scored, key=lambda s: -s[0])]

    def image_b64(asset_id: str) -> str:
        _, path = app.state.asset_path(asset_id)
        try:
            return encode_image(load_image(path))
        except ImageRejected as e:
            raise HTTPException(422, f"cannot read this photo: {str(e).split(':')[0]}")

    assistant = Assistant(store, DETECTOR_KEY, search, rank_within, image_b64, chat)
    app.state.assistant = assistant

    def _check_libs(library_ids):
        for lid in library_ids or []:
            if not store.get_library(lid):
                raise HTTPException(404, f"library not found: {lid}")

    @app.get("/api/ask/status")
    def ask_status(library_ids: str | None = None) -> dict:
        libs = [x for x in (library_ids or "").split(",") if x] or None
        _check_libs(libs)
        jobs = [j for j in app.state.runner.list() if j.kind == "count-objects"]
        job = next((j.to_dict() for j in jobs if j.status in ("queued", "running")), None)
        last = max(jobs, key=lambda j: j.created, default=None)
        error = last.error if last is not None and last.status == "failed" else None
        return {"detector": {"model": DETECTOR_ID, "key": DETECTOR_KEY,
                             "coverage": store.detection_coverage(DETECTOR_KEY, libs), "job": job, "error": error},
                "chat_model": chat.status() if chat else {"available": False, "reason": "disabled"}}

    @app.post("/api/ask/prepare")
    def ask_prepare(body: PrepareRequest) -> dict:
        """Queue the one-time object count for photos that have not been checked yet."""
        _check_libs(body.library_ids)
        for j in app.state.runner.list():
            if j.kind == "count-objects" and j.status in ("queued", "running"):
                return j.to_dict()
        try:
            return app.state.runner.submit(
                "count-objects", lambda ctx: run_detection(ctx, store, detector_host, body.library_ids)).to_dict()
        except QueueFull as e:
            raise HTTPException(503, str(e))

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict:
        _check_libs(body.library_ids)
        if body.asset_id is not None and store.get_asset(body.asset_id) is None:
            raise HTTPException(404, "asset not found")
        history = [Turn(t.role, t.text, t.asset_ids, t.action, t.edit_id) for t in body.history]
        out = assistant.ask(body.message, history, body.library_ids, body.asset_id)
        assets = [store.get_asset(a) for a in out["asset_ids"]]
        out["results"] = [app.state.public_asset(a) for a in assets if a is not None]
        out["asset_ids"] = [a["id"] for a in out["results"]]
        if (req := out.pop("create_request", None)) is not None:
            if req["op"] == "generate" and painter_factory is None:
                out["answer"] = "Making images is turned off on this computer."
            else:
                try:
                    out["job"] = app.state.runner.submit(req["op"], lambda ctx, req=req: run_create(ctx, req)).to_dict()
                except QueueFull as e:
                    raise HTTPException(503, str(e))
        if (req := out.pop("edit_request", None)) is not None:
            if editor is None:
                out["answer"] = "Editing photos is turned off on this computer."
            elif "edit_id" in req and edits.get(req["edit_id"]) is None:
                out["answer"] = "I can't find that edited copy any more. Ask for the photo again, then edit it."
            else:
                try:
                    out["job"] = app.state.runner.submit("edit", lambda ctx, req=req: run_edit(ctx, req)).to_dict()
                except QueueFull as e:
                    raise HTTPException(503, str(e))
        return out

    # ---- edits: new copies, never the originals ------------------------------------------
    def run_edit(ctx: JobContext, req: dict) -> dict:
        """Find the object, outline it, remove or replace it, check the result, save a new copy."""
        ctx.progress(0, 3, "Finding it")
        if "edit_id" in req:
            parent = edits.get(req["edit_id"])
            image = Image.open(edits.path(req["edit_id"])).convert("RGB")
            source = {"source_asset_id": parent["source_asset_id"], "parent_edit_id": parent["id"]}
        else:
            _, path = app.state.asset_path(req["asset_id"])
            image = load_image(path)
            source = {"source_asset_id": req["asset_id"], "parent_edit_id": None}
        label, new, everything = req["label"], req.get("new") or "", req.get("all", False)
        thing = display(label, 1)
        try:
            try:
                detector = detector_host.get()
            except Exception as e:
                raise RuntimeError(f"the object detector could not be loaded ({type(e).__name__})") from e
            w, h = image.size
            px = lambda b: (b[0] * w, b[1] * h, b[2] * w, b[3] * h)  # noqa: E731
            found = sorted(detector.detect(image).get(label, []), key=lambda sb: -sb[0])
            sure = [b for s, b in found if s >= SURE] or [b for s, b in found]
            if not sure:
                return {"answer": f"I couldn't find a {thing} in this photo, so I didn't change anything. I can only "
                                  f"edit things the object detector finds.", "edit": None}
            area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
            picked = sure if everything else [max(sure, key=area)]
            ctx.progress(1, message="Outlining it")

            def still_there(edited):
                return [px(b) for _, b in detector.detect(edited).get(label, [])]

            new_label = resolve_label(new) if new else None

            def new_there(edited, mask):  # the new thing is found, mostly inside the painted area
                for _, b in detector.detect(edited).get(new_label, []):
                    x0, y0, x1, y1 = int(b[0] * w), int(b[1] * h), int(np.ceil(b[2] * w)), int(np.ceil(b[3] * h))
                    region = mask[y0:y1, x0:x1]
                    if region.size and region.mean() >= 0.5:
                        return True
                return False

            try:
                if new:
                    r = editor.replace(image, [px(b) for b in picked], new, still_there,
                                       new_there if new_label else None)
                else:
                    r = editor.remove(image, [px(b) for b in picked], still_there)
            except EditRefused as e:
                return {"answer": f"I didn't change it: the {thing} is too big to {'replace' if new else 'remove'} "
                                  f"believably ({e}). Filling that much would mean making up most of the picture.",
                        "edit": None}
            except OSError as e:  # model files missing
                raise RuntimeError(f"the editing models aren't downloaded yet. Run this once in the MediaIndex "
                                   f"folder: {EDIT_DOWNLOAD_HINT}") from e
        finally:
            detector_host.unload()
        ctx.progress(3, message="Saving a copy")
        meta = edits.save(r.image, {**source, "op": "replace" if new else "remove", "label": label, "new": new,
                                    "objects": r.objects, "clean": r.clean, "attempts": r.attempts,
                                    "new_found": getattr(r, "new_found", None),
                                    "seconds": round(r.seconds, 1), "changed_share": round(float(r.mask.mean()), 4),
                                    "models": editor.models(bool(new))})
        n_found, n_done = len(sure), len(picked)
        what = display(label, n_done)
        if new:
            text = f"Done: I replaced the {what} with {new}."
        else:
            text = f"Done: I removed the {'' if n_done == 1 else str(n_done) + ' '}{what}."
        if n_found > n_done:
            text += (f" There {'is' if n_found - n_done == 1 else 'are'} {n_found - n_done} more; I changed the biggest. "
                     f"Say “{'replace' if new else 'remove'} all the {display(label)}” to change every one.")
        if new and r.new_found is False:
            text += (f" A check doesn't find {new} in the result{' after two tries' if r.attempts > 1 else ''}, so it "
                     f"may not look like one.")
        if r.clean is False:
            text += f" A check still finds something like a {thing} there, so look closely: it may not be clean."
        elif r.mask.mean() > BIG_FILL:
            text += " It was a big area to fill, so check the result closely for smudges or leftover shadows."
        text += " This is a new copy; your original photo is unchanged. Use “Save a copy…” to keep it."
        return {"answer": text, "edit": public_edit(meta)}

    def run_create(ctx: JobContext, req: dict) -> dict:
        """A new image from words, or a video of photos. Saved like edits; nothing in the libraries changes."""
        device = app.state.settings.resolved_device()
        if req["op"] == "generate":
            ctx.progress(0, 1, "Loading the image model")
            painter = painter_factory()  # the inpainting model on a blank canvas: more natural pictures
            try:
                ctx.progress(0, 1, "Making the image")
                w, h = SHAPES.get(req.get("shape"), SHAPES["square"])
                im = painter.generate(req["prompt"], w, h)
            finally:
                painter.close()
            meta = edits.save(im, {"op": "generate", "label": None, "new": req["prompt"], "objects": 0, "clean": None,
                                   "source_asset_id": None, "parent_edit_id": None,
                                   "models": [getattr(painter_factory, "model_name", "image model")]})
            return {"answer": f"Here is a new image of “{req['prompt']}”. It is made up by a local image model, so it "
                              f"isn't a real photo, and details can be wrong.", "edit": public_edit(meta)}
        ids = [a for a in req["asset_ids"] if (x := store.get_asset(a)) is not None and x["media_type"] == "image"]
        if not ids:
            return {"answer": "None of those are photos, so there's nothing to make a video from.", "edit": None}
        size = VIDEO_SHAPES.get(req.get("shape"), LANDSCAPE)

        def loader(aid):
            return lambda: load_image(app.state.asset_path(aid)[1])

        out_path = edits.new_path(".mp4")
        info = make_slideshow([loader(a) for a in ids], out_path, size,
                              progress=lambda i, n: ctx.progress(i, n, f"Adding photo {i} of {n}"),
                              cancelled=lambda: ctx.cancelled, device=device)
        meta = edits.adopt(out_path, {"op": "video", "label": None, "new": "", "objects": info["photos"], "clean": None,
                                      "source_asset_id": ids[0], "parent_edit_id": None, "asset_ids": ids,
                                      "width": info["width"], "height": info["height"], "seconds": info["seconds"],
                                      "models": []})
        return {"answer": f"Here is your video: {info['photos']} photos, {info['seconds']:.0f} seconds, "
                          f"{info['width']}×{info['height']}. It shows exactly those photos, nothing made up. "
                          f"Use “Save a copy…” to keep it.", "edit": public_edit(meta)}

    @app.get("/api/edits/{edit_id}/video")
    def edit_video(edit_id: str):
        meta = edits.get(edit_id)
        p = edits.path(edit_id)
        if meta is None or meta.get("op") != "video" or p is None:
            raise HTTPException(404, "video not found")
        return FileResponse(p, media_type="video/mp4")

    @app.get("/api/edits/{edit_id}/image")
    def edit_image(edit_id: str):
        meta = edits.get(edit_id)
        p = edits.path(edit_id)
        if meta is None or meta.get("op") == "video" or p is None:
            raise HTTPException(404, "edit not found")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/edits/{edit_id}")
    def edit_meta(edit_id: str) -> dict:
        meta = edits.get(edit_id)
        if meta is None:
            raise HTTPException(404, "edit not found")
        return public_edit(meta)

    @app.post("/api/edits/{edit_id}/save")
    def save_edit(edit_id: str, body: SaveEditRequest) -> dict:
        """Write a copy of the edited photo into a folder the user chose (outside the libraries); never overwrites."""
        meta = edits.get(edit_id)
        if meta is None:
            raise HTTPException(404, "edit not found")
        try:
            dest = validate_destination(store, body.destination)
        except SelectionError as e:
            raise HTTPException(400, str(e))
        a = store.get_asset(meta["source_asset_id"]) if meta.get("source_asset_id") else None
        stem = Path(a["rel_path"]).stem if a else "photo"
        op = meta.get("op")
        name, tag = {"video": (f"{stem} video.mp4", "Video"), "generate": ("MediaIndex image.jpg", "Made up")}.get(
            op, (f"{stem} (edited).jpg", "Edited"))
        dest.mkdir(parents=False, exist_ok=True)
        tmp = dest / f"{PARTIAL_PREFIX}{uuid.uuid4().hex}{Path(name).suffix}"
        shutil.copyfile(edits.path(edit_id), tmp)
        name = _place_exclusive(tmp, dest, _free_name(name, set(), dest), set())
        set_finder_tags(dest / name, ["MediaIndex", tag])
        return {"saved": name, "destination": str(dest)}
