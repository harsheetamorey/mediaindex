"""Ask API: the chat-style assistant over the library (see assistant.py and docs/ask.md)."""

from __future__ import annotations

import base64
import io
import json
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .api_search import embed_query, run_search
from .assistant import Assistant, Turn
from .assistant import display
from .detect import DETECTOR_ID, DETECTOR_KEY, DetectorHost, run_detection
from .llm import ChatUnavailable
from .jobs import QueueFull
from .media.images import ImageRejected, load_image

DESCRIBE_MAX_SIDE = 768  # enough detail for a description, and several times faster than full-size photos


class TurnIn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)
    asset_ids: list[str] = Field(default_factory=list, max_length=200)
    action: str | None = None


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[TurnIn] = Field(default_factory=list, max_length=40)
    library_ids: list[str] | None = None
    asset_id: str | None = None


class PrepareRequest(BaseModel):
    library_ids: list[str] | None = None


def encode_image(im) -> str:
    im = im.copy()
    im.thumbnail((DESCRIBE_MAX_SIDE, DESCRIBE_MAX_SIDE))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


CONFIRM_SCHEMA = {"type": "object", "properties": {"present": {"type": "boolean"}}, "required": ["present"]}


def make_confirm(chat):
    """Ask the local chat model a yes/no question about one photo. An error counts as 'no' (the box is dropped)."""
    def confirm(image, label: str) -> bool:
        try:
            raw = chat.chat([{"role": "user", "content": f"Is there a {display(label, 1)} in this photo? Only say yes if "
                              "you can clearly see one.", "images": [encode_image(image)]}],
                            schema=CONFIRM_SCHEMA, max_tokens=20)
            return bool(json.loads(raw).get("present"))
        except (ChatUnavailable, ValueError, AttributeError):
            return False
    return confirm


def register(app: FastAPI, detector_host: DetectorHost, chat) -> None:
    store = app.state.store

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
        job = next((j.to_dict() for j in app.state.runner.list()
                    if j.kind == "count-objects" and j.status in ("queued", "running")), None)
        return {"detector": {"model": DETECTOR_ID, "key": DETECTOR_KEY,
                             "coverage": store.detection_coverage(DETECTOR_KEY, libs), "job": job},
                "chat_model": chat.status() if chat else {"available": False, "reason": "disabled"}}

    @app.post("/api/ask/prepare")
    def ask_prepare(body: PrepareRequest) -> dict:
        """Queue the one-time object count for photos that have not been checked yet."""
        _check_libs(body.library_ids)
        for j in app.state.runner.list():
            if j.kind == "count-objects" and j.status in ("queued", "running"):
                return j.to_dict()
        confirm = make_confirm(chat) if chat is not None and chat.status().get("available") else None
        try:
            return app.state.runner.submit(
                "count-objects",
                lambda ctx: run_detection(ctx, store, detector_host, body.library_ids, confirm=confirm)).to_dict()
        except QueueFull as e:
            raise HTTPException(503, str(e))

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict:
        _check_libs(body.library_ids)
        if body.asset_id is not None and store.get_asset(body.asset_id) is None:
            raise HTTPException(404, "asset not found")
        history = [Turn(t.role, t.text, t.asset_ids, t.action) for t in body.history]
        out = assistant.ask(body.message, history, body.library_ids, body.asset_id)
        assets = [store.get_asset(a) for a in out["asset_ids"]]
        out["results"] = [app.state.public_asset(a) for a in assets if a is not None]
        out["asset_ids"] = [a["id"] for a in out["results"]]
        return out
