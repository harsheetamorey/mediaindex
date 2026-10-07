"""Selections, manifest export, copy export, and reveal/copy-path helpers."""

from __future__ import annotations

import platform
import subprocess
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import selections as sel
from .jobs import QueueFull


class SelectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ItemAdd(BaseModel):
    asset_id: str
    query_context: dict | None = None
    start_s: float | None = None
    end_s: float | None = None


class Reorder(BaseModel):
    item_ids: list[str]


class ExportPreview(BaseModel):
    destination: str = Field(min_length=1, max_length=4096)
    include_manifest: bool = True


class ExportConfirm(BaseModel):
    plan_id: str
    confirm: bool


def register(app: FastAPI) -> None:
    store = app.state.store
    plans: dict[str, sel.ExportPlan] = {}
    plans_lock = threading.Lock()

    def _sel_or_404(sid: str) -> dict:
        s = sel.get_selection(store, sid)
        if s is None:
            raise HTTPException(404, "selection not found")
        return s

    @app.get("/api/selections")
    def list_sel() -> list[dict]:
        return sel.list_selections(store)

    @app.post("/api/selections")
    def create_sel(body: SelectionCreate) -> dict:
        return sel.create_selection(store, body.name.strip())

    @app.get("/api/selections/{sid}")
    def get_sel(sid: str) -> dict:
        s = _sel_or_404(sid)
        items = sel.list_items(store, sid)
        for it in items:
            it["thumbnail_url"] = f"/api/assets/{it['asset_id']}/thumbnail" if it["status"] != "removed" else None
        return s | {"items": items}

    @app.patch("/api/selections/{sid}")
    def rename_sel(sid: str, body: SelectionCreate) -> dict:
        _sel_or_404(sid)
        sel.rename_selection(store, sid, body.name.strip())
        return sel.get_selection(store, sid)

    @app.delete("/api/selections/{sid}")
    def delete_sel(sid: str) -> dict:
        _sel_or_404(sid)
        sel.delete_selection(store, sid)  # never touches media files
        return {"deleted": sid}

    @app.post("/api/selections/{sid}/items")
    def add_item(sid: str, body: ItemAdd) -> dict:
        _sel_or_404(sid)
        try:
            return sel.add_item(store, sid, body.asset_id, body.query_context, body.start_s, body.end_s)
        except sel.SelectionError as e:
            raise HTTPException(404, str(e))

    @app.delete("/api/selections/{sid}/items/{item_id}")
    def remove_item(sid: str, item_id: str) -> dict:
        _sel_or_404(sid)
        sel.remove_item(store, sid, item_id)
        return {"removed": item_id}

    @app.put("/api/selections/{sid}/order")
    def reorder(sid: str, body: Reorder) -> dict:
        _sel_or_404(sid)
        try:
            sel.reorder(store, sid, body.item_ids)
        except sel.SelectionError as e:
            raise HTTPException(422, str(e))
        return {"ok": True}

    @app.get("/api/selections/{sid}/manifest")
    def manifest(sid: str):
        s = _sel_or_404(sid)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s["name"])[:60] or "selection"
        return JSONResponse(sel.build_manifest(store, sid),
                            headers={"Content-Disposition": f'attachment; filename="{safe}-manifest.json"'})

    @app.post("/api/selections/{sid}/export/preview")
    def export_preview(sid: str, body: ExportPreview) -> dict:
        _sel_or_404(sid)
        try:
            plan = sel.plan_export(store, sid, body.destination, body.include_manifest)
        except sel.SelectionError as e:
            raise HTTPException(400, str(e))
        with plans_lock:
            for k in [k for k, p in plans.items() if time.time() - p.created > sel.PLAN_TTL_SECONDS]:
                plans.pop(k)
            plans[plan.id] = plan
        return sel.plan_to_dict(plan)

    @app.post("/api/selections/{sid}/export")
    def export_run(sid: str, body: ExportConfirm) -> dict:
        _sel_or_404(sid)
        if not body.confirm:
            raise HTTPException(400, "export requires explicit confirmation")
        with plans_lock:
            plan = plans.pop(body.plan_id, None)
        if plan is None or plan.selection_id != sid or time.time() - plan.created > sel.PLAN_TTL_SECONDS:
            raise HTTPException(409, "export plan not found or expired; preview again")
        try:
            sel.validate_destination(store, str(plan.destination))  # re-check at execution time
        except sel.SelectionError as e:
            raise HTTPException(400, str(e))
        try:
            job = app.state.runner.submit("export", lambda ctx: sel.run_export(ctx, store, plan))
        except QueueFull as e:
            raise HTTPException(503, str(e))
        return job.to_dict()

    # ---- copy path / reveal (no shell; path comes from the ID resolver) ---------------
    @app.get("/api/assets/{asset_id}/path")
    def asset_path(asset_id: str) -> dict:
        _, p = app.state.asset_path(asset_id)
        return {"path": str(p)}

    @app.post("/api/assets/{asset_id}/reveal")
    def reveal(asset_id: str) -> dict:
        _, p = app.state.asset_path(asset_id)
        system = platform.system()
        if system == "Darwin":
            cmd = ["/usr/bin/open", "-R", str(p)]
        elif system == "Linux":
            cmd = ["xdg-open", str(p.parent)]
        elif system == "Windows":
            cmd = ["explorer", f"/select,{p}"]
        else:
            raise HTTPException(501, "reveal is not supported on this platform")
        try:
            subprocess.run(cmd, check=False, timeout=10, shell=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise HTTPException(500, f"could not open file manager: {e}")
        return {"revealed": True}


class ClipPreview(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    mode: str = "accurate"
    destination: str = Field(min_length=1, max_length=4096)
    query_context: dict | None = None


def register_clips(app: FastAPI) -> None:
    from . import clips

    store = app.state.store
    plans: dict[str, clips.ClipPlan] = {}
    lock = threading.Lock()

    @app.post("/api/assets/{asset_id}/clip/preview")
    def clip_preview(asset_id: str, body: ClipPreview) -> dict:
        a, path = app.state.asset_path(asset_id)
        try:
            plan = clips.plan_clip(store, a, path, body.start_s, body.end_s, body.mode, body.destination,
                                   body.query_context)
        except sel.SelectionError as e:
            raise HTTPException(400, str(e))
        with lock:
            for k in [k for k, p in plans.items() if time.time() - p.created > clips.PLAN_TTL_SECONDS]:
                plans.pop(k)
            plans[plan.id] = plan
        return clips.plan_dict(plan)

    @app.post("/api/assets/{asset_id}/clip")
    def clip_export(asset_id: str, body: ExportConfirm) -> dict:
        if not body.confirm:
            raise HTTPException(400, "clip export requires explicit confirmation")
        with lock:
            plan = plans.pop(body.plan_id, None)
        if plan is None or plan.asset["id"] != asset_id or time.time() - plan.created > clips.PLAN_TTL_SECONDS:
            raise HTTPException(409, "clip plan not found or expired; preview again")
        try:
            sel.validate_destination(store, str(plan.destination))
        except sel.SelectionError as e:
            raise HTTPException(400, str(e))
        try:
            job = app.state.runner.submit("clip", lambda ctx: clips.run_clip_export(ctx, plan))
        except QueueFull as e:
            raise HTTPException(503, str(e))
        return job.to_dict()
