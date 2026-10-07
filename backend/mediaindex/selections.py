"""Persistent selections, manifest export and safe copy export.

Exports never move, modify or delete originals. Copies are written to a temporary
`.mediaindex-partial-*` file, verified by SHA-256, then linked into place with an
exclusive operation that cannot overwrite an existing file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .db import now
from .jobs import JobContext
from .paths import PathRejected, is_within, resolve_in_root
from .store import Store

PARTIAL_PREFIX = ".mediaindex-partial-"
MANIFEST_NAME = "mediaindex-manifest.json"
PLAN_TTL_SECONDS = 15 * 60


class SelectionError(ValueError):
    pass


# ---- persistence -----------------------------------------------------------------
def create_selection(store: Store, name: str) -> dict:
    sid = uuid.uuid4().hex
    t = now()
    with store.db.tx() as c:
        c.execute("INSERT INTO selections(id, name, created_at, updated_at) VALUES (?,?,?,?)", (sid, name, t, t))
    return get_selection(store, sid)


def list_selections(store: Store) -> list[dict]:
    rows = store.db.query("""SELECT s.*, (SELECT COUNT(*) FROM selection_items i WHERE i.selection_id = s.id) AS item_count
                             FROM selections s ORDER BY s.updated_at DESC""")
    return [dict(r) for r in rows]


def get_selection(store: Store, sid: str) -> dict | None:
    r = store.db.one("SELECT * FROM selections WHERE id=?", (sid,))
    return dict(r) if r else None


def rename_selection(store: Store, sid: str, name: str) -> None:
    with store.db.tx() as c:
        c.execute("UPDATE selections SET name=?, updated_at=? WHERE id=?", (name, now(), sid))


def delete_selection(store: Store, sid: str) -> None:
    with store.db.tx() as c:
        c.execute("DELETE FROM selections WHERE id=?", (sid,))


def _snapshot(store: Store, asset: dict) -> dict:
    lib = store.get_library(asset["library_id"]) or {}
    meta = json.loads(asset["meta_json"]) if asset.get("meta_json") else {}
    return {
        "library_id": asset["library_id"],
        "library_name": lib.get("name"),
        "library_root": lib.get("root_path"),
        "rel_path": asset["rel_path"],
        "media_type": asset["media_type"],
        "content_hash": asset["content_hash"],
        "size": asset["size"],
        "width": asset["width"],
        "height": asset["height"],
        "duration": asset["duration"],
        "source": meta.get("source"),
    }


def add_item(store: Store, sid: str, asset_id: str, query_context: dict | None = None,
             start_s: float | None = None, end_s: float | None = None) -> dict:
    if not get_selection(store, sid):
        raise SelectionError("selection not found")
    asset = store.get_asset(asset_id)
    if asset is None:
        raise SelectionError("asset not found")
    with store.db.tx() as c:
        existing = c.execute("""SELECT id FROM selection_items WHERE selection_id=? AND asset_id=?
                                AND IFNULL(start_s,-1)=IFNULL(?,-1) AND IFNULL(end_s,-1)=IFNULL(?,-1)""",
                             (sid, asset_id, start_s, end_s)).fetchone()
        if existing:
            return {"id": existing["id"], "added": False}
        pos = c.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM selection_items WHERE selection_id=?",
                        (sid,)).fetchone()[0]
        iid = uuid.uuid4().hex
        c.execute("""INSERT INTO selection_items(id, selection_id, asset_id, position, start_s, end_s, snapshot_json,
                     query_context_json, added_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                  (iid, sid, asset_id, pos, start_s, end_s, json.dumps(_snapshot(store, asset)),
                   json.dumps(query_context) if query_context else None, now()))
        c.execute("UPDATE selections SET updated_at=? WHERE id=?", (now(), sid))
    return {"id": iid, "added": True}


def remove_item(store: Store, sid: str, item_id: str) -> None:
    with store.db.tx() as c:
        c.execute("DELETE FROM selection_items WHERE id=? AND selection_id=?", (item_id, sid))
        c.execute("UPDATE selections SET updated_at=? WHERE id=?", (now(), sid))


def reorder(store: Store, sid: str, item_ids: list[str]) -> None:
    with store.db.tx() as c:
        current = {r["id"] for r in c.execute("SELECT id FROM selection_items WHERE selection_id=?", (sid,))}
        if set(item_ids) != current or len(item_ids) != len(current):
            raise SelectionError("reorder must list every item of the selection exactly once")
        for pos, iid in enumerate(item_ids):
            c.execute("UPDATE selection_items SET position=? WHERE id=?", (pos, iid))
        c.execute("UPDATE selections SET updated_at=? WHERE id=?", (now(), sid))


def item_status(store: Store, asset_id: str, snap: dict) -> tuple[str, Path | None]:
    """'ok' | 'missing' (file gone) | 'removed' (no longer in the index) | 'changed' (content differs)."""
    asset = store.get_asset(asset_id)
    if asset is None:
        return "removed", None
    lib = store.get_library(asset["library_id"])
    try:
        p = resolve_in_root(lib["root_path"], asset["rel_path"])
    except PathRejected:
        return "missing", None
    if not p.is_file():
        return "missing", None
    if snap.get("content_hash") and asset["content_hash"] != snap["content_hash"]:
        return "changed", p
    return "ok", p


def list_items(store: Store, sid: str) -> list[dict]:
    rows = store.db.query("SELECT * FROM selection_items WHERE selection_id=? ORDER BY position", (sid,))
    out = []
    for r in rows:
        snap = json.loads(r["snapshot_json"])
        status, _ = item_status(store, r["asset_id"], snap)
        out.append({
            "id": r["id"], "asset_id": r["asset_id"], "position": r["position"],
            "start_s": r["start_s"], "end_s": r["end_s"], "status": status, "snapshot": snap,
            "query_context": json.loads(r["query_context_json"]) if r["query_context_json"] else None,
            "added_at": r["added_at"],
        })
    return out


# ---- manifest ----------------------------------------------------------------------
def build_manifest(store: Store, sid: str, copied: dict[str, str] | None = None) -> dict:
    sel = get_selection(store, sid)
    if sel is None:
        raise SelectionError("selection not found")
    items = []
    for it in list_items(store, sid):
        s = it["snapshot"]
        src = str(Path(s["library_root"]) / s["rel_path"]) if s.get("library_root") else None
        entry = {
            "position": it["position"],
            "asset_id": it["asset_id"],
            "status": it["status"],
            "media_type": s["media_type"],
            "source_path": src,
            "library": {"id": s["library_id"], "name": s["library_name"]},
            "rel_path": s["rel_path"],
            "sha256": s["content_hash"],
            "size_bytes": s["size"],
            "dimensions": [s["width"], s["height"]] if s.get("width") else None,
            "duration_s": s.get("duration"),
            "segment": [it["start_s"], it["end_s"]] if it["start_s"] is not None else None,
            "source_record": s.get("source"),
            "query_context": it["query_context"],
        }
        if copied is not None:
            entry["exported_as"] = copied.get(it["id"])
        items.append(entry)
    return {
        "format": "mediaindex-selection-manifest",
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection": {"id": sel["id"], "name": sel["name"]},
        "note": ("source_record reproduces what the data publisher declared (if anything); it is not a rights audit. "
                 "Similarity scores are not included because they are not confidence values."),
        "items": items,
    }


# ---- copy export -----------------------------------------------------------------------
@dataclass
class ExportPlan:
    id: str
    selection_id: str
    destination: Path
    entries: list[dict]  # {item_id, source, dest_name, sha256, renamed}
    skipped: list[dict]
    manifest_name: str | None
    created: float


def validate_destination(store: Store, dest: str) -> Path:
    if not dest or not os.path.isabs(os.path.expanduser(dest)):
        raise SelectionError("destination must be an absolute folder path")
    p = Path(os.path.expanduser(dest)).resolve()
    if p == Path(p.anchor) or p == Path.home().resolve():
        raise SelectionError("choose a specific destination folder, not the filesystem root or your home directory")
    if p.exists() and not p.is_dir():
        raise SelectionError("destination exists and is not a folder")
    if not p.exists() and not p.parent.is_dir():
        raise SelectionError("destination's parent folder does not exist")
    for lib in store.list_libraries():
        root = Path(lib["root_path"])
        if is_within(p, root) or is_within(root, p):
            raise SelectionError(f"destination overlaps the library folder '{lib['name']}'; choose a folder outside "
                                 "your indexed libraries so originals and copies stay separate")
    return p


def _free_name(name: str, taken: set[str], dest: Path) -> str:
    stem, suffix = os.path.splitext(name)
    candidate, n = name, 1
    while candidate.lower() in taken or (dest / candidate).exists():
        n += 1
        candidate = f"{stem} ({n}){suffix}"
    taken.add(candidate.lower())
    return candidate


def plan_export(store: Store, sid: str, destination: str, include_manifest: bool = True) -> ExportPlan:
    dest = validate_destination(store, destination)
    taken: set[str] = set()
    entries, skipped = [], []
    for it in list_items(store, sid):
        status, src = item_status(store, it["asset_id"], it["snapshot"])
        if status != "ok" or src is None:
            skipped.append({"item_id": it["id"], "rel_path": it["snapshot"]["rel_path"], "reason": status})
            continue
        if it["start_s"] is not None and it["snapshot"]["media_type"] != "video":
            skipped.append({"item_id": it["id"], "rel_path": it["snapshot"]["rel_path"],
                            "reason": "audio segments are exported as whole files; add the file instead"})
            continue
        if it["start_s"] is not None:  # video moment -> frame-accurate clip
            from .clips import fmt_ts

            base = f"{Path(it['snapshot']['rel_path']).stem}_{fmt_ts(it['start_s'])}-{fmt_ts(it['end_s'])}.mp4"
            name = _free_name(base, taken, dest)
            entries.append({"item_id": it["id"], "source": str(src), "dest_name": name, "sha256": None,
                            "renamed": name != base, "bytes": None, "clip": [it["start_s"], it["end_s"]]})
            continue
        base = Path(it["snapshot"]["rel_path"]).name
        name = _free_name(base, taken, dest)
        entries.append({"item_id": it["id"], "source": str(src), "dest_name": name,
                        "sha256": it["snapshot"]["content_hash"], "renamed": name != base,
                        "bytes": it["snapshot"]["size"]})
    manifest_name = _free_name(MANIFEST_NAME, taken, dest) if include_manifest else None
    return ExportPlan(uuid.uuid4().hex, sid, dest, entries, skipped, manifest_name, time.time())


def plan_to_dict(plan: ExportPlan) -> dict:
    return {
        "plan_id": plan.id,
        "destination": str(plan.destination),
        "destination_exists": plan.destination.exists(),
        "files": [{k: e[k] for k in ("item_id", "dest_name", "renamed", "bytes")} | {"source": e["source"],
                  "clip": e.get("clip")} for e in plan.entries],
        "skipped": plan.skipped,
        "manifest_name": plan.manifest_name,
        "total_bytes": sum(e["bytes"] or 0 for e in plan.entries),
        "overwrites": 0,
        "expires_in_s": PLAN_TTL_SECONDS,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _place_exclusive(tmp: Path, dest_dir: Path, name: str, taken: set[str]) -> str:
    """Move tmp into dest_dir under `name` without ever overwriting; picks a new name on collision."""
    while True:
        target = dest_dir / name
        try:
            os.link(tmp, target)  # fails with FileExistsError instead of overwriting
            tmp.unlink()
            return name
        except FileExistsError:
            name = _free_name(name, taken, dest_dir)


def run_export(ctx: JobContext, store: Store, plan: ExportPlan) -> dict:
    plan.destination.mkdir(parents=False, exist_ok=True)
    copied: dict[str, str] = {}
    failed: list[dict] = []
    taken = {e["dest_name"].lower() for e in plan.entries}
    total = len(plan.entries)
    ctx.progress(0, total, "copying")
    tmp: Path | None = None
    try:
        for i, e in enumerate(plan.entries):
            ctx.check_cancelled()
            src = Path(e["source"])
            tmp = plan.destination / f"{PARTIAL_PREFIX}{uuid.uuid4().hex}"
            try:
                if e.get("clip"):
                    from .clips import render_clip

                    render_clip(ctx, src, e["clip"][0], e["clip"][1], "accurate", tmp, e["clip"][0])
                else:
                    shutil.copyfile(src, tmp)  # content only; source opened read-only
                    shutil.copystat(src, tmp, follow_symlinks=True)
                if e["sha256"] and _sha256(tmp) != e["sha256"]:
                    raise OSError("copied file does not match the indexed content hash (source changed?)")
                final = _place_exclusive(tmp, plan.destination, e["dest_name"], taken)
                copied[e["item_id"]] = final
            except OSError as err:
                failed.append({"item_id": e["item_id"], "source": e["source"], "error": str(err)})
            finally:
                if tmp.exists():
                    tmp.unlink()
                tmp = None
            ctx.progress(i + 1, total, f"copied {e['dest_name']}")
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
        cancelled = ctx.cancelled
        manifest_written = None
        if plan.manifest_name and copied:
            manifest = build_manifest(store, plan.selection_id, copied)
            manifest["export"] = {"destination": str(plan.destination), "cancelled": cancelled,
                                  "copied": len(copied), "failed": failed, "skipped": plan.skipped}
            mtmp = plan.destination / f"{PARTIAL_PREFIX}{uuid.uuid4().hex}"
            mtmp.write_text(json.dumps(manifest, indent=2))
            manifest_written = _place_exclusive(mtmp, plan.destination, plan.manifest_name, taken)
        ctx.job.result = {"copied": len(copied), "failed": failed, "skipped": plan.skipped,
                          "manifest": manifest_written, "destination": str(plan.destination),
                          "cancelled": cancelled}
    return ctx.job.result
