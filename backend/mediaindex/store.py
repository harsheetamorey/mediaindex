"""Repository layer over the SQLite schema: libraries, assets, profiles, embeddings, jobs."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .db import Database, now
from .model.profiles import IndexProfile

INDEXED = "indexed"
PENDING = "pending"
FAILED = "failed"
MISSING = "missing"


def new_id() -> str:
    return uuid.uuid4().hex


def vec_to_blob(v: np.ndarray) -> bytes:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if not np.isfinite(v).all():
        raise ValueError("refusing to store non-finite vector")
    return v.tobytes()


def blob_to_vec(b: bytes, dim: int) -> np.ndarray:
    v = np.frombuffer(b, dtype=np.float32)
    if v.shape[0] != dim:
        raise ValueError(f"stored vector has {v.shape[0]} dims, expected {dim}")
    return v


@dataclass
class VectorMatrix:
    """Embedding IDs, asset IDs and an (n, dim) matrix whose rows correspond exactly to the ID arrays."""

    embedding_ids: list[str]
    asset_ids: list[str]
    matrix: np.ndarray
    starts: list[float | None]
    ends: list[float | None]
    modalities: list[str]
    generation: int


class Store:
    def __init__(self, db: Database):
        self.db = db

    # ---- libraries -------------------------------------------------------------
    def create_library(self, name: str, root: Path) -> dict:
        root = Path(root).resolve()
        existing = self.db.one("SELECT * FROM libraries WHERE root_path = ?", (str(root),))
        if existing:
            return dict(existing)
        lid = new_id()
        with self.db.tx() as c:
            c.execute("INSERT INTO libraries(id, name, root_path, created_at) VALUES (?,?,?,?)",
                      (lid, name, str(root), now()))
        return self.get_library(lid)

    def get_library(self, library_id: str) -> dict | None:
        r = self.db.one("SELECT * FROM libraries WHERE id = ?", (library_id,))
        return dict(r) if r else None

    def list_libraries(self) -> list[dict]:
        rows = self.db.query("""
            SELECT l.*,
              (SELECT COUNT(*) FROM assets a WHERE a.library_id = l.id) AS asset_count,
              (SELECT COUNT(*) FROM assets a WHERE a.library_id = l.id AND a.status = 'indexed') AS indexed_count
            FROM libraries l ORDER BY created_at""")
        return [dict(r) for r in rows]

    def delete_library(self, library_id: str) -> None:
        """Removes index records only. Original files are never touched."""
        with self.db.tx() as c:
            c.execute("DELETE FROM libraries WHERE id = ?", (library_id,))

    def set_library_watch(self, library_id: str, watch: bool) -> None:
        with self.db.tx() as c:
            c.execute("UPDATE libraries SET watch = ? WHERE id = ?", (int(watch), library_id))

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        r = self.db.one("SELECT value FROM app_settings WHERE key = ?", (key,))
        return r["value"] if r else default

    def set_setting(self, key: str, value: str) -> None:
        with self.db.tx() as c:
            c.execute("INSERT INTO app_settings(key, value) VALUES (?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def bump_generation(self, c, library_id: str) -> None:
        c.execute("UPDATE libraries SET generation = generation + 1 WHERE id = ?", (library_id,))

    # ---- profiles --------------------------------------------------------------
    def register_profile(self, profile: IndexProfile) -> str:
        with self.db.tx() as c:
            c.execute("INSERT OR IGNORE INTO index_profiles(key, json, created_at) VALUES (?,?,?)",
                      (profile.key, json.dumps(profile.to_dict(), sort_keys=True), now()))
        return profile.key

    def rekey_compatible_profiles(self, current: IndexProfile) -> dict[str, int]:
        """Move vectors stored under older profile keys that describe the *same* vector space.

        A stored profile is compatible when re-parsing it with today's key rules yields the current
        key (i.e. it differs only in KEY_EXCLUDED_FIELDS, verified not to affect vectors).
        Returns {old_key: rows_moved}.
        """
        moved: dict[str, int] = {}
        rows = self.db.query("SELECT key, json FROM index_profiles WHERE key != ?", (current.key,))
        for r in rows:
            try:
                old = IndexProfile.from_dict(json.loads(r["json"]))
            except (TypeError, ValueError):
                continue
            if old.key != current.key:
                continue
            with self.db.tx() as c:
                n = c.execute("UPDATE OR IGNORE embeddings SET profile_key=? WHERE profile_key=?",
                              (current.key, r["key"])).rowcount
                c.execute("UPDATE libraries SET generation = generation + 1")
            moved[r["key"]] = n
        return moved

    def get_profile(self, key: str) -> IndexProfile | None:
        r = self.db.one("SELECT json FROM index_profiles WHERE key = ?", (key,))
        return IndexProfile.from_dict(json.loads(r["json"])) if r else None

    # ---- assets ----------------------------------------------------------------
    def upsert_asset(self, library_id: str, rel_path: str, media_type: str, *, size: int, mtime_ns: int,
                     content_hash: str | None, width=None, height=None, duration=None, meta: dict | None = None,
                     c=None) -> tuple[str, bool]:
        """Insert or update an asset by (library, rel_path). Returns (asset_id, content_changed).

        The asset ID is immutable for a given (library, rel_path). If content changed, existing
        embeddings are dropped and the asset returns to pending.
        """
        def _do(c):
            row = c.execute("SELECT id, content_hash FROM assets WHERE library_id = ? AND rel_path = ?",
                            (library_id, rel_path)).fetchone()
            t = now()
            meta_json = json.dumps(meta) if meta else None
            if row is None:
                aid = new_id()
                c.execute("""INSERT INTO assets(id, library_id, rel_path, media_type, size, mtime_ns, content_hash,
                             width, height, duration, status, meta_json, created_at, updated_at)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                          (aid, library_id, rel_path, media_type, size, mtime_ns, content_hash, width, height,
                           duration, PENDING, meta_json, t, t))
                return aid, True
            aid = row["id"]
            changed = row["content_hash"] != content_hash
            if changed:
                c.execute("DELETE FROM embeddings WHERE asset_id = ?", (aid,))
                c.execute("""UPDATE assets SET media_type=?, size=?, mtime_ns=?, content_hash=?, width=?, height=?,
                             duration=?, status=?, error=NULL, meta_json=?, updated_at=? WHERE id=?""",
                          (media_type, size, mtime_ns, content_hash, width, height, duration, PENDING,
                           meta_json, t, aid))
                self.bump_generation(c, library_id)
            else:
                c.execute("UPDATE assets SET size=?, mtime_ns=?, updated_at=?, status=CASE WHEN status=? THEN ? ELSE status END WHERE id=?",
                          (size, mtime_ns, t, MISSING, PENDING, aid))
            return aid, changed

        if c is not None:
            return _do(c)
        with self.db.tx() as c2:
            return _do(c2)

    def get_asset(self, asset_id: str) -> dict | None:
        r = self.db.one("SELECT * FROM assets WHERE id = ?", (asset_id,))
        return dict(r) if r else None

    def list_assets(self, library_id: str, status: str | None = None, media_type: str | None = None) -> list[dict]:
        sql = "SELECT * FROM assets WHERE library_id = ?"
        params: list = [library_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        if media_type:
            sql += " AND media_type = ?"
            params.append(media_type)
        return [dict(r) for r in self.db.query(sql + " ORDER BY rel_path", params)]

    def find_by_hash(self, content_hash: str) -> list[dict]:
        return [dict(r) for r in self.db.query("SELECT * FROM assets WHERE content_hash = ?", (content_hash,))]

    def set_asset_status(self, asset_id: str, status: str, error: str | None = None, c=None) -> None:
        def _do(c):
            c.execute("UPDATE assets SET status=?, error=?, updated_at=? WHERE id=?", (status, error, now(), asset_id))
            lib = c.execute("SELECT library_id FROM assets WHERE id=?", (asset_id,)).fetchone()
            if lib:
                self.bump_generation(c, lib["library_id"])
        if c is not None:
            return _do(c)
        with self.db.tx() as c2:
            _do(c2)

    def remove_asset(self, asset_id: str) -> None:
        """Remove from the index only; the original file is not deleted."""
        with self.db.tx() as c:
            lib = c.execute("SELECT library_id FROM assets WHERE id=?", (asset_id,)).fetchone()
            c.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
            if lib:
                self.bump_generation(c, lib["library_id"])

    # ---- embeddings ------------------------------------------------------------
    def write_embeddings(self, asset_id: str, profile_key: str, modality: str, vectors: np.ndarray,
                         segments: list[tuple[float | None, float | None]] | None = None,
                         source_hash: str | None = None, mark_indexed: bool = True, c=None) -> list[str]:
        """Atomically replace an asset's vectors for (profile, modality) and mark it indexed."""
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        segments = segments or [(None, None)] * len(vectors)
        if len(segments) != len(vectors):
            raise ValueError("segments and vectors length mismatch")

        def _do(c):
            c.execute("DELETE FROM embeddings WHERE asset_id=? AND profile_key=? AND modality=?",
                      (asset_id, profile_key, modality))
            ids = []
            t = now()
            for i, (v, (s, e)) in enumerate(zip(vectors, segments)):
                eid = new_id()
                c.execute("""INSERT INTO embeddings(id, asset_id, profile_key, modality, segment_index, start_s, end_s,
                             source_hash, dim, vector, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                          (eid, asset_id, profile_key, modality, i, s, e, source_hash, v.shape[0], vec_to_blob(v), t))
                ids.append(eid)
            if mark_indexed:
                self.set_asset_status(asset_id, INDEXED, None, c=c)
            return ids

        if c is not None:
            return _do(c)
        with self.db.tx() as c2:
            return _do(c2)

    def reusable_vectors(self, source_hash: str, profile_key: str, modality: str) -> np.ndarray | None:
        """Vectors already computed for identical content under the same profile (duplicate reuse)."""
        row = self.db.one("""SELECT asset_id FROM embeddings WHERE source_hash=? AND profile_key=? AND modality=?
                             LIMIT 1""", (source_hash, profile_key, modality))
        if not row:
            return None
        rows = self.db.query("""SELECT dim, vector FROM embeddings WHERE asset_id=? AND profile_key=? AND modality=?
                                ORDER BY segment_index""", (row["asset_id"], profile_key, modality))
        return np.vstack([blob_to_vec(r["vector"], r["dim"]) for r in rows])

    def reusable_segments(self, source_hash: str, profile_key: str, modality: str,
                          window: tuple[float, float] | None = None) -> tuple[np.ndarray, list] | None:
        """Vectors + (start, end) of an asset with identical content under the same profile."""
        row = self.db.one("""SELECT e.asset_id FROM embeddings e JOIN assets a ON a.id = e.asset_id
                             WHERE e.source_hash=? AND e.profile_key=? AND e.modality=? AND a.status='indexed'
                             LIMIT 1""", (source_hash, profile_key, modality))
        if not row:
            return None
        rows = self.db.query("""SELECT dim, vector, start_s, end_s FROM embeddings WHERE asset_id=? AND profile_key=?
                                AND modality=? ORDER BY segment_index""", (row["asset_id"], profile_key, modality))
        return (np.vstack([blob_to_vec(r["vector"], r["dim"]) for r in rows]),
                [(r["start_s"], r["end_s"]) for r in rows])

    def get_asset_vectors(self, asset_id: str, profile_key: str, modality: str | None = None) -> np.ndarray | None:
        sql = "SELECT dim, vector FROM embeddings WHERE asset_id=? AND profile_key=?"
        params: list = [asset_id, profile_key]
        if modality:
            sql += " AND modality=?"
            params.append(modality)
        rows = self.db.query(sql + " ORDER BY modality, segment_index", params)
        if not rows:
            return None
        return np.vstack([blob_to_vec(r["vector"], r["dim"]) for r in rows])

    def load_matrix(self, profile_key: str, library_ids: list[str] | None = None,
                    modalities: list[str] | None = None) -> VectorMatrix:
        """Searchable vectors: only assets with status 'indexed' under the given profile."""
        sql = """SELECT e.id, e.asset_id, e.dim, e.vector, e.start_s, e.end_s, e.modality
                 FROM embeddings e JOIN assets a ON a.id = e.asset_id
                 WHERE e.profile_key = ? AND a.status = 'indexed'"""
        params: list = [profile_key]
        if library_ids:
            sql += f" AND a.library_id IN ({','.join('?' * len(library_ids))})"
            params += library_ids
        if modalities:
            sql += f" AND e.modality IN ({','.join('?' * len(modalities))})"
            params += modalities
        with self.db.tx() as c:  # consistent snapshot of generation + rows
            gen = c.execute("SELECT COALESCE(SUM(generation),0) FROM libraries").fetchone()[0]
            rows = c.execute(sql + " ORDER BY e.id", params).fetchall()
        if not rows:
            return VectorMatrix([], [], np.zeros((0, 0), np.float32), [], [], [], gen)
        dims = {r["dim"] for r in rows}
        if len(dims) != 1:
            raise ValueError(f"mixed vector dimensions under one profile: {dims}")
        mat = np.vstack([blob_to_vec(r["vector"], r["dim"]) for r in rows])
        return VectorMatrix([r["id"] for r in rows], [r["asset_id"] for r in rows], mat,
                            [r["start_s"] for r in rows], [r["end_s"] for r in rows],
                            [r["modality"] for r in rows], gen)

    # ---- object counts (Ask assistant) ---------------------------------------------
    def save_detections(self, content_hash: str, detector: str, boxes: dict[str, list]) -> None:
        """boxes: label -> [(score, [x0, y0, x1, y1]), ...]; the count is the number of boxes."""
        with self.db.tx() as c:
            c.execute("DELETE FROM detections WHERE content_hash=? AND detector=?", (content_hash, detector))
            c.executemany("INSERT INTO detections(content_hash, detector, label, count, boxes_json) VALUES (?,?,?,?,?)",
                          [(content_hash, detector, k, len(v), json.dumps(v)) for k, v in boxes.items() if v])
            c.execute("INSERT OR REPLACE INTO detection_runs(content_hash, detector, created_at) VALUES (?,?,?)",
                      (content_hash, detector, now()))

    @staticmethod
    def _scope(library_ids: list[str] | None) -> tuple[str, list]:
        sql = "a.media_type='image' AND a.status='indexed'"
        if library_ids:
            return sql + f" AND a.library_id IN ({','.join('?' * len(library_ids))})", list(library_ids)
        return sql, []

    def detection_coverage(self, detector: str, library_ids: list[str] | None = None) -> dict:
        where, params = self._scope(library_ids)
        r = self.db.one(f"""SELECT COUNT(*) AS photos, COUNT(dr.content_hash) AS checked FROM assets a
                            LEFT JOIN detection_runs dr ON dr.content_hash=a.content_hash AND dr.detector=?
                            WHERE {where}""", [detector, *params])
        return {"photos": r["photos"], "checked": r["checked"]}

    def count_objects(self, detector: str, label: str, library_ids: list[str] | None = None,
                      sure: float = 0.0) -> dict:
        """Photos (files) where the detector found `label`. A photo counts if it has a box scoring >= `sure`;
        photos with only lower-scoring boxes are returned separately as `maybe_ids`."""
        where, params = self._scope(library_ids)
        rows = self.db.query(f"""SELECT a.id, a.rel_path, d.boxes_json FROM assets a
                                 JOIN detections d ON d.content_hash=a.content_hash AND d.detector=? AND d.label=?
                                 WHERE {where}""", [detector, label, *params])
        counts, maybe = {}, {}
        for r in rows:
            scores = [b[0] for b in json.loads(r["boxes_json"] or "[]")]
            n = sum(sc >= sure for sc in scores)
            if n:
                counts[r["id"]] = n
            elif scores:
                maybe[r["id"]] = max(scores)
        ids = sorted(counts, key=lambda a: -counts[a])
        return {"photos": len(ids), "objects": sum(counts.values()), "asset_ids": ids, "counts": counts,
                "maybe_ids": sorted(maybe, key=lambda a: -maybe[a])}

    def detections_for(self, content_hash: str, detector: str, sure: float = 0.0) -> dict[str, int] | None:
        """Objects counted in one photo (boxes scoring >= `sure`), or None if it hasn't been checked."""
        if not self.db.one("SELECT 1 FROM detection_runs WHERE content_hash=? AND detector=?", (content_hash, detector)):
            return None
        out = {}
        for r in self.db.query("SELECT label, boxes_json FROM detections WHERE content_hash=? AND detector=?",
                               (content_hash, detector)):
            n = sum(b[0] >= sure for b in json.loads(r["boxes_json"] or "[]"))
            if n:
                out[r["label"]] = n
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def boxes_for(self, content_hash: str, detector: str, labels: list[str] | None = None) -> list[dict]:
        rows = self.db.query("SELECT label, boxes_json FROM detections WHERE content_hash=? AND detector=?",
                             (content_hash, detector))
        out = []
        for r in rows:
            if (labels is None or r["label"] in labels) and r["boxes_json"]:
                out += [{"label": r["label"], "score": s, "box": b} for s, b in json.loads(r["boxes_json"])]
        return out

    def verifications(self, hashes: list[str], label: str, verifier: str) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for i in range(0, len(hashes), 500):
            part = hashes[i:i + 500]
            rows = self.db.query(f"""SELECT content_hash, present FROM verifications WHERE label=? AND verifier=?
                                     AND content_hash IN ({','.join('?' * len(part))})""", [label, verifier, *part])
            out.update({r["content_hash"]: bool(r["present"]) for r in rows})
        return out

    def save_verification(self, content_hash: str, label: str, verifier: str, present: bool) -> None:
        with self.db.tx() as c:
            c.execute("INSERT OR REPLACE INTO verifications(content_hash, label, verifier, present, created_at) "
                      "VALUES (?,?,?,?,?)", (content_hash, label, verifier, int(present), now()))

    def object_summary(self, detector: str, library_ids: list[str] | None = None, sure: float = 0.0) -> list[dict]:
        """Per label: photos with a box scoring >= `sure`, and how many such boxes in total."""
        where, params = self._scope(library_ids)
        rows = self.db.query(f"""SELECT d.label, d.boxes_json FROM assets a
                                 JOIN detections d ON d.content_hash=a.content_hash AND d.detector=?
                                 WHERE {where}""", [detector, *params])
        agg: dict[str, list[int]] = {}
        for r in rows:
            n = sum(b[0] >= sure for b in json.loads(r["boxes_json"] or "[]"))
            if n:
                a = agg.setdefault(r["label"], [0, 0])
                a[0] += 1
                a[1] += n
        return [{"label": k, "photos": v[0], "objects": v[1]}
                for k, v in sorted(agg.items(), key=lambda kv: (-kv[1][0], kv[0]))]

    # ---- jobs ------------------------------------------------------------------
    def save_job(self, job: dict, library_id: str | None = None) -> None:
        with self.db.tx() as c:
            c.execute("""INSERT INTO jobs(id, kind, library_id, status, done, total, message, error, result_json,
                         created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                         ON CONFLICT(id) DO UPDATE SET status=excluded.status, done=excluded.done,
                         total=excluded.total, message=excluded.message, error=excluded.error,
                         result_json=excluded.result_json, updated_at=excluded.updated_at,
                         library_id=COALESCE(excluded.library_id, jobs.library_id)""",
                      (job["id"], job["kind"], library_id, job["status"], job.get("done", 0), job.get("total", 0),
                       job.get("message"), job.get("error"),
                       json.dumps(job["result"]) if job.get("result") is not None else None,
                       job.get("created", now()), now()))

    def get_job(self, job_id: str) -> dict | None:
        r = self.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        return dict(r) if r else None

    def recover_interrupted_jobs(self) -> int:
        """On startup: jobs left queued/running by a previous process become 'interrupted'.

        Their assets remain 'pending' and are not searchable until a re-import completes them.
        """
        with self.db.tx() as c:
            cur = c.execute("""UPDATE jobs SET status='interrupted', updated_at=?,
                               message='process exited before completion; re-run import to resume'
                               WHERE status IN ('queued','running')""", (now(),))
            return cur.rowcount


def index_state(store: Store, profile_key: str, library_ids: list[str] | None, media_types: list[str]) -> dict:
    """Describe how searchable the selected libraries are under the active profile."""
    where = "a.media_type IN (%s)" % ",".join("?" * len(media_types))
    params: list = list(media_types)
    if library_ids:
        where += " AND a.library_id IN (%s)" % ",".join("?" * len(library_ids))
        params += library_ids
    rows = store.db.query(f"SELECT a.status, COUNT(*) AS n FROM assets a WHERE {where} GROUP BY a.status", params)
    counts = {r["status"]: r["n"] for r in rows}
    total = sum(counts.values())
    other = store.db.one(f"""SELECT COUNT(DISTINCT e.asset_id) AS n FROM embeddings e JOIN assets a ON a.id=e.asset_id
                             WHERE {where} AND e.profile_key != ?""", params + [profile_key])["n"]
    current = store.db.one(f"""SELECT COUNT(DISTINCT e.asset_id) AS n FROM embeddings e JOIN assets a ON a.id=e.asset_id
                               WHERE {where} AND e.profile_key = ? AND a.status='indexed'""",
                           params + [profile_key])["n"]
    if total == 0:
        state = "empty"
    elif current == 0 and other > 0:
        state = "incompatible"
    elif current == 0:
        state = "not_indexed"
    elif current < total - counts.get("failed", 0) - counts.get("missing", 0):
        state = "partial"
    else:
        state = "ready"
    return {"state": state, "total": total, "searchable": current, "other_profile_vectors": other,
            "by_status": counts}
