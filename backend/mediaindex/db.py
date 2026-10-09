"""SQLite persistence with versioned migrations.

Vectors are stored as float32 BLOBs keyed by explicit embedding IDs in the same database,
so asset status and vectors are written in one transaction. Search matrices are always built
together with their ID arrays from a single query; SQLite row order is never assumed.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

MIGRATIONS: list[str] = [
    # v1
    """
    CREATE TABLE libraries (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        root_path TEXT NOT NULL UNIQUE,
        generation INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    );
    CREATE TABLE index_profiles (
        key TEXT PRIMARY KEY,
        json TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE TABLE assets (
        id TEXT PRIMARY KEY,
        library_id TEXT NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
        rel_path TEXT NOT NULL,
        media_type TEXT NOT NULL,            -- image | audio | video
        size INTEGER,
        mtime_ns INTEGER,
        content_hash TEXT,
        width INTEGER,
        height INTEGER,
        duration REAL,
        status TEXT NOT NULL DEFAULT 'pending', -- pending | indexed | failed | missing
        error TEXT,
        meta_json TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        UNIQUE(library_id, rel_path)
    );
    CREATE INDEX assets_hash ON assets(content_hash);
    CREATE INDEX assets_lib_status ON assets(library_id, status);
    CREATE TABLE embeddings (
        id TEXT PRIMARY KEY,
        asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
        profile_key TEXT NOT NULL REFERENCES index_profiles(key),
        modality TEXT NOT NULL,               -- image | audio | video-visual | video-audio | video-joint
        segment_index INTEGER NOT NULL DEFAULT 0,
        start_s REAL,
        end_s REAL,
        source_hash TEXT,
        dim INTEGER NOT NULL,
        vector BLOB NOT NULL,
        created_at REAL NOT NULL,
        UNIQUE(asset_id, profile_key, modality, segment_index)
    );
    CREATE INDEX emb_profile ON embeddings(profile_key, modality);
    CREATE INDEX emb_source ON embeddings(source_hash, profile_key, modality);
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        library_id TEXT,
        status TEXT NOT NULL,
        done INTEGER NOT NULL DEFAULT 0,
        total INTEGER NOT NULL DEFAULT 0,
        message TEXT,
        error TEXT,
        result_json TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    """,
    # v2: selections (asset_id is deliberately not a cascading FK: items survive index removal with a snapshot)
    """
    CREATE TABLE selections (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE TABLE selection_items (
        id TEXT PRIMARY KEY,
        selection_id TEXT NOT NULL REFERENCES selections(id) ON DELETE CASCADE,
        asset_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        start_s REAL,
        end_s REAL,
        snapshot_json TEXT NOT NULL,
        query_context_json TEXT,
        added_at REAL NOT NULL
    );
    CREATE UNIQUE INDEX sel_item_unique ON selection_items(selection_id, asset_id, IFNULL(start_s, -1), IFNULL(end_s, -1));
    CREATE INDEX sel_item_order ON selection_items(selection_id, position);
    """,
    # v3: watched folders and app preferences
    """
    ALTER TABLE libraries ADD COLUMN watch INTEGER NOT NULL DEFAULT 0;
    CREATE TABLE app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
    # v4: object counts for the Ask assistant (keyed by content hash, so duplicates share one detection run)
    """
    CREATE TABLE detection_runs (
        content_hash TEXT NOT NULL,
        detector TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (content_hash, detector)
    );
    CREATE TABLE detections (
        content_hash TEXT NOT NULL,
        detector TEXT NOT NULL,
        label TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY (content_hash, detector, label)
    );
    CREATE INDEX detections_label ON detections(detector, label);
    """,
    # v5: where each counted object is, so answers can show it
    """
    ALTER TABLE detections ADD COLUMN boxes_json TEXT;
    """,
]


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self.migrate()

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys = ON")
            c.execute("PRAGMA journal_mode = WAL")
            c.execute("PRAGMA synchronous = NORMAL")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self):
        """Serialized write transaction; rolls back fully on any exception."""
        with self._write_lock:
            c = self.conn()
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
            except BaseException:
                c.execute("ROLLBACK")
                raise
            else:
                c.execute("COMMIT")

    def query(self, sql: str, params=()) -> list[sqlite3.Row]:
        return self.conn().execute(sql, params).fetchall()

    def one(self, sql: str, params=()) -> sqlite3.Row | None:
        return self.conn().execute(sql, params).fetchone()

    @property
    def version(self) -> int:
        return self.conn().execute("PRAGMA user_version").fetchone()[0]

    def migrate(self) -> None:
        with self._write_lock:
            c = self.conn()
            current = c.execute("PRAGMA user_version").fetchone()[0]
            for v, script in enumerate(MIGRATIONS[current:], start=current + 1):
                c.execute("BEGIN IMMEDIATE")
                try:
                    for stmt in [s for s in script.split(";") if s.strip()]:
                        c.execute(stmt)
                    c.execute(f"PRAGMA user_version = {v}")
                    c.execute("COMMIT")
                except BaseException:
                    c.execute("ROLLBACK")
                    raise

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None


def now() -> float:
    return time.time()
