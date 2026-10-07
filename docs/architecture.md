# Architecture

```
Browser (React UI, served from the same origin)
   │  HTTP on 127.0.0.1:8765 only (Host/Origin checks, CSP, 60 MB body limit)
   ▼
FastAPI app (one process, one worker, flock on <data>/.mediaindex.lock)
   ├── API: libraries · jobs · assets (by ID) · search · selections · clips · capabilities
   ├── JobRunner: one background thread, bounded queue, progress + cancel between units
   ├── ModelHost: ONE EmbeddingGemma 2 instance, inference serialized by a lock
   │      (queries wait ≤ MEDIAINDEX_QUERY_WAIT s, otherwise 503 "model busy")
   ├── Store (SQLite, WAL, versioned migrations): libraries, assets, embeddings (float32 BLOBs), jobs,
   │      index_profiles, selections
   ├── MatrixCache: per (profile, libraries, modalities) matrix + ID arrays, invalidated by generation counters
   └── Media tools: Pillow (images), FFmpeg/ffprobe subprocesses (audio, video, clips; argument lists,
          protocol whitelist file,pipe)
```

## Data flow: import
1. The user adds an absolute folder path. It is validated (not `/` or `$HOME`) and stored as a library root.
2. An import job walks the root without following symlinked directories. Symlinked files must stay inside the root, and hidden files are skipped.
3. Each file is hashed (SHA-256), probed and validated, and gets a thumbnail (an image, a waveform, or a video frame plus per-window frames).
4. Embedding:
   - images go in batches of 2;
   - audio uses 10 s / 5 s windows of mono 16 kHz audio;
   - video uses 8 s / 4 s windows: 8 frames sampled at 1 fps go through the model's native video input, the window's soundtrack goes through the audio encoder, and joint video+audio is optional.
5. An asset's vectors and its `indexed` status are written **in one transaction**. Interrupted or cancelled files stay `pending` and are never
   searchable. Unchanged files (same size and mtime) are skipped on re-import. Identical content reuses vectors.

## Data flow: search
1. The query is embedded under the active **index profile**:
   - text uses the `SearchQuery` prompt;
   - an image reference uses the image path, or the stored vector for a library asset;
   - image+text, audio+text and video windows each use their native interface.
2. Exact cosine similarity (a dot product of L2-normalized float32 vectors) runs against the cached matrix for the requested modalities.
3. Windows are grouped per source file, and windows overlapping a kept window by ≥ 50% are merged. Multi-type searches are grouped by media type by default.
4. Results carry asset IDs, preview URLs, raw similarity (never shown as a percentage), the window span and timing.

## Index profiles
`IndexProfile` pins the model ID and revision, dimension, precision, prompts, image and video token budgets, video fps and normalization. Its `key` is a
hash of the fields that define the vector space. Searches only use vectors stored under the current key, and anything else returns 409.
Fields verified not to change vectors (the loaded encoder set and the windowing) are excluded from the key, and older compatible profiles are re-keyed at startup.

## Files and directories
| Path | Contents |
|---|---|
| `backend/mediaindex/` | app (`app.py`, `api_*.py`), `model/` (profiles, backend, host), `store.py`, `db.py`, `ingest.py`, `indexer.py`, `media/`, `clips.py`, `selections.py` |
| `frontend/src/` | React UI (`App.tsx`, `components/`, `lib/`) |
| `scripts/` | smoke checks, demo-data builders, UI/offline/robustness checks |
| `evaluation/` | frozen queries, run/label/report tools, benchmarks |
| `data/` (git-ignored) | SQLite DB, thumbnails, temp uploads, demo/bench data |
