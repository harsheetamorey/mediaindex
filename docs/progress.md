# MediaIndex progress

The phases follow `mediaindex-claude-build-guide.md`. A gate is marked passed only after real execution on the target laptop.

| Phase | Status |
|---|---|
| 0 Repository and machine setup | PASSED |
| 1 Real model compatibility | PASSED |
| 2 Model adapter and execution worker | PASSED |
| 3 SQLite library and persistence | PASSED |
| 4 Image import and thumbnails | PASSED |
| 5 Optional demo image pack | PASSED |

---

## Phase 0: Repository and machine setup (2026-10-06)

**Changes:** `pyproject.toml`, `uv.lock`, `.python-version`, `backend/mediaindex/{__init__,app,config,security,__main__}.py`,
`backend/tests/test_health.py`, `frontend/` (Vite React-TS), `.gitignore`, `CLAUDE.md`, `docs/environment.md`, this file.

**Commands and observed results**
- `uv sync`: installed fastapi, uvicorn, starlette 1.7.0, pytest 9.1.1 and httpx
- `uv run pytest -q`: `3 passed`
- `uv run python -m mediaindex` then `curl http://127.0.0.1:8765/api/health`:
  `{"status":"ok","version":"0.1.0","python":"3.12.13","platform":"macOS-14.0-arm64-arm-64bit"}`
- `curl -H "Host: evil.com" .../api/health`: `{"detail":"Host not allowed"}` (403)
- `lsof`: the listener is on `127.0.0.1:8765` only
- `cd frontend && npm run build`: built in 323 ms (dist JS 220 kB)

**Machine:** see `docs/environment.md`. This is the target laptop: an M1 with 8 GB unified memory.

**Limitations:** 8 GB of RAM may be tight for EmbeddingGemma 2. Phase 1 measures it.

**Gate:** PASSED: dependencies install, the health endpoint responds, the frontend builds, and the machine report is accurate.

**Next:** Phase 1, real model compatibility.

---

## Phase 1: Real model compatibility (2026-10-07)

**Changes:** `scripts/model_smoke.py`, `docs/model-compatibility.md`, and pinned deps in `pyproject.toml` and `uv.lock`
(torch 2.14.1, torchvision 0.29.1, torchcodec 0.17.0, transformers 5.19.0, sentence-transformers 6.1.0).

**Commands and observed results** (real weights, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`)
- `uv run python scripts/model_smoke.py --images data/samples/smoke` (first run, download): text, image and mixed all OK on `mps:0` bfloat16
- The same run with `--revision ... --offline`: `all_ok: true` and load 6.0 s
- `--offline --device cpu --dtype float32`: `all_ok: true`. Scores are within 0.003 of MPS and the rankings are identical.
- `--no-audio-encoder`: `all_ok: true`, peak RSS about 1.0 GB
- Text-to-image top-1 is correct for all 3 queries. The mixed query (zebra.jpg + "a musical instrument") moves piano and guitar to ranks 2–3.
- The first attempt failed with `ModuleNotFoundError: EmbeddingGemma2Processor` because torchvision is a hidden requirement. Adding torchvision fixed it.

**Limitations:** this is a smoke check only (6 images, no accuracy claim). Image encoding on MPS takes about 2.5 s/image when cold.
Sample images are macOS built-in pictures converted to JPEG under `data/` (not committed).

**Gate:** PASSED: text, image and native image+text run with real weights, both online and with local-only loading.

**Next:** Phase 2, model adapter and execution worker.

---

## Phase 2: Model adapter and execution worker (2026-10-07)

**Changes:** `backend/mediaindex/model/{profiles,backend,host}.py`, `backend/mediaindex/jobs.py`, `app.py` (model status and job endpoints,
clean shutdown in lifespan), `config.py` (device/precision settings), `backend/tests/test_worker.py`, `backend/tests/real/test_real_model.py`.

**Design**
- `GemmaBackend` exposes `embed_query_texts`, `embed_images` and `embed_image_text`, plus `capabilities()`. It uses the verified prompts from Phase 1,
  re-normalizes in float32, and converts OOM to `ResourceError`. Native image+text runs as one forward pass. An unloaded encoder raises `CapabilityUnavailable`.
- `ModelHost` holds exactly one backend per process. It loads lazily under a load lock, and an inference lock serializes all model use.
  `use(timeout)` raises `ModelBusy` when the lock is not acquired in time.
- `JobRunner` is a single background thread with a bounded queue (`QueueFull`), progress, and cancellation checked between work units.
- `IndexProfile` is frozen and pins model id, revision, dim, precision, encoders, image token budget, prompts and normalization.
  Its `key` is a SHA-256 of canonical JSON, and `ensure_compatible` raises `ProfileMismatch`.
- The default profile is text+image encoders, bf16 on MPS and fp32 on CPU. `FakeBackend` is for unit tests only and requires explicit `MEDIAINDEX_FAKE_MODEL=1` at runtime.

**Commands and observed results**
- `uv run pytest -q`: `10 passed, 1 deselected` (mocked worker tests: single load under 8 concurrent requests, ModelBusy on timeout,
  profile mismatch for precision/revision/dim/token budget/prompt, float16 rejected, job progress/cancel/failure, bounded queue)
- `uv run pytest -m real_model -q`: `1 passed`. **REAL MODEL**: text, image and mixed on MPS bf16 with load_count == 1 after repeated requests, and repeated text embeddings were identical.

**Limitations:** jobs live in memory until Phase 3 persists them. Indexing and query inference share one lock, so a query waits for at most one indexing batch.

**Gate:** PASSED.

**Next:** Phase 3, SQLite library and persistence.

---

## Phase 3: SQLite library and persistence (2026-10-07)

**Changes:** `backend/mediaindex/db.py` (WAL connection, `PRAGMA user_version` migrations, serialized write transactions with full rollback),
`backend/mediaindex/store.py` (libraries, assets, profiles, embeddings, jobs), app wiring (job persistence, interrupted-job recovery on startup),
`backend/tests/test_store.py`.

**Design**
- Tables: `libraries`, `index_profiles`, `assets` (immutable UUID per (library, root-relative path), media type, size, mtime, sha256,
  width/height/duration, status pending|indexed|failed|missing, error), `embeddings`, and `jobs`.
- Vectors are float32 BLOBs in `embeddings` with explicit `id`, `asset_id`, `profile_key`, `modality`, `segment_index` and start/end seconds.
  A write replaces an asset's vectors and sets `status='indexed'` **in one transaction**. Non-finite vectors are refused.
- `load_matrix` returns `(embedding_ids, asset_ids, matrix)` built row by row from one query, so SQLite row order is never relied on.
  It only includes `status='indexed'` assets under the requested profile.
- When content changes (hash differs), old vectors are dropped and the asset returns to pending under the same ID. Identical content in different paths
  gets separate asset rows, and vectors can be reused through `source_hash` + profile.
- Removing an asset or a library deletes index rows only, never files. Library `generation` counters invalidate caches.
- Recovery: on startup, jobs left `queued`/`running` become `interrupted`, and their assets stay `pending` (not searchable) until a re-import finishes.

**Commands and observed results**
- `uv run pytest -q`: `19 passed, 1 deselected`. Covers migration idempotence, restart identity, ID/vector alignment after out-of-order writes and deletes,
  rollback after a NaN segment and after an exception mid-transaction (asset stays pending with no vectors), profile filtering,
  content-change invalidation, duplicate reuse, original-file preservation, and interrupted-job recovery.

**Limitations:** search loads every vector into memory per profile. That is fine for tens of thousands of images, and Phase 16 measures it.

**Gate:** PASSED.

**Next:** Phase 4, image import and thumbnails.

---

## Phase 4: Image import and thumbnails (2026-10-07)

**Changes:** `backend/mediaindex/paths.py` (root validation, traversal and symlink guards, safe walk), `backend/mediaindex/media/images.py`
(format sniffing, size and pixel limits, EXIF orientation, thumbnails), `backend/mediaindex/ingest.py` (idempotent import job),
`app.py` (library CRUD, import job, asset listing, thumbnail and file serving by asset ID), `backend/tests/{conftest,test_ingest}.py`.

**Behaviour**
- Pillow decoders verified: JPEG, PNG and WebP are all available. Formats are sniffed from content (`JPEG|PNG|WEBP|MPO`), not trusted from the extension.
  Limits are 200 MB per file and 80 MP (decompression-bomb guard). Corrupt or truncated files are marked `failed` with the error text.
- Discovery covers only the explicitly selected root. It refuses `/` and `$HOME` itself, never follows symlinked dirs, accepts symlinked files only when their target is inside the root,
  and skips hidden files.
- Thumbnails are 384 px JPEGs at `data/thumbs/<hash[:2]>/<sha256>.jpg`, shared by duplicate content. Originals are opened read-only.
- Re-import skips files with unchanged size and mtime. Changed content keeps its asset ID and returns to pending. Vanished files become `missing`, and they return
  to pending when they reappear. Cancellation is checked between files, and re-running resumes.
- Files are served only through `/api/assets/{id}/file|thumbnail`, resolved from the stored root and relative path with escape checks. A missing original returns 410.
  Deleting a library removes only index rows.

**Commands and observed results**
- `uv run pytest -q`: `25 passed, 1 deselected` (mixed formats, corrupt/truncated, symlink escape, hidden files, EXIF rotation 80x40 to 40x80,
  duplicate content, originals byte-identical with identical mtime, idempotent re-import, change/missing tracking, cancel then resume, API serving,
  traversal 404, missing 410, delete-preserves-files)
- Live server on a real folder (`data/samples/p4`: 6 JPEG photos + 1 garbage file):
  first import `discovered 7, new 6, failed 1 (broken.jpg: cannot identify image file)`. **After a server restart**, re-import gave `new 0, unchanged 7`.
  `shasum -c` confirms all 7 originals are unchanged.

**Limitations:** assets stay `pending` because embedding is connected in Phase 6. Cancelling during a live import was tested in unit tests only, since the
real folder imports in under a second.

**Gate:** PASSED.

**Next:** Phase 5, optional demo image pack.

---

## Phase 5: Optional demo image pack (2026-10-07)

**Changes:** `scripts/download_demo.py`, `manifests/stockimages-cc0-500.json` (committed selection: rows and hashes, no images),
`THIRD_PARTY_DATA.md`, importer support for the `mediaindex-provenance.json` sidecar (stored as asset `source` metadata), and the `demo` optional dependency (pyarrow).

**Dataset inspection:** revision `206f357…`, 3,999 rows in 2 parquet shards (422 MB + 467 MB) with 100-row row groups. Features are `image` and `tags`.
The card declares CC0-1.0 with no per-image provenance. Some rows contain HTML error pages instead of images, and the downloader now excludes them.

**Commands and observed results**
- `uv run --extra demo python scripts/download_demo.py --write-manifest`: 5 row groups ((1,14), (0,1), (0,16), (1,0), (1,4)), 500 images,
  **about 114.8 MB transferred** (versus about 889 MB for the full dataset), 0 rows excluded
- Rerun into a fresh folder: `selection matches committed manifest`, and `shasum` across both folders matches for all 500 files
- Live API import of `data/demo/stockimages-cc0`: `discovered 500, new 500, failed 0`. **All 500 assets have a source record**
  (dataset URL and revision, row, publisher-declared license). Tags stay separate as `inspection_tags`.
- `uv run pytest -q`: `26 passed`

**Limitations:** publisher-declared CC0 only, with no rights audit (see THIRD_PARTY_DATA.md). The app works without this pack.

**Gate:** PASSED.

**Next:** Phase 6, image embeddings and text search.
