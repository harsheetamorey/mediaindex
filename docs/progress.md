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
| 6 Image embeddings and text search | PASSED |
| 7 Reference-image search | PASSED |
| 8 Image plus text refinement | PASSED (quality caveats documented) |
| 9 Creator interface: **Version 1** | PASSED |
| 10 Selections and image exports | PASSED |
| 11 Audio inference and segmentation | PASSED |
| 12 Audio search and playback | PASSED |
| 13 Cross-modal workspace | PASSED (experimental modes labelled) |
| 14 Video ingestion and indexing | PASSED |
| 15 Video-moment search and clip export | PASSED |

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

---

## Phase 6: Image embeddings and text search (2026-10-07)

**Changes:** `backend/mediaindex/indexer.py` (batched real image embedding during import, duplicate-content vector reuse, per-item fallback),
`backend/mediaindex/search.py` (exact dot-product ranking over L2-normalized vectors, generation-checked matrix cache),
`backend/mediaindex/api_search.py` (`POST /api/search/text`), `store.index_state`, ingest changes (re-embed when vectors for the
*current* profile are missing), `scripts/search_check.py`, `backend/tests/test_search.py`.

**API:** `POST /api/search/text {text, library_ids?, media_types: ["image"], limit≤200}` returns asset IDs, preview URLs, raw `similarity`,
`timing_ms.query_embedding` and `timing_ms.ranking` (measured separately), `candidates_searched`, `index_state`
(`ready|partial|empty|not_indexed|incompatible`), and a note that similarity is not a probability. An incompatible profile returns 409, and a busy model returns 503.

**Commands and observed results (REAL MODEL, MPS bf16)**
- Live import of the 500-image demo pack: `embedded 500, failed 0` in **about 913 s (about 1.8 s/image)**, plus the 6-image smoke folder
- `uv run python scripts/search_check.py text ... --library <demo>` (top-1 shown with its publisher tags, which are used only for inspection):
  - "waves crashing on a rocky beach": 0.737 stock-02096 (beach, sea, coast, rock)
  - "a city skyline at night": 0.726 stock-00163 (skyline, night, city)
  - "snowy mountains": 0.726 stock-00171 (snow, winter, mountain)
  - "close-up of a flower": 0.718 stock-02063 (flower, close up, macro)
  - "a car on a road": 0.680 stock-03482 (sports car)
  - "a laptop on a desk": 0.699 stock-00134 (work, table, mouse), with a tagged laptop image at rank 3
  - Warm query embedding takes 36–50 ms. Ranking 500 vectors takes about 0.1 ms (cache hit) or about 8 ms (first load after restart).
- **After a server restart:** the same query returned identical results. Re-import gave `unchanged 500, embedded 0` (no recomputation), and `load_count 1`.
- `uv run pytest -q`: `30 passed` (known-vector ranking, exclusion, dim mismatch, cache invalidation on write and delete (stale IDs),
  library filter, empty/not-indexed/ready states, validation, 409 on profile mismatch, reindex under the new profile)

**Limitations:** these are inspection queries, not an evaluation (see Phase 17). Raw similarities cluster between 0.63 and 0.74, so scores are not
comparable across queries. Indexing speed on the M1 is about 1.8 s/image.

**Gate:** PASSED.

**Next:** Phase 7, reference-image search.

---

## Phase 7: Reference-image search (2026-10-07)

**Changes:** `backend/mediaindex/uploads.py` (temp upload streaming with a 25 MB limit, an extension allow-list, server-generated names,
always-delete, and a stale-file sweep at startup), `api_search.py` (`POST /api/search/image`, multipart with exactly one of `file` or `asset_id`),
`backend/tests/test_reference_search.py`.

**Behaviour**
- Uploaded references are decoded with the same `load_image` (EXIF orientation, RGB) and embedded with the same backend path as library images.
- Library references are chosen by **asset ID only**: the stored vector is reused, and the file is read through the ID-based, root-restricted resolver.
- By default, the reference asset and **every asset with an identical SHA-256** are excluded. `include_identical=true` keeps them.
- Errors: 415 unsupported extension, 413 oversized, 400 empty, 422 undecodable or ambiguous input, 404 unknown asset. Client filenames are never used as paths.

**Commands and observed results (REAL MODEL)**
- Asset reference stock-00102 (purple flower): top results are 0.885 stock-02095 (purple viola), then flower images at 0.76. The reference itself is excluded.
- Asset reference stock-00150 (city): city, street and architecture scenes at 0.74–0.76.
- **Failure case:** asset reference stock-03482 (sports car) returns dark night and road scenes (0.70–0.74), not other vehicles. Overall tone and lighting
  seem to dominate for this image.
- Uploading a byte-identical copy of stock-00102 gives a computed embedding. The default run excludes 1 identical asset. With `include_identical`, the copy ranks first
  at **1.0000**, which confirms the upload preprocessing matches indexing.
- Uploading an external sunflower photo returns flower and meadow images. Its identical copy in the smoke library was excluded.
- `data/tmp_uploads` holds 0 files after the requests. All 500 demo originals match their manifest SHA-256 (0 modified).
- `uv run pytest -q`: `34 passed`

**Limitations:** uploaded references cost a full image embedding (about 1.7–3 s on MPS while warm). Library references are instant.

**Gate:** PASSED.

**Next:** Phase 8, image plus text refinement.

---

## Phase 8: Image plus text refinement (2026-10-07)

**Changes:** `api_search.py` (`POST /api/search/image-text`: multipart `text` plus exactly one of `file`/`asset_id`), `scripts/refinement_check.py`,
`docs/refinement-findings.md`, and new tests in `backend/tests/test_reference_search.py`.

**Behaviour:** one native query embedding from the reference image plus the refinement text (`embed_image_text`, verified in Phase 1). Result provenance
in `query` records the reference, text, interface description, model id and revision, prompt, identical-asset exclusions, and a caveat that no logical
constraints are applied. Empty text returns 422 (pointing to the image-only endpoint). If the loaded encoders lack image support, the request is rejected explicitly.
Text-only and image-only modes are unchanged.

**Commands and observed results (REAL MODEL)**
- `uv run python scripts/refinement_check.py --library <demo>` ran 6 cases. See `docs/refinement-findings.md`.
  Helped: "at night" and "at sunset". Partly helped: "yellow flowers" and "a car". **Ignored:** "black and white photo". **Negation failed:** "without any flowers".
- `uv run pytest -q`: `35 passed`

**Limitations:** refinement quality is inconsistent, and the reference often dominates. No negation or logical constraints. Each mixed query needs a full
image+text forward pass (about 1–3 s on MPS).

**Gate:** PASSED. Real combined-input inference returns candidates, and the API and profile are recorded. Semantic limitations are documented.

**Next:** Phase 9, creator interface (Version 1).

---

## Phase 9: Creator interface, Version 1 (2026-10-07)

**Changes:** `frontend/src/{App.tsx,api.ts,index.css}` and `frontend/src/components/{Sidebar,SearchBar,ResultsGrid,DetailPanel}.tsx`.
The backend serves `frontend/dist` from the same loopback origin and sets security headers (CSP `default-src 'self'`, nosniff,
no-referrer). Jobs carry `library_id`. Upload errors no longer echo temp paths. Also `scripts/ui_check.py` (Playwright, dev-only) and a README quick start.

**UI:** library sidebar (add folder by absolute path with a plain-language explanation, import/re-scan, progress bar with cancel,
import summary listing failed files, remove index), search bar with a reference drop zone, optional refinement text and an include-identical toggle,
results grid (keyboard: arrows and Enter), and a detail preview (Esc, ←/→, **Use as reference**, open original, source/licence record).
Empty, loading, error, cancelled and partial states are covered. Technical scores are hidden by default. When shown, they read as raw cosine, never percentages.
System fonts only, no CDNs, light and dark themes.

**Start command:** `(cd frontend && npm run build) && uv run python -m mediaindex`, then open http://127.0.0.1:8765

**Browser verification:** headless installed Chrome via Playwright on the target laptop, REAL MODEL.
- `scripts/ui_check.py --folder data/samples/p4`: added the folder in the UI, import finished (`6 new, 1 failed: broken.jpg`), and the failure was listed
  → text "a zebra" (48 results, zebra.jpg first) → arrow-key focus then Enter opened the matching preview, Esc closed it → **Use as reference** → image search
  (identical zebra copies excluded; penguin first) → refinement "at night" (HTTP 200, image+text) → scores toggle showed `0.722` labelled as raw similarity.
  External requests: none. Console errors: none.
- **After a server restart** (`--skip-import`): libraries persisted. Text "a city skyline at night" put stock-00163 first, refinement "in the snow" returned snowy
  city scenes, an invalid upload (broken.jpg) showed a readable 422 message, and a valid upload reference returned 200. External requests: none.
- Fresh data dir (`ui_states.py`): empty-state text shown, `/` rejected with an explanation, cancelling the 500-image import mid-way showed "Import cancelled",
  and searching the partly indexed library worked.
- Screenshots were reviewed. Two layout bugs found this way were fixed: the stretched checkbox and long error paths overflowing.

**Limitations:** folders are added by pasting a path, since a browser cannot hand folder paths to a local server. Drag-and-drop of the reference was not exercised
by automation (the file-input path was). Tested only in Chrome on macOS.

**Gate:** PASSED. A user can complete import → text → image → refinement → preview → restart with real data. **Version 1.**

**Next:** Phase 10, selections and image exports.

---

## Phase 10: Selections and image exports (2026-10-07)

**Changes:** DB migration v2 (`selections`, `selection_items` with a snapshot of each asset so items survive index removal), `backend/mediaindex/selections.py`,
`backend/mediaindex/api_selections.py`, UI (`SelectionView.tsx`, "+ Add" on result cells and the `a` key, detail-panel Add/Reveal/Copy path, selections list
in the sidebar), `backend/tests/test_selections.py`, `scripts/ui_check_selection.py`.

**Behaviour**
- Persistent named selections: add (idempotent, with query context: mode, text or reference, rank), remove, reorder (must list every item), rename, delete.
  Item status is `ok`, `missing` (file gone), `removed` (no longer indexed), or `changed` (hash differs).
- Manifest (`GET /api/selections/{id}/manifest`, downloaded as an attachment): asset IDs, absolute source paths, library, sha256, size, dimensions,
  publisher source/licence record (with a "not a rights audit" note), query context. No similarity scores.
- Copy export uses two steps: `export/preview` returns a plan (destination, final names, clash renames, skipped items, `overwrites: 0`), then `export` with
  `{plan_id, confirm: true}` runs a cancellable job. Destination rules: absolute; not `/` or `$HOME`; not a file; parent must exist; **must not overlap
  any library root**. Each file is copied to `.mediaindex-partial-*`, verified against the indexed SHA-256, then **hard-linked into place (never
  overwrites; it re-picks a name on a race)**. Partial files are always removed. A manifest with `exported_as` names is written the same way.
- Reveal in folder runs `open -R <path>` on macOS (or the Linux/Windows equivalent) as an argument list with `shell=False`, using a path from the ID resolver.
  Copy path returns the resolved path.

**Commands and observed results**
- `uv run pytest -q`: `40 passed` (CRUD, idempotent add, reorder validation, persistence across app restart, missing/removed status, manifest contents,
  destination restrictions: relative, `/`, inside a library, parent of a library, existing file; duplicate filenames → `shared (2).jpg`, `shared (3).jpg`; a
  pre-existing user file is not overwritten; plan is single-use; a second export gives `photo one (2).jpg` and `mediaindex-manifest (2).json`;
  cancel during the 2nd file → 2 complete files and no partials; simulated "No space left on device" → 3 failed, empty destination; reveal uses an argument list without a shell)
- Browser (real app, REAL MODEL search): searched "a zebra", added 4 results with "+ Add" and 1 with the `a` key, opened the selection, moved up, removed,
  downloaded the manifest (`query_context: {mode: text, text: "a zebra", rank: 2}`), `/` destination rejected, then previewed and exported into
  `…/export dest` (a path with a space): `zebra.jpg`, **`zebra (2).jpg` (renamed clash between two libraries)**, 2 stock images and the manifest.
- **After a server restart** the selection order persisted. Moving one source file away showed a **"File missing"** badge.
- `shasum` over all sample and demo originals before and after: identical.

**Limitations:** reveal was verified with a mocked subprocess, not by opening Finder during automation. Copy path uses the browser clipboard API (loopback origin).
Video-segment items are skipped by the file exporter until Phase 15.

**Gate:** PASSED.

**Next:** Phase 11, audio inference and segmentation.

---

## Phase 11: Audio inference and segmentation (2026-10-07)

**Changes:** `backend/mediaindex/media/audio.py` (ffprobe/ffmpeg probing and decoding to mono 16 kHz, window planning, waveform thumbnails),
`GemmaBackend.embed_audio`, `indexer.make_audio_embedder` (windowed embedding, all windows written in one transaction, duplicate reuse with offsets),
ingest support for `.wav/.flac/.mp3`, profile changes (`audio` encoder loaded by default, `audio_window_s=10`, `audio_stride_s=5`,
`KEY_EXCLUDED_FIELDS`), startup re-key of compatible profiles, `scripts/{audio_smoke,audio_offsets_check,download_demo_audio}.py`,
`manifests/fsd50k-cc0.json`, `backend/tests/test_audio.py`, and docs (model-compatibility, THIRD_PARTY_DATA).

**Commands and observed results (REAL MODEL, MPS bf16)**
- `uv run python scripts/audio_smoke.py --audio data/demo/fsd50k-cc0 --images data/samples/smoke --n 12`: audio embeddings were finite, 768-d, and unit-norm within bf16 rounding.
  **Image and text vectors were bit-identical with and without the audio encoder** (max abs diff 0.0).
- `uv run --extra demo python scripts/download_demo_audio.py`: 60 CC0 clips (1–28 s), about 16.9 MB transferred, matching the committed manifest on rerun.
- Live app after restart: the old text+image vectors were re-keyed, and "snowy mountains" returned the same top results and scores as in Phase 6.
- Live import: `fsd50k-cc0` gave 60 files, 60 embedded, **86 segments**, 0 failed. `medley.wav` (39.25 s; dog, glass, crickets and snare concatenated with 3 s gaps, stereo
  44.1 kHz) gave 7 windows. Total 123 s.
- `uv run python scripts/audio_offsets_check.py --rel-path medley.wav ...`: each stored window vector matches an **independent FFmpeg-seeked re-decode of the same span:
  cos 0.9998–1.0000**. The control (same windows shifted by +2.5 s) gave mean cos 0.948. `OFFSETS OK`. Text over windows: "a snare drum" picked the last window (snare at 32.1–39.3 s),
  and "glass shattering" picked 15–25 s (glass at 14.8–17.8 s). "a dog barking" picked 10–20 s (the dog is at 0–11.8 s, so the overlap is only partial), and "crickets" picked 15–25 s
  (crickets at 20.8–29.1 s). Window-level localisation is coarse.
- `uv run pytest -q`: `46 passed` (window planning; WAV/FLAC/MP3 stereo decode to 16 kHz mono with seeks; corrupt audio rejected; mixed import with exact offsets
  `(0,10),(5,15),(10,20),(13,23)`; duplicate reuse; originals untouched; idempotent re-import; cancellation mid-file leaves it pending and unsearchable;
  compatible-profile re-key; API mixed library with image search unaffected)

**Limitations:** indexing takes about 2 s per short clip on MPS. Timestamps are only as precise as the 10 s / 5 s windows. Decoding needs FFmpeg on PATH.

**Gate:** PASSED. Real audio segments have finite embeddings and verified source offsets, and image search still works.

**Next:** Phase 12, audio search and playback.

---

## Phase 12: Audio search and playback (2026-10-07)

**Changes:** `api_search.py` (`media_types` accepts `audio`, segment grouping with `other_segments`, `POST /api/search/audio` with an upload or
`asset_id` plus optional `start_s` to pick a window), UI (Images/Sounds toggle, sound references in the drop zone, play/stop on result cells and the `p` key,
segment time labels, detail view with waveform, matched-window bar, "Play matched segment", other windows, full-file player),
`frontend/src/lib/{player,time}.ts`, `backend/tests/test_audio_search.py`, `scripts/ui_check_audio.py`.

**Behaviour:** only audio samples are indexed (no captions or filenames). Results are grouped per file by default (best window, plus up to 5 other matching windows).
Upload references accept WAV, FLAC and MP3 up to 50 MB, embed the first 10 s (or from `start_s`), and are deleted after use. A missing FFmpeg returns 503 with install guidance.
Unsupported types return 415, undecodable files 422. Playback uses HTTP Range requests on `/api/assets/{id}/file`. A single audio element seeks to `start` and pauses at `end`.
Text refinement is disabled for sound references in the UI. Native audio+text queries are not offered here (see Phase 13).

**Commands and observed results (REAL MODEL, MPS bf16; 60 FSD50K CC0 clips + medley, 93 windows)**
- Text → sound (top 1): "a dog barking" gave the bark clip (0.757), with medley 10–20 s second. "a snare drum hit" gave drum and snare clips in the top 3. "birds singing" gave a chicken/rooster clip, a partial hit.
  **Misses:** "glass shattering" (gunshot, tearing and keys; the glass clip is not in the top 3), "a door slamming" (gunshot first; the slam clip is not in the top 3),
  "a person speaking" (snare hits first). Warm query embedding takes about 40 ms (up to about 1 s cold).
- Sound → sound by asset: dog bark → **medley 0–10 s (0.997, the same bark inside the medley)**, then other animal sounds. Snare → **medley 29.25–39.25 s
  (0.984)**, then drum clips. Crickets → **medley 20–30 s (0.972)**, then the other cricket clip. Identical files are excluded.
- Upload of an MP3 transcode of fsd-146343 finds the FLAC original first (0.979) and the medley bark window second.
- Browser (`scripts/ui_check_audio.py`, headless Chrome): with Sounds selected, "a dog barking" returned 48 results. Playing medley 0:10–0:20 gave **player currentTime 10.58 s, playing**.
  The "other window 0:20–0:30" button gave **21.15 s**. Use as reference set the reference to "medley.wav 0:10–0:20" with the text box disabled, and the `/api/search/audio` span was [10, 20].
  An MP3 upload reference returned the original first. **Stop at end:** playing the 0:10–0:20 window and waiting 12 s ended **paused at 20.21 s**. External requests: none.
- **After a server restart:** the same text→sound results came back, and image search was unchanged.
- `uv run pytest -q`: `50 passed`

**Limitations:** retrieval quality is uneven, with clear misses above. Similar-sounding clips are candidates, not guaranteed-suitable sound effects. Stop-at-end accuracy is
about ±0.25 s (browser `timeupdate` granularity). Uploaded sound references use only one 10 s window.

**Gate:** PASSED. Imported sounds can be searched by text or sound, and the matching segment plays entirely locally. Image features are unchanged.

**Next:** Phase 13, cross-modal workspace.

---

## Phase 13: Cross-modal workspace (2026-10-07)

**Changes:** `api_search.py` (`target` on image, image+text and audio endpoints; native **audio+text** via `embed_audio_text`; multi-type text search
grouped by modality, or an experimental `global_rank`; `GET /api/capabilities` with per-mode status verified, experimental or unavailable; `mode_status`/`experimental` in
query provenance), UI (Images/Sounds/Both target, "experimental" badge driven by `/api/capabilities`, grouped "Images" and "Sounds" sections,
experimental banners, a mixed-ranking toggle behind technical scores, text refinement for sound references), `backend/tests/test_crossmodal.py`,
`scripts/{crossmodal_check,ui_check_crossmodal}.py`, `docs/cross-modal-findings.md`.

**Native audio+text verification (REAL MODEL):** `encode([{"text": prompt+text, "audio": {...}}])` returns a finite 768-d unit vector distinct from both inputs
(cos 0.908 to audio-only, 0.785 to text-only, 0.947 to their normalized mean). It is a single forward pass, not an average, so it is exposed as **experimental**.

**Commands and observed results (REAL MODEL)**
- `uv run python scripts/crossmodal_check.py`: 12 fixed examples. The expected item was in the top 10 in **8 of 12** cases and at rank 1 in **3 of 12**. Image→sound did reasonably well
  (train and insect at rank 1). **Sound→image mostly failed**, with the same three "hub" images topping most audio queries. Details: `docs/cross-modal-findings.md`.
- Browser (`scripts/ui_check_crossmodal.py`): Both + "a train" showed sections **Images (48)** and **Sounds (48)**. Image result → reference → Sounds showed the mode
  "Reference image → sounds" with an experimental badge, and returned the **train clip first**. Sound result → reference → Images ran `audio→image` (experimental) with a hub image first.
  The mixed-ranking toggle returned `grouping: global (experimental)` with a warning banner.
- Index compatibility is still enforced per media type (409 on profile mismatch), and every mode uses the single active profile.
- `uv run pytest -q`: `54 passed`

**Limitations:** sound→image quality is poor on this data (hubness). Similarity scales are not calibrated across types. Cross-media suggestions are candidates,
not synchronization or quality judgements.

**Gate:** PASSED. Real cross-modal retrieval can be previewed, index compatibility is enforced, and experimental modes are clearly identified.

**Next:** Phase 14, video ingestion and indexing.

---

## Phase 14: Video ingestion and indexing (2026-10-07)

**Changes:** `backend/mediaindex/media/video.py` (ffprobe probing with container and codec checks, accurate per-slot frame sampling, frame thumbnails),
`GemmaBackend.embed_video` / `embed_video_audio` / `embed_video_text`, `indexer.make_video_embedder` (windowed native-video, audio and optional joint
embeddings, per-window thumbnails, one transaction per file), MP4 support in ingest, profile video fields (`video_fps` and `video_max_soft_tokens` are in the key,
windowing is not), `GET /api/assets/{id}/window-thumbnail?start=`, **single-instance data-dir lock** (`instance.py`), the `plan_windows` tail fix,
`scripts/{make_demo_video,video_smoke,video_offsets_check}.py`, `backend/tests/test_video.py`, and docs.

**Commands and observed results (REAL MODEL, MPS bf16)**
- `uv run python scripts/make_demo_video.py`: `scenes.mp4` (40.02 s, 1280x720 H.264 25 fps + AAC) and `scenes-silent.mp4`, built from CC0 demo items.
- `uv run python scripts/video_smoke.py ...`: **soundtrack not included in video input (diff 0.0)**. Native video differs from the frame average (cos 0.922).
  Joint runs. Visual windows rank the dog, insect and city scenes on the right intervals.
- **Interrupted ingestion (real SIGKILL of the server process mid-video):** after restart the job was `interrupted`, the half-processed `scenes.mp4` was
  `pending` with **0 vectors** (not searchable), and the finished `scenes-silent.mp4` kept its 9 windows. Re-import resumed and finished in 147 s.
  This test exposed a real bug, now fixed: `pkill -9` had hit only the `uv` wrapper, and a second server instance ran startup recovery while the
  first was still working. A per-data-dir `flock` now refuses a second instance (`another MediaIndex process is already using …`).
- After the window-tail fix, a clean reindex of the video library: 2 files, **18 windows** (9 visual + 9 audio for scenes.mp4, 9 visual for the silent copy), 232 s
  (about 13 s per 8 s window on the M1 for visual+audio).
- `uv run python scripts/video_offsets_check.py --rel-path scenes.mp4`: every stored visual window matches an **independent torchcodec re-decode of the
  same interval: cos 0.980–0.998**. The control (+2 s shift) gave mean 0.952. `OFFSETS OK`. Scene text queries: dog, insect and city landed on their scenes (8 s overlap).
  **The train query missed** (best window 8–16 s; the train scene is 0–10 s).
- `uv run pytest -q`: `60 passed` (probe, frame offsets on solid-colour segments, straddling windows, bad container and MJPEG codec rejected,
  visual/audio/joint windows `(0,8),(4,12),(8,16),(12,20)`, silent video has visual windows only, window thumbnails, duplicate reuse, idempotent re-import,
  cancel mid-video leaves it unsearchable then resumes, broken video marked failed, single-instance lock)

**Limitations:** indexing is slow on the M1 (about 13 s per window with visual+audio, about 1.6x real time). Timestamps are only as precise as the 8 s / 4 s windows.
Only H.264 MP4 was exercised. An interrupted video restarts from its first window (the all-or-nothing write keeps partial windows out of search).

**Gate:** PASSED. Real indexed windows refer to correct source intervals, and interrupted ingestion recovers.

**Next:** Phase 15, video-moment search and clip export.

---

## Phase 15: Video-moment search and clip export (2026-10-07)

**Changes:** `api_search.py` (`media_types: ["video"]` with `video_signal` visual|audio|joint, per-video grouping with **overlap de-duplication**
(windows overlapping a kept moment by ≥50% are merged; `overlapping_windows_merged` count), window thumbnails, image or image+text → video targets,
`POST /api/search/video-window` for **moment → sounds (experimental)** and moment → similar moments), `backend/mediaindex/clips.py` and clip endpoints
(`/api/assets/{id}/clip/preview`, `/clip` with confirm), video moments in selection exports become accurate clips, UI (Videos/All targets, moment cells,
`VideoMoment` player that seeks to and stops at the window, other-moment thumbnails, sound suggestions, clip export panel with accurate or fast modes),
`docs/clip-export.md`, `backend/tests/test_video_search_clips.py`, `scripts/ui_check_video.py`.

**Commands and observed results (REAL MODEL)**
- Text → video moments on `scenes.mp4` and its silent copy: "a dog" gave 12–20 s, "an insect on a flower" 20–28 s, "a city skyline at night" 32–40 s (all on their scenes).
  "a train on railway tracks" gave 8–16 s, which only partly overlaps the 0–10 s train scene. The soundtrack signal for "a dog barking" gave 12–20 s.
  Image + text (dog photo + "running in a field") → video gave 16–24 s, a partial overlap.
- Moment → sound suggestions (experimental): train moment → **train clip**, insect moment → **cricket clips**, dog moment → traffic noise 1st and **bark 2nd**.
- Clip export on the real file: accurate [12, 20] gave **8.000 s**, H.264+AAC. Fast [13.3, 18] gave an effective start of **13.023 s** (keyframe) and 5.18 s.
  The source SHA-256 was unchanged.
- Browser (`scripts/ui_check_video.py`): Videos + "a dog" gave 2 grouped results at 0:12–0:20. The detail player **seeked to 12.00 s**, Play matched moment
  **stopped by itself at 20.16 s**, and the other-moment button seeked to 4.93 s. UI export wrote the clip and manifest, and a second export got a **numbered name instead of overwriting**.
  Find sounds showed the experimental banner. External requests: none.
- `uv run pytest -q`: `67 passed` (grouping and merged overlaps, window thumbnails, soundtrack signal, moment → sounds caveat, preview validation:
  reversed or too-short ranges, destination inside a library, relative path, bad mode, non-video; clamping to duration; **exported content is exactly the green segment**;
  exported duration; no overwrite of a user file; no partials; fast-mode keyframe start; cancel before or while rendering; codec failure; selection export of a moment as a clip)

**Limitations:** timestamps are only as precise as the 8 s / 4 s windows, never exact boundaries. Fast exports start early, at a keyframe. Sound suggestions are
similarity candidates, not synchronized audio. Retrieval was verified only on a generated 40 s demo video.

**Gate:** PASSED. Users can search, watch the retrieved window, and export a playable clip with a provenance manifest.

**Next:** Phase 16, offline operation and robustness.
