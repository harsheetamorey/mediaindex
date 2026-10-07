# MediaIndex progress

The phases follow `mediaindex-claude-build-guide.md`. A gate is marked passed only after real execution on the target laptop.

| Phase | Status |
|---|---|
| 0 Repository and machine setup | PASSED |
| 1 Real model compatibility | PASSED |

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
