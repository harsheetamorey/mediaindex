# MediaIndex progress

The phases follow `mediaindex-claude-build-guide.md`. A gate is marked passed only after real execution on the target laptop.

| Phase | Status |
|---|---|
| 0 Repository and machine setup | PASSED |

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
