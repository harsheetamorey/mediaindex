# MediaIndex: project instructions

MediaIndex is a local, open-source media-search workspace for creators. It is our own app, not an Oxford WISE fork.
Build spec: `mediaindex-claude-build-guide.md`. Progress and gate status: `docs/progress.md`.

## Stack
- Backend: Python 3.12 (uv-managed `.venv`), FastAPI, SQLite metadata, normalized NumPy vectors, exact cosine search.
- Frontend: React + TypeScript + Vite in `frontend/`. Bundle all assets locally; no runtime CDNs or external fonts.
- Model: `google/embeddinggemma-2` only for search. Never substitute another model, fake embeddings, or caption-based search.
- Ask tab only (approved by the maintainer, 2026-10-08): RT-DETR v2 counts objects, and an optional Gemma 4 E2B served by a local
  Ollama (loopback only) routes and phrases answers. Numbers always come from the database, never from the chat model.

## Commands
- Install: `uv sync` and `cd frontend && npm install`
- Backend: `uv run python -m mediaindex` (binds 127.0.0.1:8765, single worker)
- Frontend dev: `cd frontend && npm run dev` (127.0.0.1:5173, proxies /api)
- Frontend build: `cd frontend && npm run build`
- Tests: `uv run pytest` (unit tests); `uv run pytest -m real_model` (opt-in real-model checks)

## Rules
- Loopback only. Validate Host/Origin. Never serve arbitrary filesystem paths; serve media by immutable IDs only.
- Import only folders the user selects explicitly. Never modify, move or delete originals.
- Keep media, databases, vectors, thumbnails and model caches out of git (`data/` is the app state dir).
- One model instance in one worker; serialize heavy inference.
- Similarity is not confidence: never show cosine scores as percentages.
- Label real-model smoke checks separately from mocked unit tests.
- Commits: short messages (3 lines or fewer), no AI attribution or co-author lines.
