# MediaIndex

A local, open-source media-search workspace for creators. Point it at a folder of images, then search by describing what you want,
by example image, or by example image plus refinement text. Everything runs on your computer: no cloud, no uploads, no telemetry.

Embeddings come from [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), run locally.

> Status: **Version 1, image search**. See `docs/progress.md` for verified phases.

## Quick start (macOS / Linux)

Requirements: Python 3.12 with [uv](https://docs.astral.sh/uv/), and Node 20+ (only to build the UI).

```bash
uv sync                                   # Python deps (pinned in uv.lock)
(cd frontend && npm install && npm run build)
MEDIAINDEX_ALLOW_DOWNLOAD=1 uv run python scripts/model_smoke.py --images <folder with 3+ images>  # first run downloads ~1.5 GB of weights
uv run python -m mediaindex               # then open http://127.0.0.1:8765
```

After the first model download, MediaIndex loads the model from the local cache only (`local_files_only`) and needs no network.

In the app: **+ Add folder**, paste the folder's absolute path (in Finder, select the folder and press ⌥⌘C), and wait for indexing. Then search.

Optional demo images (500 publisher-declared CC0 stock photos, about 115 MB): `uv run --extra demo python scripts/download_demo.py`,
then add `data/demo/stockimages-cc0` as a folder. See `THIRD_PARTY_DATA.md`.

## Search modes
| Mode | How |
|---|---|
| Text → image | Type a description |
| Image → image | Drop or choose a reference image, or click **Use as reference** on a result |
| Image + text → image | Reference image plus refinement text (one native multimodal embedding) |

Results are the nearest neighbours by cosine similarity. The closest items are always shown, even when nothing truly matches.
Scores are hidden by default. When shown, they are raw similarities, not probabilities.

## Privacy and safety
- The server binds to `127.0.0.1` only, validates Host and Origin, and sets a strict Content-Security-Policy.
- Only the folders you add are read. Originals are never modified, moved or deleted. Files are served by asset ID, never by path.
- App state (database, thumbnails, temp uploads) lives in `data/`, which is git-ignored.
