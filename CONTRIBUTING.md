# Contributing to MediaIndex

Thanks for helping! MediaIndex is a local-first app, so changes must keep it offline, loopback-only and non-destructive.

## Setup
```bash
uv sync                                  # Python 3.12 env from uv.lock
(cd frontend && npm ci)                  # UI deps from package-lock.json
brew install ffmpeg                      # or your platform's FFmpeg 6+ (needed for audio/video and many tests)
```

## Checks before a pull request
```bash
uv run pytest -q                         # unit tests (mocked model; no weights needed)
(cd frontend && npm run build && npx oxlint src)
uv run pytest -m real_model -q           # opt-in: needs cached weights + data/samples/smoke (3+ images)
```
Tests that need FFmpeg skip themselves when it is missing. Real-model checks are labelled `real_model` and are never mocked.

## Ground rules
- Never upload or phone home. No telemetry, analytics, CDNs or external fonts. Normal use must work with outbound network blocked
  (see `docs/offline-and-robustness.md`).
- Never modify, move or delete users' originals. Serve media only by asset ID.
- Keep one model instance per process, and serialize inference through `ModelHost`.
- Do not display similarity as confidence or percentages. Label experimental modes clearly.
- Do not commit media, databases, model weights or anything under `data/`.
- Any change that alters vectors (model revision, prompts, preprocessing, precision, token budgets) must change the `IndexProfile` key.
- Use evaluation labels from humans only (`evaluation/label_server.py`). Never generate relevance labels with a model.

## Reporting issues
Please include your OS, CPU or GPU, `uv run python -c "import torch; print(torch.__version__)"`, the FFmpeg version, and the output of
`curl http://127.0.0.1:8765/api/model/status`.
