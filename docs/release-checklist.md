# Release checklist

This checklist prepares a release. **Publishing (tags, GitHub releases, announcements) is a separate decision for the maintainer.**

## Code and tests
- [x] `uv run pytest -q`: all unit tests pass (mocked model, FFmpeg present)
- [x] `uv run pytest -m real_model -q`: the real-model smoke passes on the target laptop
- [x] `cd frontend && npm run build && npx oxlint src`: builds, no lint errors
- [x] Clean setup from a fresh clone (`uv sync --frozen`, `npm ci`, build, tests). See the Phase 19 notes in `docs/progress.md`
- [x] CI workflow `.github/workflows/ci.yml`: unit tests plus UI build on macOS arm64, with an opt-in real-model job (manual dispatch)
- [ ] CI run observed green on GitHub (requires the workflow to be pushed and run by the repository owner)

## Privacy and safety
- [x] Offline operation verified with outbound network blocked (`docs/offline-and-robustness.md`)
- [x] Loopback binding, Host/Origin checks, CORS limited to local UI origins, CSP, body-size limit, FFmpeg protocol whitelist
- [x] Originals are never modified (hash checks in tests and live runs)

## Repository hygiene
- [x] No media, databases, model weights, indexes or `data/` in git (`git ls-files` audit)
- [x] No secrets or credentials, and no private absolute paths except `/Users/you/…` UI placeholders
- [x] `.gitignore` covers media, `data/`, caches, `.env`

## Licences and provenance
- [x] Original code: Apache-2.0 (`LICENSE`). No individual author identity is asserted in the licence or docs
- [x] Model, third-party software and data notices kept separate (`NOTICE.md`, `THIRD_PARTY_DATA.md`)
- [x] Demo downloads keep per-item provenance, and the docs state that publisher declarations are not a rights audit
- [x] README screenshot uses only the publisher-declared CC0 demo pack (traceable rows in `manifests/stockimages-cc0-500.json`)
- [x] Benchmark data (Clotho, non-commercial) is local only and never bundled

## Documentation accuracy
- [x] README quick start, formats, query modes (verified vs experimental), hardware numbers, troubleshooting, limitations
- [x] Retrieval-quality metrics are marked **pending human labels** and no numbers are claimed
- [x] Untested platforms stated (Linux, Windows, Intel Macs, NVIDIA GPUs)
