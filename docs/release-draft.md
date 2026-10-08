# Draft release notes and handoff (not published)

These are drafts for the maintainer to review. Nothing here has been posted, tagged or released.

## Repository description (≤ 350 chars)
> Local, offline media search for creators. Search your own images, sounds and MP4 video moments by text, by example, or by example + text,
> then export copies, clips and provenance manifests. One local model (EmbeddingGemma 2), loopback-only, no uploads, no telemetry. Apache-2.0.

Topics: `local-first` `semantic-search` `multimodal` `embeddings` `creators` `fastapi` `react` `offline`

## Draft release description: v0.1.0 (preview)
**MediaIndex v0.1.0: local media search for creators (preview)**

MediaIndex indexes folders you choose and lets you search them on your own machine, with no cloud account, uploads or telemetry.

What works (verified on an Apple M1, 8 GB, macOS 14):
- Text → images, sounds and video moments. Image → images. Image + text → images and video moments. Sound → sounds. Video moment → similar moments.
- Experimental, and labelled as such in the app: image → sounds, sound → images, sound + text, moment → sounds.
- Selections with a JSON manifest (sources, hashes, licence records, how each item was found), copy export with preview, and frame-accurate
  or fast clip export with a provenance file. Originals are never modified.
- Offline after setup: verified with outbound networking blocked, and a socket audit recorded zero outgoing connections.

Speed (8 GB M1): indexing is precomputed and slow (about 1.8 s per image, about 13 s per 8 s video window). Queries are fast: text embedding is about 40 ms, and exact ranking
of 600 vectors is 0.1 ms. An image reference query takes about 1.8 s.

Measured quality (human labels, one labeller, 500-image demo pack): text → images Hit@1 0.90, Hit@5 0.95, nDCG@10 0.92 over 20 queries.
Image → image and most image+text queries are not labelled yet. Known limitations: On the Clotho text→audio benchmark,
EmbeddingGemma 2 scores well below the specialist CLAP model (R@10 0.18 vs 0.50). Refinement text is a soft steer, not
a filter, and negation is not understood. Video moments are 8 s windows. Tested only on macOS 14 with Apple silicon.
The model weights (about 1.5 GB) are downloaded once at setup and are not redistributed.

Credits: google/embeddinggemma-2 (Apache 2.0 per model card, Gemma Prohibited Use Policy applies). Demo packs: KoalaAI/StockImages-CC0 (publisher-declared CC0)
and a CC0 subset of FSD50K. See NOTICE.md and THIRD_PARTY_DATA.md.

## Final handoff

### Run
```bash
uv sync && (cd frontend && npm ci && npm run build)
MEDIAINDEX_ALLOW_DOWNLOAD=1 uv run python scripts/model_smoke.py --images data/samples/smoke   # once, needs network
uv run python -m mediaindex                                                                       # http://127.0.0.1:8765
uv run pytest -q                    # 74 unit tests (mocked model)
uv run pytest -m real_model -q      # real-model smoke (cached weights)
```
Demo: see `docs/demo.md` (`scripts/demo.sh reset|start|setup`, `scripts/demo_walkthrough.py`).

### Feature status
| Feature | Status |
|---|---|
| Image import, thumbnails, text → image, image → image, image + text | Done, verified with the real model |
| Creator UI (preview, keyboard, selections, manifest, copy export) | Done, verified in a real browser |
| Audio import (10 s / 5 s windows), text → sound, sound → sound, segment playback | Done, verified |
| Cross-media (image ↔ sound, sound + text, moment → sounds) | Experimental, labelled in the UI |
| Video (MP4) windows, text → moment, moment → moment, clip export | Done, verified (H.264/AAC tested) |
| Offline operation and robustness | Verified under a network-blocking sandbox |
| Retrieval-quality metrics (Phase 17) | Text → images measured (20 queries, Hit@1 0.90). Image+text 3/15 labelled, image → image 0/15 |
| EmbeddingGemma vs CLAP benchmark (Phase 18) | Done (audio). EmbeddingGemma R@10 0.18 vs CLAP 0.50 on Clotho. Image comparison skipped |
| Narrated demo video | **Not recorded.** Storyboard, scripts, stills and a silent 39 s rough cut are ready |
| Native folder picker, Linux/Windows support, non-MP4 video | Planned / untested |

### Gates
Phases 0–19: PASSED (evidence in `docs/progress.md`). Phase 17 passed as specified, with metrics explicitly pending.
Phase 18: audio comparison done, image comparison skipped. Phase 20: see `docs/progress.md`.

### Supported hardware
Tested: Apple M1, 8 GB, macOS 14.0, MPS bf16 (about 3.5 GB peak footprint). CPU fp32 works but is slower. Untested: Linux, Windows, Intel Macs, NVIDIA GPUs.

### Known issues
- Text→sound retrieval is weak on Clotho (R@10 0.18, CLAP 0.50; see docs/model-comparison.md).
- Uneven retrieval: glass and door sounds are missed, "hub" images appear for many sound queries, style refinements ("black and white") can be ignored, and negation fails.
- Indexing is slow on 8 GB machines (video at about 1.6× real time). Heavy indexing can make queries wait (503 after `MEDIAINDEX_QUERY_WAIT`).
- Folders are added by pasting a path. Two oxlint warnings (`set-state-in-effect`) remain.
