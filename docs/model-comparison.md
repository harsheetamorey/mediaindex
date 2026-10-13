# Optional specialist-model comparison (Phase 18)

**Status: COMPLETE for audio (all 5 configurations on the full split). Images: runs done, waiting for human labels** (see Images below).
Run on 2026-10-07 on an Apple M1, 8 GB, macOS 14, one configuration at a time.

**Finding:** on Clotho text→audio retrieval, EmbeddingGemma 2 scores far below the specialist CLAP model:
R@10 is about 0.18 for EmbeddingGemma and about 0.50 for CLAP, and the median rank of the right clip is about 62–68 versus 11. This holds in every setup tried, including the controlled one
(same first 10 s and fp32 for both models). EmbeddingGemma is well above chance (R@10 for random ranking is about 0.01), but text→sound search in MediaIndex should
be treated as a rough candidate finder, not a precise one. CLAP's lead may be inflated by training overlap (see Caveats).

## What is measured
Text→audio retrieval on the official **Clotho v2.1 evaluation split**:
- **Pool:** all 1,045 clips.
- **Queries:** all 5,225 captions (5 per clip). Each caption has exactly one relevant clip.
- **Metrics:** computed per caption, averaged per clip, then over clips, with 95% bootstrap CIs over clips.
- **Script:** `evaluation/bench_clotho.py`. Each configuration runs in its own process and writes `evaluation/bench/clotho_<config>.json`.

| Config | Meaning |
|---|---|
| `gemma-best` | EmbeddingGemma 2 (pinned revision), MPS bf16, whole clip in one native pass at 16 kHz mono, captions with the SearchQuery prompt |
| `gemma-app` | As deployed in MediaIndex: 10 s windows, 5 s stride, clip score = max over windows |
| `clap-best` | `laion/clap-htsat-fused` @ `365dea6e…`, MPS fp32, 48 kHz, documented fusion truncation (seed 0) |
| `gemma-ctrl` / `clap-ctrl` | Controlled runtime: MPS fp32 for both, identical first-10 s excerpt of every clip |

## Results (Apple M1, 8 GB, macOS 14; full 1,045-clip / 5,225-caption pool)
| Config | R@1 [95% CI] | R@5 | R@10 [95% CI] | mAP@10 | Median rank | Embed time | Peak MPS |
|---|---|---|---|---|---|---|---|
| clap-best | 0.142 [0.129, 0.157] | 0.373 | 0.498 [0.478, 0.520] | 0.241 | 11 | 247 s | 1,087 MB |
| clap-ctrl | 0.147 [0.132, 0.161] | 0.372 | 0.499 [0.476, 0.521] | 0.243 | 11 | 201 s | 1,087 MB |
| gemma-best | 0.039 [0.031, 0.048] | 0.113 | 0.175 [0.157, 0.194] | 0.071 | 66 | 3,890 s | 2,611 MB |
| gemma-app | 0.037 [0.029, 0.045] | 0.116 | 0.180 [0.162, 0.199] | 0.072 | 62 | 2,337 s | 2,590 MB |
| gemma-ctrl | 0.031 [0.024, 0.039] | 0.107 | 0.168 [0.151, 0.185] | 0.065 | 68 | 656 s | 3,884 MB |

Observations:
- The app's windowing (10 s windows with 5 s stride, clip score = best window; 3,948 windows) is statistically indistinguishable from whole-clip embedding,
  because the confidence intervals overlap. The windowed index is therefore not costing retrieval quality on clips of this length.
- Using the whole clip instead of the first 10 s gives at most a small gain for either model.
- Embedding time for EmbeddingGemma is dominated by its audio encoder (3–16× slower than CLAP here). Query-time ranking cost is the same for both.
- The caption prompt was the app's documented `SearchQuery` prompt. Other prompts were not tried, and tuning them against this split would leak the test set.

## Caveats
- **Training overlap:** CLAP's training data (LAION-Audio-630K) includes Clotho and other Freesound-derived audio, so its scores may be inflated.
  EmbeddingGemma 2's training data is not documented in enough detail to rule out overlap either.
- **Images:** Flickr30k needs a terms-gated download, so images are compared on the app's own labelled evaluation instead (see Images below).
- Clotho captions are licensed for non-commercial experimental use only (Tampere University). The audio carries per-file Freesound licences.
  The data stays local in `data/bench/clotho` and is never bundled. Clotho: K. Drossos, S. Lipping, T. Virtanen, *Clotho: an Audio Captioning Dataset*, ICASSP 2020.
- The benchmark is for evaluation only. **The app uses only EmbeddingGemma 2.**

## Reproduce
Run one configuration at a time, with other heavy apps closed (EmbeddingGemma needs about 3.5–4 GB):
```bash
uv run --extra bench python evaluation/bench_clotho.py --config gemma-ctrl   # also: gemma-best, gemma-app, clap-best, clap-ctrl
uv run --extra bench python evaluation/bench_clotho.py --summarize
```
The first attempt at the EmbeddingGemma runs was stopped because the machine ran low on memory. After other apps were closed, all three completed (11 min, 65 min and 39 min).

## Images: open baselines on the app's evaluation (runs done 2026-10-13; labels pending)
Same 500 CC0 photos, same frozen queries (`evaluation/queries.json`, spec v1), same top-10 depth and label rules as
`docs/evaluation-report.md`. Script: `evaluation/bench_images.py`. Evaluation only. **The app uses only EmbeddingGemma 2.**

| Config | Model | Precision | Notes |
|---|---|---|---|
| `clip-l14` | `openai/clip-vit-large-patch14` @ `32bd6428…` | MPS fp32 | 224 px |
| `siglip2-so400m` | `google/siglip2-so400m-patch14-384` @ `e8e48729…` | MPS bf16 | 384 px, text padded to 64 tokens (model card) |

- Query text is used as written, with no prompt template, so nothing is tuned on the test queries.
- Image references exclude the reference and identical-hash copies, as the app does. Image+text queries are not run:
  neither baseline has a native joint image+text query.
- **Fair pooling:** the human labels so far cover only EmbeddingGemma 2's top 10. Each baseline's top 10 was merged into
  `evaluation/pool-v1.json` (+289 new candidates). Still unlabelled: 155 text, 284 image, 120 image+text. `label_server.py` shows them shuffled
  with the source hidden. An unlabelled result counts as unknown, never as irrelevant.

Early look on the existing labels (text queries only; **not a result**, because 88–92 of each baseline's 200 top-10 results are unlabelled):

| System | Hit@1 | Hit@5 |
|---|---|---|
| EmbeddingGemma 2 (app run) | 0.900 (n=20) | 0.950 (n=20) |
| CLIP ViT-L/14 | 0.947 (n=19) | 1.000 (n=19) |
| SigLIP 2 so400m | 0.850 (n=20) | 1.000 (n=19) |

With 20 queries, one query moves Hit@1 by 0.05, so these three are not distinguishable yet. nDCG@10 needs every candidate labelled.

To finish: label the new candidates, then
```bash
uv run python evaluation/label_server.py --images data/demo/stockimages-cc0 --labeler "<name>"
uv run python evaluation/bench_images.py --summarize
```
Embedding time on the M1 included the first model download (497 s for CLIP, 1,768 s for SigLIP 2), so it is not a speed comparison.
