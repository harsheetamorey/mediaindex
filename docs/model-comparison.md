# Optional specialist-model comparison (Phase 18)

**Status: SKIPPED for now, partially run.** The CLAP side finished. The EmbeddingGemma side did not: the macOS low-memory
monitor killed the run on the 8 GB M1 laptop during the first Gemma configuration. The maintainer chose to skip the phase and come back later.
**No comparison between EmbeddingGemma 2 and CLAP is claimed.**

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

## Results so far (Apple M1, 8 GB, macOS 14)
| Config | R@1 [95% CI] | R@5 | R@10 | mAP@10 | Embed time | Peak MPS |
|---|---|---|---|---|---|---|
| clap-best | 0.142 [0.129, 0.157] | 0.373 | 0.498 | 0.241 | 247 s | 1,087 MB |
| clap-ctrl | 0.147 [0.132, 0.161] | 0.372 | 0.499 | 0.243 | 201 s | 1,087 MB |
| gemma-best | not run | – | – | – | – | – |
| gemma-app | not run | – | – | – | – | – |
| gemma-ctrl | killed (low memory) | – | – | – | – | – |

An earlier 20-clip trial is a far easier task than the 1,045-clip pool, so it is **not** reported as a result.

## Caveats
- **Training overlap:** CLAP's training data (LAION-Audio-630K) includes Clotho and other Freesound-derived audio, so its scores may be inflated.
  EmbeddingGemma 2's training data is not documented in enough detail to rule out overlap either.
- **Images:** the image comparison was skipped. Flickr30k needs a terms-gated download, and the CC0 demo pack has no human relevance labels yet.
- Clotho captions are licensed for non-commercial experimental use only (Tampere University). The audio carries per-file Freesound licences.
  The data stays local in `data/bench/clotho` and is never bundled. Clotho: K. Drossos, S. Lipping, T. Virtanen, *Clotho: an Audio Captioning Dataset*, ICASSP 2020.
- The benchmark is for evaluation only. **The app uses only EmbeddingGemma 2.**

## Coming back to it
The Gemma configurations need about 3.5–4 GB of free memory. Close other heavy apps (browsers, editors, database servers), then run
**one configuration at a time**:
```bash
uv run --extra bench python evaluation/bench_clotho.py --config gemma-ctrl
uv run --extra bench python evaluation/bench_clotho.py --config gemma-best
uv run --extra bench python evaluation/bench_clotho.py --config gemma-app
uv run --extra bench python evaluation/bench_clotho.py --summarize   # prints the table above, now with all results
```
The Clotho evaluation audio is already downloaded and checksum-verified in `data/bench/clotho`.
