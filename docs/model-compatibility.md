# Model compatibility: google/embeddinggemma-2

Verified 2026-10-07 on the target laptop (Apple M1, 8 GB, macOS 14.0). The outputs below come from real weights, not mocks.

## Source and license
- Hugging Face: `google/embeddinggemma-2`, public and **not gated**, license **Apache-2.0** (model card links the Gemma 4 license page).
- Pinned revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`
- Weights: `model.safetensors` 1,488,915,288 bytes (1.4 GB on disk) in `~/.cache/huggingface/hub`.
- No `trust_remote_code`: the architecture (`EmbeddingGemma2Model`, `model_type: embedding_gemma2`) ships natively in `transformers==5.19.0`.

## Pinned working stack
| Package | Version | Note |
|---|---|---|
| torch | 2.14.1 | MPS backend |
| torchvision | 0.29.1 | **required** by the Gemma4 image processor (the import fails without it) |
| torchcodec | 0.17.0 | video and audio decoding (docs install line) |
| transformers | 5.19.0 | model card config was written with 5.18.0.dev0 |
| sentence-transformers | 6.1.0 | model requires >=6.1.0 for multimodal dict inputs |
| pillow | 12.3.0 | |
| soundfile | 0.14.0 | |
| numpy | 2.5.3 | |

## Interface (verified)
- Loading: `SentenceTransformer("google/embeddinggemma-2", revision=REV, device=..., model_kwargs={"dtype": dtype}, config_kwargs=..., local_files_only=True)`
- Output: 768-d, normalized by the model's `2_Normalize` module. Supported MRL truncations are 512/256/128, which must be re-normalized. MediaIndex uses the full 768 d.
- **Text query:** `model.encode(texts, prompt_name="SearchQuery")`, which prepends `task: search result | query: `.
- **Text document:** `prompt_name="Document"`, which prepends `title: none | text: ` (not used for media).
- **Image:** `model.encode([{"image": PIL.Image}])`. Media take **no prompt prefix** (model card). Defaults: 280 vision soft tokens and the Gemma4ImageProcessor (resize, rescale 1/255, no mean/std normalize, patch 16).
- **Native image + text (one embedding):** `model.encode([{"text": "task: search result | query: <text>", "image": PIL.Image}])`. This is a single forward pass over the interleaved sequence, not an average of two vectors.
- Audio expects mono 16 kHz. Video defaults to 1 fps, at most 32 frames and 140 tokens per frame (verified in later phases).
- Selective encoder loading: `config_kwargs={"audio_config": None}` (text and image only) works. It was verified with `--no-audio-encoder`.

## Precision and device
- The model card forbids **float16**: it produces NaN or degraded output. The smoke script asserts float16 is never used.
- **MPS + bfloat16: works.** All outputs are finite. Norms are 0.998–1.003 because they are bf16-rounded, so MediaIndex re-normalizes in float32 before storage.
- **CPU + float32: works.** It serves as the reference, and its norms are exactly 1.0.
- Agreement: MPS-bf16 and CPU-fp32 similarity scores differ by ≤0.003, and the top-3 rankings match on every query.
- No CPU fallback was needed on MPS.

## Smoke results (scripts/model_smoke.py, 6 local JPEGs converted from macOS sample pictures)
Command: `uv run python scripts/model_smoke.py --images data/samples/smoke --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --offline`

| Query | Top-3 (raw cosine, MPS bf16) |
|---|---|
| "a zebra with black and white stripes" | zebra 0.753, piano 0.608, penguin 0.598 |
| "a musical instrument" | piano 0.709, guitar 0.701, parrot 0.628 |
| "a bright yellow flower" | sunflower 0.721, parrot 0.632, penguin 0.616 |
| mixed: zebra.jpg + "a musical instrument" | zebra 0.793, piano 0.732, guitar 0.696 |

The mixed query raises the instrument images (piano 0.61 → 0.73) while the reference image stays on top. **This is a smoke check, not an accuracy claim.** Raw cosine values cluster around 0.6–0.8 and are not probabilities.

## Timing and memory (cold process, M1)
| Mode | Load | 3 text queries | 6 images (batch 1) | 1 mixed |
|---|---|---|---|---|
| MPS bf16 (offline) | 6.0 s | 1.2 s | 14.8 s | 3.3 s |
| CPU fp32 (offline) | 5.0 s | 0.3 s | 27.4 s | 3.9 s |

Peak RSS for MPS bf16 without the audio encoder was about 1.0 GB, as measured by `/usr/bin/time -l`.

## Offline
After the first download, loading with `HF_HUB_OFFLINE=1` and `local_files_only=True` succeeds (`--offline` runs above).
This verifies local-only loading. A test with outbound network blocked is deferred to Phase 16.
