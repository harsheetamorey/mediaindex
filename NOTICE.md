# Notices

MediaIndex's original source code is licensed under the **Apache License 2.0** (see `LICENSE`).
Code, models and media each have their own terms, and they are kept separate below.

## Model (downloaded at setup, not included in this repository)
- **google/embeddinggemma-2**, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, by Google DeepMind.
  The model card declares **Apache 2.0** and links the Gemma licence page. Use must also follow the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), which the model card references.
- The weights are fetched from Hugging Face into your local cache. MediaIndex does not redistribute them.
- Optional benchmark only (`evaluation/bench_clotho.py`): **laion/clap-htsat-fused**, revision `365dea6ef167def6676140ed93bbc43f84dabb28`,
  Apache-2.0. It is not used by the app.

## Third-party software (installed by `uv sync` and `npm install`, not vendored)
| Package | Version (locked) | Licence (as declared in package metadata) |
|---|---|---|
| fastapi | 0.142.2 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |
| starlette | 1.7.0 | BSD-3-Clause |
| pydantic | 2.13.5 | MIT |
| python-multipart | 0.0.32 | Apache-2.0 |
| torch | 2.14.1 | BSD-3-Clause style (plus bundled components: Apache-2.0, MIT, BSL-1.0 …) |
| torchvision | 0.29.1 | BSD |
| torchcodec | 0.17.0 | BSD-3-Clause |
| transformers | 5.19.0 | Apache-2.0 |
| sentence-transformers | 6.1.0 | Apache-2.0 |
| huggingface-hub | 1.33.0 | Apache-2.0 |
| tokenizers / safetensors | 0.23.2 / 0.8.0 | Apache-2.0 |
| numpy | 2.5.3 | BSD-3-Clause (plus bundled components) |
| pillow | 12.3.0 | MIT-CMU |
| soundfile | 0.14.0 | BSD-3-Clause |
| pyarrow (optional `demo` extra) | 25.0.1 | Apache-2.0 |
| py7zr (optional `bench` extra) | 1.1.3 | LGPL-2.1-or-later |
| react / react-dom / scheduler | 19.3.0 / 19.3.0 / 0.28.0 | MIT |
| vite, @vitejs/plugin-react, oxlint (build tools) | 8.3.3, 6.1.2, 1.87.0 | MIT |
| typescript (build tool) | 6.0.3 | Apache-2.0 |

**FFmpeg / FFprobe** are external programs that you install yourself (for example `brew install ffmpeg`). MediaIndex runs them as separate
processes and does not bundle or link them. Distribution builds of FFmpeg are often GPL-licensed; check your build's licence if you redistribute it.

The built UI bundle (`frontend/dist`, not committed) contains React (MIT).

## Data
No media is included in this repository. Optional demo downloads, their publisher-declared licences, and their provenance are described in
`THIRD_PARTY_DATA.md`. **A publisher's declaration is not a rights audit.** The optional benchmark data (Clotho) is licensed for
non-commercial experimental use only. It is used only for local evaluation and is never bundled.
