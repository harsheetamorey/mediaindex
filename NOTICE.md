# Notices

MediaIndex's original source code is licensed under the **Apache License 2.0** (see `LICENSE`).
Code, models and media each have their own terms, and they are kept separate below.

## Models (downloaded at setup, not included in this repository or the app)
- **google/embeddinggemma-2**, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, by Google DeepMind.
  The model card declares **Apache 2.0** and links the Gemma licence page. Use must also follow the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), which the model card references.
- Ask: **PekingU/rtdetr_v2_r50vd** (RT-DETR v2, object detection), Apache-2.0. Optional chat model: **Gemma 4 E2B**
  (`gemma4:e2b-it-qat`), Apache-2.0, which you install yourself through Ollama.
- Photo editing (optional): **facebook/sam-vit-base** (SAM, by Meta AI), Apache-2.0; **big-lama** (LaMa, by Samsung Research,
  TorchScript copy from `fashn-ai/LaMa`), Apache-2.0; **stable-diffusion-v1-5/stable-diffusion-inpainting** (CompVis, Stability AI,
  Runway), **CreativeML OpenRAIL-M**, whose use-based restrictions apply to anything you make with it; and
  **latent-consistency/lcm-lora-sdv1-5**, OpenRAIL++. Revisions are pinned in `edit.py` and `creative.py`.
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
| pillow-heif (HEIC/HEIF photos) | 1.8.1 | BSD-3-Clause; its wheel bundles libheif (LGPL-3.0), libde265 (LGPL-3.0) and x265 (GPL-2.0) |
| soundfile | 0.14.0 | BSD-3-Clause |
| psutil | 7.2.2 | BSD-3-Clause |
| diffusers / peft / accelerate (optional `create` extra) | 0.41.0 / 0.21.2 / 1.15.0 | Apache-2.0 |
| pywebview (optional `app` extra, included in the Mac and Windows apps) | 6.2.1 | BSD-3-Clause |
| pythonnet / clr-loader (Windows app only, used by pywebview) | 3.2.0 / 0.3.1 | MIT |
| pyinstaller (optional `app` extra, builds the Mac and Windows apps) | 6.22.3 | GPL-2.0 with a bootloader exception |
| pyarrow (optional `demo` extra) | 25.0.1 | Apache-2.0 |
| py7zr (optional `bench` extra) | 1.1.3 | LGPL-2.1-or-later |
| react / react-dom / scheduler | 19.3.0 / 19.3.0 / 0.28.0 | MIT |
| vite, @vitejs/plugin-react, oxlint (build tools) | 8.3.3, 6.1.2, 1.87.0 | MIT |
| typescript (build tool) | 6.0.3 | Apache-2.0 |

**FFmpeg / FFprobe:** from source, they are external programs that you install yourself (for example `brew install ffmpeg`).
The Mac app includes static FFmpeg 9.0.2 builds for arm64 by Martin Riedl (https://ffmpeg.martin-riedl.de), licensed under the
**GPL-3.0** (built with `--enable-gpl --enable-version3`). MediaIndex runs them as separate processes and does not link them.
Their licence notice and source links ship in the app (`Contents/Resources/ffmpeg_bin/LICENSE-ffmpeg.txt`); FFmpeg source:
https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz.
The Windows app includes the FFmpeg 9.0.2 "essentials" build by Gyan Doshi (https://www.gyan.dev/ffmpeg/builds/, pinned
and checksum-checked by `scripts/build_windows_app.ps1`), also **GPL-3.0** and also run as separate processes. Its licence
and source links ship in `_internal/ffmpeg_bin/`.

The built UI bundle (`frontend/dist`, not committed) contains React (MIT).

## Data
Place names for photo GPS positions come from **GeoNames** (cities15000), © GeoNames, licensed under **CC BY 4.0**; a
reduced copy is bundled (`backend/mediaindex/resources/places.tsv.gz`, details in `THIRD_PARTY_DATA.md`).

No media is included in this repository. Optional demo downloads, their publisher-declared licences, and their provenance are described in
`THIRD_PARTY_DATA.md`. **A publisher's declaration is not a rights audit.** The optional benchmark data (Clotho) is licensed for
non-commercial experimental use only. It is used only for local evaluation and is never bundled.
