# Demo: storyboard, setup and recording

Everything shown uses the traceable demo packs only: 500 publisher-declared CC0 stock photos, 60 CC0 FSD50K sounds, and a 40 s
video generated locally from both packs. Every result named below is an **actual output** of the walkthrough run on 2026-10-07
(Apple M1, 8 GB, MPS bf16). These are single examples, not evidence of retrieval quality. Measured text→image quality is in `docs/evaluation-report.md` (20 labelled queries).

## 1. Launch and reset (exact commands)
```bash
# one-time: install (see README) and fetch the demo packs (~130 MB)
uv run --extra demo python scripts/download_demo.py
uv run --extra demo python scripts/download_demo_audio.py
uv run python scripts/make_demo_video.py

scripts/demo.sh reset     # deletes data/demo-app only (demo DB + thumbnails). Media files are never touched.
scripts/demo.sh start     # terminal 1: MediaIndex on the demo state dir, http://127.0.0.1:8765
scripts/demo.sh setup     # terminal 2: adds + indexes the 3 demo folders and waits
```
Indexing from scratch is **precomputed library work**. Based on measured rates (about 0.4 images/s, about 2 s per short sound and about 13 s per
8 s video window), expect **roughly 25 minutes** on an 8 GB M1. A full fresh demo index was not timed end to end in this phase. Re-running
`setup` on an indexed library skips unchanged files and finishes in about 3 s. Your own library DB (`data/mediaindex.sqlite3`) is not used or changed.

**Reproducible scenario (automatic):** with the server running, play every beat and save one screenshot per beat:
```bash
python scripts/demo_walkthrough.py --out data/demo-walkthrough [--video] [--pause 2.5]
```
This needs Playwright in a separate dev venv (`pip install playwright`, plus `playwright install ffmpeg` for `--video`), and it drives the installed Chrome.
It exits with a non-zero status if any request leaves the machine.

## 2. Storyboard (60–90 s with narration)
| # | Time | On screen (actual output) | Narration |
|---|---|---|---|
| 1 | 0–8 s | App open, three demo libraries in the sidebar, header "nothing leaves this computer" | "MediaIndex searches my own photos, sounds and videos, entirely on this laptop." |
| 2 | 8–20 s | Images · "a snowy mountain landscape" → top 5 are snowy mountains and forests (stock-00171, 02469, 03469, 03441, 00167) | "I describe what I want. The library was indexed ahead of time, so the query itself takes about 40 ms to embed on an M1." |
| 3 | 20–34 s | Upload stock-00171 as the reference, then add "at sunset" → sunset and dusk hills with trees and horizons (stock-03469, 03400, 03479, 02060, 02035) | "Or I start from a picture and steer it with words. The image and the text go into the model together, in one pass." |
| 4 | 34–44 s | Reference: purple viola, plus "without any flowers" → #1 is still a purple flower, and the rest are flowers and plants | "It's not magic. Negation doesn't work. 'Without flowers' just adds flowers. These are similarity results, not filters." |
| 5 | 44–56 s | Sounds · "a dog barking" → plays the matched window of fsd-146343 (publisher title: dog). #2 is a horse whinny and #3 a whip, and another dog is #4 | "Sounds work the same way. The top match is a dog, but the next two aren't, so you still listen and choose." |
| 6 | 56–70 s | Videos · "a dog" → the dog scene, window 0:12–0:20 of the demo video. The moment plays | "For video it finds the moment, an 8-second window, not the exact frame." |
| 7 | 70–84 s | Export clip → `scenes-silent_00m12_0s-00m20_0s.mp4` (8.00 s) plus a provenance JSON. Then add 3 photos to a selection → "Copied 3 files and mediaindex-manifest.json" | "I keep what I need: a clip, copies and a manifest with every source and licence. My originals are never touched." |
| 8 | 84–90 s | Back to the grid | "Local, open source, one model: EmbeddingGemma 2." |

Narration rules: separate indexing time (minutes, done ahead) from query time (milliseconds for text, about 2 s for an image reference).
Don't call any example "accurate" or a "breakthrough". Name the miss in beat 4 out loud.

## 3. What was recorded
- `scripts/demo_walkthrough.py --video --pause 2.5` produced a **silent 39 s screen capture** (`demo-rough-cut.webm`) and 8 screenshots,
  saved locally in `data/demo-walkthrough/` (git-ignored). It is a rough cut for timing. There is no narration and no edited
  60–90 s video, so the narrated recording is still to do.
- README demo stills: `docs/assets/screenshot-search.png`, `demo-refine.jpg`, `demo-honest-miss.jpg`, `demo-sounds.jpg`, `demo-video-moment.jpg`.

## 4. Recording checklist
- [ ] `scripts/demo.sh reset`, then `start` and `setup`. Wait until all three libraries show "searchable" counts equal to their file counts
- [ ] Close other apps (8 GB machine). Turn on Do Not Disturb, hide the bookmarks bar, and use a 1400×860 window at 100% zoom
- [ ] Run each query once before recording, so the model is warm (the first query includes about 13 s of model load)
- [ ] Keep "Show technical scores" off, so no numbers can be read as confidence
- [ ] Use only demo-pack media. No personal files or paths in view (export to a neutral folder such as `~/Desktop/demo-exports`)
- [ ] Record with QuickTime (File → New Screen Recording) and the narration from the storyboard. Keep beat 4 (the miss) in the cut
- [ ] Check the recording for private paths, notifications and other apps before sharing
- [ ] Credits in the description: stock photos (publisher-declared CC0, KoalaAI/StockImages-CC0), sounds (FSD50K CC0 subset), and model google/embeddinggemma-2
