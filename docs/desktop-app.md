# Desktop app and creator features

## MediaIndex.app (double-click, macOS on Apple silicon)
```bash
scripts/build_mac_app.sh        # → dist-app/MediaIndex.app and MediaIndex-<version>-macos-arm64.zip (model weights not included)
open dist-app/MediaIndex.app    # or double-click it in Finder; drag it to /Applications to keep it
```
- **What it is:** the same local server, started inside the app and shown in a native window (pywebview, WebKit).
  It needs no Terminal, uv or npm after the build. It still listens only on `127.0.0.1`, uses a free port if 8765 is taken,
  and allows one app per data folder.
- **Your data:** stored in `~/Library/Application Support/MediaIndex` (database, thumbnails). Your media stays where it is.
- **Opening:** the window appears at once with "Starting MediaIndex…", and the app loads behind it (about 2–3 s on an M1).
- **First launch:** if the models aren't in the local Hugging Face cache, the window explains and asks before downloading them
  once, at pinned revisions: EmbeddingGemma for search and the object detector for Ask (about 1.6 GB), and, unless unticked,
  the photo editing models SAM, LaMa and Stable Diffusion 1.5 inpainting with LCM-LoRA (about 2.7 GB more). Editing can be
  added later with **Models → Download photo editing models…** in the menu bar. After that the app runs offline. The Ask
  tab's optional chat model needs Ollama (see [`ask.md`](ask.md)).
- **FFmpeg:** included in the app (static FFmpeg 9.0.2 for arm64 by [Martin Riedl](https://ffmpeg.martin-riedl.de), GPL-3.0,
  pinned by checksum in the build script). It runs as a separate program; its licence is in `Contents/Resources/ffmpeg_bin`.
- **Not signed or notarized** (ad-hoc signature only). On another Mac, open it the first time with right-click → Open, or on
  macOS 15 with System Settings → Privacy & Security → Open Anyway. Signing with a Developer ID and notarizing removes that step.
- **Self-test:** `MEDIAINDEX_DESKTOP_SELFTEST="a dog" dist-app/MediaIndex.app/Contents/MacOS/MediaIndex` waits for the window to load
  the UI, then:
  - runs one real search;
  - counts objects and asks "how many dogs?" in Ask;
  - prints the results and quits.

  On the first-run page it also presses "Download the model".

## Choosing folders
**Choose folder…** opens the standard macOS folder chooser on this Mac (through `osascript`) and fills in the path.
Pasting a path still works. The chooser can only be triggered by the app's own page, because cross-site requests are rejected.

## Watched folders
Tick **Watch for changes** on a library. Every 20 s (`MEDIAINDEX_WATCH_INTERVAL`, `0` turns it off), MediaIndex checks the folder's
media files using file size and modification time only. When a change has been stable for two checks, so a copy in progress has
finished, it queues the normal read-only re-scan. New and changed files are indexed, unchanged files are skipped, and removed files
are marked missing.

## Indexing: progress estimate and low priority
- While indexing, the library shows **"about N min left"**. The estimate uses the rate of real work so far; unchanged files don't
  count. It appears after a few files have been indexed.
- **Index gently in the background (low priority)** runs the indexing thread at macOS background priority (lower CPU and disk
  priority), so other apps stay responsive. It is **slower**: on an M1 (8 GB), 12 new images took 37 s instead of 22 s (both runs
  repeated, about 1.7×).
- **Larger batches don't make indexing faster.** On the M1, 0.56 images/s at batch size 1, 0.57 at 2 and 4, and 0.52 at 8
  (which also used 1.2 GB more memory). The GPU is the limit, so the batch size was left unchanged.

## Getting files into editing apps
- **Finder tags:** exported copies get the tags **MediaIndex** and the selection's name, and exported clips get **MediaIndex** and
  **Clip**, so you can find them in Finder's sidebar or with Spotlight. Original files are never tagged. From there, drag them into
  Premiere Pro, Final Cut Pro or Canva. Those apps were **not tested** (none were installed on the test Mac).
- **Reveal in folder** on any result opens Finder at the original, ready to drag into an editing app.
- **Dragging results straight out of the browser was tried and removed.** It relies on the browser being allowed to download.
  On the test Mac, managed Chrome blocked downloads, which left empty files, and it can't work inside the app window at all.
