# Offline operation and robustness (Phase 16)

All measurements were taken on the target laptop (Apple M1, 8 GB, macOS 14.0) on 2026-10-07.

## First-time setup vs. normal use
- **First-time setup needs the network:** `uv sync`, `npm install`, the model weights (about 1.5 GB; run `scripts/model_smoke.py` with
  `MEDIAINDEX_ALLOW_DOWNLOAD=1` or a plain `huggingface_hub` download), and optionally the demo packs.
- **Normal use is offline.** `python -m mediaindex` sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1` and
  `DO_NOT_TRACK=1` (unless `MEDIAINDEX_ALLOW_DOWNLOAD=1`), and the model loads with `local_files_only=True` at the pinned revision.

## What was verified with outbound network blocked
The server ran under `sandbox-exec` with a profile that denies all outbound network except loopback (`no-network.sb`, reproduced below).
A control `curl https://huggingface.co` inside the profile failed (exit 7), and a raw Python connect to 1.1.1.1 got `Operation not permitted`.

- `scripts/offline_flow_check.py` against a fresh data directory: **13/13 steps passed**. They cover the model loaded from the local cache, import of images, audio and an MP4
  (48 s), text → image, sound and video moment, uploaded image reference, image+text, uploaded sound reference, image → sound, moment → sound,
  clip export, selection copy export with a manifest, and the UI served locally.
- `scripts/ui_check.py` in headless Chrome against the sandboxed server: text, preview, keyboard, reference, refinement and upload all worked, with
  **external requests: none** and **console errors: none**.
- `scripts/run_with_socket_audit.py`, a Python audit hook on `socket.connect`, `socket.getaddrinfo` and `socket.sendto`, ran during the full flow and recorded
  **0 events** (a control script in the same interpreter recorded its connect and DNS events). The only socket of the process was
  `127.0.0.1:8765 (LISTEN)` (lsof).
- Code and build audit: no URLs in backend runtime code. The built UI contains only XML namespace strings and React's error-doc string (never fetched),
  no external fonts, and no CDN. CSP `default-src 'self'`.
- Not verified: the macOS unified log did not show sandbox denials, even for a deliberate control, so it could not serve as a second witness.

```
(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(allow network-outbound (remote unix-socket))
(deny network-inbound)
(allow network-inbound (local ip "localhost:*"))
```

## Security review
| Area | Status |
|---|---|
| Binding | `127.0.0.1` only (uvicorn), single worker. lsof confirms. |
| Host / Origin | Non-loopback `Host` gives 403 (DNS rebinding). Mutations with a foreign or `null` Origin give 403. CORS allows only the local UI origins. |
| Single instance | New: a `flock` on `<data>/.mediaindex.lock` refuses a second server on the same data dir. Found when a stray process ran recovery concurrently. |
| Paths | Libraries are explicit absolute folders, not `/` or `$HOME`. Media is served by asset ID only, re-resolved inside the root, with symlink escapes rejected. Exports cannot target library folders. |
| Uploads | New: request bodies over 60 MB are rejected **before parsing** (ASGI middleware, also for chunked bodies). Per-type limits are 25 MB images and 50 MB audio. Server-generated names. Deleted after use and swept at startup. Multipart spooling is redirected to `<data>/tmp_spool` (emptied at start) instead of the system temp dir. |
| Decoding limits | Images ≤ 80 MP and ≤ 200 MB. Audio ≤ 4 h. Video ≤ 6 h, MP4 only, allow-listed codecs. FFprobe and FFmpeg have timeouts. |
| FFmpeg | Argument lists only (no shell), `--` before probe paths, absolute paths. New: `-protocol_whitelist file,pipe` on every invocation, so a crafted playlist inside a `.mp3` cannot reach a URL (test added). |
| Cleanup | Partial export and clip files are removed on failure, cancel or disk-full. Stale uploads are removed at startup. |
| Reveal | `open -R <path>` as an argument list, path from the ID resolver. |

## Stability under load
- **Concurrency:** 20 simultaneous searches (image, sound and all-media mixes) all returned 200 in 2.4 s with the model idle, and 4.8 s while a video import was running.
  `load_count` stayed **1** throughout (one model instance).
- **Model busy:** with `MEDIAINDEX_QUERY_WAIT=0.3` during a video import, a burst of 41 queries got **503 "model is busy with other work; try again shortly"**
  while a window held the model, and 200 in the gaps between windows (frame decoding runs outside the model lock). The default wait is 30 s.
- **Disk full:** a selection export of 4.5 MB onto a 4 MB RAM disk copied 2 of 3 files. The video failed with `[Errno 28] No space left on device`, the manifest
  listed the failure, and **no partial files** remained.
- **Restart recovery:** a real SIGKILL during video ingestion is covered in Phase 14 (job `interrupted`, file not searchable, re-import resumes).
- **Memory (server process, MPS bf16, full model):** physical footprint **3.4 GB (peak 3.5 GB)** while indexing video and serving searches. MPS allocator: 1.4 GB
  current, 2.6 GB driver. Process peak RSS (CPU side) is about 0.9 GB. This fits the 8 GB machine but leaves limited headroom. CPU fp32 mode uses more RAM.

## Search scaling (measure before optimizing)
`scripts/scale_check.py` with synthetic 768-d vectors, exact dot-product search:

| Vectors | SQLite size | Cold matrix load | Matrix RAM | Warm ranking median / p95 |
|---|---|---|---|---|
| 20,000 | 86 MB | 0.25 s | 59 MB | 2.1 / 2.5 ms |
| 100,000 | 429 MB | 2.9 s | 293 MB | 11.8 / 16.4 ms |
| 200,000 | 859 MB | 6.3 s | 586 MB | 22.1 / 24.5 ms |

The real libraries are hundreds to a few thousand vectors (embedding takes about 2 s per image on the M1, so indexing 200k items would take days). There is no measured need for
ANN or quantization, so **none was introduced** and retrieval stays exact. At 200k, RAM (matrix plus model) would be the first limit on an 8 GB machine.
