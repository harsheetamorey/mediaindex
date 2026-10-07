# Clip export

Clips are exported from a video's detail view ("Export clip…"), or automatically when a selection containing video moments is exported.
Every export runs preview first, then an explicit confirm, then renders to a hidden `.mediaindex-partial-*` file, validates it, and moves it into place with an
operation that **cannot overwrite** (a numbered name is chosen on a clash). A `<clip>.mediaindex.json` provenance manifest is written next to it.
The source video is opened read-only. Destinations inside any indexed library are refused.

| Mode | FFmpeg | Start accuracy | Quality and speed |
|---|---|---|---|
| **Accurate** (default) | `-ss S -i src -t D -c:v libx264 -crf 20 -c:a aac` | Frame-accurate: the clip starts at S | Re-encoded (small generational loss), slower |
| **Fast** | `-ss S -i src -t D -c copy -avoid_negative_ts make_zero` | Starts on the **keyframe at or before S**. The preview reports it as `effective_start_s` | No quality loss, very fast |

Observed on `scenes.mp4` (1 s GOP):

- Accurate [12.0, 20.0] gave **8.000 s** H.264/AAC, and every sampled frame belongs to the requested scene (also checked by the solid-colour test).
- Fast [13.3, 18.0] gave an effective start of **13.023 s** (keyframe) and a 5.18 s output.

Validation after rendering: FFprobe must find a video stream. The duration must be within 0.15 s of the request for accurate mode, or not shorter than
the request for fast mode. On a cancel, a failure or a codec error, the partial file is removed, so nothing half-written is left in the destination.

**Timestamps are limited by indexing.** Search returns 8 s windows (4 s stride, 1 frame/s), so a "moment" shows where a match roughly is, not an
exact event boundary. Adjust start and end in the export panel (use ▶ Preview range) before exporting.
