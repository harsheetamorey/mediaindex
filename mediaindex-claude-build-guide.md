# MediaIndex — phased Claude Code build guide

## What we are building

A local, open-source media-search workspace for creators. Import a media folder, search by text or a reference file, refine with text, preview results, and export selections. EmbeddingGemma 2 supplies embeddings. This is our own app, not an Oxford WISE fork. The working name is MediaIndex; name availability is not verified.

Version 1 is a working image-search app. Audio and video follow. Comparing models is a separate optional evaluation, not a prerequisite for the product.

Use Claude Code on the laptop that will run the model, inside your repository. An ordinary Claude chat can help write code but cannot operate your laptop without a connected execution environment.

## How to use this guide

1. Create or clone the repository, open its folder in Claude Code, and provide this entire guide.
2. Send the one-time prompt below.
3. Send one numbered phase prompt at a time. Let Claude implement and verify it before advancing.
4. This guide has 21 numbered stages: Phase 0 through Phase 20. Phase 9 is the first usable image app; Phase 10 adds creator exports. Phases 11–15 add audio/video. Phases 16–17 verify reliability and quality. Phase 18 is optional comparison work. Phases 19–20 prepare release and demo materials.
5. If Claude starts a new session, tell it to read CLAUDE.md, this guide, and docs/progress.md before continuing.

Do not label untested functionality as completed. “Runs locally” refers to the target laptop, not a remote coding container.

## Milestones

| Phases | Result |
| --- | --- |
| 0–2 | Verified local environment and real model adapter |
| 3–8 | Persistent image library and three search modes |
| 9–10 | Version 1: usable image app and selections/export |
| 11–13 | Audio and cross-modal workspace |
| 14–15 | Video moments and clip export |
| 16–17 | Offline reliability and measured product quality |
| 18 | Optional model comparisons; may be skipped |
| 19–20 | Reviewable release and demonstration materials |

Each stage is a deliverable, not a promised number of days. The phase prompts inherit the one-time operating rules. If you have already started using the old eight-stage guide, map completed work to these new gates instead of rebuilding it.

## One-time prompt — paste first

```text
You are implementing MediaIndex, a local media-search workspace for creators, in this repository. Read the attached mediaindex-claude-build-guide.md and any existing repository instructions. Implement only the phase I request, but carry that phase through real execution and verification. Do not stop at a plan or return code for me to assemble.

Product goal: import local media, search with text, example media, or example media plus text, preview results, and export selections. Use google/embeddinggemma-2 as the initial model backend. This is our own app, not an Oxford WISE fork. Do not add model comparisons until optional Phase 18.

Preferred stack for an empty repo: Python + FastAPI backend; React + TypeScript + Vite frontend; SQLite for metadata; normalized NumPy vectors and exact cosine search initially. Avoid a vector database service, Docker requirement, cloud account, or generative LLM dependency. Preserve an established stack if this repo already has one and explain material differences.

Operating rules:
- First inspect the repository and actual machine. Do not assume OS, GPU, RAM, model APIs, dependency versions, or mixed-input support.
- Verify model behavior against the official model card and executable examples. Never silently substitute original EmbeddingGemma, Gemini's API, fake embeddings, or caption-based search.
- Setup may download dependencies, model weights and requested public sample data. After setup, normal indexing and search must work offline. No telemetry, external fonts, analytics, or remote media calls in normal use.
- Bind the service to loopback only. Never upload user media. Import only folders explicitly selected by the user; do not scan their home directory automatically.
- Keep source media, model caches, local databases and generated indexes out of git. Default to indexing originals in place without modifying them. Imported query uploads belong in a temporary app directory with cleanup.
- Pin the working dependency set and model revision after real verification. Record device, precision, preprocessing, prompt formats and embedding dimensions. No blanket trust_remote_code=True.
- Use immutable media IDs; API routes must not serve arbitrary filesystem paths. Limit uploads and media decoding; reject traversal and symlink escapes outside selected roots. Restrict browser origins and validate Host/Origin on local mutation endpoints.
- Keep the model loaded once in a single worker; do not spawn a model copy per HTTP request or multiple server workers. Serialize GPU-heavy work initially; provide progress and cancellation.
- Similarity is not confidence. Do not display cosine scores as probability percentages. Never claim mixed-input queries guarantee logical constraints, negation, temporal reasoning or sound synchronization.
- Write meaningful tests for persistence, ID alignment, search behavior and file access. Mocks are fine for unit tests; real model smoke checks are mandatory and must be labelled separately.
- Maintain docs/progress.md with completed gates, commands run, actual outputs, limitations and next phase. Update CLAUDE.md with stable project instructions.
- Do not push, publish, deploy, or delete unrelated files. Proceed autonomously on routine reversible implementation choices.

At each phase end give: files changed, exact run command, observed verification results, unresolved limitations, and whether the phase gate passed. If an essential capability is blocked, say exactly why rather than pretending the phase passed.

Wait for the Phase 0 instruction.
```

## Phase 0 — Repository and machine setup

**Outcome:** A documented, reproducible local development environment.

```text
Implement Phase 0 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Inspect repository instructions and actual OS, CPU, GPU, RAM/VRAM, free disk, Python and Node. Determine whether this is the target laptop or a remote environment; do not imply remote execution tests the user's laptop.
Preserve existing code. For an empty repo, initialize Python/FastAPI and React/TypeScript/Vite with a simple directory layout, isolated environments, and working health endpoint. Add .gitignore entries for media, uploads, databases, model caches, indexes and secrets. Create CLAUDE.md and docs/progress.md.
Record setup commands and hardware in docs/environment.md. Verify backend startup and frontend build. Do not download a large media collection yet.
Gate: dependencies install, the health endpoint responds, the frontend builds, and the machine report is accurate.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 1 — Real model compatibility

**Outcome:** Real text, image and native mixed-input smoke tests.

```text
Implement Phase 1 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Read current official model documentation linked in this guide. Verify that google/embeddinggemma-2 is downloadable, its license, required package versions, local device support and native image+text API. Stop on access or availability blockers; never substitute another model silently.
Write scripts/model_smoke.py and run text, image and mixed image+text inference on at least three different local sample images. Validate finite outputs, dimensions, normalization and real ranking outputs. Record actual scores; this is a smoke check, not an accuracy claim.
Verify supported precision; do not use float16 if prohibited by the current model docs. Use bfloat16 only where supported or float32. Verify MPS rather than assuming support. Record any CPU fallback.
Pin the working dependencies/model revision and write docs/model-compatibility.md. Cache assets and repeat with local-only loading offline.
Gate: the three required modes execute with real weights, or each blocked capability is explicitly reported. Do not build around fake outputs.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 2 — Model adapter and execution worker

**Outcome:** Reusable inference with stable index profiles.

```text
Implement Phase 2 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Wrap the verified inference code behind text, image and mixed-input methods with capability reporting. Use documented prompts/preprocessing. Do not caption media with another model.
Load one model instance in one worker, not one per request. Implement a bounded job queue, resource errors, device/precision settings and clean shutdown. Serialize heavy inference initially. Expose progress and cancellation between work units.
Define immutable index profiles including model ID/revision, enabled encoders, preprocessing, prompt templates, dimension and precision. Prevent incompatible vectors from being searched together.
Native image+text must use the verified interface. If it fails, leave it unavailable with an actionable error; do not silently replace it with an average.
Gate: repeated real requests reuse the model, concurrent submissions are controlled, and profile mismatch is rejected. Test worker behavior separately from real-model smoke tests.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 3 — SQLite library and persistence

**Outcome:** Stable identities for collections, files, vectors and jobs.

```text
Implement Phase 3 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Create versioned SQLite schema/migrations for libraries, assets, index profiles, embedding references and jobs. Assets need immutable IDs, root-relative paths, media type, file stats, content hash, dimensions/duration, status and error details.
Design vector storage with explicit asset/segment IDs and atomic snapshots or transactional writes. Never assume row order aligns SQLite and vector arrays. Removing an index item must not delete its original file.
Define import/reindex/recovery states and interrupted-job behavior. Add uniqueness rules for repeated imports while preserving separate file references to duplicate content.
Test database restart, migration, vector-ID mapping and rollback after partial writes.
Gate: persisted records and vectors retain identity after restart, and incomplete work cannot appear as a valid searchable index.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 4 — Image import and thumbnails

**Outcome:** A safe local folder ingestion pipeline.

```text
Implement Phase 4 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Implement image discovery only inside explicitly selected roots. Support JPEG, PNG and WebP after verifying decoder support. Check contents, EXIF orientation, oversized images and corrupt files. Do not follow symlinks outside the root or scan home directories automatically.
Generate thumbnails locally; record hashes and metadata. Index originals in place without modifying them. Add job progress, cancellation and a summary of skipped/failed files.
Implement idempotent re-import: reuse unchanged embeddings, reindex changed content, mark missing files, and reuse vectors for duplicates where safe.
Serve thumbnails/assets through stored IDs restricted to selected roots, never arbitrary file paths.
Gate: importing a small real folder survives corrupt files, cancellation/restart and repeat import without duplicating or changing originals.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 5 — Optional demo image pack

**Outcome:** A deterministic dataset download with traceable metadata.

```text
Implement Phase 5 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Inspect the current dataset card and schema at https://huggingface.co/datasets/KoalaAI/StockImages-CC0 before implementing scripts/download_demo.py.
Create an optional deterministic 500-image download with a pinned dataset revision and selection manifest. Record row IDs, hashes, publisher-declared license, source URL and image-level provenance where available. Download minimal required shards and disclose actual transfer size.
The publisher declares CC0, but image-level provenance is incomplete. Record that limitation in THIRD_PARTY_DATA.md; do not claim an independent rights audit. Do not bundle raw images in git.
Keep tags as inspection metadata only; do not include them in image embeddings. The app must still work on a user folder with no demo download.
Gate: rerunning the command yields the same selection, the importer accepts it, and each asset has a source record.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 6 — Image embeddings and text search

**Outcome:** The first end-to-end semantic search through the API.

```text
Implement Phase 6 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Connect ingestion to the real image encoder. Normalize and persist image vectors with stable IDs and index profile. Add text query embedding and exact cosine/dot-product ranking; no separate vector database service yet.
Provide typed search requests with library filters and result limits. Return actual asset IDs, preview URLs, raw similarity and separately measured query-embedding/ranking times.
Handle empty, partially indexed and incompatible libraries explicitly. Nearest-neighbor search always has candidates; do not invent calibrated no-match confidence.
Test ranking with known small vectors, filters and stale IDs. Run several real text queries against the imported image folder.
Gate: after restart, real text queries retrieve image assets without recomputing unchanged library vectors.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 7 — Reference-image search

**Outcome:** Upload an example and find similar images.

```text
Implement Phase 7 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Add temporary query-image upload, validation and cleanup using the same documented image preprocessing. Implement image-to-image ranking against the current library.
Exclude the reference asset and identical content hashes by default, with an explicit option to include them. Support choosing an existing asset ID as the reference.
Return useful errors for invalid images, oversized payloads and unsupported files. Never accept a request path as unrestricted access to the local filesystem.
Run real searches with both uploaded references and existing library assets. Test duplicate exclusion and cleanup.
Gate: references retrieve actual related candidates, the exact same image does not monopolize default results, and source files remain untouched.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 8 — Image plus text refinement

**Outcome:** Native composed queries with honest behavior.

```text
Implement Phase 8 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Implement one native query embedding from a reference image plus refinement text, using the verified model interface. Do not silently average separate vectors or generate an image caption.
Add query validation, result provenance and search-mode metadata. Preserve standalone text/image modes for comparison.
Run a small documented set of real refinements such as different setting, color or object attributes where candidates exist. Record cases where text improves retrieval and cases where it is ignored. No guaranteed negation or logical-constraint claims.
Reject unsupported mixed inputs explicitly rather than faking completion.
Gate: real combined-input inference returns candidates and the actual API/profile is recorded. Semantic quality limitations are documented; an API success alone is not proof that refinements work well.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 9 — Creator interface — Version 1

**Outcome:** A working local image-search application.

```text
Implement Phase 9 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Build a clean UI: library sidebar, import flow and progress, text search, reference-image drop area, optional refinement field, results grid, detail preview and use-this-as-reference action.
A browser cannot silently read arbitrary folders. A confirmed folder path in this loopback-only app is acceptable initially; explain it simply. Handle empty/loading/error/cancelled states and keyboard navigation.
Show real results only. Hide technical scores by default and never format them as probability. Bundle UI assets locally; no runtime CDNs.
Provide a documented local start command. Test import -> text search -> image search -> refinement -> preview -> restart in the actual browser when tooling permits; state unavailable UI checks.
Gate: a user can complete the workflow with real data. This is the first usable app; mark Version 1 here, not before.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 10 — Selections and image exports

**Outcome:** Useful creator output without altering originals.

```text
Implement Phase 10 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Add persistent selection collections, add/remove/reorder actions and a selection manifest export. Include asset IDs, source paths, source/license metadata where available and query context.
Add optional copy-to-destination export with a preview, explicit confirmation, collision handling and no silent overwrite. Never move or delete originals. Implement safe copy-path/reveal-in-folder without arbitrary shell execution.
Preserve selected items across restarts and show missing-file status.
Test duplicate filenames, paths with spaces, destination restrictions and interrupted copy behavior.
Gate: a creator can find images, collect them and export selected copies plus a valid manifest without changing source files.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 11 — Audio inference and segmentation

**Outcome:** Real audio embeddings stored with timestamps.

```text
Implement Phase 11 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Verify the current model's audio API and decoders on the target laptop. Smoke-test actual audio inference. Add WAV/FLAC/MP3 ingestion where codecs work; use documented sample-rate/channel handling.
Split long audio into configurable windows within model limits and preserve original start/end times. Track decoding failures, duration, profile and index progress.
Avoid multiple full model instances. Before reusing image vectors across changed encoder configurations, verify compatible outputs on identical images within documented tolerance or use separate profiles/reindex.
For demo audio, inspect https://huggingface.co/datasets/quinnlue/FSD50K-16k and select only individually declared CC0 clips for an optional deterministic sample, retaining provenance. If qualifying data cannot be verified, report it and use user-provided permitted recordings.
Gate: real audio segments have finite embeddings and correct source offsets, and image search still works.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 12 — Audio search and playback

**Outcome:** Searchable sounds with meaningful previews.

```text
Implement Phase 12 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Add text-to-audio and audio-reference-to-audio search. Include result duration, source, segment timestamps and inline playback at the retrieved segment. Add media-type filters and segment grouping.
Index audio samples themselves, not captions or filenames. Clean up temporary reference uploads. Make unsupported formats and missing codecs actionable.
Verify known sound queries, audio references, segment boundary playback and persistence after restart. Evaluate actual results without claiming all semantically similar clips are suitable sound effects.
Gate: a user can search imported sounds and play the matching segment entirely locally; image features remain functional.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 13 — Cross-modal workspace

**Outcome:** Search across images and sounds without misleading rankings.

```text
Implement Phase 13 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Add experimental audio-to-image and image-to-audio search using compatible embeddings from the same model profile. Expose supported query modes in the interface rather than assuming every combination works.
Group results by target modality by default because raw similarity scales are not calibrated across media. If offering a global ranking, label it experimental and explain the scoring rule in documentation.
Support optional example-plus-text queries only where native support has been verified. A video/sound suitability score is not synchronization or artistic quality.
Run a fixed small set of cross-modal examples and record both success and failure. Keep tags/captions out of media embeddings.
Gate: real cross-modal retrieval can be previewed, with index compatibility enforced and experimental modes clearly identified.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 14 — Video ingestion and indexing

**Outcome:** Timestamped visual/audio windows from local MP4 files.

```text
Implement Phase 14 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Verify the installed model's video decoding, frame sampling and audio behavior. Do not assume passing a video includes its soundtrack.
Use FFmpeg/ffprobe with safe argument lists. Support MP4 first. Segment into configurable overlapping windows, initially about 8 seconds with 4-second stride; keep exact offsets, frame sampling and token budgets.
Distinguish visual-only, audio-only and genuinely joint audio+visual embeddings. Keep profiles explicit. An average of frame vectors must not be called native temporal modeling.
Generate local thumbnails, queue indexing, support cancellation/restart and report unsupported codecs. Use user-owned or individually verified reusable demo videos; do not assume a website allows dataset redistribution.
Gate: real indexed windows refer to correct source intervals and recover from interrupted ingestion.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 15 — Video-moment search and clip export

**Outcome:** The full media-search creator workflow.

```text
Implement Phase 15 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Implement text-to-video and reference-image-plus-text-to-video where supported. Return ranked windows with source grouping, overlap deduplication, preview thumbnails and seek-to-timestamp playback.
Add experimental selected-video-window-to-audio suggestions using verified model support. These are candidates, not generated or synchronized sound.
Implement clip export with boundary preview, output validation and no source modification. Document frame-accurate re-encoding versus approximate stream-copy behavior.
State timestamp precision is limited by indexing windows/sampling; never claim exact event boundaries. Test codec failures, seek behavior, overlap grouping, export cancellation and exported duration.
Gate: users can search, watch the retrieved window and export a playable clip with provenance manifest.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 16 — Offline operation and robustness

**Outcome:** A dependable local app on the actual target machine.

```text
Implement Phase 16 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Audit startup and normal app flows for outgoing calls, telemetry and CDN dependencies. Ensure local-only model loading works after initial downloads; distinguish first-time setup from offline use. Test with outbound network blocked if possible and document what was actually verified.
Review loopback binding, Host/Origin validation, CORS, path restrictions, symlink handling, upload/decode limits, unsafe FFmpeg invocation and cleanup. Never expose the service publicly by default.
Exercise large-library indexing, concurrent requests, model-busy errors, disk-full behavior where safely simulated, and restart recovery. Verify one model instance and bound memory use.
Measure before optimizing; introduce ANN or quantization only for a measured need and recheck retrieval quality after changes.
Gate: all implemented media flows work offline on cached assets and material stability/security failures are resolved or clearly blocked.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 17 — Product-quality evaluation

**Outcome:** A reproducible report about this app's actual retrieval.

```text
Implement Phase 17 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Create a frozen initial 50-query image evaluation: 20 text, 15 image-reference, 15 mixed. Hold out references or exclude identical assets. Keep development queries separate.
Prepare a lightweight relevance-labelling workflow for human review. Do not silently use model-generated labels or tags as ground truth. Include multiple valid results, failures and unrelated candidates. If human labels are pending, implement the harness but leave evaluation incomplete rather than fabricate scores.
Report Hit@1/5; use Recall@5 and nDCG@10 only with appropriate relevance labels. Explain incomplete judgement pools and exclude unknowns appropriately. Keep these metrics distinct.
Measure cold load, indexing, warm embedding, ranking, total latency, peak process/GPU memory and disk size, recording device, precision, profile and cache state. Synchronize accelerator timing and use repeated runs for median/p95.
Gate: executable evaluation tools and an honest report with labelled-query coverage; unlabelled metrics remain explicitly pending.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 18 — Optional specialist-model comparison

**Outcome:** Independent benchmark work, separate from the app release.

```text
Implement Phase 18 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

This phase is optional. If skipped, record that and continue to Phase 19; it must not block release.
Add verified exact LAION CLAP checkpoints for text-to-audio, OpenCLIP for text-to-image, and ImageBind only when feasible and its terms fit. Record revision, recommended preprocessing and known training-data overlap.
Use official held-out Clotho data after verifying access/terms. Use Flickr30k only under its permitted noncommercial research/educational conditions, never as the app's bundled public demo pack; a custom labelled image set is an alternative.
Use identical candidate pools and queries per modality, exclude caption/tag leakage, document audio window aggregation, and run models separately on the same hardware. Separate controlled-runtime and best-supported-runtime comparisons. Never compare vectors from different model spaces.
Report full or clearly labelled subset results, uncertainty where meaningful and all resource settings. Dataset clip/image, not individual captions, is the statistical grouping unit.
Gate: reproducible commands and measured comparisons, or an explicit optional skip. Do not assume EmbeddingGemma wins.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 19 — Documentation and public-release preparation

**Outcome:** A reviewable open-source repository.

```text
Implement Phase 19 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Finish README with real quick start, supported formats/query modes, hardware observations, offline setup, troubleshooting and limitations. Add CONTRIBUTING, architecture overview, data/model notices and Apache-2.0 for original code unless an existing repository license requires a different decision. Do not invent author identity.
Keep model, third-party code and media licenses distinct. Optional demo downloads must retain provenance; do not imply publisher declarations are a rights audit. Choose traceable reusable assets for published screenshots.
Add CI for useful lightweight tests and an opt-in real-model suite. Verify clean setup in a fresh environment where feasible and document untested platforms.
Inspect tracked files for secrets, raw media, private paths, models, indexes and databases. Preserve user files.
Gate: release checklist, passing applicable checks, accurate documentation and a clean diff. Prepare only; do not push, publish or deploy.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```

## Phase 20 — Demo recording and final handoff

**Outcome:** A demonstration package ready for the user's review.

```text
Implement Phase 20 of MediaIndex now. Read CLAUDE.md, this guide and docs/progress.md first. Preserve prior working phases.

Prepare a 60–90-second demonstration storyboard using the implemented app: text search, reference+text refinement, an honest failure, audio/video if completed, and export. Use traceable reusable assets and actual outputs.
Provide exact launch/reset-demo commands, a small reproducible scenario, recording checklist, and README demo assets. Record a screen demo if tooling permits; otherwise deliver the script and shots without pretending a video was recorded.
Summarize actual evaluation findings only. Mark unimplemented modes as planned and never call example retrieval a breakthrough established by evidence.
Prepare a draft release description and repository description. Distinguish precomputed library indexing from query-time speed in narration. Include final run commands, known issues, supported hardware and completed/skipped gates.
Do not post to LinkedIn, push to GitHub or publish a release. Hand over the working app and prepared materials for my review.
Gate: a newcomer can follow the launch instructions and reproduce the demo, with truthful status for every feature.

Implement and execute this phase, then update docs/progress.md with changes, exact commands, observed results, limitations and gate status. Do not start the next phase automatically.
```


## Recovery prompt — if Claude becomes distracted or starts a new session

```text
Read CLAUDE.md, mediaindex-claude-build-guide.md and docs/progress.md. Inspect the current repository and identify the latest phase whose acceptance gate actually passed. Continue the current requested phase from that point, preserving working features. Implement and run the missing checks. Do not restart the project, silently change the model, add unrequested cloud services, or mark mock inference as real model execution. Report the exact blocker if the required environment is unavailable.
```

## Sources and verification notes

Reviewed October 6, 2026. Model APIs and downloadable artifacts must be rechecked on the target laptop in Phase 1; this guide is a build specification, not a claim that implementation has been executed.

- Official model: https://huggingface.co/google/embeddinggemma-2
- Model card: https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
- Multimodal local inference: https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers
- Image sample publisher: https://huggingface.co/datasets/KoalaAI/StockImages-CC0
- Audio sample publisher: https://huggingface.co/datasets/quinnlue/FSD50K-16k
- CC0 terms: https://creativecommons.org/publicdomain/zero/1.0/
- Flickr30k usage notice: https://github.com/BryanPlummer/flickr30k_entities/blob/master/README.md

The model documentation describes native mixed inputs, but the quality of refinements is an empirical question. Dataset sample packs are for demonstrating the app; they are not automatically standard benchmarks. Similarity retrieval returns existing items; it does not edit images, generate sounds, establish causal relations, or guarantee every requested constraint.
