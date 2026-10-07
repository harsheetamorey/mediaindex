# Image retrieval evaluation report

Run: `evaluation/runs/20261007T090550Z_3e76b03f5fdbe69e.json` · profile `3e76b03f5fdbe69e` · device `mps` · precision `bfloat16` · spec v1 (frozen 2026-10-07)

## Relevance metrics (human labels only)

Label coverage: **0/50 queries have any labels, 0/50 have a fully judged pool** (pool = union of top-10 results across runs; 500 candidates).

| Mode | Queries | Hit@1 | Hit@5 | Recall@5 (pool-relative) | nDCG@10 |
|---|---|---|---|---|---|
| text | 20 | PENDING (no labelled queries) | PENDING (no labelled queries) | PENDING (no labelled queries) | PENDING (no labelled queries) |
| image | 15 | PENDING (no labelled queries) | PENDING (no labelled queries) | PENDING (no labelled queries) | PENDING (no labelled queries) |
| image+text | 15 | PENDING (no labelled queries) | PENDING (no labelled queries) | PENDING (no labelled queries) | PENDING (no labelled queries) |

Rules: a candidate without a human label is *unknown*, not irrelevant. Hit@k is computed only when decidable (a labelled relevant item is in the top k, or the whole top k is labelled). Recall@5 and nDCG@10 are computed only for queries whose whole pool is judged, and recall is relative to that pool, so it overstates recall when relevant items exist outside the pool. Relevant means grade ≥ 1; nDCG uses graded gains 2^g − 1. Similarity scores are never used as relevance.

## End-to-end query latency (HTTP, this run)

Client-measured per query on the warm server: embedding + ranking + JSON. Image references to library assets reuse the stored vector, so they need no embedding.

| Mode | Median | p95 | Queries |
|---|---|---|---|
| text | 52 ms | 443 ms | 20 |
| image | 13 ms | 27 ms | 15 |
| image+text | 2509 ms | 2950 ms | 15 |

## Performance

Measured 2026-10-07T09:04:35Z on arm64 macOS-14.0-arm64-arm-64bit (Apple M1, 8 GB) · device `mps` · precision `bfloat16` · profile `3e76b03f5fdbe69e` · model cache warm on disk.

| Measure | Median | p95 | Runs |
|---|---|---|---|
| cold start: import + model load + first query (new process) | 13226.7 ms | 17307.6 ms | 3 |
| warm query embedding: text | 38.7 ms | 44.6 ms | 30 |
| warm query embedding: image (reference) | 1775.2 ms | 1803.9 ms | 15 |
| warm query embedding: image + text | 1878.8 ms | 2058.4 ms | 15 |
| warm query embedding: audio 10 s | 592.0 ms | 758.6 ms | 10 |
| warm embedding: video window (8 frames, 1,120 tokens) | 7574.2 ms | 10496.5 ms | 5 |
| ranking: exact dot product over 512 image vectors | 0.1 ms | 0.2 ms | 200 |
| ranking: exact over all 632 vectors (all media) | 0.1 ms | 0.1 ms | 200 |

| Resource | Value |
|---|---|
| process peak RSS (CPU side) | 1047 MB |
| process physical footprint peak (incl. Metal) | 3552 MB |
| MPS driver allocated | 2618 MB |
| image indexing throughput (embedding only) | 0.41 images/s |
| model cache on disk | 1450.7 MB |
| SQLite database | 4.1 MB |
| thumbnails | 12.4 MB |
| indexed vectors (all media) | 632 |

Warm timings exclude decoding and HTTP. Ranking uses the in-memory matrix (cache hit). End-to-end HTTP latency per query mode is in the evaluation run file (client_latency_ms). Library indexing is precomputed; query-time cost is one query embedding plus ranking.
