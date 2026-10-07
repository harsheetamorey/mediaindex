# Evaluating MediaIndex retrieval

The image evaluation is **frozen** in `evaluation/queries.json` (v1, 2026-10-07). It has 50 queries over the pinned 500-image demo pack:

- **20 text** queries, none of which were used during development. The development queries are listed in `dev_queries` and never scored.
- **15 image-reference** queries, with references drawn using `random.Random(17017)`.
- **15 image+text** queries, with seeded references and fixed refinements.

For reference queries, the reference asset and any identical-hash copies are excluded from results, which is the app's default behaviour.

## Workflow
1. `uv run python -m mediaindex`, then import `data/demo/stockimages-cc0`.
2. `uv run python evaluation/run_eval.py --library <id>` stores ranked top-10 results in `evaluation/runs/` and adds them to the judgement
   pool `evaluation/pool-v1.json`, which is the union over runs, so later model or profile runs extend the pool.
3. **Human labelling:** `uv run python evaluation/label_server.py --images data/demo/stockimages-cc0 --labeler "<name>"`, then open
   http://127.0.0.1:8777. Each query shows its pooled candidates in a shuffled order with no scores and no ranks. Grade each one 0, 1 or 2.
   Labels save to `evaluation/labels/labels-v1.json` on every click.
4. `uv run python evaluation/report.py` writes `docs/evaluation-report.md`.

Publisher tags and model outputs are **never** used as ground truth, and the harness has no option to auto-label.

## Metrics
- **Hit@1, Hit@5** (relevant = grade ≥ 1) are computed only when decidable. Unlabelled candidates are unknown, and undecidable queries are excluded and counted.
- **Recall@5** (pool-relative) and **nDCG@10** (graded gains 2^g − 1) are computed only for queries whose whole pool is judged.
  Pool-relative recall overstates true recall whenever relevant images exist outside the pool. With one system, the pool is that system's top 10.
- Every metric is reported with its `n` and the number of excluded queries. Metrics are reported separately per query mode.

## Performance
`uv run python evaluation/benchmark.py` (server stopped) measures:

- cold start in fresh processes
- warm query embedding per mode, with `torch.mps.synchronize()` around each call and 5–30 repeats, reported as median and p95
- exact ranking on the real index
- embedding throughput, peak RSS and Metal footprint, and disk usage

It writes `evaluation/perf-latest.json`, which `report.py` includes.
