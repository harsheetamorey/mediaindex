# Cross-modal retrieval: fixed example set (Phase 13)

Run on 2026-10-07 with `uv run python scripts/crossmodal_check.py` (MPS bf16, one shared EmbeddingGemma 2 profile;
500 demo images, 60 CC0 FSD50K clips plus a medley). "Expected" items were picked from publisher tags and labels **only to judge
these examples by eye**. They are not ground truth, they are never embedded, and 12 hand-picked cases are **not an accuracy estimate**.

| Query | Reference | Text | Expected (by tags) | Result |
|---|---|---|---|---|
| sound → images | dog bark (fsd-144028) | | the single dog photo | rank 7 |
| sound → images | train (fsd-150978) | | 4 train photos | **miss** (top 10) |
| sound → images | crickets (fsd-141471) | | insect photo | **miss** |
| sound → images | organ (fsd-151373) | | keyboard photo | **miss**, but piano photos rank 1–2 (sensible) |
| sound → images | glass shatter (fsd-144321) | | glass-tagged photos | **miss** |
| image → sounds | dog photo (stock-01642) | | 2 barks | rank 5 |
| image → sounds | train photo (stock-02000) | | train clip | **rank 1** |
| image → sounds | insect photo (stock-01683) | | 2 cricket clips | **rank 1** |
| image → sounds | keyboard photo (stock-01609) | | organ clip | rank 5 |
| image → sounds | car photo (stock-03414) | | traffic noise | rank 3 |
| image + text → sounds | dog photo | "barking loudly" | barks | **rank 1** (improves on rank 5 for the image alone) |
| sound + text → images | train clip | "at night" | train photos | rank 8 (night scenes dominate) |

**Summary:** the expected item appeared in the top 10 in 8 of 12 cases, and at rank 1 in 3 of 12.

## Observations
- **Image → sound works better than sound → image** on this data. Photos often surface a plausible sound (train, crickets).
- **Sound → image shows strong "hubness".** The same few images (stock-03443, stock-03459, stock-01680) top almost every audio query. Audio vectors
  seem to sit closer to those images than to the content-matched ones. This is a real limitation of the current cross-modal path.
- Adding text helped in one case (dog photo + "barking loudly"). In the other, the text dominated ("at night" pulled night scenes).
- Raw similarity scales differ by media type. Text queries scored roughly 0.65–0.76 against images and 0.6–0.76 against sounds, so the default UI
  **groups results by media type**. A merged "global" ranking is available only behind the technical toggle, labelled experimental.

## Scoring rule for the experimental global ranking
Each media type is searched separately with exact cosine similarity (dot product of L2-normalized vectors) under the same profile. The per-type
result lists (sounds grouped to their best window) are concatenated and sorted by raw cosine, then cut to the limit. No calibration or
normalization across types is applied. That is why it is experimental.

## What these modes are not
A sound suggested for an image (or the reverse) is a **similarity candidate**. It is not evidence of synchronization, timing, or artistic
suitability. Tags and captions are never part of any media embedding.
