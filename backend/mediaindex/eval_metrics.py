"""Retrieval metrics over human relevance labels with incomplete judgement pools.

Labels: {candidate_id: grade} with grade 0 (not relevant), 1 (partial), 2 (clearly relevant).
A candidate missing from the labels is UNKNOWN, never assumed irrelevant.

- hit_at_k: 1 if a labelled-relevant item is in the top k; 0 only if all top-k items are labelled and none is
  relevant; None (excluded) if undecidable because of unknowns.
- recall_at_k / ndcg_at_k: only defined when every pooled candidate of the query is labelled (pool = what was
  judged, so recall is *pool-relative*); None otherwise.
"""

from __future__ import annotations

import math
from statistics import mean


def _rel(grade: int | None, threshold: int) -> bool:
    return grade is not None and grade >= threshold


def hit_at_k(ranked: list[str], labels: dict[str, int], k: int, threshold: int = 1) -> int | None:
    top = ranked[:k]
    if any(_rel(labels.get(c), threshold) for c in top):
        return 1
    if all(c in labels for c in top):
        return 0
    return None


def fully_judged(pool: list[str], labels: dict[str, int]) -> bool:
    return bool(pool) and all(c in labels for c in pool)


def recall_at_k(ranked: list[str], labels: dict[str, int], pool: list[str], k: int, threshold: int = 1) -> float | None:
    if not fully_judged(pool, labels):
        return None
    relevant = {c for c in pool if _rel(labels[c], threshold)}
    if not relevant:
        return None  # undefined: no relevant item known for this query
    return len(relevant & set(ranked[:k])) / len(relevant)


def ndcg_at_k(ranked: list[str], labels: dict[str, int], pool: list[str], k: int) -> float | None:
    if not fully_judged(pool, labels):
        return None
    gains = [(2 ** labels.get(c, 0) - 1) for c in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((2 ** labels[c] - 1 for c in pool), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return None if idcg == 0 else dcg / idcg


def aggregate(values: list[float | int | None]) -> dict:
    known = [v for v in values if v is not None]
    return {"value": round(mean(known), 4) if known else None, "n": len(known), "excluded": len(values) - len(known)}
