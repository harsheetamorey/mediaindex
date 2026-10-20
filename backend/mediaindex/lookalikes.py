"""Look-alike photos: burst shots, re-saved, cropped or edited copies, grouped for review. Nothing is ever deleted.

Groups come from the stored EmbeddingGemma image vectors (cosine similarity, not confidence). Thresholds were set with
the real model (evaluation/bench_lookalikes.py, docs/progress.md): on the 500-photo stock library, copies made smaller,
cropped, brighter or tilted scored 0.87-0.99 against their original, and different photos above 0.92 were nearly all
shots of the same scene. Each group is a star around one photo, so a chain of similar scenes never snowballs into one
big group.
"""

from __future__ import annotations

import numpy as np

LEVELS = {"strict": 0.95, "normal": 0.92, "loose": 0.90}
BLOCK = 1024


def group_lookalikes(ids: list[str], matrix: np.ndarray, threshold: float) -> list[list[str]]:
    """Groups of 2+ ids whose vectors are within `threshold` of the group's first photo. Biggest groups first."""
    n = len(ids)
    if n < 2:
        return []
    neighbours: list[np.ndarray] = []
    for s in range(0, n, BLOCK):  # never the full n x n matrix at once
        sims = matrix[s:s + BLOCK] @ matrix.T
        for k, row in enumerate(sims):
            hit = np.flatnonzero(row >= threshold)
            neighbours.append(hit[hit != s + k])
    order = sorted(range(n), key=lambda i: (-len(neighbours[i]), ids[i]))  # best-connected photos lead a group
    taken = np.zeros(n, bool)
    groups = []
    for i in order:
        if taken[i] or len(neighbours[i]) == 0:
            continue
        members = [j for j in neighbours[i] if not taken[j]]
        if not members:
            continue
        taken[i] = True
        taken[members] = True
        groups.append([ids[i], *(ids[j] for j in sorted(members, key=lambda j: -float(matrix[j] @ matrix[i])))])
    return sorted(groups, key=lambda g: -len(g))
